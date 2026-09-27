from datetime import UTC, date, datetime, timedelta

import pytest

from predictor import db
from predictor.backtest import tipoff
from real_archive import open_real_archive_or_skip
from schedule_rows import insert_schedule_row


@pytest.fixture
def con(tmp_path):
    c = db.connect(tmp_path / "t.duckdb")
    db.migrate(c)
    return c


# --- parse_game_time: retained for the injury-PDF cross-check ------------


def test_evening_game_is_pm():
    # 7pm ET in January (EST, UTC-5) is 00:00 UTC the next day.
    got = tipoff.parse_game_time("07:00 (ET)", date(2025, 1, 15))
    assert got == datetime(2025, 1, 16, 0, 0, tzinfo=UTC)


def test_noon_game_is_not_shifted_to_midnight():
    got = tipoff.parse_game_time("12:00 (ET)", date(2025, 1, 15))
    assert got == datetime(2025, 1, 15, 17, 0, tzinfo=UTC)


def test_afternoon_game():
    got = tipoff.parse_game_time("03:30 (ET)", date(2025, 1, 15))
    assert got == datetime(2025, 1, 15, 20, 30, tzinfo=UTC)


def test_layout_without_space_before_paren_parses_identically():
    with_space = tipoff.parse_game_time("08:00 (ET)", date(2025, 1, 15))
    without = tipoff.parse_game_time("08:00(ET)", date(2025, 1, 15))
    assert with_space == without


def test_daylight_saving_is_honoured():
    summer = tipoff.parse_game_time("07:00 (ET)", date(2025, 6, 10))
    winter = tipoff.parse_game_time("07:00 (ET)", date(2025, 1, 10))
    assert summer.hour == 23
    assert winter.hour == 0


def test_unparseable_returns_none():
    for bad in ["", "TBD", "25:00 (ET)", "07:00 (PT)", "7pm"]:
        assert tipoff.parse_game_time(bad, date(2025, 1, 15)) is None


# --- tipoff_index: built from schedule vintages --------------------------


def test_index_keys_both_teams_to_the_tipoff(con):
    gd = date(2025, 1, 15)
    tip = datetime(2025, 1, 16, 0, 0, tzinfo=UTC)
    insert_schedule_row(con, "0022400561", gd, "PHI", "NYK", tip)
    index = tipoff.tipoff_index(con)
    assert index[(gd, "PHI")] == tip
    assert index[(gd, "NYK")] == tip


def test_a_later_vintage_moving_the_game_earlier_is_honoured(con):
    # FIX 14's shape: a correction to an EARLIER time arrives later. Ignoring
    # it would put the cutoff after the real tip-off.
    gd = date(2022, 11, 9)
    insert_schedule_row(con, "0022200161", gd, "ORL", "DAL",
                        datetime(2022, 11, 10, 0, 0, tzinfo=UTC),
                        observed_at=datetime(2022, 11, 1, tzinfo=UTC))
    insert_schedule_row(con, "0022200161", gd, "ORL", "DAL",
                        datetime(2022, 11, 9, 22, 30, tzinfo=UTC),
                        observed_at=datetime(2022, 11, 9, 12, tzinfo=UTC))
    index = tipoff.tipoff_index(con)
    assert index[(gd, "ORL")] == datetime(2022, 11, 9, 22, 30, tzinfo=UTC)


def test_a_later_vintage_moving_the_game_later_still_resolves_to_the_earlier_time(con):
    gd = date(2025, 1, 15)
    insert_schedule_row(con, "0022400561", gd, "PHI", "NYK",
                        datetime(2025, 1, 16, 0, 0, tzinfo=UTC),
                        observed_at=datetime(2025, 1, 1, tzinfo=UTC))
    insert_schedule_row(con, "0022400561", gd, "PHI", "NYK",
                        datetime(2025, 1, 16, 2, 30, tzinfo=UTC),
                        observed_at=datetime(2025, 1, 10, tzinfo=UTC))
    assert tipoff.tipoff_index(con)[(gd, "PHI")] == datetime(2025, 1, 16, 0, 0, tzinfo=UTC)


def test_tbd_rows_never_enter_the_index(con):
    gd = date(2026, 12, 4)
    insert_schedule_row(con, "0022601201", gd, "PHI", "NYK", None,
                        observed_at=datetime(2026, 9, 27, tzinfo=UTC))
    assert tipoff.tipoff_index(con) == {}


def test_injury_report_game_time_is_no_longer_a_tipoff_source(con):
    i = db.POINT_IN_TIME_TABLES["injury_status"]
    con.execute(
        f"INSERT INTO {i} (report_date, game_date, game_time, team, player, status,"
        " observed_at) VALUES (?,?,?,?,?,?,?)",
        [date(2025, 1, 15), date(2025, 1, 15), "07:00 (ET)", "PHI", "p", "Out",
         datetime(2025, 1, 15, 22, 30, tzinfo=UTC)],
    )
    assert tipoff.tipoff_index(con) == {}


# --- resolve_tipoff: unchanged pure function over the index --------------


def test_resolve_tipoff_takes_the_minimum_when_home_is_earlier():
    gd = date(2025, 1, 15)
    early = datetime(2025, 1, 16, 0, 0, tzinfo=UTC)
    late = datetime(2025, 1, 16, 0, 30, tzinfo=UTC)
    index = {(gd, "PHI"): early, (gd, "NYK"): late}
    assert tipoff.resolve_tipoff(index, gd, "PHI", "NYK") == early


def test_resolve_tipoff_takes_the_minimum_when_away_is_earlier():
    gd = date(2025, 1, 15)
    early = datetime(2025, 1, 16, 0, 30, tzinfo=UTC)
    late = datetime(2025, 1, 16, 1, 30, tzinfo=UTC)
    index = {(gd, "NYK"): late, (gd, "PHI"): early}
    assert tipoff.resolve_tipoff(index, gd, "NYK", "PHI") == early


def test_resolve_tipoff_falls_back_to_away_team():
    gd = date(2025, 1, 15)
    tip = datetime(2025, 1, 16, 0, 30, tzinfo=UTC)
    assert tipoff.resolve_tipoff({(gd, "NYK"): tip}, gd, "PHI", "NYK") == tip


def test_resolve_tipoff_returns_none_when_neither_team_has_an_entry():
    assert tipoff.resolve_tipoff({}, date(2025, 1, 15), "PHI", "NYK") is None


# --- archive-wide invariants (real archive, read-only) -------------------


def test_every_archived_regular_season_game_resolves_a_tipoff():
    """Sub-project 2.5's coverage promise: 8,289 of 8,289, not 7,200."""
    con = open_real_archive_or_skip()
    try:
        games = db.POINT_IN_TIME_TABLES["games"]
        index = tipoff.tipoff_index(con)
        rows = con.execute(
            f"SELECT game_id, MIN(game_date), MIN(home_team), MIN(away_team) "
            f"FROM {games} WHERE game_id LIKE '002%' GROUP BY game_id"
        ).fetchall()
        assert len(rows) >= 8289
        unresolved = [
            gid for gid, gd, h, a in rows if tipoff.resolve_tipoff(index, gd, h, a) is None
        ]
        assert unresolved == [], f"{len(unresolved)} game(s) have no tip-off: {unresolved[:10]}"
    finally:
        con.close()


@pytest.mark.parametrize("buffer_minutes", [30, 60, 120])
def test_archive_wide_cutoff_is_strictly_before_every_recorded_schedule_vintage(
    buffer_minutes,
):
    """FIX 23's archive-wide shape, now over schedule vintages.

    Compares every scored game's cutoff with EVERY tip-off recorded for
    either team on that date -- not with the resolved value, which is the
    test shape that let two tip-off leaks ship green. It becomes stricter
    every day the schedule job adds a vintage.
    """
    con = open_real_archive_or_skip()
    try:
        games = db.POINT_IN_TIME_TABLES["games"]
        sched = db.POINT_IN_TIME_TABLES["schedule"]
        index = tipoff.tipoff_index(con)
        vintages: dict[tuple[date, str], list[datetime]] = {}
        for gd, home, away, tip in con.execute(
            f"SELECT game_date, home_team, away_team, tip_off_utc FROM {sched} "
            "WHERE tip_off_utc IS NOT NULL"
        ).fetchall():
            vintages.setdefault((gd, home), []).append(tip)
            vintages.setdefault((gd, away), []).append(tip)
        rows = con.execute(
            f"SELECT DISTINCT game_id, game_date, home_team, away_team "
            f"FROM {games} WHERE game_id LIKE '002%'"
        ).fetchall()
        comparisons = 0
        unsafe = []
        for gid, gd, h, a in rows:
            tip = tipoff.resolve_tipoff(index, gd, h, a)
            if tip is None:
                continue
            cutoff = tip - timedelta(minutes=buffer_minutes)
            for v in vintages.get((gd, h), []) + vintages.get((gd, a), []):
                comparisons += 1
                if not cutoff < v:
                    unsafe.append((gid, cutoff, v))
        assert comparisons > 0
        assert unsafe == [], f"{len(unsafe)} unsafe cutoff(s): {unsafe[:5]}"
    finally:
        con.close()
