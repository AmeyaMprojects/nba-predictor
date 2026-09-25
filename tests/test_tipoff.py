from datetime import UTC, date, datetime, timedelta

import pytest

from predictor import db
from predictor.backtest import tipoff


@pytest.fixture
def con(tmp_path):
    c = db.connect(tmp_path / "t.duckdb")
    db.migrate(c)
    return c


def _insert_injury_row(con, game_date, team, game_time, observed_at, player="p"):
    con.execute(
        "INSERT INTO injury_status_raw (report_date, game_date, game_time, team,"
        " player, status, observed_at) VALUES (?,?,?,?,?,?,?)",
        [
            observed_at.date(),
            game_date,
            game_time,
            team,
            player,
            "Out",
            observed_at,
        ],
    )


def test_evening_game_is_pm():
    # '07:00 (ET)' on a January date means 7pm EST = 00:00 UTC next day
    got = tipoff.parse_game_time("07:00 (ET)", date(2025, 1, 15))
    assert got == datetime(2025, 1, 16, 0, 0, tzinfo=UTC)


def test_noon_game_is_not_shifted_to_midnight():
    # '12:00 (ET)' means noon, not midnight
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
    # June is EDT (UTC-4); January is EST (UTC-5)
    summer = tipoff.parse_game_time("07:00 (ET)", date(2025, 6, 10))
    winter = tipoff.parse_game_time("07:00 (ET)", date(2025, 1, 10))
    assert summer.hour == 23
    assert winter.hour == 0  # rolled into the next UTC day


def test_unparseable_returns_none():
    for bad in ["", "TBD", "not a time", "25:00 (ET)"]:
        assert tipoff.parse_game_time(bad, date(2025, 1, 15)) is None


def test_tipoff_index_resolves_a_conflicting_time_to_the_minimum_clock_time(con):
    # A rescheduled game: an early-filed report says 8pm, a later one
    # corrects it to 5:30pm -- the real tip-off. FIX 14 (final review, part
    # 3): the MINIMUM clock time across vintages must win, deterministically
    # -- NOT the earliest-observed filing (that was the CRITICAL regression:
    # it resolved to 8pm here, an hour after a 5:30-7:30pm-ish real tip-off
    # would already be under way, putting the cutoff after the game started).
    gd = date(2022, 11, 9)
    early = datetime(2022, 11, 8, 12, 0, tzinfo=UTC)
    late = early + timedelta(hours=6)
    _insert_injury_row(con, gd, "DAL", "08:00 (ET)", early)
    _insert_injury_row(con, gd, "DAL", "05:30 (ET)", late)

    expected = tipoff.parse_game_time("05:30 (ET)", gd)
    for _ in range(5):
        index = tipoff.tipoff_index(con)
        assert index[(gd, "DAL")] == expected


def test_flagged_regression_games_resolve_to_the_earlier_real_tipoff(con):
    """FIX 14 (final review, part 3) -- the two archive games the CRITICAL
    regression was verified against. Under the buggy earliest-observed
    rule these resolved to 07:00 (ET) and 08:30 (ET) respectively -- both
    AFTER the real tip-off -- because the earliest-FILED report happened to
    carry the later clock time. Both must resolve to the earlier (real)
    tip-off instead.
    """
    # 0022200161  2022-11-09 ORL v DAL
    gd1 = date(2022, 11, 9)
    _insert_injury_row(
        con, gd1, "DAL", "07:00 (ET)", datetime(2022, 11, 8, 22, 30, tzinfo=UTC)
    )
    _insert_injury_row(
        con, gd1, "DAL", "05:30 (ET)", datetime(2022, 11, 9, 22, 30, tzinfo=UTC),
        player="q",
    )

    # 0022400521  2025-01-09 DAL v POR
    gd2 = date(2025, 1, 9)
    _insert_injury_row(
        con, gd2, "DAL", "08:30 (ET)", datetime(2025, 1, 8, 22, 30, tzinfo=UTC)
    )
    _insert_injury_row(
        con, gd2, "DAL", "07:30 (ET)", datetime(2025, 1, 9, 22, 30, tzinfo=UTC),
        player="q",
    )

    index = tipoff.tipoff_index(con)
    assert index[(gd1, "DAL")] == tipoff.parse_game_time("05:30 (ET)", gd1)
    assert index[(gd2, "DAL")] == tipoff.parse_game_time("07:30 (ET)", gd2)


def test_a_tipoff_moved_later_in_a_subsequent_vintage_still_resolves_to_the_earlier_time(con):
    """FIX 4(c): the missing test. A game whose tip-off moves LATER in a
    subsequent vintage (e.g. a broadcast-driven push-back) must still
    resolve to the EARLIER time -- the later filing was published after the
    earlier one, and the resolved value must not depend on information
    published after any particular game's cutoff.
    """
    gd = date(2025, 2, 1)
    early = datetime(2025, 1, 31, 12, 0, tzinfo=UTC)
    late = early + timedelta(hours=6)
    _insert_injury_row(con, gd, "BOS", "07:00 (ET)", early)
    _insert_injury_row(con, gd, "BOS", "09:30 (ET)", late)

    expected = tipoff.parse_game_time("07:00 (ET)", gd)
    index = tipoff.tipoff_index(con)
    assert index[(gd, "BOS")] == expected


def test_resolve_tipoff_prefers_home_team(con):
    gd = date(2025, 1, 15)
    observed = datetime(2025, 1, 14, 12, 0, tzinfo=UTC)
    _insert_injury_row(con, gd, "PHI", "07:00 (ET)", observed)
    _insert_injury_row(con, gd, "NYK", "07:30 (ET)", observed, player="q")

    index = tipoff.tipoff_index(con)
    got = tipoff.resolve_tipoff(index, gd, "PHI", "NYK")
    assert got == tipoff.parse_game_time("07:00 (ET)", gd)


def test_resolve_tipoff_falls_back_to_away_team(con):
    gd = date(2025, 1, 15)
    observed = datetime(2025, 1, 14, 12, 0, tzinfo=UTC)
    _insert_injury_row(con, gd, "NYK", "07:30 (ET)", observed)

    index = tipoff.tipoff_index(con)
    got = tipoff.resolve_tipoff(index, gd, "PHI", "NYK")
    assert got == tipoff.parse_game_time("07:30 (ET)", gd)


def test_resolve_tipoff_returns_none_when_neither_team_has_an_entry(con):
    index = tipoff.tipoff_index(con)
    assert tipoff.resolve_tipoff(index, date(2025, 1, 15), "PHI", "NYK") is None
