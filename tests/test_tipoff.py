from datetime import UTC, date, datetime, timedelta

import duckdb
import pytest

from predictor import db
from predictor.backtest import tipoff
from predictor.config import PROJECT_ROOT

_REAL_ARCHIVE = PROJECT_ROOT / "data" / "predictor.duckdb"


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


def test_resolve_tipoff_takes_the_minimum_when_home_is_earlier(con):
    gd = date(2025, 1, 15)
    observed = datetime(2025, 1, 14, 12, 0, tzinfo=UTC)
    _insert_injury_row(con, gd, "PHI", "07:00 (ET)", observed)
    _insert_injury_row(con, gd, "NYK", "07:30 (ET)", observed, player="q")

    index = tipoff.tipoff_index(con)
    got = tipoff.resolve_tipoff(index, gd, "PHI", "NYK")
    assert got == tipoff.parse_game_time("07:00 (ET)", gd)


def test_resolve_tipoff_takes_the_minimum_when_away_is_earlier(con):
    """FIX 23 (final review, part 4) -- the regression this wave closes. The
    old rule (`index.get(home) or index.get(away)`) returned the HOME
    team's entry whenever it existed, even when the AWAY team's entry was
    earlier -- exactly the shape of 0022400624 (2025-01-23, MIA at MIL) and
    0021900701 (2020-01-28, BOS at MIA) in the real archive. Both teams
    have an entry here; the away team's (PHI, this fixture's away side) is
    earlier and must win.
    """
    gd = date(2025, 1, 15)
    observed = datetime(2025, 1, 14, 12, 0, tzinfo=UTC)
    _insert_injury_row(con, gd, "NYK", "08:30 (ET)", observed)
    _insert_injury_row(con, gd, "PHI", "07:30 (ET)", observed, player="q")

    index = tipoff.tipoff_index(con)
    got = tipoff.resolve_tipoff(index, gd, "NYK", "PHI")
    assert got == tipoff.parse_game_time("07:30 (ET)", gd)


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


# --- FIX 23 (final review, part 4): the archive-wide invariant -------------
#
# The absence of a check like this one is what let both the CRITICAL
# regression (FIX 14) and this one (FIX 23) through: a test whose fixture
# only ever exercises ONE team's entries cannot catch a bug in how the
# other team's entries are combined with it. This reads the real,
# read-only archive (never written to -- `duckdb.connect(..., read_only=True)`
# is used directly, deliberately bypassing `db.connect()`'s test-isolation
# rail, which exists to stop a test from migrating or writing to the real
# database, not from reading it) and checks the actual property the
# harness needs, for every scored game against EVERY recorded vintage for
# EITHER team -- not just the two games this wave's evidence happened to
# name.
@pytest.mark.skipif(
    not _REAL_ARCHIVE.exists(), reason="real archive not present in this environment"
)
@pytest.mark.parametrize("buffer_minutes", [30, 60, 120])
def test_archive_wide_cutoff_is_strictly_before_every_recorded_tipoff_vintage(
    buffer_minutes,
):
    # The scheduled `poll-news` job holds a write lock on the real archive at
    # 09:00/14:00/19:00, and DuckDB allows one writer XOR readers. Without
    # this, a suite run during one of those windows fails inside a TIP-OFF
    # test, which reads to a non-coding user as "tip-off resolution is
    # broken" when nothing is wrong at all.
    try:
        con = duckdb.connect(str(_REAL_ARCHIVE), read_only=True)
    except duckdb.Error as exc:  # pragma: no cover - timing-dependent
        pytest.skip(f"real archive is locked by another process: {exc}")
    try:
        games_table = db.POINT_IN_TIME_TABLES["games"]
        inj_table = db.POINT_IN_TIME_TABLES["injury_status"]

        index = tipoff.tipoff_index(con)

        # Every parsed vintage per (game_date, team) -- NOT collapsed to the
        # minimum, unlike `index` -- so every recorded filing is checked,
        # not just the one that happened to win.
        vintage_rows = con.execute(
            f"SELECT game_date, team, game_time FROM {inj_table} "
            "WHERE game_date IS NOT NULL AND game_time IS NOT NULL AND game_time <> ''"
        ).fetchall()
        vintages: dict[tuple[date, str], list[datetime]] = {}
        for game_date, team, raw in vintage_rows:
            parsed = tipoff.parse_game_time(raw, game_date)
            if parsed is None:
                continue
            vintages.setdefault((game_date, team), []).append(parsed)

        games = con.execute(
            f"SELECT DISTINCT game_id, game_date, home_team, away_team "
            f"FROM {games_table} WHERE game_id LIKE '002%'"
        ).fetchall()

        scored = 0
        comparisons = 0
        unsafe: list[tuple[str, datetime, datetime]] = []
        for game_id, game_date, home_team, away_team in games:
            tip = tipoff.resolve_tipoff(index, game_date, home_team, away_team)
            if tip is None:
                continue
            cutoff = tip - timedelta(minutes=buffer_minutes)
            scored += 1
            for v in (
                vintages.get((game_date, home_team), [])
                + vintages.get((game_date, away_team), [])
            ):
                comparisons += 1
                if not (cutoff < v):
                    unsafe.append((game_id, cutoff, v))

        assert scored > 0, "the real archive produced no scorable games -- check the path"
        assert comparisons > 0
        assert unsafe == [], (
            f"{len(unsafe)} cutoff-vs-vintage comparison(s) failed at "
            f"buffer_minutes={buffer_minutes} out of {comparisons:,} across "
            f"{scored:,} games: {unsafe[:5]}"
        )
    finally:
        con.close()
