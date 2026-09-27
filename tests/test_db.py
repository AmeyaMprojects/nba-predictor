from datetime import UTC, datetime, timedelta, timezone

import duckdb
import pytest

from predictor import db
from predictor.config import PROJECT_ROOT


@pytest.fixture
def con(tmp_path):
    c = db.connect(tmp_path / "t.duckdb")
    db.migrate(c)
    return c


def test_connect_pins_session_timezone_to_utc(tmp_path):
    c = db.connect(tmp_path / "tz.duckdb")
    assert c.execute("SELECT current_setting('TimeZone')").fetchone()[0] == "UTC"


def test_migrate_creates_expected_tables(con):
    names = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert {
        "games_raw",
        "injury_status_raw",
        "odds_snapshots_raw",
        "news_items_raw",
    } <= names


def test_migrate_is_idempotent(con):
    """migrate() must never lose data, even if called again with rows present.

    Calling it twice on an empty database (asserting still-0-rows) is nearly
    tautological -- a future refactor to DROP TABLE/CREATE TABLE would wipe
    data and that assertion would still pass. Insert a row into every
    point-in-time table first, then prove migrate() is a true no-op on
    existing data.
    """
    moment = datetime(2025, 1, 15, 12, 0, tzinfo=UTC)
    con.execute(
        "INSERT INTO games_raw (game_id, season, game_date, home_team, away_team,"
        " status, observed_at) VALUES (?,?,?,?,?,?,?)",
        ["g1", "2024-25", moment.date(), "LAL", "BOS", "scheduled", moment],
    )
    con.execute(
        "INSERT INTO injury_status_raw (report_date, game_date, team, player,"
        " status, observed_at) VALUES (?,?,?,?,?,?)",
        [moment.date(), moment.date(), "LAL", "someone", "Out", moment],
    )
    con.execute(
        "INSERT INTO odds_snapshots_raw (game_key, book, home_team, away_team,"
        " observed_at) VALUES (?,?,?,?,?)",
        ["g1", "draftkings", "LAL", "BOS", moment],
    )
    con.execute(
        "INSERT INTO news_items_raw (item_key, feed, observed_at) VALUES (?,?,?)",
        ["n1", "rss", moment],
    )

    db.migrate(con)
    db.migrate(con)

    assert con.execute("SELECT count(*) FROM games_raw").fetchone()[0] == 1
    assert con.execute("SELECT count(*) FROM injury_status_raw").fetchone()[0] == 1
    assert con.execute("SELECT count(*) FROM odds_snapshots_raw").fetchone()[0] == 1
    assert con.execute("SELECT count(*) FROM news_items_raw").fetchone()[0] == 1


def test_every_point_in_time_table_has_observed_at(con):
    for physical in db.POINT_IN_TIME_TABLES.values():
        cols = {r[0] for r in con.execute(f"DESCRIBE {physical}").fetchall()}
        assert "observed_at" in cols, f"{physical} missing observed_at"


def test_point_in_time_tables_matches_schema_exactly(con):
    """POINT_IN_TIME_TABLES must not silently drift from the schema.

    The as-of accessor only guards the PHYSICAL tables named as values in
    this mapping. A table added later that carries observed_at but is never
    added to the mapping would leak future knowledge into the backtest with
    no error -- exactly the failure this schema exists to prevent. So the
    set of physical names must match, in both directions, the tables that
    actually carry observed_at according to DuckDB itself. (ingest_runs is
    not point-in-time data and is deliberately excluded -- it has no
    observed_at column.)
    """
    rows = con.execute(
        "SELECT table_name FROM information_schema.columns"
        " WHERE column_name = 'observed_at' AND table_schema = 'main'"
    ).fetchall()
    actual = {r[0] for r in rows}
    assert actual == set(db.POINT_IN_TIME_TABLES.values())


def test_observed_at_is_timestamptz(con):
    for physical in db.POINT_IN_TIME_TABLES.values():
        rows = con.execute(f"DESCRIBE {physical}").fetchall()
        kind = {r[0]: r[1] for r in rows}["observed_at"]
        assert "TIMESTAMP WITH TIME ZONE" in kind, f"{physical}.observed_at is {kind}"


def test_timestamps_roundtrip_in_utc_regardless_of_machine_timezone(con):
    """Guards the SET TimeZone='UTC' in connect().

    Without it DuckDB returns values in the local zone, which still compare
    equal but make failures machine-dependent and very hard to read.
    """
    moment = datetime(2025, 1, 15, 22, 30, tzinfo=UTC)
    con.execute(
        "INSERT INTO injury_status_raw (report_date, game_date, team, player,"
        " status, observed_at) VALUES (?,?,?,?,?,?)",
        [moment.date(), moment.date(), "LAL", "someone", "Out", moment],
    )
    got = con.execute("SELECT observed_at FROM injury_status_raw").fetchone()[0]
    assert got == moment
    assert got.utcoffset().total_seconds() == 0, f"returned in non-UTC zone: {got}"


# --- require_utc ---------------------------------------------------------


def test_require_utc_rejects_naive_datetime():
    with pytest.raises(ValueError, match="timezone-aware"):
        db.require_utc(datetime(2025, 1, 15, 12, 0))


def test_require_utc_rejects_non_utc_offset():
    tz = timezone(timedelta(hours=-5))
    with pytest.raises(ValueError, match="UTC"):
        db.require_utc(datetime(2025, 1, 15, 12, 0, tzinfo=tz))


def test_require_utc_passes_through_utc_datetime():
    moment = datetime(2025, 1, 15, 12, 0, tzinfo=UTC)
    assert db.require_utc(moment) is moment


def test_require_utc_uses_field_name_in_message():
    with pytest.raises(ValueError, match="published_at must be timezone-aware"):
        db.require_utc(datetime(2025, 1, 15, 12, 0), field="published_at")


# --- injury_status primary key --------------------------------------------


def test_injury_status_pk_allows_same_player_two_game_dates_one_report(con):
    """A report published at one observed_at can span two game dates.

    5 of 11 real injury reports sampled during review spanned two different
    game dates in a single report -- exactly the precondition for the same
    player appearing twice with the same (observed_at, team, player). The
    old PK (observed_at, team, player) would collide there, and because
    ingestion uses INSERT OR REPLACE, the collision would silently delete a
    row instead of erroring. Prove both rows now persist.
    """
    moment = datetime(2025, 1, 15, 12, 0, tzinfo=UTC)
    con.execute(
        "INSERT INTO injury_status_raw (report_date, game_date, team, player,"
        " status, observed_at) VALUES (?,?,?,?,?,?)",
        [moment.date(), datetime(2025, 1, 15).date(), "LAL", "LeBron James",
         "Questionable", moment],
    )
    con.execute(
        "INSERT INTO injury_status_raw (report_date, game_date, team, player,"
        " status, observed_at) VALUES (?,?,?,?,?,?)",
        [moment.date(), datetime(2025, 1, 17).date(), "LAL", "LeBron James",
         "Probable", moment],
    )
    rows = con.execute(
        "SELECT game_date, status FROM injury_status_raw"
        " WHERE team='LAL' AND player='LeBron James' ORDER BY game_date"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0][0] == datetime(2025, 1, 15).date()
    assert rows[1][0] == datetime(2025, 1, 17).date()


def test_injury_status_insert_or_replace_collapses_identical_key(con):
    """Re-ingesting the identical row must still collapse to one, not grow."""
    moment = datetime(2025, 1, 15, 12, 0, tzinfo=UTC)
    game_date = datetime(2025, 1, 15).date()
    con.execute(
        "INSERT OR REPLACE INTO injury_status_raw (report_date, game_date, team,"
        " player, status, observed_at) VALUES (?,?,?,?,?,?)",
        [moment.date(), game_date, "LAL", "LeBron James", "Questionable", moment],
    )
    con.execute(
        "INSERT OR REPLACE INTO injury_status_raw (report_date, game_date, team,"
        " player, status, observed_at) VALUES (?,?,?,?,?,?)",
        [moment.date(), game_date, "LAL", "LeBron James", "Out", moment],
    )
    rows = con.execute(
        "SELECT status FROM injury_status_raw"
        " WHERE team='LAL' AND player='LeBron James' AND game_date=?",
        [game_date],
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "Out"


# --- FIX 1 rail: refuse to open the real database during a test run ------


def test_connect_refuses_the_real_database_path_during_a_test_run():
    """A regression here would let a future test silently write to the

    real, irreplaceable archive again -- exactly FIX 1's failure mode. This
    calls connect() with the real path explicitly (not via settings.db_path,
    which conftest.py already redirects); PYTEST_CURRENT_TEST is set by
    pytest itself for the whole duration of this test, so the rail must
    trip -- and it must do so before ever touching the real file on disk.
    """
    real_path = PROJECT_ROOT / "data" / "predictor.duckdb"
    with pytest.raises(RuntimeError, match="PREDICTOR_DATA_DIR"):
        db.connect(real_path)


def test_connect_allows_the_real_database_path_outside_a_test_run(monkeypatch, tmp_path):
    """The rail is scoped to test runs only -- it must not fire in production.

    Uses a decoy file standing in for the real path (monkeypatched onto
    db._REAL_DB_PATH) so this test cannot accidentally open the genuine
    archive even with PYTEST_CURRENT_TEST cleared.
    """
    decoy = tmp_path / "predictor.duckdb"
    monkeypatch.setattr(db, "_REAL_DB_PATH", decoy)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    con = db.connect(decoy)
    con.close()


def test_injury_status_game_date_is_not_null(con):
    with pytest.raises(duckdb.ConstraintException):
        con.execute(
            "INSERT INTO injury_status_raw (report_date, team, player, status,"
            " observed_at) VALUES (?,?,?,?,?)",
            [datetime(2025, 1, 15).date(), "LAL", "someone", "Out",
             datetime(2025, 1, 15, 12, 0, tzinfo=UTC)],
        )


def test_schedule_table_is_registered_and_has_no_outcome_columns(con):
    assert db.POINT_IN_TIME_TABLES["schedule"] == "schedule_raw"
    cols = {r[0] for r in con.execute("DESCRIBE schedule_raw").fetchall()}
    assert cols == {
        "game_id", "season", "game_date", "tip_off_utc", "home_team",
        "away_team", "arena_name", "arena_city", "arena_state",
        "is_neutral_reported", "is_neutral", "observed_at",
    }
    # Spec 1.1: scores are never ingested -- a column that does not exist
    # cannot leak.
    for forbidden in ("home_points", "away_points", "score", "status", "wins", "losses"):
        assert not any(forbidden in c for c in cols), forbidden
