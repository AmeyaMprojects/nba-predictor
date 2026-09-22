import pytest

from predictor import db


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
    assert {"games", "injury_status", "odds_snapshots", "news_items"} <= names


def test_migrate_is_idempotent(con):
    db.migrate(con)
    db.migrate(con)
    assert con.execute("SELECT count(*) FROM games").fetchone()[0] == 0


def test_every_point_in_time_table_has_observed_at(con):
    for table in db.POINT_IN_TIME_TABLES:
        cols = {r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()}
        assert "observed_at" in cols, f"{table} missing observed_at"


def test_observed_at_is_timestamptz(con):
    for table in db.POINT_IN_TIME_TABLES:
        rows = con.execute(f"DESCRIBE {table}").fetchall()
        kind = {r[0]: r[1] for r in rows}["observed_at"]
        assert "TIMESTAMP WITH TIME ZONE" in kind, f"{table}.observed_at is {kind}"


def test_timestamps_roundtrip_in_utc_regardless_of_machine_timezone(con):
    """Guards the SET TimeZone='UTC' in connect().

    Without it DuckDB returns values in the local zone, which still compare
    equal but make failures machine-dependent and very hard to read.
    """
    from datetime import UTC, datetime

    moment = datetime(2025, 1, 15, 22, 30, tzinfo=UTC)
    con.execute(
        "INSERT INTO injury_status (report_date, team, player, status, observed_at)"
        " VALUES (?,?,?,?,?)",
        [moment.date(), "LAL", "someone", "Out", moment],
    )
    got = con.execute("SELECT observed_at FROM injury_status").fetchone()[0]
    assert got == moment
    assert got.utcoffset().total_seconds() == 0, f"returned in non-UTC zone: {got}"
