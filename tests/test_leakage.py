"""Adversarial tests: deliberately attempt to leak future data.

If any test here fails, the backtest's results are meaningless. Treat a
failure as a critical defect, never as a test to relax.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import pytest

from predictor import db
from predictor.asof import AsOfError, AsOfView

TIP_OFF = datetime(2025, 1, 15, 0, 0, tzinfo=UTC)


@pytest.fixture
def con(tmp_path):
    c = db.connect(tmp_path / "t.duckdb")
    db.migrate(c)
    return c


def _insert_game(con, game_id, observed_at, status, home_points, away_points):
    con.execute(
        "INSERT INTO games_raw (game_id, season, game_date, home_team, away_team,"
        " home_points, away_points, status, observed_at) VALUES (?,?,?,?,?,?,?,?,?)",
        [
            game_id,
            "2024-25",
            TIP_OFF.date(),
            "PHI",
            "NYK",
            home_points,
            away_points,
            status,
            observed_at,
        ],
    )


def test_final_score_is_invisible_before_the_game_finishes(con):
    _insert_game(con, "001", TIP_OFF - timedelta(days=1), "SCHEDULED", None, None)
    _insert_game(con, "001", TIP_OFF + timedelta(hours=3), "FINAL", 110, 104)

    view = AsOfView(con, TIP_OFF - timedelta(minutes=30))
    rows = view.table("games").project("status, home_points").fetchall()

    assert rows == [("SCHEDULED", None)]
    assert all(r[1] is None for r in rows), "final score leaked into pre-game view"


def test_injury_report_published_after_cutoff_is_invisible(con):
    for offset, player in [(-1, "early"), (1, "late")]:
        con.execute(
            "INSERT INTO injury_status_raw (report_date, game_date, team, player, status, observed_at)"
            " VALUES (?,?,?,?,?,?)",
            [
                TIP_OFF.date(),
                TIP_OFF.date(),
                "PHI",
                player,
                "Out",
                TIP_OFF + timedelta(hours=offset),
            ],
        )
    view = AsOfView(con, TIP_OFF)
    players = {r[0] for r in view.table("injury_status").project("player").fetchall()}
    assert players == {"early"}


def test_odds_moved_after_cutoff_are_invisible(con):
    for offset, spread in [(-2, -3.5), (2, -7.5)]:
        con.execute(
            "INSERT INTO odds_snapshots_raw (game_key, book, home_team, away_team,"
            " spread, observed_at) VALUES (?,?,?,?,?,?)",
            ["g1", "bookA", "PHI", "NYK", spread, TIP_OFF + timedelta(hours=offset)],
        )
    view = AsOfView(con, TIP_OFF)
    spreads = [r[0] for r in view.table("odds_snapshots").project("spread").fetchall()]
    assert spreads == [-3.5]


def test_guard_cannot_be_bypassed_with_a_crafted_table_name(con):
    view = AsOfView(con, TIP_OFF)
    for hostile in [
        "games WHERE 1=1 OR observed_at > now()",
        "(SELECT * FROM games)",
        "games--",
        "GAMES",
    ]:
        with pytest.raises(AsOfError):
            view.table(hostile)


def test_every_point_in_time_table_is_reachable_through_the_view(con):
    """A new table added to the schema must not silently bypass the guard."""
    view = AsOfView(con, TIP_OFF)
    for table in db.POINT_IN_TIME_TABLES:
        view.table(table).fetchall()


# --- FIX 1 pins: physical-table rename closes every relation-API escape ---
#
# Adversarial review (I-1) found that con.sql() binds to the connection's
# catalog, so any SQL fragment passed to .project()/.aggregate()/.filter()/
# .query()/.join()/.union() that names the LOGICAL table "games" reads it
# completely unfiltered -- past the observed_at <= cutoff guard entirely.
# Before the rename, this returned the FINAL score hours before tip-off.
# After the rename (games -> games_raw), "games" is not a real table, so
# each of these now raises a DuckDB CatalogException instead of leaking.


@pytest.fixture
def leaky_con(con):
    """A games table with a pre-game row and a later FINAL row, both
    observable only through the raw connection -- exactly the I-1 setup."""
    _insert_game(con, "001", TIP_OFF - timedelta(days=1), "SCHEDULED", None, None)
    _insert_game(con, "001", TIP_OFF + timedelta(hours=3), "FINAL", 110, 104)
    return con


def test_project_naming_logical_table_raises_not_leaks(leaky_con):
    view = AsOfView(leaky_con, TIP_OFF - timedelta(minutes=30))
    base = view.table("games").project("status, home_points")
    with pytest.raises(duckdb.CatalogException):
        base.project(
            "status, home_points, (SELECT max(home_points) FROM games) AS leaked"
        ).fetchall()


def test_aggregate_naming_logical_table_raises_not_leaks(leaky_con):
    view = AsOfView(leaky_con, TIP_OFF - timedelta(minutes=30))
    base = view.table("games")
    with pytest.raises(duckdb.CatalogException):
        base.aggregate("(SELECT max(away_points) FROM games)").fetchall()


def test_filter_exists_naming_logical_table_raises_not_leaks(leaky_con):
    view = AsOfView(leaky_con, TIP_OFF - timedelta(minutes=30))
    base = view.table("games")
    with pytest.raises(duckdb.CatalogException):
        base.filter(
            "EXISTS (SELECT 1 FROM games WHERE status = 'FINAL')"
        ).fetchall()


def test_query_naming_logical_table_raises_not_leaks(leaky_con):
    view = AsOfView(leaky_con, TIP_OFF - timedelta(minutes=30))
    base = view.table("games")
    with pytest.raises(duckdb.CatalogException):
        base.query("v", "SELECT * FROM games").fetchall()


def test_union_naming_logical_table_raises_not_leaks(leaky_con):
    view = AsOfView(leaky_con, TIP_OFF - timedelta(minutes=30))
    base = view.table("games").project("status, home_points")
    with pytest.raises(duckdb.CatalogException):
        # con.sql() is a fresh, unfiltered read bound to the connection's
        # catalog -- exactly the I-1 escape, now closed by the rename.
        other = leaky_con.sql("SELECT status, home_points FROM games")
        base.union(other).fetchall()


def test_join_naming_logical_table_raises_not_leaks(leaky_con):
    view = AsOfView(leaky_con, TIP_OFF - timedelta(minutes=30))
    base = view.table("games").project("status, home_points")
    with pytest.raises(duckdb.CatalogException):
        other = leaky_con.sql("SELECT status AS s2, home_points AS hp2 FROM games")
        base.join(other, "status = s2").fetchall()


def test_physical_raw_table_name_still_leaks_if_named_deliberately(leaky_con):
    """Documented, accepted residual: renaming closes the ACCIDENTAL path
    (typing the logical name), not a DELIBERATE one. A caller who explicitly
    writes the physical "_raw" name -- which AsOfView never exposes and no
    caller would type by accident -- still bypasses the cutoff filter. This
    is intentional per FIX 1's scope; the enforcement mechanism for it is
    test_no_physical_table_name_appears_outside_db_and_asof below, which
    fails the suite if any other module ever spells a "_raw" table name.
    """
    view = AsOfView(leaky_con, TIP_OFF - timedelta(minutes=30))
    base = view.table("games").project("status, home_points")
    leaked = base.project(
        "status, home_points, (SELECT max(home_points) FROM games_raw) AS leaked"
    ).fetchall()
    assert leaked == [("SCHEDULED", None, 110)], "expected residual leak via physical name"


# --- FIX 5: the boundary's enforcement mechanism for future feature code --


def test_no_physical_table_name_appears_outside_db_and_asof():
    """The physical "_raw" table names must only ever be spelled in db.py
    (schema DDL) and asof.py (the resolver). Any other module in src/ that
    contains one of these names in a string literal is feature code reading
    point-in-time data directly, bypassing AsOfView -- exactly the mistake
    FIX 1 makes structurally hard, but only if nothing else names the raw
    tables either.
    """
    src_root = Path(__file__).resolve().parents[1] / "src" / "predictor"
    allowed = {src_root / "db.py", src_root / "asof.py"}
    physical_names = set(db.POINT_IN_TIME_TABLES.values())

    offenders = []
    for path in src_root.rglob("*.py"):
        if path in allowed:
            continue
        text = path.read_text()
        for name in physical_names:
            if name in text:
                offenders.append((path, name))
    assert offenders == [], f"physical table name(s) leaked into feature code: {offenders}"
