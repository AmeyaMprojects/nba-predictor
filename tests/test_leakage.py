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
    for table in db.point_in_time_logical_names():
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


# --- Round 2, FINDING A [Critical] pins: catalog shadowing cannot leak ----
# future data back THROUGH AsOfView itself.
#
# Adversarial re-review found that table()/latest() emitted an UNQUALIFIED
# identifier ("FROM games_raw"). DuckDB resolves an unqualified (or only
# schema-qualified) name through the temp/registered-view namespace BEFORE
# the real database. So ANY code sharing this connection that binds the
# name "games_raw" -- via .query() with that virtual-table alias,
# con.register(), or CREATE TEMP VIEW -- silently replaces what every
# table()/latest() call reads, even for callers who only ever touch
# AsOfView and never name a table themselves. A later USE on the same
# connection has the same effect for a different reason (it changes which
# database an unqualified/schema-qualified name resolves against). The fix
# is to qualify every reference with the database name captured once at
# construction. These tests pin all four vectors.


def test_query_alias_shadowing_the_raw_table_does_not_leak(leaky_con):
    view = AsOfView(leaky_con, TIP_OFF - timedelta(minutes=30))
    assert view.table("games").project("status, home_points").fetchall() == [
        ("SCHEDULED", None)
    ]

    leaky_con.execute(
        "CREATE TABLE staging AS SELECT game_id, season, game_date, home_team,"
        " away_team, home_points, away_points, status,"
        " observed_at - INTERVAL 10 YEAR AS observed_at FROM"
        f' "{view._db_name}"."main"."games_raw"'
    )
    # This is the exact I-1-style escape: registering an alias named after
    # the physical table on the shared connection.
    leaky_con.sql("SELECT * FROM staging").query("games_raw", "SELECT 1")

    rows = view.table("games").project("status, home_points").fetchall()
    assert rows == [("SCHEDULED", None)], f"future data leaked through table(): {rows}"
    latest_rows = view.latest("games").project("status, home_points").fetchall()
    assert latest_rows == [
        ("SCHEDULED", None)
    ], f"future data leaked through latest(): {latest_rows}"


def test_con_register_shadowing_the_raw_table_does_not_leak(leaky_con):
    view = AsOfView(leaky_con, TIP_OFF - timedelta(minutes=30))
    assert view.table("games").project("status").fetchall() == [("SCHEDULED",)]

    import pandas as pd

    poisoned = pd.DataFrame(
        {"status": ["FINAL"], "observed_at": [TIP_OFF + timedelta(hours=3)]}
    )
    leaky_con.register("games_raw", poisoned)
    try:
        rows = view.table("games").project("status").fetchall()
        assert rows == [("SCHEDULED",)], f"future data leaked through table(): {rows}"
    finally:
        leaky_con.unregister("games_raw")


def test_create_temp_view_shadowing_the_raw_table_does_not_leak(leaky_con):
    view = AsOfView(leaky_con, TIP_OFF - timedelta(minutes=30))
    assert view.table("games").project("status").fetchall() == [("SCHEDULED",)]

    leaky_con.execute(
        "CREATE TABLE staging2 AS SELECT game_id, season, game_date, home_team,"
        " away_team, home_points, away_points, status,"
        " observed_at - INTERVAL 10 YEAR AS observed_at FROM"
        f' "{view._db_name}"."main"."games_raw"'
    )
    leaky_con.execute("CREATE TEMP VIEW games_raw AS SELECT * FROM staging2")
    try:
        rows = view.table("games").project("status").fetchall()
        assert rows == [("SCHEDULED",)], f"future data leaked through table(): {rows}"
    finally:
        leaky_con.execute("DROP VIEW games_raw")


def test_use_other_database_does_not_repoint_the_view(leaky_con, tmp_path):
    view = AsOfView(leaky_con, TIP_OFF - timedelta(minutes=30))
    assert view.table("games").project("status").fetchall() == [("SCHEDULED",)]

    leaky_con.execute(f"ATTACH '{tmp_path / 'other.duckdb'}' AS other")
    leaky_con.execute("USE other")
    try:
        # The view was constructed against the original database; USE must
        # not silently re-point it at (or break it against) "other".
        rows = view.table("games").project("status").fetchall()
        assert rows == [("SCHEDULED",)], f"USE re-pointed the view: {rows}"
    finally:
        leaky_con.execute(f'USE "{view._db_name}"')


# --- FIX 5: the boundary's enforcement mechanism for future feature code --
#
# Round 2 widened this scan (Finding D): it previously walked src/predictor
# only and globbed *.py only, missing a scripts/ directory (this project
# already has one, wired to launchd) and any .sql files. It now walks the
# whole repository -- excluding data/ (never touched, per the absolute
# constraint on this task), .venv/, .git/, .superpowers/, and __pycache__ --
# and covers both .py and .sql. tests/ is also excluded: test fixtures
# legitimately write physical "_raw" names directly, simulating the
# ingestion writer code that does not exist yet; that is sanctioned setup,
# not feature code reading around AsOfView.


def test_no_physical_table_name_appears_outside_db_and_asof():
    """The physical "_raw" table names must only ever be spelled in db.py
    (schema DDL) and asof.py (the resolver). Any other module that contains
    one of these names in a string literal is feature code reading
    point-in-time data directly, bypassing AsOfView -- exactly the mistake
    FIX 1 makes structurally hard, but only if nothing else names the raw
    tables either.
    """
    import os

    repo_root = Path(__file__).resolve().parents[1]
    excluded_dirs = {
        "data",
        ".venv",
        ".git",
        ".superpowers",
        "tests",
        "__pycache__",
        ".pytest_cache",
    }
    allowed = {
        (repo_root / "src" / "predictor" / "db.py").resolve(),
        (repo_root / "src" / "predictor" / "asof.py").resolve(),
    }
    physical_names = set(db.POINT_IN_TIME_TABLES.values())

    offenders = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        # Prune BEFORE descending -- this never lists or reads anything
        # under data/, satisfying the absolute constraint on this task.
        dirnames[:] = [d for d in dirnames if d not in excluded_dirs]
        for filename in filenames:
            if not (filename.endswith(".py") or filename.endswith(".sql")):
                continue
            path = Path(dirpath) / filename
            if path.resolve() in allowed:
                continue
            text = path.read_text(errors="ignore")
            for name in physical_names:
                if name in text:
                    offenders.append((str(path), name))
    assert offenders == [], f"physical table name(s) leaked into feature code: {offenders}"


# --- Round 2, FINDING C [Important]: latest() error handling -------------


def test_default_latest_key_covers_every_point_in_time_table():
    """_DEFAULT_LATEST_KEY must not silently drift from POINT_IN_TIME_TABLES.

    Without this, a table added to POINT_IN_TIME_TABLES but not to
    _DEFAULT_LATEST_KEY would raise a raw KeyError the first time someone
    called latest() on it with no explicit key, instead of a clear
    AsOfError -- exactly the kind of drift test_point_in_time_tables_
    matches_schema_exactly guards for the table mapping itself.
    """
    from predictor.asof import _DEFAULT_LATEST_KEY

    assert set(_DEFAULT_LATEST_KEY) == set(db.point_in_time_logical_names())


def test_latest_on_table_missing_default_key_raises_asof_error(con, monkeypatch):
    from predictor import asof as asof_module

    monkeypatch.delitem(asof_module._DEFAULT_LATEST_KEY, "games")
    view = AsOfView(con, TIP_OFF)
    with pytest.raises(AsOfError, match="no default latest\\(\\) key"):
        view.latest("games")


def test_latest_rejects_empty_key(con):
    view = AsOfView(con, TIP_OFF)
    with pytest.raises(AsOfError, match="must not be empty"):
        view.latest("games", key=())


def test_latest_rejects_bare_string_key(con):
    view = AsOfView(con, TIP_OFF)
    with pytest.raises(AsOfError, match="bare"):
        view.latest("games", key="game_id")


# --- Round 2, FINDING F [Minor]: latest() coverage across every table ----


def test_latest_default_key_works_for_every_point_in_time_table(con):
    """latest() with no explicit key is exercised by test_asof.py only for
    injury_status. Prove every default key in _DEFAULT_LATEST_KEY is a real,
    usable partition for its table, so a stale default against a future
    schema change fails here instead of only in production.
    """
    view = AsOfView(con, TIP_OFF)
    for table in db.point_in_time_logical_names():
        view.latest(table).fetchall()
