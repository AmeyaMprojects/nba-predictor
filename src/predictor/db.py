from __future__ import annotations

import os
import time
from collections.abc import KeysView
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import duckdb

from predictor.config import PROJECT_ROOT, settings

# FIX 1 (final review, part 1): the real, irreplaceable archive. A test run
# (PYTEST_CURRENT_TEST is set by pytest itself for the duration of every
# test) must never open this file -- see the rail in connect() below and
# tests/conftest.py, which points settings.data_dir at a throwaway temp
# directory before predictor.config is ever imported.
_REAL_DB_PATH = PROJECT_ROOT / "data" / "predictor.duckdb"

# Logical name (what feature code and AsOfView callers use) -> physical
# table name (what actually exists in the DuckDB catalog). The physical
# names are deliberately NOT "games", "injury_status", etc. -- see FIX 1 in
# the task-6 hardening report: con.sql()/.project()/.aggregate()/.filter()/
# .query()/.join()/.union() all resolve identifiers against the connection's
# catalog, completely bypassing AsOfView's observed_at <= cutoff filter, for
# ANY caller who happens to name a real table in a SQL fragment passed to
# one of those methods. Renaming the physical tables to a "_raw" suffix that
# nobody would type by accident makes that mistake structurally impossible:
# there is no table literally named "games" for a stray "FROM games" to
# resolve to.
POINT_IN_TIME_TABLES: Mapping[str, str] = MappingProxyType(
    {
        "games": "games_raw",
        "injury_status": "injury_status_raw",
        "odds_snapshots": "odds_snapshots_raw",
        "news_items": "news_items_raw",
        "schedule": "schedule_raw",
    }
)


def point_in_time_logical_names() -> KeysView[str]:
    """The logical point-in-time table names (what callers pass to AsOfView)."""
    return POINT_IN_TIME_TABLES.keys()


def require_utc(value: datetime, field: str = "observed_at") -> datetime:
    """Reject naive or non-UTC datetimes before they reach DuckDB.

    DuckDB silently accepts a naive datetime and stores it as if it were
    already UTC wall-clock, with no error. That is exactly how a future
    ingestion path using ``datetime.utcnow()`` instead of
    ``datetime.now(UTC)`` would silently corrupt point-in-time data. Every
    write path for a point-in-time column must call this first. Mirrors the
    equivalent guard in ``raw_store.store()`` for the file archive.
    """
    if value.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    offset = value.utcoffset()
    if offset is not None and offset.total_seconds() != 0:
        raise ValueError(
            f"{field} must be UTC (zero UTC offset); got an offset of {offset} instead"
        )
    return value


_SCHEMA = """
CREATE TABLE IF NOT EXISTS games_raw (
    game_id       VARCHAR NOT NULL,
    season        VARCHAR NOT NULL,
    game_date     DATE NOT NULL,
    home_team     VARCHAR NOT NULL,
    away_team     VARCHAR NOT NULL,
    home_points   INTEGER,
    away_points   INTEGER,
    status        VARCHAR NOT NULL,
    reconstructed BOOLEAN NOT NULL DEFAULT FALSE,
    observed_at   TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (game_id, observed_at)
);

-- A single injury report can span two different game dates (e.g. a
-- back-to-back), and the same player can appear once per game date within
-- one report. PRIMARY KEY (observed_at, team, player) alone collides on
-- that case, and since ingestion uses INSERT OR REPLACE, a collision
-- silently deletes a row instead of erroring. game_date is part of the key
-- to prevent that. game_date cannot be NULL because it is part of the
-- primary key; if a row's game date is unparseable, ingestion must
-- substitute the report's own publication (report_date) rather than
-- dropping the row -- losing an injury row is worse than an imperfect date.
-- team/player hold the CANONICAL key form (a 3-letter team abbreviation --
-- also the join key against games_raw.home_team/away_team -- and a
-- whitespace-stripped player name); team_display/player_display hold the
-- original human-readable text as parsed from the PDF ("Golden State
-- Warriors" / "Curry, Stephen"). See the idempotent
-- `_add_injury_normalization_columns` migration below and its docstring
-- for why this is columns rather than an in-place rewrite, and
-- `injury_report.ingest_report` for where the split happens. game_time
-- (e.g. "07:00(ET)") is parsed from every report; without it there is no
-- tip-off time anywhere in the schema for a consumer to compute a correct
-- per-game cutoff from.
CREATE TABLE IF NOT EXISTS injury_status_raw (
    report_date    DATE NOT NULL,
    game_date      DATE NOT NULL,
    game_time      VARCHAR,
    matchup        VARCHAR,
    team           VARCHAR NOT NULL,
    team_display   VARCHAR,
    player         VARCHAR NOT NULL,
    player_display VARCHAR,
    status         VARCHAR NOT NULL,
    reason         VARCHAR,
    reconstructed  BOOLEAN NOT NULL DEFAULT FALSE,
    observed_at    TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (observed_at, team, player, game_date)
);

CREATE TABLE IF NOT EXISTS odds_snapshots_raw (
    game_key      VARCHAR NOT NULL,
    book          VARCHAR NOT NULL,
    home_team     VARCHAR NOT NULL,
    away_team     VARCHAR NOT NULL,
    home_price    INTEGER,
    away_price    INTEGER,
    spread        DOUBLE,
    total         DOUBLE,
    observed_at   TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (game_key, book, observed_at)
);

CREATE TABLE IF NOT EXISTS news_items_raw (
    item_key      VARCHAR NOT NULL,
    feed          VARCHAR NOT NULL,
    title         VARCHAR,
    link          VARCHAR,
    summary       VARCHAR,
    observed_at   TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (item_key)
);

-- The league schedule (nba_api ScheduleLeagueV2), one row per game per
-- fetch. observed_at is the real fetch time, so daily fetches accumulate
-- genuine schedule vintages from 2026-09-27 onward; rows for games already
-- played at fetch time are post-hoc. game_date is the Eastern calendar
-- date. tip_off_utc is NULL when the league lists the time as TBD -- a
-- placeholder must never become a tip-off, because the harness takes the
-- MINIMUM across vintages and a 00:00 placeholder would win forever.
-- is_neutral_reported is the league's own flag, which is false for every
-- game before 2024-25 (Paris, Mexico City and Las Vegas included);
-- is_neutral also marks a game whose arena differs from the home team's
-- usual regular-season venue (see sources/schedule.py). Scores, game
-- status and team records are deliberately NOT columns: the endpoint
-- carries them, and a field that is not stored cannot leak.
CREATE TABLE IF NOT EXISTS schedule_raw (
    game_id             VARCHAR NOT NULL,
    season              VARCHAR NOT NULL,
    game_date           DATE NOT NULL,
    tip_off_utc         TIMESTAMP WITH TIME ZONE,
    home_team           VARCHAR NOT NULL,
    away_team           VARCHAR NOT NULL,
    arena_name          VARCHAR,
    arena_city          VARCHAR,
    arena_state         VARCHAR,
    is_neutral_reported BOOLEAN NOT NULL,
    is_neutral          BOOLEAN NOT NULL,
    observed_at         TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (game_id, observed_at)
);

CREATE TABLE IF NOT EXISTS ingest_runs (
    source        VARCHAR NOT NULL,
    started_at    TIMESTAMP WITH TIME ZONE NOT NULL,
    finished_at   TIMESTAMP WITH TIME ZONE,
    ok            BOOLEAN,
    detail        VARCHAR
);
"""


def connect(
    path: Path | None = None, *, read_only: bool = False
) -> duckdb.DuckDBPyConnection:
    target = path or settings.db_path
    # FIX 1 rail: this must trip even when a caller passes an explicit
    # `path` pointing at the real file, not just when it falls back to
    # `settings.db_path` -- a regression could reintroduce either. Resolved
    # so a relative or symlinked path pointing at the same file cannot slip
    # past a straight string/Path equality check.
    if os.environ.get("PYTEST_CURRENT_TEST") and target.resolve() == _REAL_DB_PATH.resolve():
        raise RuntimeError(
            f"refusing to open the real database at {target} during a test "
            "run (PYTEST_CURRENT_TEST is set). Tests must never touch the "
            "real, irreplaceable archive -- point PREDICTOR_DATA_DIR at a "
            "throwaway directory, or pass db.connect() a tmp_path-based "
            "path, instead."
        )
    if read_only:
        con = duckdb.connect(str(target), read_only=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(str(target))
    # Without this, DuckDB returns timestamptz values in the machine's local
    # zone, so identical code yields different-looking results per machine.
    con.execute("SET TimeZone='UTC'")
    return con


def connect_with_retry(
    path: Path | None = None,
    *,
    read_only: bool = False,
    attempts: int = 6,
    wait_seconds: float = 20.0,
    sleep=time.sleep,
) -> duckdb.DuckDBPyConnection:
    """connect(), waiting out another predictor process's write lock.

    DuckDB allows one writer per file. The news and schedule launchd jobs
    both fire on wake after a laptop sleeps through their slots, so one of
    them routinely finds the other holding the lock for a few seconds.
    Only that specific error is retried (matched on DuckDB's own message,
    as the backtest command does); anything else raises immediately.
    ``read_only`` is passed to every attempt (a read-only open still
    conflicts with another process's write lock).
    """
    for attempt in range(1, attempts + 1):
        try:
            return connect(path, read_only=read_only)
        except duckdb.Error as exc:
            if "conflicting lock is held" not in str(exc).lower() or attempt == attempts:
                raise
            sleep(wait_seconds)
    raise AssertionError("unreachable")


def migrate(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(_SCHEMA)
    _add_games_reconstructed_column(con)
    _add_injury_normalization_columns(con)


def _add_games_reconstructed_column(con: duckdb.DuckDBPyConnection) -> None:
    """Idempotent upgrade for databases created before `reconstructed` existed.

    `_SCHEMA` above uses CREATE TABLE IF NOT EXISTS, which is a no-op on a
    database that already has `games_raw` without this column -- it does
    NOT retroactively add it. This runs on every migrate() call (including
    against a fresh database, where it is a harmless no-op since the
    column already exists) so an existing database picks up the column
    without any separate one-time script. DuckDB's ALTER TABLE ADD COLUMN
    does not support inline NOT NULL, so the constraint is applied as a
    separate step after backfilling any existing rows -- both steps are
    safe to repeat: ADD COLUMN IF NOT EXISTS is a no-op once the column is
    present, the UPDATE only touches rows that are still NULL (none, after
    the first run), and SET NOT NULL on an already-NOT-NULL column
    succeeds without error. This never touches injury_status_raw or any
    other table.
    """
    con.execute("ALTER TABLE games_raw ADD COLUMN IF NOT EXISTS reconstructed BOOLEAN DEFAULT FALSE")
    con.execute("UPDATE games_raw SET reconstructed = FALSE WHERE reconstructed IS NULL")
    con.execute("ALTER TABLE games_raw ALTER COLUMN reconstructed SET NOT NULL")


def _add_injury_normalization_columns(con: duckdb.DuckDBPyConnection) -> None:
    """Idempotent upgrade for databases created before C2's fix existed.

    C2 (final whole-branch review): the two injury-report PDF layouts
    render the same player/team differently ("Curry, Stephen" vs
    "Curry,Stephen", "Miami Heat" vs "MiamiHeat"), splitting 41% of
    players into two identities and leaving NO join key at all between
    injury_status_raw and games_raw. The fix normalizes `team`/`player`
    themselves into the canonical key form (a 3-letter team abbreviation
    -- the same form games_raw already uses -- and a whitespace-stripped
    player name) at ingest time, going forward.

    Columns, not an in-place rewrite of existing rows here, because: (1)
    normalizing requires re-deriving the canonical form from the ORIGINAL
    parsed text, which this migration does not have -- only a full
    `reingest-injuries` re-parse of the archived PDFs (already the
    documented recovery path for any injury_report fix) can actually
    populate correct values, exactly like `reconstructed` above; a blind
    SQL rewrite of existing `team`/`player` values here could not resolve
    "MiamiHeat" back to "MIA" without the same team-name table
    `injury_report.py` already has, and duplicating that here would be a
    second copy of the same knowledge; (2) a human-readable display form
    is an explicit part of the fix's requirements, which an in-place
    rewrite would have nowhere to keep. This mirrors
    `_add_games_reconstructed_column`: safe to call on a fresh database
    (all three ADD COLUMNs are no-ops once the columns exist) and safe to
    call repeatedly on an existing one. Pre-migration rows are left with
    NULL team_display/player_display/game_time until the next
    `reingest-injuries` run repopulates them -- NULL, not a guessed value,
    is the honest state for data this migration cannot itself derive.
    """
    con.execute("ALTER TABLE injury_status_raw ADD COLUMN IF NOT EXISTS team_display VARCHAR")
    con.execute("ALTER TABLE injury_status_raw ADD COLUMN IF NOT EXISTS player_display VARCHAR")
    con.execute("ALTER TABLE injury_status_raw ADD COLUMN IF NOT EXISTS game_time VARCHAR")
