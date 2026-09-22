from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb

from predictor.config import settings

POINT_IN_TIME_TABLES = frozenset(
    {"games", "injury_status", "odds_snapshots", "news_items"}
)


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
CREATE TABLE IF NOT EXISTS games (
    game_id       VARCHAR NOT NULL,
    season        VARCHAR NOT NULL,
    game_date     DATE NOT NULL,
    home_team     VARCHAR NOT NULL,
    away_team     VARCHAR NOT NULL,
    home_points   INTEGER,
    away_points   INTEGER,
    status        VARCHAR NOT NULL,
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
CREATE TABLE IF NOT EXISTS injury_status (
    report_date   DATE NOT NULL,
    game_date     DATE NOT NULL,
    matchup       VARCHAR,
    team          VARCHAR NOT NULL,
    player        VARCHAR NOT NULL,
    status        VARCHAR NOT NULL,
    reason        VARCHAR,
    reconstructed BOOLEAN NOT NULL DEFAULT FALSE,
    observed_at   TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (observed_at, team, player, game_date)
);

CREATE TABLE IF NOT EXISTS odds_snapshots (
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

CREATE TABLE IF NOT EXISTS news_items (
    item_key      VARCHAR NOT NULL,
    feed          VARCHAR NOT NULL,
    title         VARCHAR,
    link          VARCHAR,
    summary       VARCHAR,
    observed_at   TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (item_key)
);

CREATE TABLE IF NOT EXISTS ingest_runs (
    source        VARCHAR NOT NULL,
    started_at    TIMESTAMP WITH TIME ZONE NOT NULL,
    finished_at   TIMESTAMP WITH TIME ZONE,
    ok            BOOLEAN,
    detail        VARCHAR
);
"""


def connect(path: Path | None = None) -> duckdb.DuckDBPyConnection:
    target = path or settings.db_path
    target.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(target))
    # Without this, DuckDB returns timestamptz values in the machine's local
    # zone, so identical code yields different-looking results per machine.
    con.execute("SET TimeZone='UTC'")
    return con


def migrate(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(_SCHEMA)
