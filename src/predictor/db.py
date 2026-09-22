from __future__ import annotations

from pathlib import Path

import duckdb

from predictor.config import settings

POINT_IN_TIME_TABLES = frozenset(
    {"games", "injury_status", "odds_snapshots", "news_items"}
)

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

CREATE TABLE IF NOT EXISTS injury_status (
    report_date   DATE NOT NULL,
    game_date     DATE,
    matchup       VARCHAR,
    team          VARCHAR NOT NULL,
    player        VARCHAR NOT NULL,
    status        VARCHAR NOT NULL,
    reason        VARCHAR,
    reconstructed BOOLEAN NOT NULL DEFAULT FALSE,
    observed_at   TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (observed_at, team, player)
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
