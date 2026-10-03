"""Captured NBA game results (nba_api LeagueGameFinder) -- live result capture.

Unlike `nba_stats.ingest_season` (which DERIVES observed_at from game_date
for a one-shot historical backfill -- see its docstring), this module is the
LIVE path: `fetched_at` is the real wall-clock moment this process asked the
NBA for results, so a FINAL row written here carries a genuinely OBSERVED
capture time, not a reconstructed one. That is what lets
`backtest.replay`'s leak guard (see its FIX 2 note) start doing real work --
a FINAL row from this module is only ever visible to an `AsOfView` cut at or
after the real moment the result was captured.

Raw-first: the fetched DataFrame is archived (gzipped JSON, `orient="split"`)
before it is parsed, and the parser reads the archived bytes back -- same
discipline as `sources/schedule.py`.
"""

from __future__ import annotations

import gzip
import io
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pandas as pd

from predictor import db, raw_store
from predictor.sources import nba_stats

SOURCE = "results"

# A round-trip through `DataFrame.to_json(orient="split")` -> JSON -> back
# loses GAME_ID's leading zeros and its string dtype: a purely-numeric JSON
# string like "0022600001" is otherwise silently re-inferred as the int
# 22600001 by `pd.read_json`'s default dtype inference. That would break
# every later lookup keyed on the full, zero-padded game_id (notably the
# "does this game already have a FINAL row" check in `load()` below, and
# `pair_team_rows`' own `str(game_id)` grouping). Forcing the column back to
# `str` on read is what makes the archived bytes round-trip byte-for-byte
# equivalent in meaning, not just in appearance.
_READ_DTYPE = {"GAME_ID": str}


@dataclass(frozen=True)
class Downloaded:
    season: str
    fetched_at: datetime
    blob_key: str
    games: list[nba_stats.GameRow]


@dataclass(frozen=True)
class CaptureResult:
    season: str
    new_finals: int
    already_known: int
    blob_key: str


def archive_key(season: str, fetched_at: datetime) -> str:
    return f"{season}_{fetched_at:%Y%m%dT%H%M%S}Z.json.gz"


def download(
    season: str,
    fetched_at: datetime | None = None,
    fetch: Callable[[str], pd.DataFrame] = nba_stats.fetch_season,
) -> Downloaded:
    """Fetch, archive and parse one season's results. Touches no database.

    Raw-first: `fetch(season)` is archived BEFORE it is parsed, and parsing
    reads the archived bytes back via `raw_store.load` -- so what this
    returns is provably what got archived, not merely what was fetched (a
    parser bug stays re-parsable instead of losing the original payload).

    Only FINAL games (both scores present) are kept in `.games` -- a
    scheduled-but-unplayed game carries nothing new for `load()` to record.
    """
    fetched_at = db.require_utc(
        fetched_at if fetched_at is not None else datetime.now(UTC), "fetched_at"
    )
    df = fetch(season)
    key = archive_key(season, fetched_at)
    payload = df.to_json(orient="split", date_format="iso").encode()
    raw_store.store(
        SOURCE, key, gzip.compress(payload, mtime=0), fetched_at, meta={"season": season}
    )
    archived = pd.read_json(
        io.BytesIO(gzip.decompress(raw_store.load(SOURCE, key))),
        orient="split",
        dtype=_READ_DTYPE,
    )
    games = [
        game
        for game in nba_stats.pair_team_rows(archived, season)
        if game.status == "FINAL"
    ]
    return Downloaded(season=season, fetched_at=fetched_at, blob_key=key, games=games)


# See the long comment in `load()` below: a brand-new game (no row at all
# yet) gets a SCHEDULED stub alongside its FINAL row so that
# `backtest.replay`'s earliest-SCHEDULED sanity bound has something to find.
# Both rows would otherwise collide on the games table's PRIMARY KEY
# (game_id, observed_at) if stamped with the exact same instant, so the stub
# is stamped one microsecond earlier -- functionally the same capture moment
# for every real purpose (nothing reads observed_at at sub-second
# granularity), but a distinct primary key.
_STUB_LEAD = timedelta(microseconds=1)


def load(con, downloaded: Downloaded) -> CaptureResult:
    """Record newly FINAL games from one `download()` result, all-or-nothing.

    For each FINAL GameRow whose game_id has no FINAL row in the games
    table yet, this inserts a FINAL row stamped `observed_at =
    downloaded.fetched_at`, `reconstructed = False` -- a real captured
    result, not a derived one. A game that already has ANY FINAL row
    (whether captured live by an earlier run of this command, or
    reconstructed by the historical `nba_stats.ingest_season` backfill) is
    counted in `already_known` and never inserted again.

    A game with NO row at all yet (the schedule ingest never saw it --
    expected to be rare) also gets a SCHEDULED stub (NULL points) alongside
    its FINAL row, so `backtest.replay`'s earliest-SCHEDULED sanity check
    (see replay.py's FIX 7/21) finds something instead of silently skipping
    its bound for this game; see `_STUB_LEAD` above for why the stub's
    timestamp is one microsecond earlier rather than identical.

    Rolls back and re-raises on any error -- nothing is left half-written.
    """
    table = db.POINT_IN_TIME_TABLES["games"]
    observed_at = db.require_utc(downloaded.fetched_at, "fetched_at")
    stub_at = observed_at - _STUB_LEAD
    new_finals = 0
    already_known = 0
    con.execute("BEGIN")
    try:
        for game in downloaded.games:
            has_final = con.execute(
                f"SELECT 1 FROM {table} WHERE game_id = ? AND status = 'FINAL' LIMIT 1",
                [game.game_id],
            ).fetchone()
            if has_final is not None:
                already_known += 1
                continue

            has_any_row = con.execute(
                f"SELECT 1 FROM {table} WHERE game_id = ? LIMIT 1",
                [game.game_id],
            ).fetchone()
            if has_any_row is None:
                con.execute(
                    f"INSERT INTO {table} (game_id, season, game_date, home_team,"
                    " away_team, home_points, away_points, status, reconstructed,"
                    " observed_at) VALUES (?,?,?,?,?,NULL,NULL,'SCHEDULED',FALSE,?)",
                    [
                        game.game_id,
                        game.season,
                        game.game_date,
                        game.home_team,
                        game.away_team,
                        stub_at,
                    ],
                )

            con.execute(
                f"INSERT INTO {table} (game_id, season, game_date, home_team,"
                " away_team, home_points, away_points, status, reconstructed,"
                " observed_at) VALUES (?,?,?,?,?,?,?,'FINAL',FALSE,?)",
                [
                    game.game_id,
                    game.season,
                    game.game_date,
                    game.home_team,
                    game.away_team,
                    game.home_points,
                    game.away_points,
                    observed_at,
                ],
            )
            new_finals += 1
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise

    return CaptureResult(
        season=downloaded.season,
        new_finals=new_finals,
        already_known=already_known,
        blob_key=downloaded.blob_key,
    )
