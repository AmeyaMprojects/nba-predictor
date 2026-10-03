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
from datetime import UTC, datetime

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
    # game_ids `pair_team_rows` could not resolve into a GameRow at all
    # (e.g. a GAME_ID group with other than 2 team-rows, or disagreeing
    # home/away parses) -- never silently lost, see `download()` below.
    dropped: list[str]


@dataclass(frozen=True)
class CaptureResult:
    season: str
    new_finals: int
    already_known: int
    blob_key: str
    dropped: list[str]


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
    Two anomalies are never silently swallowed, only logged and surfaced on
    the returned object for the caller (the CLI) to report loudly:

    - A GAME_ID group `pair_team_rows` could not resolve at all (wrong row
      count, unparseable/disagreeing MATCHUP text) -- collected in
      `.dropped`, mirroring `nba_stats.ingest_season`'s own `dropped` list.
    - A game with a score for only ONE team -- genuinely corrupt/partial
      data, not an ordinary not-yet-played game (which has NEITHER score).
      `pair_team_rows` cannot tell these apart itself (both come back
      `status="SCHEDULED"`), so it is detected here and printed loudly;
      the game is still excluded from `.games` (it is not a confirmed
      FINAL), but silently treating it exactly like an unplayed game would
      hide a real data problem.
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
    dropped: list[nba_stats.DroppedGame] = []
    paired = nba_stats.pair_team_rows(archived, season, dropped=dropped)

    partial_score = [
        g for g in paired if (g.home_points is None) != (g.away_points is None)
    ]
    if partial_score:
        ids = ", ".join(g.game_id for g in partial_score)
        print(
            f"results: WARNING -- {len(partial_score)} game(s) have a score "
            f"for only one team (partial/corrupt data), NOT treated as "
            f"final: {ids}"
        )

    games = [game for game in paired if game.status == "FINAL"]
    return Downloaded(
        season=season,
        fetched_at=fetched_at,
        blob_key=key,
        games=games,
        dropped=[d.game_id for d in dropped],
    )


def load(con, downloaded: Downloaded) -> CaptureResult:
    """Record newly FINAL games from one `download()` result, all-or-nothing.

    For each FINAL GameRow whose game_id has no FINAL row in the games
    table yet, this inserts a FINAL row stamped `observed_at =
    downloaded.fetched_at`, `reconstructed = False` -- a real captured
    result, not a derived one. A game that already has ANY FINAL row
    (whether captured live by an earlier run of this command, or
    reconstructed by the historical `nba_stats.ingest_season` backfill) is
    counted in `already_known` and never inserted again.

    No companion SCHEDULED row is written for a game with no row at all.
    An earlier draft of this function added one (stamped at the same
    capture instant) so that `backtest.replay`'s earliest-SCHEDULED sanity
    bound (see replay.py's FIX 7/21) would have something to find -- but
    that stub is stamped AFTER tip-off (capture happens once a game is
    already final), so it made the sanity bound itself backwards: every
    live-captured game's cutoff (tip - buffer) would then fall BEFORE this
    "earliest SCHEDULED" timestamp, and replay would skip the game as
    "buffer too early" -- mislabelling it an OBSERVED schedule timestamp
    besides, which it is not. `replay.py` already handles a game with NO
    SCHEDULED row in the games table correctly: `earliest_scheduled` comes
    back `None` and that particular sanity bound is simply not applied for it
    (the real leak guard -- "is this game's own FINAL row already visible
    at the cutoff" -- does not depend on a SCHEDULED row at all). See
    `tests/test_results_capture.py::test_captured_result_is_predicted_by_replay`.

    Rolls back and re-raises on any error -- nothing is left half-written.
    """
    table = db.POINT_IN_TIME_TABLES["games"]
    observed_at = db.require_utc(downloaded.fetched_at, "fetched_at")
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
        dropped=downloaded.dropped,
    )
