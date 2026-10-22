"""Player game logs (nba_api LeagueGameLog, player mode), stored point-in-time.

One row per player per game, taken from that game's own box score. Only
per-game logs are used: an endpoint that returns season totals or
season-level estimates would carry the whole season into every earlier
date, so none is ever called from this package (a source-scan test
enforces it).

Raw-first, like `results.py`: each fetched DataFrame is archived (gzipped
JSON, `orient="split"`, `mtime=0`) before it is parsed, and the parser reads
the archived bytes back with GAME_ID forced to a string so its leading
zeros survive.

`observed_at` is derived exactly like a reconstructed FINAL games row
(`nba_stats._derive_observed_at`: game date + 1 day, 12:00 UTC), so every
row here is `reconstructed=TRUE`, `source='nba_stats'`.

Season types: `SEASON_TYPES` are the ones whose logs feed player ratings
and minutes. Preseason logs ("Pre Season") are fetched too
(`ALL_SEASON_TYPES`) and stored with `season_type='preseason'`, but only
for roster membership -- so a summer signing or trade counts from his first
preseason game. They must never feed ratings or minutes.
"""

from __future__ import annotations

import gzip
import io
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
from tenacity import retry, stop_after_attempt, wait_exponential

from predictor import db, raw_store
from predictor.config import season_label
from predictor.sources import nba_stats

SOURCE = "player_games"
DB_SOURCE = "nba_stats"

PRESEASON = "Pre Season"
SEASON_TYPES = ("Regular Season", "PlayIn", "Playoffs")
ALL_SEASON_TYPES = (PRESEASON, *SEASON_TYPES)

# nba_api season-type name -> the `season_type` value stored in the table.
SEASON_TYPE_TAGS = {
    PRESEASON: "preseason",
    "Regular Season": "regular",
    "PlayIn": "playin",
    "Playoffs": "playoffs",
}

FIRST_SEASON = "2014-15"

# See results._READ_DTYPE: without this, "0022300001" reads back as an int.
_READ_DTYPE = {"GAME_ID": str}

_BOX_COLUMNS = (
    ("pts", "PTS"),
    ("fgm", "FGM"),
    ("fga", "FGA"),
    ("fg3m", "FG3M"),
    ("fg3a", "FG3A"),
    ("ftm", "FTM"),
    ("fta", "FTA"),
    ("oreb", "OREB"),
    ("dreb", "DREB"),
    ("ast", "AST"),
    ("stl", "STL"),
    ("blk", "BLK"),
    ("tov", "TOV"),
    ("pf", "PF"),
)


@dataclass(frozen=True)
class PlayerGameRow:
    game_id: str
    season: str
    game_date: date
    player_id: int
    player_name: str
    team: str
    minutes: float | None
    pts: int | None
    fgm: int | None
    fga: int | None
    fg3m: int | None
    fg3a: int | None
    ftm: int | None
    fta: int | None
    oreb: int | None
    dreb: int | None
    ast: int | None
    stl: int | None
    blk: int | None
    tov: int | None
    pf: int | None
    plus_minus: float | None


@dataclass(frozen=True)
class Downloaded:
    season: str
    season_type: str
    fetched_at: datetime
    blob_key: str
    rows: list[PlayerGameRow]


@dataclass(frozen=True)
class LoadResult:
    season: str
    season_type: str
    rows_written: int
    games: int
    # Game ids with no row at all in the games table. Their rows are still
    # stored -- reported, never dropped.
    unknown_game_ids: list[str]


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=30))
def fetch_season_type(season: str, season_type: str) -> pd.DataFrame:
    """Every player-game row for one season and season type."""
    log = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star=season_type,
        player_or_team_abbreviation="P",
        league_id="00",
        timeout=60,
    )
    return log.get_data_frames()[0]


def archive_key(season: str, season_type: str, fetched_at: datetime) -> str:
    return f"{season}_{season_type.lower().replace(' ', '-')}_{fetched_at:%Y%m%dT%H%M%S}Z.json.gz"


def default_seasons(now: datetime) -> list[str]:
    """Every season from FIRST_SEASON through the one `now` belongs to."""
    last = int(season_label(now)[:4])
    first = int(FIRST_SEASON[:4])
    return [f"{y}-{str(y + 1)[-2:]}" for y in range(first, last + 1)]


def _missing(value) -> bool:
    return value is None or (not isinstance(value, str) and pd.isna(value))


def _as_int(value) -> int | None:
    if _missing(value):
        return None
    return int(value)


def _as_float(value) -> float | None:
    if _missing(value):
        return None
    return float(value)


def _as_minutes(value) -> float | None:
    """Minutes as a number; also accepts clock-style "MM:SS" text."""
    if _missing(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if ":" in text:
            mins, secs = text.split(":", 1)
            return int(mins) + int(secs) / 60
        return float(text)
    return float(value)


def _parse(frame: pd.DataFrame, season: str) -> list[PlayerGameRow]:
    rows = []
    for rec in frame.to_dict("records"):
        rows.append(
            PlayerGameRow(
                game_id=str(rec["GAME_ID"]),
                season=season,
                game_date=pd.to_datetime(rec["GAME_DATE"]).date(),
                player_id=int(rec["PLAYER_ID"]),
                player_name=str(rec["PLAYER_NAME"]),
                team=str(rec["TEAM_ABBREVIATION"]),
                minutes=_as_minutes(rec.get("MIN")),
                **{field: _as_int(rec.get(col)) for field, col in _BOX_COLUMNS},
                plus_minus=_as_float(rec.get("PLUS_MINUS")),
            )
        )
    return rows


def download(
    season: str,
    season_type: str,
    fetched_at: datetime | None = None,
    fetch: Callable[[str, str], pd.DataFrame] = fetch_season_type,
) -> Downloaded:
    """Fetch, archive and parse one season type's logs. Touches no database.

    The frame is archived BEFORE parsing and parsing reads the archived
    bytes back (`raw_store.load`), so what this returns is what was
    archived. `fetched_at`, when not given, is stamped after the fetch
    returns.
    """
    if season_type not in SEASON_TYPE_TAGS:
        raise ValueError(f"unknown season type {season_type!r}")
    df = fetch(season, season_type)
    fetched_at = db.require_utc(
        fetched_at if fetched_at is not None else datetime.now(UTC), "fetched_at"
    )
    key = archive_key(season, season_type, fetched_at)
    payload = df.to_json(orient="split", date_format="iso").encode()
    raw_store.store(
        SOURCE,
        key,
        gzip.compress(payload, mtime=0),
        fetched_at,
        meta={"season": season, "season_type": season_type},
    )
    archived = pd.read_json(
        io.BytesIO(gzip.decompress(raw_store.load(SOURCE, key))),
        orient="split",
        dtype=_READ_DTYPE,
    )
    return Downloaded(
        season=season,
        season_type=season_type,
        fetched_at=fetched_at,
        blob_key=key,
        rows=_parse(archived, season),
    )


_COLUMNS = (
    "game_id", "season", "game_date", "season_type", "player_id", "player_name",
    "team", "minutes", *(field for field, _ in _BOX_COLUMNS), "plus_minus",
    "source", "reconstructed", "observed_at",
)


def load(con, downloaded: Downloaded) -> LoadResult:
    """INSERT OR REPLACE one download's rows in a single transaction.

    Rolls back and re-raises on any error -- nothing is left half-written.
    """
    table = db.POINT_IN_TIME_TABLES["player_games"]
    games_table = db.POINT_IN_TIME_TABLES["games"]
    tag = SEASON_TYPE_TAGS[downloaded.season_type]
    sql = (
        f"INSERT OR REPLACE INTO {table} ({', '.join(_COLUMNS)})"
        f" VALUES ({', '.join('?' for _ in _COLUMNS)})"
    )
    con.execute("BEGIN")
    try:
        params = []
        for row in downloaded.rows:
            observed_at = db.require_utc(
                nba_stats._derive_observed_at(row.game_date, row.game_id)[1], "observed_at"
            )
            params.append(
                [
                    row.game_id, row.season, row.game_date, tag, row.player_id,
                    row.player_name, row.team, row.minutes,
                    *(getattr(row, field) for field, _ in _BOX_COLUMNS),
                    row.plus_minus, DB_SOURCE, True, observed_at,
                ]
            )
        if params:
            con.executemany(sql, params)
        game_ids = sorted({row.game_id for row in downloaded.rows})
        known = {
            r[0]
            for r in con.execute(
                f"SELECT DISTINCT game_id FROM {games_table} WHERE list_contains(?, game_id)",
                [game_ids],
            ).fetchall()
        }
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise
    return LoadResult(
        season=downloaded.season,
        season_type=downloaded.season_type,
        rows_written=len(params),
        games=len(game_ids),
        unknown_game_ids=[g for g in game_ids if g not in known],
    )
