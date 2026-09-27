"""NBA league schedule (nba_api ScheduleLeagueV2) -- spec section 1.1.

What may be read from this source:

- tip_off_utc ESTABLISHES the backtest cutoff (see backtest/tipoff.py). It
  is never a predictor feature.
- Arena and neutral-site columns are static venue facts, not
  outcome-bearing, safe to read at any time.
- Scores, the numeric game status, team records and points leaders are in
  the payload but are never read by this parser. `gameStatusText` is read
  only to recognise the league's "TBD" placeholder, and is never stored. A
  field that is not stored cannot leak.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from nba_api.stats.endpoints import scheduleleaguev2
from tenacity import retry, stop_after_attempt, wait_exponential

from predictor import db, raw_store

EASTERN = ZoneInfo("America/New_York")
_REGULAR_SEASON_PREFIX = "002"
SOURCE = "schedule"


@dataclass(frozen=True)
class ScheduleRow:
    game_id: str
    season: str
    game_date: date
    tip_off_utc: datetime | None
    home_team: str
    away_team: str
    arena_name: str | None
    arena_city: str | None
    arena_state: str | None
    is_neutral_reported: bool
    is_neutral: bool


@dataclass(frozen=True)
class ParseResult:
    rows: list[ScheduleRow]
    # Games saved without a tip-off: the league lists the time as TBD, or
    # the listed instant does not fall on the listed Eastern date.
    no_tipoff: list[str]
    # Games left out entirely because their teams are not decided yet
    # (NBA Cup knockout slots, the Cup final).
    undetermined: list[str]


def _text(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _tipoff(game: dict, game_date: date) -> datetime | None:
    """The tip-off instant, or None rather than a guess.

    A TBD game carries a 00:00 ET placeholder in gameDateTimeUTC. Accepting
    it would be conservative once, but tipoff_index takes the MINIMUM
    across vintages, so the placeholder would outlive the real time
    forever. None keeps it out of the index entirely.
    """
    if (_text(game.get("gameStatusText")) or "").upper() == "TBD":
        return None
    raw = _text(game.get("gameDateTimeUTC"))
    if raw is None:
        return None
    try:
        tip = datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None
    if tip.astimezone(EASTERN).date() != game_date:
        return None
    return tip


def parse_schedule(payload: bytes, season: str) -> ParseResult:
    try:
        doc = json.loads(payload)
        league = doc["leagueSchedule"]
        reported_season = league["seasonYear"]
        staged = []
        undetermined: list[str] = []
        for day in league["gameDates"]:
            for game in day["games"]:
                game_id = str(game["gameId"])
                home = _text((game.get("homeTeam") or {}).get("teamTricode"))
                away = _text((game.get("awayTeam") or {}).get("teamTricode"))
                if home is None or away is None:
                    undetermined.append(game_id)
                    continue
                game_date = date.fromisoformat(str(game["gameDateEst"])[:10])
                staged.append((game, game_id, game_date, home, away))
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            f"the NBA schedule response for {season} was not in the expected "
            f"shape ({exc!r})"
        ) from None
    if reported_season != season:
        raise ValueError(
            f"asked for the {season} schedule but the NBA returned the "
            f"{reported_season!r} schedule"
        )

    # Each team's usual home venue: the (city, state) it plays most of its
    # regular-season home games at, in this payload. The league's own
    # isNeutral is false for every game before 2024-25 -- Paris, Mexico
    # City and Las Vegas included -- so a game away from that venue is
    # neutral too. Measured effect: the 2019-20 Orlando bubble, the
    # Paris/Mexico City/Las Vegas games, and San Antonio's Austin games.
    venues: dict[str, Counter] = {}
    for game, game_id, _, home, _ in staged:
        if game_id.startswith(_REGULAR_SEASON_PREFIX) and not bool(game.get("isNeutral")):
            key = (_text(game.get("arenaCity")), _text(game.get("arenaState")))
            venues.setdefault(home, Counter())[key] += 1
    usual = {team: counts.most_common(1)[0][0] for team, counts in venues.items()}

    rows: list[ScheduleRow] = []
    no_tipoff: list[str] = []
    for game, game_id, game_date, home, away in staged:
        city = _text(game.get("arenaCity"))
        state = _text(game.get("arenaState"))
        reported = bool(game.get("isNeutral"))
        away_from_home = home in usual and (city, state) != usual[home]
        tip = _tipoff(game, game_date)
        if tip is None:
            no_tipoff.append(game_id)
        rows.append(
            ScheduleRow(
                game_id=game_id,
                season=season,
                game_date=game_date,
                tip_off_utc=tip,
                home_team=home,
                away_team=away,
                arena_name=_text(game.get("arenaName")),
                arena_city=city,
                arena_state=state,
                is_neutral_reported=reported,
                is_neutral=reported or away_from_home,
            )
        )
    return ParseResult(rows=rows, no_tipoff=no_tipoff, undetermined=undetermined)


@dataclass(frozen=True)
class IngestResult:
    season: str
    written: int
    no_tipoff: list[str]
    undetermined: list[str]
    mismatches: list[str]
    blob_key: str


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
)
def fetch_season_payload(season: str) -> bytes:
    """The raw schedule JSON for one season, historical or forward."""
    endpoint = scheduleleaguev2.ScheduleLeagueV2(
        season=season, league_id="00", timeout=60
    )
    return endpoint.get_json().encode("utf-8")


def archive_key(season: str, fetched_at: datetime) -> str:
    return f"{season}_{fetched_at:%Y%m%dT%H%M%S}Z.json.gz"


def ingest_season(
    con,
    season: str,
    fetched_at: datetime | None = None,
    fetch: Callable[[str], bytes] = fetch_season_payload,
) -> IngestResult:
    """Fetch, archive, parse and load one season's schedule.

    Raw-first: the gzipped payload is archived BEFORE parsing, and the
    parser reads the archived bytes back, so a parser bug is re-parsable
    rather than lost. Each call writes a new vintage stamped with the real
    fetch time -- the table accumulates genuine point-in-time schedule
    history from the first daily run onward.
    """
    fetched_at = db.require_utc(
        fetched_at if fetched_at is not None else datetime.now(UTC), "fetched_at"
    )
    payload = fetch(season)
    key = archive_key(season, fetched_at)
    raw_store.store(
        SOURCE, key, gzip.compress(payload, mtime=0), fetched_at, meta={"season": season}
    )
    parsed = parse_schedule(gzip.decompress(raw_store.load(SOURCE, key)), season)

    table = db.POINT_IN_TIME_TABLES["schedule"]
    if parsed.rows:
        con.executemany(
            f"INSERT OR REPLACE INTO {table} (game_id, season, game_date,"
            " tip_off_utc, home_team, away_team, arena_name, arena_city,"
            " arena_state, is_neutral_reported, is_neutral, observed_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                [
                    r.game_id, r.season, r.game_date, r.tip_off_utc, r.home_team,
                    r.away_team, r.arena_name, r.arena_city, r.arena_state,
                    r.is_neutral_reported, r.is_neutral, fetched_at,
                ]
                for r in parsed.rows
            ],
        )

    mismatches = compare_with_games(con, season, parsed.rows)
    for line in mismatches:
        print(f"schedule: MISMATCH {line}")
    return IngestResult(
        season=season,
        written=len(parsed.rows),
        no_tipoff=parsed.no_tipoff,
        undetermined=parsed.undetermined,
        mismatches=mismatches,
        blob_key=key,
    )


def compare_with_games(con, season: str, rows: list[ScheduleRow]) -> list[str]:
    """Where the schedule and the games table disagree, say so -- never reconcile.

    Spec 1.1: the schedule is ground truth for when a game tipped, and a
    disagreement (chiefly a rescheduled game) is logged loudly. Measured
    2026-09-27: zero disagreements across all 8,289 regular-season games,
    so any line printed here is news.
    """
    games = db.POINT_IN_TIME_TABLES["games"]
    by_id = {r.game_id: r for r in rows}
    found = con.execute(
        f"SELECT game_id, MIN(game_date), MIN(home_team), MIN(away_team) "
        f"FROM {games} WHERE season = ? GROUP BY game_id ORDER BY game_id",
        [season],
    ).fetchall()
    out: list[str] = []
    for game_id, game_date, home, away in found:
        row = by_id.get(game_id)
        if row is None:
            if game_id.startswith(_REGULAR_SEASON_PREFIX):
                out.append(
                    f"{game_id}: in the games table ({game_date} {away}@{home}) "
                    "but not in the schedule"
                )
            continue
        if (row.game_date, row.home_team, row.away_team) != (game_date, home, away):
            out.append(
                f"{game_id}: schedule says {row.game_date} {row.away_team}@"
                f"{row.home_team}, games table says {game_date} {away}@{home}"
            )
    return out
