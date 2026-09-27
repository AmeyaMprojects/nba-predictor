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

import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")
_REGULAR_SEASON_PREFIX = "002"


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
