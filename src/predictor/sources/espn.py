"""ESPN's public NBA scoreboard (schedule, tip-off times, final scores) and
game summaries (box scores).

One request per ET date (``dates=YYYYMMDD``). Each day's response bytes go
to ``raw_store`` exactly as received (``download_scoreboards``), and parsing
reads them back from the archive (``load_scoreboards``).

Response shape, confirmed against real responses on 2026-10-24 (see
``tests/fixtures/espn/scoreboard_sample.json``):

- ``events[].id`` (string), ``events[].date`` ("2026-10-21T01:50Z", UTC),
  ``events[].season`` = ``{"year": 2027, "type": 2, ...}`` (year is the
  season's end year; type 1 preseason, 2 regular, 3 postseason, 5 play-in).
- ``events[].status.type.completed`` (bool).
- ``competitions[0]``: ``timeValid``, ``neutralSite``, ``venue.fullName``,
  ``venue.address.city/state`` and ``competitors[]`` with ``homeAway``,
  ``team.abbreviation`` / ``team.displayName`` and ``score`` -- a string
  ("121"), and "0" before tip-off, so a score counts only once completed.
- ``status.type.name`` (STATUS_SCHEDULED, STATUS_FINAL; STATUS_POSTPONED and
  STATUS_CANCELED are assumed from ESPN's naming, not yet seen live).

A competition without a ``timeValid`` field is treated as having a valid
time (every real sample carried it). An event missing a field the parser
needs (season year/type, date, id) is reported as malformed and skipped;
the rest of the response is kept.

Game summary (``summary?event=<id>``), confirmed against a real response
for a final on 2026-10-24 (see ``tests/fixtures/espn/summary_sample.json``):
``boxscore.players[]`` has one block per team with ``team.abbreviation`` /
``displayName`` and ``statistics[0]`` holding ``keys`` (``minutes``,
``points``, ..., ``rebounds``, ``assists``, ..., ``plusMinus``), ``labels``
and ``athletes[]``. Each athlete has ``athlete.id`` (string) and
``athlete.displayName``, ``starter`` and ``didNotPlay`` (bools) and
``stats``: strings in ``keys`` order, minutes as whole minutes ("31"),
plus/minus signed ("+9", "-5", "0"). A DNP player has ``stats: []``.

The range form ``dates=YYYYMMDD-YYYYMMDD`` answered HTTP 400 ("Failed to get
events endpoint.") on 2026-10-24 for every range tried, so it is never used.
Politeness: at most MAX_SCOREBOARD_REQUESTS day requests per run (3 near
days every run, 7 far days at most hourly; see cloud/games.py), 20 s
timeout, 2 retries on network errors or 5xx; at most 15 summaries per run
(cloud/players.py).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

import requests

from predictor import raw_store
from predictor.teams import espn_team_abbr

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
RAW_SOURCE = "espn"
EASTERN = ZoneInfo("America/New_York")

TIMEOUT_SECONDS = 20
RETRIES = 2  # extra attempts after the first, on network errors or 5xx
# slate-2 .. slate+7: the most day requests one run may make.
MAX_SCOREBOARD_REQUESTS = 10
_RETRY_PAUSE_SECONDS = 2.0


class EspnFetchError(Exception):
    """ESPN did not give a usable response."""


class EspnFormatError(Exception):
    """An ESPN response does not have the shape the parser relies on."""


@dataclass(frozen=True)
class EspnGame:
    espn_id: str
    season: str
    season_type: int
    game_date: date
    tip_off_utc: datetime | None
    home_team: str | None
    away_team: str | None
    home_name: str | None
    away_name: str | None
    neutral: bool
    venue_name: str | None
    venue_city: str | None
    venue_state: str | None
    completed: bool
    home_points: int | None
    away_points: int | None
    status_name: str | None = None

    @property
    def postponed(self) -> bool:
        """Postponed, cancelled or suspended: no tip and no result to use."""
        name = (self.status_name or "").upper()
        return any(word in name for word in _OFF_WORDS)


_OFF_WORDS = ("POSTPONED", "CANCEL", "SUSPENDED")


@dataclass(frozen=True)
class EspnPlayerLine:
    """One player's line in an ESPN box score. ``team`` is the NBA
    abbreviation. A player who did not play has no minutes or stats."""

    espn_player_id: str
    name: str
    team: str
    starter: bool
    did_not_play: bool
    minutes: float | None
    points: int | None
    rebounds: int | None
    assists: int | None
    plus_minus: int | None


# --- fetch --------------------------------------------------------------------


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _get(session, url: str, params: dict, label: str) -> tuple[int, bytes]:
    """(status, body) of one request, retrying network errors and 5xx."""
    last = ""
    for attempt in range(RETRIES + 1):
        if attempt:
            _sleep(_RETRY_PAUSE_SECONDS * attempt)
        try:
            response = session.get(url, params=params, timeout=TIMEOUT_SECONDS)
        except requests.RequestException as exc:
            last = f"could not reach ESPN ({type(exc).__name__})"
            continue
        if response.status_code >= 500:
            last = f"ESPN returned HTTP {response.status_code}"
            continue
        return response.status_code, response.content
    raise EspnFetchError(f"{last} for {label} after {RETRIES + 1} attempts")


def _fetch_json_bytes(session, url: str, params: dict, label: str) -> bytes:
    """The exact bytes of a 200 response whose body is a JSON object."""
    status, content = _get(session or requests.Session(), url, params, label)
    if status != 200:
        raise EspnFetchError(f"ESPN returned HTTP {status} for {label}")
    try:
        payload = json.loads(content)
    except ValueError:
        raise EspnFetchError(f"ESPN returned a response that is not JSON for {label}") from None
    if not isinstance(payload, dict):
        raise EspnFetchError(f"ESPN returned an unexpected response for {label}")
    return content


def fetch_scoreboard_day(day: date, session=None) -> bytes:
    """ESPN's scoreboard for one ET date, as the exact response bytes."""
    dates = f"{day:%Y%m%d}"
    return _fetch_json_bytes(session, SCOREBOARD_URL, {"dates": dates, "limit": 1000},
                             f"dates={dates}")


def fetch_summary(espn_id: str, session=None) -> bytes:
    """ESPN's game summary (box score included) for one event, as the exact
    response bytes. Archive them (``archive("summary", body, fetched_at,
    espn_id)``) before parsing."""
    return _fetch_json_bytes(session, SUMMARY_URL, {"event": str(espn_id)},
                             f"event={espn_id}")


def archive(kind: str, body: bytes, fetched_at: datetime, ident: str) -> str:
    """Store the response bytes in raw_store; returns the key."""
    key = f"{kind}_{ident}_{fetched_at.astimezone(UTC):%Y%m%dT%H%M%SZ}.json"
    raw_store.store(RAW_SOURCE, key, body, fetched_at,
                    meta={"kind": kind, "fetched_at": fetched_at.isoformat()})
    return key


def download_scoreboards(days: list[date], fetched_at: datetime, session=None) -> list[str]:
    """Fetch each day and archive its exact bytes (key ``scoreboard_<YYYYMMDD>_
    <fetched_at>.json``); returns the keys in ``days`` order. A failed day
    raises EspnFetchError; days archived before it stay archived."""
    if len(days) > MAX_SCOREBOARD_REQUESTS:
        raise ValueError(f"{len(days)} scoreboard days exceeds {MAX_SCOREBOARD_REQUESTS}")
    session = session or requests.Session()
    keys = []
    for day in days:
        body = fetch_scoreboard_day(day, session=session)
        keys.append(archive("scoreboard", body, fetched_at, f"{day:%Y%m%d}"))
    return keys


def load_scoreboards(keys: list[str], problems: list[str] | None = None) -> list[EspnGame]:
    """Parse archived scoreboard responses; each event once (first wins).
    Malformed events are appended to ``problems`` (see `parse_scoreboard`)."""
    seen: dict[str, EspnGame] = {}
    for key in keys:
        for game in parse_scoreboard(raw_store.load(RAW_SOURCE, key), problems):
            seen.setdefault(game.espn_id, game)
    return list(seen.values())


# --- parse --------------------------------------------------------------------


def _utc(text: str) -> datetime:
    moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        raise ValueError(f"ESPN date without a zone: {text!r}")
    return moment.astimezone(UTC)


def _season_label(end_year: int) -> str:
    return f"{end_year - 1}-{str(end_year)[-2:]}"


def _points(score) -> int | None:
    if isinstance(score, dict):
        score = score.get("value", score.get("displayValue"))
    if score is None or score == "":
        return None
    return int(float(score))


def _parse_event(event: dict) -> EspnGame:
    comp = (event.get("competitions") or [{}])[0]
    when = _utc(event.get("date") or comp["date"])
    sides = {c.get("homeAway"): c for c in comp.get("competitors") or []}
    home, away = sides.get("home", {}), sides.get("away", {})
    home_team_obj, away_team_obj = home.get("team") or {}, away.get("team") or {}
    season = event.get("season") or {}
    status = (event.get("status") or comp.get("status") or {}).get("type") or {}
    completed = bool(status.get("completed"))
    venue = comp.get("venue") or {}
    address = venue.get("address") or {}
    tip_valid = comp.get("timeValid", True)
    return EspnGame(
        espn_id=str(event["id"]),
        season=_season_label(int(season["year"])),
        season_type=int(season["type"]),
        game_date=when.astimezone(EASTERN).date(),
        tip_off_utc=when if tip_valid else None,
        home_team=espn_team_abbr(home_team_obj.get("abbreviation"),
                                 home_team_obj.get("displayName")),
        away_team=espn_team_abbr(away_team_obj.get("abbreviation"),
                                 away_team_obj.get("displayName")),
        home_name=home_team_obj.get("displayName") or home_team_obj.get("abbreviation"),
        away_name=away_team_obj.get("displayName") or away_team_obj.get("abbreviation"),
        neutral=bool(comp.get("neutralSite", False)),
        venue_name=venue.get("fullName"),
        venue_city=address.get("city"),
        venue_state=address.get("state"),
        completed=completed,
        home_points=_points(home.get("score")) if completed else None,
        away_points=_points(away.get("score")) if completed else None,
        status_name=status.get("name"),
    )


def _malformed_label(event, exc: Exception) -> str:
    if not isinstance(event, dict):
        return f"? ? (malformed event: {type(exc).__name__} {exc})"
    short = event.get("shortName") or event.get("name") or "?"
    day = str(event.get("date") or "?")[:10]
    return f"{short} {day} (malformed event {event.get('id', '?')}: {type(exc).__name__} {exc})"


def parse_scoreboard(body: bytes, problems: list[str] | None = None) -> list[EspnGame]:
    """Every event in one scoreboard response, once. An event that cannot be
    parsed is skipped and described in ``problems`` (when given); the other
    events are still returned."""
    seen: dict[str, EspnGame] = {}
    for event in json.loads(body).get("events") or []:
        try:
            game = _parse_event(event)
        except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
            if problems is not None:
                problems.append(_malformed_label(event, exc))
            continue
        seen.setdefault(game.espn_id, game)
    return list(seen.values())


# --- box score (game summary) -------------------------------------------------

# Stat keys the box score must carry (``statistics[0].keys``); values are
# always looked up by key, never by position or label.
BOX_KEYS = ("minutes", "points", "rebounds", "assists", "plusMinus")
_EMPTY = ("", "--", "-")


def _stat_int(text) -> int | None:
    if text is None or str(text).strip() in _EMPTY:
        return None
    return int(str(text).strip())  # int() reads "+12" and "-3"


def _stat_minutes(text) -> float | None:
    if text is None or str(text).strip() in _EMPTY:
        return None
    text = str(text).strip()
    if ":" in text:
        whole, seconds = text.split(":", 1)
        return int(whole) + int(seconds) / 60
    return float(text)


def _player_line(entry: dict, team: str, index: dict[str, int]) -> EspnPlayerLine:
    who = entry["athlete"]
    stats = entry.get("stats") or []
    did_not_play = bool(entry.get("didNotPlay")) or not stats

    def stat(key: str):
        if did_not_play:
            return None
        return stats[index[key]] if index[key] < len(stats) else None

    return EspnPlayerLine(
        espn_player_id=str(who["id"]),
        name=who.get("displayName") or who.get("shortName") or str(who["id"]),
        team=team,
        starter=bool(entry.get("starter")),
        did_not_play=did_not_play,
        minutes=_stat_minutes(stat("minutes")),
        points=_stat_int(stat("points")),
        rebounds=_stat_int(stat("rebounds")),
        assists=_stat_int(stat("assists")),
        plus_minus=_stat_int(stat("plusMinus")),
    )


def parse_boxscore(body: bytes) -> list[EspnPlayerLine]:
    """Every player line in one game summary, team by team in ESPN's order.

    Raises EspnFormatError when the response has no ``boxscore``, a team
    block lacks one of BOX_KEYS (the message names it), a team is not an NBA
    team, or a value cannot be read. A summary whose box score has no player
    blocks yet gives an empty list."""
    payload = json.loads(body)
    if not isinstance(payload, dict) or not isinstance(payload.get("boxscore"), dict):
        raise EspnFormatError("ESPN summary has no boxscore")
    lines: list[EspnPlayerLine] = []
    for block in payload["boxscore"].get("players") or []:
        team_obj = block.get("team") or {}
        team = espn_team_abbr(team_obj.get("abbreviation"), team_obj.get("displayName"))
        if team is None:
            raise EspnFormatError(
                f"ESPN box score team {team_obj.get('displayName')!r} is not an NBA team"
            )
        groups = block.get("statistics") or []
        if not groups:
            raise EspnFormatError(f"ESPN box score for {team} has no statistics")
        keys = groups[0].get("keys") or []
        index = {}
        for key in BOX_KEYS:
            if key not in keys:
                raise EspnFormatError(f"ESPN box score for {team} has no stat key {key!r}")
            index[key] = keys.index(key)
        for entry in groups[0].get("athletes") or []:
            try:
                lines.append(_player_line(entry, team, index))
            except (KeyError, TypeError, ValueError) as exc:
                raise EspnFormatError(
                    f"ESPN box score line for {team} cannot be read: {type(exc).__name__} {exc}"
                ) from exc
    return lines
