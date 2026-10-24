"""ESPN's public NBA scoreboard: schedule, tip-off times and final scores.

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

The range form ``dates=YYYYMMDD-YYYYMMDD`` answered HTTP 400 ("Failed to get
events endpoint.") on 2026-10-24 for every range tried, so it is never used.
Politeness: at most MAX_SCOREBOARD_REQUESTS day requests per run (3 near
days every run, 7 far days at most hourly; see cloud/games.py), 20 s
timeout, 2 retries on network errors or 5xx.
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
RAW_SOURCE = "espn"
EASTERN = ZoneInfo("America/New_York")

TIMEOUT_SECONDS = 20
RETRIES = 2  # extra attempts after the first, on network errors or 5xx
# slate-2 .. slate+7: the most day requests one run may make.
MAX_SCOREBOARD_REQUESTS = 10
_RETRY_PAUSE_SECONDS = 2.0


class EspnFetchError(Exception):
    """ESPN did not give a usable response."""


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


# --- fetch --------------------------------------------------------------------


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _get(session, dates: str) -> tuple[int, bytes]:
    """(status, body) of one request, retrying network errors and 5xx."""
    last = ""
    for attempt in range(RETRIES + 1):
        if attempt:
            _sleep(_RETRY_PAUSE_SECONDS * attempt)
        try:
            response = session.get(
                SCOREBOARD_URL, params={"dates": dates, "limit": 1000},
                timeout=TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            last = f"could not reach ESPN ({type(exc).__name__})"
            continue
        if response.status_code >= 500:
            last = f"ESPN returned HTTP {response.status_code}"
            continue
        return response.status_code, response.content
    raise EspnFetchError(f"{last} for dates={dates} after {RETRIES + 1} attempts")


def fetch_scoreboard_day(day: date, session=None) -> bytes:
    """ESPN's scoreboard for one ET date, as the exact response bytes."""
    session = session or requests.Session()
    dates = f"{day:%Y%m%d}"
    status, content = _get(session, dates)
    if status != 200:
        raise EspnFetchError(f"ESPN returned HTTP {status} for dates={dates}")
    try:
        payload = json.loads(content)
    except ValueError:
        raise EspnFetchError(
            f"ESPN returned a response that is not JSON for dates={dates}"
        ) from None
    if not isinstance(payload, dict):
        raise EspnFetchError(f"ESPN returned an unexpected response for dates={dates}")
    return content


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
