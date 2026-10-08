"""Live odds from The Odds API, linked to schedule games.

Spec section 8. One fetch per run (the free tier allows 500 requests a
month). The flow is split so the network call never holds the database:

    download()  fetch -> stamp observed_at -> archive the response bytes
    ingest_current(con, fetch=...)  parse the ARCHIVED bytes -> link -> store

`predictor ingest-odds` runs them in that order with `connect_with_retry`
in between. The API key is never printed, logged, archived or stored:
request errors are re-raised as `OddsFetchError` without the URL (which
carries the key as a query parameter).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import requests

from predictor import config, db, raw_store
from predictor.model.venues import COMPETITIVE_PREFIXES
from predictor.teams import team_abbr

API_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
SOURCE = "theoddsapi"
RAW_SOURCE = "odds"
EASTERN = ZoneInfo("America/New_York")

_COMPETITIVE_SQL = ", ".join(f"'{p}'" for p in COMPETITIVE_PREFIXES)


class OddsQuotaExceeded(Exception):
    """The Odds API rejected the request for quota or auth reasons.

    Raised only for a 401 (bad/expired key, or monthly credits used up) or
    429 (rate limit) response -- both mean "this call did not get usable
    data because of the account, not the market", and must be reported
    loudly and distinctly from a legitimate "no games right now" response
    (an empty payload with a 200 status), exactly as injury_report's
    TransientFetchError is kept distinct from a confirmed "not published"
    403/404.
    """

    def __init__(self, status_code: int, detail: str = "", requests_remaining: str | None = None):
        # The message is the status alone; the API's response text is kept
        # on `detail` for whoever needs it, never echoed by default.
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.detail = detail[:200]
        self.requests_remaining = requests_remaining


class OddsFetchError(Exception):
    """Any other failure to get a usable response. Never contains the key."""


class MissingOddsKey(ValueError):
    """No key in the key file or in env ODDS_API_KEY."""


def missing_key_message() -> str:
    path = config.odds_api_key_path()
    return (
        "No Odds API key found. Get a free key at https://the-odds-api.com, "
        f"save it (just the key, one line) in the file {path}, then run: "
        f"chmod 600 {path}"
    )


@dataclass(frozen=True)
class OddsResponse:
    """One API response: the raw body bytes and the quota header, if sent."""

    body: bytes
    requests_remaining: str | None

    @property
    def payload(self) -> list[dict]:
        return json.loads(self.body)


@dataclass(frozen=True)
class OddsDownload:
    """A fetched-and-archived snapshot, ready to load into the database."""

    archive_key: str
    observed_at: datetime
    requests_remaining: str | None


@dataclass(frozen=True)
class OddsIngestSummary:
    rows: int
    events: int
    linked: int
    # "AWAY@HOME YYYY-MM-DD" (ET date of commence_time) per event that
    # matched no scheduled game; its rows are stored with game_id NULL.
    unlinked: list[str] = field(default_factory=list)
    # "AWAY@HOME odds-date -> schedule-date" per event linked through the
    # +-1 day fallback.
    shifted: list[str] = field(default_factory=list)
    requests_remaining: str | None = None


def fetch_current(api_key: str, session=None) -> OddsResponse:
    session = session or requests.Session()
    try:
        response = session.get(
            API_URL,
            params={
                "apiKey": api_key,
                "regions": "us",
                "markets": "h2h,spreads,totals",
                "oddsFormat": "american",
            },
            timeout=30,
        )
    except requests.RequestException as exc:
        # The exception text carries the request URL, key included.
        raise OddsFetchError(
            f"could not reach The Odds API ({type(exc).__name__})"
        ) from None
    remaining = (response.headers or {}).get("x-requests-remaining")
    if response.status_code in (401, 429):
        raise OddsQuotaExceeded(response.status_code, response.text or "", remaining)
    if response.status_code != 200:
        raise OddsFetchError(f"The Odds API returned HTTP {response.status_code}")
    body = response.content
    try:
        payload = json.loads(body)
    except ValueError:
        raise OddsFetchError("The Odds API returned a response that is not JSON") from None
    if not isinstance(payload, list):
        raise OddsFetchError("The Odds API returned an unexpected response (not a list of games)")
    return OddsResponse(body, remaining)


def parse_odds_payload(payload: list[dict], observed_at: datetime) -> list[dict]:
    rows: list[dict] = []
    for event in payload:
        home, away = event.get("home_team"), event.get("away_team")
        for book in event.get("bookmakers") or []:
            row = {
                "game_key": event["id"],
                "book": book["key"],
                "home_team": home,
                "away_team": away,
                "home_price": None,
                "away_price": None,
                "spread": None,
                "total": None,
                "observed_at": observed_at,
            }
            for market in book.get("markets") or []:
                outcomes = {o["name"]: o for o in market.get("outcomes") or []}
                if market["key"] == "h2h":
                    row["home_price"] = outcomes.get(home, {}).get("price")
                    row["away_price"] = outcomes.get(away, {}).get("price")
                elif market["key"] == "spreads":
                    row["spread"] = outcomes.get(home, {}).get("point")
                elif market["key"] == "totals":
                    row["total"] = next(
                        (o.get("point") for o in outcomes.values()), None
                    )
            rows.append(row)
    return rows


def _utcnow() -> datetime:
    return datetime.now(UTC)


def download(api_key: str, now: datetime | None = None, session=None) -> OddsDownload:
    """Fetch one snapshot and archive its bytes. Touches no database.

    ``observed_at`` is stamped after the fetch returns -- the moment the
    lines became known to us -- and must be UTC (db.require_utc). The
    archive key is content-addressed (sha256 of the body), so an identical
    body is a raw_store no-op. Neither the key nor the URL is archived.
    """
    response = fetch_current(api_key, session=session)
    observed_at = db.require_utc(now if now is not None else _utcnow(), "observed_at")
    digest = hashlib.sha256(response.body).hexdigest()[:16]
    archive_key = f"odds_{digest}.json"
    raw_store.store(
        RAW_SOURCE,
        archive_key,
        response.body,
        observed_at,
        meta={"observed_at": observed_at.isoformat()},
    )
    return OddsDownload(archive_key, observed_at, response.requests_remaining)


def _eastern_date(commence_time: str | None) -> date | None:
    if not commence_time:
        return None
    try:
        moment = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        return None
    return moment.astimezone(EASTERN).date()


def _event_label(home: str | None, away: str | None, day: date | None) -> str:
    home_text = team_abbr(home) or home or "?"
    away_text = team_abbr(away) or away or "?"
    return f"{away_text}@{home_text} {day.isoformat() if day else '?'}"


@dataclass(frozen=True)
class ScheduleGame:
    """One game as its LATEST schedule vintage lists it."""

    game_id: str
    season: str
    game_date: date
    tip_off_utc: datetime | None
    home_team: str
    away_team: str


def latest_schedule_games(
    con, start: date | None = None, end: date | None = None
) -> list[ScheduleGame]:
    """The LATEST schedule vintage of each competitive game, optionally only
    those dated ``start``..``end`` (inclusive, ET dates).

    Read directly rather than through AsOfView, like model/live.py's slate:
    linking a line to its game is bookkeeping, not a feature, and the
    current listing is the right one to link against.
    """
    table = db.POINT_IN_TIME_TABLES["schedule"]
    rows = con.execute(
        f"""
        WITH latest AS (
            SELECT game_id, season, game_date, tip_off_utc, home_team, away_team,
                   row_number() OVER (
                       PARTITION BY game_id ORDER BY observed_at DESC
                   ) AS rn
            FROM {table}
        )
        SELECT game_id, season, game_date, tip_off_utc, home_team, away_team
        FROM latest
        WHERE rn = 1
          AND game_date BETWEEN coalesce(?, DATE '0001-01-01')
                            AND coalesce(?, DATE '9999-12-31')
          AND substr(game_id, 1, 3) IN ({_COMPETITIVE_SQL})
        ORDER BY game_date, game_id
        """,
        [start, end],
    ).fetchall()
    return [ScheduleGame(*row) for row in rows]


def build_link_index(games) -> dict[tuple[date, str, str], list[str]]:
    """(ET date, home, away) -> game_ids, for `link_game`."""
    index: dict[tuple[date, str, str], list[str]] = {}
    for game in games:
        index.setdefault((game.game_date, game.home_team, game.away_team), []).append(
            game.game_id
        )
    return index


def _schedule_index(con, days: set[date]) -> dict[tuple[date, str, str], list[str]]:
    """`build_link_index` over the games dated within a day of any of ``days``."""
    if not days:
        return {}
    return build_link_index(
        latest_schedule_games(
            con, min(days) - timedelta(days=1), max(days) + timedelta(days=1)
        )
    )


def link_game(index, day: date, home: str, away: str) -> tuple[str | None, date | None]:
    """The one game this event is, as (game_id, matched schedule date).

    Exact ET date first; failing that, the day before or after -- but only
    if exactly one game fits, never a guess between two.
    """
    exact = index.get((day, home, away), [])
    if len(exact) == 1:
        return exact[0], day
    if exact:
        return None, None
    candidates = [
        (game_id, d)
        for d in (day - timedelta(days=1), day + timedelta(days=1))
        for game_id in index.get((d, home, away), [])
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None, None


_link = link_game


def load(con, downloaded: OddsDownload) -> OddsIngestSummary:
    """Parse the archived snapshot, link each event to a game, store rows."""
    payload = json.loads(raw_store.load(RAW_SOURCE, downloaded.archive_key))
    observed_at = downloaded.observed_at

    events = []
    for event in payload:
        day = _eastern_date(event.get("commence_time"))
        events.append(
            (event, day, team_abbr(event.get("home_team")), team_abbr(event.get("away_team")))
        )
    index = _schedule_index(con, {day for _, day, _, _ in events if day is not None})

    game_by_event: dict[str, str | None] = {}
    linked = 0
    unlinked: list[str] = []
    shifted: list[str] = []
    for event, day, home, away in events:
        game_id = matched_day = None
        if day is not None and home and away:
            game_id, matched_day = _link(index, day, home, away)
        game_by_event[event.get("id")] = game_id
        label = _event_label(event.get("home_team"), event.get("away_team"), day)
        if game_id is None:
            unlinked.append(label)
            continue
        linked += 1
        if matched_day != day:
            shifted.append(f"{label} -> {matched_day.isoformat()}")

    rows = parse_odds_payload(payload, observed_at)

    # Resolved through db.POINT_IN_TIME_TABLES rather than spelled as a
    # literal here -- the physical "_raw" table names are only allowed to
    # appear as string literals in db.py/asof.py (see
    # test_no_physical_table_name_appears_outside_db_and_asof).
    table = db.POINT_IN_TIME_TABLES["odds_snapshots"]
    insert_sql = (
        f"INSERT OR REPLACE INTO {table} (game_key, book, home_team,"
        " away_team, home_price, away_price, spread, total, observed_at,"
        " game_id, source, reconstructed)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,FALSE)"
    )
    for row in rows:
        con.execute(
            insert_sql,
            [
                row["game_key"], row["book"], row["home_team"], row["away_team"],
                row["home_price"], row["away_price"], row["spread"], row["total"],
                row["observed_at"], game_by_event.get(row["game_key"]), SOURCE,
            ],
        )
    return OddsIngestSummary(
        rows=len(rows),
        events=len(payload),
        linked=linked,
        unlinked=unlinked,
        shifted=shifted,
        requests_remaining=downloaded.requests_remaining,
    )


def ingest_current(
    con,
    api_key: str | None = None,
    now: datetime | None = None,
    session=None,
    fetch: OddsDownload | None = None,
) -> OddsIngestSummary:
    """Load one snapshot into ``con``.

    ``fetch`` is an already-made `download()` (the CLI's path: it downloads
    before opening the database). Without it, this downloads first, using
    ``api_key`` or `config.odds_api_key()`; with no key anywhere it raises
    `MissingOddsKey`, whose message names the key file to create.
    """
    if fetch is None:
        api_key = api_key or config.odds_api_key()
        if not api_key:
            raise MissingOddsKey(missing_key_message())
        fetch = download(api_key, now=now, session=session)
    return load(con, fetch)
