from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime

import requests

from predictor import db, raw_store

API_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"


class OddsQuotaExceeded(Exception):
    """The Odds API rejected the request for quota or auth reasons.

    Raised only for a 401 (bad/expired key) or 429 (rate limit / monthly
    quota exhausted) response -- both mean "this call did not get usable
    data because of the account, not the market", and must be reported
    loudly and distinctly from a legitimate "no games right now" response
    (an empty payload with a 200 status), exactly as injury_report's
    TransientFetchError is kept distinct from a confirmed "not published"
    403/404.
    """


def fetch_current(api_key: str, session=None) -> list[dict]:
    session = session or requests.Session()
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
    if response.status_code in (401, 429):
        raise OddsQuotaExceeded(f"{response.status_code}: {response.text[:200]}")
    response.raise_for_status()
    return response.json()


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


def ingest_current(con, api_key: str | None = None, now=None, session=None) -> int:
    api_key = api_key or os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise ValueError(
            "ODDS_API_KEY is not set. Get a free key at https://the-odds-api.com "
            "and export it before running this command."
        )
    # observed_at is when this fact became knowable -- the moment we
    # fetched the snapshot, not some later processing time. Runs through
    # db.require_utc() before it touches either the raw archive or the
    # database, mirroring injury_report.ingest_report and
    # nba_stats.ingest_season (naive/non-UTC must be a hard error here too).
    now = db.require_utc(now or datetime.now(UTC), "observed_at")
    payload = fetch_current(api_key, session=session)

    # Archive before parsing, content-addressed by sha256 -- mirrors
    # news_rss.archive_raw_feed. fetch_current's required return type is
    # already-parsed JSON (list[dict]), not the raw wire bytes (the test
    # suite monkeypatches fetch_current itself to return parsed payloads),
    # so the bytes archived here are a deterministic re-serialization of
    # that payload rather than the literal HTTP response body. It is still
    # content-addressed: polling a quiet market repeatedly produces the
    # same bytes and therefore the same key, so raw_store.store() is a
    # no-op on the second call instead of piling up duplicate snapshots.
    content = json.dumps(payload, sort_keys=True).encode()
    digest = hashlib.sha256(content).hexdigest()[:16]
    raw_store.store(
        "odds",
        f"odds_{digest}.json",
        content,
        now,
        meta={"observed_at": now.isoformat()},
    )

    rows = parse_odds_payload(payload, now)

    # Resolved through db.POINT_IN_TIME_TABLES rather than spelled as a
    # literal here -- the physical "_raw" table names are only allowed to
    # appear as string literals in db.py/asof.py (see
    # test_no_physical_table_name_appears_outside_db_and_asof); this
    # ingestion module must not name the physical table directly either.
    # Mirrors the pattern already used by injury_report.py and nba_stats.py.
    table = db.POINT_IN_TIME_TABLES["odds_snapshots"]
    insert_sql = (
        f"INSERT OR REPLACE INTO {table} (game_key, book, home_team,"
        " away_team, home_price, away_price, spread, total, observed_at)"
        " VALUES (?,?,?,?,?,?,?,?,?)"
    )
    for row in rows:
        con.execute(
            insert_sql,
            [
                row["game_key"], row["book"], row["home_team"], row["away_team"],
                row["home_price"], row["away_price"], row["spread"], row["total"],
                row["observed_at"],
            ],
        )
    return len(rows)
