from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import feedparser

from predictor import raw_store

# Verified live 2026-09-22. nba.com/rss/nba_rss.xml returns 403 — excluded.
FEEDS: dict[str, str] = {
    "espn": "https://www.espn.com/espn/rss/nba/news",
    "yahoo": "https://sports.yahoo.com/nba/rss.xml",
    "cbs": "https://www.cbssports.com/rss/headlines/nba/",
}


def item_key(feed_name: str, entry_id: str) -> str:
    digest = hashlib.sha256(entry_id.encode()).hexdigest()[:20]
    return f"{feed_name}_{digest}.json"


def _entry_id(entry) -> str:
    for attr in ("id", "link", "title"):
        value = getattr(entry, attr, None)
        if value:
            return str(value)
    raise ValueError("feed entry has no usable identifier")


def archive_entries(feed_name: str, parsed, now: datetime) -> int:
    stored = 0
    for entry in getattr(parsed, "entries", []):
        entry_id = _entry_id(entry)
        key = item_key(feed_name, entry_id)
        if raw_store.exists("news", key):
            continue
        payload = {
            "feed": feed_name,
            "entry_id": entry_id,
            "title": getattr(entry, "title", ""),
            "link": getattr(entry, "link", ""),
            "summary": getattr(entry, "summary", ""),
            "published": getattr(entry, "published", ""),
            "observed_at": now.isoformat(),
        }
        raw_store.store(
            "news",
            key,
            json.dumps(payload).encode(),
            now,
            meta={"feed": feed_name},
        )
        stored += 1
    return stored


def poll_all(now: datetime | None = None) -> dict[str, int]:
    now = now or datetime.now(UTC)
    results: dict[str, int] = {}
    for name, url in FEEDS.items():
        try:
            parsed = feedparser.parse(url)
            results[name] = archive_entries(name, parsed, now)
        except Exception as exc:  # a dead feed must not stop the others
            results[name] = -1
            print(f"feed {name} failed: {exc}")
    return results
