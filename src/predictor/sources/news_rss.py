from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime

import feedparser
import requests

from predictor import raw_store

# Verified live 2026-09-22. nba.com/rss/nba_rss.xml returns 403 — excluded.
FEEDS: dict[str, str] = {
    "espn": "https://www.espn.com/espn/rss/nba/news",
    "yahoo": "https://sports.yahoo.com/nba/rss.xml",
    "cbs": "https://www.cbssports.com/rss/headlines/nba/",
}

_TIMEOUT_SECONDS = 30
_USER_AGENT = (
    "predictor-news-rss-archiver/0.1 "
    "(NBA game prediction data pipeline; contact: ameya.s.mhatre@gmail.com)"
)

# HTTP statuses that still mean "we got a usable response". 301/302 are
# accepted because `requests` follows redirects by default, so a 301/302
# reaching here means it's the *original* response of a session/mock that
# doesn't follow redirects -- treating it as fine is more forgiving than
# strictly necessary but never masks a real failure (a redirect loop or a
# redirect to an error page still ends up non-2xx after requests follows
# it, or produces zero/garbage entries and gets caught by the bozo check).
_OK_STATUSES = (200, 301, 302)


@dataclass(frozen=True)
class EntryStats:
    """Outcome of archiving one feed's parsed entries."""

    new: int
    skipped: int
    conflicts: int


@dataclass(frozen=True)
class FeedResult:
    """Outcome of polling one feed, fit for direct display to an end user."""

    feed: str
    ok: bool
    new: int
    skipped: int
    conflicts: int
    warning: str | None
    error: str | None


def item_key(feed_name: str, entry_id: str) -> str:
    digest = hashlib.sha256(entry_id.encode()).hexdigest()[:20]
    return f"{feed_name}_{digest}.json"


def _entry_id(entry) -> str:
    # A headline is too weak to be an identity: two distinct stories can
    # share a title, and falling back to it would silently collide them.
    for attr in ("id", "link"):
        value = getattr(entry, attr, None)
        if value:
            return str(value)
    raise ValueError("feed entry has no usable identifier (id or link)")


def archive_entries(feed_name: str, parsed, now: datetime) -> EntryStats:
    new = 0
    skipped = 0
    conflicts = 0
    for entry in getattr(parsed, "entries", []):
        # Each entry is handled independently: one malformed entry, or one
        # entry whose id collides with different archived content, must
        # never stop the rest of the batch from being archived.
        try:
            entry_id = _entry_id(entry)
            key = item_key(feed_name, entry_id)
            already_present = raw_store.exists("news", key)
            # NOTE: `observed_at` deliberately lives in `meta`, not in the
            # hashed content. It is poll-time metadata (it changes on every
            # single poll, even for an entry whose content never changes),
            # and raw_store's conflict detection hashes only `content`. Put
            # a volatile field like this inside `content` and every re-poll
            # of an unchanged entry becomes a spurious RawStoreConflict --
            # which was observed live while verifying this very fix. The
            # first-archive timestamp is separately, correctly captured by
            # raw_store's own `fetched_at` on the manifest line.
            payload = {
                "feed": feed_name,
                "entry_id": entry_id,
                "title": getattr(entry, "title", ""),
                "link": getattr(entry, "link", ""),
                "summary": getattr(entry, "summary", ""),
                "published": getattr(entry, "published", ""),
            }
            # Always call store() -- never skip it just because exists()
            # said the key is present -- so raw_store's hash comparison
            # (and RawStoreConflict) actually runs. Pre-checking exists()
            # only to decide whether to skip archiving would silently drop
            # a publisher reusing an id for genuinely different content.
            raw_store.store(
                "news",
                key,
                json.dumps(payload).encode(),
                now,
                meta={"feed": feed_name, "observed_at": now.isoformat()},
            )
        except raw_store.RawStoreConflict as exc:
            conflicts += 1
            print(f"news conflict for feed {feed_name!r}: {exc}")
            continue
        except Exception as exc:  # a malformed entry must not abort the batch
            skipped += 1
            print(f"skipping malformed entry in feed {feed_name!r}: {exc}")
            continue
        if not already_present:
            new += 1
    return EntryStats(new=new, skipped=skipped, conflicts=conflicts)


def _fetch(url: str, session) -> tuple[bytes, int | None]:
    sess = session if session is not None else requests
    response = sess.get(
        url, timeout=_TIMEOUT_SECONDS, headers={"User-Agent": _USER_AGENT}
    )
    status = getattr(response, "status_code", None)
    return response.content, status


def archive_raw_feed(feed_name: str, raw_bytes: bytes, now: datetime) -> raw_store.RawRef:
    """Archive the exact bytes fetched over the wire, before any parsing.

    Content-addressed under a separate `news_feeds` source: the key embeds
    the sha256 of the bytes, so an unchanged feed re-archives to the same
    key every poll and is stored exactly once, while a genuinely changed
    feed body gets a new key. RSS cannot be recovered retroactively, so
    this raw snapshot -- not feedparser's interpretation of it -- is the
    thing that must never be lost.
    """
    digest = hashlib.sha256(raw_bytes).hexdigest()[:16]
    key = f"{feed_name}_{digest}.xml"
    return raw_store.store("news_feeds", key, raw_bytes, now, meta={"feed": feed_name})


def poll_feed(feed_name: str, url: str, now: datetime, session=None) -> FeedResult:
    raw_bytes, status = _fetch(url, session)

    # Raw bytes are archived before parsing, always -- even if the status
    # or the parse below turns out to indicate failure, so a human can
    # inspect exactly what the publisher sent.
    archive_raw_feed(feed_name, raw_bytes, now)

    if status is not None and status not in _OK_STATUSES:
        return FeedResult(feed_name, False, 0, 0, 0, None, f"HTTP {status}")

    parsed = feedparser.parse(raw_bytes)
    entries = getattr(parsed, "entries", [])
    bozo = bool(getattr(parsed, "bozo", False))

    if bozo and not entries:
        # feedparser sets bozo for all sorts of quirks, many of them
        # harmless while entries still parse fine -- so bozo alone is not
        # treated as failure. Bozo *and* zero entries means nothing usable
        # came out of parsing, which is a real failure.
        exc = getattr(parsed, "bozo_exception", None)
        return FeedResult(feed_name, False, 0, 0, 0, None, f"feed failed to parse: {exc}")

    warning = None
    if bozo and entries:
        exc = getattr(parsed, "bozo_exception", None)
        warning = f"feed parsed with warnings: {exc}"

    stats = archive_entries(feed_name, parsed, now)
    return FeedResult(
        feed_name, True, stats.new, stats.skipped, stats.conflicts, warning, None
    )


def poll_all(now: datetime | None = None, session=None) -> dict[str, FeedResult]:
    now = now or datetime.now(UTC)
    results: dict[str, FeedResult] = {}
    for name, url in FEEDS.items():
        try:
            results[name] = poll_feed(name, url, now, session=session)
        except Exception as exc:  # a dead feed must not stop the others
            results[name] = FeedResult(name, False, 0, 0, 0, None, str(exc))
            print(f"feed {name} failed: {exc}")
    return results
