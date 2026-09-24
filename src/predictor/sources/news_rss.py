from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime

import feedparser
import requests

from predictor import db, raw_store

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
    """Outcome of archiving one feed's parsed entries.

    `conflicts` counts entries whose identifier was already archived with
    DIFFERENT content (a publisher edit under the same id). Despite the
    name, nothing is discarded: the updated content is archived under a
    separate, deterministic VERSIONED key (see `_versioned_key`) and this
    just tells an operator how often that happened this poll.
    """

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


@dataclass(frozen=True)
class IngestStats:
    """Outcome of `ingest_archived_news`."""

    written: int
    skipped_unknown_feed: int


def item_key(feed_name: str, entry_id: str) -> str:
    digest = hashlib.sha256(entry_id.encode()).hexdigest()[:20]
    return f"{feed_name}_{digest}.json"


def _versioned_key(key: str, content: bytes) -> str:
    """Deterministic archive key for content that conflicts with `key`.

    Used when a publisher edits a story under the same identifier: `key`
    already has DIFFERENT bytes recorded for it, so the updated content
    must land somewhere else instead of being dropped. The suffix is
    derived from a hash of the new content itself, not a counter, so:
      - the same updated content always maps to the same versioned key
        (re-polling an unchanged edit is a no-op, not a growing pile of
        copies), and
      - a later, DIFFERENT edit of the same story gets its own distinct
        versioned key (a counter would need external state to avoid
        reusing "v1" after a restart; a content hash needs none).
    """
    digest = hashlib.sha256(content).hexdigest()[:12]
    stem = key[:-5] if key.endswith(".json") else key
    return f"{stem}.v{digest}.json"


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
            content = json.dumps(payload).encode()
            # Always call store() -- never skip it just because exists()
            # said the key is present -- so raw_store's hash comparison
            # (and RawStoreConflict) actually runs. Pre-checking exists()
            # only to decide whether to skip archiving would silently drop
            # a publisher reusing an id for genuinely different content.
            try:
                raw_store.store(
                    "news",
                    key,
                    content,
                    now,
                    meta={"feed": feed_name, "observed_at": now.isoformat()},
                )
            except raw_store.RawStoreConflict:
                # The publisher edited this story under the same identifier.
                # News is the one source in this project that cannot be
                # re-fetched, so the updated bytes must never be dropped:
                # archive them under a NEW, deterministic key instead (see
                # `_versioned_key`). The original key's blob and its
                # original `observed_at` are left completely untouched --
                # this never overwrites, it only adds a new archive entry.
                versioned_key = _versioned_key(key, content)
                version_already_present = raw_store.exists("news", versioned_key)
                raw_store.store(
                    "news",
                    versioned_key,
                    content,
                    now,
                    meta={
                        "feed": feed_name,
                        "observed_at": now.isoformat(),
                        "versions_of": key,
                    },
                )
                conflicts += 1
                print(
                    f"news item changed for feed {feed_name!r}: {key!r} was "
                    f"already archived with different content; the update "
                    f"was archived as a new version {versioned_key!r} "
                    "(original left untouched, nothing was discarded)"
                )
                if not version_already_present:
                    new += 1
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
    version = getattr(parsed, "version", None)

    if not version:
        # feedparser sets `version` to the identified feed format
        # (`'rss20'`, `'atom10'`, ...) or to '' / None when the response
        # body isn't recognizable as any feed format at all -- e.g. an
        # HTML "this feed has moved" page, or an empty body. Neither of
        # those cases reliably sets `bozo`, so they slip past the bozo
        # check below and would otherwise be reported identically to a
        # quiet news day (0 new). Checked ahead of, and independent of,
        # bozo for that reason.
        exc = getattr(parsed, "bozo_exception", None)
        detail = f": {exc}" if exc else ""
        return FeedResult(
            feed_name, False, 0, 0, 0, None,
            f"response was not a recognizable feed{detail}",
        )

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


def ingest_archived_news(con) -> IngestStats:
    """Load archived news JSON from the raw store into `news_items`.

    Returns an `IngestStats` with the number of rows actually written
    (newly-seen items only; see the idempotency note below) and the number
    of archived manifest entries skipped because their feed is not one of
    `FEEDS`.

    FEED ALLOWLIST: the raw store under the "news" source can contain
    entries from anything that was ever archived there, including one-off
    probe/test runs against the real archive that used a feed name never
    wired into this module (this happened for real: 22 rows from a feed
    named "good_espn", which appears nowhere in this file, ended up in
    `news_items` before this filter existed). Only entries whose `feed`
    is currently a key of `FEEDS` are ingested; anything else is skipped
    and counted, and logged so an operator can see it -- never dropped
    silently, since a legitimately renamed feed would show up the same
    way and needs a human to notice and update `FEEDS`. This filter is
    forward-only: `ON CONFLICT DO NOTHING` never deletes, so it cannot
    retroactively remove rows a prior, unfiltered run already wrote.

    `observed_at` is read from the manifest entry's `meta`, NOT from the
    archived blob's own JSON. The blob written by `archive_entries` above
    only ever contains `feed`, `entry_id`, `title`, `link`, `summary`, and
    `published` -- `observed_at` deliberately never enters it (see the NOTE
    in `archive_entries`: it's poll-time metadata, kept out of the content
    hash), so it lives solely in `meta["observed_at"]` on the manifest
    line (with `fetched_at` on that same line as a fallback, for a
    manifest entry archived without meta at all).

    IDEMPOTENCY / point-in-time safety: the physical table behind the
    logical "news_items" name (see `db.POINT_IN_TIME_TABLES`) has
    `item_key` as its ONLY primary key column; `observed_at` is not part
    of it. Using `INSERT OR REPLACE`
    here (as an earlier version of this function did) would let
    re-ingesting an already-seen item silently REWRITE its `observed_at`
    every time this function runs -- retroactively changing when that
    fact became knowable, which is exactly what `AsOfView`'s point-in-time
    guarantee depends on never happening, and it has no defence against
    it. This is not just theoretical: `raw_store.store()`'s own
    self-heal path (blob present on disk, but its manifest line missing,
    e.g. after a crash between the two writes) appends a FRESH manifest
    line -- with today's `fetched_at`/`observed_at` -- for a blob that was
    actually first archived earlier. Re-running this function afterwards
    must not let that newer, wrong timestamp overwrite the true
    first-seen time already recorded in the table.
    Ruling: the FIRST time an item is seen is when it became knowable,
    and that must never move once recorded. So an existing `item_key` is
    left completely untouched (`ON CONFLICT (item_key) DO NOTHING`)
    rather than replaced; only genuinely new keys are written. This makes
    the whole function safe to re-run at any time: re-ingesting an
    unchanged archive twice writes nothing new the second time, and even
    a duplicate/corrected manifest line for an already-known key can never
    move its recorded `observed_at`.
    """
    written = 0
    skipped_unknown_feed = 0
    # Resolved through db.POINT_IN_TIME_TABLES rather than spelled as a
    # literal here -- the physical "_raw" table names are only allowed to
    # appear as string literals in db.py/asof.py (see
    # test_no_physical_table_name_appears_outside_db_and_asof); this
    # ingestion module must not name the physical table directly either.
    table = db.POINT_IN_TIME_TABLES["news_items"]
    insert_sql = (
        f"INSERT INTO {table} (item_key, feed, title, link, summary, observed_at)"
        " VALUES (?,?,?,?,?,?)"
        " ON CONFLICT (item_key) DO NOTHING"
        " RETURNING item_key"
    )
    for entry in raw_store.iter_manifest("news"):
        key = entry["key"]
        payload = json.loads(raw_store.load("news", key))
        meta = entry.get("meta") or {}
        feed = payload.get("feed", meta.get("feed", ""))
        if feed not in FEEDS:
            skipped_unknown_feed += 1
            print(
                f"skipping archived news item {key!r}: feed {feed!r} is not "
                "a configured feed (predictor.sources.news_rss.FEEDS) -- "
                "not ingested into news_items"
            )
            continue
        observed_at_raw = meta.get("observed_at") or entry.get("fetched_at")
        if not observed_at_raw:
            # Never silently lose data: a manifest line with no timestamp
            # anywhere on it is corrupt, not a legitimate item to skip
            # quietly.
            raise ValueError(
                f"news item {key!r} has no observed_at recorded anywhere "
                "in its manifest entry -- refusing to guess when it "
                "became knowable"
            )
        observed_at = db.require_utc(
            datetime.fromisoformat(observed_at_raw), "observed_at"
        )
        rows = con.execute(
            insert_sql,
            [
                key,
                feed,
                payload.get("title", ""),
                payload.get("link", ""),
                payload.get("summary", ""),
                observed_at,
            ],
        ).fetchall()
        if rows:
            written += 1
    return IngestStats(written=written, skipped_unknown_feed=skipped_unknown_feed)
