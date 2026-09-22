import hashlib
import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from predictor import raw_store
from predictor.config import Settings
from predictor.sources import news_rss


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(raw_store, "settings", s)
    return s


NOW = datetime(2026, 1, 2, 3, 4, tzinfo=UTC)


def _parsed(ids):
    return SimpleNamespace(
        entries=[
            SimpleNamespace(id=i, title=f"title {i}", link=f"http://x/{i}", summary="s")
            for i in ids
        ]
    )


class _FakeResponse:
    """Stands in for a `requests.Response` without touching the network."""

    def __init__(self, content: bytes, status_code: int):
        self.content = content
        self.status_code = status_code


class _FakeSession:
    """Stands in for `requests` (or a `requests.Session`) in tests.

    Either returns a canned `_FakeResponse` from `.get()`, or raises a
    canned exception, so failure paths can be exercised without any real
    network access.
    """

    def __init__(self, response: _FakeResponse | None = None, exc: Exception | None = None):
        self._response = response
        self._exc = exc

    def get(self, url, timeout=None, headers=None):
        if self._exc is not None:
            raise self._exc
        return self._response


def test_item_key_is_stable_and_filesystem_safe():
    k1 = news_rss.item_key("espn", "http://a/b?c=1")
    k2 = news_rss.item_key("espn", "http://a/b?c=1")
    assert k1 == k2
    assert "/" not in k1 and "?" not in k1


def test_archive_entries_stores_each_item_once(store):
    # `archive_entries` now returns an `EntryStats` (new/skipped/conflicts)
    # rather than a bare int of new items, so conflicts (F3) and skipped
    # malformed entries (F4) can be surfaced distinctly. Checking `.new`
    # preserves the original intent of this test.
    assert news_rss.archive_entries("espn", _parsed(["a", "b"]), NOW).new == 2
    assert news_rss.archive_entries("espn", _parsed(["a", "b"]), NOW).new == 0
    assert news_rss.archive_entries("espn", _parsed(["a", "b", "c"]), NOW).new == 1


def test_archived_item_is_loadable_json(store):
    news_rss.archive_entries("espn", _parsed(["a"]), NOW)

    key = news_rss.item_key("espn", "a")
    payload = json.loads(raw_store.load("news", key))
    assert payload["title"] == "title a"
    assert payload["feed"] == "espn"


def test_feeds_are_configured():
    assert "espn" in news_rss.FEEDS
    assert all(u.startswith("https://") for u in news_rss.FEEDS.values())


# --- F3: dedup must not bypass raw_store's conflict guard -----------------


def test_archive_entries_counts_conflict_for_id_reused_with_different_content(store):
    first = SimpleNamespace(id="dup", title="first story", link="http://x/1", summary="s")
    stats1 = news_rss.archive_entries("espn", SimpleNamespace(entries=[first]), NOW)
    assert stats1.new == 1
    assert stats1.conflicts == 0

    # Same id, genuinely different content -- must be recorded as a
    # conflict, not silently dropped by an `exists()` pre-check.
    reused = SimpleNamespace(id="dup", title="a completely different story", link="http://x/1", summary="s")
    stats2 = news_rss.archive_entries("espn", SimpleNamespace(entries=[reused]), NOW)
    assert stats2.new == 0
    assert stats2.conflicts == 1


# --- F4: a malformed entry must not abort the rest of the batch -----------


def test_archive_entries_skips_malformed_entry_without_aborting_batch(store):
    good1 = SimpleNamespace(id="a", title="t", link="http://x/a", summary="s")
    malformed = SimpleNamespace(title="no id or link on this one")
    good2 = SimpleNamespace(id="b", title="t2", link="http://x/b", summary="s")

    stats = news_rss.archive_entries(
        "espn", SimpleNamespace(entries=[good1, malformed, good2]), NOW
    )
    assert stats.new == 2
    assert stats.skipped == 1
    assert stats.conflicts == 0


# --- F5: title alone is not a usable identity ------------------------------


def test_entry_with_only_title_is_treated_as_malformed(store):
    title_only = SimpleNamespace(title="a title and nothing else")
    stats = news_rss.archive_entries("espn", SimpleNamespace(entries=[title_only]), NOW)
    assert stats.new == 0
    assert stats.skipped == 1


# --- F1: raw bytes archived before parsing ---------------------------------


def test_poll_feed_archives_raw_bytes_before_parsing(store):
    raw = b"<rss><channel><item><title>x</title></item></channel></rss>"
    session = _FakeSession(response=_FakeResponse(raw, 200))

    news_rss.poll_feed("espn", "http://example.invalid/feed", NOW, session=session)

    digest = hashlib.sha256(raw).hexdigest()[:16]
    key = f"espn_{digest}.xml"
    assert raw_store.load("news_feeds", key) == raw


def test_poll_feed_raw_archive_is_content_addressed_and_dedupes(store):
    raw = b"<rss><channel></channel></rss>"
    session = _FakeSession(response=_FakeResponse(raw, 200))

    news_rss.poll_feed("espn", "http://example.invalid/feed", NOW, session=session)
    news_rss.poll_feed("espn", "http://example.invalid/feed", NOW, session=session)

    digest = hashlib.sha256(raw).hexdigest()[:16]
    key = f"espn_{digest}.xml"
    manifest_lines = list(raw_store.iter_manifest("news_feeds"))
    assert len(manifest_lines) == 1
    assert manifest_lines[0]["key"] == key


# --- F2: failures must surface loudly, not as "0 new" ----------------------


def test_poll_feed_classifies_non_200_as_failure(store):
    session = _FakeSession(response=_FakeResponse(b"not found", 404))
    result = news_rss.poll_feed("espn", "http://example.invalid/feed", NOW, session=session)
    assert result.ok is False
    assert "404" in result.error


def test_poll_all_classifies_transport_exception_as_failure(store):
    session = _FakeSession(exc=ConnectionError("name resolution failed"))
    results = news_rss.poll_all(NOW, session=session)
    assert results  # at least one configured feed
    for result in results.values():
        assert result.ok is False
        assert "name resolution failed" in result.error


def test_poll_feed_bozo_with_zero_entries_is_failure(store, monkeypatch):
    raw = b"<totally not rss"
    session = _FakeSession(response=_FakeResponse(raw, 200))
    fake_parsed = SimpleNamespace(entries=[], bozo=True, bozo_exception=ValueError("broken xml"))
    monkeypatch.setattr(news_rss.feedparser, "parse", lambda content: fake_parsed)

    result = news_rss.poll_feed("espn", "http://example.invalid/feed", NOW, session=session)
    assert result.ok is False
    assert "broken xml" in result.error


def test_poll_feed_bozo_with_entries_is_success_with_warning(store, monkeypatch):
    raw = b"<rss>slightly off but parseable</rss>"
    session = _FakeSession(response=_FakeResponse(raw, 200))
    fake_parsed = SimpleNamespace(
        entries=_parsed(["a", "b"]).entries,
        bozo=True,
        bozo_exception=ValueError("minor quirk"),
    )
    monkeypatch.setattr(news_rss.feedparser, "parse", lambda content: fake_parsed)

    result = news_rss.poll_feed("espn", "http://example.invalid/feed", NOW, session=session)
    assert result.ok is True
    assert result.new == 2
    assert result.warning is not None
    assert "minor quirk" in result.warning
