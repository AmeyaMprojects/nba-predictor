import hashlib
import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
import requests

from predictor import raw_store
from predictor.config import Settings
from predictor.sources import news_rss


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(raw_store, "settings", s)
    return s


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    """Every test in this module must run fast -- stub the sleep
    indirection instead of letting exponential backoff actually block.
    Mirrors `tests/test_injury_fetch.py`'s identically-named fixture.
    """
    monkeypatch.setattr(news_rss, "_sleep", lambda seconds: None)


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
        self.calls = getattr(self, "calls", 0) + 1
        if self._exc is not None:
            raise self._exc
        return self._response


class _FlakySession:
    """Raises `exc` for the first `fail_times` calls, then returns `response`.

    Simulates a machine whose network comes back up partway through the
    retry window -- the case that actually fixes the launchd-wake bug.
    """

    def __init__(self, exc: Exception, fail_times: int, response: "_FakeResponse"):
        self._exc = exc
        self._fail_times = fail_times
        self._response = response
        self.calls = 0

    def get(self, url, timeout=None, headers=None):
        self.calls += 1
        if self.calls <= self._fail_times:
            raise self._exc
        return self._response


class _PerUrlSession:
    """Routes `.get()` per URL -- one feed can fail while others succeed,
    exercising per-feed isolation through `poll_all` with a single shared
    session object (mirroring how the real CLI passes one session).
    """

    def __init__(self, responses: dict | None = None, excs: dict | None = None):
        self._responses = responses or {}
        self._excs = excs or {}
        self.calls: list[str] = []

    def get(self, url, timeout=None, headers=None):
        self.calls.append(url)
        if url in self._excs:
            raise self._excs[url]
        return self._responses[url]


_VALID_RSS = (
    b"<rss version='2.0'><channel><item>"
    b"<title>t</title><link>http://x/1</link>"
    b"</item></channel></rss>"
)


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
# --- Finding 1: an updated story must be VERSIONED, never discarded -------


def test_archive_entries_versions_updated_content_instead_of_discarding_it(store):
    first = SimpleNamespace(id="dup", title="first story", link="http://x/1", summary="s")
    stats1 = news_rss.archive_entries("espn", SimpleNamespace(entries=[first]), NOW)
    assert stats1.new == 1
    assert stats1.conflicts == 0

    original_key = news_rss.item_key("espn", "dup")

    # Same id, genuinely different content (a publisher edit) -- must be
    # archived under a NEW versioned key, not silently dropped by an
    # `exists()` pre-check, and not discarded when raw_store refuses to
    # overwrite the original.
    reused = SimpleNamespace(
        id="dup", title="a completely different story", link="http://x/1", summary="s"
    )
    later = datetime(2026, 1, 3, 9, 30, tzinfo=UTC)
    stats2 = news_rss.archive_entries("espn", SimpleNamespace(entries=[reused]), later)
    # The updated content lands as an ADDITIONAL archive entry (a new
    # `new` item under a new key), and `conflicts` still reports that an
    # identifier's content changed -- it just no longer means "discarded".
    assert stats2.new == 1
    assert stats2.conflicts == 1
    assert stats2.skipped == 0

    # The ORIGINAL blob and its ORIGINAL observed_at are completely
    # untouched: bytes...
    original_payload = json.loads(raw_store.load("news", original_key))
    assert original_payload["title"] == "first story"
    # ...and manifest metadata (observed_at), including after the later poll.
    original_manifest_entry = next(
        e for e in raw_store.iter_manifest("news") if e["key"] == original_key
    )
    assert original_manifest_entry["meta"]["observed_at"] == NOW.isoformat()

    # The updated content is archived under a DIFFERENT key.
    manifest = list(raw_store.iter_manifest("news"))
    assert len(manifest) == 2
    versioned_entries = [e for e in manifest if e["key"] != original_key]
    assert len(versioned_entries) == 1
    versioned_key = versioned_entries[0]["key"]
    assert versioned_key != original_key
    versioned_payload = json.loads(raw_store.load("news", versioned_key))
    assert versioned_payload["title"] == "a completely different story"

    # IDEMPOTENT: re-polling the exact same updated content again must not
    # create a third copy -- it maps to the same versioned key.
    stats3 = news_rss.archive_entries("espn", SimpleNamespace(entries=[reused]), later)
    assert stats3.new == 0
    assert stats3.conflicts == 1
    manifest_after_repoll = list(raw_store.iter_manifest("news"))
    assert len(manifest_after_repoll) == 2, "re-polling the same edit must not add a 3rd copy"


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
    # `version="rss20"` because this represents a feed that IS recognized
    # as a known format but whose body is malformed enough that feedparser
    # couldn't extract any entries -- distinct from Finding A's "not a
    # feed at all" case below, which fails on `version` before this branch
    # is ever reached.
    fake_parsed = SimpleNamespace(
        entries=[], bozo=True, bozo_exception=ValueError("broken xml"), version="rss20"
    )
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
        version="rss20",
    )
    monkeypatch.setattr(news_rss.feedparser, "parse", lambda content: fake_parsed)

    result = news_rss.poll_feed("espn", "http://example.invalid/feed", NOW, session=session)
    assert result.ok is True
    assert result.new == 2
    assert result.warning is not None
    assert "minor quirk" in result.warning


# --- Finding A: an unrecognizable response body must not read as a quiet day


def test_poll_feed_html_body_is_classified_as_failure(store, monkeypatch):
    # A publisher serving a "this feed has moved" HTML page for what used
    # to be the feed URL. feedparser typically does NOT set bozo for
    # well-formed HTML, and reports zero entries -- so this must be caught
    # by the `version` check, not by the bozo check.
    raw = b"<html><body>This feed has moved.</body></html>"
    session = _FakeSession(response=_FakeResponse(raw, 200))
    fake_parsed = SimpleNamespace(entries=[], bozo=False, bozo_exception=None, version="")
    monkeypatch.setattr(news_rss.feedparser, "parse", lambda content: fake_parsed)

    result = news_rss.poll_feed("espn", "http://example.invalid/feed", NOW, session=session)
    assert result.ok is False
    assert "not a recognizable feed" in result.error


def test_poll_feed_empty_body_is_classified_as_failure(store, monkeypatch):
    raw = b""
    session = _FakeSession(response=_FakeResponse(raw, 200))
    fake_parsed = SimpleNamespace(entries=[], bozo=False, bozo_exception=None, version=None)
    monkeypatch.setattr(news_rss.feedparser, "parse", lambda content: fake_parsed)

    result = news_rss.poll_feed("espn", "http://example.invalid/feed", NOW, session=session)
    assert result.ok is False
    assert "not a recognizable feed" in result.error


def test_poll_feed_valid_but_empty_feed_is_success_with_zero_new(store, monkeypatch):
    # The discriminator for Finding A is `version`, not entry count: a
    # feed that IS recognized as e.g. rss20 but genuinely has zero items
    # right now is a legitimate quiet day, not a failure.
    raw = b"<rss version='2.0'><channel></channel></rss>"
    session = _FakeSession(response=_FakeResponse(raw, 200))
    fake_parsed = SimpleNamespace(entries=[], bozo=False, bozo_exception=None, version="rss20")
    monkeypatch.setattr(news_rss.feedparser, "parse", lambda content: fake_parsed)

    result = news_rss.poll_feed("espn", "http://example.invalid/feed", NOW, session=session)
    assert result.ok is True
    assert result.new == 0
    assert result.error is None


# --- Finding B: pin the observed_at-must-not-be-hashed invariant -----------


def test_rearchiving_unchanged_entry_with_a_different_now_produces_no_conflict(store):
    entry = SimpleNamespace(id="a", title="t", link="http://x/a", summary="s")
    key = news_rss.item_key("espn", "a")

    first_now = NOW
    stats1 = news_rss.archive_entries("espn", SimpleNamespace(entries=[entry]), first_now)
    assert stats1.new == 1
    assert stats1.conflicts == 0

    # Logically identical content (same entry, unchanged), but polled at a
    # later time. If `observed_at` (or any other poll-time-only value) ever
    # leaks back into the hashed content, this second archive of the exact
    # same story would falsely look like a content change and raise
    # RawStoreConflict on every single already-archived item of every poll
    # -- exactly the regression caught live while implementing F3.
    later_now = datetime(2026, 1, 3, 9, 30, tzinfo=UTC)
    assert later_now != first_now
    stats2 = news_rss.archive_entries("espn", SimpleNamespace(entries=[entry]), later_now)
    assert stats2.new == 0
    assert stats2.conflicts == 0
    assert stats2.skipped == 0

    manifest_entries = [e for e in raw_store.iter_manifest("news") if e["key"] == key]
    assert len(manifest_entries) == 1


# --- News retry: launchd-wake DNS failures are transient, not fatal --------
#
# Regression guard for the live defect: launchd fires `poll-news` at a fixed
# wall-clock time, and a laptop asleep at that moment wakes with networking
# not yet up, so ALL THREE feeds fail in the same run with
# `NameResolutionError` (a `requests.exceptions.ConnectionError`). These
# tests prove the fetch now retries that specific, transient condition with
# exponential backoff before giving up, while an HTTP status failure (a
# completed response, not a transient network condition) is still NOT
# retried, and one feed's exhausted retries still don't stop the others.


def _isolated_failure_session(exc: Exception) -> _PerUrlSession:
    """One feed (espn) always raises `exc`; the other two configured feeds
    succeed trivially. Isolates the attempt-count assertion to the single
    failing feed -- `poll_all` always iterates all of `FEEDS`, so a session
    that fails uniformly for every URL would let a later feed's calls
    contaminate the failing feed's own call count.
    """
    espn_url = news_rss.FEEDS["espn"]
    other_urls = [u for name, u in news_rss.FEEDS.items() if name != "espn"]
    return _PerUrlSession(
        responses={u: _FakeResponse(_VALID_RSS, 200) for u in other_urls},
        excs={espn_url: exc},
    )


def test_poll_feed_retries_connection_error_then_reports_failed(store):
    # `poll_feed` itself does not catch a fetch exception -- only `poll_all`
    # does (see `test_poll_all_classifies_transport_exception_as_failure`,
    # unchanged by this fix) -- so this is exercised through `poll_all`,
    # matching the module's existing division of responsibility.
    session = _isolated_failure_session(
        requests.exceptions.ConnectionError(
            "Failed to resolve 'www.espn.com' "
            "([Errno 8] nodename nor servname provided, or not known)"
        )
    )
    results = news_rss.poll_all(NOW, session=session)
    result = results["espn"]

    assert result.ok is False
    assert "www.espn.com" in result.error
    assert session.calls.count(news_rss.FEEDS["espn"]) == news_rss._NEWS_MAX_ATTEMPTS, (
        "a persistent connection error should retry up to the configured "
        "attempt limit, not fail on the first attempt"
    )


def test_poll_feed_retries_timeout_then_reports_failed(store):
    session = _isolated_failure_session(requests.exceptions.Timeout("timed out"))
    results = news_rss.poll_all(NOW, session=session)
    result = results["espn"]

    assert result.ok is False
    assert "timed out" in result.error
    assert session.calls.count(news_rss.FEEDS["espn"]) == news_rss._NEWS_MAX_ATTEMPTS


def test_poll_feed_succeeds_when_network_recovers_mid_retry(store):
    # This is the case that actually fixes the reported bug: the machine's
    # network comes back up partway through the retry window, so the feed
    # is archived successfully instead of being reported FAILED.
    session = _FlakySession(
        exc=requests.exceptions.ConnectionError("Failed to resolve 'www.espn.com'"),
        fail_times=news_rss._NEWS_MAX_ATTEMPTS - 1,  # succeeds on the last attempt
        response=_FakeResponse(_VALID_RSS, 200),
    )
    result = news_rss.poll_feed("espn", "http://example.invalid/feed", NOW, session=session)

    assert result.ok is True
    assert result.error is None
    assert result.new == 1
    assert session.calls == news_rss._NEWS_MAX_ATTEMPTS

    key = news_rss.item_key("espn", "http://x/1")
    assert raw_store.exists("news", key), "the archived item must actually be on disk"


def test_poll_feed_does_not_retry_a_non_2xx_status(store):
    session = _FakeSession(response=_FakeResponse(b"not found", 404))
    result = news_rss.poll_feed("espn", "http://example.invalid/feed", NOW, session=session)

    assert result.ok is False
    assert "404" in result.error
    assert session.calls == 1, (
        "a completed HTTP error response is not a transient network "
        "condition and must not be retried"
    )


def test_one_feed_exhausting_retries_does_not_block_the_others(store):
    espn_url = news_rss.FEEDS["espn"]
    yahoo_url = news_rss.FEEDS["yahoo"]
    cbs_url = news_rss.FEEDS["cbs"]

    session = _PerUrlSession(
        responses={
            yahoo_url: _FakeResponse(_VALID_RSS, 200),
            cbs_url: _FakeResponse(_VALID_RSS, 200),
        },
        excs={
            espn_url: requests.exceptions.ConnectionError("Failed to resolve 'www.espn.com'"),
        },
    )

    results = news_rss.poll_all(NOW, session=session)

    assert results["espn"].ok is False
    assert "www.espn.com" in results["espn"].error
    assert results["yahoo"].ok is True
    assert results["yahoo"].new == 1
    assert results["cbs"].ok is True
    assert results["cbs"].new == 1
    assert session.calls.count(espn_url) == news_rss._NEWS_MAX_ATTEMPTS
    assert session.calls.count(yahoo_url) == 1
    assert session.calls.count(cbs_url) == 1


def test_poll_news_cli_exits_nonzero_when_retries_are_exhausted(store, monkeypatch):
    # Full stack, real retry path (no CLI-level stubbing of poll_all): a
    # feed whose connection error survives every retry attempt must still
    # make `poll-news` exit non-zero, exactly as an immediate failure did
    # before this retry was added.
    from typer.testing import CliRunner

    from predictor import cli, config, db

    monkeypatch.setattr(config, "settings", store)
    monkeypatch.setattr(db, "settings", store)

    espn_url = news_rss.FEEDS["espn"]
    yahoo_url = news_rss.FEEDS["yahoo"]
    cbs_url = news_rss.FEEDS["cbs"]
    session = _PerUrlSession(
        responses={
            yahoo_url: _FakeResponse(_VALID_RSS, 200),
            cbs_url: _FakeResponse(_VALID_RSS, 200),
        },
        excs={
            espn_url: requests.exceptions.ConnectionError("Failed to resolve 'www.espn.com'"),
        },
    )
    real_poll_all = news_rss.poll_all
    monkeypatch.setattr(news_rss, "poll_all", lambda: real_poll_all(NOW, session=session))

    runner = CliRunner()
    result = runner.invoke(cli.app, ["poll-news"])

    assert result.exit_code == 1
    assert "espn: FAILED" in result.stdout
    assert "WARNING: 1 feed(s) failed this run: espn." in result.stdout
