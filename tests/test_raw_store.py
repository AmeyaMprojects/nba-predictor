import hashlib
import json
from datetime import UTC, datetime, timedelta, timezone

import pytest

from predictor import raw_store
from predictor.config import Settings


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(raw_store, "settings", s)
    return s


NOW = datetime(2026, 1, 2, 3, 4, tzinfo=UTC)


def test_store_writes_bytes_and_returns_ref(store):
    ref = raw_store.store("injury", "2025-01-15_05PM.pdf", b"hello", NOW)
    assert ref.path.read_bytes() == b"hello"
    assert ref.size == 5
    assert len(ref.sha256) == 64


def test_exists_and_load_roundtrip(store):
    raw_store.store("injury", "a.pdf", b"payload", NOW)
    assert raw_store.exists("injury", "a.pdf")
    assert raw_store.load("injury", "a.pdf") == b"payload"
    assert not raw_store.exists("injury", "missing.pdf")


def test_store_is_idempotent_and_does_not_duplicate_manifest(store):
    raw_store.store("injury", "a.pdf", b"same", NOW)
    raw_store.store("injury", "a.pdf", b"same", NOW)
    assert len(list(raw_store.iter_manifest("injury"))) == 1


def test_rejects_naive_datetime(store):
    with pytest.raises(ValueError, match="timezone-aware"):
        raw_store.store("injury", "a.pdf", b"x", datetime(2026, 1, 1))


def test_manifest_records_metadata(store):
    raw_store.store("news", "item1", b"x", NOW, meta={"feed": "espn"})
    entry = next(raw_store.iter_manifest("news"))
    assert entry["meta"]["feed"] == "espn"
    assert entry["fetched_at"] == NOW.isoformat()


# --- F1: conflicting content must raise loudly, never silently drop bytes ---


def test_conflicting_content_raises(store):
    raw_store.store("injury", "a.pdf", b"first version", NOW)
    with pytest.raises(raw_store.RawStoreConflict):
        raw_store.store("injury", "a.pdf", b"different version", NOW)
    # The original content must survive untouched.
    assert raw_store.load("injury", "a.pdf") == b"first version"
    assert len(list(raw_store.iter_manifest("injury"))) == 1


# --- F2: a key that looks like the manifest filename must not corrupt it ---


def test_manifest_named_key_is_harmless(store):
    raw_store.store("injury", "_manifest.jsonl", b"not a manifest", NOW)
    raw_store.store("injury", "other.pdf", b"payload", NOW)

    assert raw_store.load("injury", "_manifest.jsonl") == b"not a manifest"
    entries = list(raw_store.iter_manifest("injury"))
    assert len(entries) == 2
    keys = {e["key"] for e in entries}
    assert keys == {"_manifest.jsonl", "other.pdf"}


# --- F4: percent-encoding must be injective; distinct keys never collide ---


def test_distinct_keys_with_unsafe_chars_do_not_collide(store):
    raw_store.store("odds", "a?b", b"question mark", NOW)
    raw_store.store("odds", "a b", b"space", NOW)

    assert raw_store.load("odds", "a?b") == b"question mark"
    assert raw_store.load("odds", "a b") == b"space"
    assert raw_store.blob_path("odds", "a?b") != raw_store.blob_path("odds", "a b")


def test_already_safe_key_is_unchanged_on_disk(store):
    key = "Injury-Report_2025-01-15_05PM.pdf"
    ref = raw_store.store("injury", key, b"payload", NOW)
    assert ref.path.name == key


def test_percent_itself_is_escaped(store):
    # '%' must not be a safe character, otherwise encoding is not injective
    # (e.g. a literal "%41" could collide with an encoded "A").
    raw_store.store("odds", "100%done", b"literal percent", NOW)
    raw_store.store("odds", "100%2541done", b"looks like encoded percent", NOW)
    assert raw_store.load("odds", "100%done") == b"literal percent"
    assert raw_store.load("odds", "100%2541done") == b"looks like encoded percent"


# --- F3: manifest <-> disk divergence must self-heal in both directions ---


def test_missing_manifest_record_self_heals(store):
    # Simulate a crash: the blob was written but the manifest line never
    # made it to disk.
    blob = raw_store.blob_path("injury", "orphan.pdf")
    blob.write_bytes(b"orphan content")

    raw_store.store("injury", "orphan.pdf", b"orphan content", NOW)

    entries = list(raw_store.iter_manifest("injury"))
    assert len(entries) == 1
    assert entries[0]["key"] == "orphan.pdf"

    # Calling store again must not duplicate the now-healed manifest entry.
    raw_store.store("injury", "orphan.pdf", b"orphan content", NOW)
    assert len(list(raw_store.iter_manifest("injury"))) == 1


def test_missing_blob_self_heals(store):
    raw_store.store("injury", "a.pdf", b"payload", NOW)
    raw_store.blob_path("injury", "a.pdf").unlink()
    assert not raw_store.exists("injury", "a.pdf")

    raw_store.store("injury", "a.pdf", b"payload", NOW)

    assert raw_store.load("injury", "a.pdf") == b"payload"
    # No duplicate manifest line was written for the self-heal.
    assert len(list(raw_store.iter_manifest("injury"))) == 1


# --- F5: only UTC (zero offset) timestamps are accepted ---


def test_rejects_non_utc_offset(store):
    non_utc = datetime(2026, 1, 2, 3, 4, tzinfo=timezone(timedelta(hours=-5)))
    with pytest.raises(ValueError, match="UTC"):
        raw_store.store("injury", "a.pdf", b"x", non_utc)


# --- Regression: a stale in-memory index cache must never cause a
# duplicate manifest append on an idempotent retry ---


def test_stale_cache_does_not_duplicate_manifest_entry(store):
    # Prime this process's cache for the source while it is still empty,
    # exactly as an unrelated earlier call in the same process would.
    raw_store._index("injury")

    # Simulate another writer -- a concurrent process, or an earlier run of
    # this one -- recording a key after our cache was built: write the blob
    # and append a correct manifest line directly, bypassing store().
    content = b"payload"
    blob = raw_store.blob_path("injury", "a.pdf")
    blob.write_bytes(content)
    entry = {
        "source": "injury",
        "key": "a.pdf",
        "sha256": hashlib.sha256(content).hexdigest(),
        "fetched_at": NOW.isoformat(),
        "size": len(content),
        "meta": {},
    }
    with raw_store._manifest_path("injury").open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    assert len(list(raw_store.iter_manifest("injury"))) == 1

    # The canonical idempotent retry: same key, matching content, through
    # this (stale-cached) process.
    raw_store.store("injury", "a.pdf", content, NOW)

    entries = list(raw_store.iter_manifest("injury"))
    assert len(entries) == 1


# --- I4: a torn manifest line must not take down the whole source --------


def test_iter_manifest_skips_a_torn_trailing_line_and_keeps_the_rest(store, capsys):
    raw_store.store("injury", "a.pdf", b"first", NOW)
    raw_store.store("injury", "b.pdf", b"second", NOW)
    manifest = raw_store._manifest_path("injury")
    # Simulate the exact damage an interrupted write (no fsync -- Ruling
    # 10) leaves behind: a truncated, unparseable JSON fragment with no
    # trailing newline, appended after two good lines.
    with manifest.open("a") as fh:
        fh.write('{"source": "injury", "key": "torn.pdf", "sha256": "dead')

    entries = list(raw_store.iter_manifest("injury"))
    assert {e["key"] for e in entries} == {"a.pdf", "b.pdf"}
    assert "WARNING" in capsys.readouterr().out


def test_index_load_skips_a_torn_line_instead_of_raising(store):
    raw_store.store("injury", "a.pdf", b"first", NOW)
    manifest = raw_store._manifest_path("injury")
    with manifest.open("a") as fh:
        fh.write("{not even close to valid json")

    # Building the index from disk (what _refresh_index/_load_index_from_disk
    # do) must not raise -- a fresh process reading this exact manifest for
    # the first time must still see "a.pdf" as archived.
    raw_store._index_cache.clear()
    idx = raw_store._index("injury")
    assert "a.pdf" in idx


def test_a_torn_trailing_line_does_not_permanently_break_store(store):
    # The compounding half of I4: before this fix, `_refresh_index` (which
    # `store()` calls on the path that decides whether to append) re-read
    # the manifest on every new-key append -- so a single torn line, once
    # written, made EVERY future store() call for that source raise
    # forever. This is the regression that matters most: the news
    # archiver (the one source that cannot be recovered retroactively)
    # must keep archiving after a torn line, not stop permanently.
    raw_store.store("injury", "a.pdf", b"first", NOW)
    manifest = raw_store._manifest_path("injury")
    with manifest.open("a") as fh:
        fh.write('{"source": "injury", "key": "torn')  # no trailing newline

    raw_store.store("injury", "c.pdf", b"third", NOW)

    assert raw_store.exists("injury", "c.pdf")
    entries = {e["key"] for e in raw_store.iter_manifest("injury")}
    assert entries == {"a.pdf", "c.pdf"}


def test_store_isolates_a_torn_line_instead_of_fusing_onto_it(store):
    # Second half of I4: a torn fragment has no trailing newline, so a
    # naive append would concatenate the new, well-formed JSON line onto
    # the tail of the torn one -- permanently fusing them into a second,
    # different unparseable line. store() must insert a newline first so
    # the torn fragment stays isolated as its own (still-torn, still
    # skipped) line, and the new entry becomes its own clean line.
    raw_store.store("injury", "a.pdf", b"first", NOW)
    manifest = raw_store._manifest_path("injury")
    with manifest.open("a") as fh:
        fh.write('{"source": "injury", "key": "torn.pdf", "sha256": "dead')

    raw_store.store("injury", "c.pdf", b"third", NOW)

    lines = manifest.read_text().splitlines()
    # The line the append produced for "c.pdf" must parse on its own and
    # must not have absorbed the torn fragment's text.
    matching = [json.loads(line) for line in lines if '"c.pdf"' in line]
    assert len(matching) == 1
    assert matching[0]["key"] == "c.pdf"
