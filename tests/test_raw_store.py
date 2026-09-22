import json
from datetime import UTC, datetime

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
