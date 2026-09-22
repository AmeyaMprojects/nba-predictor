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


def test_item_key_is_stable_and_filesystem_safe():
    k1 = news_rss.item_key("espn", "http://a/b?c=1")
    k2 = news_rss.item_key("espn", "http://a/b?c=1")
    assert k1 == k2
    assert "/" not in k1 and "?" not in k1


def test_archive_entries_stores_each_item_once(store):
    assert news_rss.archive_entries("espn", _parsed(["a", "b"]), NOW) == 2
    assert news_rss.archive_entries("espn", _parsed(["a", "b"]), NOW) == 0
    assert news_rss.archive_entries("espn", _parsed(["a", "b", "c"]), NOW) == 1


def test_archived_item_is_loadable_json(store):
    news_rss.archive_entries("espn", _parsed(["a"]), NOW)
    import json

    key = news_rss.item_key("espn", "a")
    payload = json.loads(raw_store.load("news", key))
    assert payload["title"] == "title a"
    assert payload["feed"] == "espn"


def test_feeds_are_configured():
    assert "espn" in news_rss.FEEDS
    assert all(u.startswith("https://") for u in news_rss.FEEDS.values())
