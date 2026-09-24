"""Tests for loading archived news items into the database.

See `news_rss.ingest_archived_news` for the two deviations from a naive
reading of the raw-store archive:

1. `observed_at` is read from the manifest entry's `meta`, not from the
   archived blob's own JSON -- the blob (see `archive_entries`) never
   contains `observed_at` at all.
2. Re-ingesting an already-known `item_key` must never move its recorded
   `observed_at` -- `news_items_raw` has no `observed_at` in its primary
   key, so an `INSERT OR REPLACE` would silently rewrite history. This is
   pinned by `test_reingest_with_a_later_manifest_entry_keeps_first_observed_at`
   below.
"""

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from predictor import db, raw_store
from predictor.config import Settings
from predictor.sources import news_rss

NOW = datetime(2026, 1, 2, 3, 4, tzinfo=UTC)
LATER = datetime(2026, 1, 3, 9, 30, tzinfo=UTC)


@pytest.fixture
def env(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(raw_store, "settings", s)
    con = db.connect(tmp_path / "t.duckdb")
    db.migrate(con)
    return con


def _parsed(ids):
    return SimpleNamespace(
        entries=[
            SimpleNamespace(id=i, title=f"title {i}", link=f"http://x/{i}", summary="s")
            for i in ids
        ]
    )


# news_items_raw is the physical table behind the logical "news_items" name
# (see predictor.db.POINT_IN_TIME_TABLES); tests are exempt from the
# repo-wide "_raw" literal ban enforced by test_leakage.py's
# test_no_physical_table_name_appears_outside_db_and_asof, same as
# test_db.py and test_nba_stats.py already rely on.
_TABLE = db.POINT_IN_TIME_TABLES["news_items"]


def test_ingest_loads_archived_items_into_the_table(env):
    news_rss.archive_entries("espn", _parsed(["a", "b"]), NOW)
    assert news_rss.ingest_archived_news(env) == 2
    rows = env.execute(
        f"SELECT feed, title, observed_at FROM {_TABLE} ORDER BY item_key"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0][0] == "espn"
    assert rows[0][2] == NOW


def test_ingest_is_idempotent(env):
    news_rss.archive_entries("espn", _parsed(["a"]), NOW)
    news_rss.ingest_archived_news(env)
    news_rss.ingest_archived_news(env)
    assert env.execute(f"SELECT count(*) FROM {_TABLE}").fetchone()[0] == 1


def test_ingest_with_no_archive_returns_zero(env):
    assert news_rss.ingest_archived_news(env) == 0


def test_reingest_with_a_later_manifest_entry_keeps_first_observed_at(env):
    """The known defect this function must NOT reproduce.

    `news_items_raw` is keyed by `item_key` alone -- `observed_at` is not
    part of the primary key. `raw_store.store()`'s own self-heal path
    (blob present on disk, its manifest line missing, e.g. after a crash
    between the two writes) can append a SECOND manifest line for the
    SAME key, carrying today's timestamp instead of the item's true
    first-seen time. `iter_manifest` has no dedup, so `ingest_archived_news`
    would see both lines. This simulates exactly that: a duplicate
    manifest line for an already-ingested item_key, with a LATER
    observed_at. The row already written for that item_key must be left
    completely alone.
    """
    news_rss.archive_entries("espn", _parsed(["a"]), NOW)
    news_rss.ingest_archived_news(env)

    key = news_rss.item_key("espn", "a")
    manifest_path = raw_store._manifest_path("news")
    duplicate_line = {
        "source": "news",
        "key": key,
        "sha256": "irrelevant-for-this-test",
        "fetched_at": LATER.isoformat(),
        "size": 0,
        "meta": {"feed": "espn", "observed_at": LATER.isoformat()},
    }
    with manifest_path.open("a") as fh:
        fh.write(json.dumps(duplicate_line) + "\n")

    # Sanity check: iter_manifest really does yield the key twice now, so
    # this test is exercising the duplicate-handling path, not a no-op.
    assert sum(1 for e in raw_store.iter_manifest("news") if e["key"] == key) == 2

    news_rss.ingest_archived_news(env)

    rows = env.execute(
        f"SELECT observed_at FROM {_TABLE} WHERE item_key = ?", [key]
    ).fetchall()
    assert rows == [(NOW,)], "re-ingesting must never move observed_at forward"


def test_ingest_reads_observed_at_from_manifest_meta_not_the_blob(env):
    """Pins that the blob JSON itself has no `observed_at` key.

    `archive_entries` deliberately keeps `observed_at` out of the hashed
    blob content (see its own NOTE) -- it lives only in the manifest
    entry's `meta`. This asserts that invariant directly, so a change to
    `archive_entries` that silently stopped doing this would be caught
    here rather than only surfacing as a KeyError deep in ingestion.
    """
    news_rss.archive_entries("espn", _parsed(["a"]), NOW)
    key = news_rss.item_key("espn", "a")
    payload = json.loads(raw_store.load("news", key))
    assert "observed_at" not in payload

    assert news_rss.ingest_archived_news(env) == 1
    row = env.execute(
        f"SELECT observed_at FROM {_TABLE} WHERE item_key = ?", [key]
    ).fetchone()
    assert row[0] == NOW
