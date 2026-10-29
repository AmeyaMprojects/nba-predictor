"""The odds cache: the latest odds body, kept in the loop's base dir so a
later iteration (a fresh data dir and DB) still has live market lines.
No network: bodies are archived with raw_store directly."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta

import pytest

from predictor import config, db, raw_store
from predictor.cloud import archive, odds_cache
from predictor.config import Settings
from predictor.model import market
from predictor.sources import odds
from schedule_rows import insert_schedule_row

OBSERVED = datetime(2025, 1, 15, 20, 0, tzinfo=UTC)
GID = "0022400555"


def _payload(price=-150) -> list[dict]:
    home, away = "Philadelphia 76ers", "New York Knicks"
    return [{
        "id": "abc123", "commence_time": "2025-01-16T00:00:00Z",
        "home_team": home, "away_team": away,
        "bookmakers": [{"key": "draftkings", "markets": [
            {"key": "h2h", "outcomes": [{"name": home, "price": price},
                                        {"name": away, "price": 130}]},
        ]}],
    }]


def _point(path, monkeypatch) -> Settings:
    s = Settings(data_dir=path)
    s.ensure_dirs()
    monkeypatch.setattr(raw_store, "settings", s)
    monkeypatch.setattr(config, "settings", s)
    monkeypatch.setattr(db, "settings", s)
    return s


def _con(path):
    con = db.connect(path)
    db.migrate(con)
    insert_schedule_row(con, GID, date(2025, 1, 15), "PHI", "NYK",
                        datetime(2025, 1, 16, 0, 0, tzinfo=UTC))
    return con


def _download(payload, at=OBSERVED) -> odds.OddsDownload:
    body = json.dumps(payload).encode()
    key = f"odds_{abs(hash(body)) % 10**12:012d}.json"
    raw_store.store(odds.RAW_SOURCE, key, body, at, meta={"observed_at": at.isoformat()})
    return odds.OddsDownload(key, at, "480")


def test_odds_cache_round_trip(tmp_path, monkeypatch):
    _point(tmp_path / "it1", monkeypatch)
    cache = tmp_path / "odds-cache"
    downloaded = _download(_payload())
    odds_cache.save(cache, downloaded)

    # A later iteration: a fresh data dir and a fresh DB.
    _point(tmp_path / "it2", monkeypatch)
    con = _con(tmp_path / "it2.duckdb")
    now = OBSERVED + timedelta(hours=3)
    assert market.live_lines(con, GID, now) == []
    assert odds_cache.restore(con, cache, now) == 1
    lines = market.live_lines(con, GID, now)
    assert lines and lines[0].home_price == -150
    # The blob is in this iteration's raw store, under the archive-excluded source.
    assert raw_store.exists(odds.RAW_SOURCE, downloaded.archive_key)
    assert odds.RAW_SOURCE in archive.EXCLUDED_SOURCES


def test_odds_cache_older_than_live_max_age_restores_nothing(tmp_path, monkeypatch):
    _point(tmp_path / "it1", monkeypatch)
    cache = tmp_path / "odds-cache"
    odds_cache.save(cache, _download(_payload()))
    _point(tmp_path / "it2", monkeypatch)
    con = _con(tmp_path / "it2.duckdb")
    late = OBSERVED + market.LIVE_MAX_AGE + timedelta(seconds=1)
    assert odds_cache.restore(con, cache, late) == 0
    assert market.live_lines(con, GID, late) == []
    assert odds_cache.restore(con, cache, OBSERVED + market.LIVE_MAX_AGE) == 1


def test_no_cache_restores_nothing(tmp_path, monkeypatch):
    _point(tmp_path / "d", monkeypatch)
    con = _con(tmp_path / "d.duckdb")
    assert odds_cache.restore(con, tmp_path / "missing", OBSERVED) == 0
    (tmp_path / "empty").mkdir()
    assert odds_cache.restore(con, tmp_path / "empty", OBSERVED) == 0


def test_a_cache_observed_after_now_restores_nothing(tmp_path, monkeypatch):
    _point(tmp_path / "d", monkeypatch)
    cache = tmp_path / "odds-cache"
    odds_cache.save(cache, _download(_payload()))
    con = _con(tmp_path / "d.duckdb")
    assert odds_cache.restore(con, cache, OBSERVED - timedelta(minutes=1)) == 0


def test_save_keeps_only_the_latest_body(tmp_path, monkeypatch):
    _point(tmp_path / "d", monkeypatch)
    cache = tmp_path / "odds-cache"
    first = _download(_payload(-150))
    second = _download(_payload(-170), OBSERVED + timedelta(hours=2))
    odds_cache.save(cache, first)
    odds_cache.save(cache, second)
    assert sorted(p.name for p in cache.iterdir()) == sorted(
        [second.archive_key, odds_cache.LATEST])
    latest = json.loads((cache / odds_cache.LATEST).read_text(encoding="utf-8"))
    assert latest == {"archive_key": second.archive_key,
                      "observed_at": second.observed_at.isoformat(),
                      "requests_remaining": "480"}
    assert (cache / second.archive_key).read_bytes() == raw_store.load(
        odds.RAW_SOURCE, second.archive_key)


@pytest.mark.parametrize("bad", ["../x.json", "a/b.json", ".hidden", ""])
def test_save_refuses_an_unsafe_key(tmp_path, monkeypatch, bad):
    _point(tmp_path / "d", monkeypatch)
    with pytest.raises(odds_cache.OddsCacheError):
        odds_cache.save(tmp_path / "c", odds.OddsDownload(bad, OBSERVED, None))


def test_a_corrupt_cache_raises(tmp_path, monkeypatch):
    _point(tmp_path / "d", monkeypatch)
    con = _con(tmp_path / "d.duckdb")
    cache = tmp_path / "c"
    cache.mkdir()
    (cache / odds_cache.LATEST).write_text("{not json", encoding="utf-8")
    with pytest.raises(odds_cache.OddsCacheError):
        odds_cache.restore(con, cache, OBSERVED)


def test_odds_cache_dir_from_env(monkeypatch, tmp_path):
    monkeypatch.delenv(config.ODDS_CACHE_ENV, raising=False)
    assert config.ODDS_CACHE_ENV == "PREDICTOR_ODDS_CACHE"
    assert config.odds_cache_dir() is None
    monkeypatch.setenv(config.ODDS_CACHE_ENV, "  ")
    assert config.odds_cache_dir() is None
    monkeypatch.setenv(config.ODDS_CACHE_ENV, str(tmp_path / "oc"))
    assert config.odds_cache_dir() == tmp_path / "oc"
