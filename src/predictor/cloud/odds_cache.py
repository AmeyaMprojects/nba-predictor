"""The cloud loop's odds cache: the latest odds response body, kept across
loop iterations.

Each loop iteration has a fresh data dir and DB, and the odds budget allows
two downloads a slate day, so without a cache a revision or final call made
by a later iteration would see no live line. The cache lives in the loop's
base dir (``config.odds_cache_dir()``), outside every iteration's data dir:

- ``save`` (after a successful odds load) copies the archived body to
  ``<cache>/<archive_key>``, writes ``<cache>/latest.json`` (archive key,
  observed_at, requests remaining -- never the API key) and removes older
  bodies.
- ``restore`` (after the rebuild) puts the body back into this run's raw
  store under ``odds.RAW_SOURCE`` -- a source the raw archive never
  publishes -- and loads it into the run's DB, when it is no older than
  ``market.LIVE_MAX_AGE``.

Licensing: odds content never goes to ``state/`` or the raw release; the
cache is never committed or archived, and the loop removes it at the end.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from predictor import db, raw_store
from predictor.model import market
from predictor.sources import odds

LATEST = "latest.json"


class OddsCacheError(Exception):
    """The cache is unusable (bad key, corrupt latest.json, missing body)."""


def _check_key(key: str) -> str:
    if not key or key.startswith(".") or Path(key).name != key or key == LATEST:
        raise OddsCacheError(f"unusable odds archive key {key!r}")
    return key


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def save(cache_dir: Path, downloaded: odds.OddsDownload) -> None:
    """Cache the archived body of ``downloaded`` as the latest odds."""
    key = _check_key(downloaded.archive_key)
    observed_at = db.require_utc(downloaded.observed_at, "observed_at")
    body = raw_store.load(odds.RAW_SOURCE, key)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _write_atomic(cache_dir / key, body)
    latest = {"archive_key": key, "observed_at": observed_at.isoformat(),
              "requests_remaining": downloaded.requests_remaining}
    _write_atomic(cache_dir / LATEST, json.dumps(latest, sort_keys=True).encode())
    for path in cache_dir.iterdir():
        if path.is_file() and path.name not in (key, LATEST):
            path.unlink(missing_ok=True)


def restore(con, cache_dir: Path, now: datetime) -> int:
    """Load the cached odds into ``con``; the number of rows loaded. 0 when
    there is no cache, or it was observed after ``now`` or more than
    ``market.LIVE_MAX_AGE`` before it."""
    db.require_utc(now, "now")
    path = Path(cache_dir) / LATEST
    if not path.is_file():
        return 0
    try:
        latest = json.loads(path.read_text(encoding="utf-8"))
        key = _check_key(latest["archive_key"])
        observed_at = db.require_utc(datetime.fromisoformat(latest["observed_at"]),
                                     "observed_at")
        remaining = latest.get("requests_remaining")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise OddsCacheError(f"the odds cache is unreadable ({type(exc).__name__})") from None
    if observed_at > now or now - observed_at > market.LIVE_MAX_AGE:
        return 0
    blob = Path(cache_dir) / key
    if not blob.is_file():
        raise OddsCacheError(f"the cached odds body {key} is missing")
    raw_store.store(odds.RAW_SOURCE, key, blob.read_bytes(), observed_at,
                    meta={"observed_at": observed_at.isoformat()})
    downloaded = odds.OddsDownload(key, observed_at,
                                   None if remaining is None else str(remaining))
    return odds.load(con, downloaded).rows
