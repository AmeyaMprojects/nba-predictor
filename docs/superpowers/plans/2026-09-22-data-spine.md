# Data Spine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the point-in-time-correct data foundation for the NBA predictor — raw archival, a DuckDB store with an enforced as-of accessor, and ingestion of games, injury reports, odds, and news.

**Architecture:** Raw-first ingestion writes every fetched byte to a content-addressed archive on disk before parsing. Parsed facts land in DuckDB, every row stamped with `observed_at` (when the fact became *knowable*). All downstream feature reads go through a single `AsOfView` accessor that hard-filters `observed_at > as_of`, making leakage structurally difficult rather than merely discouraged.

**Tech Stack:** Python 3.14.6, `uv`, DuckDB 1.5.5, `nba_api` 1.11.4, `pdfplumber`, `feedparser`, `typer`, `tenacity`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-09-22-nba-predictor-design.md`

## Global Constraints

- **Python 3.14**; environments managed with `uv`. Verified 2026-09-22 that all dependencies below install and run on 3.14.6.
- **Free data sources only.** No paid API integrations. The Odds API free tier is budgeted to at most one request per day.
- **No LLM calls anywhere in the pipeline.** It emits structured data only.
- **All timestamps stored in UTC**, timezone-aware. Naive datetimes are a hard error, never silently coerced.
- **`observed_at` is mandatory** on every point-in-time table. It records when a fact became knowable, not when the event occurred.
- **Raw bytes are archived before parsing**, always. Forward-only sources (news) are irreplaceable if dropped.
- **Never silently publish stale data.** Missing or stale inputs must surface loudly, not degrade quietly.
- **Politeness:** Basketball-Reference throttled to ≤20 requests/minute; NBA endpoints get exponential backoff and never run unthrottled loops.

## Verified ground truth

Established by direct probing on 2026-09-22 — implementers should trust these over assumptions:

- **Injury reports:** `https://ak-static.cms.nba.com/referee/injury/Injury-Report_YYYY-MM-DD_HHPM.pdf`. Published **hourly** (all 24 slots resolve in season). Archive reaches back to ~**2019-12-10**; 2018-12-11 returns 403.
- The PDF's first line carries the authoritative publication timestamp, e.g. `Injury Report: 01/15/25 05:30 PM` (Eastern). This **differs from the filename hour** (`05PM` file → `05:30 PM` content). The content timestamp wins.
- Only **page 1** carries the column header row; later pages must reuse page 1's column boundaries.
- Extracted text **drops intra-field spaces** (`NewYorkKnicks`, `Brunson,Jalen`). Parsing must use word x-coordinates.
- `Reason` wraps across lines, and fragments can appear **above as well as below** the player's line.
- Reference fixture: `2025-01-15_05PM` → 10 pages, **161 rows**, 14 matchups, statuses `{Out, Questionable, Probable, Doubtful, Available}`.
- **Live RSS feeds (200 OK):** ESPN `https://www.espn.com/espn/rss/nba/news`, Yahoo `https://sports.yahoo.com/nba/rss.xml`, CBS `https://www.cbssports.com/rss/headlines/nba/`. NBA.com's `nba_rss.xml` returns 403 — do not use.
- **DuckDB requires `pytz`** to hand `TIMESTAMP WITH TIME ZONE` values back to Python. Without it, any query selecting `observed_at` raises `ModuleNotFoundError: No module named 'pytz'`. It is not pulled in automatically — it is declared explicitly in `pyproject.toml`.
- DuckDB returns timestamptz values in the **machine's local timezone** unless the session is pinned. `connect()` issues `SET TimeZone='UTC'` so results are identical on every machine. Values still compare equal either way (comparison is instant-based), but unpinned output is confusing to debug.
- `INSERT OR REPLACE` with a composite primary key gives idempotent re-ingestion — verified that inserting the same row twice leaves one row.
- `nba_api` `LeagueGameFinder` returns **two rows per game** (one per team) with `GAME_ID`, `GAME_DATE`, `MATCHUP`, `WL`, `PTS`. Season `2024-25` yields 2802 rows.

## File structure

```
pyproject.toml
src/predictor/
  __init__.py
  config.py            # paths and settings, env-overridable
  raw_store.py         # content-addressed raw byte archive
  db.py                # DuckDB connection + schema migrations
  asof.py              # AsOfView point-in-time accessor (critical)
  status.py            # health reporting
  cli.py               # typer entry point
  sources/
    __init__.py
    injury_report.py   # URL building, fetch, PDF parse, ingest
    nba_stats.py       # nba_api games ingestion
    news_rss.py        # RSS archiver
    odds.py            # The Odds API client
tests/
  fixtures/Injury-Report_2025-01-15_05PM.pdf
  test_config.py
  test_raw_store.py
  test_db.py
  test_asof.py
  test_leakage.py      # deliberate-attack tests
  test_injury_parse.py
  test_injury_fetch.py
  test_injury_backfill.py
  test_nba_stats.py
  test_news_rss.py
  test_news_ingest.py
  test_odds.py
  test_status.py
scripts/
  com.predictor.daily.plist
```

**Task ordering note:** the news archiver and its schedule come early (Tasks 3-4) because RSS is the only genuinely irreplaceable source — every day it is not running is permanently lost. Injury data is backfillable, so it follows.

---

### Task 1: Project scaffolding and CLI skeleton

**Files:**
- Create: `pyproject.toml`, `src/predictor/__init__.py`, `src/predictor/config.py`, `src/predictor/cli.py`, `.gitignore`
- Test: `tests/conftest.py`, `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `config.settings` (a `Settings` instance with `data_dir: Path`, `raw_dir: Path`, `db_path: Path`); `cli.app` (a `typer.Typer`).

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "predictor"
version = "0.1.0"
requires-python = ">=3.14"
dependencies = [
    "duckdb>=1.5.5",
    "nba_api>=1.11.4",
    "pandas>=3.0.6",
    "pdfplumber>=0.11",
    "feedparser>=6.0.14",
    "typer>=0.27",
    "tenacity>=9.1",
    "requests>=2.34",
    # Required by DuckDB to return TIMESTAMP WITH TIME ZONE values to Python.
    # Without it, any query selecting observed_at raises ModuleNotFoundError.
    "pytz>=2024.1",
]

[project.optional-dependencies]
dev = ["pytest>=9.1"]

[project.scripts]
predictor = "predictor.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/predictor"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `.gitignore`**

```
.venv/
data/
__pycache__/
*.pyc
.pytest_cache/
```

`data/` is ignored: raw archives and the DuckDB file are large and regenerable-in-principle. The prediction log added in a later sub-project lives outside `data/` precisely so it *is* committed.

- [ ] **Step 3: Write the failing test**

Create `tests/test_config.py`:

```python
from pathlib import Path
from predictor.config import Settings


def test_settings_derive_paths_from_data_dir(tmp_path):
    s = Settings(data_dir=tmp_path)
    assert s.raw_dir == tmp_path / "raw"
    assert s.db_path == tmp_path / "predictor.duckdb"


def test_ensure_dirs_creates_them(tmp_path):
    s = Settings(data_dir=tmp_path / "nested")
    s.ensure_dirs()
    assert s.raw_dir.is_dir()
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'predictor'`

- [ ] **Step 5: Implement `config.py`**

```python
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    data_dir: Path

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "predictor.duckdb"

    def ensure_dirs(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)


def _default_data_dir() -> Path:
    env = os.environ.get("PREDICTOR_DATA_DIR")
    return Path(env).expanduser() if env else PROJECT_ROOT / "data"


settings = Settings(data_dir=_default_data_dir())
```

- [ ] **Step 6: Implement `cli.py` skeleton and `__init__.py`**

`src/predictor/__init__.py` is empty. `src/predictor/cli.py`:

```python
from __future__ import annotations

import typer

app = typer.Typer(help="NBA prediction data spine and pipeline.")


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo("predictor 0.1.0")


if __name__ == "__main__":
    app()
```

- [ ] **Step 7: Create the environment and verify**

```bash
cd /Users/meya/Desktop/projects/predictor
uv venv --python 3.14
uv pip install -e ".[dev]"
uv run pytest tests/test_config.py -v
uv run predictor version
```

Expected: tests PASS; `predictor 0.1.0` printed.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .gitignore src tests
git commit -m "feat: project scaffolding and CLI skeleton"
```

---

### Task 2: Raw byte archive

Stores every fetched artifact verbatim before parsing, with a sidecar manifest. Deliberately independent of DuckDB so archival can begin before any schema exists.

**Files:**
- Create: `src/predictor/raw_store.py`
- Test: `tests/test_raw_store.py`

**Interfaces:**
- Consumes: `config.settings`.
- Produces:
  - `RawRef` dataclass — fields `source: str`, `key: str`, `sha256: str`, `path: Path`, `fetched_at: datetime`, `size: int`.
  - `store(source: str, key: str, content: bytes, fetched_at: datetime, meta: dict | None = None) -> RawRef`
  - `exists(source: str, key: str) -> bool`
  - `load(source: str, key: str) -> bytes`
  - `iter_manifest(source: str) -> Iterator[dict]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_raw_store.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_raw_store.py -v`
Expected: FAIL — `ImportError: cannot import name 'raw_store'`

- [ ] **Step 3: Implement `raw_store.py`**

```python
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from predictor.config import settings

_SAFE = re.compile(r"[^A-Za-z0-9._@-]")


@dataclass(frozen=True)
class RawRef:
    source: str
    key: str
    sha256: str
    path: Path
    fetched_at: datetime
    size: int


def _safe(name: str) -> str:
    cleaned = _SAFE.sub("_", name)
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError(f"unusable name: {name!r}")
    return cleaned


def _source_dir(source: str) -> Path:
    d = settings.raw_dir / _safe(source)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _manifest_path(source: str) -> Path:
    return _source_dir(source) / "_manifest.jsonl"


def blob_path(source: str, key: str) -> Path:
    return _source_dir(source) / _safe(key)


def exists(source: str, key: str) -> bool:
    return blob_path(source, key).exists()


def load(source: str, key: str) -> bytes:
    return blob_path(source, key).read_bytes()


def iter_manifest(source: str) -> Iterator[dict]:
    path = _manifest_path(source)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def store(
    source: str,
    key: str,
    content: bytes,
    fetched_at: datetime,
    meta: dict | None = None,
) -> RawRef:
    if fetched_at.tzinfo is None:
        raise ValueError("fetched_at must be timezone-aware")

    path = blob_path(source, key)
    digest = hashlib.sha256(content).hexdigest()
    already = path.exists()
    if not already:
        path.write_bytes(content)
        entry = {
            "source": source,
            "key": key,
            "sha256": digest,
            "fetched_at": fetched_at.isoformat(),
            "size": len(content),
            "meta": meta or {},
        }
        with _manifest_path(source).open("a") as fh:
            fh.write(json.dumps(entry) + "\n")

    return RawRef(source, key, digest, path, fetched_at, len(content))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_raw_store.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/predictor/raw_store.py tests/test_raw_store.py
git commit -m "feat: content-addressed raw byte archive with manifest"
```

---

### Task 3: News RSS archiver

Started early: RSS is the only source that cannot be recovered retroactively.

**Files:**
- Create: `src/predictor/sources/__init__.py`, `src/predictor/sources/news_rss.py`
- Modify: `src/predictor/cli.py`
- Test: `tests/test_news_rss.py`

**Interfaces:**
- Consumes: `raw_store.store`, `raw_store.exists`.
- Produces:
  - `FEEDS: dict[str, str]` — feed name to URL.
  - `item_key(feed_name: str, entry_id: str) -> str`
  - `archive_entries(feed_name: str, parsed, now: datetime) -> int` — returns count newly archived.
  - `poll_all(now: datetime | None = None) -> dict[str, int]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_news_rss.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_news_rss.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'predictor.sources'`

- [ ] **Step 3: Implement `news_rss.py`**

Create empty `src/predictor/sources/__init__.py`, then:

```python
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import feedparser

from predictor import raw_store

# Verified live 2026-09-22. nba.com/rss/nba_rss.xml returns 403 — excluded.
FEEDS: dict[str, str] = {
    "espn": "https://www.espn.com/espn/rss/nba/news",
    "yahoo": "https://sports.yahoo.com/nba/rss.xml",
    "cbs": "https://www.cbssports.com/rss/headlines/nba/",
}


def item_key(feed_name: str, entry_id: str) -> str:
    digest = hashlib.sha256(entry_id.encode()).hexdigest()[:20]
    return f"{feed_name}_{digest}.json"


def _entry_id(entry) -> str:
    for attr in ("id", "link", "title"):
        value = getattr(entry, attr, None)
        if value:
            return str(value)
    raise ValueError("feed entry has no usable identifier")


def archive_entries(feed_name: str, parsed, now: datetime) -> int:
    stored = 0
    for entry in getattr(parsed, "entries", []):
        entry_id = _entry_id(entry)
        key = item_key(feed_name, entry_id)
        if raw_store.exists("news", key):
            continue
        payload = {
            "feed": feed_name,
            "entry_id": entry_id,
            "title": getattr(entry, "title", ""),
            "link": getattr(entry, "link", ""),
            "summary": getattr(entry, "summary", ""),
            "published": getattr(entry, "published", ""),
            "observed_at": now.isoformat(),
        }
        raw_store.store(
            "news",
            key,
            json.dumps(payload).encode(),
            now,
            meta={"feed": feed_name},
        )
        stored += 1
    return stored


def poll_all(now: datetime | None = None) -> dict[str, int]:
    now = now or datetime.now(UTC)
    results: dict[str, int] = {}
    for name, url in FEEDS.items():
        try:
            parsed = feedparser.parse(url)
            results[name] = archive_entries(name, parsed, now)
        except Exception as exc:  # a dead feed must not stop the others
            results[name] = -1
            print(f"feed {name} failed: {exc}")
    return results
```

The bare `except` here is deliberate and is the one place it is warranted: one publisher breaking must never prevent archiving the others, and the loss is permanent if the run aborts.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_news_rss.py -v`
Expected: 4 passed

- [ ] **Step 5: Add the CLI command**

Append to `src/predictor/cli.py`:

```python
@app.command("poll-news")
def poll_news() -> None:
    """Fetch all configured NBA news feeds and archive new items."""
    from predictor.config import settings
    from predictor.sources import news_rss

    settings.ensure_dirs()
    for feed, count in news_rss.poll_all().items():
        state = "FAILED" if count < 0 else f"{count} new"
        typer.echo(f"{feed}: {state}")
```

- [ ] **Step 6: Verify against the live feeds**

Run: `uv run predictor poll-news`
Expected: each feed reports a count; running it twice reports `0 new` the second time.

- [ ] **Step 7: Commit**

```bash
git add src/predictor/sources tests/test_news_rss.py src/predictor/cli.py
git commit -m "feat: NBA news RSS archiver"
```

---

### Task 4: Scheduled daily runs via launchd

Gets the irreplaceable news archiver collecting immediately, before the rest of the spine exists.

**Files:**
- Create: `scripts/com.predictor.daily.plist`, `scripts/install_schedule.sh`
- Test: manual verification (launchd behaviour is not unit-testable)

**Interfaces:**
- Consumes: the `predictor poll-news` command from Task 3.
- Produces: a loaded launchd agent writing logs to `data/logs/`.

- [ ] **Step 1: Create the plist**

`scripts/com.predictor.daily.plist` — note `PROJECT_DIR` is substituted by the installer:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.predictor.daily</string>
    <key>ProgramArguments</key>
    <array>
        <string>PROJECT_DIR/.venv/bin/predictor</string>
        <string>poll-news</string>
    </array>
    <key>WorkingDirectory</key>
    <string>PROJECT_DIR</string>
    <key>StartCalendarInterval</key>
    <array>
        <dict><key>Hour</key><integer>9</integer><key>Minute</key><integer>0</integer></dict>
        <dict><key>Hour</key><integer>14</integer><key>Minute</key><integer>0</integer></dict>
        <dict><key>Hour</key><integer>19</integer><key>Minute</key><integer>0</integer></dict>
    </array>
    <key>StandardOutPath</key>
    <string>PROJECT_DIR/data/logs/daily.out.log</string>
    <key>StandardErrorPath</key>
    <string>PROJECT_DIR/data/logs/daily.err.log</string>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
```

Three runs a day: news moves continuously, and launchd fires a missed
`StartCalendarInterval` once the machine wakes, so a closed laptop
delays rather than skips collection.

- [ ] **Step 2: Create the installer**

`scripts/install_schedule.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$HOME/Library/LaunchAgents/com.predictor.daily.plist"

mkdir -p "$HOME/Library/LaunchAgents" "$PROJECT_DIR/data/logs"
sed "s|PROJECT_DIR|$PROJECT_DIR|g" \
    "$PROJECT_DIR/scripts/com.predictor.daily.plist" > "$TARGET"

launchctl unload "$TARGET" 2>/dev/null || true
launchctl load "$TARGET"

echo "Installed and loaded: $TARGET"
echo "Verify with: launchctl list | grep predictor"
```

- [ ] **Step 3: Install and verify**

```bash
chmod +x scripts/install_schedule.sh
./scripts/install_schedule.sh
launchctl list | grep predictor
cat data/logs/daily.out.log
```

Expected: the agent appears in `launchctl list`, and because `RunAtLoad` is set, the log shows feed counts within a few seconds.

- [ ] **Step 4: Commit**

```bash
git add scripts
git commit -m "feat: launchd schedule for daily news archiving"
```

---

### Task 5: DuckDB schema and migrations

**Files:**
- Create: `src/predictor/db.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Consumes: `config.settings`.
- Produces:
  - `connect(path: Path | None = None) -> duckdb.DuckDBPyConnection`
  - `migrate(con) -> None` — idempotent.
  - `POINT_IN_TIME_TABLES: frozenset[str]` — tables that must carry `observed_at`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_db.py`:

```python
import pytest

from predictor import db


@pytest.fixture
def con(tmp_path):
    c = db.connect(tmp_path / "t.duckdb")
    db.migrate(c)
    return c


def test_connect_pins_session_timezone_to_utc(tmp_path):
    c = db.connect(tmp_path / "tz.duckdb")
    assert c.execute("SELECT current_setting('TimeZone')").fetchone()[0] == "UTC"


def test_migrate_creates_expected_tables(con):
    names = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert {"games", "injury_status", "odds_snapshots", "news_items"} <= names


def test_migrate_is_idempotent(con):
    db.migrate(con)
    db.migrate(con)
    assert con.execute("SELECT count(*) FROM games").fetchone()[0] == 0


def test_every_point_in_time_table_has_observed_at(con):
    for table in db.POINT_IN_TIME_TABLES:
        cols = {r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()}
        assert "observed_at" in cols, f"{table} missing observed_at"


def test_observed_at_is_timestamptz(con):
    for table in db.POINT_IN_TIME_TABLES:
        rows = con.execute(f"DESCRIBE {table}").fetchall()
        kind = {r[0]: r[1] for r in rows}["observed_at"]
        assert "TIMESTAMP WITH TIME ZONE" in kind, f"{table}.observed_at is {kind}"


def test_timestamps_roundtrip_in_utc_regardless_of_machine_timezone(con):
    """Guards the SET TimeZone='UTC' in connect().

    Without it DuckDB returns values in the local zone, which still compare
    equal but make failures machine-dependent and very hard to read.
    """
    from datetime import UTC, datetime

    moment = datetime(2025, 1, 15, 22, 30, tzinfo=UTC)
    con.execute(
        "INSERT INTO injury_status (report_date, team, player, status, observed_at)"
        " VALUES (?,?,?,?,?)",
        [moment.date(), "LAL", "someone", "Out", moment],
    )
    got = con.execute("SELECT observed_at FROM injury_status").fetchone()[0]
    assert got == moment
    assert got.utcoffset().total_seconds() == 0, f"returned in non-UTC zone: {got}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_db.py -v`
Expected: FAIL — `ImportError: cannot import name 'db'`

- [ ] **Step 3: Implement `db.py`**

```python
from __future__ import annotations

from pathlib import Path

import duckdb

from predictor.config import settings

POINT_IN_TIME_TABLES = frozenset(
    {"games", "injury_status", "odds_snapshots", "news_items"}
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    game_id       VARCHAR NOT NULL,
    season        VARCHAR NOT NULL,
    game_date     DATE NOT NULL,
    home_team     VARCHAR NOT NULL,
    away_team     VARCHAR NOT NULL,
    home_points   INTEGER,
    away_points   INTEGER,
    status        VARCHAR NOT NULL,
    observed_at   TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (game_id, observed_at)
);

CREATE TABLE IF NOT EXISTS injury_status (
    report_date   DATE NOT NULL,
    game_date     DATE,
    matchup       VARCHAR,
    team          VARCHAR NOT NULL,
    player        VARCHAR NOT NULL,
    status        VARCHAR NOT NULL,
    reason        VARCHAR,
    reconstructed BOOLEAN NOT NULL DEFAULT FALSE,
    observed_at   TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (observed_at, team, player)
);

CREATE TABLE IF NOT EXISTS odds_snapshots (
    game_key      VARCHAR NOT NULL,
    book          VARCHAR NOT NULL,
    home_team     VARCHAR NOT NULL,
    away_team     VARCHAR NOT NULL,
    home_price    INTEGER,
    away_price    INTEGER,
    spread        DOUBLE,
    total         DOUBLE,
    observed_at   TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (game_key, book, observed_at)
);

CREATE TABLE IF NOT EXISTS news_items (
    item_key      VARCHAR NOT NULL,
    feed          VARCHAR NOT NULL,
    title         VARCHAR,
    link          VARCHAR,
    summary       VARCHAR,
    observed_at   TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (item_key)
);

CREATE TABLE IF NOT EXISTS ingest_runs (
    source        VARCHAR NOT NULL,
    started_at    TIMESTAMP WITH TIME ZONE NOT NULL,
    finished_at   TIMESTAMP WITH TIME ZONE,
    ok            BOOLEAN,
    detail        VARCHAR
);
"""


def connect(path: Path | None = None) -> duckdb.DuckDBPyConnection:
    target = path or settings.db_path
    target.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(target))
    # Without this, DuckDB returns timestamptz values in the machine's local
    # zone, so identical code yields different-looking results per machine.
    con.execute("SET TimeZone='UTC'")
    return con


def migrate(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(_SCHEMA)
```

`games` is keyed by `(game_id, observed_at)` rather than `game_id` alone: a
game is observed first as scheduled and later as final, and both
observations must coexist for point-in-time queries to be truthful.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_db.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/predictor/db.py tests/test_db.py
git commit -m "feat: DuckDB schema with mandatory observed_at columns"
```

---

### Task 6: As-of accessor and leakage guard

**The most important task in this sub-project.** A leak here silently invalidates every downstream result.

**Files:**
- Create: `src/predictor/asof.py`
- Test: `tests/test_asof.py`, `tests/test_leakage.py`

**Interfaces:**
- Consumes: `db.POINT_IN_TIME_TABLES`.
- Produces:
  - `AsOfError(Exception)`
  - `AsOfView(con, as_of: datetime)` with `.table(name: str) -> duckdb.DuckDBPyRelation` and `.as_of: datetime`

- [ ] **Step 1: Write the failing test**

Create `tests/test_asof.py`:

```python
from datetime import UTC, datetime, timedelta

import pytest

from predictor import db
from predictor.asof import AsOfError, AsOfView

CUTOFF = datetime(2025, 1, 15, 22, 0, tzinfo=UTC)


@pytest.fixture
def con(tmp_path):
    c = db.connect(tmp_path / "t.duckdb")
    db.migrate(c)
    for offset, player in [(-2, "past"), (2, "future")]:
        c.execute(
            "INSERT INTO injury_status "
            "(report_date, team, player, status, observed_at) VALUES (?,?,?,?,?)",
            [
                CUTOFF.date(),
                "LAL",
                player,
                "Out",
                CUTOFF + timedelta(hours=offset),
            ],
        )
    return c


def test_returns_only_rows_observed_at_or_before_cutoff(con):
    view = AsOfView(con, CUTOFF)
    players = {r[0] for r in view.table("injury_status").project("player").fetchall()}
    assert players == {"past"}


def test_boundary_row_exactly_at_cutoff_is_included(con):
    con.execute(
        "INSERT INTO injury_status "
        "(report_date, team, player, status, observed_at) VALUES (?,?,?,?,?)",
        [CUTOFF.date(), "BOS", "boundary", "Out", CUTOFF],
    )
    view = AsOfView(con, CUTOFF)
    players = {r[0] for r in view.table("injury_status").project("player").fetchall()}
    assert "boundary" in players


def test_naive_as_of_is_rejected(con):
    with pytest.raises(AsOfError, match="timezone-aware"):
        AsOfView(con, datetime(2025, 1, 15, 22, 0))


def test_unregistered_table_is_rejected(con):
    view = AsOfView(con, CUTOFF)
    with pytest.raises(AsOfError, match="not a point-in-time table"):
        view.table("ingest_runs")


def test_unknown_table_is_rejected(con):
    view = AsOfView(con, CUTOFF)
    with pytest.raises(AsOfError, match="not a point-in-time table"):
        view.table("games; DROP TABLE games")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_asof.py -v`
Expected: FAIL — `ImportError: cannot import name 'asof'`

- [ ] **Step 3: Implement `asof.py`**

```python
from __future__ import annotations

from datetime import datetime

import duckdb

from predictor.db import POINT_IN_TIME_TABLES


class AsOfError(Exception):
    """Raised when a point-in-time access rule is violated."""


class AsOfView:
    """The only sanctioned way to read point-in-time data.

    Every read is filtered to rows whose ``observed_at`` is at or before
    ``as_of``. Feature code must never query these tables directly.
    """

    def __init__(self, con: duckdb.DuckDBPyConnection, as_of: datetime) -> None:
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise AsOfError("as_of must be timezone-aware")
        self.con = con
        self.as_of = as_of

    def table(self, name: str) -> duckdb.DuckDBPyRelation:
        if name not in POINT_IN_TIME_TABLES:
            raise AsOfError(
                f"{name!r} is not a point-in-time table; "
                f"known tables: {sorted(POINT_IN_TIME_TABLES)}"
            )
        return self.con.sql(
            f"SELECT * FROM {name} WHERE observed_at <= $cutoff",
            params={"cutoff": self.as_of},
        )
```

Table names are validated against a fixed allow-list before interpolation,
so the f-string cannot carry injected SQL; the timestamp is bound as a
parameter.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_asof.py -v`
Expected: 5 passed

- [ ] **Step 5: Write the deliberate-attack leakage tests**

Create `tests/test_leakage.py`. These are permanent regression tests — they
try to break the guard on purpose:

```python
"""Adversarial tests: deliberately attempt to leak future data.

If any test here fails, the backtest's results are meaningless. Treat a
failure as a critical defect, never as a test to relax.
"""

from datetime import UTC, datetime, timedelta

import pytest

from predictor import db
from predictor.asof import AsOfError, AsOfView

TIP_OFF = datetime(2025, 1, 15, 0, 0, tzinfo=UTC)


@pytest.fixture
def con(tmp_path):
    c = db.connect(tmp_path / "t.duckdb")
    db.migrate(c)
    return c


def _insert_game(con, game_id, observed_at, status, home_points, away_points):
    con.execute(
        "INSERT INTO games (game_id, season, game_date, home_team, away_team,"
        " home_points, away_points, status, observed_at) VALUES (?,?,?,?,?,?,?,?,?)",
        [
            game_id,
            "2024-25",
            TIP_OFF.date(),
            "PHI",
            "NYK",
            home_points,
            away_points,
            status,
            observed_at,
        ],
    )


def test_final_score_is_invisible_before_the_game_finishes(con):
    _insert_game(con, "001", TIP_OFF - timedelta(days=1), "SCHEDULED", None, None)
    _insert_game(con, "001", TIP_OFF + timedelta(hours=3), "FINAL", 110, 104)

    view = AsOfView(con, TIP_OFF - timedelta(minutes=30))
    rows = view.table("games").project("status, home_points").fetchall()

    assert rows == [("SCHEDULED", None)]
    assert all(r[1] is None for r in rows), "final score leaked into pre-game view"


def test_injury_report_published_after_cutoff_is_invisible(con):
    for offset, player in [(-1, "early"), (1, "late")]:
        con.execute(
            "INSERT INTO injury_status (report_date, team, player, status, observed_at)"
            " VALUES (?,?,?,?,?)",
            [TIP_OFF.date(), "PHI", player, "Out", TIP_OFF + timedelta(hours=offset)],
        )
    view = AsOfView(con, TIP_OFF)
    players = {r[0] for r in view.table("injury_status").project("player").fetchall()}
    assert players == {"early"}


def test_odds_moved_after_cutoff_are_invisible(con):
    for offset, spread in [(-2, -3.5), (2, -7.5)]:
        con.execute(
            "INSERT INTO odds_snapshots (game_key, book, home_team, away_team,"
            " spread, observed_at) VALUES (?,?,?,?,?,?)",
            ["g1", "bookA", "PHI", "NYK", spread, TIP_OFF + timedelta(hours=offset)],
        )
    view = AsOfView(con, TIP_OFF)
    spreads = [r[0] for r in view.table("odds_snapshots").project("spread").fetchall()]
    assert spreads == [-3.5]


def test_guard_cannot_be_bypassed_with_a_crafted_table_name(con):
    view = AsOfView(con, TIP_OFF)
    for hostile in [
        "games WHERE 1=1 OR observed_at > now()",
        "(SELECT * FROM games)",
        "games--",
        "GAMES",
    ]:
        with pytest.raises(AsOfError):
            view.table(hostile)


def test_every_point_in_time_table_is_reachable_through_the_view(con):
    """A new table added to the schema must not silently bypass the guard."""
    view = AsOfView(con, TIP_OFF)
    for table in db.POINT_IN_TIME_TABLES:
        view.table(table).fetchall()
```

- [ ] **Step 6: Run the leakage tests**

Run: `uv run pytest tests/test_leakage.py -v`
Expected: 5 passed

- [ ] **Step 7: Commit**

```bash
git add src/predictor/asof.py tests/test_asof.py tests/test_leakage.py
git commit -m "feat: point-in-time as-of accessor with adversarial leakage tests"
```

---

### Task 7: Injury report fetcher

**Files:**
- Create: `src/predictor/sources/injury_report.py`
- Test: `tests/test_injury_fetch.py`

**Interfaces:**
- Consumes: `raw_store`.
- Produces:
  - `HOUR_LABELS: tuple[str, ...]` — the 24 hourly slot labels.
  - `report_url(day: date, hour_label: str) -> str`
  - `raw_key(day: date, hour_label: str) -> str`
  - `fetch_report(day, hour_label, session=None) -> bytes | None` — `None` on 403/404.
  - `archive_report(day, hour_label, now=None, session=None) -> bool` — True if newly stored.

- [ ] **Step 1: Write the failing test**

Create `tests/test_injury_fetch.py`:

```python
from datetime import UTC, date, datetime

import pytest

from predictor import raw_store
from predictor.config import Settings
from predictor.sources import injury_report


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(raw_store, "settings", s)
    return s


class FakeResponse:
    def __init__(self, status_code, content=b""):
        self.status_code = status_code
        self.content = content


class FakeSession:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        return self.mapping.get(url, FakeResponse(403))


def test_report_url_matches_verified_pattern():
    url = injury_report.report_url(date(2025, 1, 15), "05PM")
    assert url == (
        "https://ak-static.cms.nba.com/referee/injury/"
        "Injury-Report_2025-01-15_05PM.pdf"
    )


def test_hour_labels_cover_all_24_slots():
    assert len(injury_report.HOUR_LABELS) == 24
    assert "12AM" in injury_report.HOUR_LABELS
    assert "05PM" in injury_report.HOUR_LABELS


def test_fetch_returns_none_on_403():
    session = FakeSession({})
    assert injury_report.fetch_report(date(2018, 12, 11), "05PM", session) is None


def test_fetch_returns_bytes_on_200():
    day, hour = date(2025, 1, 15), "05PM"
    url = injury_report.report_url(day, hour)
    session = FakeSession({url: FakeResponse(200, b"%PDF-1.4 data")})
    assert injury_report.fetch_report(day, hour, session) == b"%PDF-1.4 data"


def test_archive_report_stores_once_and_skips_refetch(store):
    day, hour = date(2025, 1, 15), "05PM"
    url = injury_report.report_url(day, hour)
    session = FakeSession({url: FakeResponse(200, b"%PDF-1.4 data")})

    assert injury_report.archive_report(day, hour, session=session) is True
    assert injury_report.archive_report(day, hour, session=session) is False
    assert len(session.calls) == 1, "already-archived report was re-fetched"


def test_archive_rejects_non_pdf_payload(store):
    day, hour = date(2025, 1, 15), "05PM"
    url = injury_report.report_url(day, hour)
    session = FakeSession({url: FakeResponse(200, b"<html>error</html>")})
    assert injury_report.archive_report(day, hour, session=session) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_injury_fetch.py -v`
Expected: FAIL — `ImportError: cannot import name 'injury_report'`

- [ ] **Step 3: Implement the fetching half of `injury_report.py`**

```python
from __future__ import annotations

from datetime import UTC, date, datetime

import requests

from predictor import raw_store

BASE_URL = "https://ak-static.cms.nba.com/referee/injury"

HOUR_LABELS: tuple[str, ...] = tuple(
    f"{h:02d}{suffix}" for suffix in ("AM", "PM") for h in list(range(1, 13))
)

# Archive verified to begin here; earlier dates return 403.
ARCHIVE_START = date(2019, 12, 1)


def report_url(day: date, hour_label: str) -> str:
    return f"{BASE_URL}/Injury-Report_{day.isoformat()}_{hour_label}.pdf"


def raw_key(day: date, hour_label: str) -> str:
    return f"Injury-Report_{day.isoformat()}_{hour_label}.pdf"


def fetch_report(day: date, hour_label: str, session=None) -> bytes | None:
    session = session or requests.Session()
    response = session.get(report_url(day, hour_label), timeout=30)
    if response.status_code != 200:
        return None
    return response.content


def archive_report(
    day: date,
    hour_label: str,
    now: datetime | None = None,
    session=None,
) -> bool:
    key = raw_key(day, hour_label)
    if raw_store.exists("injury", key):
        return False

    content = fetch_report(day, hour_label, session)
    if content is None or not content.startswith(b"%PDF"):
        return False

    raw_store.store(
        "injury",
        key,
        content,
        now or datetime.now(UTC),
        meta={"day": day.isoformat(), "hour_label": hour_label},
    )
    return True
```

The `%PDF` magic-byte check matters: the CDN occasionally serves an HTML
error page with a 200 status, and archiving that as if it were a report
would poison the backfill silently.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_injury_fetch.py -v`
Expected: 6 passed

- [ ] **Step 5: Verify against the live endpoint**

```bash
uv run python -c "
from datetime import date
from predictor.config import settings
from predictor.sources import injury_report
settings.ensure_dirs()
print('stored:', injury_report.archive_report(date(2025,1,15), '05PM'))
print('again :', injury_report.archive_report(date(2025,1,15), '05PM'))
"
```

Expected: `stored: True` then `again : False`.

- [ ] **Step 6: Commit**

```bash
git add src/predictor/sources/injury_report.py tests/test_injury_fetch.py
git commit -m "feat: NBA injury report fetcher and archiver"
```

---

### Task 8: Injury report PDF parser

Parsing logic below is **prototyped and verified** against the reference fixture — it produced 161 rows across 14 matchups with zero blank fields.

**Files:**
- Modify: `src/predictor/sources/injury_report.py`
- Create: `tests/fixtures/Injury-Report_2025-01-15_05PM.pdf`
- Test: `tests/test_injury_parse.py`

**Interfaces:**
- Consumes: `pdfplumber`.
- Produces:
  - `InjuryRow` dataclass — `game_date: date | None`, `game_time: str`, `matchup: str`, `team: str`, `player: str`, `status: str`, `reason: str`
  - `ParsedReport` dataclass — `published_at: datetime` (UTC), `rows: list[InjuryRow]`
  - `parse_report(pdf_bytes: bytes) -> ParsedReport`
  - `ingest_report(con, pdf_bytes) -> int`

- [ ] **Step 1: Save the fixture**

```bash
mkdir -p tests/fixtures
curl -s -o tests/fixtures/Injury-Report_2025-01-15_05PM.pdf \
  "https://ak-static.cms.nba.com/referee/injury/Injury-Report_2025-01-15_05PM.pdf"
ls -la tests/fixtures/
```

Expected: a file of roughly 89 KB. Commit it — the suite must never depend on the network.

- [ ] **Step 2: Write the failing test**

Create `tests/test_injury_parse.py`:

```python
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from predictor import db
from predictor.sources import injury_report

FIXTURE = Path(__file__).parent / "fixtures" / "Injury-Report_2025-01-15_05PM.pdf"


@pytest.fixture(scope="module")
def parsed():
    return injury_report.parse_report(FIXTURE.read_bytes())


def test_published_at_comes_from_pdf_content_not_filename(parsed):
    # Filename says 05PM; the document header says 05:30 PM Eastern.
    assert parsed.published_at == datetime(2025, 1, 15, 22, 30, tzinfo=UTC)


def test_row_count_matches_verified_baseline(parsed):
    assert len(parsed.rows) == 161


def test_all_fourteen_matchups_are_present(parsed):
    assert len({r.matchup for r in parsed.rows}) == 14


def test_group_columns_are_forward_filled(parsed):
    assert all(r.team for r in parsed.rows)
    assert all(r.matchup for r in parsed.rows)
    assert all(r.game_date == date(2025, 1, 15) for r in parsed.rows)


def test_wrapped_reason_text_is_reassembled(parsed):
    by_player = {r.player: r for r in parsed.rows}
    assert by_player["Brunson,Jalen"].reason == "Injury/Illness-RightShoulder;Soreness"
    assert by_player["Towns,Karl-Anthony"].reason == "Injury/Illness-RightThumb;Sprained"


def test_statuses_are_from_the_known_vocabulary(parsed):
    known = {"Out", "Questionable", "Probable", "Doubtful", "Available"}
    assert {r.status for r in parsed.rows} <= known


def test_pages_after_the_first_are_parsed(parsed):
    # Page 1 alone yields 15 rows; anything near that means later pages were dropped.
    assert len(parsed.rows) > 100


def test_title_line_is_not_emitted_as_a_row(parsed):
    assert not any("InjuryReport" in r.player.replace(" ", "") for r in parsed.rows)


def test_ingest_writes_rows_with_published_at_as_observed_at(tmp_path):
    con = db.connect(tmp_path / "t.duckdb")
    db.migrate(con)
    count = injury_report.ingest_report(con, FIXTURE.read_bytes())
    assert count == 161
    distinct = con.execute("SELECT DISTINCT observed_at FROM injury_status").fetchall()
    assert distinct == [(datetime(2025, 1, 15, 22, 30, tzinfo=UTC),)]


def test_ingest_is_idempotent(tmp_path):
    con = db.connect(tmp_path / "t.duckdb")
    db.migrate(con)
    injury_report.ingest_report(con, FIXTURE.read_bytes())
    injury_report.ingest_report(con, FIXTURE.read_bytes())
    total = con.execute("SELECT count(*) FROM injury_status").fetchone()[0]
    assert total == 161
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_injury_parse.py -v`
Expected: FAIL — `AttributeError: module has no attribute 'parse_report'`

- [ ] **Step 4: Implement the parser**

Append to `src/predictor/sources/injury_report.py`:

```python
import io
import re
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import pdfplumber

EASTERN = ZoneInfo("America/New_York")

COLS = ["GameDate", "GameTime", "Matchup", "Team", "PlayerName", "CurrentStatus", "Reason"]
GAME_DATE, GAME_TIME, MATCHUP, TEAM, PLAYER, STATUS, REASON = range(7)

_STAMP = re.compile(r"(\d{2}/\d{2}/\d{2})\s+(\d{1,2}:\d{2})\s*(AM|PM)")


@dataclass(frozen=True)
class InjuryRow:
    game_date: date | None
    game_time: str
    matchup: str
    team: str
    player: str
    status: str
    reason: str


@dataclass(frozen=True)
class ParsedReport:
    published_at: datetime
    rows: list[InjuryRow]


def _column_bounds(page) -> list[float] | None:
    header: dict[str, float] = {}
    for word in page.extract_words():
        if word["text"] in COLS and word["text"] not in header:
            header[word["text"]] = word["x0"]
    return [header[c] for c in COLS] if len(header) == len(COLS) else None


def _column_index(x0: float, bounds: list[float]) -> int:
    index = 0
    for i, edge in enumerate(bounds):
        if x0 >= edge - 2:
            index = i
    return index


def _lines(page, bounds):
    groups: dict[int, list] = {}
    for word in page.extract_words():
        if word["text"] in COLS:
            continue
        groups.setdefault(round(word["top"] / 3), []).append(word)

    out = []
    for key in sorted(groups):
        words = groups[key]
        cells = [""] * len(COLS)
        for word in sorted(words, key=lambda w: w["x0"]):
            i = _column_index(word["x0"], bounds)
            cells[i] = (cells[i] + " " + word["text"]).strip()
        if "InjuryReport:" in "".join(cells).replace(" ", ""):
            continue
        out.append((min(w["top"] for w in words), cells))
    return out


def _parse_page(page, bounds, carry):
    lines = _lines(page, bounds)
    anchors = [(t, c) for t, c in lines if c[PLAYER] and c[STATUS]]
    fragments = [
        (t, c[REASON]) for t, c in lines if not (c[PLAYER] and c[STATUS]) and c[REASON]
    ]

    rows = [
        {"top": t, "cells": c, "pieces": [(t, c[REASON])] if c[REASON] else []}
        for t, c in anchors
    ]

    # A wrapped Reason can sit above or below its player line, so each
    # fragment attaches to the vertically nearest anchor.
    for top, text in fragments:
        if rows:
            nearest = min(rows, key=lambda r: abs(r["top"] - top))
            nearest["pieces"].append((top, text))

    parsed = []
    for row in rows:
        cells = row["cells"]
        for i in (GAME_DATE, GAME_TIME, MATCHUP, TEAM):
            if cells[i]:
                carry[i] = cells[i]
            else:
                cells[i] = carry.get(i, "")
        reason = "".join(text for _, text in sorted(row["pieces"]))
        parsed.append(
            InjuryRow(
                game_date=_to_date(cells[GAME_DATE]),
                game_time=cells[GAME_TIME],
                matchup=cells[MATCHUP],
                team=cells[TEAM],
                player=cells[PLAYER],
                status=cells[STATUS],
                reason=reason,
            )
        )
    return parsed, carry


def _to_date(text: str) -> date | None:
    try:
        return datetime.strptime(text, "%m/%d/%Y").date()
    except ValueError:
        return None


def _published_at(first_line: str) -> datetime:
    match = _STAMP.search(first_line)
    if not match:
        raise ValueError(f"no publication timestamp in header: {first_line!r}")
    day, clock, meridiem = match.groups()
    naive = datetime.strptime(f"{day} {clock} {meridiem}", "%m/%d/%y %I:%M %p")
    return naive.replace(tzinfo=EASTERN).astimezone(UTC)


def parse_report(pdf_bytes: bytes) -> ParsedReport:
    rows: list[InjuryRow] = []
    carry: dict[int, str] = {}
    bounds = None

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        published_at = _published_at(pdf.pages[0].extract_text().split("\n")[0])
        for page in pdf.pages:
            # Only page 1 carries the header; reuse its bounds thereafter.
            bounds = _column_bounds(page) or bounds
            if bounds is None:
                continue
            page_rows, carry = _parse_page(page, bounds, carry)
            rows.extend(page_rows)

    return ParsedReport(published_at=published_at, rows=rows)


def ingest_report(con, pdf_bytes: bytes) -> int:
    report = parse_report(pdf_bytes)
    for row in report.rows:
        con.execute(
            "INSERT OR REPLACE INTO injury_status (report_date, game_date, matchup,"
            " team, player, status, reason, reconstructed, observed_at)"
            " VALUES (?,?,?,?,?,?,?,FALSE,?)",
            [
                report.published_at.date(),
                row.game_date,
                row.matchup,
                row.team,
                row.player,
                row.status,
                row.reason,
                report.published_at,
            ],
        )
    return len(report.rows)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_injury_parse.py -v`
Expected: 10 passed

- [ ] **Step 6: Commit**

```bash
git add src/predictor/sources/injury_report.py tests/test_injury_parse.py tests/fixtures
git commit -m "feat: injury report PDF parser with point-in-time timestamps"
```

---

### Task 9: Injury backfill command

Sweeps the archive from 2019-12 to present. Resumable and rate-limited.

**Files:**
- Modify: `src/predictor/sources/injury_report.py`, `src/predictor/cli.py`
- Test: `tests/test_injury_backfill.py`

**Interfaces:**
- Consumes: `archive_report`, `ingest_report`, `raw_store`.
- Produces:
  - `backfill_range(con, start: date, end: date, hours: Sequence[str], delay: float = 0.4, session=None) -> dict[str, int]` — keys `fetched`, `skipped`, `missing`, `ingested`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_injury_backfill.py`:

```python
from datetime import date

import pytest

from predictor import db, raw_store
from predictor.config import Settings
from predictor.sources import injury_report

FIXTURE_BYTES = (
    __import__("pathlib").Path(__file__).parent
    / "fixtures"
    / "Injury-Report_2025-01-15_05PM.pdf"
).read_bytes()


@pytest.fixture
def env(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(raw_store, "settings", s)
    con = db.connect(tmp_path / "t.duckdb")
    db.migrate(con)
    return con


class FakeResponse:
    def __init__(self, status_code, content=b""):
        self.status_code = status_code
        self.content = content


class FakeSession:
    def __init__(self, available):
        self.available = available
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        if url in self.available:
            return FakeResponse(200, FIXTURE_BYTES)
        return FakeResponse(403)


def test_backfill_counts_missing_slots(env):
    session = FakeSession(available=set())
    stats = injury_report.backfill_range(
        env, date(2025, 1, 15), date(2025, 1, 15), ["05PM"], delay=0, session=session
    )
    assert stats["missing"] == 1
    assert stats["fetched"] == 0


def test_backfill_fetches_and_ingests_available_slots(env):
    url = injury_report.report_url(date(2025, 1, 15), "05PM")
    session = FakeSession(available={url})
    stats = injury_report.backfill_range(
        env, date(2025, 1, 15), date(2025, 1, 15), ["05PM"], delay=0, session=session
    )
    assert stats["fetched"] == 1
    assert stats["ingested"] == 161


def test_backfill_is_resumable_and_skips_archived_days(env):
    url = injury_report.report_url(date(2025, 1, 15), "05PM")
    session = FakeSession(available={url})
    args = (env, date(2025, 1, 15), date(2025, 1, 15), ["05PM"])
    injury_report.backfill_range(*args, delay=0, session=session)
    stats = injury_report.backfill_range(*args, delay=0, session=session)
    assert stats["skipped"] == 1
    assert len(session.calls) == 1, "resumed run re-fetched an archived report"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_injury_backfill.py -v`
Expected: FAIL — `AttributeError: module has no attribute 'backfill_range'`

- [ ] **Step 3: Implement `backfill_range`**

Append to `src/predictor/sources/injury_report.py`:

```python
import time
from collections.abc import Sequence
from datetime import timedelta


def backfill_range(
    con,
    start: date,
    end: date,
    hours: Sequence[str] = ("05PM",),
    delay: float = 0.4,
    session=None,
) -> dict[str, int]:
    session = session or requests.Session()
    stats = {"fetched": 0, "skipped": 0, "missing": 0, "ingested": 0}

    day = start
    while day <= end:
        for hour in hours:
            key = raw_key(day, hour)
            if raw_store.exists("injury", key):
                stats["skipped"] += 1
                continue
            if archive_report(day, hour, session=session):
                stats["fetched"] += 1
                stats["ingested"] += ingest_report(con, raw_store.load("injury", key))
            else:
                stats["missing"] += 1
            if delay:
                time.sleep(delay)
        day += timedelta(days=1)

    return stats
```

Default `hours=("05PM",)` keeps the first backfill to roughly 2,500
requests rather than 60,000. The 5pm Eastern report is the last one
published before most tip-offs, making it the single most useful slot.
Additional hours can be swept later for finer resolution.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_injury_backfill.py -v`
Expected: 3 passed

- [ ] **Step 5: Add the CLI command**

Append to `src/predictor/cli.py`:

```python
@app.command("backfill-injuries")
def backfill_injuries(
    start: str = typer.Option("2019-12-01", help="ISO start date."),
    end: str = typer.Option(None, help="ISO end date; defaults to today."),
    hours: str = typer.Option("05PM", help="Comma-separated hour slots."),
) -> None:
    """Backfill archived NBA injury reports into the store."""
    from datetime import date as _date

    from predictor import db
    from predictor.config import settings
    from predictor.sources import injury_report

    settings.ensure_dirs()
    con = db.connect()
    db.migrate(con)

    stats = injury_report.backfill_range(
        con,
        _date.fromisoformat(start),
        _date.fromisoformat(end) if end else _date.today(),
        [h.strip() for h in hours.split(",") if h.strip()],
    )
    for key, value in stats.items():
        typer.echo(f"{key}: {value}")
```

- [ ] **Step 6: Verify on a small live range**

```bash
uv run predictor backfill-injuries --start 2025-01-14 --end 2025-01-16
```

Expected: `fetched: 3`, `ingested` in the hundreds, `missing: 0`. Re-running reports `skipped: 3`.

- [ ] **Step 7: Commit**

```bash
git add src/predictor/sources/injury_report.py tests/test_injury_backfill.py src/predictor/cli.py
git commit -m "feat: resumable injury report backfill from 2019"
```

---

### Task 10: NBA game ingestion

**Files:**
- Create: `src/predictor/sources/nba_stats.py`
- Modify: `src/predictor/cli.py`
- Test: `tests/test_nba_stats.py`

**Interfaces:**
- Consumes: `nba_api`, `raw_store`, `db`.
- Produces:
  - `GameRow` dataclass — `game_id, season, game_date, home_team, away_team, home_points, away_points, status`
  - `pair_team_rows(df) -> list[GameRow]` — collapses `LeagueGameFinder`'s two-rows-per-game into one.
  - `fetch_season(season: str) -> pandas.DataFrame`
  - `ingest_season(con, season: str, observed_at: datetime | None = None) -> int`

- [ ] **Step 1: Write the failing test**

Create `tests/test_nba_stats.py`:

```python
from datetime import UTC, date, datetime

import pandas as pd
import pytest

from predictor import db
from predictor.sources import nba_stats

OBSERVED = datetime(2025, 6, 23, 0, 0, tzinfo=UTC)


def _frame(rows):
    return pd.DataFrame(
        rows,
        columns=["GAME_ID", "GAME_DATE", "MATCHUP", "WL", "PTS", "TEAM_ABBREVIATION"],
    )


def test_pair_team_rows_collapses_two_rows_into_one_game():
    df = _frame(
        [
            ["0042400407", "2025-06-22", "IND @ OKC", "L", 91, "IND"],
            ["0042400407", "2025-06-22", "OKC vs. IND", "W", 103, "OKC"],
        ]
    )
    games = nba_stats.pair_team_rows(df)
    assert len(games) == 1
    game = games[0]
    assert game.home_team == "OKC"
    assert game.away_team == "IND"
    assert game.home_points == 103
    assert game.away_points == 91
    assert game.game_date == date(2025, 6, 22)
    assert game.status == "FINAL"


def test_unplayed_game_is_marked_scheduled_with_no_score():
    df = _frame(
        [
            ["0022500001", "2026-10-21", "LAL @ GSW", None, None, "LAL"],
            ["0022500001", "2026-10-21", "GSW vs. LAL", None, None, "GSW"],
        ]
    )
    game = nba_stats.pair_team_rows(df)[0]
    assert game.status == "SCHEDULED"
    assert game.home_points is None


def test_unpaired_row_is_dropped_rather_than_guessed():
    df = _frame([["0042400407", "2025-06-22", "IND @ OKC", "L", 91, "IND"]])
    assert nba_stats.pair_team_rows(df) == []


def test_ingest_writes_rows_with_supplied_observed_at(tmp_path, monkeypatch):
    con = db.connect(tmp_path / "t.duckdb")
    db.migrate(con)
    df = _frame(
        [
            ["0042400407", "2025-06-22", "IND @ OKC", "L", 91, "IND"],
            ["0042400407", "2025-06-22", "OKC vs. IND", "W", 103, "OKC"],
        ]
    )
    monkeypatch.setattr(nba_stats, "fetch_season", lambda season: df)

    count = nba_stats.ingest_season(con, "2024-25", observed_at=OBSERVED)
    assert count == 1
    row = con.execute(
        "SELECT home_team, away_team, home_points, observed_at FROM games"
    ).fetchone()
    assert row == ("OKC", "IND", 103, OBSERVED)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_nba_stats.py -v`
Expected: FAIL — `ImportError: cannot import name 'nba_stats'`

- [ ] **Step 3: Implement `nba_stats.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass(frozen=True)
class GameRow:
    game_id: str
    season: str
    game_date: date
    home_team: str
    away_team: str
    home_points: int | None
    away_points: int | None
    status: str


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=30))
def fetch_season(season: str) -> pd.DataFrame:
    """Fetch every team-game row for a season. Two rows per game."""
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season, league_id_nullable="00", timeout=60
    )
    return finder.get_data_frames()[0]


def _as_int(value) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def pair_team_rows(df: pd.DataFrame, season: str = "") -> list[GameRow]:
    games: list[GameRow] = []
    for game_id, group in df.groupby("GAME_ID"):
        if len(group) != 2:
            continue  # unpaired row: drop rather than invent an opponent

        # "OKC vs. IND" is the home listing; "IND @ OKC" is the away listing.
        home_mask = group["MATCHUP"].str.contains("vs.", regex=False)
        if home_mask.sum() != 1:
            continue
        home = group[home_mask].iloc[0]
        away = group[~home_mask].iloc[0]

        home_points = _as_int(home["PTS"])
        away_points = _as_int(away["PTS"])
        played = home_points is not None and away_points is not None

        games.append(
            GameRow(
                game_id=str(game_id),
                season=season,
                game_date=pd.to_datetime(home["GAME_DATE"]).date(),
                home_team=str(home["TEAM_ABBREVIATION"]),
                away_team=str(away["TEAM_ABBREVIATION"]),
                home_points=home_points,
                away_points=away_points,
                status="FINAL" if played else "SCHEDULED",
            )
        )
    return games


def ingest_season(con, season: str, observed_at: datetime | None = None) -> int:
    observed_at = observed_at or datetime.now(UTC)
    games = pair_team_rows(fetch_season(season), season)
    for game in games:
        con.execute(
            "INSERT OR REPLACE INTO games (game_id, season, game_date, home_team,"
            " away_team, home_points, away_points, status, observed_at)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            [
                game.game_id,
                season,
                game.game_date,
                game.home_team,
                game.away_team,
                game.home_points,
                game.away_points,
                game.status,
                observed_at,
            ],
        )
    return len(games)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_nba_stats.py -v`
Expected: 4 passed

- [ ] **Step 5: Add the CLI command and verify live**

Append to `src/predictor/cli.py`:

```python
@app.command("ingest-season")
def ingest_season_cmd(season: str = typer.Argument(..., help="e.g. 2024-25")) -> None:
    """Ingest all games for one season."""
    from predictor import db
    from predictor.config import settings
    from predictor.sources import nba_stats

    settings.ensure_dirs()
    con = db.connect()
    db.migrate(con)
    typer.echo(f"ingested {nba_stats.ingest_season(con, season)} games for {season}")
```

Run: `uv run predictor ingest-season 2024-25`
Expected: roughly 1,400 games (2,802 team-rows collapsed into game-rows).

- [ ] **Step 6: Commit**

```bash
git add src/predictor/sources/nba_stats.py tests/test_nba_stats.py src/predictor/cli.py
git commit -m "feat: NBA season game ingestion via nba_api"
```

---

### Task 11: Odds ingestion

**Files:**
- Create: `src/predictor/sources/odds.py`
- Modify: `src/predictor/cli.py`
- Test: `tests/test_odds.py`

**Interfaces:**
- Consumes: `raw_store`, `db`.
- Produces:
  - `OddsQuotaExceeded(Exception)`
  - `parse_odds_payload(payload: list[dict], observed_at: datetime) -> list[dict]`
  - `fetch_current(api_key: str, session=None) -> list[dict]`
  - `ingest_current(con, api_key: str | None = None, now=None, session=None) -> int`

- [ ] **Step 1: Write the failing test**

Create `tests/test_odds.py`:

```python
from datetime import UTC, datetime

import pytest

from predictor import db
from predictor.sources import odds

NOW = datetime(2025, 1, 15, 20, 0, tzinfo=UTC)

PAYLOAD = [
    {
        "id": "abc123",
        "home_team": "Philadelphia 76ers",
        "away_team": "New York Knicks",
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Philadelphia 76ers", "price": -150},
                            {"name": "New York Knicks", "price": 130},
                        ],
                    }
                ],
            }
        ],
    }
]


def test_parse_extracts_one_row_per_bookmaker():
    rows = odds.parse_odds_payload(PAYLOAD, NOW)
    assert len(rows) == 1
    row = rows[0]
    assert row["book"] == "draftkings"
    assert row["home_price"] == -150
    assert row["away_price"] == 130
    assert row["observed_at"] == NOW


def test_parse_skips_events_without_bookmakers():
    assert odds.parse_odds_payload([{"id": "x", "home_team": "A", "away_team": "B"}], NOW) == []


def test_ingest_writes_rows(tmp_path, monkeypatch):
    con = db.connect(tmp_path / "t.duckdb")
    db.migrate(con)
    monkeypatch.setattr(odds, "fetch_current", lambda key, session=None: PAYLOAD)
    assert odds.ingest_current(con, api_key="k", now=NOW) == 1
    stored = con.execute("SELECT book, home_price FROM odds_snapshots").fetchone()
    assert stored == ("draftkings", -150)


def test_missing_api_key_raises_a_clear_error(tmp_path, monkeypatch):
    con = db.connect(tmp_path / "t.duckdb")
    db.migrate(con)
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ODDS_API_KEY"):
        odds.ingest_current(con)


class FakeResponse:
    status_code = 401
    text = "quota"

    def json(self):
        return {}


def test_quota_exhaustion_raises_named_error():
    session = type("S", (), {"get": lambda self, *a, **k: FakeResponse()})()
    with pytest.raises(odds.OddsQuotaExceeded):
        odds.fetch_current("k", session=session)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_odds.py -v`
Expected: FAIL — `ImportError: cannot import name 'odds'`

- [ ] **Step 3: Implement `odds.py`**

```python
from __future__ import annotations

import json
import os
from datetime import UTC, datetime

import requests

from predictor import raw_store

API_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"


class OddsQuotaExceeded(Exception):
    """The Odds API rejected the request for quota or auth reasons."""


def fetch_current(api_key: str, session=None) -> list[dict]:
    session = session or requests.Session()
    response = session.get(
        API_URL,
        params={
            "apiKey": api_key,
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american",
        },
        timeout=30,
    )
    if response.status_code in (401, 429):
        raise OddsQuotaExceeded(f"{response.status_code}: {response.text[:200]}")
    response.raise_for_status()
    return response.json()


def parse_odds_payload(payload: list[dict], observed_at: datetime) -> list[dict]:
    rows: list[dict] = []
    for event in payload:
        home, away = event.get("home_team"), event.get("away_team")
        for book in event.get("bookmakers") or []:
            row = {
                "game_key": event["id"],
                "book": book["key"],
                "home_team": home,
                "away_team": away,
                "home_price": None,
                "away_price": None,
                "spread": None,
                "total": None,
                "observed_at": observed_at,
            }
            for market in book.get("markets") or []:
                outcomes = {o["name"]: o for o in market.get("outcomes") or []}
                if market["key"] == "h2h":
                    row["home_price"] = outcomes.get(home, {}).get("price")
                    row["away_price"] = outcomes.get(away, {}).get("price")
                elif market["key"] == "spreads":
                    row["spread"] = outcomes.get(home, {}).get("point")
                elif market["key"] == "totals":
                    row["total"] = next(
                        (o.get("point") for o in outcomes.values()), None
                    )
            rows.append(row)
    return rows


def ingest_current(con, api_key: str | None = None, now=None, session=None) -> int:
    api_key = api_key or os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise ValueError(
            "ODDS_API_KEY is not set. Get a free key at https://the-odds-api.com "
            "and export it before running this command."
        )
    now = now or datetime.now(UTC)
    payload = fetch_current(api_key, session)

    raw_store.store(
        "odds", f"odds_{now.strftime('%Y%m%dT%H%M%S')}.json",
        json.dumps(payload).encode(), now,
    )

    rows = parse_odds_payload(payload, now)
    for row in rows:
        con.execute(
            "INSERT OR REPLACE INTO odds_snapshots (game_key, book, home_team,"
            " away_team, home_price, away_price, spread, total, observed_at)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            [
                row["game_key"], row["book"], row["home_team"], row["away_team"],
                row["home_price"], row["away_price"], row["spread"], row["total"],
                row["observed_at"],
            ],
        )
    return len(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_odds.py -v`
Expected: 5 passed

- [ ] **Step 5: Add the CLI command**

Append to `src/predictor/cli.py`:

```python
@app.command("ingest-odds")
def ingest_odds_cmd() -> None:
    """Fetch and store one odds snapshot. Budgeted to one call per run."""
    from predictor import db
    from predictor.config import settings
    from predictor.sources import odds

    settings.ensure_dirs()
    con = db.connect()
    db.migrate(con)
    try:
        typer.echo(f"stored {odds.ingest_current(con)} odds rows")
    except odds.OddsQuotaExceeded as exc:
        typer.echo(f"ODDS QUOTA EXCEEDED: {exc}", err=True)
        raise typer.Exit(code=1)
```

- [ ] **Step 6: Commit**

```bash
git add src/predictor/sources/odds.py tests/test_odds.py src/predictor/cli.py
git commit -m "feat: odds snapshot ingestion with quota handling"
```

---

### Task 12: News archive to database ingestion

Task 3 archives news to raw storage only. This task loads those archived items into `news_items` so the table registered in `POINT_IN_TIME_TABLES` is actually populated and queryable through `AsOfView`.

**Files:**
- Modify: `src/predictor/sources/news_rss.py`, `src/predictor/cli.py`
- Test: `tests/test_news_ingest.py`

**Interfaces:**
- Consumes: `raw_store.iter_manifest`, `raw_store.load`, `db`.
- Produces: `ingest_archived_news(con) -> int` — number of rows written.

- [ ] **Step 1: Write the failing test**

Create `tests/test_news_ingest.py`:

```python
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from predictor import db, raw_store
from predictor.config import Settings
from predictor.sources import news_rss

NOW = datetime(2026, 1, 2, 3, 4, tzinfo=UTC)


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


def test_ingest_loads_archived_items_into_the_table(env):
    news_rss.archive_entries("espn", _parsed(["a", "b"]), NOW)
    assert news_rss.ingest_archived_news(env) == 2
    rows = env.execute("SELECT feed, title, observed_at FROM news_items").fetchall()
    assert len(rows) == 2
    assert rows[0][0] == "espn"
    assert rows[0][2] == NOW


def test_ingest_is_idempotent(env):
    news_rss.archive_entries("espn", _parsed(["a"]), NOW)
    news_rss.ingest_archived_news(env)
    news_rss.ingest_archived_news(env)
    assert env.execute("SELECT count(*) FROM news_items").fetchone()[0] == 1


def test_ingest_with_no_archive_returns_zero(env):
    assert news_rss.ingest_archived_news(env) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_news_ingest.py -v`
Expected: FAIL — `AttributeError: module has no attribute 'ingest_archived_news'`

- [ ] **Step 3: Implement `ingest_archived_news`**

Append to `src/predictor/sources/news_rss.py`:

```python
def ingest_archived_news(con) -> int:
    """Load archived news JSON from the raw store into news_items."""
    written = 0
    for entry in raw_store.iter_manifest("news"):
        key = entry["key"]
        payload = json.loads(raw_store.load("news", key))
        con.execute(
            "INSERT OR REPLACE INTO news_items"
            " (item_key, feed, title, link, summary, observed_at)"
            " VALUES (?,?,?,?,?,?)",
            [
                key,
                payload.get("feed", ""),
                payload.get("title", ""),
                payload.get("link", ""),
                payload.get("summary", ""),
                datetime.fromisoformat(payload["observed_at"]),
            ],
        )
        written += 1
    return written
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_news_ingest.py -v`
Expected: 3 passed

- [ ] **Step 5: Wire it into the poll command**

In `src/predictor/cli.py`, replace the body of `poll_news` with:

```python
@app.command("poll-news")
def poll_news() -> None:
    """Fetch all configured NBA news feeds, archive new items, and load them."""
    from predictor import db
    from predictor.config import settings
    from predictor.sources import news_rss

    settings.ensure_dirs()
    for feed, count in news_rss.poll_all().items():
        state = "FAILED" if count < 0 else f"{count} new"
        typer.echo(f"{feed}: {state}")

    con = db.connect()
    db.migrate(con)
    typer.echo(f"news_items rows: {news_rss.ingest_archived_news(con)}")
```

- [ ] **Step 6: Verify end to end**

Run: `uv run predictor poll-news`
Expected: feed counts, then a non-zero `news_items rows` line.

- [ ] **Step 7: Commit**

```bash
git add src/predictor/sources/news_rss.py tests/test_news_ingest.py src/predictor/cli.py
git commit -m "feat: load archived news into news_items table"
```

---

### Task 13: Status command and data-health reporting

Implements the spec's governing rule — failures surface in plain English where they will actually be seen.

**Files:**
- Create: `src/predictor/status.py`
- Modify: `src/predictor/cli.py`
- Test: `tests/test_status.py`

**Interfaces:**
- Consumes: `db`, `config.settings`.
- Produces:
  - `SourceHealth` dataclass — `name: str`, `latest: datetime | None`, `row_count: int`, `age_hours: float | None`, `stale: bool`, `advice: str`
  - `check_sources(con, now: datetime | None = None) -> list[SourceHealth]`
  - `format_report(health: list[SourceHealth]) -> str`
  - `STALENESS_HOURS: dict[str, float]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_status.py`:

```python
from datetime import UTC, datetime, timedelta

import pytest

from predictor import db, status

NOW = datetime(2025, 1, 15, 20, 0, tzinfo=UTC)


@pytest.fixture
def con(tmp_path):
    c = db.connect(tmp_path / "t.duckdb")
    db.migrate(c)
    return c


def _add_injury(con, observed_at, player="x"):
    con.execute(
        "INSERT INTO injury_status (report_date, team, player, status, observed_at)"
        " VALUES (?,?,?,?,?)",
        [observed_at.date(), "LAL", player, "Out", observed_at],
    )


def test_empty_source_is_reported_stale_with_advice(con):
    health = {h.name: h for h in status.check_sources(con, NOW)}
    injuries = health["injury_status"]
    assert injuries.row_count == 0
    assert injuries.stale is True
    assert injuries.advice


def test_fresh_source_is_not_stale(con):
    _add_injury(con, NOW - timedelta(hours=1))
    health = {h.name: h for h in status.check_sources(con, NOW)}
    assert health["injury_status"].stale is False


def test_old_source_is_flagged_stale(con):
    _add_injury(con, NOW - timedelta(days=5))
    health = {h.name: h for h in status.check_sources(con, NOW)}
    injuries = health["injury_status"]
    assert injuries.stale is True
    assert injuries.age_hours == pytest.approx(120, abs=1)


def test_report_is_plain_english_and_names_problem_sources(con):
    _add_injury(con, NOW - timedelta(days=5))
    text = status.format_report(status.check_sources(con, NOW))
    assert "injury_status" in text
    assert "STALE" in text
    assert "OK" in text or "stale" in text.lower()


def test_report_leads_with_overall_verdict(con):
    text = status.format_report(status.check_sources(con, NOW))
    assert text.splitlines()[0].startswith(("PROBLEMS", "ALL OK"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_status.py -v`
Expected: FAIL — `ImportError: cannot import name 'status'`

- [ ] **Step 3: Implement `status.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from predictor.db import POINT_IN_TIME_TABLES

STALENESS_HOURS: dict[str, float] = {
    "games": 48,
    "injury_status": 36,
    "odds_snapshots": 36,
    "news_items": 24,
}

ADVICE: dict[str, str] = {
    "games": "Run: predictor ingest-season 2026-27",
    "injury_status": "Run: predictor backfill-injuries --start <recent date>",
    "odds_snapshots": "Run: predictor ingest-odds (needs ODDS_API_KEY)",
    "news_items": "Run: predictor poll-news, and check the launchd agent is loaded",
}


@dataclass(frozen=True)
class SourceHealth:
    name: str
    latest: datetime | None
    row_count: int
    age_hours: float | None
    stale: bool
    advice: str


def check_sources(con, now: datetime | None = None) -> list[SourceHealth]:
    now = now or datetime.now(UTC)
    out: list[SourceHealth] = []

    for table in sorted(POINT_IN_TIME_TABLES):
        count, latest = con.execute(
            f"SELECT count(*), max(observed_at) FROM {table}"
        ).fetchone()

        if latest is None:
            out.append(
                SourceHealth(table, None, 0, None, True, ADVICE.get(table, ""))
            )
            continue

        age = (now - latest).total_seconds() / 3600
        stale = age > STALENESS_HOURS.get(table, 48)
        out.append(
            SourceHealth(
                table, latest, count, age, stale, ADVICE.get(table, "") if stale else ""
            )
        )

    return out


def format_report(health: list[SourceHealth]) -> str:
    problems = [h for h in health if h.stale]
    lines = [
        f"PROBLEMS: {len(problems)} of {len(health)} sources need attention"
        if problems
        else "ALL OK: every data source is fresh"
    ]
    lines.append("")

    for h in health:
        if h.latest is None:
            lines.append(f"  [STALE] {h.name}: no data at all")
        else:
            mark = "STALE" if h.stale else "OK"
            lines.append(
                f"  [{mark}] {h.name}: {h.row_count:,} rows, "
                f"newest {h.age_hours:.0f}h old"
            )
        if h.advice:
            lines.append(f"          -> {h.advice}")

    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_status.py -v`
Expected: 5 passed

- [ ] **Step 5: Add the CLI command**

Append to `src/predictor/cli.py`:

```python
@app.command()
def status() -> None:
    """Report data freshness in plain English."""
    from predictor import db
    from predictor import status as status_mod
    from predictor.config import settings

    settings.ensure_dirs()
    con = db.connect()
    db.migrate(con)
    typer.echo(status_mod.format_report(status_mod.check_sources(con)))


@app.command()
def setup() -> None:
    """Create directories and initialise the database."""
    from predictor import db
    from predictor.config import settings

    settings.ensure_dirs()
    db.migrate(db.connect())
    typer.echo(f"ready. data dir: {settings.data_dir}")
```

- [ ] **Step 6: Run the full suite and the command**

```bash
uv run pytest -v
uv run predictor status
```

Expected: all tests pass; status prints a verdict line plus one line per source.

- [ ] **Step 7: Commit**

```bash
git add src/predictor/status.py tests/test_status.py src/predictor/cli.py
git commit -m "feat: plain-English data health reporting"
```

---

## Definition of done

The data spine is complete when all of the following hold:

- [ ] `uv run pytest -v` passes, including every test in `tests/test_leakage.py`.
- [ ] `uv run predictor status` reports all four sources present.
- [ ] The launchd agent is loaded and `data/logs/daily.out.log` shows recent news polls.
- [ ] Injury reports are backfilled from 2019-12 to the present at the 5pm slot.
- [ ] At least the 2019-20 through 2025-26 seasons are ingested into `games`.
- [ ] No feature-style code reads a point-in-time table except through `AsOfView`.

## What this deliberately does not include

Deferred to later sub-projects, per the spec's decomposition:

- Basketball-Reference scraping and advanced stats (needed by the prediction core, not the spine).
- Historical odds datasets — the live odds path is built here; bulk historical odds loading belongs with the backtest work that consumes it.
- Play-by-play ingestion (needed for post-game key moments, sub-project 4).
- Any modelling, feature engineering, or backtesting.
