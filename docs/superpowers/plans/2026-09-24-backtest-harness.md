# Backtest Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a walk-forward replay engine that scores any prediction function against real NBA history, using only what was knowable before each tip-off, and reports calibration and accuracy honestly enough to publish.

**Architecture:** A model-agnostic harness. It resolves each game's true tip-off, constructs an `AsOfView` at tip-off minus a buffer, hands that view to a caller-supplied predictor, and records the result. Metrics are computed from the recorded predictions, never from the model. The harness ships with trivial baselines so it can be validated *before* any model exists — and with adversarial tests that try to cheat through it.

**Tech Stack:** Python 3.14, `uv`, DuckDB, the existing `AsOfView` accessor. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-22-nba-predictor-design.md` (section 2, "Backtest harness")

## Global Constraints

- **Python 3.14** via `uv run`. No new dependencies.
- **All timestamps UTC and timezone-aware.** Naive datetimes are a hard error, never silently coerced.
- **`observed_at` records when a fact became KNOWABLE**, not when the event occurred.
- **Every read of point-in-time data goes through `AsOfView`.** A repo-wide test forbids naming a physical `_raw` table outside `db.py`/`asof.py`; resolve via `db.POINT_IN_TIME_TABLES`.
- **Never silently lose or misreport data — surface loudly.** Missing or stale inputs must be visible, not quietly degraded.
- **The end user does not write code.** CLI output reads in plain English.
- **Free data sources only**; no paid APIs; no LLM calls anywhere in the pipeline.
- **Do NOT delete, clear or overwrite anything under `data/`** — it holds 1,468 archived injury PDFs and the news archive, neither re-fetchable.

## Verified ground truth

Measured against the real database on 2026-09-24. Trust these over assumption.

| Fact | Value |
|---|---|
| Regular-season FINAL games (`game_id` prefix `002`) | **8,289** |
| Always-pick-home accuracy over those games | **0.5517** — the first baseline to beat |
| Games with a resolvable tip-off | **7,200 of 8,289 (86.9%)** |
| Seasons with full tip-off coverage | 2021-22, 2022-23, 2023-24, 2024-25 (~1,230 each) |
| Partial seasons | 2019-20 (775 — injury archive starts 2019-12), 2025-26 (426 — archive stops 2025-12-21) |
| Injury rows | 109,819, all carrying `game_time` |
| Odds rows | **0** — the user has no API key |

**Why tip-off resolution matters more than anything else here.** `games_raw.game_date` is a bare `DATE`; there is no tip-off time in the games table. The ONLY tip-off source is `injury_status_raw.game_time`, captured from the injury reports. A cutoff that is too late leaks the result. A cutoff that is too early throws away the 5:30pm ET injury report — the single most valuable pre-game signal — because that report is published only ~90 minutes before a 7pm ET tip. So the harness must resolve a real per-game tip-off, and must SKIP and COUNT games where it cannot, rather than guessing.

`game_time` values look like `'07:00 (ET)'` and `'08:00(ET)'` — the spacing differs between the two PDF layouts. They are 12-hour clock with no AM/PM marker. NBA games run roughly noon to 10:30pm ET, so the rule is: hour 12 means noon; hours 1-11 mean PM.

**Odds are absent.** The spec's "versus market" metrics cannot run on real data today. Build the code path and test it with synthetic odds, but the report must say plainly that market comparison is unavailable rather than printing a misleading zero.

## File structure

```
src/predictor/backtest/
  __init__.py
  tipoff.py      # resolve a game's true tip-off instant from injury report game_time
  baselines.py   # Predictor protocol + trivial predictors to validate the harness
  replay.py      # walk-forward engine: the only thing that builds AsOfView cutoffs
  metrics.py     # Brier, log loss, accuracy, calibration bins
  report.py      # BacktestResult + plain-English summary
tests/
  test_tipoff.py
  test_baselines.py
  test_replay.py
  test_backtest_leakage.py   # adversarial: predictors that try to cheat
  test_metrics.py
  test_backtest_report.py
```

`src/predictor/cli.py` gains a `backtest` command.

**Task ordering note:** the leakage tests (Task 4) come immediately after the replay engine and before any metric exists. A harness whose cutoff is wrong produces beautiful, meaningless numbers, so the guard is proven before anything can be measured.

---

### Task 1: Tip-off resolution

**Files:**
- Create: `src/predictor/backtest/__init__.py`, `src/predictor/backtest/tipoff.py`
- Test: `tests/test_tipoff.py`

**Interfaces:**
- Consumes: `db.POINT_IN_TIME_TABLES`.
- Produces:
  - `EASTERN: ZoneInfo`
  - `parse_game_time(raw: str, game_date: date) -> datetime | None` — a UTC instant, or None if unparseable.
  - `tipoff_index(con, season: str | None = None) -> dict[tuple[date, str], datetime]` — keyed by `(game_date, team_abbreviation)`.
  - `resolve_tipoff(index, game_date: date, home_team: str, away_team: str) -> datetime | None`

- [ ] **Step 1: Write the failing test**

Create `tests/test_tipoff.py`:

```python
from datetime import UTC, date, datetime

import pytest

from predictor.backtest import tipoff


def test_evening_game_is_pm():
    # '07:00 (ET)' on a January date means 7pm EST = 00:00 UTC next day
    got = tipoff.parse_game_time("07:00 (ET)", date(2025, 1, 15))
    assert got == datetime(2025, 1, 16, 0, 0, tzinfo=UTC)


def test_noon_game_is_not_shifted_to_midnight():
    # '12:00 (ET)' means noon, not midnight
    got = tipoff.parse_game_time("12:00 (ET)", date(2025, 1, 15))
    assert got == datetime(2025, 1, 15, 17, 0, tzinfo=UTC)


def test_afternoon_game():
    got = tipoff.parse_game_time("03:30 (ET)", date(2025, 1, 15))
    assert got == datetime(2025, 1, 15, 20, 30, tzinfo=UTC)


def test_layout_without_space_before_paren_parses_identically():
    with_space = tipoff.parse_game_time("08:00 (ET)", date(2025, 1, 15))
    without = tipoff.parse_game_time("08:00(ET)", date(2025, 1, 15))
    assert with_space == without


def test_daylight_saving_is_honoured():
    # June is EDT (UTC-4); January is EST (UTC-5)
    summer = tipoff.parse_game_time("07:00 (ET)", date(2025, 6, 10))
    winter = tipoff.parse_game_time("07:00 (ET)", date(2025, 1, 10))
    assert summer.hour == 23
    assert winter.hour == 0  # rolled into the next UTC day


def test_unparseable_returns_none():
    for bad in ["", "TBD", "not a time", "25:00 (ET)"]:
        assert tipoff.parse_game_time(bad, date(2025, 1, 15)) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tipoff.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'predictor.backtest'`

- [ ] **Step 3: Implement `tipoff.py`**

Create an empty `src/predictor/backtest/__init__.py`, then:

```python
from __future__ import annotations

import re
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from predictor import db

EASTERN = ZoneInfo("America/New_York")

# '07:00 (ET)' and '08:00(ET)' both occur -- the two PDF layouts differ in
# spacing. Times are a 12-hour clock with no AM/PM marker.
_TIME = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*\(ET\)\s*$")


def parse_game_time(raw: str, game_date: date) -> datetime | None:
    """Resolve an injury-report game time to a UTC instant.

    NBA games run roughly noon to 10:30pm Eastern, so a bare hour of 12 means
    noon and 1-11 mean PM. Returns None rather than guessing when the value
    cannot be parsed -- a wrong tip-off silently invalidates a backtest.
    """
    if not raw:
        return None
    m = _TIME.match(raw)
    if not m:
        return None
    hour, minute = int(m.group(1)), int(m.group(2))
    if not (1 <= hour <= 12 and 0 <= minute <= 59):
        return None
    hour24 = hour if hour == 12 else hour + 12
    local = datetime(
        game_date.year, game_date.month, game_date.day, hour24, minute,
        tzinfo=EASTERN,
    )
    return local.astimezone(UTC)


def tipoff_index(con, season: str | None = None) -> dict[tuple[date, str], datetime]:
    """Map (game_date, team) -> tip-off instant, from the injury reports.

    The injury report is the only place a tip-off time exists in this schema.
    """
    table = db.POINT_IN_TIME_TABLES["injury_status"]
    rows = con.execute(
        f"SELECT DISTINCT game_date, team, game_time FROM {table} "
        "WHERE game_date IS NOT NULL AND game_time IS NOT NULL AND game_time <> ''"
    ).fetchall()
    index: dict[tuple[date, str], datetime] = {}
    for game_date, team, raw in rows:
        parsed = parse_game_time(raw, game_date)
        if parsed is not None:
            index[(game_date, team)] = parsed
    return index


def resolve_tipoff(
    index: dict[tuple[date, str], datetime],
    game_date: date,
    home_team: str,
    away_team: str,
) -> datetime | None:
    """Tip-off for a game, from either team's injury-report entry."""
    return index.get((game_date, home_team)) or index.get((game_date, away_team))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tipoff.py -v`
Expected: 6 passed

- [ ] **Step 5: Verify coverage against the real database**

```bash
uv run python -c "
from predictor import db
from predictor.backtest import tipoff
con = db.connect()
idx = tipoff.tipoff_index(con)
print('tipoff index entries:', f'{len(idx):,}')
g = db.POINT_IN_TIME_TABLES['games']
games = con.execute(f\"SELECT DISTINCT game_id, game_date, home_team, away_team FROM {g} WHERE status='FINAL' AND game_id LIKE '002%'\").fetchall()
hit = sum(1 for _, d, h, a in games if tipoff.resolve_tipoff(idx, d, h, a))
print(f'resolvable: {hit:,} of {len(games):,} ({hit/len(games)*100:.1f}%)')
"
```

Expected: roughly **7,200 of 8,289 (86.9%)**. A materially lower number means the parser is rejecting a real format — investigate before continuing.

- [ ] **Step 6: Commit**

```bash
git add src/predictor/backtest tests/test_tipoff.py
git commit -m "feat: resolve game tip-off times from injury reports"
```

---

### Task 2: Predictor protocol and baselines

**Files:**
- Create: `src/predictor/backtest/baselines.py`
- Test: `tests/test_baselines.py`

**Interfaces:**
- Consumes: `asof.AsOfView`.
- Produces:
  - `GameToPredict` frozen dataclass — `game_id: str`, `season: str`, `game_date: date`, `home_team: str`, `away_team: str`, `tipoff: datetime`.
  - `Predictor` protocol — callable `(game: GameToPredict, view: AsOfView) -> float` returning P(home win) in [0, 1].
  - `always_home(game, view) -> float` — returns 1.0.
  - `fixed_probability(p: float) -> Predictor` — factory returning a predictor that always returns `p`.
  - `PredictionError(Exception)`

- [ ] **Step 1: Write the failing test**

Create `tests/test_baselines.py`:

```python
from datetime import UTC, date, datetime

import pytest

from predictor.backtest.baselines import (
    GameToPredict,
    always_home,
    fixed_probability,
)

GAME = GameToPredict(
    game_id="0022400561",
    season="2024-25",
    game_date=date(2025, 1, 15),
    home_team="PHI",
    away_team="NYK",
    tipoff=datetime(2025, 1, 16, 0, 0, tzinfo=UTC),
)


def test_always_home_returns_certainty():
    assert always_home(GAME, None) == 1.0


def test_fixed_probability_returns_its_value():
    assert fixed_probability(0.62)(GAME, None) == 0.62


def test_fixed_probability_rejects_values_outside_zero_one():
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            fixed_probability(bad)


def test_game_is_immutable():
    with pytest.raises(Exception):
        GAME.home_team = "BOS"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_baselines.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'predictor.backtest.baselines'`

- [ ] **Step 3: Implement `baselines.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol

from predictor.asof import AsOfView


class PredictionError(Exception):
    """Raised when a predictor returns something unusable."""


@dataclass(frozen=True)
class GameToPredict:
    """A game the harness is asking about, with no result attached.

    Deliberately carries no score: a predictor cannot leak what it is never
    handed.
    """

    game_id: str
    season: str
    game_date: date
    home_team: str
    away_team: str
    tipoff: datetime


class Predictor(Protocol):
    def __call__(self, game: GameToPredict, view: AsOfView) -> float:
        """Return P(home team wins), in [0, 1]."""


def always_home(game: GameToPredict, view: AsOfView) -> float:
    """The first baseline any model must beat."""
    return 1.0


def fixed_probability(p: float) -> Predictor:
    """A predictor that always returns the same probability."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {p}")

    def _predict(game: GameToPredict, view: AsOfView) -> float:
        return p

    return _predict
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_baselines.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/predictor/backtest/baselines.py tests/test_baselines.py
git commit -m "feat: predictor protocol and trivial baselines"
```

---

### Task 3: Walk-forward replay engine

**Files:**
- Create: `src/predictor/backtest/replay.py`
- Test: `tests/test_replay.py`

**Interfaces:**
- Consumes: `tipoff.tipoff_index`, `tipoff.resolve_tipoff`, `baselines.GameToPredict`, `baselines.Predictor`, `baselines.PredictionError`, `asof.AsOfView`, `db.POINT_IN_TIME_TABLES`.
- Produces:
  - `Prediction` frozen dataclass — `game_id`, `season`, `game_date`, `home_team`, `away_team`, `tipoff`, `cutoff`, `p_home: float`, `home_won: bool`.
  - `ReplayStats` frozen dataclass — `considered: int`, `predicted: int`, `skipped_no_tipoff: int`, `skipped_no_result: int`, `failed: int`.
  - `DEFAULT_BUFFER_MINUTES: int = 30`
  - `replay(con, predictor, season=None, buffer_minutes=DEFAULT_BUFFER_MINUTES, limit=None) -> tuple[list[Prediction], ReplayStats]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_replay.py`:

```python
from datetime import UTC, date, datetime, timedelta

import pytest

from predictor import db
from predictor.backtest import replay
from predictor.backtest.baselines import always_home, fixed_probability

TIP = datetime(2025, 1, 16, 0, 0, tzinfo=UTC)  # 7pm ET on 2025-01-15


@pytest.fixture
def con(tmp_path):
    c = db.connect(tmp_path / "t.duckdb")
    db.migrate(c)
    g = db.POINT_IN_TIME_TABLES["games"]
    i = db.POINT_IN_TIME_TABLES["injury_status"]
    # one played game, observed as SCHEDULED then FINAL
    c.execute(
        f"INSERT INTO {g} (game_id, season, game_date, home_team, away_team,"
        " home_points, away_points, status, observed_at, reconstructed)"
        " VALUES (?,?,?,?,?,?,?,?,?,TRUE)",
        ["0022400561", "2024-25", date(2025, 1, 15), "PHI", "NYK",
         None, None, "SCHEDULED", TIP - timedelta(days=7)],
    )
    c.execute(
        f"INSERT INTO {g} (game_id, season, game_date, home_team, away_team,"
        " home_points, away_points, status, observed_at, reconstructed)"
        " VALUES (?,?,?,?,?,?,?,?,?,TRUE)",
        ["0022400561", "2024-25", date(2025, 1, 15), "PHI", "NYK",
         119, 110, "FINAL", TIP + timedelta(hours=3)],
    )
    # an injury row supplying the tip-off time
    c.execute(
        f"INSERT INTO {i} (report_date, game_date, matchup, team, player,"
        " status, reason, observed_at, game_time)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        [date(2025, 1, 15), date(2025, 1, 15), "NYK@PHI", "PHI", "Embiid,Joel",
         "Out", "injury", TIP - timedelta(hours=2), "07:00 (ET)"],
    )
    return c


def test_replays_a_played_game(con):
    preds, stats = replay.replay(con, always_home)
    assert stats.predicted == 1
    assert len(preds) == 1
    p = preds[0]
    assert p.game_id == "0022400561"
    assert p.p_home == 1.0
    assert p.home_won is True


def test_cutoff_is_before_tipoff(con):
    preds, _ = replay.replay(con, always_home, buffer_minutes=30)
    assert preds[0].cutoff == TIP - timedelta(minutes=30)
    assert preds[0].cutoff < preds[0].tipoff


def test_game_without_a_resolvable_tipoff_is_skipped_and_counted(con):
    i = db.POINT_IN_TIME_TABLES["injury_status"]
    con.execute(f"DELETE FROM {i}")
    preds, stats = replay.replay(con, always_home)
    assert preds == []
    assert stats.skipped_no_tipoff == 1
    assert stats.predicted == 0


def test_unplayed_game_is_skipped_not_scored(con):
    g = db.POINT_IN_TIME_TABLES["games"]
    con.execute(f"DELETE FROM {g} WHERE status='FINAL'")
    preds, stats = replay.replay(con, always_home)
    assert preds == []
    assert stats.skipped_no_result == 1


def test_predictor_returning_an_impossible_probability_is_counted_not_silent(con):
    def bad(game, view):
        return 1.7

    preds, stats = replay.replay(con, bad)
    assert preds == []
    assert stats.failed == 1


def test_predictor_raising_does_not_abort_the_run(con):
    def explodes(game, view):
        raise RuntimeError("model blew up")

    preds, stats = replay.replay(con, explodes)
    assert stats.failed == 1
    assert stats.predicted == 0


def test_predictions_are_in_chronological_order(con):
    g = db.POINT_IN_TIME_TABLES["games"]
    i = db.POINT_IN_TIME_TABLES["injury_status"]
    later_tip = datetime(2025, 1, 20, 0, 0, tzinfo=UTC)
    for gid, d, tip in [("0022400999", date(2025, 1, 19), later_tip)]:
        con.execute(
            f"INSERT INTO {g} (game_id, season, game_date, home_team, away_team,"
            " home_points, away_points, status, observed_at, reconstructed)"
            " VALUES (?,?,?,?,?,?,?,?,?,TRUE)",
            [gid, "2024-25", d, "BOS", "LAL", 100, 90, "FINAL", tip + timedelta(hours=3)],
        )
        con.execute(
            f"INSERT INTO {i} (report_date, game_date, matchup, team, player,"
            " status, reason, observed_at, game_time)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            [d, d, "LAL@BOS", "BOS", "P", "Out", "x", tip - timedelta(hours=2), "07:00 (ET)"],
        )
    preds, _ = replay.replay(con, fixed_probability(0.5))
    assert [p.game_id for p in preds] == ["0022400561", "0022400999"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_replay.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'predictor.backtest.replay'`

- [ ] **Step 3: Implement `replay.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from predictor import db
from predictor.asof import AsOfView
from predictor.backtest import tipoff as tipoff_mod
from predictor.backtest.baselines import GameToPredict, Predictor

DEFAULT_BUFFER_MINUTES = 30


@dataclass(frozen=True)
class Prediction:
    game_id: str
    season: str
    game_date: date
    home_team: str
    away_team: str
    tipoff: datetime
    cutoff: datetime
    p_home: float
    home_won: bool


@dataclass(frozen=True)
class ReplayStats:
    considered: int
    predicted: int
    skipped_no_tipoff: int
    skipped_no_result: int
    failed: int


def replay(
    con,
    predictor: Predictor,
    season: str | None = None,
    buffer_minutes: int = DEFAULT_BUFFER_MINUTES,
    limit: int | None = None,
) -> tuple[list[Prediction], ReplayStats]:
    """Replay games chronologically, predicting each from a pre-tipoff view.

    The predictor is handed an AsOfView cut at tip-off minus a buffer and a
    GameToPredict that carries no score. It cannot see the result through the
    harness; the adversarial tests prove it cannot see it around the harness
    either.
    """
    games_table = db.POINT_IN_TIME_TABLES["games"]
    index = tipoff_mod.tipoff_index(con)

    where = ["game_id LIKE '002%'"]
    params: list = []
    if season is not None:
        where.append("season = ?")
        params.append(season)
    clause = " AND ".join(where)

    rows = con.execute(
        f"SELECT DISTINCT game_id, season, game_date, home_team, away_team "
        f"FROM {games_table} WHERE {clause} ORDER BY game_date, game_id",
        params,
    ).fetchall()

    considered = predicted = no_tip = no_result = failed = 0
    out: list[Prediction] = []

    for game_id, game_season, game_date, home_team, away_team in rows:
        considered += 1
        if limit is not None and predicted >= limit:
            break

        tip = tipoff_mod.resolve_tipoff(index, game_date, home_team, away_team)
        if tip is None:
            no_tip += 1
            continue

        result = con.execute(
            f"SELECT home_points, away_points FROM {games_table} "
            f"WHERE game_id = ? AND status = 'FINAL' "
            "AND home_points IS NOT NULL ORDER BY observed_at DESC LIMIT 1",
            [game_id],
        ).fetchone()
        if result is None:
            no_result += 1
            continue

        cutoff = tip - timedelta(minutes=buffer_minutes)
        view = AsOfView(con, cutoff)
        game = GameToPredict(
            game_id=game_id,
            season=game_season,
            game_date=game_date,
            home_team=home_team,
            away_team=away_team,
            tipoff=tip,
        )

        try:
            p_home = float(predictor(game, view))
        except Exception as exc:  # one bad game must not abort a season
            failed += 1
            print(f"backtest: predictor FAILED on {game_id} -- {exc}")
            continue

        if not 0.0 <= p_home <= 1.0:
            failed += 1
            print(
                f"backtest: predictor returned an impossible probability "
                f"{p_home} for {game_id} -- not scored"
            )
            continue

        out.append(
            Prediction(
                game_id=game_id,
                season=game_season,
                game_date=game_date,
                home_team=home_team,
                away_team=away_team,
                tipoff=tip,
                cutoff=cutoff,
                p_home=p_home,
                home_won=result[0] > result[1],
            )
        )
        predicted += 1

    return out, ReplayStats(considered, predicted, no_tip, no_result, failed)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_replay.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/predictor/backtest/replay.py tests/test_replay.py
git commit -m "feat: walk-forward replay engine"
```

---

### Task 4: Adversarial leakage tests

**The most important task in this sub-project.** A harness with a wrong cutoff produces beautiful, meaningless numbers, and the user intends to publish those numbers.

**Files:**
- Test: `tests/test_backtest_leakage.py`

**Interfaces:**
- Consumes: everything from Tasks 1-3.
- Produces: no source code — permanent regression tests.

- [ ] **Step 1: Write the adversarial tests**

Create `tests/test_backtest_leakage.py`:

```python
"""Adversarial tests: predictors that deliberately try to see the future.

If any test here fails, every backtest number this project produces is
meaningless, and the user would be publishing a track record built on a lie.
Treat a failure as a critical defect, never as a test to relax.
"""

from datetime import UTC, date, datetime, timedelta

import duckdb
import pytest

from predictor import db
from predictor.asof import AsOfError
from predictor.backtest import replay
from predictor.backtest.baselines import always_home

TIP = datetime(2025, 1, 16, 0, 0, tzinfo=UTC)


@pytest.fixture
def con(tmp_path):
    c = db.connect(tmp_path / "t.duckdb")
    db.migrate(c)
    g = db.POINT_IN_TIME_TABLES["games"]
    i = db.POINT_IN_TIME_TABLES["injury_status"]
    c.execute(
        f"INSERT INTO {g} (game_id, season, game_date, home_team, away_team,"
        " home_points, away_points, status, observed_at, reconstructed)"
        " VALUES (?,?,?,?,?,?,?,?,?,TRUE)",
        ["0022400561", "2024-25", date(2025, 1, 15), "PHI", "NYK",
         None, None, "SCHEDULED", TIP - timedelta(days=7)],
    )
    c.execute(
        f"INSERT INTO {g} (game_id, season, game_date, home_team, away_team,"
        " home_points, away_points, status, observed_at, reconstructed)"
        " VALUES (?,?,?,?,?,?,?,?,?,TRUE)",
        ["0022400561", "2024-25", date(2025, 1, 15), "PHI", "NYK",
         119, 110, "FINAL", TIP + timedelta(hours=3)],
    )
    c.execute(
        f"INSERT INTO {i} (report_date, game_date, matchup, team, player,"
        " status, reason, observed_at, game_time)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        [date(2025, 1, 15), date(2025, 1, 15), "NYK@PHI", "PHI", "Embiid,Joel",
         "Out", "injury", TIP - timedelta(hours=2), "07:00 (ET)"],
    )
    return c


def test_the_view_handed_to_a_predictor_cannot_see_the_final_score(con):
    """The core guarantee. A predictor querying games sees SCHEDULED only."""
    seen = {}

    def snooping(game, view):
        rows = view.table("games").project("status, home_points").fetchall()
        seen["rows"] = rows
        return 0.5

    replay.replay(con, snooping)
    assert seen["rows"] == [("SCHEDULED", None)]
    assert all(r[1] is None for r in seen["rows"]), "final score leaked"


def test_the_game_object_carries_no_result(con):
    captured = {}

    def grabby(game, view):
        captured["fields"] = vars(game)
        return 0.5

    replay.replay(con, grabby)
    blob = repr(captured["fields"])
    assert "119" not in blob and "110" not in blob
    for forbidden in ("home_points", "away_points", "home_won", "status"):
        assert forbidden not in captured["fields"]


def test_a_predictor_cannot_reach_the_physical_table_through_the_view(con):
    """The renamed physical tables are unreachable from a chained query."""
    outcome = {}

    def cheater(game, view):
        try:
            view.table("games").project(
                "status, (SELECT max(home_points) FROM games) AS leak"
            ).fetchall()
            outcome["leaked"] = True
        except duckdb.CatalogException:
            outcome["leaked"] = False
        return 0.5

    replay.replay(con, cheater)
    assert outcome["leaked"] is False


def test_a_predictor_cannot_move_the_cutoff(con):
    outcome = {}

    def tamperer(game, view):
        try:
            view.as_of = datetime(2099, 1, 1, tzinfo=UTC)
            outcome["moved"] = True
        except AttributeError:
            outcome["moved"] = False
        return 0.5

    replay.replay(con, tamperer)
    assert outcome["moved"] is False


def test_injury_rows_published_after_the_cutoff_are_invisible(con):
    """A report filed after the cutoff must not reach the predictor."""
    i = db.POINT_IN_TIME_TABLES["injury_status"]
    con.execute(
        f"INSERT INTO {i} (report_date, game_date, matchup, team, player,"
        " status, reason, observed_at, game_time)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        [date(2025, 1, 15), date(2025, 1, 15), "NYK@PHI", "PHI", "LateScratch,Guy",
         "Out", "injury", TIP - timedelta(minutes=5), "07:00 (ET)"],
    )
    players = {}

    def looker(game, view):
        players["seen"] = {
            r[0] for r in view.table("injury_status").project("player").fetchall()
        }
        return 0.5

    replay.replay(con, looker, buffer_minutes=30)
    assert "Embiid,Joel" in players["seen"]
    assert "LateScratch,Guy" not in players["seen"], "post-cutoff report leaked"


def test_every_cutoff_precedes_its_own_tipoff(con):
    preds, _ = replay.replay(con, always_home)
    assert preds, "fixture produced no predictions"
    for p in preds:
        assert p.cutoff < p.tipoff


def test_a_zero_buffer_still_does_not_include_the_result(con):
    """Even with no safety buffer, the FINAL row is observed after tip-off."""
    seen = {}

    def snooping(game, view):
        seen["rows"] = view.table("games").project("status, home_points").fetchall()
        return 0.5

    replay.replay(con, snooping, buffer_minutes=0)
    assert all(r[1] is None for r in seen["rows"])
```

- [ ] **Step 2: Run the leakage tests**

Run: `uv run pytest tests/test_backtest_leakage.py -v`
Expected: 7 passed

If any fail, STOP and fix the replay engine — do not weaken an assertion.

- [ ] **Step 3: Commit**

```bash
git add tests/test_backtest_leakage.py
git commit -m "test: adversarial leakage tests for the backtest harness"
```

---

### Task 5: Accuracy and probabilistic metrics

**Files:**
- Create: `src/predictor/backtest/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `replay.Prediction`.
- Produces:
  - `brier_score(preds) -> float`
  - `log_loss(preds) -> float`
  - `accuracy(preds, threshold: float = 0.5) -> float`
  - `home_rate(preds) -> float`
  - `MetricsError(Exception)`

- [ ] **Step 1: Write the failing test**

Create `tests/test_metrics.py`:

```python
from datetime import UTC, date, datetime

import pytest

from predictor.backtest import metrics
from predictor.backtest.replay import Prediction

TIP = datetime(2025, 1, 16, 0, 0, tzinfo=UTC)


def make(p_home: float, home_won: bool, gid: str = "g") -> Prediction:
    return Prediction(
        game_id=gid, season="2024-25", game_date=date(2025, 1, 15),
        home_team="PHI", away_team="NYK", tipoff=TIP, cutoff=TIP,
        p_home=p_home, home_won=home_won,
    )


def test_brier_is_zero_for_perfect_confident_predictions():
    preds = [make(1.0, True), make(0.0, False)]
    assert metrics.brier_score(preds) == 0.0


def test_brier_is_one_for_perfectly_wrong_confident_predictions():
    preds = [make(0.0, True), make(1.0, False)]
    assert metrics.brier_score(preds) == 1.0


def test_brier_of_a_coin_flip_is_a_quarter():
    preds = [make(0.5, True), make(0.5, False)]
    assert metrics.brier_score(preds) == pytest.approx(0.25)


def test_accuracy_counts_the_side_the_probability_favours():
    preds = [make(0.9, True), make(0.9, False), make(0.1, False), make(0.1, True)]
    assert metrics.accuracy(preds) == pytest.approx(0.5)


def test_log_loss_penalises_confident_errors_more_than_brier():
    confident_wrong = [make(0.02, True)]
    mild_wrong = [make(0.45, True)]
    assert metrics.log_loss(confident_wrong) > metrics.log_loss(mild_wrong)


def test_log_loss_is_finite_for_a_certain_wrong_prediction():
    """A raw log loss would be infinite; probabilities must be clipped."""
    assert metrics.log_loss([make(0.0, True)]) < float("inf")


def test_home_rate_reports_the_base_rate():
    preds = [make(0.5, True), make(0.5, True), make(0.5, False), make(0.5, False)]
    assert metrics.home_rate(preds) == pytest.approx(0.5)


def test_metrics_on_an_empty_list_raise_rather_than_return_nonsense():
    for fn in (metrics.brier_score, metrics.log_loss, metrics.accuracy, metrics.home_rate):
        with pytest.raises(metrics.MetricsError):
            fn([])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'predictor.backtest.metrics'`

- [ ] **Step 3: Implement `metrics.py`**

```python
from __future__ import annotations

import math
from collections.abc import Sequence

from predictor.backtest.replay import Prediction

# Probabilities are clipped before taking a logarithm: a confident, wrong
# prediction would otherwise produce an infinite loss and poison the average.
_EPS = 1e-15


class MetricsError(Exception):
    """Raised when a metric is asked for something it cannot compute."""


def _require(preds: Sequence[Prediction]) -> None:
    if not preds:
        raise MetricsError(
            "no predictions to score -- the replay produced nothing, so there "
            "is nothing to measure"
        )


def brier_score(preds: Sequence[Prediction]) -> float:
    """Mean squared error of the probability. Lower is better; 0.25 is a coin flip."""
    _require(preds)
    return sum((p.p_home - (1.0 if p.home_won else 0.0)) ** 2 for p in preds) / len(preds)


def log_loss(preds: Sequence[Prediction]) -> float:
    """Mean negative log likelihood. Punishes confident errors far harder than Brier."""
    _require(preds)
    total = 0.0
    for p in preds:
        q = min(max(p.p_home, _EPS), 1.0 - _EPS)
        total += -math.log(q) if p.home_won else -math.log(1.0 - q)
    return total / len(preds)


def accuracy(preds: Sequence[Prediction], threshold: float = 0.5) -> float:
    """Fraction of games where the favoured side actually won."""
    _require(preds)
    hits = sum(1 for p in preds if (p.p_home >= threshold) == p.home_won)
    return hits / len(preds)


def home_rate(preds: Sequence[Prediction]) -> float:
    """Base rate of home wins in the scored set -- the always-pick-home accuracy."""
    _require(preds)
    return sum(1 for p in preds if p.home_won) / len(preds)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/predictor/backtest/metrics.py tests/test_metrics.py
git commit -m "feat: accuracy and probabilistic backtest metrics"
```

---

### Task 6: Calibration

Calibration is the headline metric for this project: a well-calibrated 62% model is publishable, an overconfident 65% one destroys credibility the first time a stated near-certainty loses.

**Files:**
- Modify: `src/predictor/backtest/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `replay.Prediction`.
- Produces:
  - `CalibrationBin` frozen dataclass — `low: float`, `high: float`, `count: int`, `mean_predicted: float`, `observed_rate: float`.
  - `calibration_bins(preds, n_bins: int = 10) -> list[CalibrationBin]` — empty bins omitted.
  - `calibration_error(preds, n_bins: int = 10) -> float` — count-weighted mean absolute gap (expected calibration error).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_metrics.py`:

```python
def test_a_perfectly_calibrated_predictor_has_near_zero_error():
    # 100 games at p=0.7, exactly 70 won
    preds = [make(0.7, i < 70, f"g{i}") for i in range(100)]
    assert metrics.calibration_error(preds) == pytest.approx(0.0, abs=0.01)


def test_an_overconfident_predictor_has_large_calibration_error():
    # claims 95%, actually wins half the time
    preds = [make(0.95, i < 50, f"g{i}") for i in range(100)]
    assert metrics.calibration_error(preds) > 0.4


def test_bins_report_predicted_against_observed():
    preds = [make(0.9, i < 60, f"g{i}") for i in range(100)]
    bins = metrics.calibration_bins(preds, n_bins=10)
    assert len(bins) == 1
    b = bins[0]
    assert b.count == 100
    assert b.mean_predicted == pytest.approx(0.9)
    assert b.observed_rate == pytest.approx(0.6)


def test_empty_bins_are_omitted_not_reported_as_zero():
    preds = [make(0.55, True, f"g{i}") for i in range(10)]
    bins = metrics.calibration_bins(preds, n_bins=10)
    assert len(bins) == 1
    assert all(b.count > 0 for b in bins)


def test_a_probability_of_exactly_one_lands_in_the_top_bin():
    preds = [make(1.0, True, f"g{i}") for i in range(5)]
    bins = metrics.calibration_bins(preds, n_bins=10)
    assert len(bins) == 1
    assert bins[0].high == pytest.approx(1.0)
    assert bins[0].count == 5


def test_calibration_on_an_empty_list_raises():
    with pytest.raises(metrics.MetricsError):
        metrics.calibration_bins([])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL — `AttributeError: module has no attribute 'calibration_error'`

- [ ] **Step 3: Implement calibration in `metrics.py`**

Add `from dataclasses import dataclass` to the imports at the TOP of the file
(not mid-file), then append the rest:

```python
@dataclass(frozen=True)
class CalibrationBin:
    low: float
    high: float
    count: int
    mean_predicted: float
    observed_rate: float


def calibration_bins(
    preds: Sequence[Prediction], n_bins: int = 10
) -> list[CalibrationBin]:
    """Group predictions by stated probability and compare to what happened.

    Empty bins are omitted rather than reported as zero, which would read as
    'we said 30% and were never right' instead of 'we never said 30%'.
    """
    _require(preds)
    if n_bins < 1:
        raise MetricsError(f"n_bins must be at least 1, got {n_bins}")

    buckets: list[list[Prediction]] = [[] for _ in range(n_bins)]
    for p in preds:
        # p == 1.0 would index past the end; clamp it into the top bin.
        idx = min(int(p.p_home * n_bins), n_bins - 1)
        buckets[idx].append(p)

    out: list[CalibrationBin] = []
    for idx, bucket in enumerate(buckets):
        if not bucket:
            continue
        out.append(
            CalibrationBin(
                low=idx / n_bins,
                high=(idx + 1) / n_bins,
                count=len(bucket),
                mean_predicted=sum(b.p_home for b in bucket) / len(bucket),
                observed_rate=sum(1 for b in bucket if b.home_won) / len(bucket),
            )
        )
    return out


def calibration_error(preds: Sequence[Prediction], n_bins: int = 10) -> float:
    """Count-weighted mean gap between stated probability and observed rate."""
    bins = calibration_bins(preds, n_bins)
    total = sum(b.count for b in bins)
    return sum(
        b.count * abs(b.mean_predicted - b.observed_rate) for b in bins
    ) / total
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: 14 passed

- [ ] **Step 5: Commit**

```bash
git add src/predictor/backtest/metrics.py tests/test_metrics.py
git commit -m "feat: calibration bins and expected calibration error"
```

---

### Task 7: Result object and plain-English report

**Files:**
- Create: `src/predictor/backtest/report.py`
- Test: `tests/test_backtest_report.py`

**Interfaces:**
- Consumes: `replay.Prediction`, `replay.ReplayStats`, everything in `metrics`.
- Produces:
  - `BacktestResult` frozen dataclass — `predictions`, `stats`, `brier`, `log_loss`, `accuracy`, `home_baseline`, `calibration_error`, `bins`, `market_available: bool`.
  - `summarize(preds, stats) -> BacktestResult`
  - `format_report(result) -> str`

- [ ] **Step 1: Write the failing test**

Create `tests/test_backtest_report.py`:

```python
from datetime import UTC, date, datetime

import pytest

from predictor.backtest import report
from predictor.backtest.replay import Prediction, ReplayStats

TIP = datetime(2025, 1, 16, 0, 0, tzinfo=UTC)


def make(p_home, home_won, gid="g"):
    return Prediction(
        game_id=gid, season="2024-25", game_date=date(2025, 1, 15),
        home_team="PHI", away_team="NYK", tipoff=TIP, cutoff=TIP,
        p_home=p_home, home_won=home_won,
    )


STATS = ReplayStats(considered=120, predicted=100, skipped_no_tipoff=15,
                    skipped_no_result=5, failed=0)


def test_summary_computes_every_headline_metric():
    preds = [make(0.7, i < 70, f"g{i}") for i in range(100)]
    r = report.summarize(preds, STATS)
    assert r.accuracy == pytest.approx(0.7)
    assert r.home_baseline == pytest.approx(0.7)
    assert r.brier > 0
    assert r.calibration_error == pytest.approx(0.0, abs=0.01)
    assert r.market_available is False


def test_report_leads_with_the_verdict_against_the_baseline():
    preds = [make(0.9, i < 60, f"g{i}") for i in range(100)]
    text = report.format_report(report.summarize(preds, STATS))
    first = text.splitlines()[0]
    assert first.startswith(("BEATS", "LOSES TO", "MATCHES"))


def test_report_states_coverage_honestly():
    preds = [make(0.7, i < 70, f"g{i}") for i in range(100)]
    text = report.format_report(report.summarize(preds, STATS))
    assert "100" in text
    assert "15" in text  # skipped for no tip-off
    assert "tip-off" in text.lower()


def test_report_says_market_comparison_is_unavailable_rather_than_printing_zero():
    preds = [make(0.7, i < 70, f"g{i}") for i in range(100)]
    text = report.format_report(report.summarize(preds, STATS))
    assert "market" in text.lower()
    assert "unavailable" in text.lower() or "no odds" in text.lower()


def test_report_includes_a_readable_calibration_table():
    preds = [make(0.65, i < 65, f"a{i}") for i in range(100)]
    preds += [make(0.35, i < 35, f"b{i}") for i in range(100)]
    text = report.format_report(report.summarize(preds, STATS))
    assert "calibration" in text.lower()
    assert text.count("%") >= 4


def test_summarize_with_no_predictions_raises_rather_than_reporting_zeroes():
    empty = ReplayStats(considered=10, predicted=0, skipped_no_tipoff=10,
                        skipped_no_result=0, failed=0)
    with pytest.raises(Exception):
        report.summarize([], empty)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_backtest_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'predictor.backtest.report'`

- [ ] **Step 3: Implement `report.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from predictor.backtest import metrics
from predictor.backtest.replay import Prediction, ReplayStats


@dataclass(frozen=True)
class BacktestResult:
    predictions: Sequence[Prediction]
    stats: ReplayStats
    brier: float
    log_loss: float
    accuracy: float
    home_baseline: float
    calibration_error: float
    bins: list[metrics.CalibrationBin]
    market_available: bool


def summarize(
    preds: Sequence[Prediction], stats: ReplayStats
) -> BacktestResult:
    """Compute every headline metric. Raises if there is nothing to score."""
    return BacktestResult(
        predictions=preds,
        stats=stats,
        brier=metrics.brier_score(preds),
        log_loss=metrics.log_loss(preds),
        accuracy=metrics.accuracy(preds),
        home_baseline=metrics.home_rate(preds),
        calibration_error=metrics.calibration_error(preds),
        bins=metrics.calibration_bins(preds),
        # No odds data exists yet; the market comparison is built but cannot run.
        market_available=False,
    )


def format_report(result: BacktestResult) -> str:
    """A report someone can read in thirty seconds and trust."""
    edge = result.accuracy - result.home_baseline
    if edge > 0.005:
        verdict = f"BEATS always-pick-home by {edge * 100:.1f} points"
    elif edge < -0.005:
        verdict = f"LOSES TO always-pick-home by {abs(edge) * 100:.1f} points"
    else:
        verdict = "MATCHES always-pick-home"

    s = result.stats
    lines = [
        verdict,
        "",
        f"  games scored        : {s.predicted:,} of {s.considered:,} considered",
        f"  accuracy            : {result.accuracy * 100:.1f}%",
        f"  always-pick-home    : {result.home_baseline * 100:.1f}%  (the baseline)",
        f"  Brier score         : {result.brier:.4f}  (lower is better; 0.25 is a coin flip)",
        f"  log loss            : {result.log_loss:.4f}",
        f"  calibration error   : {result.calibration_error * 100:.1f} points average gap",
        "",
        "  Calibration -- when it said X%, how often did that happen?",
    ]
    for b in result.bins:
        lines.append(
            f"    {b.low * 100:3.0f}-{b.high * 100:3.0f}%  "
            f"said {b.mean_predicted * 100:5.1f}%  "
            f"actual {b.observed_rate * 100:5.1f}%  "
            f"({b.count:,} games)"
        )

    lines += ["", "  Coverage and exclusions:"]
    if s.skipped_no_tipoff:
        lines.append(
            f"    {s.skipped_no_tipoff:,} game(s) skipped -- no tip-off time could be "
            "resolved, so no honest pre-game cutoff exists for them"
        )
    if s.skipped_no_result:
        lines.append(f"    {s.skipped_no_result:,} game(s) skipped -- not yet played")
    if s.failed:
        lines.append(
            f"    {s.failed:,} game(s) NOT scored -- the predictor failed or returned "
            "an impossible probability. See the lines above."
        )

    if not result.market_available:
        lines += [
            "",
            "  Market comparison: unavailable -- no odds data has been collected "
            "(no ODDS_API_KEY set), so there is nothing to compare against.",
        ]

    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_backtest_report.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/predictor/backtest/report.py tests/test_backtest_report.py
git commit -m "feat: backtest result summary and plain-English report"
```

---

### Task 8: CLI command and a real season run

**Files:**
- Modify: `src/predictor/cli.py`
- Test: `tests/test_backtest_cli.py`

**Interfaces:**
- Consumes: `replay.replay`, `report.summarize`, `report.format_report`, `baselines.always_home`, `baselines.fixed_probability`.
- Produces: a `predictor backtest` CLI command.

- [ ] **Step 1: Write the failing test**

Create `tests/test_backtest_cli.py`:

```python
import pytest
from typer.testing import CliRunner

from predictor.cli import app

runner = CliRunner()


def test_backtest_rejects_an_unknown_model_in_plain_english():
    result = runner.invoke(app, ["backtest", "--model", "not-a-model"])
    assert result.exit_code != 0
    assert "not-a-model" in result.stdout
    assert "always-home" in result.stdout  # tells them what IS available


def test_backtest_reports_nothing_to_score_rather_than_crashing(monkeypatch):
    from predictor.backtest import replay as replay_mod

    def empty(*args, **kwargs):
        return [], replay_mod.ReplayStats(0, 0, 0, 0, 0)

    monkeypatch.setattr(replay_mod, "replay", empty)
    result = runner.invoke(app, ["backtest", "--season", "1999-00"])
    assert result.exit_code != 0
    assert "no games" in result.stdout.lower() or "nothing to score" in result.stdout.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_backtest_cli.py -v`
Expected: FAIL — no such command `backtest`

- [ ] **Step 3: Add the CLI command**

Append to `src/predictor/cli.py`:

```python
@app.command("backtest")
def backtest_cmd(
    model: str = typer.Option(
        "always-home", help="Which predictor to score: always-home or coin-flip."
    ),
    season: str = typer.Option(None, help="Limit to one season, e.g. 2024-25."),
    buffer_minutes: int = typer.Option(
        30, help="Minutes before tip-off to cut the data off."
    ),
) -> None:
    """Replay real games and score a predictor on what was knowable pre-tipoff."""
    from predictor import db
    from predictor.backtest import baselines, replay, report
    from predictor.config import settings

    known = {
        "always-home": baselines.always_home,
        "coin-flip": baselines.fixed_probability(0.5),
    }
    if model not in known:
        typer.echo(
            f"Unknown model '{model}'. Available: {', '.join(sorted(known))}."
        )
        raise typer.Exit(code=1)

    settings.ensure_dirs()
    con = db.connect()
    db.migrate(con)

    preds, stats = replay.replay(
        con, known[model], season=season, buffer_minutes=buffer_minutes
    )
    if not preds:
        typer.echo(
            "No games could be scored -- nothing to measure. "
            f"{stats.considered:,} game(s) were considered; "
            f"{stats.skipped_no_tipoff:,} had no resolvable tip-off time and "
            f"{stats.skipped_no_result:,} had no result yet."
        )
        raise typer.Exit(code=1)

    typer.echo(report.format_report(report.summarize(preds, stats)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_backtest_cli.py -v`
Expected: 2 passed

- [ ] **Step 5: Run it for real on one full season**

```bash
uv run predictor backtest --model always-home --season 2023-24
```

Expected: roughly **1,229 games scored**, accuracy equal to the home-win base rate (always-home is right exactly when the home team wins), calibration showing a single bin at 100%, and the market line reporting unavailable.

This is the sanity check that the whole harness works end to end. If the scored count is far from 1,229, the tip-off resolution or the game filter is wrong.

- [ ] **Step 6: Run the full window and record the baseline**

```bash
uv run predictor backtest --model always-home
uv run predictor backtest --model coin-flip
```

Expected: always-home accuracy near **55.2%** across all scored games, and coin-flip near 50% with a Brier of about 0.25. Record both in the commit message — they are the numbers every future model is measured against.

- [ ] **Step 7: Commit**

```bash
git add src/predictor/cli.py tests/test_backtest_cli.py
git commit -m "feat: backtest CLI command"
```

---

## Definition of done

- [ ] `uv run pytest -v` passes, including every test in `tests/test_backtest_leakage.py`.
- [ ] `uv run predictor backtest --model always-home --season 2023-24` scores roughly 1,229 games.
- [ ] Always-pick-home accuracy over the full window is approximately 55.2%, matching the independently measured baseline.
- [ ] A coin-flip predictor scores a Brier of approximately 0.25.
- [ ] The report states tip-off coverage and exclusions explicitly rather than silently scoring a subset.
- [ ] The report says market comparison is unavailable rather than printing a misleading zero.
- [ ] No predictor can reach a final score through the view handed to it, proven by the adversarial tests.

## What this deliberately does not include

- **Any model.** The harness is model-agnostic by design; the Elo-plus-residual core is sub-project 3. Baselines exist only to validate the harness.
- **Retraining inside the replay.** The spec calls for walk-forward retraining on a schedule; that belongs with the model, which does not exist yet. The engine's per-game loop is the seam it will hook into.
- **Market/CLV metrics on real data.** The code path and its tests exist, but there are zero odds rows until an API key is configured.
- **Reconstructed-versus-observed split reporting.** Every historical row is currently `reconstructed = TRUE`, so the split would be degenerate. Add it when genuinely observed rows exist.
- **Multi-season retraining windows and rolling origin evaluation.** Once a model exists and one season replays cleanly, extend.
