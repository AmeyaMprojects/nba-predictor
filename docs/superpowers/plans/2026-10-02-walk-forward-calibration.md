# Walk-Forward Recalibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retire 2023-26 as the test, choose Stage 1 settings by walk-forward evaluation over seven tuning seasons with three recency variants, and make 2026-27 (live) the only test.

**Architecture:**
- **Shared selection engine.** A new `src/predictor/model/tuning.py` holds the selection engine used by both `fit-model` and the new `evaluate-model`. It simulates each rating-grid combination once over all games. Within that same pass it solves every weighted least-squares "job" (a set of seasons plus a recency weighting), keeping the best combination per job.
- **Walk-forward.** `evaluate()` predicts each tuning season from 2020-21 to 2025-26 with the job fitted only on earlier seasons. It picks the recency variant with the lowest pooled log loss, and returns the final settings fitted on all seven tuning seasons with that variant.

**Tech Stack:** Python 3.14, numpy, DuckDB, typer, pytest. Run with `uv run`.

**Spec:** `docs/superpowers/specs/2026-09-22-nba-predictor-design.md`, section 3, subsection **"Calibration redesign — decided 2026-10-02"** (and "Recalibration — 2026-10-02 (second look)" for the background).

## Global Constraints

- **Season roles (exact):**
  - `WARMUP_SEASONS = ("2014-15", "2015-16", "2016-17", "2017-18", "2018-19")`
  - `TUNING_SEASONS = ("2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26")`
  - `TEST_SEASONS = ("2026-27",)`
- **Walk-forward seasons:** `TUNING_SEASONS[1:]` (2020-21 → 2025-26). Each is predicted only with settings chosen on `TUNING_SEASONS[:i]`, the seasons strictly before it. Code must raise if a fold's seasons include the predicted season or a later one.
- **Grids (exact, iterated in this nesting order: k, cap, regression, window):**
  - `GRID_K = (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20)`
  - `GRID_CAP = (15.0, 20.0, 25.0, 30.0, 40.0)`
  - `GRID_REGRESSION = (0.0, 0.1, 0.2, 0.33, 0.5, 0.66)`
  - `GRID_WINDOW = (400, 800, 1230)`
  - `SIGMA_GRID = tuple(i / 100 for i in range(800, 2001, 5))`
  - Ties: strictly lower wins, so the first combination in grid order is kept.
- **Recency variants (exact, in preference order):** `HALF_LIVES = (None, 3.0, 1.0)`.
  - The weight of season `s` in a job whose seasons are `S` (oldest → newest) is `1.0` when the half-life is None, otherwise `0.5 ** ((len(S) - 1 - S.index(s)) / half_life)`.
  - Weights apply to the grid MSE, the least squares, and the sigma log loss.
- **Winner rule:** the lowest mean walk-forward log loss. A later variant replaces the current best only if its loss is lower by more than `TIE_TOLERANCE = 0.001`.
- **Unchanged:**
  - The rating update, adjustments, and `Stage1Predictor`. No change to model math beyond which settings are chosen.
  - Physical `_raw` table names only in db.py/asof.py.
  - data/predictor.duckdb is never written by tests.
  - Plain-English CLI errors.
- **Test hygiene.** Tests compare against values worked out by hand where arithmetic is involved. Slow real-archive tests (> 60 s) get `@pytest.mark.slow`, which is excluded by default and run with `uv run pytest -m slow`.
- Baseline before this plan: `uv run pytest -q` gives **533 passed** on branch `walk-forward` (HEAD 2b9f1d6).

---

### Task 1: Season roles and settings file version 2

**Files:**
- Modify: `src/predictor/model/settings.py`, `src/predictor/model/fit.py` (minimal), `src/predictor/cli.py` (stage1 headline message only), `pyproject.toml` (pytest marker)
- Modify tests: `tests/test_model_settings.py`, `tests/test_model_fit.py`, `tests/test_model_stage1.py`, `tests/test_model_backtest.py`

**Interfaces produced:**
- `settings.WARMUP_SEASONS`, `settings.TUNING_SEASONS`, `settings.TEST_SEASONS`: exact values from Global Constraints. `FIT_SEASONS` and `CALIBRATE_SEASON` are deleted.
- `settings.season_role(season) -> "warm-up" | "tuning" | "test" | "unassigned"`
- `ModelSettings(ratings: RatingParams, coefficients: Coefficients, sigma: float, half_life: float | None, tuning_games: int)`. JSON `"version": 2` stores `half_life` as a number or `null`. `from_json` rejects version 1 with SettingsError mentioning `predictor fit-model`.

- [ ] **Step 1: Tests first**

In `tests/test_model_settings.py`:
- Rebuild `S` with the new fields (`half_life=3.0, tuning_games=7889`).
- Add a round-trip with `half_life=None`, which must serialise as JSON `null` and come back as `None`.
- Add a test that a version-1 document raises SettingsError matching "predictor fit-model".
- Replace the role parametrisation with:

```python
@pytest.mark.parametrize(
    "season, role",
    [("2014-15", "warm-up"), ("2018-19", "warm-up"), ("2019-20", "tuning"),
     ("2023-24", "tuning"), ("2025-26", "tuning"), ("2026-27", "test"),
     ("2027-28", "unassigned")],
)
def test_season_roles(season, role):
    assert ms.season_role(season) == role


def test_roles_do_not_overlap():
    groups = [set(ms.WARMUP_SEASONS), set(ms.TUNING_SEASONS), set(ms.TEST_SEASONS)]
    assert sum(len(g) for g in groups) == len(set().union(*groups)) == 13
```

In every other test file, change the `ModelSettings(...)` constructions from `fit_games=…, calibrate_games=…` to `half_life=None, tuning_games=…`.

- [ ] **Step 2: Implement `settings.py`**

- Set `_VERSION = 2`.
- Replace the role constants.
- `season_role`: return "warm-up" if in WARMUP, "tuning" if in TUNING, "test" if in TEST, else "unassigned".
- `to_json` writes `"half_life": s.half_life, "tuning_games": s.tuning_games`.
- `from_json` reads them; `half_life` is `None` if the JSON holds null, else a float.
- Validation:
  - `half_life`, when not None, must be finite and > 0.
  - `tuning_games >= 0`.
  - Otherwise raise SettingsError with the "Run: predictor fit-model" hint.
- The version check message for any version other than 2 stays as it is.

- [ ] **Step 3: Keep `fit.py` working (temporary, replaced in Task 2)**

- Import `TUNING_SEASONS, TEST_SEASONS, WARMUP_SEASONS` (drop FIT/CALIBRATE).
- `_load` loads `WARMUP_SEASONS + TUNING_SEASONS`.
- In `fit()`, use `TUNING_SEASONS` wherever FIT_SEASONS or the sigma seasons were used, and drop the separate calibrate count.
- Return `ModelSettings(..., half_life=None, tuning_games=<count of 002 games in TUNING_SEASONS>)`.
- `describe`'s last line becomes: `f"Chosen on {s.tuning_games:,} tuning-season games ({TUNING_SEASONS[0]} to {TUNING_SEASONS[-1]}); recency: {'equal weight' if s.half_life is None else f'half-life {s.half_life:g} season(s)'}."`

Update `tests/test_model_fit.py`:
- `_history` builds `WARMUP_SEASONS[-1:] + TUNING_SEASONS + TEST_SEASONS`, 40 games each.
- Count assertions: `tuning_games == 7 * 40`.
- The mislabeled-test-season planting uses `TEST_SEASONS[0]` (`"2026-27"`) dated inside the 2019-20 window.
- Drop `test_sigma_is_chosen_on_fit_and_calibrate_seasons_pooled`.
- Update the describe test to the new last line.
- Mark `test_committed_settings_reproduce_from_the_real_archive` `@pytest.mark.slow`.

- [ ] **Step 4: The `backtest --model stage1` headline**

When `headline` is empty and no `--season` was given, print:

`"No live 2026-27 games have been scored yet, so there is no test headline. The pre-season evaluation is 'predictor evaluate-model'."`

and exit 1. The `--season <non-test season>` message keeps naming the filter.

Update `tests/test_model_backtest.py`:
- Fixtures that used `"2023-24"` / `"2024-25"` as test seasons now use `"2026-27"` / `"2026-27"`-dated games (e.g. `date(2026, 11, 1)`), with game ids starting `00226`.
- The "no test-season games" test expects the new message.
- Header assertions use `test season 2026-27`.
- The real-archive test `test_every_test_season_game_is_predicted_and_every_explanation_adds_up` is parametrised over `["2023-24", "2024-25", "2025-26"]` explicitly (now tuning seasons) and renamed `test_every_recent_tuning_season_game_is_predicted_and_every_explanation_adds_up`. Keep it in the default run.

- [ ] **Step 5: pytest marker**

In `pyproject.toml` under `[tool.pytest.ini_options]` add:

```toml
markers = ["slow: real-archive checks over 60s; run with `uv run pytest -m slow`"]
addopts = "-m 'not slow'"
```

- [ ] **Step 6: Regenerate settings and run**

```bash
uv run predictor fit-model
uv run pytest -q
```

The fit now runs on 2019-26 with the old grid; the values will change. The suite must pass, except that the slow reproducibility test is deselected. Then run `uv run pytest -m slow -q`: the reproducibility test must pass against the regenerated file.

- [ ] **Step 7: Commit**

```bash
git add -A src/predictor/model/settings.py src/predictor/model/fit.py src/predictor/model/stage1_settings.json src/predictor/cli.py pyproject.toml tests/
git commit -m "feat: retire 2023-26 as the test; seven tuning seasons, 2026-27 live test, settings v2

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: The selection engine (`tuning.py`) with recency weights

**Files:**
- Create: `src/predictor/model/tuning.py`
- Modify: `src/predictor/model/fit.py` (`fit()` delegates to the engine; grids move to tuning.py)
- Test: `tests/test_model_tuning.py` (new), `tests/test_model_fit.py`

**Interfaces produced (tuning.py):**
- Grid constants and `SIGMA_GRID`, `HALF_LIVES`, `TIE_TOLERANCE`: exact values from Global Constraints. fit.py re-exports the grids for compatibility.
- `season_weights(seasons: tuple[str, ...], half_life: float | None) -> dict[str, float]`
- `Job(seasons: tuple[str, ...], half_life: float | None)` (frozen, hashable)
- `Choice(params: RatingParams, coefficients: Coefficients, sigma: float, games: int)` (frozen)
- `choose(games: list[fit._Game], jobs: list[Job]) -> dict[Job, Choice]`
- `spreads_and_outcomes(games, params, coefficients, seasons) -> list[tuple[_Game, float, bool]]`: pre-game spreads for the `002` games in `seasons`.

- [ ] **Step 1: Tests first** (`tests/test_model_tuning.py`)

```python
import pytest

from predictor.model import tuning


def test_equal_weight_is_one_everywhere():
    assert tuning.season_weights(("a", "b", "c"), None) == {"a": 1.0, "b": 1.0, "c": 1.0}


def test_half_life_three_by_hand():
    w = tuning.season_weights(("a", "b", "c", "d"), 3.0)
    # newest 'd' age 0 -> 1.0; 'a' age 3 -> 0.5; 'b' age 2 -> 0.5**(2/3)
    assert w["d"] == 1.0
    assert w["a"] == pytest.approx(0.5)
    assert w["b"] == pytest.approx(0.5 ** (2 / 3))


def test_half_life_one_by_hand():
    w = tuning.season_weights(("a", "b", "c"), 1.0)
    assert w == pytest.approx({"a": 0.25, "b": 0.5, "c": 1.0})


def test_grids_are_exact():
    assert tuning.GRID_K == (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20)
    assert tuning.GRID_CAP == (15.0, 20.0, 25.0, 30.0, 40.0)
    assert tuning.GRID_REGRESSION == (0.0, 0.1, 0.2, 0.33, 0.5, 0.66)
    assert tuning.GRID_WINDOW == (400, 800, 1230)
    assert tuning.HALF_LIVES == (None, 3.0, 1.0)
    assert tuning.TIE_TOLERANCE == 0.001
```

Plus fixture-archive tests. Reuse `_history` from tests/test_model_fit.py by moving it into `tests/model_fixtures.py` as `build_history(con, games_per_season=40)`, and import it in both files.

1. `choose` is deterministic: two calls give equal dicts.
2. `choose` with two jobs over different seasons gives the same result as calling `choose` separately for each job. This proves the shared pass does not mix jobs.
3. A job's result ignores games outside its seasons. Corrupt the scores of a season NOT in `job.seasons` that comes *after* all of them, and the Choice must be identical. A later season can't affect ratings before it, so this is a real check.
4. Weighting matters. Construct a fixture where an old season's games are planted with a huge home margin (e.g. 40 points). Then `Choice` for `Job(S, 1.0)` must have a smaller fitted home-court influence than `Job(S, None)`. Make this concrete: compare the mean predicted spread over that old season's home games, which should be higher under equal weight. If this proves too fiddly, instead assert that the per-job weighted MSE minimiser differs (`choice_none != choice_hl1`) on a fixture built so that the two weightings disagree, and explain the fixture in a comment.
5. `Job` with seasons that contain no `002` games raises `fit.FitError` with a plain message.

- [ ] **Step 2: Implement `tuning.py`**

```python
"""Shared settings-selection engine for fit-model and evaluate-model (spec 3,
"Calibration redesign — decided 2026-10-02").

A rating-grid combination fixes every game's pre-game rating gap and home
court, whatever seasons are later used to fit the adjustments. So each
combination is simulated ONCE over all games, and every job (a set of
seasons plus a recency weighting) is scored from that single simulation.
This turns ~16,000 simulations into 900.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

import numpy as np

from predictor.model import adjustments as adj
from predictor.model.adjustments import Coefficients
from predictor.model.fit import FitError, _Game, _simulate
from predictor.model.ratings import RatingParams, win_probability

GRID_K = (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20)
GRID_CAP = (15.0, 20.0, 25.0, 30.0, 40.0)
GRID_REGRESSION = (0.0, 0.1, 0.2, 0.33, 0.5, 0.66)
GRID_WINDOW = (400, 800, 1230)
SIGMA_GRID = tuple(i / 100 for i in range(800, 2001, 5))
HALF_LIVES = (None, 3.0, 1.0)
TIE_TOLERANCE = 0.001


def season_weights(seasons, half_life):
    n = len(seasons)
    if half_life is None:
        return {s: 1.0 for s in seasons}
    return {s: 0.5 ** ((n - 1 - i) / half_life) for i, s in enumerate(seasons)}


@dataclass(frozen=True)
class Job:
    seasons: tuple[str, ...]
    half_life: float | None


@dataclass(frozen=True)
class Choice:
    params: RatingParams
    coefficients: Coefficients
    sigma: float
    games: int


def _job_rows(games, job):
    weights = season_weights(job.seasons, job.half_life)
    idx = [
        i for i, g in enumerate(games)
        if g.result.season in weights and g.result.game_id.startswith("002")
    ]
    if not idx:
        raise FitError(
            f"no regular-season results in {', '.join(job.seasons)} to fit on. "
            "Run 'predictor ingest-season <season>' for each"
        )
    w = np.array([weights[games[i].result.season] for i in idx])
    return np.array(idx), w


def choose(games: list[_Game], jobs: list[Job]) -> dict[Job, Choice]:
    X_all = np.array([g.x for g in games], dtype=float).reshape(len(games), 5)
    margin = np.array(
        [g.result.home_points - g.result.away_points for g in games], dtype=float
    )
    rows = {job: _job_rows(games, job) for job in jobs}
    best: dict[Job, tuple] = {}
    for k, cap, reg, window in product(GRID_K, GRID_CAP, GRID_REGRESSION, GRID_WINDOW):
        params = RatingParams(k, cap, reg, window)
        pre = np.array(_simulate(params, games), dtype=float)
        base = margin - pre[:, 0] - pre[:, 1]
        for job in jobs:
            idx, w = rows[job]
            X, y = X_all[idx], base[idx]
            sw = np.sqrt(w)
            coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
            mse = float(np.sum(w * (y - X @ coef) ** 2) / np.sum(w))
            if job not in best or mse < best[job][0]:
                best[job] = (mse, params, coef)

    out: dict[Job, Choice] = {}
    sims: dict[RatingParams, np.ndarray] = {}
    for job in jobs:
        _, params, coef = best[job]
        coefficients = Coefficients(*(round(float(c), 6) for c in coef))
        if params not in sims:
            sims[params] = np.array(_simulate(params, games), dtype=float)
        pre = sims[params]
        idx, w = rows[job]
        spreads = [
            pre[i, 0] + pre[i, 1] + sum(adj.astuple_terms(coefficients, games[i].x))
            for i in idx
        ]
        won = [games[i].result.home_points > games[i].result.away_points for i in idx]
        best_sigma = None
        for sigma in SIGMA_GRID:
            loss = 0.0
            for spread, h, wt in zip(spreads, won, w):
                p = min(max(win_probability(spread, sigma), 1e-12), 1 - 1e-12)
                loss -= wt * math.log(p if h else 1 - p)
            if best_sigma is None or loss < best_sigma[0]:
                best_sigma = (loss, sigma)
        out[job] = Choice(params, coefficients, best_sigma[1], len(idx))
    return out


def spreads_and_outcomes(games, params, coefficients, seasons):
    pre = _simulate(params, games)
    out = []
    for g, (gap, hc) in zip(games, pre):
        if g.result.season in seasons and g.result.game_id.startswith("002"):
            spread = gap + hc + sum(adj.astuple_terms(coefficients, g.x))
            out.append((g, spread, g.result.home_points > g.result.away_points))
    return out
```

Notes:
- `neutral` games: `_simulate` already returns home court 0 for them.
- The sigma loop is the hot path (241 × n games). For a ~8,000-game job that is about 2M `win_probability` calls, roughly 2–4 s per job. That is acceptable. If profiling shows more than 10 s per job, vectorise with numpy (`0.5 * (1 + erf(...))` via `math.erf` mapped over an array, or `scipy` if already installed — check before adding any dependency).

- [ ] **Step 3: `fit.fit` delegates**

`fit(con) -> ModelSettings` is the Task 1 behaviour on the new engine, with the equal-weight job only. Task 3 switches it to the walk-forward winner.

```python
def fit(con) -> ModelSettings:
    from predictor.model import tuning
    venues = VenueIndex.from_db(con)
    games = _load(con, venues)
    job = tuning.Job(TUNING_SEASONS, None)
    c = tuning.choose(games, [job])[job]
    return ModelSettings(c.params, c.coefficients, c.sigma, None, c.games)
```

- Delete the old grid loop and `_residuals` from fit.py if they are now unused.
- Re-export the grids: `from predictor.model.tuning import GRID_K, GRID_CAP, GRID_REGRESSION, GRID_WINDOW, SIGMA_GRID`. Avoid a circular import: tuning imports `_Game`/`_simulate`/`FitError` from fit at module level, so fit must import tuning lazily (inside functions) or re-export via a function-level import. If the re-export can't be done cleanly, update tests to import the grids from `tuning`.
- Update `test_fit_counts_its_games_and_picks_values_from_the_grids` to use the tuning grids.

- [ ] **Step 4: Run, regenerate, commit**

```bash
uv run pytest tests/test_model_tuning.py tests/test_model_fit.py -q
uv run predictor fit-model
uv run pytest -q
uv run pytest -m slow -q
git add -A src/predictor/model/ tests/
git commit -m "feat: shared settings-selection engine with recency weights and a wider grid

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Report `fit-model`'s output and its wall-clock time; it should be at most about 3 minutes. Over 6 minutes, report DONE_WITH_CONCERNS with a profile.

---

### Task 3: Walk-forward evaluation, `evaluate-model`, and the winner-based `fit-model`

**Files:**
- Create: `src/predictor/model/evaluate.py`
- Modify: `src/predictor/model/fit.py` (`fit` returns the evaluation's final settings), `src/predictor/cli.py` (new `evaluate-model`; `fit-model` prints the variant summary), `src/predictor/backtest/report.py` (`format_publishing_bar(bar, season_word="test season")`)
- Test: `tests/test_model_evaluate.py` (new), `tests/test_backtest_report.py` (season_word)

**Interfaces produced (evaluate.py):**
- `WALK_FORWARD_SEASONS = TUNING_SEASONS[1:]`
- `fold_job(season: str, half_life) -> tuning.Job`: the seasons strictly before `season` within TUNING_SEASONS. Raises ValueError if `season` is not in WALK_FORWARD_SEASONS.
- `VariantResult(half_life, predictions: list[Prediction], log_loss: float, brier: float, accuracy: float, choices: dict[str, tuning.Choice])` (frozen; `choices` keyed by predicted season)
- `Evaluation(variants: tuple[VariantResult, ...], winner: VariantResult, final: ModelSettings)` (frozen)
- `evaluate(con) -> Evaluation`
- `format_evaluation(ev: Evaluation) -> str`

**Rules:**
- Jobs:
  - `fold_job(S, hl)` for every S in WALK_FORWARD_SEASONS and hl in HALF_LIVES (18 jobs)
  - plus `Job(TUNING_SEASONS, hl)` for each hl (3 final jobs)
  - All 21 jobs are solved in ONE `tuning.choose` call.
- Guard: before predicting season S, assert every season in the fold job's seasons is `< S`, comparing strings, which works for "YYYY-YY". Otherwise raise `RuntimeError("walk-forward fold for S includes S or a later season")`.
- Predictions for season S under variant hl:
  - Take the `spreads_and_outcomes` (with that fold's Choice) restricted to S.
  - `p = win_probability(spread, choice.sigma)`.
  - Build `backtest.replay.Prediction` objects with `tipoff = cutoff = datetime.combine(game_date, time(0), tzinfo=UTC)` (unused by the metrics; comment it) and `reconstructed=True`.
- Metrics per variant: `metrics.log_loss`, `metrics.brier_score`, `metrics.accuracy` over all walk-forward predictions.
- Winner:
  - Start with `variants[0]` (equal weight).
  - For each later variant in HALF_LIVES order, replace the current best only if `loss < best.log_loss - TIE_TOLERANCE`.
- Final: `ModelSettings(choice.params, choice.coefficients, choice.sigma, winner.half_life, choice.games)`, from `Job(TUNING_SEASONS, winner.half_life)`.
- `fit.fit(con)` returns `evaluate(con).final`. `fit-model` prints `describe(...)` plus one line per variant (log loss, Brier, accuracy) and "Chosen: …".

**`format_evaluation` output (exact structure):**

```
  Walk-forward evaluation -- each season predicted with settings chosen only on earlier seasons
  (method designed on 2026-10-02 after a first look at 2023-26; the clean test is 2026-27, predicted live)

  Recency variant          log loss   Brier    accuracy   (N games, 2020-21 to 2025-26)
    equal weight            0.xxxx   0.xxxx    xx.x%
    half-life 3 seasons     0.xxxx   0.xxxx    xx.x%
    half-life 1 season      0.xxxx   0.xxxx    xx.x%
  Chosen: <label> (lowest log loss; a later variant must win by more than 0.001)

  By season (chosen variant; settings chosen on earlier seasons only):
    2020-21  k 0.xx cap xx reg 0.xx window xxx sigma xx.xx   model xx.x%   home xx.x%   Brier 0.xxxx
    ...
  <calibration table: the same 10-bucket "said / actual / games" format the backtest report uses>
  <publishing-bar block from report.format_publishing_bar(..., season_word="evaluated season")>

  Final settings for 2026-27 (all seven tuning seasons, <label>):
    k … cap … regression … window … sigma …
```

Reuse `metrics.calibration_bins` and the calibration-row formatting from report.py. If that formatting is inline in `format_report`, extract it to `report.format_calibration_table(bins)` without changing the backtest's output, and add a test pinning one row's exact text.

`format_publishing_bar` gains `season_word: str = "test season"`. It replaces "test season" in its lines with that word; plural "test seasons" becomes `season_word + "s"`. The default output is unchanged, and the existing tests still pass.

**Tests (`tests/test_model_evaluate.py`, fixture archive via `build_history`):**
1. `fold_job("2024-25", None).seasons == ("2019-20", ..., "2023-24")`, and `fold_job("2019-20", None)` raises ValueError.
2. **No future leakage:** corrupt every 2024-25 FINAL score in the fixture (e.g. 200–1). The fold Choices for 2020-21 … 2024-25 must be unchanged, and the 2024-25 predictions' `p_home` must be unchanged. The 2025-26 fold Choice must change. This proves a season is predicted with settings that never saw it.
3. **Winner rule by hand:** construct `VariantResult`s with log losses (0.6500, 0.6495, 0.6480). The winner is the third: 0.6480 < 0.6500 − 0.001. With (0.6500, 0.6495, 0.6492), the winner is the first. Test via a small pure helper `pick_winner(variants)`.
4. `evaluate` is deterministic, and `final.half_life == winner.half_life`.
5. The `format_evaluation` text contains each variant label, "Chosen:", every walk-forward season, "Final settings for 2026-27", and the publishing-bar block with "evaluated season".
6. CLI `evaluate-model`, on a tmp archive built with `build_history`, exits 0 and prints the evaluation. `fit-model` saves settings whose `half_life` equals the printed winner.
7. A real-archive `@pytest.mark.slow` test: `evaluate(con).final == ms.load()` (reproducibility).

- [ ] **Step 1:** Write the tests above (RED).
- [ ] **Step 2:** Implement evaluate.py, the fit.py change, the report.py `season_word` and calibration-table extraction, and the CLI. `evaluate-model` opens the database read-only, catches `duckdb.Error` and `fit.FitError` with plain messages, and prints a per-stage line to stderr ("simulating 900 rating settings...", "fitting 21 season sets...").
- [ ] **Step 3:** `uv run pytest -q` (fast). Run `uv run predictor fit-model` and record its output and time. Run `uv run pytest -m slow -q`.
- [ ] **Step 4:** Commit: `feat: walk-forward evaluation picks the recency variant; fit-model saves its final settings` + trailer.

---

### Task 4: Run, record, decide (controller)

- [ ] `uv run predictor evaluate-model > <scratch>/walk-forward.txt` and read it.
- [ ] Record in the spec a dated subsection **"Walk-forward result — <date>"** under the calibration redesign. Include:
  - the variant table
  - the chosen variant and the final settings
  - per-season accuracy against home
  - the publishing-bar verdict, stated plainly
  - the reminder that 2026-27 live is the clean test
- [ ] Update memory `project_prediction_core.md`.
- [ ] Whole-branch review, then the finishing skill (merge needs the user's yes).
