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


def _validate_seasons_and_half_life(seasons: tuple[str, ...], half_life: float | None) -> None:
    if len(set(seasons)) != len(seasons) or tuple(seasons) != tuple(sorted(seasons)):
        raise ValueError(
            f"seasons must be strictly chronological (sorted, no duplicates), got {seasons!r}"
        )
    if half_life is not None and half_life <= 0:
        raise ValueError(f"half_life must be None or positive, got {half_life!r}")


def season_weights(seasons: tuple[str, ...], half_life: float | None) -> dict[str, float]:
    """Weight of each season in `seasons` (oldest -> newest). `None` gives
    equal weight 1.0 everywhere; otherwise the newest season is 1.0 and each
    season `half_life` seasons older is half the weight."""
    _validate_seasons_and_half_life(seasons, half_life)
    n = len(seasons)
    if half_life is None:
        return {s: 1.0 for s in seasons}
    return {s: 0.5 ** ((n - 1 - i) / half_life) for i, s in enumerate(seasons)}


@dataclass(frozen=True)
class Job:
    seasons: tuple[str, ...]
    half_life: float | None

    def __post_init__(self) -> None:
        _validate_seasons_and_half_life(self.seasons, self.half_life)


@dataclass(frozen=True)
class Choice:
    params: RatingParams
    coefficients: Coefficients
    sigma: float
    games: int


def _job_rows(games: list[_Game], job: Job):
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
    """Pick the grid combination, adjustment coefficients and sigma that
    minimise each job's weighted loss, from ONE pass over the grid.

    Within a job, strictly-lower MSE (and, for sigma, strictly-lower log
    loss) replaces the current best, so a tie keeps whichever combination
    `product`'s fixed iteration order saw first. TIE_TOLERANCE is not used
    here -- that is the walk-forward winner rule between jobs (Task 3).

    After the grid loop, each job's winning `RatingParams` is re-simulated
    once to compute its sigma -- cached per distinct winning `RatingParams`
    (`sims`), so when several jobs land on the same params that re-simulation
    happens only once, not once per job.
    """
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
        best_sigma = _best_sigma(spreads, won, w)
        out[job] = Choice(params, coefficients, best_sigma, len(idx))
    return out


def _best_sigma(spreads: list[float], won: list[bool], weights: np.ndarray) -> float:
    """Pick the sigma in SIGMA_GRID minimising weighted log loss.

    This is `ratings.win_probability` evaluated at every (game, sigma) pair,
    so it matches the predictor exactly (no erf approximation of its own).
    """
    best = None
    for sigma in SIGMA_GRID:
        loss = 0.0
        for spread, h, wt in zip(spreads, won, weights):
            p = min(max(win_probability(spread, sigma), 1e-12), 1 - 1e-12)
            loss -= wt * math.log(p if h else 1 - p)
        if best is None or loss < best[0]:
            best = (loss, sigma)
    return best[1]


def spreads_and_outcomes(games, params, coefficients, seasons):
    """Pre-game spreads and outcomes for the `002` games in `seasons`."""
    pre = _simulate(params, games)
    out = []
    for g, (gap, hc) in zip(games, pre):
        if g.result.season in seasons and g.result.game_id.startswith("002"):
            spread = gap + hc + sum(adj.astuple_terms(coefficients, g.x))
            out.append((g, spread, g.result.home_points > g.result.away_points))
    return out
