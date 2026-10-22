"""Optional gradient-boosted correction on top of the additive candidate.

A ``HistGradientBoostingRegressor`` learns the additive candidate's
walk-forward error (home margin minus additive spread) from the spread's own
pieces plus three extra point-in-time features. For each walk-forward season
S it is trained only on the walk-forward residuals of earlier walk-forward
seasons, and only when there are at least ``MIN_TRAINING_SEASONS`` of them
(otherwise S keeps its additive spread and sigma). Those residuals come from
models fitted strictly before each of those seasons, so nothing from S or
later reaches S's correction.

The model is trained on final-call rows and applied to both calls, each with
its own call's inputs. Sigma for S is re-chosen with ``tuning._best_sigma``
on the corrected training spreads (equal weights).

Training is deterministic (``random_state=0``) and the model is never
pickled: a saved setting records the training seasons and parameters, and
the model is retrained on load. The correction ships only if
``correction_helps``: pooled final-call Brier and log loss both strictly
lower, on the same games.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from predictor.backtest.replay import Prediction
from predictor.model import evaluate as evaluate_mod
from predictor.model import tuning
from predictor.model.adjustments import FEATURE_NAMES
from predictor.model.candidate import CandidateResult, GameParts, _variant
from predictor.model.fit import FitError
from predictor.model.player_sim import GameFeatures
from predictor.model.ratings import win_probability

EXTRA_FEATURES = ("recent_form_gap", "pace_sum", "continuity_gap")
CORRECTION_FEATURES = (
    ("additive_spread", "team_gap", "home_court")
    + FEATURE_NAMES
    + ("roster_gap", "injury_gap")
    + EXTRA_FEATURES
)
GBM_PARAMS = dict(
    max_depth=3,
    learning_rate=0.05,
    max_iter=200,
    min_samples_leaf=100,
    l2_regularization=1.0,
    random_state=0,
)
MIN_TRAINING_SEASONS = 2

_CONTINUITY_KEY = {"final": "continuity_gap", "early": "continuity_gap_early"}
_CALLS = (("final", "final_call"), ("early", "early_call"))


def feature_row(parts: GameParts, features: GameFeatures, call: str) -> tuple[float, ...]:
    """One game's correction inputs, in ``CORRECTION_FEATURES`` order.

    ``recent_form_gap`` is home minus away of (recent mean margin - team
    rating), i.e. the stored recent-margin gap minus the fold's rating gap.
    """
    ex = features.extras
    return (
        (parts.additive_spread, parts.team_gap, parts.home_court)
        + tuple(parts.x)
        + (parts.roster, parts.injury)
        + (
            ex["recent_margin_gap"] - parts.team_gap,
            ex["pace_sum"],
            ex[_CONTINUITY_KEY[call]],
        )
    )


@dataclass(frozen=True)
class Correction:
    model: HistGradientBoostingRegressor
    seasons: tuple[str, ...]

    @classmethod
    def train(
        cls,
        rows: Sequence[tuple[tuple[float, ...], float]],
        *,
        seasons: Sequence[str] = (),
        season: str | None = None,
    ) -> Correction:
        """Fit on (features, residual = margin - additive spread) rows.

        ``seasons`` are the seasons the rows come from and ``season`` the one
        the model will correct. Neither changes the fit; a training season on
        or after ``season`` is refused.
        """
        if season is not None and any(s >= season for s in seasons):
            raise FitError(
                f"internal error: the correction for {season} was given rows from "
                f"{season} or later; refusing to train"
            )
        X = np.array([r for r, _ in rows], dtype=float).reshape(len(rows), -1)
        y = np.array([res for _, res in rows], dtype=float)
        model = HistGradientBoostingRegressor(**GBM_PARAMS).fit(X, y)
        return cls(model, tuple(seasons))

    def predict(self, x: Sequence[float]) -> float:
        return float(self.predict_many([x])[0])

    def predict_many(self, xs: Sequence[Sequence[float]]) -> np.ndarray:
        X = np.array(xs, dtype=float).reshape(len(xs), -1)
        return self.model.predict(X)


def training_rows(
    result: CandidateResult, features: dict[str, GameFeatures], seasons: Sequence[str]
) -> list[tuple[tuple[float, ...], float]]:
    """Final-call (features, walk-forward residual) rows for ``seasons``."""
    wanted = set(seasons)
    parts = result.parts["final"]
    details = result.final_call.details
    rows = []
    for p in result.final_call.predictions:
        if p.season not in wanted:
            continue
        gp = parts[p.game_id]
        rows.append((
            feature_row(gp, features[p.game_id], "final"),
            float(details[p.game_id].margin) - gp.additive_spread,
        ))
    return rows


def apply_corrections(
    result: CandidateResult, features: dict[str, GameFeatures]
) -> CandidateResult | None:
    """The walk-forward result with each season's correction applied, for
    both calls; None when no season has ``MIN_TRAINING_SEASONS`` earlier
    walk-forward seasons to train on. Same games, same order."""
    seasons = sorted({p.season for p in result.final_call.predictions})
    # season -> (correction, sigma), only for seasons that get one
    fitted: dict[str, tuple[Correction, float]] = {}
    for season in seasons:
        earlier = tuple(s for s in seasons if s < season)
        if len(earlier) < MIN_TRAINING_SEASONS:
            continue
        rows = training_rows(result, features, earlier)
        model = Correction.train(rows, seasons=earlier, season=season)
        additive = np.array([r[0] for r, _ in rows])
        spreads = additive + model.predict_many([r for r, _ in rows])
        margins = additive + np.array([res for _, res in rows])
        sigma = tuning._best_sigma(
            [float(s) for s in spreads], [bool(m > 0) for m in margins], np.ones(len(rows))
        )
        fitted[season] = (model, sigma)
    if not fitted:
        return None

    out = {}
    for call, attr in _CALLS:
        base: evaluate_mod.VariantResult = getattr(result, attr)
        parts = result.parts[call]
        delta: dict[str, float] = {}
        for season, (model, _) in fitted.items():
            ids = [p.game_id for p in base.predictions if p.season == season]
            if ids:
                xs = [feature_row(parts[g], features[g], call) for g in ids]
                delta.update(zip(ids, (float(v) for v in model.predict_many(xs)), strict=True))
        predictions: list[Prediction] = []
        details: dict[str, evaluate_mod.GameDetail] = {}
        for p in base.predictions:
            d = base.details[p.game_id]
            if p.season in fitted:
                s = parts[p.game_id].additive_spread + delta[p.game_id]
                sigma = fitted[p.season][1]
                d = evaluate_mod.GameDetail(s, sigma, d.margin)
                p = dataclasses.replace(p, p_home=win_probability(s, sigma))
            predictions.append(p)
            details[p.game_id] = d
        out[attr] = _variant(base.half_life, predictions, details)
    return dataclasses.replace(result, **out)


def correction_helps(base: CandidateResult, corrected: CandidateResult) -> bool:
    """Pooled final-call Brier AND log loss both strictly lower, on the
    same games."""
    b, c = base.final_call, corrected.final_call
    b_ids = [p.game_id for p in b.predictions]
    c_ids = [p.game_id for p in c.predictions]
    if sorted(b_ids) != sorted(c_ids) or len(set(b_ids)) != len(b_ids):
        raise ValueError("the base and corrected results must cover the same games")
    return c.brier < b.brier and c.log_loss < b.log_loss
