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
