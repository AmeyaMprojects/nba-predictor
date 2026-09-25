from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

from predictor.asof import AsOfView


class PredictionError(Exception):
    """Raised when a predictor returns something unusable."""


@dataclass(frozen=True)
class GameToPredict:
    """A game the harness is asking about, with no result attached.

    Deliberately carries no score: a predictor cannot leak what it is never
    handed.

    FIX 4 (final review, part 1): does NOT carry `tipoff`. It used to, but
    that was both false advertising and a leak surface: the value is
    derived from the LATEST injury-report vintage available at cutoff
    time, which is a post-hoc quantity (8 (game_date, team) pairs in the
    real archive disagree across vintages -- see `tipoff.py`), and it is
    redundant besides -- a predictor holding `view.as_of` (the cutoff) and
    the harness's buffer can already recover `tip = view.as_of + buffer`
    exactly, since `cutoff = tip - buffer`. Removing the field is
    structural: there is no longer a post-hoc value here to (re)introduce
    a leak through.
    """

    game_id: str
    season: str
    game_date: date
    home_team: str
    away_team: str


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
