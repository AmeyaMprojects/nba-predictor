"""Team ratings in points, updated on margin of victory (spec 3, Stage 1).

A rating gap of 4 means "4 points better on a neutral floor". After each
game both teams move by K x (capped actual margin - predicted margin), where
predicted = rating gap + home court. Rest, travel and altitude are NOT part
of the update: they are small, and leaving them out keeps ratings
independent of the fitted adjustment coefficients.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class RatingParams:
    k: float
    margin_cap: float
    season_regression: float
    hca_window: int


@dataclass(frozen=True)
class Result:
    game_id: str
    season: str
    game_date: date
    home_team: str
    away_team: str
    home_points: int
    away_points: int
    neutral: bool


class Ratings:
    def __init__(self, params: RatingParams) -> None:
        self.params = params
        self._ratings: dict[str, float] = {}
        self._season: str | None = None
        self._home_margins: deque[int] = deque(maxlen=params.hca_window)

    def rating(self, team: str) -> float:
        return self._ratings.get(team, 0.0)

    def home_court(self) -> float:
        """Mean home margin over the most recent `hca_window` non-neutral games."""
        if not self._home_margins:
            return 0.0
        return sum(self._home_margins) / len(self._home_margins)

    def enter_season(self, season: str) -> None:
        """Regress every rating toward 0 the first time a new season is seen."""
        if self._season is not None and season != self._season:
            keep = 1.0 - self.params.season_regression
            self._ratings = {t: r * keep for t, r in self._ratings.items()}
        self._season = season

    def apply(self, result: Result) -> None:
        self.enter_season(result.season)
        home_court = 0.0 if result.neutral else self.home_court()
        predicted = (
            self.rating(result.home_team) - self.rating(result.away_team) + home_court
        )
        margin = result.home_points - result.away_points
        cap = self.params.margin_cap
        delta = self.params.k * (max(-cap, min(cap, margin)) - predicted)
        self._ratings[result.home_team] = self.rating(result.home_team) + delta
        self._ratings[result.away_team] = self.rating(result.away_team) - delta
        if not result.neutral:
            self._home_margins.append(margin)


def win_probability(spread: float, sigma: float) -> float:
    """P(home wins) = Phi(spread / sigma)."""
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    return 0.5 * (1.0 + math.erf(spread / (sigma * math.sqrt(2.0))))
