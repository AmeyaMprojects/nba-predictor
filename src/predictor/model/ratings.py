"""Team ratings in points, updated on margin of victory (spec 3, Stage 1).

A rating gap of 4 means "4 points better on a neutral floor". After each
game both teams move by K x (capped actual margin - predicted margin), where
predicted = rating gap + home court. Rest, travel and altitude are NOT part
of the update: they are small, and leaving them out keeps ratings
independent of the fitted adjustment coefficients.

Home court (owner decision 2026-10-25): the league value is the mean home
margin over the last `hca_window` non-neutral games, as before. Each team
also has its own edge: the mean of its home residuals minus the mean of its
away residuals (residual = margin - rating gap - league value, with the
altitude allowance taken out at altitude games so altitude is not counted
twice), over its last `HomeParams.window` games per side, shrunk toward 0
by `n_eff / (n_eff + shrinkage)`. Infinite shrinkage (`FLAT_HOME`, the
default) is the flat league value, byte-identical to the old model. Every
residual is recorded by `apply`, so a team's edge read before a game on
date D uses only results already applied (games before D).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from collections.abc import Sequence
from datetime import date


@dataclass(frozen=True)
class RatingParams:
    k: float
    margin_cap: float
    season_regression: float
    hca_window: int


TEAM_HOME_WINDOW = 164
SHRINKAGE_GRID = (math.inf, 400.0, 150.0, 50.0)


@dataclass(frozen=True)
class HomeParams:
    shrinkage: float
    altitude_allowance: float = 0.0
    window: int = TEAM_HOME_WINDOW

    def __post_init__(self) -> None:
        if not (self.shrinkage >= 0):  # also rejects NaN
            raise ValueError(f"shrinkage must be >= 0 or inf, got {self.shrinkage}")
        if not math.isfinite(self.altitude_allowance):
            raise ValueError(f"altitude_allowance must be finite, got {self.altitude_allowance}")
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window}")

    @property
    def flat(self) -> bool:
        return self.shrinkage == math.inf


FLAT_HOME = HomeParams(math.inf)


def shrunk_deviation(home: Sequence[float], away: Sequence[float], shrinkage: float) -> float:
    """(mean(home) - mean(away)) * n_eff / (n_eff + shrinkage), where
    n_eff = 1 / (1/n_home + 1/n_away). 0.0 with no evidence on either side
    or infinite shrinkage."""
    if not home or not away or shrinkage == math.inf:
        return 0.0
    raw = sum(home) / len(home) - sum(away) / len(away)
    n_eff = 1.0 / (1.0 / len(home) + 1.0 / len(away))
    return raw * n_eff / (n_eff + shrinkage)


@dataclass(frozen=True)
class TeamHomeSummary:
    team: str
    home_games: int
    away_games: int
    raw: float
    deviation: float


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
    def __init__(self, params: RatingParams, home: HomeParams = FLAT_HOME) -> None:
        self.params = params
        self.home = home
        self._ratings: dict[str, float] = {}
        self._season: str | None = None
        self._home_margins: deque[int] = deque(maxlen=params.hca_window)
        # Per team: residuals of its home games, and negated residuals of
        # its away games (so both read "how much better than expected the
        # team did at that venue type").
        self._team_home: dict[str, deque[float]] = {}
        self._team_away: dict[str, deque[float]] = {}

    def rating(self, team: str) -> float:
        return self._ratings.get(team, 0.0)

    def home_court(self, team: str | None = None) -> float:
        """With no team: mean home margin over the most recent `hca_window`
        non-neutral games (the league value). With a team: the league value
        plus that team's shrunk edge."""
        if not self._home_margins:
            league = 0.0
        else:
            league = sum(self._home_margins) / len(self._home_margins)
        if team is None or self.home.flat:
            return league
        return league + self.deviation(team)

    def _sides(self, team: str) -> tuple[Sequence[float], Sequence[float]]:
        return self._team_home.get(team, ()), self._team_away.get(team, ())

    def deviation(self, team: str) -> float:
        home, away = self._sides(team)
        return shrunk_deviation(home, away, self.home.shrinkage)

    def team_home_summary(self, team: str) -> TeamHomeSummary:
        home, away = self._sides(team)
        raw = shrunk_deviation(home, away, 0.0)
        return TeamHomeSummary(team, len(home), len(away), raw, self.deviation(team))

    def enter_season(self, season: str) -> None:
        """Regress every rating toward 0 the first time a new season is seen."""
        if self._season is not None and season != self._season:
            keep = 1.0 - self.params.season_regression
            self._ratings = {t: r * keep for t, r in self._ratings.items()}
        self._season = season

    def apply(self, result: Result, altitude_game: bool = False) -> None:
        self.enter_season(result.season)
        gap = self.rating(result.home_team) - self.rating(result.away_team)
        league = self.home_court()
        home_court = 0.0 if result.neutral else self.home_court(result.home_team)
        predicted = gap + home_court
        margin = result.home_points - result.away_points
        cap = self.params.margin_cap
        delta = self.params.k * (max(-cap, min(cap, margin)) - predicted)
        self._ratings[result.home_team] = self.rating(result.home_team) + delta
        self._ratings[result.away_team] = self.rating(result.away_team) - delta
        if not result.neutral:
            allowance = self.home.altitude_allowance if altitude_game else 0.0
            resid = margin - gap - league - allowance
            self._side(self._team_home, result.home_team).append(resid)
            self._side(self._team_away, result.away_team).append(-resid)
            self._home_margins.append(margin)

    def _side(self, sides: dict[str, deque[float]], team: str) -> deque[float]:
        d = sides.get(team)
        if d is None:
            d = sides[team] = deque(maxlen=self.home.window)
        return d


def win_probability(spread: float, sigma: float) -> float:
    """P(home wins) = Phi(spread / sigma)."""
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    return 0.5 * (1.0 + math.erf(spread / (sigma * math.sqrt(2.0))))
