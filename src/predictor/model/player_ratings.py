"""Point-in-time player ratings, per 48 minutes, relative to the league.

Each player carries two ratings, kept side by side so the walk-forward can
pick between them:

- ``"box"``: Hollinger game score per 48 minutes, minus the league's
  cumulative minutes-weighted mean of that quantity over every row applied
  before the current game.
- ``"pm"``: plus-minus per 48 minutes, minus his team's final margin, i.e.
  how much better the team did with him on the floor than overall.

Both are minutes-weighted EWMAs: a game of ``m`` minutes keeps
``0.5 ** (m / half_life)`` of the old rating. Ratings are keyed by player id,
never by team, so a traded player brings his rating with him. Every rating
regresses toward 0 by ``season_regression`` when a new season is first seen.

The book is point-in-time only if the caller applies games in date order
and reads a rating for a game on date D before applying any game on D.
Rows must already exclude preseason (see
``player_games.rating_rows_sql``): preseason never feeds ratings or minutes.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import Literal

from predictor.sources.player_games import PlayerGameRow

HALF_LIFE_GRID: tuple[float, ...] = (500.0, 1500.0, 4000.0)
REGRESSION_GRID: tuple[float, ...] = (0.2, 0.4)
ROOKIE_PRIOR = -2.0
OBS_CLIP = 60.0
MIN_MINUTES = 1.0
RECENT_GAMES = 10

Formula = Literal["box", "pm"]
FORMULAS: tuple[Formula, ...] = ("box", "pm")


@dataclass(frozen=True)
class PlayerParams:
    half_life_minutes: float
    season_regression: float


def _n(value: int | float | None) -> float:
    return 0.0 if value is None else float(value)


def game_score(row: PlayerGameRow) -> float:
    """Hollinger game score. A missing stat counts as 0."""
    return (
        _n(row.pts)
        + 0.4 * _n(row.fgm)
        - 0.7 * _n(row.fga)
        - 0.4 * (_n(row.fta) - _n(row.ftm))
        + 0.7 * _n(row.oreb)
        + 0.3 * _n(row.dreb)
        + _n(row.stl)
        + 0.7 * _n(row.ast)
        + 0.7 * _n(row.blk)
        - 0.4 * _n(row.pf)
        - _n(row.tov)
    )


def ewma_update(r: float, obs: float, minutes: float, half_life: float) -> float:
    lam = 0.5 ** (minutes / half_life)
    clipped = max(-OBS_CLIP, min(OBS_CLIP, obs))
    return lam * r + (1.0 - lam) * clipped


def _played(row: PlayerGameRow) -> bool:
    return row.minutes is not None and row.minutes >= MIN_MINUTES


class PlayerBook:
    """Running player ratings, recent minutes and team rotations."""

    def __init__(self, params: PlayerParams) -> None:
        self.params = params
        self._ratings: dict[int, dict[str, float]] = {}
        self._minutes: dict[int, deque[float]] = {}
        self._last_team: dict[int, str] = {}
        self._team_games: dict[str, deque[set[int]]] = {}
        self._seasons: set[str] = set()
        self._box_sum = 0.0  # sum of 48 * game_score over applied rows
        self._minutes_sum = 0.0  # sum of minutes over applied rows
        self._last_date: date | None = None

    # -- seasons -----------------------------------------------------------
    def enter_season(self, season: str) -> None:
        """Regress every rating toward 0 the first time ``season`` is seen."""
        if season in self._seasons:
            return
        self._seasons.add(season)
        keep = 1.0 - self.params.season_regression
        for ratings in self._ratings.values():
            for formula in ratings:
                ratings[formula] *= keep

    # -- updates -----------------------------------------------------------
    @property
    def league_box_mean(self) -> float:
        if self._minutes_sum <= 0.0:
            return 0.0
        return self._box_sum / self._minutes_sum

    def apply_game(self, rows: Iterable[PlayerGameRow], margins: dict[str, int]) -> None:
        """Fold one game's player rows into the book.

        ``margins`` maps each team to its final margin. Rows with no minutes
        or under one minute are skipped entirely.
        """
        rows = list(rows)
        if not rows:
            return
        seasons = {row.season for row in rows}
        if len(seasons) != 1:
            raise ValueError(f"one game's rows span several seasons: {sorted(seasons)}")
        game_date = max(row.game_date for row in rows)
        if self._last_date is not None and game_date < self._last_date:
            raise ValueError(
                f"games must be applied in date order: {game_date} after {self._last_date}"
            )
        self._last_date = game_date
        self.enter_season(seasons.pop())

        league_mean = self.league_box_mean
        half_life = self.params.half_life_minutes
        lineups: dict[str, set[int]] = {}
        box_sum = 0.0
        minutes_sum = 0.0
        for row in rows:
            if not _played(row):
                continue
            minutes = float(row.minutes)  # type: ignore[arg-type]
            per48 = 48.0 * game_score(row) / minutes
            ratings = self._ratings.setdefault(
                row.player_id, {f: ROOKIE_PRIOR for f in FORMULAS}
            )
            ratings["box"] = ewma_update(ratings["box"], per48 - league_mean, minutes, half_life)
            if row.plus_minus is not None and row.team in margins:
                pm_obs = 48.0 * row.plus_minus / minutes - margins[row.team]
                ratings["pm"] = ewma_update(ratings["pm"], pm_obs, minutes, half_life)
            box_sum += 48.0 * game_score(row)
            minutes_sum += minutes
            self._minutes.setdefault(row.player_id, deque(maxlen=RECENT_GAMES)).append(minutes)
            self._last_team[row.player_id] = row.team
            lineups.setdefault(row.team, set()).add(row.player_id)
        # The league mean moves only after the whole game is applied, so
        # every row in a game is compared with the same earlier-games mean.
        self._box_sum += box_sum
        self._minutes_sum += minutes_sum
        for team, players in lineups.items():
            self._team_games.setdefault(team, deque(maxlen=RECENT_GAMES)).append(players)

    # -- reads -------------------------------------------------------------
    def rating(self, player_id: int, formula: Formula) -> float:
        ratings = self._ratings.get(player_id)
        if ratings is None:
            return ROOKIE_PRIOR
        return ratings[formula]

    def recent_minutes(self, player_id: int) -> list[float]:
        return list(self._minutes.get(player_id, ()))

    def last_team(self, player_id: int) -> str | None:
        return self._last_team.get(player_id)

    def team_recent_games(self, team: str) -> list[set[int]]:
        return [set(players) for players in self._team_games.get(team, ())]
