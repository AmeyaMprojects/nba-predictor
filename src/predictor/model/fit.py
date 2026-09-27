"""Choose every Stage 1 setting from past seasons only (spec 3).

Fit seasons choose the rating settings and adjustment sizes; the calibrate
season chooses only sigma; test seasons are never read. The fit reads the
games table directly -- it trains on completed past seasons and is not a
prediction path -- and asserts that no test season was loaded.

Simulation mirrors Stage1Predictor exactly: a date's results are applied
only after every game on that date has been given its pre-game numbers,
because in the harness a result becomes visible the day after it is played.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from itertools import groupby, product

import numpy as np

from predictor import db
from predictor.model import adjustments as adj
from predictor.model.adjustments import Coefficients
from predictor.model.ratings import RatingParams, Ratings, Result, win_probability
from predictor.model.settings import (
    CALIBRATE_SEASON,
    FIT_SEASONS,
    TEST_SEASONS,
    WARMUP_SEASONS,
    ModelSettings,
)
from predictor.model.venues import COMPETITIVE_PREFIXES, VenueIndex

GRID_K = (0.04, 0.06, 0.08, 0.10, 0.12, 0.15)
GRID_CAP = (15.0, 20.0, 25.0, 30.0)
GRID_REGRESSION = (0.2, 0.33, 0.5, 0.66)
GRID_WINDOW = (400, 800, 1230)
SIGMA_GRID = tuple(i / 100 for i in range(800, 2001, 5))


class FitError(Exception):
    """Not enough history to fit."""


@dataclass(frozen=True)
class _Game:
    result: Result
    x: tuple[float, float, float, float, float]


def _load(con, venues: VenueIndex) -> list[_Game]:
    seasons = WARMUP_SEASONS + FIT_SEASONS + (CALIBRATE_SEASON,)
    table = db.POINT_IN_TIME_TABLES["games"]
    prefixes = ", ".join(f"'{p}'" for p in COMPETITIVE_PREFIXES)
    placeholders = ", ".join("?" for _ in seasons)
    rows = con.execute(
        f"SELECT game_id, season, game_date, home_team, away_team, home_points, away_points "
        f"FROM {table} WHERE status = 'FINAL' AND home_points IS NOT NULL "
        f"AND away_points IS NOT NULL AND substr(game_id, 1, 3) IN ({prefixes}) "
        f"AND season IN ({placeholders}) "
        "QUALIFY row_number() OVER (PARTITION BY game_id ORDER BY observed_at DESC) = 1 "
        "ORDER BY game_date, game_id",
        list(seasons),
    ).fetchall()
    games: list[_Game] = []
    for gid, season, gd, home, away, hp, ap in rows:
        assert season not in TEST_SEASONS, "fit must never read a test season"
        venue = venues.venue(gid)
        city = venue.city if venue else None
        neutral = venue.is_neutral if venue else False
        x = adj.feature_vector(
            adj.situation(venues, home, gd, city),
            adj.situation(venues, away, gd, city),
            adj.is_altitude_game(city, neutral),
        )
        games.append(_Game(Result(gid, season, gd, home, away, hp, ap, neutral), x))
    return games


def _simulate(params: RatingParams, games: list[_Game]) -> list[tuple[float, float]]:
    """Pre-game (rating gap, home court) for every game, in input order."""
    ratings = Ratings(params)
    out: list[tuple[float, float]] = []
    for _, day in groupby(games, key=lambda g: g.result.game_date):
        day = list(day)
        for g in day:
            ratings.enter_season(g.result.season)
            r = g.result
            gap = ratings.rating(r.home_team) - ratings.rating(r.away_team)
            out.append((gap, 0.0 if r.neutral else ratings.home_court()))
        for g in day:
            ratings.apply(g.result)
    return out


def _residuals(games, pre, seasons):
    rows = [
        (g, gap, hc)
        for g, (gap, hc) in zip(games, pre)
        if g.result.season in seasons and g.result.game_id.startswith("002")
    ]
    X = np.array([g.x for g, _, _ in rows], dtype=float).reshape(len(rows), 5)
    y = np.array(
        [g.result.home_points - g.result.away_points - gap - hc for g, gap, hc in rows],
        dtype=float,
    )
    return rows, X, y


def fit(con) -> ModelSettings:
    venues = VenueIndex.from_db(con)
    games = _load(con, venues)
    fit_count = sum(
        1 for g in games if g.result.season in FIT_SEASONS and g.result.game_id.startswith("002")
    )
    cal_count = sum(
        1 for g in games
        if g.result.season == CALIBRATE_SEASON and g.result.game_id.startswith("002")
    )
    if fit_count == 0 or cal_count == 0:
        raise FitError(
            "not enough history to fit the model: the fit seasons "
            f"({', '.join(FIT_SEASONS)}) and the calibrate season ({CALIBRATE_SEASON}) "
            "must have results. Run 'predictor ingest-season <season>' for each."
        )

    best = None
    for k, cap, reg, window in product(GRID_K, GRID_CAP, GRID_REGRESSION, GRID_WINDOW):
        params = RatingParams(k, cap, reg, window)
        pre = _simulate(params, games)
        _, X, y = _residuals(games, pre, FIT_SEASONS)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        mse = float(np.mean((y - X @ coef) ** 2))
        if best is None or mse < best[0]:
            best = (mse, params, coef)
    _, params, coef = best
    coefficients = Coefficients(*(round(float(c), 6) for c in coef))

    pre = _simulate(params, games)
    rows, X, _ = _residuals(games, pre, (CALIBRATE_SEASON,))
    spreads = [
        gap + hc + sum(adj.astuple_terms(coefficients, g.x))
        for (g, gap, hc) in rows
    ]
    outcomes = [g.result.home_points > g.result.away_points for g, _, _ in rows]
    best_sigma = None
    for sigma in SIGMA_GRID:
        loss = 0.0
        for spread, won in zip(spreads, outcomes):
            p = min(max(win_probability(spread, sigma), 1e-12), 1 - 1e-12)
            loss -= math.log(p if won else 1 - p)
        if best_sigma is None or loss < best_sigma[0]:
            best_sigma = (loss, sigma)

    return ModelSettings(
        ratings=params,
        coefficients=coefficients,
        sigma=best_sigma[1],
        fit_games=fit_count,
        calibrate_games=cal_count,
    )


def describe(s: ModelSettings) -> str:
    r, c = s.ratings, s.coefficients
    return "\n".join([
        f"Ratings move {r.k * 100:.0f}% of each game's surprise; blowouts count as at "
        f"most {r.margin_cap:.0f} points.",
        f"Each new season, ratings fall back {r.season_regression * 100:.0f}% toward average.",
        f"Home court is the average home margin over the last {r.hca_window} games.",
        "Adjustments, in points for the home team:",
        f"  back-to-back (home minus away)       {c.back_to_back:+.1f}",
        f"  third game in four nights            {c.third_in_four:+.1f}",
        f"  per 1,000 km travelled               {c.travel_per_1000km:+.1f}",
        f"  per time zone crossed                {c.tz_per_hour:+.1f}",
        f"  playing at altitude (Denver, Utah)   {c.altitude:+.1f}",
        f"Typical game-to-game spread (sigma): {s.sigma:.2f} points.",
        f"Chosen on {s.fit_games:,} fit-season games; sigma set on "
        f"{s.calibrate_games:,} calibrate-season games.",
    ])
