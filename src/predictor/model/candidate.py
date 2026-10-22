"""The player-aware candidate, walked forward, and the switch-over bar.

The candidate's expected home margin is

    (1 - w) * team_gap + home_court + X . coefficients

where ``team_gap`` and ``home_court`` come from the current model's team
ratings (``fit._simulate``), and ``X`` is the five rest/travel/altitude
features followed by the player roster-strength and injury terms for the
chosen formula (``player_sim``, at the final call). ``w`` is how much of the
team-rating gap the player terms replace.

Walk-forward: every season from 2020-21 to 2025-26 is predicted with
settings chosen only on the tuning seasons strictly before it
(``evaluate.fold_job``). Within a fold:

- The team-rating settings are the current model's own choice for that
  fold. They are reused, not re-tuned with the player terms (controller
  ruling), which makes the comparison with the current model conservative:
  the candidate never gets team settings tuned for it.
- Weighted least squares on the final-call features of the fold's ``002``
  games, leaving out games with no injury report at the final call (they are
  still predicted, with an injury term of 0). The lowest weighted MSE over
  player settings x formula x w wins; ties keep the first one seen.
- Sigma is chosen with ``tuning._best_sigma`` on the winner.

The six player simulations are shared by every fold and recency variant.
The early call is scored with the same settings, applied to the early-call
features.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, time
from itertools import product

import numpy as np

from predictor import db
from predictor.backtest import metrics
from predictor.backtest.report import (
    CALIBRATION_BAR_MIN_GAMES,
    CALIBRATION_BAR_POINTS,
    publishing_bar,
)
from predictor.backtest.replay import Prediction
from predictor.model import evaluate as evaluate_mod
from predictor.model import fit, tuning
from predictor.model.adjustments import FEATURE_NAMES
from predictor.model.availability import InjuryHistory
from predictor.model.fit import FitError, _Game
from predictor.model.player_ids import PlayerIndex
from predictor.model.player_ratings import HALF_LIFE_GRID, REGRESSION_GRID, PlayerParams
from predictor.model.player_sim import (
    CallFeatures,
    GameFeatures,
    PlayerLines,
    SimStats,
    collect_listings,
    simulate,
)
from predictor.model.ratings import RatingParams, win_probability
from predictor.model.settings import TEST_SEASONS, TUNING_SEASONS
from predictor.model.venues import VenueIndex

FORMULAS = ("box", "pm", "both")
W_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
PLAYER_GRID: tuple[PlayerParams, ...] = tuple(
    PlayerParams(h, r) for h, r in product(HALF_LIFE_GRID, REGRESSION_GRID)
)
SEASONS_BETTER_NEEDED = 4

TEAM_SETTINGS_NOTE = (
    "The candidate reuses the current model's team-rating settings for each "
    "season instead of re-tuning them with the player terms, so this "
    "comparison is conservative: it can only understate the candidate."
)

_COLUMNS = {"box": ("box",), "pm": ("pm",), "both": ("box", "pm")}


def feature_names(formula: str) -> tuple[str, ...]:
    cols = _COLUMNS[formula]
    return (
        FEATURE_NAMES
        + tuple(f"roster_{f}" for f in cols)
        + tuple(f"injury_{f}" for f in cols)
    )


def design_row(x: Sequence[float], call: CallFeatures, formula: str) -> tuple[float, ...]:
    """One game's regressors, in ``feature_names(formula)`` order."""
    cols = _COLUMNS[formula]
    return (
        tuple(float(v) for v in x)
        + tuple(call.roster_gap[f] for f in cols)
        + tuple(call.injury_gap[f] for f in cols)
    )


def spread(
    team_gap: float,
    home_court: float,
    row: Sequence[float],
    coefficients: Sequence[float],
    w: float,
) -> float:
    """Expected home margin: ``(1 - w) * team_gap + home_court + X . coef``."""
    return (1.0 - w) * team_gap + home_court + sum(
        c * v for c, v in zip(coefficients, row, strict=True)
    )


@dataclass(frozen=True)
class CandidateChoice:
    team: RatingParams
    player: PlayerParams
    formula: str
    w: float
    coefficients: tuple[float, ...]
    feature_names: tuple[str, ...]
    sigma: float
    games: int


@dataclass(frozen=True)
class CandidateResult:
    half_life: float | None
    final_call: evaluate_mod.VariantResult
    early_call: evaluate_mod.VariantResult
    choices: dict[str, CandidateChoice]  # keyed by the season it predicted
    stats: SimStats


@dataclass(frozen=True)
class CandidateEvaluation:
    variants: tuple[CandidateResult, ...]
    winner: CandidateResult
    final: CandidateChoice
    current: evaluate_mod.Evaluation


def load_player_index(con) -> PlayerIndex:
    """Identity-only name index from every stored player row (never test
    seasons): (player_id, player_name, team, season)."""
    table = db.POINT_IN_TIME_TABLES["player_games"]
    tests = ", ".join("?" for _ in TEST_SEASONS)
    rows = con.execute(
        f"SELECT DISTINCT player_id, player_name, team, season FROM {table} "
        f"WHERE season NOT IN ({tests}) ORDER BY player_id, season, team, player_name",
        list(TEST_SEASONS),
    ).fetchall()
    return PlayerIndex.from_rows(rows)


# -- fitting -------------------------------------------------------------------

def _weighted_lstsq(
    games: Sequence[_Game], X: np.ndarray, y: np.ndarray, w: np.ndarray, *,
    season: str | None,
) -> tuple[np.ndarray, float]:
    """Weighted least squares; returns (coefficients, weighted MSE).

    ``games`` are the rows being fitted and ``season`` the season the fit
    will predict (None for the final, all-tuning-seasons fit). Neither
    changes the arithmetic; they are here so tests can check what a fold
    was fitted on.
    """
    if season is not None and any(g.result.season >= season for g in games):
        raise FitError(
            f"internal error: the fit for {season} was given a game from {season} "
            "or later; refusing to fit"
        )
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    mse = float(np.sum(w * (y - X @ coef) ** 2) / np.sum(w))
    return coef, mse


@dataclass
class _World:
    games: list[_Game]
    features: dict[PlayerParams, dict[str, GameFeatures]]
    # (params, formula, call) -> design matrix over every game, in game order
    designs: dict[tuple[PlayerParams, str, str], np.ndarray]
    team: dict[RatingParams, np.ndarray]  # params -> (gap, home court) per game
    margin: np.ndarray
    no_report: np.ndarray  # bool per game, final call

    def team_sim(self, params: RatingParams) -> np.ndarray:
        if params not in self.team:
            self.team[params] = np.array(
                fit._simulate(params, self.games), dtype=float
            ).reshape(len(self.games), 2)
        return self.team[params]


def _fit_job(
    world: _World, job: tuning.Job, team: RatingParams, season: str | None
) -> CandidateChoice:
    games = world.games
    weights = tuning.season_weights(job.seasons, job.half_life)
    idx = np.array([
        i for i, g in enumerate(games)
        if g.result.season in weights
        and g.result.game_id.startswith("002")
        and not world.no_report[i]
    ], dtype=int)
    if idx.size == 0:
        raise FitError(
            f"no regular-season games with an injury report in "
            f"{', '.join(job.seasons)} to fit the player model on. Run "
            "'predictor ingest-player-logs' and backfill the injury reports"
        )
    fit_games = [games[i] for i in idx]
    w = np.array([weights[g.result.season] for g in fit_games])
    pre = world.team_sim(team)
    gap, hc = pre[idx, 0], pre[idx, 1]
    margin = world.margin[idx]

    best = None
    for player in PLAYER_GRID:
        for formula in FORMULAS:
            X = world.designs[(player, formula, "final")][idx]
            for wt in W_GRID:
                y = margin - (1.0 - wt) * gap - hc
                coef, mse = _weighted_lstsq(fit_games, X, y, w, season=season)
                if best is None or mse < best[0]:
                    best = (mse, player, formula, wt, coef)
    _, player, formula, wt, coef = best
    coefficients = tuple(round(float(c), 6) + 0.0 for c in coef)  # never -0.0
    X = world.designs[(player, formula, "final")][idx]
    spreads = [
        spread(g_, h_, row, coefficients, wt) for g_, h_, row in zip(gap, hc, X, strict=True)
    ]
    won = [g.result.home_points > g.result.away_points for g in fit_games]
    sigma = tuning._best_sigma(spreads, won, w)
    return CandidateChoice(
        team=team,
        player=player,
        formula=formula,
        w=wt,
        coefficients=coefficients,
        feature_names=feature_names(formula),
        sigma=sigma,
        games=int(idx.size),
    )


def _predict(
    world: _World, choice: CandidateChoice, season: str, call: str,
    predictions: list[Prediction], details: dict[str, evaluate_mod.GameDetail],
) -> None:
    """Predict the ``002`` games of ``season`` (the same games
    ``tuning.spreads_and_outcomes`` scores for the current model)."""
    pre = world.team_sim(choice.team)
    X = world.designs[(choice.player, choice.formula, call)]
    for i, g in enumerate(world.games):
        r = g.result
        if r.season != season or not r.game_id.startswith("002"):
            continue
        s = spread(pre[i, 0], pre[i, 1], X[i], choice.coefficients, choice.w)
        won = r.home_points > r.away_points
        details[r.game_id] = evaluate_mod.GameDetail(
            s, choice.sigma, r.home_points - r.away_points
        )
        # Placeholder stamps, as in evaluate.evaluate: the metrics read only
        # p_home and home_won.
        stamp = datetime.combine(r.game_date, time(0), tzinfo=UTC)
        predictions.append(Prediction(
            game_id=r.game_id, season=r.season, game_date=r.game_date,
            home_team=r.home_team, away_team=r.away_team, tipoff=stamp, cutoff=stamp,
            p_home=win_probability(s, choice.sigma), home_won=won, reconstructed=True,
        ))


def _variant(
    half_life: float | None, predictions: list[Prediction],
    details: dict[str, evaluate_mod.GameDetail],
) -> evaluate_mod.VariantResult:
    return evaluate_mod.VariantResult(
        half_life=half_life,
        predictions=predictions,
        log_loss=metrics.log_loss(predictions),
        brier=metrics.brier_score(predictions),
        accuracy=metrics.accuracy(predictions),
        choices={},
        details=details,
    )


def evaluate_candidate(
    con, current: evaluate_mod.Evaluation, progress: Callable[[str], None] | None = None
) -> CandidateEvaluation:
    """Walk the candidate forward over every recency variant and pick the
    variant with ``evaluate.pick_winner`` on final-call log loss."""
    if progress is None:
        def progress(_message: str) -> None:
            pass

    venues = VenueIndex.from_db(con)
    games = fit._load(con, venues)
    lines = PlayerLines.from_db(con)
    history = InjuryHistory.from_db(con)
    index = load_player_index(con)
    progress("collecting injury-report listings...")
    listings = collect_listings(con, games, history, index)

    features: dict[PlayerParams, dict[str, GameFeatures]] = {}
    stats: SimStats | None = None
    for n, player in enumerate(PLAYER_GRID, start=1):
        progress(f"simulating player settings {n} of {len(PLAYER_GRID)}...")
        feats, st = simulate(games, lines, history, index, player, listings)
        features[player] = feats
        # Data-quality counts (fallbacks, missing reports, unlinked names,
        # unknown statuses) do not depend on the rating settings.
        if stats is None:
            stats = st

    designs: dict[tuple[PlayerParams, str, str], np.ndarray] = {}
    for player, formula, call in product(PLAYER_GRID, FORMULAS, ("final", "early")):
        width = len(feature_names(formula))
        designs[(player, formula, call)] = np.array([
            design_row(g.x, getattr(features[player][g.result.game_id], call), formula)
            for g in games
        ], dtype=float).reshape(len(games), width)
    first = features[PLAYER_GRID[0]]
    world = _World(
        games=games,
        features=features,
        designs=designs,
        team={},
        margin=np.array(
            [g.result.home_points - g.result.away_points for g in games], dtype=float
        ),
        no_report=np.array(
            [first[g.result.game_id].final.no_report for g in games], dtype=bool
        ),
    )

    progress("fitting the candidate on earlier seasons...")
    results: list[CandidateResult] = []
    for hl in tuning.HALF_LIVES:
        choices: dict[str, CandidateChoice] = {}
        preds = {"final": ([], {}), "early": ([], {})}
        for season in evaluate_mod.WALK_FORWARD_SEASONS:
            job = evaluate_mod.fold_job(season, hl)
            if any(s >= season for s in job.seasons):
                raise RuntimeError(
                    f"walk-forward fold for {season} includes {season} or a later season"
                )
            team = current.winner.choices[season].params
            choice = _fit_job(world, job, team, season)
            choices[season] = choice
            for call, (p, d) in preds.items():
                _predict(world, choice, season, call, p, d)
        results.append(CandidateResult(
            half_life=hl,
            final_call=_variant(hl, *preds["final"]),
            early_call=_variant(hl, *preds["early"]),
            choices=choices,
            stats=stats,
        ))

    best = evaluate_mod.pick_winner([r.final_call for r in results])
    winner = next(r for r in results if r.final_call is best)
    progress("fitting the candidate on every tuning season...")
    final = _fit_job(
        world, tuning.Job(TUNING_SEASONS, winner.half_life), current.final.ratings, None
    )
    return CandidateEvaluation(tuple(results), winner, final, current)


# -- the switch-over bar ---------------------------------------------------------

@dataclass(frozen=True)
class SwitchOverBar:
    """The spec's switch-over bar. Each tuple is (current, candidate)."""

    pooled_brier: tuple[float, float]
    pooled_log_loss: tuple[float, float]
    pooled_accuracy: tuple[float, float]
    seasons_better: int
    seasons: int
    worst_bucket_gap: float | None
    calibration_met: bool
    includes_terms: bool

    @property
    def scores_better(self) -> bool:
        return (
            self.pooled_brier[1] < self.pooled_brier[0]
            and self.pooled_log_loss[1] < self.pooled_log_loss[0]
        )

    @property
    def accuracy_not_lower(self) -> bool:
        return self.pooled_accuracy[1] >= self.pooled_accuracy[0]

    @property
    def enough_seasons(self) -> bool:
        return self.seasons_better >= SEASONS_BETTER_NEEDED

    @property
    def met(self) -> bool:
        return (
            self.scores_better
            and self.accuracy_not_lower
            and self.enough_seasons
            and self.calibration_met
            and self.includes_terms
        )

    def reasons(self) -> list[str]:
        out: list[str] = []
        if not self.scores_better:
            (bc, bk), (lc, lk) = self.pooled_brier, self.pooled_log_loss
            out.append(
                f"Pooled Brier and log loss are not both lower than the current "
                f"model's: Brier {bk:.4f} against {bc:.4f}, log loss {lk:.4f} "
                f"against {lc:.4f}."
            )
        if not self.accuracy_not_lower:
            ac, ak = self.pooled_accuracy
            out.append(
                f"Accuracy is lower than the current model's: {ak * 100:.1f}% "
                f"against {ac * 100:.1f}%."
            )
        if not self.enough_seasons:
            out.append(
                f"Brier is lower in only {self.seasons_better} of {self.seasons} "
                f"seasons; the bar needs at least {SEASONS_BETTER_NEEDED}."
            )
        if not self.calibration_met:
            out.append(
                f"Calibration is off by {self.worst_bucket_gap:.1f} points in a "
                f"bucket of {CALIBRATION_BAR_MIN_GAMES}+ games; the bar allows "
                f"{CALIBRATION_BAR_POINTS:g}."
            )
        if not self.includes_terms:
            out.append(
                "The chosen settings do not use both a roster-strength term and "
                "an injury term (a coefficient rounds to zero)."
            )
        return out


def _nonzero(choice: CandidateChoice, prefix: str) -> bool:
    return any(
        round(c, 6) != 0.0
        for name, c in zip(choice.feature_names, choice.coefficients, strict=True)
        if name.startswith(prefix)
    )


def switch_over_bar(
    current: Sequence[Prediction],
    candidate: Sequence[Prediction],
    choice: CandidateChoice,
) -> SwitchOverBar:
    """Score the candidate against the current model on the same games."""
    cur_ids = {p.game_id for p in current}
    cand_ids = {p.game_id for p in candidate}
    if cur_ids != cand_ids or len(cur_ids) != len(current) or len(cand_ids) != len(candidate):
        raise ValueError(
            "the current and candidate predictions must cover the same games "
            f"({len(cur_ids - cand_ids)} only in the current model, "
            f"{len(cand_ids - cur_ids)} only in the candidate)"
        )
    seasons = sorted({p.season for p in current})
    better = 0
    for season in seasons:
        cur = [p for p in current if p.season == season]
        new = [p for p in candidate if p.season == season]
        if metrics.brier_score(new) < metrics.brier_score(cur):
            better += 1
    bar = publishing_bar(candidate)
    return SwitchOverBar(
        pooled_brier=(metrics.brier_score(current), metrics.brier_score(candidate)),
        pooled_log_loss=(metrics.log_loss(current), metrics.log_loss(candidate)),
        pooled_accuracy=(metrics.accuracy(current), metrics.accuracy(candidate)),
        seasons_better=better,
        seasons=len(seasons),
        worst_bucket_gap=bar.worst_bucket_gap,
        calibration_met=bar.calibration_met,
        includes_terms=_nonzero(choice, "roster_") and _nonzero(choice, "injury_"),
    )
