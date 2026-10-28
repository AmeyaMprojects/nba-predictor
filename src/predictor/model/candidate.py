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
- Sigma is chosen with ``best_sigma`` (``tuning._best_sigma``, vectorised)
  on the winner.

The six player simulations are shared by every fold and recency variant.
The early call is scored with the same settings, applied to the early-call
features.

Home court per team (plan 3): ``home_court`` is the home team's own edge
(``ratings.HomeParams``). Its shrinkage is searched walk-forward like the
other settings: the grid (``(math.inf,)`` = flat by default, or
``SHRINKAGE_GRID``) is the outermost loop, flat first, and only a strictly
lower weighted MSE replaces the best, so per-team wins only on strictly lower
error. The altitude allowance for a fold is the current model's altitude
coefficient for that same fold (fitted on earlier seasons only).
``evaluate_home_variants`` scores flat and per-team on one shared world.
"""

from __future__ import annotations

import math
import textwrap
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, time
from itertools import product

import numpy as np
from scipy.special import erf

from predictor import db
from predictor.backtest import metrics
from predictor.backtest.report import (
    CALIBRATION_BAR_MIN_GAMES,
    CALIBRATION_BAR_POINTS,
    publishing_bar,
)
from predictor.backtest.replay import Prediction
from predictor.model import evaluate as evaluate_mod
from predictor.model import fit, player_sim, tuning
from predictor.model.adjustments import FEATURE_NAMES
from predictor.model.availability import InjuryHistory, Listing
from predictor.model.fit import FitError, _Game
from predictor.model.player_ids import PlayerIndex
from predictor.model.player_ratings import HALF_LIFE_GRID, REGRESSION_GRID, PlayerParams
from predictor.model.player_sim import (
    CallFeatures,
    GameFeatures,
    PlayerLines,
    SimStats,
    collect_listings,
)
from predictor.model.ratings import (
    FLAT_HOME,
    SHRINKAGE_GRID,
    HomeParams,
    RatingParams,
    win_probability,
)
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
    home: HomeParams = FLAT_HOME


@dataclass(frozen=True)
class GameParts:
    """One predicted game's additive spread and its pieces, in points (the
    optional correction's inputs, Task 8). ``team_gap`` is the raw rating gap
    (before the ``1 - w`` scaling); ``roster``/``injury`` are the chosen
    formula's coefficient x feature terms, summed over its columns."""

    additive_spread: float
    team_gap: float
    home_court: float
    x: tuple[float, ...]
    roster: float
    injury: float


@dataclass(frozen=True)
class CandidateResult:
    half_life: float | None
    final_call: evaluate_mod.VariantResult
    early_call: evaluate_mod.VariantResult
    choices: dict[str, CandidateChoice]  # keyed by the season it predicted
    stats: SimStats
    # call ("final"/"early") -> game_id -> parts
    parts: dict[str, dict[str, GameParts]] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateEvaluation:
    variants: tuple[CandidateResult, ...]
    winner: CandidateResult
    final: CandidateChoice
    current: evaluate_mod.Evaluation
    # Per-game features of the first player setting. Only ``extras`` is
    # meant to be read from here: extras do not depend on the player
    # settings, the roster and injury gaps do.
    features: dict[str, GameFeatures] = field(default_factory=dict)
    # Every injury-report listing the replay collected (Task 10 estimates the
    # shipped availability table from them).
    listings: list[Listing] = field(default_factory=list)


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
    # (params, home) -> (gap, home court) per game
    team: dict[tuple[RatingParams, HomeParams], np.ndarray]
    margin: np.ndarray
    no_report: np.ndarray  # bool per game, final call

    def team_sim(self, params: RatingParams, home: HomeParams = FLAT_HOME) -> np.ndarray:
        # Flat output does not depend on the allowance or window (they only
        # feed the per-team edges, which flat never reads), so every flat
        # HomeParams shares one simulation.
        key = (params, FLAT_HOME if home.flat else home)
        if key not in self.team:
            self.team[key] = np.array(
                fit._simulate(params, self.games, home), dtype=float
            ).reshape(len(self.games), 2)
        return self.team[key]


def _fit_job(
    world: _World, job: tuning.Job, team: RatingParams, season: str | None, *,
    shrinkage_grid: tuple[float, ...], altitude_allowance: float,
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
    margin = world.margin[idx]

    # Shrinkage outermost, flat first; strictly lower MSE only, so a tie
    # keeps the earlier (flatter) setting.
    best = None
    for shrinkage in shrinkage_grid:
        home = HomeParams(shrinkage, altitude_allowance)
        pre = world.team_sim(team, home)
        gap, hc = pre[idx, 0], pre[idx, 1]
        for player in PLAYER_GRID:
            for formula in FORMULAS:
                X = world.designs[(player, formula, "final")][idx]
                for wt in W_GRID:
                    y = margin - (1.0 - wt) * gap - hc
                    coef, mse = _weighted_lstsq(fit_games, X, y, w, season=season)
                    if best is None or mse < best[0]:
                        best = (mse, home, player, formula, wt, coef)
    _, home, player, formula, wt, coef = best
    pre = world.team_sim(team, home)
    gap, hc = pre[idx, 0], pre[idx, 1]
    coefficients = tuple(round(float(c), 6) + 0.0 for c in coef)  # never -0.0
    X = world.designs[(player, formula, "final")][idx]
    spreads = [
        spread(g_, h_, row, coefficients, wt) for g_, h_, row in zip(gap, hc, X, strict=True)
    ]
    won = [g.result.home_points > g.result.away_points for g in fit_games]
    sigma = best_sigma(spreads, won, w)
    return CandidateChoice(
        team=team,
        player=player,
        formula=formula,
        w=wt,
        coefficients=coefficients,
        feature_names=feature_names(formula),
        sigma=sigma,
        games=int(idx.size),
        home=home,
    )


def best_sigma(spreads: Sequence[float], won: Sequence[bool], weights: np.ndarray) -> float:
    """``tuning._best_sigma``, vectorised; picks exactly the same sigma.

    Weighted log loss for every sigma in ``tuning.SIGMA_GRID`` at once (scipy
    ``erf``, numpy sums). Floating-point differences from the loop are ~1e-12
    relative, so every sigma whose vectorised loss is within 1e-9 of the
    minimum is re-scored with the loop's exact arithmetic (first in grid
    order wins), which makes the result identical. (``tuning.py`` is not
    modified by this plan.)
    """
    grid = np.array(tuning.SIGMA_GRID, dtype=float)
    s = np.asarray(spreads, dtype=float)
    if s.size == 0:
        return tuning.SIGMA_GRID[0]
    h = np.asarray(won, dtype=bool)
    w = np.asarray(weights, dtype=float)
    p = 0.5 * (1.0 + erf(s[:, None] / (grid[None, :] * math.sqrt(2.0))))
    p = np.clip(p, 1e-12, 1 - 1e-12)
    loss = -(w[:, None] * np.log(np.where(h[:, None], p, 1.0 - p))).sum(axis=0)
    best = float(loss.min())
    near = grid[loss <= best + 1e-9 * max(1.0, abs(best))]
    if near.size == 1:
        return float(near[0])
    # The loop's own arithmetic, over the near-ties only, in grid order.
    exact = None
    for sigma in (float(v) for v in near):
        total = 0.0
        for sp, hw, wt in zip(spreads, won, weights):
            q = min(max(win_probability(sp, sigma), 1e-12), 1 - 1e-12)
            total -= wt * math.log(q if hw else 1 - q)
        if exact is None or total < exact[0]:
            exact = (total, sigma)
    return exact[1]


def _terms(choice: CandidateChoice, row: Sequence[float], prefix: str) -> float:
    return float(sum(
        c * v for name, c, v in zip(choice.feature_names, choice.coefficients, row, strict=True)
        if name.startswith(prefix)
    ))


def _predict(
    world: _World, choice: CandidateChoice, season: str, call: str,
    predictions: list[Prediction], details: dict[str, evaluate_mod.GameDetail],
    parts: dict[str, GameParts] | None = None,
) -> None:
    """Predict the ``002`` games of ``season`` (the same games
    ``tuning.spreads_and_outcomes`` scores for the current model)."""
    pre = world.team_sim(choice.team, choice.home)
    X = world.designs[(choice.player, choice.formula, call)]
    for i, g in enumerate(world.games):
        r = g.result
        if r.season != season or not r.game_id.startswith("002"):
            continue
        s = spread(pre[i, 0], pre[i, 1], X[i], choice.coefficients, choice.w)
        if parts is not None:
            parts[r.game_id] = GameParts(
                additive_spread=float(s),
                team_gap=float(pre[i, 0]),
                home_court=float(pre[i, 1]),
                x=tuple(float(v) for v in g.x),
                roster=_terms(choice, X[i], "roster_"),
                injury=_terms(choice, X[i], "injury_"),
            )
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


def _build_world(
    con, progress: Callable[[str], None]
) -> tuple[_World, SimStats, dict[str, GameFeatures], list[Listing]]:
    """Load the games and run one player simulation per ``PLAYER_GRID``
    entry. Shared by every fold, recency variant and home-court variant."""
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
        feats, st = player_sim.simulate(games, lines, history, index, player, listings)
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
    return world, stats, first, listings


def _check_grid(shrinkage_grid: tuple[float, ...]) -> None:
    if not shrinkage_grid or shrinkage_grid[0] != math.inf:
        raise ValueError(
            f"the home-court shrinkage grid must start with math.inf (flat), got "
            f"{shrinkage_grid!r}"
        )


def _evaluate(
    world: _World, stats: SimStats, first: dict[str, GameFeatures], listings: list[Listing],
    current: evaluate_mod.Evaluation, progress: Callable[[str], None],
    shrinkage_grid: tuple[float, ...],
) -> CandidateEvaluation:
    _check_grid(shrinkage_grid)
    progress("fitting the candidate on earlier seasons...")
    results: list[CandidateResult] = []
    for hl in tuning.HALF_LIVES:
        choices: dict[str, CandidateChoice] = {}
        preds = {"final": ([], {}), "early": ([], {})}
        parts: dict[str, dict[str, GameParts]] = {"final": {}, "early": {}}
        for season in evaluate_mod.WALK_FORWARD_SEASONS:
            job = evaluate_mod.fold_job(season, hl)
            if any(s >= season for s in job.seasons):
                raise RuntimeError(
                    f"walk-forward fold for {season} includes {season} or a later season"
                )
            fold = current.winner.choices[season]
            choice = _fit_job(
                world, job, fold.params, season,
                shrinkage_grid=shrinkage_grid,
                altitude_allowance=fold.coefficients.altitude,
            )
            choices[season] = choice
            for call, (p, d) in preds.items():
                _predict(world, choice, season, call, p, d, parts[call])
        results.append(CandidateResult(
            half_life=hl,
            final_call=_variant(hl, *preds["final"]),
            early_call=_variant(hl, *preds["early"]),
            choices=choices,
            stats=stats,
            parts=parts,
        ))

    best = evaluate_mod.pick_winner([r.final_call for r in results])
    winner = next(r for r in results if r.final_call is best)
    progress("fitting the candidate on every tuning season...")
    final = _fit_job(
        world, tuning.Job(TUNING_SEASONS, winner.half_life), current.final.ratings, None,
        shrinkage_grid=shrinkage_grid,
        altitude_allowance=current.final.coefficients.altitude,
    )
    return CandidateEvaluation(tuple(results), winner, final, current, first, listings)


def _quiet(_message: str) -> None:
    pass


def evaluate_candidate(
    con, current: evaluate_mod.Evaluation, progress: Callable[[str], None] | None = None,
    *, shrinkage_grid: tuple[float, ...] = (math.inf,),
) -> CandidateEvaluation:
    """Walk the candidate forward over every recency variant and pick the
    variant with ``evaluate.pick_winner`` on final-call log loss. The
    default grid is flat home court only (plan 1)."""
    _check_grid(shrinkage_grid)
    progress = progress or _quiet
    world, stats, first, listings = _build_world(con, progress)
    return _evaluate(world, stats, first, listings, current, progress, shrinkage_grid)


@dataclass(frozen=True)
class HomeVariants:
    flat: CandidateEvaluation
    per_team: CandidateEvaluation
    # The shared world's games: every warm-up and tuning game, in replay
    # order (``fit._load``). ``home_edges`` replays them.
    games: tuple[_Game, ...]


def evaluate_home_variants(
    con, current: evaluate_mod.Evaluation, progress: Callable[[str], None] | None = None
) -> HomeVariants:
    """The candidate with flat home court and with per-team home court
    (shrinkage walked forward over ``SHRINKAGE_GRID``), on one shared world."""
    progress = progress or _quiet
    world, stats, first, listings = _build_world(con, progress)
    progress("home court: flat...")
    flat = _evaluate(world, stats, first, listings, current, progress, (math.inf,))
    progress("home court: per team...")
    per_team = _evaluate(world, stats, first, listings, current, progress, SHRINKAGE_GRID)
    return HomeVariants(flat=flat, per_team=per_team, games=tuple(world.games))


@dataclass(frozen=True)
class TeamHomeEdge:
    team: str
    home_games: int
    away_games: int
    raw: float  # unshrunk home-minus-away residual mean
    edge: float  # the home court the team gets (league value + shrunk edge)


def home_edges(variants: HomeVariants) -> list[TeamHomeEdge]:
    """Each team's home court under the per-team variant's final choice
    (fitted on every tuning season: the setting that would ship), after
    replaying every warm-up and tuning game of the variants' own world.
    Teams that played in the last tuning season; highest edge first, then
    team.

    Taking the whole ``HomeVariants`` keeps the two inputs together: a fold
    choice, the flat variant, or a partial list of games cannot be passed
    by mistake.
    """
    return _edges_after(list(variants.games), variants.per_team.final)


def _edges_after(games: Sequence[_Game], choice: CandidateChoice) -> list[TeamHomeEdge]:
    """``home_edges`` for any choice. ``games`` must be the world's games:
    no test season, in replay order, and including the last tuning season
    (the edges are as of its end)."""
    if any(g.result.season in TEST_SEASONS for g in games):
        raise ValueError("home edges were given a test season game; refusing")
    if not any(g.result.season == TUNING_SEASONS[-1] for g in games):
        raise ValueError(
            f"home edges need the games of the last tuning season ({TUNING_SEASONS[-1]})"
        )
    keys = [(g.result.game_date, g.result.game_id) for g in games]
    if keys != sorted(keys):
        raise ValueError("home edges need the games in replay order (date, then game id)")
    ratings = fit.final_ratings(choice.team, list(games), choice.home)
    teams = {
        t for g in games if g.result.season == TUNING_SEASONS[-1]
        for t in (g.result.home_team, g.result.away_team)
    }
    out = []
    for team in teams:
        summary = ratings.team_home_summary(team)
        out.append(TeamHomeEdge(
            team, summary.home_games, summary.away_games, summary.raw,
            ratings.home_court(team),
        ))
    return sorted(out, key=lambda e: (-e.edge, e.team))


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


# -- the report (evaluate-model) -------------------------------------------------

REPORT_TITLE = (
    "  Player model candidate (walk-forward, final call = tip-off minus 45 minutes)"
)
SIGMA_NOTE = (
    "Its sigma (how far results stray from the expected margin) is fitted only on "
    "games that had an injury report at the final call."
)
NAME_INDEX_NOTE = (
    "Injury-report names are matched to players using every season's game logs, for "
    "identity only (who is who on which team), never for performance data."
)
NO_PLAYER_LOGS = "  No player game logs -- run predictor ingest-player-logs"
TOP_UNLINKED = 10

WIDTH = 96  # no report line is wider than this


def _wrap(text: str) -> list[str]:
    return textwrap.wrap(
        text, width=WIDTH, initial_indent="  ", subsequent_indent="  ",
        break_on_hyphens=False,
    )


def _verdict(cur: float, new: float) -> str:
    if new < cur:
        return "better"
    if new > cur:
        return "worse"
    return "same"


def _metrics_cells(preds: Sequence[Prediction]) -> str:
    return (
        f"  {metrics.brier_score(preds):>8.4f}{metrics.log_loss(preds):>10.4f}"
        f"{metrics.accuracy(preds) * 100:>9.1f}%"
    )


def _season_table(
    current: Sequence[Prediction], candidate: Sequence[Prediction]
) -> list[str]:
    """Per-season current vs candidate, then the pooled row. ``better`` /
    ``worse`` compares Brier (strictly lower is better)."""
    lines = [
        " " * 23 + f"  {' current model ':-^28}  {' candidate ':-^28}",
        f"    {'Season':<12}{'games':>7}"
        + f"  {'Brier':>8}{'log loss':>10}{'accuracy':>10}" * 2,
    ]

    def row(label: str, cur: Sequence[Prediction], new: Sequence[Prediction]) -> str:
        return (
            f"    {label:<12}{len(cur):>7,}{_metrics_cells(cur)}{_metrics_cells(new)}"
            f"   {_verdict(metrics.brier_score(cur), metrics.brier_score(new))}"
        )

    for season in sorted({p.season for p in current}):
        lines.append(row(
            season,
            [p for p in current if p.season == season],
            [p for p in candidate if p.season == season],
        ))
    lines.append(row("all seasons", current, candidate))
    return lines


def _mark(ok: bool) -> str:
    return "[pass]" if ok else "[fail]"


def _bar_lines(bar: SwitchOverBar) -> list[str]:
    (bc, bk), (lc, lk), (ac, ak) = bar.pooled_brier, bar.pooled_log_loss, bar.pooled_accuracy
    if bar.worst_bucket_gap is None:
        calibration = f"no bucket of {CALIBRATION_BAR_MIN_GAMES}+ games to check"
    else:
        calibration = f"worst bucket off by {bar.worst_bucket_gap:.1f} points"
    checks = [
        f"{_mark(bar.scores_better)} Pooled Brier and log loss both lower: "
        f"Brier {bk:.4f} vs {bc:.4f}, log loss {lk:.4f} vs {lc:.4f}",
        f"{_mark(bar.accuracy_not_lower)} Accuracy not lower: "
        f"{ak * 100:.1f}% vs {ac * 100:.1f}%",
        f"{_mark(bar.enough_seasons)} Brier lower in at least "
        f"{SEASONS_BETTER_NEEDED} of {bar.seasons} seasons: "
        f"lower in {bar.seasons_better}",
        f"{_mark(bar.calibration_met)} Calibration within "
        f"{CALIBRATION_BAR_POINTS:g} points in every bucket of "
        f"{CALIBRATION_BAR_MIN_GAMES}+ games: {calibration}",
        f"{_mark(bar.includes_terms)} Uses both a roster-strength term and an "
        f"injury term: {'yes' if bar.includes_terms else 'no'}",
    ]
    lines = [f"  Switch-over bar: {'MET' if bar.met else 'NOT MET'}"]
    for check in checks:
        lines += textwrap.wrap(
            check, width=WIDTH, initial_indent="    ", subsequent_indent=" " * 11,
            break_on_hyphens=False,
        )
    if not bar.met:
        lines.append("  The current model stays live.")
        for reason in bar.reasons():
            lines += textwrap.wrap(
                reason, width=WIDTH, initial_indent="    - ", subsequent_indent="      ",
                break_on_hyphens=False,
            )
    return lines


def _choice_lines(label: str, c: CandidateChoice) -> list[str]:
    terms = "  ".join(
        f"{name.replace('_', ' ')} {coef:+.3f}"
        for name, coef in zip(c.feature_names, c.coefficients, strict=True)
        if name.startswith(("roster_", "injury_"))
    )
    return [
        f"    {label:<9}formula {c.formula}  w {c.w:g}  "
        f"half-life {c.player.half_life_minutes:g} min  "
        f"regression {c.player.season_regression:g}",
        f"{' ' * 6}{terms}  sigma {c.sigma:.2f}",
    ]


def _correction_line(
    base: CandidateResult, corrected: CandidateResult | None,
    tried: CandidateResult | None,
) -> list[str]:
    def change(r: CandidateResult) -> str:
        b, c = base.final_call, r.final_call
        return (
            f"pooled Brier {b.brier:.4f} -> {c.brier:.4f}, "
            f"log loss {b.log_loss:.4f} -> {c.log_loss:.4f}"
        )

    if corrected is not None:
        text = f"Correction model: included ({change(corrected)})"
    elif tried is not None:
        text = (
            "Correction model: dropped -- it did not improve the walk-forward "
            f"({change(tried)})"
        )
    else:
        text = (
            "Correction model: dropped -- it could not be trained (no season had two "
            "earlier seasons of errors)."
        )
    return _wrap(text)


def _market_cells(preds: Sequence[Prediction]) -> str:
    return (
        f"  {metrics.accuracy(preds) * 100:>7.1f}%{metrics.brier_score(preds):>8.4f}"
        f"{metrics.log_loss(preds):>10.4f}"
    )


def _market_row(label: str, games: Sequence[evaluate_mod.MarketGame]) -> str:
    """``evaluate._market_row``'s numbers, in a narrower layout (<= WIDTH)."""
    from_spread = sum(1 for g in games if g.market.from_spread)
    return (
        f"    {label:<10}{len(games):>6,}"
        f"{_market_cells([g.prediction for g in games])}"
        f"{_market_cells([g.market_prediction for g in games])}"
        f"  {len(games) - from_spread:>9,}{from_spread:>7,}"
    )


def _market_block(
    market_final: Sequence[evaluate_mod.MarketGame],
    market_current: Sequence[evaluate_mod.MarketGame],
) -> list[str]:
    lines = ["  Gap to the closing market (for information only; there is no bar here)"]
    if not market_final and not market_current:
        return lines + [f"  {evaluate_mod.NO_MARKET_DATA}"]
    lines += _wrap(
        "Each model's walk-forward predictions next to the closing line, for the games "
        "that have one."
    )
    lines += _wrap(evaluate_mod.CLOSING_LINE_CAVEAT)
    lines += [
        " " * 20 + f"  {' model ':-^26}  {' market ':-^26}  {'market chance from':>18}",
        f"    {'Model':<10}{'games':>6}"
        + f"  {'accuracy':>8}{'Brier':>8}{'log loss':>10}" * 2
        + f"  {'moneyline':>9}{'spread':>7}",
        _market_row("current", market_current),
        _market_row("candidate", market_final),
    ]
    return lines


def _data_quality(stats: SimStats) -> list[str]:
    seasons = sorted(
        set().union(
            *(stats.fallback.get(c, Counter()) for c in ("final", "early")),
            *(stats.no_report.get(c, Counter()) for c in ("final", "early")),
            stats.unlinked_by_season,
            stats.unknown_by_season,
        )
    )
    first_scored = evaluate_mod.WALK_FORWARD_SEASONS[0]
    lines = [
        "  Data quality (injury reports and names)",
        "    fallback = the report for that time was missing, so an earlier one was used;",
        "    no report = no report listed the game, so the injury term is 0;",
        "    unlinked = injury-report names matched to no player (treated as available);",
        "    unknown = statuses outside the five known ones (treated as Questionable).",
        " " * 22 + f"{' final call ':-^20}  {' early call ':-^20}",
        f"    {'Season':<18}" + f"{'fallback':>10}{'no report':>10}  " * 2
        + f"{'unlinked':>9}{'unknown':>9}",
    ]
    for s in seasons:
        cells = "".join(
            f"{stats.fallback.get(c, Counter())[s]:>10,}"
            f"{stats.no_report.get(c, Counter())[s]:>10,}  "
            for c in ("final", "early")
        )
        rep = stats.unlinked_by_season.get(s)
        unlinked = sum(rep.unlinked.values()) if rep else 0
        unknown = sum(stats.unknown_by_season.get(s, {}).values())
        label = f"{s} (warm-up)" if s < first_scored else s
        lines.append(f"    {label:<18}{cells}{unlinked:>9,}{unknown:>9,}")
    if any(s < first_scored for s in seasons):
        lines += _wrap(
            f"Seasons before {first_scored} are warm-up only: they train the ratings "
            "and are not scored."
        )
    names = stats.unlinked.unlinked
    total = sum(names.values())
    lines += _wrap(
        f"Unlinked injury-report names (treated as available): {total:,} listings, "
        f"{len(names):,} distinct names"
        + (f"; the {min(TOP_UNLINKED, len(names))} most common:" if names else ".")
    )
    top = sorted(names.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_UNLINKED]
    for (name, team), n in top:
        lines.append(f"    {name} ({team}) ×{n}")
    unknown = stats.unknown_status
    if unknown:
        listed = ", ".join(
            f"{status} ×{n}"
            for status, n in sorted(unknown.items(), key=lambda kv: (-kv[1], kv[0]))
        )
    else:
        listed = "none"
    lines += _wrap(f"Unknown injury statuses (treated as Questionable): {listed}")
    lines.append(f"  Games with no known tip-off: {stats.no_tip:,}")
    return lines


def format_candidate(
    cev: CandidateEvaluation,
    bar: SwitchOverBar,
    corrected: CandidateResult | None,
    market_final: list[evaluate_mod.MarketGame],
    market_current: list[evaluate_mod.MarketGame],
    *,
    tried: CandidateResult | None = None,
) -> str:
    """The candidate section of ``evaluate-model``.

    ``corrected`` is the corrected result when the correction is included
    (then it is the candidate scored here), else None. ``tried`` is the
    corrected result that was dropped, if there was one, so the report can
    show what it would have done. ``bar`` and ``market_final`` must be for
    the same candidate (corrected if included).
    """
    shipped = corrected if corrected is not None else cev.winner
    current = cev.current.winner.predictions
    lines = [REPORT_TITLE]
    lines += _wrap(
        "The candidate adds each team's player roster strength and the latest injury "
        "report to the current model. Every season is predicted with settings chosen "
        "only on earlier seasons, on exactly the same games as the current model. "
        "Brier and log loss: lower is better; 'better' or 'worse' compares the "
        "candidate's Brier with the current model's for that season."
    )
    lines += _wrap(TEAM_SETTINGS_NOTE)
    lines += _wrap(SIGMA_NOTE)
    lines += _wrap(NAME_INDEX_NOTE)
    lines.append("")
    lines += _season_table(current, shipped.final_call.predictions)
    lines.append(
        f"  Recency variant: {evaluate_mod.variant_label(cev.winner.half_life)} "
        "(lowest log loss, as for the current model)"
    )
    lines.append("")
    lines += _bar_lines(bar)
    lines.append("")
    lines.append("  Chosen settings by season (fitted on earlier seasons only):")
    lines += _wrap(
        "w = how much of the team-rating gap the player terms replace (0 = none, "
        "1 = all); half-life = minutes played over which a player's past games "
        "lose half their weight; regression = pull toward average at a new season; "
        "roster/injury = points per unit of each term (box score, plus-minus)."
    )
    for season, choice in sorted(shipped.choices.items()):
        lines += _choice_lines(season, choice)
    lines += _choice_lines(TEST_SEASONS[0], cev.final)
    lines.append(
        f"    ({TEST_SEASONS[0]} = fitted on all tuning seasons; it would be used if the "
        "candidate went live)"
    )
    lines.append("")
    lines += _correction_line(cev.winner, corrected, tried)
    lines.append("")
    lines += _wrap(
        "The early call -- 12:30 UTC on game day (08:30 EDT, 07:30 EST), with the injury report "
        "known by then. Same candidate settings as above. It is compared with the same "
        "current-model numbers as the final call: the current model has no injury "
        "input, so its call does not depend on the call time."
    )
    lines += _season_table(current, shipped.early_call.predictions)
    lines.append("")
    lines += _market_block(market_final, market_current)
    lines.append("")
    lines += _data_quality(shipped.stats)
    return "\n".join(lines)


# -- home court per team (evaluate-model) ------------------------------------------

HOME_TITLE = "  Home court per team (walk-forward; flat = one league-wide value)"
HOME_KEPT = "  Per-team home court: kept"
HOME_DROPPED = (
    "  Per-team home court: dropped -- the flat league value scored as well or better"
)
REPORT_LAG_NOTE = (
    "  Reports count only from 15 minutes before each call time (live runs fetch every "
    "15 minutes)."
)


def _home_label(home: HomeParams) -> str:
    return "flat" if home.flat else f"shrinkage {home.shrinkage:g} games"


def flat_home_line(flat: CandidateEvaluation) -> str:
    """The plan-1 (flat home court) candidate's pooled walk-forward scores."""
    v = flat.winner.final_call
    return (
        f"  Player model with flat home court (plan 1): Brier {v.brier:.4f}, "
        f"log loss {v.log_loss:.4f}, accuracy {v.accuracy * 100:.1f}%"
    )


def format_home_court(
    variants: HomeVariants, edges: list[TeamHomeEdge], altitude: float
) -> str:
    """Flat against per-team home court, the shrinkage chosen per season,
    and each team's home edge. Per-team is kept when the final choice (the
    setting that would ship) is not flat: its fit, like every fold's, takes
    a per-team shrinkage only on strictly lower error than flat."""
    per_team = variants.per_team
    lines = [HOME_TITLE]
    lines += _wrap(
        "Flat gives every team the league's home-court value. Per team adds each team's "
        "own home edge (its home results minus its away results, against the ratings), "
        "pulled toward the league value: shrinkage N means N games of evidence count as "
        "much as the league value. Each season's shrinkage is chosen only on earlier "
        "seasons."
    )
    for season, choice in sorted(per_team.winner.choices.items()):
        lines.append(f"  {season}  {_home_label(choice.home)}")
    lines.append(
        f"  {TEST_SEASONS[0]}  {_home_label(per_team.final.home)}  "
        "(fitted on all tuning seasons; used if the candidate went live)"
    )
    lines.append("")
    lines.append(f"    {'':<10}  {'Brier':>8}{'log loss':>10}{'accuracy':>10}")
    lines.append(f"    {'flat':<10}{_metrics_cells(variants.flat.winner.final_call.predictions)}")
    lines.append(f"    {'per team':<10}{_metrics_cells(per_team.winner.final_call.predictions)}")
    lines.append(HOME_DROPPED if per_team.final.home.flat else HOME_KEPT)
    flat_v, team_v = variants.flat.winner.final_call, per_team.winner.final_call
    if not per_team.final.home.flat and not (
        team_v.brier < flat_v.brier and team_v.log_loss < flat_v.log_loss
    ):
        lines += _wrap(
            "It is kept because it fitted better on all tuning seasons, but its walk-forward "
            f"did not beat flat on both scores (Brier {team_v.brier:.4f} against "
            f"{flat_v.brier:.4f}, log loss {team_v.log_loss:.4f} against "
            f"{flat_v.log_loss:.4f}). The switch-over bar above is what decides."
        )
    lines.append("")
    lines += _wrap(
        "Each team's home court in points at the end of the last tuning season, with the "
        "setting above (league value plus the team's shrunk edge); own = the team's "
        "unshrunk edge."
    )
    for e in edges:
        lines.append(
            f"  {e.team}  {e.edge:+.1f}  (own {e.raw:+.1f} from {e.home_games} home / "
            f"{e.away_games} away games)"
        )
    lines.append(
        f"  Altitude is a separate term: {altitude:+.2f} points, not inside DEN's or "
        "UTA's edge."
    )
    lines.append(REPORT_LAG_NOTE)
    return "\n".join(lines)
