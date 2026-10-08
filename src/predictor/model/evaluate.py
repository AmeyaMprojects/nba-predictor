"""Walk-forward evaluation: each tuning season after the first is predicted
only with settings chosen on the seasons strictly before it, and the
recency variant (equal weight vs a half-life) with the lowest mean
walk-forward log loss is used for the final, all-tuning-seasons fit (spec
3, "Calibration redesign -- decided 2026-10-02", Task 3).

This is the one place in the codebase that re-scores the tuning seasons
themselves (never the test season, 2026-27) -- it exists to pick a recency
variant honestly, not to publish a result. `predictor evaluate-model`
prints it; `predictor fit-model` runs it and saves `Evaluation.final`.

Every walk-forward fold job PLUS the three final (all-tuning-seasons) jobs
are solved in a single `tuning.choose` call (21 jobs: 6 walk-forward
seasons x 3 recency variants, plus 3 final jobs), so the whole evaluation
costs exactly one pass over the 900-combination grid, not 21 of them.
"""

from __future__ import annotations

import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, time

from predictor.backtest import metrics
from predictor.backtest.report import (
    format_calibration_table,
    format_publishing_bar,
    publishing_bar,
)
from predictor.backtest.replay import Prediction
from predictor.model import fit, market, tuning
from predictor.model.market import MarketView
from predictor.model.ratings import win_probability
from predictor.model.settings import TEST_SEASONS, TUNING_SEASONS, ModelSettings
from predictor.model.venues import VenueIndex

# Every tuning season except the very first -- that one has no earlier
# tuning season to fold on, so it is never walk-forward predicted.
WALK_FORWARD_SEASONS = TUNING_SEASONS[1:]

_NUMBER_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
}


def _number_word(n: int) -> str:
    return _NUMBER_WORDS.get(n, str(n))


def fold_job(season: str, half_life: float | None) -> tuning.Job:
    """The job that could have been fit before `season` started: every
    tuning season strictly before it, pooled (or recency-weighted)."""
    if season not in WALK_FORWARD_SEASONS:
        raise ValueError(
            f"{season!r} is not a walk-forward season -- it has no earlier "
            f"tuning season to fold on. Walk-forward seasons: "
            f"{', '.join(WALK_FORWARD_SEASONS)}"
        )
    seasons = TUNING_SEASONS[: TUNING_SEASONS.index(season)]
    return tuning.Job(seasons, half_life)


@dataclass(frozen=True)
class GameDetail:
    """What the model-vs-market comparison needs beyond a `Prediction`."""

    model_spread: float  # model's expected home margin, positive = home favoured
    sigma: float  # the walk-forward sigma used for this game's season
    margin: int  # actual home points minus away points


@dataclass(frozen=True)
class VariantResult:
    half_life: float | None
    predictions: list[Prediction]
    log_loss: float
    brier: float
    accuracy: float
    choices: dict[str, tuning.Choice]  # keyed by the season it predicted
    details: Mapping[str, GameDetail] = field(default_factory=dict)  # by game_id


@dataclass(frozen=True)
class Evaluation:
    variants: tuple[VariantResult, ...]
    winner: VariantResult
    final: ModelSettings


def pick_winner(variants) -> VariantResult:
    """The lowest mean walk-forward log loss wins, starting from the first
    (equal-weight) variant; a later variant only displaces the current best
    if it wins by more than `tuning.TIE_TOLERANCE` -- see Global
    Constraints. Pure, so it is testable by hand with constructed
    `VariantResult`s."""
    best = variants[0]
    for v in variants[1:]:
        if v.log_loss < best.log_loss - tuning.TIE_TOLERANCE:
            best = v
    return best


def variant_label(half_life: float | None) -> str:
    if half_life is None:
        return "equal weight"
    word = "season" if half_life == 1 else "seasons"
    return f"half-life {half_life:g} {word}"


def evaluate(con, progress=None) -> Evaluation:
    """Walk-forward evaluate every recency variant and return the winner's
    final (all-tuning-seasons) settings.

    `progress`, if given, is called with a short plain-English stage message
    at each of the two slow stages (the one `tuning.choose` call, then
    turning its 21 Choices into walk-forward predictions and metrics) -- so
    a long-running caller (the CLI) can print real progress instead of
    guessing when each stage starts.
    """
    if progress is None:
        def progress(_message: str) -> None:
            pass

    venues = VenueIndex.from_db(con)
    games = fit._load(con, venues)

    fold_jobs: dict[float | None, list[tuple[str, tuning.Job]]] = {
        hl: [(season, fold_job(season, hl)) for season in WALK_FORWARD_SEASONS]
        for hl in tuning.HALF_LIVES
    }
    final_jobs = {hl: tuning.Job(TUNING_SEASONS, hl) for hl in tuning.HALF_LIVES}

    all_jobs = [job for jobs in fold_jobs.values() for _, job in jobs]
    all_jobs += list(final_jobs.values())
    grid_size = (
        len(tuning.GRID_K) * len(tuning.GRID_CAP)
        * len(tuning.GRID_REGRESSION) * len(tuning.GRID_WINDOW)
    )
    progress(f"simulating {grid_size:,} rating settings...")
    choices = tuning.choose(games, all_jobs)
    progress("scoring walk-forward predictions...")

    variants = []
    for hl in tuning.HALF_LIVES:
        predictions: list[Prediction] = []
        details: dict[str, GameDetail] = {}
        choices_by_season: dict[str, tuning.Choice] = {}
        for season, job in fold_jobs[hl]:
            # Belt-and-braces: fold_job already builds `seasons` strictly
            # before `season`, but this is the one guard that would catch a
            # future refactor leaking a same-or-later season into a fold.
            if any(s >= season for s in job.seasons):
                raise RuntimeError(
                    f"walk-forward fold for {season} includes {season} or a "
                    "later season"
                )
            choice = choices[job]
            choices_by_season[season] = choice
            for g, spread, won in tuning.spreads_and_outcomes(
                games, choice.params, choice.coefficients, (season,)
            ):
                p = win_probability(spread, choice.sigma)
                details[g.result.game_id] = GameDetail(
                    spread, choice.sigma, g.result.home_points - g.result.away_points
                )
                # tipoff/cutoff are unused by the metrics below (log loss,
                # Brier, accuracy, calibration all read p_home/home_won
                # only) -- midnight UTC on the game date is a placeholder,
                # not an observed time.
                stamp = datetime.combine(g.result.game_date, time(0), tzinfo=UTC)
                predictions.append(
                    Prediction(
                        game_id=g.result.game_id,
                        season=g.result.season,
                        game_date=g.result.game_date,
                        home_team=g.result.home_team,
                        away_team=g.result.away_team,
                        tipoff=stamp,
                        cutoff=stamp,
                        p_home=p,
                        home_won=won,
                        reconstructed=True,
                    )
                )
        variants.append(
            VariantResult(
                half_life=hl,
                predictions=predictions,
                log_loss=metrics.log_loss(predictions),
                brier=metrics.brier_score(predictions),
                accuracy=metrics.accuracy(predictions),
                choices=choices_by_season,
                details=details,
            )
        )

    winner = pick_winner(variants)
    final_choice = choices[final_jobs[winner.half_life]]
    final = ModelSettings(
        final_choice.params,
        final_choice.coefficients,
        final_choice.sigma,
        winner.half_life,
        final_choice.games,
    )
    return Evaluation(tuple(variants), winner, final)


def format_evaluation(ev: Evaluation) -> str:
    n_games = len(ev.variants[0].predictions)
    lines = [
        "  Walk-forward evaluation -- each season predicted with settings "
        "chosen only on earlier seasons",
        "  (method designed on 2026-10-02 after a first look at 2023-26; the "
        "clean test is 2026-27, predicted live)",
        "",
        "  Recency variant          log loss   Brier    accuracy   "
        f"({n_games:,} games, {WALK_FORWARD_SEASONS[0]} to {WALK_FORWARD_SEASONS[-1]})",
    ]
    for v in ev.variants:
        lines.append(
            f"    {variant_label(v.half_life):<22} {v.log_loss:.4f}   "
            f"{v.brier:.4f}    {v.accuracy * 100:4.1f}%"
        )
    lines.append(
        f"  Chosen: {variant_label(ev.winner.half_life)} (lowest log loss; a "
        f"later variant must beat it by more than {tuning.TIE_TOLERANCE:g})"
    )
    lines.append("")
    lines.append("  By season (chosen variant; settings chosen on earlier seasons only):")
    for season in WALK_FORWARD_SEASONS:
        choice = ev.winner.choices[season]
        season_preds = [p for p in ev.winner.predictions if p.season == season]
        r = choice.params
        lines.append(
            f"    {season}  k {r.k:g} cap {r.margin_cap:g} reg {r.season_regression:g} "
            f"window {r.hca_window:g} sigma {choice.sigma:.2f}   "
            f"model {metrics.accuracy(season_preds) * 100:.1f}%   "
            f"home {metrics.home_rate(season_preds) * 100:.1f}%   "
            f"Brier {metrics.brier_score(season_preds):.4f}"
        )
    lines.append("")
    lines.append(format_calibration_table(metrics.calibration_bins(ev.winner.predictions)))
    lines.append("")
    lines.append(
        format_publishing_bar(
            publishing_bar(ev.winner.predictions), season_word="evaluated season"
        )
    )
    lines.append(
        "  This is a walk-forward result on the tuning seasons, not a clean test: the "
        "method was chosen after a first look at 2023-26. The clean test, including any "
        "calibration claim, is 2026-27 predicted live."
    )
    lines.append("")
    fr = ev.final.ratings
    lines.append(
        f"  Final settings for {TEST_SEASONS[0]} (all {_number_word(len(TUNING_SEASONS))} "
        f"tuning seasons, {variant_label(ev.final.half_life)}; the walk-forward above "
        "scores the method, not these exact settings):"
    )
    lines.append(
        f"    k {fr.k:g} cap {fr.margin_cap:g} regression {fr.season_regression:g} "
        f"window {fr.hca_window:g} sigma {ev.final.sigma:.2f}"
    )
    return "\n".join(lines)


# --- Model vs market (closing lines) -----------------------------------
#
# Reads the historical closing lines (market.historical_lines) and lines
# them up against the chosen variant's walk-forward predictions. Nothing
# here feeds back into the model: the predictions are already made.

DISAGREEMENT_GAP = 0.10
# 0.60 - 0.50 is 0.0999999... in floating point; a gap that is 0.10 on
# paper must count.
_GAP_TOLERANCE = 1e-9

NO_MARKET_DATA = "No market data -- run predictor ingest-odds-history"
CLOSING_LINE_CAVEAT = (
    "Closing lines include information up to tip-off (injuries, lineups) "
    "that the model did not have."
)


@dataclass(frozen=True)
class MarketGame:
    prediction: Prediction
    model_spread: float  # positive = home favoured
    margin: int  # actual home points minus away points
    market: MarketView

    @property
    def market_prediction(self) -> Prediction:
        """The same game scored with the market's probability."""
        return replace(self.prediction, p_home=self.market.p_home)


@dataclass(frozen=True)
class DisagreementZone:
    games: int
    model_right: int
    market_right: int


@dataclass(frozen=True)
class SpreadCheck:
    games: int  # model and closing spread differ, and not a push
    model_side: int  # actual margin landed on the model's side of the line


def match_market(variant: VariantResult, lines_by_game) -> list[MarketGame]:
    """Every prediction of `variant` that has a usable market line. A
    spread-only line is turned into a probability with that season's
    walk-forward sigma (the model's own)."""
    out = []
    for p in variant.predictions:
        lines = lines_by_game.get(p.game_id)
        if not lines:
            continue
        d = variant.details[p.game_id]
        view = market.market_p_home(lines, d.sigma)
        if view is not None:
            out.append(MarketGame(p, d.model_spread, d.margin, view))
    return out


def market_games(con, variant: VariantResult) -> list[MarketGame]:
    return match_market(variant, market.historical_lines(con))


def _picks_home(p: float) -> bool:
    # The same rule metrics.accuracy uses (p >= 0.5 is a home pick).
    return p >= 0.5


def disagreement_zone(
    games: Sequence[MarketGame], gap: float = DISAGREEMENT_GAP
) -> DisagreementZone:
    """Games where the picks differ OR the two probabilities are at least
    `gap` apart, and how often each side picked the winner there."""
    n = model_right = market_right = 0
    for g in games:
        pm, pk = g.prediction.p_home, g.market.p_home
        if _picks_home(pm) == _picks_home(pk) and abs(pm - pk) < gap - _GAP_TOLERANCE:
            continue
        n += 1
        won = g.prediction.home_won
        model_right += _picks_home(pm) == won
        market_right += _picks_home(pk) == won
    return DisagreementZone(n, model_right, market_right)


def against_closing_spread(games: Sequence[MarketGame]) -> SpreadCheck:
    """Games with a closing spread where the model expects a different
    margin than the line, leaving out pushes (the margin landed exactly on
    the line): how often the result landed on the model's side.

    The stored spread is the home spread (negative = home favoured), so the
    margin the market expects is `-spread`."""
    n = model_side = 0
    for g in games:
        if g.market.spread is None:
            continue
        line = -g.market.spread
        if g.model_spread == line or g.margin == line:
            continue
        n += 1
        model_side += (g.model_spread > line) == (g.margin > line)
    return SpreadCheck(n, model_side)


def _pct(part: int, whole: int) -> str:
    return f"{part / whole * 100:.1f}%" if whole else "--"


def _market_row(label: str, games: Sequence[MarketGame]) -> str:
    model = [g.prediction for g in games]
    mkt = [g.market_prediction for g in games]
    from_spread = sum(1 for g in games if g.market.from_spread)
    return (
        f"    {label:<12}{len(games):>7,}  "
        f"{metrics.accuracy(model) * 100:>7.1f}%{metrics.brier_score(model):>8.4f}"
        f"{metrics.log_loss(model):>10.4f}   "
        f"{metrics.accuracy(mkt) * 100:>7.1f}%{metrics.brier_score(mkt):>8.4f}"
        f"{metrics.log_loss(mkt):>10.4f}   "
        f"{len(games) - from_spread:>10,}{from_spread:>8,}"
    )


def _wrap(text: str) -> list[str]:
    return textwrap.wrap(text, width=96, initial_indent="  ", subsequent_indent="  ")


def format_market_comparison(games: Sequence[MarketGame]) -> str:
    """The "Model vs market (closing lines)" section of evaluate-model."""
    title = "  Model vs market (closing lines)"
    if not games:
        return "\n".join([title, f"  {NO_MARKET_DATA}"])
    lines = [
        title,
        "  The chosen variant's walk-forward predictions, for the games that have a "
        "stored closing line.",
        f"  {CLOSING_LINE_CAVEAT}",
        "",
        " " * 25 + f"{' model ':-^26}   {' market ':-^26}   {'market chance from':>18}",
        f"    {'Season':<12}{'games':>7}  "
        + f"{'accuracy':>8}{'Brier':>8}{'log loss':>10}   " * 2
        + f"{'moneyline':>10}{'spread':>8}",
    ]
    seasons = sorted({g.prediction.season for g in games})
    for season in seasons:
        lines.append(_market_row(season, [g for g in games if g.prediction.season == season]))
    lines.append(_market_row("all seasons", games))
    lines += _wrap(
        "Market chance: the closing moneyline with the bookmaker's margin taken out. "
        "Where no moneyline was stored, it is worked out from the closing spread (in "
        "points) using the model's own sigma for that season -- so for those games the "
        "market number borrows the model's sense of how far results stray from the "
        "spread."
    )
    lines.append("")
    zone = disagreement_zone(games)
    lines += _wrap(
        "Where they disagree (different pick, or win chances at least "
        f"{DISAGREEMENT_GAP * 100:.0f} percentage points apart): {zone.games:,} games -- "
        f"model picked the winner {_pct(zone.model_right, zone.games)}, "
        f"market {_pct(zone.market_right, zone.games)}"
    )
    check = against_closing_spread(games)
    lines += _wrap(
        "Against the closing spread (games where the model expected a different "
        f"winning margin than the line, pushes left out): {check.games:,} games -- the "
        f"result landed on the model's side {_pct(check.model_side, check.games)}"
    )
    return "\n".join(lines)
