from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from predictor.backtest import metrics
from predictor.backtest.replay import Prediction, ReplayStats


@dataclass(frozen=True)
class BacktestResult:
    predictions: Sequence[Prediction]
    stats: ReplayStats
    brier: float
    log_loss: float
    accuracy: float
    home_baseline: float
    calibration_error: float
    bins: list[metrics.CalibrationBin]
    market_available: bool
    reconstructed_share: float


def summarize(
    preds: Sequence[Prediction], stats: ReplayStats
) -> BacktestResult:
    """Compute every headline metric. Raises if there is nothing to score."""
    return BacktestResult(
        predictions=preds,
        stats=stats,
        brier=metrics.brier_score(preds),
        log_loss=metrics.log_loss(preds),
        accuracy=metrics.accuracy(preds),
        home_baseline=metrics.home_rate(preds),
        calibration_error=metrics.calibration_error(preds),
        bins=metrics.calibration_bins(preds),
        # No odds data exists yet; the market comparison is built but cannot run.
        market_available=False,
        # FIX 2: computed over `preds` (each Prediction already carries its
        # own `reconstructed` flag, read by replay.py through
        # db.POINT_IN_TIME_TABLES -- no physical table name belongs here).
        # metrics.brier_score(preds) above already raised MetricsError if
        # `preds` is empty, so this division is safe.
        reconstructed_share=sum(1 for p in preds if p.reconstructed) / len(preds),
    )


def format_report(result: BacktestResult) -> str:
    """A report someone can read in thirty seconds and trust."""
    edge = result.accuracy - result.home_baseline
    if edge > 0.005:
        verdict = f"BEATS always-pick-home by {edge * 100:.1f} points"
    elif edge < -0.005:
        verdict = f"LOSES TO always-pick-home by {abs(edge) * 100:.1f} points"
    else:
        verdict = "MATCHES always-pick-home"

    # FIX 3: `metrics.accuracy` counts p_home >= threshold as a home pick, so
    # a predictor that puts EVERY game on the same side of the line (a flat
    # 0.5 coin flip included) collapses to the home base rate and reads as
    # if it "MATCHES always-pick-home" -- a publishable, confidently wrong
    # sentence, since a coin flip does not match always-pick-home. When that
    # happens, accuracy cannot discriminate this predictor from the
    # baseline at all, so the verdict must not claim a comparison. This also
    # fires for always-home itself (home_pick_share == 1.0) -- that is
    # correct and desirable: its accuracy genuinely IS the baseline, and
    # saying accuracy isn't a meaningful comparison here is still honest,
    # not wrong.
    home_share = metrics.home_pick_share(result.predictions)
    if home_share in (0.0, 1.0):
        side = "home" if home_share == 1.0 else "away"
        verdict = (
            f"ACCURACY NOT MEANINGFUL -- every prediction favoured the {side} "
            "side, so accuracy cannot distinguish this predictor from "
            "always-pick-home. See the Brier score and calibration table "
            "below instead."
        )

    s = result.stats
    lines = [
        verdict,
        "",
        f"  games scored        : {s.predicted:,} of {s.considered:,} considered",
        f"  accuracy            : {result.accuracy * 100:.1f}%",
        f"  always-pick-home    : {result.home_baseline * 100:.1f}%  (the baseline)",
        f"  Brier score         : {result.brier:.4f}  (lower is better; 0.25 is a coin flip)",
        f"  log loss            : {result.log_loss:.4f}",
        f"  calibration error   : {result.calibration_error * 100:.1f} points average gap",
        "",
        "  Calibration -- when it said X%, how often did that happen?",
    ]
    for b in result.bins:
        lines.append(
            f"    {b.low * 100:3.0f}-{b.high * 100:3.0f}%  "
            f"said {b.mean_predicted * 100:5.1f}%  "
            f"actual {b.observed_rate * 100:5.1f}%  "
            f"({b.count:,} games)"
        )

    lines += ["", "  Coverage and exclusions:"]
    if s.skipped_no_tipoff:
        lines.append(
            f"    {s.skipped_no_tipoff:,} game(s) skipped -- no tip-off time could be "
            "resolved, so no honest pre-game cutoff exists for them"
        )
    if s.skipped_no_result:
        lines.append(f"    {s.skipped_no_result:,} game(s) skipped -- not yet played")
    if s.skipped_result_visible:
        lines.append(
            f"    {s.skipped_result_visible:,} game(s) skipped -- result already visible at "
            "cutoff (leak guard), so the predictor would have seen the outcome"
        )
    if s.declined:
        lines.append(
            f"    {s.declined:,} game(s) DECLINED -- predictor deliberately declined to make "
            "a prediction"
        )
    if s.failed:
        lines.append(
            f"    {s.failed:,} game(s) NOT scored -- the predictor failed or returned "
            "an impossible probability. See the lines above."
        )

    if not result.market_available:
        lines += [
            "",
            "  Market comparison: unavailable -- no odds data has been collected "
            "(no ODDS_API_KEY set), so there is nothing to compare against.",
        ]

    # FIX 2: every published number rests on reconstructed timestamps this
    # report used to never mention. Say so plainly whenever any scored game
    # used one -- do not let a reader assume the leak guard proved anything
    # about the timing of THIS run's data.
    if result.reconstructed_share > 0:
        pct = result.reconstructed_share * 100
        lines += [
            "",
            f"  Timing provenance: {pct:.0f}% of the scored games use RECONSTRUCTED timestamps --",
            "  the NBA archive does not record when a result was published, so the harness",
            "  derived it (results assumed public 36 hours after the game date). The leak",
            "  guard therefore cannot fire on this data: it is a live check that will",
            "  matter once results are captured as they arrive, not evidence that this",
            "  backtest's timing was verified.",
        ]

    return "\n".join(lines)
