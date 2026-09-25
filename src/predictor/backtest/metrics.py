from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

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
    """Fraction of games where the favoured side actually won.

    WARNING: if `home_pick_share` is exactly 1.0 -- every prediction favours
    home -- this collapses to the home win rate, the SAME number
    `home_rate` reports as the always-pick-home baseline. A flat 0.5
    predictor (a coin flip) is the case to watch for: 0.5 >= threshold
    counts as a home pick, so it picks home every time and its accuracy
    becomes exactly the baseline's, making it LOOK identical to
    always-pick-home even though the two are not the same predictor.

    FIX 17 (final review, part 3): the opposite extreme -- `home_pick_share
    == 0.0`, a predictor that favours away every time -- does NOT have this
    problem. It disagrees with always-pick-home on every single game, so
    its accuracy (the away win rate) is a real, well-defined comparison
    against the baseline, not a collapse into looking like it. Only the
    `== 1.0` case needs guarding against. Do not compare this number to a
    baseline without also checking `home_pick_share == 1.0`;
    `report.format_report` does this before printing its verdict line.
    """
    _require(preds)
    hits = sum(1 for p in preds if (p.p_home >= threshold) == p.home_won)
    return hits / len(preds)


def home_pick_share(preds: Sequence[Prediction], threshold: float = 0.5) -> float:
    """Fraction of predictions that count as a home pick (`p_home >= threshold`).

    A value of exactly 0.0 or 1.0 means every single prediction landed on the
    same side of the line -- accuracy against that threshold then reduces to
    a base rate and cannot meaningfully be compared to a baseline. See the
    warning on `accuracy`.
    """
    _require(preds)
    return sum(1 for p in preds if p.p_home >= threshold) / len(preds)


def home_rate(preds: Sequence[Prediction]) -> float:
    """Base rate of home wins in the scored set -- the always-pick-home accuracy."""
    _require(preds)
    return sum(1 for p in preds if p.home_won) / len(preds)


@dataclass(frozen=True)
class CalibrationBin:
    low: float
    high: float
    count: int
    mean_predicted: float
    observed_rate: float


def calibration_bins(
    preds: Sequence[Prediction], n_bins: int = 10
) -> list[CalibrationBin]:
    """Group predictions by stated probability and compare to what happened.

    Empty bins are omitted rather than reported as zero, which would read as
    'we said 30% and were never right' instead of 'we never said 30%'.
    """
    _require(preds)
    if n_bins < 1:
        raise MetricsError(f"n_bins must be at least 1, got {n_bins}")

    buckets: list[list[Prediction]] = [[] for _ in range(n_bins)]
    for p in preds:
        # p == 1.0 would index past the end; clamp it into the top bin.
        idx = min(int(p.p_home * n_bins), n_bins - 1)
        buckets[idx].append(p)

    out: list[CalibrationBin] = []
    for idx, bucket in enumerate(buckets):
        if not bucket:
            continue
        out.append(
            CalibrationBin(
                low=idx / n_bins,
                high=(idx + 1) / n_bins,
                count=len(bucket),
                mean_predicted=sum(b.p_home for b in bucket) / len(bucket),
                observed_rate=sum(1 for b in bucket if b.home_won) / len(bucket),
            )
        )
    return out


def sign_test_p_value(wins: int, losses: int) -> float:
    """Exact two-sided binomial sign test p-value for `wins` vs `losses`.

    FIX 24 (final review, part 4): replaces a normal approximation, a
    continuity correction, and a flat 25-discordant-game floor that
    together printed sentences that could be false as stated -- an edge
    that visibly EXCEEDED its own printed margin while being called "too
    small to tell from chance" (reachable at any N), and lopsided splits
    below the floor (e.g. 24-0, exact p = 1.19e-07) dismissed as "too few
    to tell them apart". The exact binomial test subsumes all three: it
    needs no approximation, no correction, and no arbitrary minimum -- it
    is correct at every N, including N as low as 1.

    With `n = wins + losses`, under the null hypothesis (each disagreement
    game is an independent coin flip between the two predictors), the
    probability of a split at least as extreme as the observed one in
    either direction is `2 * P(X >= max(wins, losses))` for `X ~
    Binomial(n, 0.5)`, capped at 1.0 (the two tails can overlap when
    wins == losses).
    """
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(max(wins, losses), n + 1))
    return min(1.0, 2 * tail / 2**n)


@dataclass(frozen=True)
class PairedComparison:
    """A paired comparison against the always-pick-home baseline.

    Only games where the predictor DISAGREES with always-home (i.e. the
    predictor's p_home falls below `threshold`, so it favours away) can tell
    the two predictors apart -- on every game where they agree, both are
    right or both are wrong together, so that game carries no information
    about which one is better. `wins + losses` is exactly the count of
    disagreement games.
    """

    wins: int
    losses: int
    edge: float
    standard_error: float
    p_value: float


def paired_comparison(preds: Sequence[Prediction], threshold: float = 0.5) -> PairedComparison:
    """Paired comparison of `preds` against always-pick-home.

    `wins` = disagreement games the predictor got right (away won).
    `losses` = disagreement games the predictor got wrong (home won, so
    always-home was right instead).
    `edge` = (wins - losses) / N, over ALL scored games N (not just the
    disagreement games) -- this is the same denominator `accuracy` uses.
    `standard_error` = sqrt(wins + losses) / N -- the standard error of
    (wins - losses), expressed as a fraction of N so it is on the same
    scale as `edge`. Both wins and losses come from the SAME disagreement
    games, so wins + losses is also the count of those games; the standard
    error of a difference of two counts drawn from one binomial split is
    sqrt(wins + losses) (see e.g. the McNemar test). Informational only --
    `report.format_report` no longer uses it to decide the verdict (see
    `p_value`), only, if at all, as a labelled scale of chance variation.
    `p_value` = the exact two-sided binomial sign test p-value (see
    `sign_test_p_value`) -- this, not `standard_error`, is what
    `report.format_report` uses to decide the verdict.
    """
    _require(preds)
    disagreements = [p for p in preds if p.p_home < threshold]
    wins = sum(1 for p in disagreements if not p.home_won)
    losses = sum(1 for p in disagreements if p.home_won)
    n = len(preds)
    edge = (wins - losses) / n
    standard_error = math.sqrt(wins + losses) / n
    p_value = sign_test_p_value(wins, losses)
    return PairedComparison(
        wins=wins, losses=losses, edge=edge, standard_error=standard_error, p_value=p_value,
    )


def calibration_error(preds: Sequence[Prediction], n_bins: int = 10) -> float:
    """Count-weighted mean gap between stated probability and observed rate."""
    bins = calibration_bins(preds, n_bins)
    total = sum(b.count for b in bins)
    return sum(
        b.count * abs(b.mean_predicted - b.observed_rate) for b in bins
    ) / total
