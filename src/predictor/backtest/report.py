from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from predictor.backtest import metrics
from predictor.backtest.replay import Prediction, ReplayStats

# FIX 24 (final review, part 4): the verdict is decided by the EXACT
# two-sided binomial sign test p-value (`metrics.sign_test_p_value`), not by
# a normal approximation, a continuity correction, or a flat minimum
# discordant-game count -- see `_paired_verdict` below. There is no sample
# size below which a lopsided-enough split stops being evidence: a 24-0
# split (24 discordant games, far below the OLD 25-game floor this wave
# removes) has an exact p-value of 1.19e-07.
_SIGNIFICANCE_LEVEL = 0.05


@dataclass(frozen=True)
class BacktestResult:
    predictions: Sequence[Prediction]
    stats: ReplayStats
    model: str
    buffer_minutes: int
    brier: float
    log_loss: float
    accuracy: float
    home_baseline: float
    calibration_error: float
    bins: list[metrics.CalibrationBin]
    market_available: bool
    market_reason: str | None
    # FIX 20 (final review, part 3): total odds_snapshots rows currently in
    # the archive, regardless of whether they cover any of THESE scored
    # games -- there is no join key yet between the odds table (keyed by
    # the odds provider's own event id) and the games table (keyed by the
    # NBA's game_id), so this count is honestly total-archive, not
    # matched-to-this-run. See the market section of format_report.
    market_row_count: int
    reconstructed_share: float


def summarize(
    preds: Sequence[Prediction],
    stats: ReplayStats,
    *,
    model: str,
    buffer_minutes: int,
    market_available: bool,
    market_reason: str | None = None,
    # FIX 25(c) (final review, part 4): a default of 0 here made
    # `"odds data exists (0 row(s) in the archive)"` constructible --
    # `market_row_count=0` alongside `market_available=True` claims odds
    # rows exist while also stating there are zero of them. Not reachable
    # from the CLI today (cli.py always passes the real count through),
    # but the type itself should not make that sentence possible to
    # construct by omission -- the caller must state the count.
    market_row_count: int,
) -> BacktestResult:
    """Compute every headline metric. Raises if there is nothing to score."""
    return BacktestResult(
        predictions=preds,
        stats=stats,
        # FIX 6 (final review, part 2): the report used to say nothing about
        # which model produced it, what season/buffer it ran with, or which
        # games it actually scored -- a reader could not tell two runs
        # apart. `model` and `buffer_minutes` cannot be derived from `preds`
        # (they are inputs to the run, not outputs of it), so the caller
        # must pass them through.
        model=model,
        buffer_minutes=buffer_minutes,
        brier=metrics.brier_score(preds),
        log_loss=metrics.log_loss(preds),
        accuracy=metrics.accuracy(preds),
        home_baseline=metrics.home_rate(preds),
        calibration_error=metrics.calibration_error(preds),
        bins=metrics.calibration_bins(preds),
        # FIX 10 (final review, part 2): whether odds data actually exists
        # is determined by the caller (predictor.status's own row-count
        # check, reused rather than duplicated -- see cli.py) and passed in
        # here, rather than hardcoded. Hardcoding `False` with a guessed
        # cause ("no ODDS_API_KEY set") was only true by coincidence: the
        # moment `ingest-odds` runs, that sentence goes false while the
        # report keeps printing it.
        market_available=market_available,
        market_reason=market_reason,
        market_row_count=market_row_count,
        # FIX 2: computed over `preds` (each Prediction already carries its
        # own `reconstructed` flag, read by replay.py through
        # db.POINT_IN_TIME_TABLES -- no physical table name belongs here).
        # metrics.brier_score(preds) above already raised MetricsError if
        # `preds` is empty, so this division is safe.
        reconstructed_share=sum(1 for p in preds if p.reconstructed) / len(preds),
    )


def _provenance_header(result: BacktestResult) -> list[str]:
    """FIX 6: what this report is a report OF -- printed above everything else.

    A reader months from now, or anyone the user publishes this to, must be
    able to tell which model produced it, which season(s), what buffer, and
    which games were actually scored -- without knowing anything about how
    this harness works.
    """
    seasons = sorted({p.season for p in result.predictions})
    season_label = seasons[0] if len(seasons) == 1 else "all seasons"
    game_dates = sorted(p.game_date for p in result.predictions)
    return [
        f"  Model               : {result.model}",
        f"  Season              : {season_label}",
        f"  Buffer              : {result.buffer_minutes:,} minutes before tip-off",
        f"  Date range          : {game_dates[0].isoformat()} to {game_dates[-1].isoformat()}",
        "",
    ]


def format_report(result: BacktestResult) -> str:
    """A report someone can read in thirty seconds and trust."""
    header = _provenance_header(result)

    # FIX 3: `metrics.accuracy` counts p_home >= threshold as a home pick, so
    # a predictor that puts EVERY game on the HOME side of the line (a flat
    # 0.5 coin flip included) collapses to the home base rate and reads as
    # if it "MATCHES always-pick-home" -- a publishable, confidently wrong
    # sentence, since a coin flip does not match always-pick-home. When that
    # happens, accuracy cannot discriminate this predictor from the
    # baseline at all, so the verdict must not claim a comparison. This also
    # fires for always-home itself (home_pick_share == 1.0).
    #
    # FIX 17 (final review, part 3): this used to also fire for
    # home_share == 0.0 (a predictor that picks AWAY every game), printing
    # the same "cannot distinguish this predictor from always-pick-home"
    # sentence -- which is simply false for an all-away predictor. Unlike
    # all-home, all-away DISAGREES with always-pick-home on every single
    # game, so the paired comparison below is perfectly well defined for it
    # (edge and margin both meaningful) and it must get a normal verdict
    # through that path instead. Only home_share == 1.0 collapses accuracy
    # into the baseline's own base rate.
    home_share = metrics.home_pick_share(result.predictions)

    if home_share == 1.0:
        # FIX 12(d): a flat 0.5 coin-flip predictor also lands here (0.5 >=
        # threshold counts as a home pick), even though it is NOT the same
        # predictor as always-home -- it only happens to make the same PICK
        # on every game, at a different stated probability. When the
        # probabilities are ALSO identical to the baseline's (every
        # p_home == 1.0), it is a stronger and clearer statement to say the
        # predictor simply IS always-pick-home.
        if all(p.p_home == 1.0 for p in result.predictions):
            verdict = (
                "PREDICTOR IS always-pick-home -- every prediction, and every "
                "stated probability, is identical to the baseline's, so there "
                "is nothing to compare. See the Brier score and calibration "
                "table below instead."
            )
        else:
            verdict = (
                "ACCURACY NOT MEANINGFUL -- every prediction favoured the home "
                "side, so accuracy cannot distinguish this predictor from "
                "always-pick-home. See the Brier score and calibration table "
                "below instead."
            )
    else:
        # FIX 8 (final review, part 2): a hardcoded `edge > 0.005` verdict
        # had no notion of sample size -- on a single season (~400 games)
        # noise alone is worth several points of "edge". Use a paired
        # comparison against always-pick-home instead: only games where the
        # predictor disagrees with the baseline (picks away) carry any
        # information.
        #
        # FIX 24 (final review, part 4): the decision now rests entirely on
        # `pc.p_value` -- the EXACT two-sided binomial sign test -- which
        # replaces a normal approximation, a continuity correction, and the
        # old flat `_MIN_DISCORDANT_GAMES` floor (all three removed; the
        # exact test needs none of them and is correct at every N). Wave 3's
        # version decided the verdict with one statistic (the
        # continuity-corrected normal approximation) but PRINTED another
        # (the uncorrected `+/- margin_pts`) -- for 446 reachable
        # (wins, losses) combinations the printed edge visibly EXCEEDED its
        # own printed margin while the sentence called it "too small to tell
        # from chance". And 146 combinations below the old 25-game floor had
        # an exact two-sided p < 0.05 (a 24-0 split: p = 1.19e-07) yet were
        # printed as "too few to tell them apart". No sentence below states
        # a margin at all, so neither failure mode is reachable any more --
        # every branch states the edge, the number of disagreement games,
        # and the exact p-value, and none of those three can contradict
        # each other because none is derived from a different statistic
        # than the one that decided the verdict.
        pc = metrics.paired_comparison(result.predictions)
        edge_pts = pc.edge * 100
        # FIX 22(a) (final review, part 3): guard wins == losses == 0
        # explicitly rather than relying on it being unreachable by an
        # implicit coupling to the home_share == 0/1 branch above -- with
        # no discordant games at all there is nothing paired to compare.
        n_discordant = pc.wins + pc.losses
        if n_discordant == 0:
            verdict = (
                "TOO CLOSE TO CALL -- the predictor never disagreed with "
                "always-pick-home on a single scored game, so there is "
                "nothing to compare"
            )
        elif pc.p_value < _SIGNIFICANCE_LEVEL:
            direction = "BEATS" if edge_pts > 0 else "LOSES TO"
            verdict = (
                f"{direction} always-pick-home by {abs(edge_pts):.1f} points "
                f"-- {n_discordant:,} disagreement(s), exact sign-test "
                f"p={pc.p_value:.3g}"
            )
        else:
            verdict = (
                f"TOO CLOSE TO CALL -- edge over always-pick-home is "
                f"{edge_pts:+.1f} points over {n_discordant:,} "
                f"disagreement(s); an edge this size or larger arises by "
                f"chance with p={pc.p_value:.3g} (not below the "
                f"{_SIGNIFICANCE_LEVEL:g} significance threshold used here)"
            )

    s = result.stats
    lines = header + [
        verdict,
        "",
        f"  games scored        : {s.predicted:,} of {s.considered:,} considered",
        f"  accuracy            : {result.accuracy * 100:.1f}%",
        f"  always-pick-home    : {result.home_baseline * 100:.1f}%  (the baseline)",
        f"  Brier score         : {result.brier:.4f}  (lower is better; 0.25 is a coin flip)",
        f"  log loss            : {result.log_loss:.4f}  (lower is better; 0.6931 is a "
        "coin flip; punishes confident wrong answers far harder than Brier does)",
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
    # FIX 12(c): the harness only ever scores REGULAR-SEASON games (the
    # replay's own game_id filter is 'LIKE 002%') -- playoffs, play-in, and
    # preseason games are excluded entirely and never appear in the
    # "considered" count above. Said plainly, always, not just when asked.
    lines.append(
        "    Scope: regular-season games only -- playoffs, play-in, and preseason "
        "games are out of scope and are not counted above"
    )
    if s.skipped_conflicting_metadata:
        lines.append(
            f"    {s.skipped_conflicting_metadata:,} game(s) skipped -- the archive holds "
            "contradictory rows for these games (season, date, or home/away team "
            "disagrees between ingested rows), so re-run 'predictor ingest-season' "
            "for the affected season(s) to fix them"
        )
    if s.skipped_buffer_too_early:
        # FIX 21 (final review, part 3): this used to say the buffer reached
        # "before the game was even on the schedule" -- but the timestamp it
        # is compared against is, for every game in the archive TODAY,
        # RECONSTRUCTED rather than observed: the NBA archive does not
        # record when a game was actually first announced, so the harness
        # derived it from game_date (7 days before for the regular season,
        # 1 day before for the postseason). That made the old sentence
        # false for every game it fired on (measured: none of 1,229 games
        # flagged at --buffer-minutes 14400 were genuinely unscheduled at
        # that cutoff). The guard is still a useful sanity bound -- only
        # the claim about what it checks is corrected.
        #
        # FIX 25(b) (final review, part 4): "RECONSTRUCTED" used to be
        # HARDCODED here regardless of what the data actually says -- the
        # same anti-pattern FIX 10 removed from the market line. Read it
        # off `skipped_buffer_too_early_reconstructed`, which replay.py
        # computes from the `reconstructed` flag on the row it actually
        # selected for each skipped game, so this sentence stays true the
        # moment a live ingest path starts writing genuinely OBSERVED
        # SCHEDULED rows instead of derived ones.
        n_early = s.skipped_buffer_too_early
        n_reconstructed = s.skipped_buffer_too_early_reconstructed
        if n_reconstructed == n_early:
            provenance = (
                "the harness's own RECONSTRUCTED schedule timestamp for these games, "
                "derived from game_date rather than observed (the NBA archive does "
                "not record when a game was actually first announced)"
            )
        elif n_reconstructed == 0:
            provenance = "the harness's own OBSERVED schedule timestamp for these games"
        else:
            provenance = (
                "the harness's own schedule timestamp for these games -- RECONSTRUCTED "
                f"(derived from game_date, not observed) for {n_reconstructed:,} of "
                f"them, OBSERVED for the other {n_early - n_reconstructed:,}"
            )
        lines.append(
            f"    {n_early:,} game(s) skipped -- the buffer reaches back past "
            f"{provenance}, so this run is not measuring anything meaningful for them"
        )
    if s.skipped_no_tipoff:
        lines.append(
            f"    {s.skipped_no_tipoff:,} game(s) skipped -- no tip-off time could be "
            "resolved, so no honest pre-game cutoff exists for them"
        )
        # FIX 11: a pooled count reads as scattered noise; broken down by
        # season it can reveal that the exclusion is concentrated in one or
        # two seasons instead (measured: 65% of 2025-26, 27% of 2019-20,
        # near zero everywhere else).
        for season in sorted(s.skipped_no_tipoff_by_season):
            n = s.skipped_no_tipoff_by_season[season]
            if n:
                total = s.considered_by_season.get(season, n)
                lines.append(f"        {season}: {n:,} of {total:,}")
    if s.skipped_no_result:
        lines.append(f"    {s.skipped_no_result:,} game(s) skipped -- not yet played")
    if s.skipped_score_missing:
        lines.append(
            f"    {s.skipped_score_missing:,} game(s) skipped -- played, but the archive "
            "did not record the score, so re-run 'predictor ingest-season' for the "
            "affected season(s) to fix them"
        )
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
            f"  Market comparison: unavailable -- {result.market_reason}, so there is "
            "nothing to compare against.",
        ]
    else:
        # FIX 20 (final review, part 3): the `market_available` branch had
        # no `else` at all, so the first `ingest-odds` row made this whole
        # section disappear silently -- no comparison, no explanation, no
        # signal to the reader that anything had changed. Print something
        # true instead: odds rows now exist, but there is no join key yet
        # between odds_snapshots (keyed by the odds provider's own event
        # id) and games (keyed by the NBA's game_id), so how many of them
        # actually cover THESE scored games is not something this report
        # can honestly claim to know -- building that match is the market
        # comparison itself, deliberately not built this wave.
        lines += [
            "",
            f"  Market comparison: odds data exists ({result.market_row_count:,} row(s) "
            "in the archive), but there is no comparison yet -- matching those rows to "
            "the scored games above (the comparison itself) has not been built, so how "
            "many of them cover these particular games is not known.",
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
