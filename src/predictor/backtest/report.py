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

# Final review ruling (2026-09-27): spec 3 ("First delivery design") sets a
# bar before the calibration claim may be published -- calibration must land
# "within a few points" of the stated probability. "A few points" = 5
# percentage points, measured only over calibration buckets with at least 50
# games (a smaller bucket is noise, not evidence either way). These are the
# only two numbers this bar depends on; the owner can change them here.
CALIBRATION_BAR_POINTS = 5.0
CALIBRATION_BAR_MIN_GAMES = 50


@dataclass(frozen=True)
class SeasonBeat:
    """One test season's displayed accuracy vs the displayed home baseline.

    Both figures are the DISPLAYED (rounded-to-one-decimal) accuracy
    figures, matching what a reader actually sees in the per-season table --
    not the unrounded metric -- so `beats` can never disagree with the two
    numbers printed next to it.
    """

    season: str
    model_accuracy: float
    home_accuracy: float
    beats: bool


@dataclass(frozen=True)
class PublishingBar:
    """Spec 3's publishing bar: beats always-pick-home in every test season,
    AND calibration is within CALIBRATION_BAR_POINTS points in every bucket
    with at least CALIBRATION_BAR_MIN_GAMES games. Pure data -- no I/O, no
    formatting -- so it is trivially testable by hand."""

    season_beats: tuple[SeasonBeat, ...]
    all_seasons_beat: bool
    worst_bucket: metrics.CalibrationBin | None
    worst_bucket_gap: float | None  # percentage points; None iff worst_bucket is None
    calibration_met: bool

    @property
    def met(self) -> bool:
        return self.all_seasons_beat and self.calibration_met


def _displayed_calibration_gap(b: metrics.CalibrationBin) -> float:
    """The gap between the two DISPLAYED (rounded-to-one-decimal) percentages
    printed side by side in the calibration table -- "said X%, actual Y%" --
    not the gap of the underlying unrounded rates.

    Wording-fix (2026-09-27, post-launch-content review): the bar used to be
    decided from `abs(mean_predicted - observed_rate)`, the UNROUNDED gap.
    For the real stage1 output that gap rounds to 10.4 while the two
    printed figures (16.8% and 6.3%) actually differ by 10.5 -- a reader
    doing the subtraction themselves gets a different number than the
    report prints. Deciding and printing the bar from this same displayed
    gap means the printed sentence can never disagree with the two
    percentages next to it.

    The final `round(..., 1)` only clears float noise from subtracting two
    already-rounded numbers (e.g. 16.8 - 6.3 landing on 10.499999999999998
    instead of 10.5); it never changes which side of the bar a bucket
    lands on.
    """
    said = round(b.mean_predicted * 100, 1)
    actual = round(b.observed_rate * 100, 1)
    return round(abs(said - actual), 1)


def publishing_bar(preds: Sequence[Prediction]) -> PublishingBar:
    """Compute spec 3's publishing bar over the scored (headline) predictions.

    Per-season "beats" uses the DISPLAYED (rounded to one decimal) model
    accuracy and home rate for that season's own predictions -- the same
    figures `format_season_table` prints -- so this can never call a season
    a win or a loss that disagrees with what the reader sees in that table.
    Calibration uses the existing 10-bin `metrics.calibration_bins`; only
    buckets with >= CALIBRATION_BAR_MIN_GAMES games count, and the worst
    (largest-gap) qualifying bucket is always reported, whether or not it
    breaches the bar, so a reader can see the bar was actually checked.
    """
    by_season: dict[str, list[Prediction]] = {}
    for p in preds:
        by_season.setdefault(p.season, []).append(p)
    season_beats = tuple(
        SeasonBeat(
            season=season,
            model_accuracy=round(metrics.accuracy(by_season[season]) * 100, 1),
            home_accuracy=round(metrics.home_rate(by_season[season]) * 100, 1),
            beats=round(metrics.accuracy(by_season[season]) * 100, 1)
            > round(metrics.home_rate(by_season[season]) * 100, 1),
        )
        for season in sorted(by_season)
    )
    all_seasons_beat = bool(season_beats) and all(sb.beats for sb in season_beats)

    qualifying = [
        b for b in metrics.calibration_bins(preds, 10) if b.count >= CALIBRATION_BAR_MIN_GAMES
    ]
    worst_bucket = None
    worst_gap = None
    calibration_met = True
    if qualifying:
        worst_bucket = max(qualifying, key=_displayed_calibration_gap)
        worst_gap = _displayed_calibration_gap(worst_bucket)
        calibration_met = worst_gap <= CALIBRATION_BAR_POINTS

    return PublishingBar(
        season_beats=season_beats,
        all_seasons_beat=all_seasons_beat,
        worst_bucket=worst_bucket,
        worst_bucket_gap=worst_gap,
        calibration_met=calibration_met,
    )


def _english_list(items: Sequence[str]) -> str:
    """Join season names the way a person would say them out loud."""
    items = list(items)
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def format_publishing_bar(bar: PublishingBar, season_word: str = "test season") -> str:
    """Render `publishing_bar`'s verdict plainly -- "If it misses, the
    report says so plainly" (spec 3).

    `season_word` names what each `SeasonBeat` actually is -- "test season"
    for a real backtest (the default, unchanged), "evaluated season" for
    the walk-forward evaluation (task 3), which beats always-pick-home on
    WALK-FORWARD seasons, not held-out test seasons. A caller that passes a
    plural noun here (there is no such caller today) would need the 's'
    itself; nothing in this function pluralizes it.
    """
    lines = [
        "  Publishing bar (from the design spec; 'a few points' read as 5 "
        "percentage points, in buckets of 50+ games -- a reading fixed after "
        "the first test run):"
    ]
    lines.append(
        f"    Beats always-pick-home in every {season_word}:  "
        + ("YES" if bar.all_seasons_beat else "NO")
    )
    for sb in bar.season_beats:
        lines.append(
            f"        {sb.season}  {sb.model_accuracy:.1f}% vs {sb.home_accuracy:.1f}%   "
            + ("yes" if sb.beats else "no")
        )
    lines.append(
        f"    Calibration within {CALIBRATION_BAR_POINTS:g} percentage points in every "
        f"bucket of {CALIBRATION_BAR_MIN_GAMES:g}+ games:  "
        + ("YES" if bar.calibration_met else "NO")
    )
    if bar.worst_bucket is not None:
        b = bar.worst_bucket
        lines.append(
            f"        worst: said {b.mean_predicted * 100:.1f}%, actual "
            f"{b.observed_rate * 100:.1f}% ({b.count:,} games) -- off by "
            f"{bar.worst_bucket_gap:.1f} points"
        )
    if bar.met:
        lines.append(
            f"    Verdict: MET -- it beats always-pick-home in every {season_word} and "
            "its probabilities are within 5 points in every bucket of 50+ games."
        )
    elif bar.all_seasons_beat:
        # Seasons hold; only calibration fails.
        lines.append(
            "    Verdict: NOT MET -- the accuracy result holds, but its stated "
            "probabilities are off by more than 5 points in at least one bucket. "
            "Publish the accuracy result; do not claim the probabilities are "
            "calibrated yet."
        )
    else:
        lost_seasons = _english_list([sb.season for sb in bar.season_beats if not sb.beats])
        verdict = f"    Verdict: NOT MET -- it did not beat always-pick-home in {lost_seasons}."
        if not bar.calibration_met:
            verdict += (
                " Its stated probabilities are also off by more than 5 points in at "
                "least one bucket."
            )
        verdict += " Do not publish yet."
        lines.append(verdict)
    return "\n".join(lines)


def _verdict_direction(pc: metrics.PairedComparison) -> str:
    """BEATS vs LOSES TO, decided from the UNROUNDED paired result.

    t7-fix2 item B: the direction used to be decided from `edge_pts`, the
    difference of the two DISPLAYED (rounded-to-one-decimal) accuracy
    figures -- a real, significant, but tiny positive edge (e.g. 0.04
    percentage points) can round to 0.0 and then fail `edge_pts > 0`,
    printing "LOSES TO ... by 0.0" for a predictor that, in fact, beats the
    baseline. `pc.wins`/`pc.losses` (and therefore `pc.edge`, which is
    `(wins - losses) / N`) are exact counts, never rounded, so comparing
    them directly can never disagree with the true sign of the edge. This
    function only ever runs once `n_discordant > 0` has already been
    checked by the caller, so `wins == losses` (an exact tie) is the only
    remaining edge case -- it cannot occur on the branch that calls this
    (a perfect tie has the maximum possible sign-test p-value, 1.0, so it
    never reaches the `pc.p_value < _SIGNIFICANCE_LEVEL` branch), but is
    resolved to "LOSES TO" rather than raising, so this helper is total.
    """
    return "BEATS" if pc.wins > pc.losses else "LOSES TO"


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
    scope: str | None = None
    # Final review (minor): a stage1 run's report never named the settings
    # it used -- a reader could not tell two runs with different settings
    # apart without opening the JSON file by hand. Plain text, built by the
    # caller (cli.py, which already has the loaded ModelSettings and its
    # source path); None for always-home/coin-flip, which have no settings.
    settings_summary: str | None = None


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
    scope: str | None = None,
    settings_summary: str | None = None,
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
        scope=scope,
        settings_summary=settings_summary,
    )


def _format_p(p: float) -> str:
    """Never print `p=0`, which is false for any finite sample.

    `2 * tail / 2**n` underflows to 0.0 for a large, lopsided discordant set
    -- from about a 70.8% win rate at the full archive's 8,289 disagreements.
    The true p-value is tiny but never zero, so say that instead.
    """
    return "<1e-300" if p <= 0.0 else f"{p:.3g}"


def _provenance_header(result: BacktestResult) -> list[str]:
    """FIX 6: what this report is a report OF -- printed above everything else.

    A reader months from now, or anyone the user publishes this to, must be
    able to tell which model produced it, which season(s), what buffer, and
    which games were actually scored -- without knowing anything about how
    this harness works.
    """
    seasons = sorted({p.season for p in result.predictions})
    season_label = seasons[0] if len(seasons) == 1 else "all seasons"
    if result.scope is not None:
        season_label = result.scope
    game_dates = sorted(p.game_date for p in result.predictions)
    lines = [
        f"  Model               : {result.model}",
        f"  Season              : {season_label}",
        f"  Buffer              : {result.buffer_minutes:,} minutes before tip-off",
        f"  Date range          : {game_dates[0].isoformat()} to {game_dates[-1].isoformat()}",
    ]
    # Final review (minor): name the settings a stage1 run actually used, so
    # a reader can tell two runs apart without opening the JSON file by hand.
    if result.settings_summary is not None:
        lines.append(f"  Settings            : {result.settings_summary}")
    lines.append("")
    return lines


def format_calibration_table(bins: Sequence[metrics.CalibrationBin]) -> str:
    """The 10-bucket "said / actual / games" calibration table, extracted
    from `format_report` so `evaluate.format_evaluation` (task 3) can print
    the identical rows for its own walk-forward predictions."""
    lines = ["  Calibration -- when it said X%, how often did that happen?"]
    for b in bins:
        lines.append(
            f"    {b.low * 100:3.0f}-{b.high * 100:3.0f}%  "
            f"said {b.mean_predicted * 100:5.1f}%  "
            f"actual {b.observed_rate * 100:5.1f}%  "
            f"({b.count:,} games)"
        )
    return "\n".join(lines)


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
        # Finding 3 (t7-fix1): print the edge as the difference of the
        # DISPLAYED (already-rounded-to-one-decimal) accuracy and baseline
        # figures, not the unrounded pc.edge*100 -- otherwise the two
        # numbers on screen can visibly disagree (67.3 - 54.7 = 12.6, but
        # the old unrounded edge rounded to 12.5). pc.edge*100 and
        # (accuracy - home_baseline)*100 are the same real number before
        # rounding (see metrics.paired_comparison's docstring), so this is
        # only ever a display fix, never a different verdict.
        acc_disp = round(result.accuracy * 100, 1)
        base_disp = round(result.home_baseline * 100, 1)
        edge_pts = acc_disp - base_disp
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
            # t7-fix2 item B: direction comes from the unrounded paired
            # result (wins vs losses), never from `edge_pts` (the rounded
            # DISPLAY figure) -- see `_verdict_direction`. `edge_pts` is
            # still what gets printed as the size of the edge.
            direction = _verdict_direction(pc)
            verdict = (
                f"{direction} always-pick-home by {abs(edge_pts):.1f} "
                f"percentage points -- {n_discordant:,} disagreement(s), "
                f"exact sign-test p={_format_p(pc.p_value)}"
            )
        else:
            verdict = (
                f"TOO CLOSE TO CALL -- edge over always-pick-home is "
                f"{edge_pts:+.1f} percentage points over {n_discordant:,} "
                f"disagreement(s); an edge this size or larger arises by "
                f"chance with p={_format_p(pc.p_value)} (not below the "
                f"{_SIGNIFICANCE_LEVEL:g} significance threshold used here)"
            )

    s = result.stats
    # Finding 1 (t7-fix1): when `scope` narrows the headline to a subset of
    # seasons (today, the test seasons), `s` here is still the POOLED
    # ReplayStats over every replayed season -- printing it under
    # "of ... considered" as if it counted only the headline's games would
    # be a lie by omission (a real run: "14,439 of 14,439 considered" under
    # a header that says "test seasons ... only"). State the headline's own
    # scored count and the pooled total separately instead.
    #
    # t7-fix2 item A: the "(N replayed in total, including warm-up and
    # tuning seasons)" clause is only true when the pooled total actually
    # EXCEEDS the headline's own count -- e.g. under `--season <test
    # season>`, replay.replay only ever walks that one season, so
    # `s.predicted == len(result.predictions)` exactly and nothing besides
    # the headline's own games was replayed. Printing the "including..."
    # clause there would falsely claim warm-up/tuning seasons were replayed
    # when none were.
    n_test = len(result.predictions)
    if result.scope is not None and s.predicted > n_test:
        games_scored_line = (
            f"  games scored        : {n_test:,} test-season games "
            f"({s.predicted:,} replayed in total, including warm-up and "
            "tuning seasons)"
        )
    elif result.scope is not None:
        games_scored_line = f"  games scored        : {n_test:,} test-season games"
    else:
        games_scored_line = f"  games scored        : {s.predicted:,} of {s.considered:,} considered"
    lines = header + [
        verdict,
        "",
        games_scored_line,
        f"  accuracy            : {result.accuracy * 100:.1f}%",
        f"  always-pick-home    : {result.home_baseline * 100:.1f}%  (the baseline)",
        f"  Brier score         : {result.brier:.4f}  (lower is better; 0.25 is a coin flip)",
        f"  log loss            : {result.log_loss:.4f}  (lower is better; 0.6931 is a "
        "coin flip; punishes confident wrong answers far harder than Brier does)",
        f"  calibration error   : {result.calibration_error * 100:.1f} percentage points "
        "average gap",
        "",
    ]
    lines.append(format_calibration_table(result.bins))

    # Final review: only a scoped (stage1 test-season headline) run has a
    # publishing claim to check -- always-home/coin-flip output is
    # unchanged.
    if result.scope is not None:
        lines += ["", format_publishing_bar(publishing_bar(result.predictions))]

    lines += ["", "  Coverage and exclusions:"]
    # Finding 1 (t7-fix1): every count below comes from the POOLED
    # ReplayStats (every replayed season), even when `scope` narrows the
    # headline above to a subset of seasons -- say so on each pooled line
    # so a reader cannot mistake, say, "3 games skipped" for "3 of the
    # headline's test-season games", when it may really be 3 warm-up-season
    # games that never touch the headline at all.
    pooled_note = " (across all replayed seasons)" if result.scope is not None else ""
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
            f"for the affected season(s) to fix them{pooled_note}"
        )
    if s.skipped_buffer_too_early:
        # FIX 21 (final review, part 3): this used to say the buffer reached
        # "before the game was even on the schedule" -- but the timestamp it
        # is compared against is, for every game in the archive TODAY,
        # RECONSTRUCTED rather than observed: the NBA archive does not
        # record when a game was actually first announced, so the harness
        # derived it from game_date (7 days before for the regular season,
        # 1 day before for the postseason). That made the old sentence
        # false for every game it fired on (measured: none of the 7,200
        # scored games flagged at --buffer-minutes 14400 were genuinely
        # unscheduled at that cutoff; the 1,229 figure first recorded here
        # was one season alone). The guard is still a useful sanity bound -- only
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
            f"{provenance}, so this run is not measuring anything meaningful "
            f"for them{pooled_note}"
        )
    if s.skipped_no_tipoff:
        lines.append(
            f"    {s.skipped_no_tipoff:,} game(s) skipped -- no tip-off time could be "
            f"resolved, so no honest pre-game cutoff exists for them{pooled_note}"
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
        lines.append(
            f"    {s.skipped_no_result:,} game(s) skipped -- not yet played{pooled_note}"
        )
    if s.skipped_score_missing:
        lines.append(
            f"    {s.skipped_score_missing:,} game(s) skipped -- played, but the archive "
            "did not record the score, so re-run 'predictor ingest-season' for the "
            f"affected season(s) to fix them{pooled_note}"
        )
    if s.skipped_result_visible:
        lines.append(
            f"    {s.skipped_result_visible:,} game(s) skipped -- result already visible at "
            f"cutoff (leak guard), so the predictor would have seen the outcome{pooled_note}"
        )
    if s.declined:
        lines.append(
            f"    {s.declined:,} game(s) DECLINED -- predictor deliberately declined to make "
            f"a prediction{pooled_note}"
        )
    if s.failed:
        lines.append(
            f"    {s.failed:,} game(s) NOT scored -- the predictor failed or returned "
            f"an impossible probability{pooled_note}. See the lines above."
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


def format_season_table(preds, role_of) -> str:
    """Accuracy, home rate and Brier for every season, labeled by role, so
    one lucky season cannot carry a pooled headline unseen."""
    lines = ["  By season (model accuracy / always-pick-home / Brier):"]
    by_season: dict[str, list] = {}
    for p in preds:
        by_season.setdefault(p.season, []).append(p)
    for season in sorted(by_season):
        group = by_season[season]
        lines.append(
            f"    {season}  {role_of(season):<9}  {len(group):>5,} games   "
            f"model {metrics.accuracy(group) * 100:5.1f}%   "
            f"home {metrics.home_rate(group) * 100:5.1f}%   "
            f"Brier {metrics.brier_score(group):.4f}"
        )
    return "\n".join(lines)
