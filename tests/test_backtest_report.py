from datetime import UTC, date, datetime

import pytest

from predictor.backtest import metrics, report
from predictor.backtest.replay import Prediction, ReplayStats

TIP = datetime(2025, 1, 16, 0, 0, tzinfo=UTC)


def make(p_home, home_won, gid="g", reconstructed=False, season="2024-25"):
    return Prediction(
        game_id=gid, season=season, game_date=date(2025, 1, 15),
        home_team="PHI", away_team="NYK", tipoff=TIP, cutoff=TIP,
        p_home=p_home, home_won=home_won, reconstructed=reconstructed,
    )


STATS = ReplayStats(
    considered=120, predicted=100, skipped_conflicting_metadata=0,
    skipped_buffer_too_early=0,
    skipped_no_tipoff=15,
    considered_by_season={"2024-25": 120},
    skipped_no_tipoff_by_season={"2024-25": 15},
    skipped_no_result=5, skipped_score_missing=0,
    skipped_result_visible=0, declined=0, failed=0,
)

# Every test below runs the same run-provenance ("what is this a report OF")
# through summarize() -- these defaults are overridden only by the handful
# of tests that specifically exercise FIX 6/10's behaviour.
_DEFAULTS = dict(
    model="always-home",
    buffer_minutes=30,
    market_available=False,
    market_reason="no odds data has been collected yet (0 row(s) in the archive)",
)


def summarize(preds, stats=STATS, **overrides):
    kwargs = {**_DEFAULTS, **overrides}
    return report.summarize(preds, stats, **kwargs)


def verdict_line(text: str) -> str:
    """The verdict is the first un-indented line, below the FIX 6 header."""
    return next(line for line in text.splitlines() if line and not line.startswith(" "))


def test_summary_computes_every_headline_metric():
    preds = [make(0.7, i < 70, f"g{i}") for i in range(100)]
    r = summarize(preds)
    assert r.accuracy == pytest.approx(0.7)
    assert r.home_baseline == pytest.approx(0.7)
    assert r.brier > 0
    assert r.calibration_error == pytest.approx(0.0, abs=0.01)
    assert r.market_available is False


def test_report_leads_with_the_verdict_against_the_baseline():
    # A mix of home- and away-favoured predictions (home_pick_share == 0.5),
    # so this exercises the ordinary BEATS/LOSES TO/TOO CLOSE TO CALL
    # verdict path rather than FIX 3's "accuracy not meaningful" override.
    preds = [make(0.9, i < 60, f"g{i}") for i in range(50)]
    preds += [make(0.1, i < 60, f"h{i}") for i in range(50)]
    text = report.format_report(summarize(preds))
    assert verdict_line(text).startswith(("BEATS", "LOSES TO", "TOO CLOSE TO CALL"))


def test_report_states_coverage_honestly():
    preds = [make(0.7, i < 70, f"g{i}") for i in range(100)]
    text = report.format_report(summarize(preds))
    assert "games scored        : 100 of 120 considered" in text
    assert "15 game(s) skipped -- no tip-off time could be resolved" in text
    assert "5 game(s) skipped -- not yet played" in text


def test_report_says_market_comparison_is_unavailable_rather_than_printing_zero():
    preds = [make(0.7, i < 70, f"g{i}") for i in range(100)]
    text = report.format_report(summarize(preds))
    assert (
        "Market comparison: unavailable -- no odds data has been collected yet "
        "(0 row(s) in the archive), so there is nothing to compare against." in text
    )


def test_report_says_something_true_once_odds_rows_exist_rather_than_nothing():
    """FIX 20 (final review, part 3): the `market_available=True` path had
    no `else` branch at all, so the first `ingest-odds` row made the whole
    market section disappear silently -- no comparison, no explanation, no
    signal that anything had changed. It must instead say plainly that odds
    data exists, how many rows are in the archive, and that the comparison
    itself (matching those rows to these scored games) has not been built.
    """
    preds = [make(0.7, i < 70, f"g{i}") for i in range(100)]
    text = report.format_report(
        summarize(preds, market_available=True, market_reason=None, market_row_count=42)
    )
    assert "Market comparison" in text
    assert "unavailable" not in text
    assert "42" in text
    assert "has not been built" in text


def test_report_includes_a_readable_calibration_table():
    preds = [make(0.65, i < 65, f"a{i}") for i in range(100)]
    preds += [make(0.35, i < 35, f"b{i}") for i in range(100)]
    text = report.format_report(summarize(preds))
    assert "Calibration -- when it said X%, how often did that happen?" in text
    assert " 60- 70%  said  65.0%  actual  65.0%  (100 games)" in text
    assert " 30- 40%  said  35.0%  actual  35.0%  (100 games)" in text


def test_summarize_with_no_predictions_raises_rather_than_reporting_zeroes():
    empty = ReplayStats(
        considered=10, predicted=0, skipped_conflicting_metadata=0,
        skipped_buffer_too_early=0,
        skipped_no_tipoff=10,
        considered_by_season={"2024-25": 10},
        skipped_no_tipoff_by_season={"2024-25": 10},
        skipped_no_result=0, skipped_score_missing=0,
        skipped_result_visible=0, declined=0, failed=0,
    )
    with pytest.raises(metrics.MetricsError, match="no predictions to score"):
        summarize([], empty)


# --- FIX 2: reconstructed-timestamp provenance ----------------------------


def test_reconstructed_share_is_computed_over_the_scored_games():
    preds = [make(0.7, True, f"g{i}", reconstructed=True) for i in range(3)]
    preds += [make(0.7, True, f"h{i}", reconstructed=False) for i in range(1)]
    r = summarize(preds)
    assert r.reconstructed_share == pytest.approx(0.75)


def test_reconstructed_share_is_zero_when_no_prediction_is_reconstructed():
    preds = [make(0.7, i < 70, f"g{i}") for i in range(100)]
    r = summarize(preds)
    assert r.reconstructed_share == 0.0


def test_report_states_reconstructed_timing_provenance_when_present():
    preds = [make(0.7, i < 70, f"g{i}", reconstructed=True) for i in range(100)]
    text = report.format_report(summarize(preds))
    assert "100%" in text
    assert "RECONSTRUCTED" in text
    assert "leak" in text.lower() and "guard" in text.lower()
    assert "not evidence that this" in text.lower()
    assert "backtest's timing was verified" in text.lower()


def test_report_omits_provenance_block_when_nothing_is_reconstructed():
    preds = [make(0.7, i < 70, f"g{i}") for i in range(100)]
    text = report.format_report(summarize(preds))
    assert "Timing provenance" not in text


# --- FIX 3 / FIX 12(d): the one-sided message ------------------------------


def test_a_flat_coin_flip_does_not_claim_to_match_the_baseline():
    """A flat p_home=0.5 predictor picks home every time (0.5 >= threshold),
    so accuracy collapses to the home base rate. The old verdict logic read
    that as "MATCHES always-pick-home" -- a publishable, confidently wrong
    sentence, since a coin flip is not always-pick-home. Its stated
    probability (0.5) also is NOT identical to always-home's (1.0), so this
    must get the generic "not meaningful" message, not "IS always-pick-home"."""
    preds = [make(0.5, i < 55, f"g{i}") for i in range(100)]
    text = report.format_report(summarize(preds))
    first = verdict_line(text)
    assert not first.startswith(("BEATS", "LOSES TO", "TOO CLOSE TO CALL"))
    assert "not meaningful" in first.lower()
    assert "IS always-pick-home" not in text
    assert "brier" in text.lower()
    assert "calibration" in text.lower()
    # accuracy and baseline lines are still printed -- only the verdict's
    # claim of a comparison is withheld.
    assert "accuracy" in text.lower()
    assert "always-pick-home" in text.lower()


def test_always_home_states_it_is_the_baseline_rather_than_not_meaningful():
    """FIX 12(d): always-home (p_home == 1.0 on every game, identical to the
    baseline's own stated probability, not just the same pick) gets a more
    specific message than the generic "not meaningful" one -- it doesn't
    just happen to make the same picks as always-pick-home, it IS
    always-pick-home."""
    preds = [make(1.0, i < 55, f"g{i}") for i in range(100)]
    text = report.format_report(summarize(preds))
    first = verdict_line(text)
    assert not first.startswith(("BEATS", "LOSES TO", "TOO CLOSE TO CALL"))
    assert "IS always-pick-home" in first
    assert "not meaningful" not in first.lower()


def test_an_all_away_predictor_gets_a_normal_verdict_not_the_not_meaningful_one():
    """FIX 17 (final review, part 3): an all-away predictor DISAGREES with
    always-pick-home on every single game -- unlike an all-home predictor,
    whose picks are identical to the baseline's, its accuracy is perfectly
    able to distinguish it from the baseline. It must get a normal verdict
    through the paired path, not the "cannot distinguish" message (which
    was, incorrectly, printed for this case before this fix). Measured:
    accuracy 45.0% vs baseline 55.0%, edge -10.0 +/- 20.0 -- a real,
    well-defined comparison, not nothing.
    """
    preds = [make(0.1, i < 55, f"g{i}") for i in range(100)]
    r = summarize(preds)
    assert r.accuracy == pytest.approx(0.45)
    assert r.home_baseline == pytest.approx(0.55)
    text = report.format_report(r)
    first = verdict_line(text)
    assert first.startswith(("BEATS", "LOSES TO", "TOO CLOSE TO CALL"))
    assert "not meaningful" not in first.lower()


# --- FIX 5: conflicting-metadata games are surfaced in the coverage section


def test_report_states_conflicting_metadata_coverage_honestly():
    preds = [make(0.7, i < 70, f"g{i}") for i in range(100)]
    stats = ReplayStats(
        considered=123, predicted=100, skipped_conflicting_metadata=3,
        skipped_buffer_too_early=0,
        skipped_no_tipoff=15,
        considered_by_season={"2024-25": 123},
        skipped_no_tipoff_by_season={"2024-25": 15},
        skipped_no_result=5, skipped_score_missing=0,
        skipped_result_visible=0, declined=0, failed=0,
    )
    text = report.format_report(summarize(preds, stats))
    assert "3" in text
    assert "contradictory" in text.lower()
    assert "ingest-season" in text.lower()


def test_report_omits_conflicting_metadata_line_when_there_are_none():
    preds = [make(0.7, i < 70, f"g{i}") for i in range(100)]
    text = report.format_report(summarize(preds))
    assert "contradictory" not in text.lower()


# --- FIX 6: provenance header ----------------------------------------------


def test_provenance_header_states_model_season_buffer_and_date_range():
    preds = [
        Prediction(
            game_id="a", season="2023-24", game_date=date(2024, 1, 1),
            home_team="PHI", away_team="NYK", tipoff=TIP, cutoff=TIP,
            p_home=0.7, home_won=True, reconstructed=False,
        ),
        Prediction(
            game_id="b", season="2023-24", game_date=date(2024, 3, 15),
            home_team="PHI", away_team="NYK", tipoff=TIP, cutoff=TIP,
            p_home=0.3, home_won=False, reconstructed=False,
        ),
    ]
    text = report.format_report(summarize(preds, model="coin-flip", buffer_minutes=45))
    lines = text.splitlines()
    assert "Model               : coin-flip" in text
    assert "Season              : 2023-24" in text
    assert "Buffer              : 45 minutes before tip-off" in text
    assert "Date range          : 2024-01-01 to 2024-03-15" in text
    # The header must be ABOVE the verdict.
    header_idx = next(i for i, line in enumerate(lines) if "Model" in line)
    verdict_idx = next(
        i for i, line in enumerate(lines)
        if line.startswith(("BEATS", "LOSES TO", "TOO CLOSE TO CALL", "ACCURACY", "PREDICTOR"))
    )
    assert header_idx < verdict_idx


def test_provenance_header_says_all_seasons_when_predictions_span_more_than_one():
    preds = [make(0.7, True, "a", season="2023-24"), make(0.3, False, "b", season="2024-25")]
    text = report.format_report(summarize(preds))
    assert "Season              : all seasons" in text


# --- FIX 7: buffer reaching before the schedule existed --------------------


def test_report_states_buffer_too_early_coverage_honestly():
    preds = [make(0.7, i < 70, f"g{i}") for i in range(100)]
    stats = ReplayStats(
        considered=110, predicted=100, skipped_conflicting_metadata=0,
        skipped_buffer_too_early=10,
        skipped_no_tipoff=0,
        considered_by_season={"2024-25": 110},
        skipped_no_tipoff_by_season={},
        skipped_no_result=0, skipped_score_missing=0,
        skipped_result_visible=0, declined=0, failed=0,
    )
    text = report.format_report(summarize(preds, stats))
    assert "10 game(s) skipped -- the buffer reaches back before the game" in text
    assert "not yet on the schedule" in text or "even on the schedule" in text


# --- FIX 8: the verdict is a paired comparison with a stated margin --------


def _paired_preds(wins, losses, agreeing=10):
    """`wins` + `losses` disagreement games (predictor picks away), plus
    `agreeing` home-favoured games that keep home_pick_share off 0/1 so the
    FIX 3 one-sided override never fires."""
    preds = [make(0.9, True, f"h{i}") for i in range(agreeing)]
    preds += [make(0.2, False, f"w{i}") for i in range(wins)]  # away won: predictor right
    preds += [make(0.2, True, f"l{i}") for i in range(losses)]  # home won: predictor wrong
    return preds


def test_verdict_is_too_close_to_call_within_the_margin():
    preds = _paired_preds(wins=52, losses=48)
    text = report.format_report(summarize(preds))
    first = verdict_line(text)
    assert first.startswith("TOO CLOSE TO CALL")
    assert "+3.6 +/- 18.2 points" in first


def test_verdict_beats_outside_the_margin_states_the_margin():
    preds = _paired_preds(wins=80, losses=20)
    text = report.format_report(summarize(preds))
    first = verdict_line(text)
    assert first == "BEATS always-pick-home by 54.5 +/- 18.2 points"
    assert "accuracy            :" in text


def test_accuracy_line_never_carries_the_edges_margin():
    """FIX 15: the margin on the verdict line is the standard error of the
    EDGE (wins - losses); printing it again on the accuracy line stated the
    wrong quantity -- 2.5x too wide, and unlabelled. The verdict line is the
    only place a margin belongs."""
    preds = _paired_preds(wins=80, losses=20)
    text = report.format_report(summarize(preds))
    accuracy_line = next(
        line for line in text.splitlines() if line.strip().startswith("accuracy")
    )
    assert "+/-" not in accuracy_line
    assert "%" in accuracy_line


def test_verdict_loses_to_outside_the_margin_states_the_margin():
    preds = _paired_preds(wins=20, losses=80)
    text = report.format_report(summarize(preds))
    first = verdict_line(text)
    assert first == "LOSES TO always-pick-home by 54.5 +/- 18.2 points"


# --- FIX 16: a minimum discordant-pair requirement and a continuity
# correction -- a handful of lucky games must not read as a verdict --------


def test_verdict_is_too_close_to_call_below_the_minimum_discordant_games():
    """4 wins / 0 losses is a perfect record but on only 4 disagreement
    games -- exact McNemar p there is 0.125, not evidence. Below the
    minimum, the verdict must say so regardless of the ratio."""
    preds = _paired_preds(wins=4, losses=0)
    text = report.format_report(summarize(preds))
    first = verdict_line(text)
    assert first.startswith("TOO CLOSE TO CALL")
    assert "too few" in first.lower()
    assert "4" in first


def test_verdict_is_too_close_to_call_below_the_minimum_even_at_5_to_0():
    """5/0 (exact McNemar p = 0.0625) is still not evidence."""
    preds = _paired_preds(wins=5, losses=0)
    text = report.format_report(summarize(preds))
    first = verdict_line(text)
    assert first.startswith("TOO CLOSE TO CALL")


def test_paired_comparison_is_well_defined_with_no_disagreement_games():
    """FIX 22(a): wins == losses == 0 (a predictor that agrees with
    always-pick-home on every single game) is only unreachable through
    format_report's own paired branch by an IMPLICIT coupling -- n_discordant
    == 0 forces home_share == 1.0, which the branch above always catches
    first. metrics.paired_comparison itself must still behave sanely (no
    division by zero, no NaN) rather than relying on that coupling to never
    be asked -- report.format_report's own explicit `n_discordant == 0`
    guard exists for exactly this reason, so a future change to the
    home_share branch cannot silently resurrect a bogus
    'LOSES TO ... by 0.0 +/- 0.0' verdict.
    """
    preds = [make(0.9, True, f"h{i}") for i in range(10)]
    pc = metrics.paired_comparison(preds)
    assert pc.wins == 0
    assert pc.losses == 0
    assert pc.edge == 0.0
    assert pc.standard_error == 0.0


def test_continuity_correction_flips_a_borderline_verdict_to_too_close_to_call():
    """Without the continuity correction, wins=60/losses=40 (110 total
    games) sits exactly on the normal-approximation threshold
    (|60-40| == 2*sqrt(100)) and reads as BEATS. The continuity correction
    (|wins - losses| - 1) pulls it back under the threshold, where a
    discrete count this close to the boundary honestly belongs."""
    preds = _paired_preds(wins=60, losses=40)
    text = report.format_report(summarize(preds))
    first = verdict_line(text)
    assert first.startswith("TOO CLOSE TO CALL")


def test_paired_comparison_counts_only_disagreement_games():
    pc = metrics.paired_comparison(_paired_preds(wins=52, losses=48))
    assert pc.wins == 52
    assert pc.losses == 48


# --- FIX 9: "not yet played" vs "played but the score is missing" ---------


def test_report_distinguishes_not_yet_played_from_score_missing():
    preds = [make(0.7, i < 70, f"g{i}") for i in range(100)]
    stats = ReplayStats(
        considered=108, predicted=100, skipped_conflicting_metadata=0,
        skipped_buffer_too_early=0,
        skipped_no_tipoff=0,
        considered_by_season={"2024-25": 108},
        skipped_no_tipoff_by_season={},
        skipped_no_result=3, skipped_score_missing=5,
        skipped_result_visible=0, declined=0, failed=0,
    )
    text = report.format_report(summarize(preds, stats))
    assert "3 game(s) skipped -- not yet played" in text
    assert (
        "5 game(s) skipped -- played, but the archive did not record the score" in text
    )
    assert "ingest-season" in text


# --- FIX 11: the no-tip-off exclusion broken down by season ----------------


def test_no_tipoff_line_is_broken_down_by_season():
    preds = [make(0.7, i < 70, f"g{i}") for i in range(100)]
    stats = ReplayStats(
        considered=1_230 + 1_059, predicted=100, skipped_conflicting_metadata=0,
        skipped_buffer_too_early=0,
        skipped_no_tipoff=804 + 284,
        considered_by_season={"2025-26": 1_230, "2019-20": 1_059},
        skipped_no_tipoff_by_season={"2025-26": 804, "2019-20": 284},
        skipped_no_result=0, skipped_score_missing=0,
        skipped_result_visible=0, declined=0, failed=0,
    )
    text = report.format_report(summarize(preds, stats))
    assert "2025-26: 804 of 1,230" in text
    assert "2019-20: 284 of 1,059" in text


# --- FIX 12(b): log loss gets an interpretation, like Brier does ----------


def test_log_loss_line_is_interpreted():
    preds = [make(0.7, i < 70, f"g{i}") for i in range(100)]
    text = report.format_report(summarize(preds))
    assert "0.6931 is a coin flip" in text
    assert "punishes confident wrong answers far harder than Brier" in text


# --- FIX 12(c): the report states its regular-season-only scope -----------


def test_report_always_states_regular_season_scope():
    preds = [make(0.7, i < 70, f"g{i}") for i in range(100)]
    text = report.format_report(summarize(preds))
    assert "regular-season games only" in text
    assert "playoffs, play-in, and preseason" in text
