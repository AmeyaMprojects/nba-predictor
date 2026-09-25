from datetime import UTC, date, datetime

import pytest

from predictor.backtest import report
from predictor.backtest.replay import Prediction, ReplayStats

TIP = datetime(2025, 1, 16, 0, 0, tzinfo=UTC)


def make(p_home, home_won, gid="g", reconstructed=False):
    return Prediction(
        game_id=gid, season="2024-25", game_date=date(2025, 1, 15),
        home_team="PHI", away_team="NYK", tipoff=TIP, cutoff=TIP,
        p_home=p_home, home_won=home_won, reconstructed=reconstructed,
    )


STATS = ReplayStats(considered=120, predicted=100, skipped_no_tipoff=15,
                    skipped_no_result=5, skipped_result_visible=0, declined=0,
                    failed=0)


def test_summary_computes_every_headline_metric():
    preds = [make(0.7, i < 70, f"g{i}") for i in range(100)]
    r = report.summarize(preds, STATS)
    assert r.accuracy == pytest.approx(0.7)
    assert r.home_baseline == pytest.approx(0.7)
    assert r.brier > 0
    assert r.calibration_error == pytest.approx(0.0, abs=0.01)
    assert r.market_available is False


def test_report_leads_with_the_verdict_against_the_baseline():
    # A mix of home- and away-favoured predictions (home_pick_share == 0.5),
    # so this exercises the ordinary BEATS/LOSES TO/MATCHES verdict path
    # rather than FIX 3's "accuracy not meaningful" override.
    preds = [make(0.9, i < 60, f"g{i}") for i in range(50)]
    preds += [make(0.1, i < 60, f"h{i}") for i in range(50)]
    text = report.format_report(report.summarize(preds, STATS))
    first = text.splitlines()[0]
    assert first.startswith(("BEATS", "LOSES TO", "MATCHES"))


def test_report_states_coverage_honestly():
    preds = [make(0.7, i < 70, f"g{i}") for i in range(100)]
    text = report.format_report(report.summarize(preds, STATS))
    assert "100" in text
    assert "15" in text  # skipped for no tip-off
    assert "tip-off" in text.lower()


def test_report_says_market_comparison_is_unavailable_rather_than_printing_zero():
    preds = [make(0.7, i < 70, f"g{i}") for i in range(100)]
    text = report.format_report(report.summarize(preds, STATS))
    assert "market" in text.lower()
    assert "unavailable" in text.lower() or "no odds" in text.lower()


def test_report_includes_a_readable_calibration_table():
    preds = [make(0.65, i < 65, f"a{i}") for i in range(100)]
    preds += [make(0.35, i < 35, f"b{i}") for i in range(100)]
    text = report.format_report(report.summarize(preds, STATS))
    assert "calibration" in text.lower()
    assert text.count("%") >= 4


def test_summarize_with_no_predictions_raises_rather_than_reporting_zeroes():
    empty = ReplayStats(considered=10, predicted=0, skipped_no_tipoff=10,
                        skipped_no_result=0, skipped_result_visible=0,
                        declined=0, failed=0)
    with pytest.raises(Exception):
        report.summarize([], empty)


# --- FIX 2: reconstructed-timestamp provenance ----------------------------


def test_reconstructed_share_is_computed_over_the_scored_games():
    preds = [make(0.7, True, f"g{i}", reconstructed=True) for i in range(3)]
    preds += [make(0.7, True, f"h{i}", reconstructed=False) for i in range(1)]
    r = report.summarize(preds, STATS)
    assert r.reconstructed_share == pytest.approx(0.75)


def test_reconstructed_share_is_zero_when_no_prediction_is_reconstructed():
    preds = [make(0.7, i < 70, f"g{i}") for i in range(100)]
    r = report.summarize(preds, STATS)
    assert r.reconstructed_share == 0.0


def test_report_states_reconstructed_timing_provenance_when_present():
    preds = [make(0.7, i < 70, f"g{i}", reconstructed=True) for i in range(100)]
    text = report.format_report(report.summarize(preds, STATS))
    assert "100%" in text
    assert "RECONSTRUCTED" in text
    assert "leak" in text.lower() and "guard" in text.lower()
    assert "not evidence that this" in text.lower()
    assert "backtest's timing was verified" in text.lower()


def test_report_omits_provenance_block_when_nothing_is_reconstructed():
    preds = [make(0.7, i < 70, f"g{i}", reconstructed=False) for i in range(100)]
    text = report.format_report(report.summarize(preds, STATS))
    assert "provenance" not in text.lower()


# --- FIX 3: a coin flip must not print "MATCHES always-pick-home" --------


def test_a_flat_coin_flip_does_not_claim_to_match_the_baseline():
    """A flat p_home=0.5 predictor picks home every time (0.5 >= threshold),
    so accuracy collapses to the home base rate. The old verdict logic read
    that as "MATCHES always-pick-home" -- a publishable, confidently wrong
    sentence, since a coin flip is not always-pick-home."""
    preds = [make(0.5, i < 55, f"g{i}") for i in range(100)]
    text = report.format_report(report.summarize(preds, STATS))
    first = text.splitlines()[0]
    assert not first.startswith(("BEATS", "LOSES TO", "MATCHES"))
    assert "not meaningful" in first.lower()
    assert "brier" in text.lower()
    assert "calibration" in text.lower()
    # accuracy and baseline lines are still printed -- only the verdict's
    # claim of a comparison is withheld.
    assert "accuracy" in text.lower()
    assert "always-pick-home" in text.lower()


def test_always_home_also_gets_the_not_meaningful_verdict_honestly():
    """always-home has a home_pick_share of 1.0 too. Its accuracy genuinely
    IS the baseline -- suppressing the BEATS/LOSES/MATCHES claim there is
    still honest, not incorrect, since accuracy still cannot discriminate
    it from the baseline it defines."""
    preds = [make(1.0, i < 55, f"g{i}") for i in range(100)]
    text = report.format_report(report.summarize(preds, STATS))
    first = text.splitlines()[0]
    assert not first.startswith(("BEATS", "LOSES TO", "MATCHES"))
    assert "not meaningful" in first.lower()


def test_an_all_away_predictor_also_gets_the_not_meaningful_verdict():
    preds = [make(0.1, i < 55, f"g{i}") for i in range(100)]
    text = report.format_report(report.summarize(preds, STATS))
    first = text.splitlines()[0]
    assert not first.startswith(("BEATS", "LOSES TO", "MATCHES"))
    assert "not meaningful" in first.lower()
    assert "away" in first.lower()
