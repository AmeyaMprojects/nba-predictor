from datetime import UTC, date, datetime

import pytest

from predictor.backtest import report
from predictor.backtest.replay import Prediction, ReplayStats

TIP = datetime(2025, 1, 16, 0, 0, tzinfo=UTC)


def make(p_home, home_won, gid="g"):
    return Prediction(
        game_id=gid, season="2024-25", game_date=date(2025, 1, 15),
        home_team="PHI", away_team="NYK", tipoff=TIP, cutoff=TIP,
        p_home=p_home, home_won=home_won,
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
    preds = [make(0.9, i < 60, f"g{i}") for i in range(100)]
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
