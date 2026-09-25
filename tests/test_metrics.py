from datetime import UTC, date, datetime

import pytest

from predictor.backtest import metrics
from predictor.backtest.replay import Prediction

TIP = datetime(2025, 1, 16, 0, 0, tzinfo=UTC)


def make(p_home: float, home_won: bool, gid: str = "g") -> Prediction:
    return Prediction(
        game_id=gid, season="2024-25", game_date=date(2025, 1, 15),
        home_team="PHI", away_team="NYK", tipoff=TIP, cutoff=TIP,
        p_home=p_home, home_won=home_won,
    )


def test_brier_is_zero_for_perfect_confident_predictions():
    preds = [make(1.0, True), make(0.0, False)]
    assert metrics.brier_score(preds) == 0.0


def test_brier_is_one_for_perfectly_wrong_confident_predictions():
    preds = [make(0.0, True), make(1.0, False)]
    assert metrics.brier_score(preds) == 1.0


def test_brier_of_a_coin_flip_is_a_quarter():
    preds = [make(0.5, True), make(0.5, False)]
    assert metrics.brier_score(preds) == pytest.approx(0.25)


def test_accuracy_counts_the_side_the_probability_favours():
    preds = [make(0.9, True), make(0.9, False), make(0.1, False), make(0.1, True)]
    assert metrics.accuracy(preds) == pytest.approx(0.5)


def test_log_loss_penalises_confident_errors_more_than_brier():
    confident_wrong = [make(0.02, True)]
    mild_wrong = [make(0.45, True)]
    assert metrics.log_loss(confident_wrong) > metrics.log_loss(mild_wrong)


def test_log_loss_is_finite_for_a_certain_wrong_prediction():
    """A raw log loss would be infinite; probabilities must be clipped."""
    assert metrics.log_loss([make(0.0, True)]) < float("inf")


def test_home_rate_reports_the_base_rate():
    preds = [make(0.5, True), make(0.5, True), make(0.5, False), make(0.5, False)]
    assert metrics.home_rate(preds) == pytest.approx(0.5)


def test_metrics_on_an_empty_list_raise_rather_than_return_nonsense():
    for fn in (metrics.brier_score, metrics.log_loss, metrics.accuracy, metrics.home_rate):
        with pytest.raises(metrics.MetricsError):
            fn([])
