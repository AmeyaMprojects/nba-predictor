from datetime import UTC, date, datetime

import pytest

from predictor.backtest import metrics
from predictor.backtest.replay import Prediction

TIP = datetime(2025, 1, 16, 0, 0, tzinfo=UTC)


def make(p_home: float, home_won: bool, gid: str = "g") -> Prediction:
    return Prediction(
        game_id=gid, season="2024-25", game_date=date(2025, 1, 15),
        home_team="PHI", away_team="NYK", tipoff=TIP, cutoff=TIP,
        p_home=p_home, home_won=home_won, reconstructed=False,
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


def test_a_perfectly_calibrated_predictor_has_near_zero_error():
    # 100 games at p=0.7, exactly 70 won
    preds = [make(0.7, i < 70, f"g{i}") for i in range(100)]
    assert metrics.calibration_error(preds) == pytest.approx(0.0, abs=0.01)


def test_an_overconfident_predictor_has_large_calibration_error():
    # claims 95%, actually wins half the time
    preds = [make(0.95, i < 50, f"g{i}") for i in range(100)]
    assert metrics.calibration_error(preds) > 0.4


def test_bins_report_predicted_against_observed():
    preds = [make(0.9, i < 60, f"g{i}") for i in range(100)]
    bins = metrics.calibration_bins(preds, n_bins=10)
    assert len(bins) == 1
    b = bins[0]
    assert b.count == 100
    assert b.mean_predicted == pytest.approx(0.9)
    assert b.observed_rate == pytest.approx(0.6)


def test_empty_bins_are_omitted_not_reported_as_zero():
    preds = [make(0.55, True, f"g{i}") for i in range(10)]
    bins = metrics.calibration_bins(preds, n_bins=10)
    assert len(bins) == 1
    assert all(b.count > 0 for b in bins)


def test_a_probability_of_exactly_one_lands_in_the_top_bin():
    preds = [make(1.0, True, f"g{i}") for i in range(5)]
    bins = metrics.calibration_bins(preds, n_bins=10)
    assert len(bins) == 1
    assert bins[0].high == pytest.approx(1.0)
    assert bins[0].count == 5


def test_calibration_on_an_empty_list_raises():
    with pytest.raises(metrics.MetricsError):
        metrics.calibration_bins([])


# --- FIX 3: home_pick_share ------------------------------------------------


def test_home_pick_share_is_one_when_every_prediction_favours_home():
    preds = [make(0.5, True), make(0.9, False), make(1.0, True)]
    assert metrics.home_pick_share(preds) == pytest.approx(1.0)


def test_home_pick_share_is_zero_when_every_prediction_favours_away():
    preds = [make(0.49, True), make(0.1, False)]
    assert metrics.home_pick_share(preds) == pytest.approx(0.0)


def test_home_pick_share_is_the_fraction_of_home_picks():
    preds = [make(0.9, True), make(0.9, False), make(0.1, False), make(0.1, True)]
    assert metrics.home_pick_share(preds) == pytest.approx(0.5)


def test_home_pick_share_on_an_empty_list_raises():
    with pytest.raises(metrics.MetricsError):
        metrics.home_pick_share([])


# --- FIX 8: paired_comparison ------------------------------------------


def test_paired_comparison_ignores_games_where_the_predictor_agrees_with_home():
    # Home picks (p_home >= 0.5) never count toward wins/losses -- they
    # carry no information about whether this predictor beats always-home.
    preds = [make(0.9, True), make(0.9, False), make(0.5, True)]
    pc = metrics.paired_comparison(preds)
    assert pc.wins == 0
    assert pc.losses == 0
    assert pc.edge == 0.0


def test_paired_comparison_counts_wins_and_losses_from_disagreement_games():
    # 3 away picks: 2 correct (away won), 1 wrong (home won) -- plus 2 home
    # picks that must be ignored.
    preds = [
        make(0.2, False), make(0.2, False), make(0.2, True),
        make(0.9, True), make(0.9, False),
    ]
    pc = metrics.paired_comparison(preds)
    assert pc.wins == 2
    assert pc.losses == 1
    assert pc.edge == pytest.approx((2 - 1) / 5)
    assert pc.standard_error == pytest.approx(3**0.5 / 5)


def test_paired_comparison_on_an_empty_list_raises():
    with pytest.raises(metrics.MetricsError):
        metrics.paired_comparison([])


# --- FIX 24 (final review, part 4): exact binomial sign test --------------


def test_sign_test_p_value_of_a_24_to_0_split_matches_the_verified_archive_value():
    # Matches the exact value verified in the final-fix-4 brief.
    assert metrics.sign_test_p_value(24, 0) == pytest.approx(1.1920928955078125e-07)


def test_sign_test_p_value_is_one_with_no_disagreement_games():
    assert metrics.sign_test_p_value(0, 0) == 1.0


def test_sign_test_p_value_is_symmetric_in_wins_and_losses():
    assert metrics.sign_test_p_value(80, 20) == metrics.sign_test_p_value(20, 80)


def test_sign_test_p_value_is_one_when_the_split_is_even():
    assert metrics.sign_test_p_value(50, 50) == pytest.approx(1.0)


def test_paired_comparison_carries_the_exact_sign_test_p_value():
    preds = [
        make(0.2, False), make(0.2, False), make(0.2, True),
        make(0.9, True), make(0.9, False),
    ]
    pc = metrics.paired_comparison(preds)
    assert pc.p_value == metrics.sign_test_p_value(pc.wins, pc.losses)
