from datetime import date

import pytest

from predictor.model.ratings import RatingParams, Ratings, Result, win_probability

P = RatingParams(k=0.1, margin_cap=20.0, season_regression=0.5, hca_window=2)


def _r(home, away, hp, ap, season="2024-25", neutral=False, gid="0022400001"):
    return Result(gid, season, date(2025, 1, 15), home, away, hp, ap, neutral)


def test_unknown_team_is_average():
    assert Ratings(P).rating("PHI") == 0.0


def test_first_game_update_by_hand():
    # home_court is 0 before any game; predicted = 0; margin 10.
    # delta = 0.1 * (10 - 0) = 1.0
    r = Ratings(P)
    r.apply(_r("PHI", "NYK", 110, 100))
    assert r.rating("PHI") == pytest.approx(1.0)
    assert r.rating("NYK") == pytest.approx(-1.0)


def test_second_game_uses_ratings_and_home_court_by_hand():
    r = Ratings(P)
    r.apply(_r("PHI", "NYK", 110, 100))  # PHI +1, NYK -1, home margins [10]
    # predicted = 1 - (-1) + 10 = 12; margin -4; delta = 0.1 * (-4 - 12) = -1.6
    r.apply(_r("PHI", "NYK", 100, 104))
    assert r.rating("PHI") == pytest.approx(-0.6)
    assert r.rating("NYK") == pytest.approx(0.6)
    # home margins [10, -4] -> mean 3.0
    assert r.home_court() == pytest.approx(3.0)


def test_blowout_is_capped_by_hand():
    r = Ratings(P)
    r.apply(_r("PHI", "NYK", 150, 100))  # margin 50 capped to 20; delta 2.0
    assert r.rating("PHI") == pytest.approx(2.0)
    # but home court uses the real margin
    assert r.home_court() == pytest.approx(50.0)


def test_neutral_site_has_no_home_court_and_does_not_feed_it():
    r = Ratings(P)
    r.apply(_r("PHI", "NYK", 110, 100))           # home margins [10]
    r.apply(_r("BOS", "MIA", 100, 100, neutral=True))
    # neutral: predicted = 0 - 0 + 0 = 0; margin 0; delta 0
    assert r.rating("BOS") == 0.0
    assert r.home_court() == pytest.approx(10.0)


def test_home_court_window_keeps_only_the_most_recent_games():
    r = Ratings(P)  # window 2
    for margin in (10, 20, 30):
        r.apply(_r("PHI", "NYK", 100 + margin, 100))
    assert r.home_court() == pytest.approx(25.0)


def test_new_season_regresses_every_rating_by_hand():
    r = Ratings(P)
    r.apply(_r("PHI", "NYK", 110, 100))  # +1 / -1
    r.enter_season("2025-26")
    assert r.rating("PHI") == pytest.approx(0.5)
    assert r.rating("NYK") == pytest.approx(-0.5)


def test_entering_the_same_season_twice_regresses_once():
    r = Ratings(P)
    r.apply(_r("PHI", "NYK", 110, 100))
    r.enter_season("2025-26")
    r.enter_season("2025-26")
    assert r.rating("PHI") == pytest.approx(0.5)


def test_apply_enters_the_results_season():
    r = Ratings(P)
    r.apply(_r("PHI", "NYK", 110, 100))
    r.apply(_r("BOS", "MIA", 100, 100, season="2025-26", gid="0022500001"))
    assert r.rating("PHI") == pytest.approx(0.5)


def test_win_probability_by_hand():
    assert win_probability(0.0, 13.0) == pytest.approx(0.5)
    # Phi(1) = 0.841344746...
    assert win_probability(13.0, 13.0) == pytest.approx(0.8413447461, abs=1e-9)
    assert win_probability(-13.0, 13.0) == pytest.approx(0.1586552539, abs=1e-9)


def test_win_probability_rejects_nonpositive_sigma():
    with pytest.raises(ValueError):
        win_probability(1.0, 0.0)
