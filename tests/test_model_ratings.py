import math
from datetime import date

import pytest

from predictor.model.ratings import (
    FLAT_HOME,
    TEAM_HOME_WINDOW,
    HomeParams,
    RatingParams,
    Ratings,
    Result,
    shrunk_deviation,
    win_probability,
)

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


# --- Home court per team (plan 3, Task 1) ---------------------------------

TP = RatingParams(0.1, 40.0, 0.0, 10)


def _two_games(r, altitude_first=False):
    # Game 1: A at home beats B 110-100. gap 0, league 0, dev 0 -> delta 1.0:
    #   A +1, B -1. resid = 10 (- allowance if altitude). A home [10], B away [-10].
    r.apply(Result("g1", "2023-24", date(2024, 1, 1), "A", "B", 110, 100, False),
            altitude_game=altitude_first)
    # Game 2: B at home beats A 102-100. gap = -1 - 1 = -2, league 10, dev(B) 0
    #   (B has no home games) -> predicted 8, delta 0.1 * (2 - 8) = -0.6:
    #   B -1.6, A +1.6. resid = 2 - (-2) - 10 = -6. B home [-6], A away [6].
    r.apply(Result("g2", "2023-24", date(2024, 1, 2), "B", "A", 102, 100, False))


def test_shrunk_deviation_values():
    # raw 4, n_eff = 1/(1/4 + 1/4) = 2, 4 * 2 / (2 + 2) = 2
    assert shrunk_deviation([4.0] * 4, [0.0] * 4, 2.0) == 2.0
    # raw 4 - 1 = 3, n_eff = 1/(1/2 + 1) = 2/3, 3 * (2/3) / (5/3) = 1.2
    assert shrunk_deviation([3.0, 5.0], [1.0], 1.0) == pytest.approx(1.2)
    assert shrunk_deviation([4.0], [0.0], math.inf) == 0.0
    assert shrunk_deviation([], [1.0], 1.0) == 0.0


def test_home_params_validation():
    assert FLAT_HOME.flat and not HomeParams(0.0).flat
    assert FLAT_HOME.window == TEAM_HOME_WINDOW == 164
    with pytest.raises(ValueError):
        HomeParams(-1.0)
    with pytest.raises(ValueError):
        HomeParams(math.nan)


def test_team_home_court_two_games():
    r = Ratings(TP, HomeParams(0.0))
    _two_games(r)
    # league margins [10, 2] -> 6
    assert r.home_court() == 6.0
    # A: home [10], away [6] -> raw 4, unshrunk -> 6 + 4
    assert r.home_court("A") == pytest.approx(10.0)
    # B: home [-6], away [-10] -> raw 4
    assert r.home_court("B") == pytest.approx(10.0)
    assert r.rating("A") == pytest.approx(1.6)

    half = Ratings(TP, HomeParams(0.5))
    _two_games(half)
    # n_eff 0.5 -> 4 * 0.5 / (0.5 + 0.5) = 2
    assert half.home_court("A") == pytest.approx(8.0)
    s = half.team_home_summary("A")
    assert (s.team, s.home_games, s.away_games) == ("A", 1, 1)
    assert s.raw == pytest.approx(4.0)
    assert s.deviation == pytest.approx(2.0)


def test_altitude_allowance_kept_out_of_team_edge():
    r = Ratings(TP, HomeParams(0.0, altitude_allowance=1.5))
    _two_games(r, altitude_first=True)
    # A home resid 10 - 1.5 = 8.5, away [6] -> raw 2.5 -> 6 + 2.5
    assert r.home_court("A") == pytest.approx(8.5)


def test_neutral_game_leaves_team_edges_untouched():
    r = Ratings(TP, HomeParams(0.0))
    r.apply(Result("g1", "2023-24", date(2024, 1, 1), "C", "D", 110, 100, False))
    before = r.home_court()
    r.apply(Result("g2", "2023-24", date(2024, 1, 2), "A", "B", 120, 100, True))
    assert r.team_home_summary("A").home_games == 0
    assert r.team_home_summary("B").away_games == 0
    assert r.home_court() == before


def test_team_window_keeps_last_164_per_side():
    r = Ratings(TP, HomeParams(0.0))
    for i in range(170):
        r.apply(Result(f"g{i}", "2023-24", date(2024, 1, 1), "A", "B", 101, 100, False))
    assert r.team_home_summary("A").home_games == 164
    assert r.team_home_summary("B").away_games == 164


def test_flat_home_equals_league_value():
    r = Ratings(P)
    assert r.home is FLAT_HOME
    r.apply(_r("PHI", "NYK", 110, 100))
    r.apply(_r("NYK", "BOS", 100, 104))
    r.apply(_r("BOS", "PHI", 120, 99))
    for t in ("PHI", "NYK", "BOS", "MIA"):
        assert r.home_court(t) == r.home_court()
        assert r.deviation(t) == 0.0
