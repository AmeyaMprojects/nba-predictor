from datetime import date

import pytest

from predictor.model import adjustments as adj
from predictor.model.cities import CITIES, distance_km
from predictor.model.venues import Venue, VenueIndex


def _v(gid, d, home, away, city):
    return Venue(gid, d, home, away, city, False)


IDX = VenueIndex([
    _v("0022400001", date(2025, 1, 10), "PHI", "NYK", "Philadelphia"),
    _v("0022400002", date(2025, 1, 12), "BOS", "PHI", "Boston"),
    _v("0022400003", date(2025, 1, 13), "PHI", "MIA", "Philadelphia"),
    _v("0022400004", date(2025, 1, 30), "LAL", "NYK", "Los Angeles"),
    _v("0022400005", date(2025, 1, 12), "NYK", "ORL", "Atlantis"),
])


def test_no_history_is_rested_and_untravelled():
    s = adj.situation(IDX, "CHI", date(2025, 1, 14), "Chicago")
    assert s == adj.Situation(False, False, 0.0, 0.0, has_history=False, unknown_city=None)


def test_back_to_back_and_third_in_four_by_hand():
    # PHI played 01-12 (Boston) and 01-13 (Philadelphia); game on 01-14.
    s = adj.situation(IDX, "PHI", date(2025, 1, 14), "Philadelphia")
    assert s.back_to_back is True          # previous game 1 day earlier
    assert s.third_in_four is True         # 01-12 and 01-13 are within 3 days
    assert s.travel_km == 0.0              # Philadelphia -> Philadelphia
    assert s.tz_hours == 0.0


def test_two_days_off_is_neither_by_hand():
    s = adj.situation(IDX, "PHI", date(2025, 1, 16), "Philadelphia")
    assert s.back_to_back is False
    assert s.third_in_four is False        # 01-12 is 4 days before 01-16


def test_travel_distance_and_time_zones_by_hand():
    # PHI's last game was in Philadelphia (01-13); now playing in Los Angeles.
    s = adj.situation(IDX, "PHI", date(2025, 1, 15), "Los Angeles")
    expected = distance_km(CITIES["Philadelphia"], CITIES["Los Angeles"])
    assert s.travel_km == pytest.approx(expected)
    assert s.tz_hours == 3.0               # UTC-5 -> UTC-8 in January


def test_long_break_means_no_travel():
    # NYK's last game before 01-30 is 01-12, 18 days earlier.
    s = adj.situation(IDX, "NYK", date(2025, 1, 30), "Los Angeles")
    assert s.travel_km == 0.0 and s.tz_hours == 0.0 and s.has_history is True


def test_unknown_city_is_reported_not_guessed():
    # NYK's previous game (01-12) was in "Atlantis", not in the table.
    s = adj.situation(IDX, "NYK", date(2025, 1, 14), "New York")
    assert s.travel_km == 0.0 and s.tz_hours == 0.0
    assert s.unknown_city == "Atlantis"


def test_missing_game_city_is_reported():
    s = adj.situation(IDX, "PHI", date(2025, 1, 15), None)
    assert s.unknown_city == "(no city recorded)"


def test_altitude_game():
    assert adj.is_altitude_game("Denver", neutral=False) is True
    assert adj.is_altitude_game("Salt Lake City", neutral=False) is True
    assert adj.is_altitude_game("Denver", neutral=True) is False
    assert adj.is_altitude_game("Boston", neutral=False) is False
    assert adj.is_altitude_game(None, neutral=False) is False


def test_feature_vector_is_home_minus_away_by_hand():
    home = adj.Situation(True, False, 1500.0, 2.0, True, None)
    away = adj.Situation(False, True, 500.0, 3.0, True, None)
    assert adj.feature_vector(home, away, altitude_game=True) == (1.0, -1.0, 1.0, -1.0, 1.0)


def test_terms_by_hand():
    c = adj.Coefficients(back_to_back=-2.0, third_in_four=-1.0,
                         travel_per_1000km=-0.5, tz_per_hour=-0.25, altitude=1.5)
    t = adj.terms(c, (1.0, -1.0, 2.0, -1.0, 1.0))
    assert t.rest == pytest.approx(-2.0 * 1 + -1.0 * -1)       # -1.0
    assert t.travel == pytest.approx(-0.5 * 2 + -0.25 * -1)    # -0.75
    assert t.altitude == pytest.approx(1.5)


def test_feature_names_match_coefficients():
    import dataclasses
    assert adj.FEATURE_NAMES == tuple(f.name for f in dataclasses.fields(adj.Coefficients))
