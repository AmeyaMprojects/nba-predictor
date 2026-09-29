import math
from datetime import date

import pytest

from predictor.model import cities
from predictor.model.cities import City


def test_one_degree_of_longitude_on_the_equator():
    # 2 * pi * 6371.0 / 360 = 111.19492664...
    d = cities.distance_km(City(0.0, 0.0, "UTC"), City(0.0, 1.0, "UTC"))
    assert d == pytest.approx(111.19492664, abs=1e-6)


def test_distance_to_self_is_zero():
    boston = cities.CITIES["Boston"]
    assert cities.distance_km(boston, boston) == 0.0


def test_distance_is_symmetric():
    a, b = cities.CITIES["Boston"], cities.CITIES["Los Angeles"]
    assert cities.distance_km(a, b) == pytest.approx(cities.distance_km(b, a))


def test_new_york_offset_follows_daylight_saving():
    ny = cities.CITIES["New York"]
    assert cities.utc_offset_hours(ny, date(2025, 1, 15)) == -5.0
    assert cities.utc_offset_hours(ny, date(2025, 7, 15)) == -4.0


def test_phoenix_does_not_observe_daylight_saving():
    phx = cities.CITIES["Phoenix"]
    assert cities.utc_offset_hours(phx, date(2025, 1, 15)) == -7.0
    assert cities.utc_offset_hours(phx, date(2025, 7, 15)) == -7.0


@pytest.mark.parametrize(
    "raw, key",
    [
        ("Mexico City, Mexico", "Mexico City"),
        ("Mexico City", "Mexico City"),
        ("  Boston ", "Boston"),
        ("", None),
        (None, None),
    ],
)
def test_city_key_normalises_the_schedule_text(raw, key):
    assert cities.city_key(raw) == key


def test_altitude_cities():
    assert cities.ALTITUDE_CITIES == {"Denver", "Salt Lake City"}
    assert all(c in cities.CITIES for c in cities.ALTITUDE_CITIES)


def test_every_city_has_a_real_time_zone():
    for name, city in cities.CITIES.items():
        assert isinstance(cities.utc_offset_hours(city, date(2025, 1, 15)), float), name
        assert -90 <= city.lat <= 90 and -180 <= city.lon <= 180, name
