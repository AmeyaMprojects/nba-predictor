"""NBA arena cities: coordinates, time zones, altitude.

Static reference data for the travel and altitude adjustments (spec 3,
"First delivery design"). Keyed by the schedule's `arena_city` with any
", Country" suffix removed ("Mexico City, Mexico" -> "Mexico City").
Coordinates are the arena's, to three decimals. Every competitive-game city
in the archive from 2014-15 on is covered (enforced by
tests/test_model_venues.py against the real archive).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

_EARTH_RADIUS_KM = 6371.0


@dataclass(frozen=True)
class City:
    lat: float
    lon: float
    tz: str


CITIES: dict[str, City] = {
    "Atlanta": City(33.757, -84.396, "America/New_York"),
    "Austin": City(30.282, -97.732, "America/Chicago"),
    "Berlin": City(52.508, 13.443, "Europe/Berlin"),
    "Boston": City(42.366, -71.062, "America/New_York"),
    "Brooklyn": City(40.683, -73.975, "America/New_York"),
    "Charlotte": City(35.225, -80.839, "America/New_York"),
    "Chicago": City(41.881, -87.674, "America/Chicago"),
    "Cleveland": City(41.496, -81.688, "America/New_York"),
    "Dallas": City(32.790, -96.810, "America/Chicago"),
    "Denver": City(39.749, -105.008, "America/Denver"),
    "Detroit": City(42.341, -83.055, "America/Detroit"),
    "Houston": City(29.751, -95.362, "America/Chicago"),
    "Indianapolis": City(39.764, -86.155, "America/Indiana/Indianapolis"),
    "Inglewood": City(33.945, -118.343, "America/Los_Angeles"),
    "Las Vegas": City(36.103, -115.178, "America/Los_Angeles"),
    "London": City(51.503, 0.003, "Europe/London"),
    "Los Angeles": City(34.043, -118.267, "America/Los_Angeles"),
    "Manchester": City(53.486, -2.199, "Europe/London"),
    "Memphis": City(35.138, -90.051, "America/Chicago"),
    "Mexico City": City(19.404, -99.096, "America/Mexico_City"),
    "Miami": City(25.781, -80.188, "America/New_York"),
    "Milwaukee": City(43.045, -87.917, "America/Chicago"),
    "Minneapolis": City(44.979, -93.276, "America/Chicago"),
    "New Orleans": City(29.949, -90.082, "America/Chicago"),
    "New York": City(40.751, -73.993, "America/New_York"),
    "Oakland": City(37.750, -122.203, "America/Los_Angeles"),
    "Oklahoma City": City(35.463, -97.515, "America/Chicago"),
    "Orlando": City(28.539, -81.384, "America/New_York"),
    "Paris": City(48.838, 2.379, "Europe/Paris"),
    "Philadelphia": City(39.901, -75.172, "America/New_York"),
    "Phoenix": City(33.446, -112.071, "America/Phoenix"),
    "Portland": City(45.532, -122.667, "America/Los_Angeles"),
    "Sacramento": City(38.580, -121.500, "America/Los_Angeles"),
    "Salt Lake City": City(40.768, -111.901, "America/Denver"),
    "San Antonio": City(29.427, -98.438, "America/Chicago"),
    "San Francisco": City(37.768, -122.388, "America/Los_Angeles"),
    "Tampa": City(27.943, -82.452, "America/New_York"),
    "Toronto": City(43.643, -79.379, "America/Toronto"),
    "Washington": City(38.898, -77.021, "America/New_York"),
}

# Arenas high enough that visitors are measurably affected.
ALTITUDE_CITIES: frozenset[str] = frozenset({"Denver", "Salt Lake City"})


def city_key(arena_city: str | None) -> str | None:
    if arena_city is None:
        return None
    key = arena_city.split(",")[0].strip()
    return key or None


def distance_km(a: City, b: City) -> float:
    """Great-circle (haversine) distance."""
    lat1, lon1, lat2, lon2 = map(math.radians, (a.lat, a.lon, b.lat, b.lon))
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def utc_offset_hours(city: City, on: date) -> float:
    """The city's UTC offset at noon local time on `on`, in hours."""
    local = datetime.combine(on, time(12, 0), tzinfo=ZoneInfo(city.tz))
    return local.utcoffset().total_seconds() / 3600
