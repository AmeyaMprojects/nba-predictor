"""Rest, travel and altitude adjustments, in points (spec 3, Stage 1).

Every input is a fact of games already played (dates and arena cities from
the schedule), so nothing here can see a result. Features are expressed
home-minus-away, so one coefficient per feature gives the home team's edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from predictor.model.cities import ALTITUDE_CITIES, CITIES, distance_km, utc_offset_hours
from predictor.model.venues import VenueIndex

# Beyond this many days since the previous game (All-Star break, season
# start), a team is treated as travelling from home: no travel penalty.
TRAVEL_GAP_DAYS = 7

FEATURE_NAMES = ("back_to_back", "third_in_four", "travel_per_1000km", "tz_per_hour", "altitude")


@dataclass(frozen=True)
class Situation:
    back_to_back: bool
    third_in_four: bool
    travel_km: float
    tz_hours: float
    has_history: bool
    unknown_city: str | None


@dataclass(frozen=True)
class Coefficients:
    back_to_back: float
    third_in_four: float
    travel_per_1000km: float
    tz_per_hour: float
    altitude: float


@dataclass(frozen=True)
class AdjustmentTerms:
    rest: float
    travel: float
    altitude: float


def situation(venues: VenueIndex, team: str, game_date: date, city: str | None) -> Situation:
    recent = venues.recent_games(team, game_date, n=2)
    if not recent:
        return Situation(False, False, 0.0, 0.0, has_history=False, unknown_city=None)
    previous = recent[-1]
    gap = (game_date - previous.game_date).days
    back_to_back = gap == 1
    # This game is the third in four nights when the two previous games both
    # fall within the three days before it.
    third_in_four = len(recent) == 2 and (game_date - recent[0].game_date).days <= 3
    travel_km = tz_hours = 0.0
    unknown: str | None = None
    if gap <= TRAVEL_GAP_DAYS:
        start = CITIES.get(previous.city) if previous.city else None
        end = CITIES.get(city) if city else None
        if start is None or end is None:
            unknown = (previous.city if start is None else city) or "(no city recorded)"
        else:
            travel_km = distance_km(start, end)
            tz_hours = abs(utc_offset_hours(start, game_date) - utc_offset_hours(end, game_date))
    return Situation(back_to_back, third_in_four, travel_km, tz_hours, True, unknown)


def is_altitude_game(city: str | None, neutral: bool) -> bool:
    return (not neutral) and city in ALTITUDE_CITIES


def feature_vector(
    home: Situation, away: Situation, altitude_game: bool
) -> tuple[float, float, float, float, float]:
    return (
        float(home.back_to_back) - float(away.back_to_back),
        float(home.third_in_four) - float(away.third_in_four),
        (home.travel_km - away.travel_km) / 1000.0,
        home.tz_hours - away.tz_hours,
        1.0 if altitude_game else 0.0,
    )


def terms(c: Coefficients, x: tuple[float, ...]) -> AdjustmentTerms:
    return AdjustmentTerms(
        rest=c.back_to_back * x[0] + c.third_in_four * x[1],
        travel=c.travel_per_1000km * x[2] + c.tz_per_hour * x[3],
        altitude=c.altitude * x[4],
    )


def astuple_terms(c: Coefficients, x: tuple[float, ...]) -> tuple[float, float, float]:
    t = terms(c, x)
    return (t.rest, t.travel, t.altitude)
