"""Static venue facts from the schedule, read outside AsOfView.

Spec 1.1: arena and neutral-site columns are static venue facts, not
outcome-bearing, safe to read at any time. They cannot come through
AsOfView: every schedule row was observed on 2026-09-27 or later, so a view
cut at any historical cutoff would hide them all. A team's past game DATES
are likewise facts of games already played. This module never reads scores
or game status, and never exposes a tip-off time.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

from predictor import db
from predictor.model.cities import city_key

COMPETITIVE_PREFIXES = ("002", "004", "005", "006")


@dataclass(frozen=True)
class Venue:
    game_id: str
    game_date: date
    home_team: str
    away_team: str
    city: str | None
    is_neutral: bool


class VenueIndex:
    def __init__(self, venues: Iterable[Venue]) -> None:
        self._by_id: dict[str, Venue] = {v.game_id: v for v in venues}
        self._by_team: dict[str, list[Venue]] = {}
        ordered = sorted(self._by_id.values(), key=lambda v: (v.game_date, v.game_id))
        for v in ordered:
            if v.game_id[:3] not in COMPETITIVE_PREFIXES:
                continue
            self._by_team.setdefault(v.home_team, []).append(v)
            self._by_team.setdefault(v.away_team, []).append(v)
        self._dates = {t: [v.game_date for v in vs] for t, vs in self._by_team.items()}

    @classmethod
    def from_db(cls, con) -> VenueIndex:
        table = db.POINT_IN_TIME_TABLES["schedule"]
        rows = con.execute(
            f"SELECT game_id, game_date, home_team, away_team, arena_city, is_neutral "
            f"FROM {table} "
            "QUALIFY row_number() OVER (PARTITION BY game_id ORDER BY observed_at DESC) = 1"
        ).fetchall()
        return cls(
            Venue(gid, gd, home, away, city_key(city), bool(neutral))
            for gid, gd, home, away, city, neutral in rows
        )

    def venue(self, game_id: str) -> Venue | None:
        return self._by_id.get(game_id)

    def recent_games(self, team: str, before: date, n: int) -> list[Venue]:
        """Up to `n` most recent competitive games strictly before `before`, oldest first."""
        dates = self._dates.get(team)
        if not dates:
            return []
        end = bisect_left(dates, before)
        return self._by_team[team][max(0, end - n):end]
