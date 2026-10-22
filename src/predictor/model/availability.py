"""Player availability from the NBA injury report.

Two pieces:

- ``estimate`` turns past listings (a status at a call time, and whether the
  player then played) into a probability of playing per status. For season
  S it uses only seasons strictly before S, so no season's own outcomes leak
  into its probabilities. A status with fewer than ``MIN_SAMPLES`` earlier
  listings keeps its prior.
- ``InjuryHistory`` answers "what did the injury report say about game date
  D as it stood at time T": the latest report with ``observed_at <= T`` that
  has a row for D. A later report that lists only other dates is skipped,
  so the caller falls back to the latest earlier report that lists D.

A player not on the report counts as available (1.0). A status outside the
five known ones uses ``UNKNOWN_STATUS_PRIOR`` (the Questionable prior), is
never treated as available, and is counted per season by
``UnknownStatusCounter`` for the report.
"""

from __future__ import annotations

import bisect
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Literal

from predictor import db

STATUS_PRIORS: dict[str, float] = {
    "Out": 0.0,
    "Doubtful": 0.2,
    "Questionable": 0.5,
    "Probable": 0.9,
    "Available": 1.0,
}
UNKNOWN_STATUS_PRIOR = 0.5
MIN_SAMPLES = 100

Call = Literal["early", "final"]
CALLS: tuple[str, ...] = ("early", "final")


_CANONICAL = {name.lower(): name for name in STATUS_PRIORS}


def _known(status: object) -> str | None:
    """The status as a known key, or None when it is not one of the five.

    Matching ignores surrounding whitespace and case ("  OUT " is "Out").
    """
    if isinstance(status, str):
        return _CANONICAL.get(status.strip().lower())
    return None


@dataclass(frozen=True)
class Listing:
    season: str
    call: Call
    status: str
    played: bool

    def __post_init__(self) -> None:
        if self.call not in CALLS:
            raise ValueError(f"call must be one of {CALLS}, got {self.call!r}")
        # A NaN or None "played" would silently count as played/not played.
        if not isinstance(self.played, bool):
            raise TypeError(f"played must be a bool, got {self.played!r}")
        if not isinstance(self.season, str):
            raise TypeError(f"season must be a str, got {self.season!r}")


@dataclass(frozen=True)
class AvailabilityTable:
    probs: dict[str, float]
    samples: dict[str, int]

    def prob(self, status: str | None) -> float:
        """Probability of playing given a report status.

        None (not on the report) is 1.0. Anything that is not one of the
        five known statuses, including NaN, is ``UNKNOWN_STATUS_PRIOR``.
        """
        if status is None:
            return 1.0
        key = _known(status)
        if key is None:
            return UNKNOWN_STATUS_PRIOR
        p = self.probs.get(key)
        if p is None or not isinstance(p, (int, float)) or math.isnan(p):
            return STATUS_PRIORS[key]
        return float(p)


def estimate(listings: Iterable[Listing], season: str, call: str) -> AvailabilityTable:
    """Played share per known status, from listings of earlier seasons only."""
    if call not in CALLS:
        raise ValueError(f"call must be one of {CALLS}, got {call!r}")
    if not isinstance(season, str):
        raise TypeError(f"season must be a str, got {season!r}")
    n: Counter[str] = Counter()
    played: Counter[str] = Counter()
    for listing in listings:
        if listing.call != call or not listing.season < season:
            continue
        key = _known(listing.status)
        if key is None:
            continue
        n[key] += 1
        if listing.played:
            played[key] += 1
    probs: dict[str, float] = {}
    samples: dict[str, int] = {}
    for status, prior in STATUS_PRIORS.items():
        samples[status] = n[status]
        probs[status] = played[status] / n[status] if n[status] >= MIN_SAMPLES else prior
    return AvailabilityTable(probs=probs, samples=samples)


class UnknownStatusCounter:
    """Per-season counts of status strings outside the five known ones."""

    def __init__(self) -> None:
        self._counts: dict[str, Counter[str]] = defaultdict(Counter)

    def record(self, season: str, status: object) -> bool:
        """Count ``status`` if it is unknown; returns whether it was counted.

        None means "not listed" and is not counted.
        """
        if status is None or _known(status) is not None:
            return False
        self._counts[season][str(status)] += 1
        return True

    def total(self, season: str) -> int:
        return sum(self._counts[season].values()) if season in self._counts else 0

    def report(self) -> dict[str, dict[str, int]]:
        return {s: dict(c) for s, c in sorted(self._counts.items())}


@dataclass(frozen=True)
class Snapshot:
    observed_at: datetime
    statuses: dict[tuple[str, str], str]


class InjuryHistory:
    """Every injury report, indexed by game date, for point-in-time lookups."""

    def __init__(self, rows: Iterable[tuple[datetime, date, str, str, str]]) -> None:
        by_date: dict[date, dict[datetime, dict[tuple[str, str], str]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        for observed_at, game_date, team, player, status in rows:
            if observed_at.tzinfo is None:
                raise ValueError("injury_status observed_at must be timezone-aware")
            observed_at = observed_at.astimezone(UTC)
            by_date[game_date][observed_at][(team, player)] = status
        self._times: dict[date, list[datetime]] = {}
        self._snaps: dict[date, list[Snapshot]] = {}
        all_times: set[datetime] = set()
        for game_date, reports in by_date.items():
            times = sorted(reports)
            all_times.update(times)
            self._times[game_date] = times
            self._snaps[game_date] = [Snapshot(t, dict(reports[t])) for t in times]
        self._all_times = sorted(all_times)

    _COLUMNS = "observed_at, game_date, team, player, status"

    @classmethod
    def from_db(cls, con) -> InjuryHistory:
        """Read the whole table directly (training path, like ``fit._load``)."""
        table = db.POINT_IN_TIME_TABLES["injury_status"]
        return cls(con.execute(f"SELECT {cls._COLUMNS} FROM {table}").fetchall())

    @classmethod
    def from_view(cls, view) -> InjuryHistory:
        """Read through an ``AsOfView``, so nothing after its cutoff is seen."""
        return cls(view.table("injury_status").project(cls._COLUMNS).fetchall())

    def snapshot_at(self, game_date: date, at: datetime) -> Snapshot | None:
        """Latest report with ``observed_at <= at`` that lists ``game_date``."""
        at = db.require_utc(at, "at").astimezone(UTC)
        times = self._times.get(game_date)
        if not times:
            return None
        i = bisect.bisect_right(times, at)
        return self._snaps[game_date][i - 1] if i else None

    def latest_report_at(self, at: datetime) -> datetime | None:
        """Time of the latest report of any game date with ``observed_at <= at``.

        Compared with ``snapshot_at(...).observed_at`` it tells whether the
        snapshot fell back past a later report that did not list the date.
        """
        at = db.require_utc(at, "at").astimezone(UTC)
        i = bisect.bisect_right(self._all_times, at)
        return self._all_times[i - 1] if i else None
