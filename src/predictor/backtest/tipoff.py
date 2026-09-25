from __future__ import annotations

import re
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from predictor import db

# This module reads the injury-report point-in-time table directly rather
# than through AsOfView. That is a deliberate exemption, not an oversight: tip-off
# resolution ESTABLISHES the as-of cutoff for everything else, so it
# cannot itself be filtered by a cutoff without circularity -- you need
# the tip-off time before you can know what "before tip-off" means. The
# resolved tip-off datetime is used only to COMPUTE a cutoff -- FIX 4
# (final review, part 1) removed it from ``GameToPredict``, so it is not
# handed to a predictor as a feature at all, and reading it unfiltered
# here cannot leak future information into a model.
EASTERN = ZoneInfo("America/New_York")

# '07:00 (ET)' and '08:00(ET)' both occur -- the two PDF layouts differ in
# spacing. Times are a 12-hour clock with no AM/PM marker.
_TIME = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*\(ET\)\s*$")


def parse_game_time(raw: str, game_date: date) -> datetime | None:
    """Resolve an injury-report game time to a UTC instant.

    NBA games run roughly noon to 10:30pm Eastern, so a bare hour of 12 means
    noon and 1-11 mean PM. Returns None rather than guessing when the value
    cannot be parsed -- a wrong tip-off silently invalidates a backtest.
    """
    if not raw:
        return None
    m = _TIME.match(raw)
    if not m:
        return None
    hour, minute = int(m.group(1)), int(m.group(2))
    if not (1 <= hour <= 12 and 0 <= minute <= 59):
        return None
    hour24 = hour if hour == 12 else hour + 12
    local = datetime(
        game_date.year, game_date.month, game_date.day, hour24, minute,
        tzinfo=EASTERN,
    )
    return local.astimezone(UTC)


def tipoff_index(con) -> dict[tuple[date, str], datetime]:
    """Map (game_date, team) -> tip-off instant, from the injury reports.

    The injury report is the only place a tip-off time exists in this schema.

    Different report vintages for the same (game_date, team) can disagree
    -- a later report can carry a corrected time (real rescheduled games
    swing by hours between vintages), and in the real archive 8
    (game_date, team) pairs do disagree across vintages this way. Without a
    deterministic tie-break, `SELECT DISTINCT` has no defined row order and
    the Python dict takes whichever row DuckDB happens to emit last, so the
    same query can silently return a different tip-off across runs.

    FIX 4 (final review, part 1): the `QUALIFY` clause below breaks the tie
    by keeping the row with the EARLIEST `observed_at` per (game_date,
    team) -- not the latest, which this used to do. A game's tip-off must
    be resolvable from what was knowable as of the earliest filing that
    states it; taking the latest vintage instead made the resolved value
    (and therefore the cutoff derived from it: `cutoff = tip - buffer`) a
    post-hoc quantity that could depend on a report filed after this
    game's own cutoff -- for 2 of 7,200 games in the real archive, the
    latest-vintage tip-off differed from the latest filing actually
    available before the cutoff. Taking the earliest vintage instead means
    the resolved value can never depend on information published later,
    which is exactly the property this harness requires everywhere else.
    """
    table = db.POINT_IN_TIME_TABLES["injury_status"]
    rows = con.execute(
        f"SELECT game_date, team, game_time FROM {table} "
        "WHERE game_date IS NOT NULL AND game_time IS NOT NULL AND game_time <> '' "
        "QUALIFY row_number() OVER "
        "(PARTITION BY game_date, team ORDER BY observed_at ASC) = 1"
    ).fetchall()
    index: dict[tuple[date, str], datetime] = {}
    for game_date, team, raw in rows:
        parsed = parse_game_time(raw, game_date)
        if parsed is not None:
            index[(game_date, team)] = parsed
    return index


def resolve_tipoff(
    index: dict[tuple[date, str], datetime],
    game_date: date,
    home_team: str,
    away_team: str,
) -> datetime | None:
    """Tip-off for a game, from either team's injury-report entry."""
    return index.get((game_date, home_team)) or index.get((game_date, away_team))
