from __future__ import annotations

import re
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from predictor import db

# This module reads the schedule point-in-time table directly rather than
# through AsOfView. That is a deliberate exemption, not an oversight:
# tip-off resolution ESTABLISHES the as-of cutoff for everything else, so it
# cannot itself be filtered by a cutoff without circularity. The resolved
# tip-off is used only to COMPUTE a cutoff -- it is not carried by
# ``GameToPredict`` (FIX 4) and is never a predictor feature.
#
# Sub-project 2.5: the schedule (sources/schedule.py) replaced the
# injury-report PDFs as the tip-off source. The PDFs' CDN froze after
# 2025-12-21, and they resolved 7,200 of 8,289 games; the schedule
# resolves all 8,289. `parse_game_time` below remains only for
# tests/test_tipoff_crosscheck.py, which checks the two sources against
# each other -- a second source without a second runtime code path
# producing cutoffs, which is where both earlier tip-off leaks lived.
EASTERN = ZoneInfo("America/New_York")

# '07:00 (ET)' and '08:00(ET)' both occur -- the two PDF layouts differ in
# spacing. Times are a 12-hour clock with no AM/PM marker.
_TIME = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*\(ET\)\s*$")


def parse_game_time(raw: str, game_date: date) -> datetime | None:
    """Resolve an injury-report game time to a UTC instant (cross-check only).

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
    """Map (game_date, team) -> tip-off instant, from the league schedule.

    Every schedule vintage contributes, and each key keeps the MINIMUM
    instant across them. This is FIX 14's rule, carried over from the
    injury reports: a later vintage that moves a game EARLIER must be
    honoured (ignoring it puts the cutoff after the real tip-off -- a
    leak), while one that moves it LATER is safely ignored (the earlier
    time only makes the cutoff more conservative). The minimum satisfies
    both without knowing which way a change runs.

    Rows whose tip_off_utc is NULL (the league lists the time as TBD) never
    enter the index -- see sources/schedule.py for why a placeholder must
    not.

    Both teams of a game are keyed to it, so ``resolve_tipoff``'s
    minimum-across-both-teams rule (FIX 23) is preserved unchanged.
    """
    table = db.POINT_IN_TIME_TABLES["schedule"]
    rows = con.execute(
        f"SELECT game_date, home_team, away_team, tip_off_utc FROM {table} "
        "WHERE tip_off_utc IS NOT NULL"
    ).fetchall()
    index: dict[tuple[date, str], datetime] = {}
    for game_date, home_team, away_team, tip in rows:
        for team in (home_team, away_team):
            key = (game_date, team)
            current = index.get(key)
            if current is None or tip < current:
                index[key] = tip
    return index


def resolve_tipoff(
    index: dict[tuple[date, str], datetime],
    game_date: date,
    home_team: str,
    away_team: str,
) -> datetime | None:
    """Tip-off for a game: the MINIMUM across both teams' index entries.

    FIX 23 (final review, part 4) -- CRITICAL regression fix. This used to
    be `index.get(home) or index.get(away)`: when the home team had an
    entry, the away team's was never consulted, even when it recorded an
    earlier (and therefore more conservative / correct) tip-off. Verified
    in the archive: 0022400624 (2025-01-23, MIA at MIL) and 0021900701
    (2020-01-28, BOS at MIA) both have an away-team vintage earlier than
    the home team's, and the home-first rule silently discarded it --
    resolving the tip-off (and therefore the cutoff derived from it) to a
    time AFTER the away team's own recorded vintage, the exact leak class
    this harness exists to prevent. Each team's own index entry is already
    the minimum across that team's vintages (see `tipoff_index`); taking
    the minimum of the two here extends that same guarantee across both
    teams, so the resolved value can never land after any vintage recorded
    for either one. `None` is returned only when NEITHER team has an entry.
    """
    home = index.get((game_date, home_team))
    away = index.get((game_date, away_team))
    if home is None:
        return away
    if away is None:
        return home
    return min(home, away)
