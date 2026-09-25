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
    (game_date, team) pairs do disagree across vintages this way.

    FIX 14 (final review, part 3) -- CRITICAL regression fix. FIX 4 (final
    review, part 1) resolved a conflict by taking the tip-off from the
    EARLIEST-OBSERVED filing, reasoning that "the resolved value can never
    depend on information published later." That reasoning was backwards
    and it put the computed cutoff AFTER the game's real tip-off for at
    least 2 games verified in the real archive (0022200161, 0022400521): a
    later filing corrected the tip-off to an EARLIER clock time (a
    placeholder slot replaced by the real broadcast time), and ignoring
    that correction because it arrived "later" left the harness computing
    `cutoff = 07:00 ET - buffer` for a game that actually tipped off at
    05:30 ET -- an hour before the naive cutoff. `AsOfView` for those games
    could then contain injury/news rows published while the game was
    already in progress. The leak guard on the FINAL row does not catch
    this: every FINAL row in the current archive is reconstructed at
    `game_date + 36h`, always long after any conceivable cutoff, so the
    guard can never trip on a bad tip-off specifically -- only a genuinely
    live FINAL timestamp would.

    The fix: resolve a game's tip-off as the MINIMUM parsed clock time
    across ALL vintages for that (game_date, team), not the
    earliest-FILED one. This is safe in both directions, which is why it
    is correct without needing to know which direction a correction runs:
      - A later filing that moves a game EARLIER must be honoured, because
        ignoring it (as the earliest-observed rule did) puts the cutoff
        AFTER the real tip-off -- a leak.
      - A later filing that moves a game LATER can be safely ignored,
        because using the earlier time only makes the cutoff MORE
        conservative (earlier than strictly necessary), never later than
        the real tip-off.
    Taking the minimum satisfies both simultaneously. It is deliberately
    pessimistic: where vintages disagree, the harness predicts from the
    EARLIEST time the game could plausibly have started, not from
    whichever filing happened to be seen first or last.

    Consequently the resolved value CAN depend on information published
    after the earliest filing -- and must, whenever a later filing moves
    the game earlier. What it can never do is resolve to a time LATER
    than any recorded vintage, which is the actual property this harness
    needs: the cutoff derived from it must never land after the true
    tip-off.
    """
    table = db.POINT_IN_TIME_TABLES["injury_status"]
    rows = con.execute(
        f"SELECT game_date, team, game_time FROM {table} "
        "WHERE game_date IS NOT NULL AND game_time IS NOT NULL AND game_time <> ''"
    ).fetchall()
    index: dict[tuple[date, str], datetime] = {}
    for game_date, team, raw in rows:
        parsed = parse_game_time(raw, game_date)
        if parsed is None:
            continue
        key = (game_date, team)
        current = index.get(key)
        if current is None or parsed < current:
            index[key] = parsed
    return index


def resolve_tipoff(
    index: dict[tuple[date, str], datetime],
    game_date: date,
    home_team: str,
    away_team: str,
) -> datetime | None:
    """Tip-off for a game, from either team's injury-report entry."""
    return index.get((game_date, home_team)) or index.get((game_date, away_team))
