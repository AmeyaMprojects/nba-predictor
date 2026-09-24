from __future__ import annotations

import re
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from predictor import db

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
    """
    table = db.POINT_IN_TIME_TABLES["injury_status"]
    rows = con.execute(
        f"SELECT DISTINCT game_date, team, game_time FROM {table} "
        "WHERE game_date IS NOT NULL AND game_time IS NOT NULL AND game_time <> ''"
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
