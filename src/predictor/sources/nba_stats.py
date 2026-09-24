from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
from tenacity import retry, stop_after_attempt, wait_exponential

from predictor import db


@dataclass(frozen=True)
class GameRow:
    game_id: str
    season: str
    game_date: date
    home_team: str
    away_team: str
    home_points: int | None
    away_points: int | None
    status: str


@dataclass(frozen=True)
class DroppedGame:
    """A GAME_ID group that could not be resolved into a single GameRow.

    Recorded (and always printed) rather than silently discarded -- see
    the module docstring note on neutral-site games below. Callers that
    want the list (rather than just the printed log) pass a mutable list
    via `pair_team_rows(..., dropped=...)`.
    """

    game_id: str
    matchup: str
    reason: str


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=30))
def fetch_season(season: str) -> pd.DataFrame:
    """Fetch every team-game row for a season. Two rows per game."""
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season, league_id_nullable="00", timeout=60
    )
    return finder.get_data_frames()[0]


def _as_int(value) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _parse_matchup(matchup: str) -> tuple[str, str] | None:
    """Parse a MATCHUP string into (home_abbr, away_abbr), or None.

    nba_api emits two textual forms:
      "HOME vs. AWAY"  -- the home team's own row.
      "AWAY @ HOME"    -- the away team's row.

    Critically, for a NEUTRAL-SITE game (NBA Cup semifinals/final, Paris/
    London games, some neutral-site preseason) BOTH rows carry the exact
    same "AWAY @ HOME" text -- neither row uses "vs.", because neither
    team is truly the home team. The "@" form alone still fully encodes
    the pairing (and which side the schedule nominally calls "home"), so
    it is parsed the same way regardless of which row it came from. This
    is what lets a neutral-site group resolve correctly instead of being
    dropped by a rule that requires seeing a "vs." row (the bug found in
    task-10 review: 9 real, played neutral-site games -- including an NBA
    Cup final -- were silently dropped every ingest).
    """
    if not isinstance(matchup, str):
        return None
    if " vs. " in matchup:
        home, away = matchup.split(" vs. ", 1)
        return home.strip(), away.strip()
    if " @ " in matchup:
        away, home = matchup.split(" @ ", 1)
        return home.strip(), away.strip()
    return None


def _log_dropped(
    game_id: str,
    matchup: str,
    reason: str,
    dropped: list[DroppedGame] | None,
) -> None:
    # "Never silently lose data -- surface loudly": every group that fails
    # to become a GameRow is printed here, unconditionally, whether or not
    # a caller is also collecting the structured list.
    print(f"nba_stats: DROPPED game_id={game_id} matchup=[{matchup}] -- {reason}")
    if dropped is not None:
        dropped.append(DroppedGame(game_id=game_id, matchup=matchup, reason=reason))


def pair_team_rows(
    df: pd.DataFrame,
    season: str = "",
    dropped: list[DroppedGame] | None = None,
) -> list[GameRow]:
    games: list[GameRow] = []
    for game_id, group in df.groupby("GAME_ID"):
        game_id = str(game_id)
        matchup_text = ", ".join(str(m) for m in group["MATCHUP"])

        if len(group) != 2:
            _log_dropped(
                game_id, matchup_text, f"expected 2 team-rows, got {len(group)}", dropped
            )
            continue

        parsed = [_parse_matchup(m) for m in group["MATCHUP"]]
        if any(p is None for p in parsed):
            _log_dropped(
                game_id, matchup_text, "could not parse MATCHUP into home/away teams", dropped
            )
            continue
        if parsed[0] != parsed[1]:
            _log_dropped(
                game_id,
                matchup_text,
                f"rows disagree on home/away: {parsed[0]} vs {parsed[1]}",
                dropped,
            )
            continue
        home_abbr, away_abbr = parsed[0]

        home_rows = group[group["TEAM_ABBREVIATION"] == home_abbr]
        away_rows = group[group["TEAM_ABBREVIATION"] == away_abbr]
        if len(home_rows) != 1 or len(away_rows) != 1:
            _log_dropped(
                game_id,
                matchup_text,
                f"TEAM_ABBREVIATION did not match parsed home={home_abbr!r} "
                f"away={away_abbr!r}",
                dropped,
            )
            continue
        home = home_rows.iloc[0]
        away = away_rows.iloc[0]

        home_points = _as_int(home["PTS"])
        away_points = _as_int(away["PTS"])
        played = home_points is not None and away_points is not None

        games.append(
            GameRow(
                game_id=game_id,
                season=season,
                game_date=pd.to_datetime(home["GAME_DATE"]).date(),
                home_team=str(home["TEAM_ABBREVIATION"]),
                away_team=str(away["TEAM_ABBREVIATION"]),
                home_points=home_points,
                away_points=away_points,
                status="FINAL" if played else "SCHEDULED",
            )
        )
    return games


def ingest_season(con, season: str, observed_at: datetime | None = None) -> int:
    observed_at = db.require_utc(observed_at or datetime.now(UTC), "observed_at")
    dropped: list[DroppedGame] = []
    games = pair_team_rows(fetch_season(season), season, dropped=dropped)
    # Resolved through db.POINT_IN_TIME_TABLES rather than spelled as a
    # literal here -- the physical "_raw" table names are only allowed to
    # appear as string literals in db.py/asof.py (see
    # test_no_physical_table_name_appears_outside_db_and_asof); this
    # ingestion module must not name the physical table directly either.
    # Mirrors the pattern already used by predictor.sources.injury_report.
    table = db.POINT_IN_TIME_TABLES["games"]
    insert_sql = (
        f"INSERT OR REPLACE INTO {table} (game_id, season, game_date, home_team,"
        " away_team, home_points, away_points, status, observed_at)"
        " VALUES (?,?,?,?,?,?,?,?,?)"
    )
    for game in games:
        con.execute(
            insert_sql,
            [
                game.game_id,
                season,
                game.game_date,
                game.home_team,
                game.away_team,
                game.home_points,
                game.away_points,
                game.status,
                observed_at,
            ],
        )
    if dropped:
        # Loud, unconditional report of anything that did NOT make it into
        # the database for this season -- individual DROPPED lines were
        # already printed by pair_team_rows/_log_dropped above; this is the
        # summary a caller (CLI or otherwise) sees at the end of the run.
        print(
            f"nba_stats: WARNING -- {len(dropped)} game(s) for season {season} "
            "could not be paired into a game row and were NOT ingested: "
            f"{[d.game_id for d in dropped]}"
        )
    return len(games)
