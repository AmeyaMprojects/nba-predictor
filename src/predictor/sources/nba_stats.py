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


def pair_team_rows(df: pd.DataFrame, season: str = "") -> list[GameRow]:
    games: list[GameRow] = []
    for game_id, group in df.groupby("GAME_ID"):
        if len(group) != 2:
            continue  # unpaired row: drop rather than invent an opponent

        # "OKC vs. IND" is the home listing; "IND @ OKC" is the away listing.
        home_mask = group["MATCHUP"].str.contains("vs.", regex=False)
        if home_mask.sum() != 1:
            continue
        home = group[home_mask].iloc[0]
        away = group[~home_mask].iloc[0]

        home_points = _as_int(home["PTS"])
        away_points = _as_int(away["PTS"])
        played = home_points is not None and away_points is not None

        games.append(
            GameRow(
                game_id=str(game_id),
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
    games = pair_team_rows(fetch_season(season), season)
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
    return len(games)
