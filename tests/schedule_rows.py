"""Seed one schedule row -- the harness's tip-off source since sub-project 2.5."""

from datetime import timedelta

from predictor import db


def insert_schedule_row(
    con,
    game_id,
    game_date,
    home_team,
    away_team,
    tip_off_utc,
    observed_at=None,
    season="2024-25",
):
    if observed_at is None:
        observed_at = tip_off_utc - timedelta(days=30)
    table = db.POINT_IN_TIME_TABLES["schedule"]
    con.execute(
        f"INSERT INTO {table} (game_id, season, game_date, tip_off_utc, home_team,"
        " away_team, arena_name, arena_city, arena_state, is_neutral_reported,"
        " is_neutral, observed_at) VALUES (?,?,?,?,?,?,?,?,?,FALSE,FALSE,?)",
        [game_id, season, game_date, tip_off_utc, home_team, away_team,
         "Test Arena", "Testville", "TS", observed_at],
    )
