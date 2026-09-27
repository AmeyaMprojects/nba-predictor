"""Build small, fully controlled archives for model tests.

Each game gets a SCHEDULED row (7 days before, reconstructed), a FINAL row
when scores are given (game_date + 1 day, 12:00 UTC -- the same stamp the
real archive uses), and a schedule row (7pm ET tip-off, observed
2026-09-27, like the real backfill).
"""

from datetime import UTC, date, datetime, time, timedelta

from predictor import db

SCHEDULE_OBSERVED = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)


def fixture_con(tmp_path):
    con = db.connect(tmp_path / "model.duckdb")
    db.migrate(con)
    return con


def add_game(
    con,
    game_id,
    season,
    game_date,
    home,
    away,
    home_pts=None,
    away_pts=None,
    *,
    city="Boston",
    neutral=False,
    final_observed_at=None,
):
    games = db.POINT_IN_TIME_TABLES["games"]
    sched = db.POINT_IN_TIME_TABLES["schedule"]
    scheduled_at = datetime.combine(game_date - timedelta(days=7), time(12), tzinfo=UTC)
    con.execute(
        f"INSERT INTO {games} (game_id, season, game_date, home_team, away_team,"
        " home_points, away_points, status, reconstructed, observed_at)"
        " VALUES (?,?,?,?,?,NULL,NULL,'SCHEDULED',TRUE,?)",
        [game_id, season, game_date, home, away, scheduled_at],
    )
    if home_pts is not None:
        if final_observed_at is None:
            final_observed_at = datetime.combine(
                game_date + timedelta(days=1), time(12), tzinfo=UTC
            )
        con.execute(
            f"INSERT INTO {games} (game_id, season, game_date, home_team, away_team,"
            " home_points, away_points, status, reconstructed, observed_at)"
            " VALUES (?,?,?,?,?,?,?,'FINAL',TRUE,?)",
            [game_id, season, game_date, home, away, home_pts, away_pts, final_observed_at],
        )
    tip = datetime.combine(game_date + timedelta(days=1), time(0), tzinfo=UTC)
    con.execute(
        f"INSERT INTO {sched} (game_id, season, game_date, tip_off_utc, home_team,"
        " away_team, arena_name, arena_city, arena_state, is_neutral_reported,"
        " is_neutral, observed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [game_id, season, game_date, tip, home, away, "Arena", city, None,
         neutral, neutral, SCHEDULE_OBSERVED],
    )
