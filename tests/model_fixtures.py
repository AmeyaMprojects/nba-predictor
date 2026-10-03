"""Build small, fully controlled archives for model tests.

Each game gets a SCHEDULED row (7 days before, reconstructed), a FINAL row
when scores are given (game_date + 1 day, 12:00 UTC -- the same stamp the
real archive uses), and a schedule row (7pm ET tip-off, observed
2026-09-27, like the real backfill).
"""

from datetime import UTC, date, datetime, time, timedelta

from predictor import db
from predictor.model import settings as ms

SCHEDULE_OBSERVED = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)

TEAMS = ["PHI", "NYK", "BOS", "MIA"]


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
    reconstructed=True,
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
            " VALUES (?,?,?,?,?,?,?,'FINAL',?,?)",
            [game_id, season, game_date, home, away, home_pts, away_pts,
             reconstructed, final_observed_at],
        )
    tip = datetime.combine(game_date + timedelta(days=1), time(0), tzinfo=UTC)
    con.execute(
        f"INSERT INTO {sched} (game_id, season, game_date, tip_off_utc, home_team,"
        " away_team, arena_name, arena_city, arena_state, is_neutral_reported,"
        " is_neutral, observed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [game_id, season, game_date, tip, home, away, "Arena", city, None,
         neutral, neutral, SCHEDULE_OBSERVED],
    )


def _season_games(con, season, start, n_days, home_edge, gid_prefix="002"):
    """A tiny round-robin: every day two games, home team wins by `home_edge`
    plus a deterministic team-strength term."""
    strength = {"PHI": 3, "NYK": -3, "BOS": 1, "MIA": -1}
    n = 0
    for day in range(n_days):
        d = start + timedelta(days=2 * day)
        pairs = [
            (TEAMS[day % 4], TEAMS[(day + 1) % 4]),
            (TEAMS[(day + 2) % 4], TEAMS[(day + 3) % 4]),
        ]
        for home, away in pairs:
            n += 1
            margin = home_edge + strength[home] - strength[away]
            add_game(con, f"{gid_prefix}{season[2:4]}{n:05d}", season, d, home, away,
                     100 + max(margin, 0), 100 + max(-margin, 0), city="Boston")


def build_history(con, games_per_season=40):
    """One warm-up season plus every tuning and test season, each with
    `games_per_season` games (so games-per-season counts in tests are easy
    to predict: round-robin, 2 games/day)."""
    seasons = ms.WARMUP_SEASONS[-1:] + ms.TUNING_SEASONS + ms.TEST_SEASONS
    n_days = games_per_season // 2
    for i, season in enumerate(seasons):
        _season_games(con, season, date(2015 + i, 11, 1), n_days, home_edge=3)
