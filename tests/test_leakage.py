"""Adversarial tests: deliberately attempt to leak future data.

If any test here fails, the backtest's results are meaningless. Treat a
failure as a critical defect, never as a test to relax.
"""

from datetime import UTC, datetime, timedelta

import pytest

from predictor import db
from predictor.asof import AsOfError, AsOfView

TIP_OFF = datetime(2025, 1, 15, 0, 0, tzinfo=UTC)


@pytest.fixture
def con(tmp_path):
    c = db.connect(tmp_path / "t.duckdb")
    db.migrate(c)
    return c


def _insert_game(con, game_id, observed_at, status, home_points, away_points):
    con.execute(
        "INSERT INTO games (game_id, season, game_date, home_team, away_team,"
        " home_points, away_points, status, observed_at) VALUES (?,?,?,?,?,?,?,?,?)",
        [
            game_id,
            "2024-25",
            TIP_OFF.date(),
            "PHI",
            "NYK",
            home_points,
            away_points,
            status,
            observed_at,
        ],
    )


def test_final_score_is_invisible_before_the_game_finishes(con):
    _insert_game(con, "001", TIP_OFF - timedelta(days=1), "SCHEDULED", None, None)
    _insert_game(con, "001", TIP_OFF + timedelta(hours=3), "FINAL", 110, 104)

    view = AsOfView(con, TIP_OFF - timedelta(minutes=30))
    rows = view.table("games").project("status, home_points").fetchall()

    assert rows == [("SCHEDULED", None)]
    assert all(r[1] is None for r in rows), "final score leaked into pre-game view"


def test_injury_report_published_after_cutoff_is_invisible(con):
    for offset, player in [(-1, "early"), (1, "late")]:
        con.execute(
            "INSERT INTO injury_status (report_date, game_date, team, player, status, observed_at)"
            " VALUES (?,?,?,?,?,?)",
            [
                TIP_OFF.date(),
                TIP_OFF.date(),
                "PHI",
                player,
                "Out",
                TIP_OFF + timedelta(hours=offset),
            ],
        )
    view = AsOfView(con, TIP_OFF)
    players = {r[0] for r in view.table("injury_status").project("player").fetchall()}
    assert players == {"early"}


def test_odds_moved_after_cutoff_are_invisible(con):
    for offset, spread in [(-2, -3.5), (2, -7.5)]:
        con.execute(
            "INSERT INTO odds_snapshots (game_key, book, home_team, away_team,"
            " spread, observed_at) VALUES (?,?,?,?,?,?)",
            ["g1", "bookA", "PHI", "NYK", spread, TIP_OFF + timedelta(hours=offset)],
        )
    view = AsOfView(con, TIP_OFF)
    spreads = [r[0] for r in view.table("odds_snapshots").project("spread").fetchall()]
    assert spreads == [-3.5]


def test_guard_cannot_be_bypassed_with_a_crafted_table_name(con):
    view = AsOfView(con, TIP_OFF)
    for hostile in [
        "games WHERE 1=1 OR observed_at > now()",
        "(SELECT * FROM games)",
        "games--",
        "GAMES",
    ]:
        with pytest.raises(AsOfError):
            view.table(hostile)


def test_every_point_in_time_table_is_reachable_through_the_view(con):
    """A new table added to the schema must not silently bypass the guard."""
    view = AsOfView(con, TIP_OFF)
    for table in db.POINT_IN_TIME_TABLES:
        view.table(table).fetchall()
