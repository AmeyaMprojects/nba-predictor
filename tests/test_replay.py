from datetime import UTC, date, datetime, timedelta

import pytest

from predictor import db
from predictor.backtest import replay
from predictor.backtest.baselines import always_home, fixed_probability

TIP = datetime(2025, 1, 16, 0, 0, tzinfo=UTC)  # 7pm ET on 2025-01-15


@pytest.fixture
def con(tmp_path):
    c = db.connect(tmp_path / "t.duckdb")
    db.migrate(c)
    g = db.POINT_IN_TIME_TABLES["games"]
    i = db.POINT_IN_TIME_TABLES["injury_status"]
    # one played game, observed as SCHEDULED then FINAL
    c.execute(
        f"INSERT INTO {g} (game_id, season, game_date, home_team, away_team,"
        " home_points, away_points, status, observed_at, reconstructed)"
        " VALUES (?,?,?,?,?,?,?,?,?,TRUE)",
        ["0022400561", "2024-25", date(2025, 1, 15), "PHI", "NYK",
         None, None, "SCHEDULED", TIP - timedelta(days=7)],
    )
    c.execute(
        f"INSERT INTO {g} (game_id, season, game_date, home_team, away_team,"
        " home_points, away_points, status, observed_at, reconstructed)"
        " VALUES (?,?,?,?,?,?,?,?,?,TRUE)",
        ["0022400561", "2024-25", date(2025, 1, 15), "PHI", "NYK",
         119, 110, "FINAL", TIP + timedelta(hours=3)],
    )
    # an injury row supplying the tip-off time
    c.execute(
        f"INSERT INTO {i} (report_date, game_date, matchup, team, player,"
        " status, reason, observed_at, game_time)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        [date(2025, 1, 15), date(2025, 1, 15), "NYK@PHI", "PHI", "Embiid,Joel",
         "Out", "injury", TIP - timedelta(hours=2), "07:00 (ET)"],
    )
    return c


def test_replays_a_played_game(con):
    preds, stats = replay.replay(con, always_home)
    assert stats.predicted == 1
    assert len(preds) == 1
    p = preds[0]
    assert p.game_id == "0022400561"
    assert p.p_home == 1.0
    assert p.home_won is True


def test_cutoff_is_before_tipoff(con):
    preds, _ = replay.replay(con, always_home, buffer_minutes=30)
    assert preds[0].cutoff == TIP - timedelta(minutes=30)
    assert preds[0].cutoff < preds[0].tipoff


def test_game_without_a_resolvable_tipoff_is_skipped_and_counted(con):
    i = db.POINT_IN_TIME_TABLES["injury_status"]
    con.execute(f"DELETE FROM {i}")
    preds, stats = replay.replay(con, always_home)
    assert preds == []
    assert stats.skipped_no_tipoff == 1
    assert stats.predicted == 0


def test_unplayed_game_is_skipped_not_scored(con):
    g = db.POINT_IN_TIME_TABLES["games"]
    con.execute(f"DELETE FROM {g} WHERE status='FINAL'")
    preds, stats = replay.replay(con, always_home)
    assert preds == []
    assert stats.skipped_no_result == 1


def test_predictor_returning_an_impossible_probability_is_counted_not_silent(con):
    def bad(game, view):
        return 1.7

    preds, stats = replay.replay(con, bad)
    assert preds == []
    assert stats.failed == 1


def test_predictor_raising_does_not_abort_the_run(con):
    def explodes(game, view):
        raise RuntimeError("model blew up")

    preds, stats = replay.replay(con, explodes)
    assert stats.failed == 1
    assert stats.predicted == 0


def test_predictions_are_in_chronological_order(con):
    g = db.POINT_IN_TIME_TABLES["games"]
    i = db.POINT_IN_TIME_TABLES["injury_status"]
    later_tip = datetime(2025, 1, 20, 0, 0, tzinfo=UTC)
    for gid, d, tip in [("0022400999", date(2025, 1, 19), later_tip)]:
        con.execute(
            f"INSERT INTO {g} (game_id, season, game_date, home_team, away_team,"
            " home_points, away_points, status, observed_at, reconstructed)"
            " VALUES (?,?,?,?,?,?,?,?,?,TRUE)",
            [gid, "2024-25", d, "BOS", "LAL", 100, 90, "FINAL", tip + timedelta(hours=3)],
        )
        con.execute(
            f"INSERT INTO {i} (report_date, game_date, matchup, team, player,"
            " status, reason, observed_at, game_time)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            [d, d, "LAL@BOS", "BOS", "P", "Out", "x", tip - timedelta(hours=2), "07:00 (ET)"],
        )
    preds, _ = replay.replay(con, fixed_probability(0.5))
    assert [p.game_id for p in preds] == ["0022400561", "0022400999"]
