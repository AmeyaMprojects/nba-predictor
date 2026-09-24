from datetime import UTC, date, datetime, timedelta

import pytest

from predictor import db
from predictor.backtest import replay
from predictor.backtest.baselines import PredictionError, always_home, fixed_probability

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


# --- Finding 1: the leak guard is a real check against the data ---


def test_final_visible_at_cutoff_is_skipped_and_not_shown_to_predictor(con, capsys):
    g = db.POINT_IN_TIME_TABLES["games"]
    # Simulate a mis-derived cutoff by moving the FINAL observation to
    # strictly before it -- the exact defect Finding 1 describes.
    con.execute(
        f"UPDATE {g} SET observed_at = ? WHERE game_id = '0022400561' AND status = 'FINAL'",
        [TIP - timedelta(hours=1)],
    )
    seen: list[str] = []

    def spy(game, view):
        seen.append(game.game_id)
        return 1.0

    preds, stats = replay.replay(con, spy, buffer_minutes=30)

    assert preds == []
    assert stats.skipped_result_visible == 1
    assert stats.predicted == 0
    assert seen == [], "predictor must never be invoked for a leaked game"
    assert "0022400561" in capsys.readouterr().out


def test_final_observed_exactly_at_cutoff_is_treated_as_visible(con):
    """AsOfView's own filter is observed_at <= cutoff -- inclusive -- so an
    exact match at the boundary must be skipped, not predicted."""
    g = db.POINT_IN_TIME_TABLES["games"]
    cutoff = TIP - timedelta(minutes=30)
    con.execute(
        f"UPDATE {g} SET observed_at = ? WHERE game_id = '0022400561' AND status = 'FINAL'",
        [cutoff],
    )
    preds, stats = replay.replay(con, always_home, buffer_minutes=30)
    assert preds == []
    assert stats.skipped_result_visible == 1


# --- Finding 2: a partially-NULL FINAL row must not abort the run ---


def test_final_row_with_one_null_score_does_not_abort_run(con):
    g = db.POINT_IN_TIME_TABLES["games"]
    i = db.POINT_IN_TIME_TABLES["injury_status"]
    # Corrupt the existing game's FINAL row: one score present, one NULL.
    con.execute(
        f"UPDATE {g} SET away_points = NULL "
        "WHERE game_id = '0022400561' AND status = 'FINAL'"
    )
    later_tip = datetime(2025, 1, 20, 0, 0, tzinfo=UTC)
    con.execute(
        f"INSERT INTO {g} (game_id, season, game_date, home_team, away_team,"
        " home_points, away_points, status, observed_at, reconstructed)"
        " VALUES (?,?,?,?,?,?,?,?,?,TRUE)",
        ["0022400999", "2024-25", date(2025, 1, 19), "BOS", "LAL",
         100, 90, "FINAL", later_tip + timedelta(hours=3)],
    )
    con.execute(
        f"INSERT INTO {i} (report_date, game_date, matchup, team, player,"
        " status, reason, observed_at, game_time)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        [date(2025, 1, 19), date(2025, 1, 19), "LAL@BOS", "BOS", "P", "Out", "x",
         later_tip - timedelta(hours=2), "07:00 (ET)"],
    )

    preds, stats = replay.replay(con, always_home)

    assert stats.skipped_no_result == 1
    assert stats.predicted == 1
    assert [p.game_id for p in preds] == ["0022400999"]


# --- Finding 3: `considered` must reconcile even when `limit` triggers a break ---


def test_counters_reconcile_with_limit_set(con):
    g = db.POINT_IN_TIME_TABLES["games"]
    i = db.POINT_IN_TIME_TABLES["injury_status"]
    for n in range(3):
        d = date(2025, 1, 17 + n)
        tip = datetime(2025, 1, 18 + n, 0, 0, tzinfo=UTC)
        gid = f"002240099{n}"
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
            [d, d, "LAL@BOS", "BOS", "P", "Out", "x",
             tip - timedelta(hours=2), "07:00 (ET)"],
        )

    preds, stats = replay.replay(con, always_home, limit=2)

    assert stats.predicted == 2
    assert len(preds) == 2
    assert stats.considered == 2, "a game beyond the limit must not be counted as considered"
    total = (
        stats.predicted
        + stats.skipped_no_tipoff
        + stats.skipped_no_result
        + stats.skipped_result_visible
        + stats.declined
        + stats.failed
    )
    assert total == stats.considered


# --- Finding 5: a negative buffer moves the cutoff past tip-off ---


def test_negative_buffer_minutes_is_rejected(con):
    with pytest.raises(ValueError):
        replay.replay(con, always_home, buffer_minutes=-1)


# --- Finding 6: a deliberate refusal is not the same as a bug ---


def test_predictor_declining_is_counted_separately_from_failed(con):
    def declines(game, view):
        raise PredictionError("not enough pre-game data")

    preds, stats = replay.replay(con, declines)

    assert preds == []
    assert stats.declined == 1
    assert stats.failed == 0
