from datetime import UTC, date, datetime, timedelta

import pytest

from predictor import db
from predictor.backtest import replay
from predictor.backtest.baselines import PredictionError, always_home, fixed_probability
from schedule_rows import insert_schedule_row

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
    # the tip-off source (sub-project 2.5)
    insert_schedule_row(c, "0022400561", date(2025, 1, 15), "PHI", "NYK", TIP)
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
    s = db.POINT_IN_TIME_TABLES["schedule"]
    con.execute(f"DELETE FROM {s}")
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
    assert stats.skipped_score_missing == 0


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
        insert_schedule_row(con, gid, d, "BOS", "LAL", tip)
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
    insert_schedule_row(con, "0022400999", date(2025, 1, 19), "BOS", "LAL", later_tip)

    preds, stats = replay.replay(con, always_home)

    # FIX 9 (final review, part 2): this game WAS played -- it has a FINAL
    # row -- but the archive failed to record its score. That is a
    # different, more actionable situation than "not yet played" and must
    # be pinned under its own counter, not the same one.
    assert stats.skipped_score_missing == 1
    assert stats.skipped_no_result == 0
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
        insert_schedule_row(con, gid, d, "BOS", "LAL", tip)

    preds, stats = replay.replay(con, always_home, limit=2)

    assert stats.predicted == 2
    assert len(preds) == 2
    assert stats.considered == 2, "a game beyond the limit must not be counted as considered"
    total = (
        stats.predicted
        + stats.skipped_conflicting_metadata
        + stats.skipped_buffer_too_early
        + stats.skipped_no_tipoff
        + stats.skipped_no_result
        + stats.skipped_score_missing
        + stats.skipped_result_visible
        + stats.declined
        + stats.failed
    )
    assert total == stats.considered


def test_counters_reconcile_with_no_limit_set(con):
    # FIX 13(e): only the `limit` case was covered -- the far more common
    # path (a full, unbounded replay) had no reconciliation test at all.
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
        insert_schedule_row(con, gid, d, "BOS", "LAL", tip)
    # Also add one game with no resolvable tip-off, so more than one
    # counter is nonzero -- a reconciliation bug that only shows up when a
    # skip path fires would otherwise slip past a test with zero skips.
    con.execute(
        f"INSERT INTO {g} (game_id, season, game_date, home_team, away_team,"
        " home_points, away_points, status, observed_at, reconstructed)"
        " VALUES (?,?,?,?,?,?,?,?,?,TRUE)",
        ["0022400888", "2024-25", date(2025, 1, 25), "MIA", "ORL",
         100, 90, "FINAL", datetime(2025, 1, 26, 12, 0, tzinfo=UTC) + timedelta(hours=3)],
    )

    preds, stats = replay.replay(con, always_home)

    assert stats.considered == 5  # fixture's 1 + the 3 added + the no-tip-off one
    assert stats.skipped_no_tipoff == 1
    assert stats.predicted == 4
    total = (
        stats.predicted
        + stats.skipped_conflicting_metadata
        + stats.skipped_buffer_too_early
        + stats.skipped_no_tipoff
        + stats.skipped_no_result
        + stats.skipped_score_missing
        + stats.skipped_result_visible
        + stats.declined
        + stats.failed
    )
    assert total == stats.considered


# --- FIX 5 (final review, part 1): conflicting metadata across rows -----


def test_conflicting_metadata_across_rows_is_skipped_not_double_predicted(con):
    """A re-ingest that corrects a game's home/away or game_date leaves the
    OLD row sitting alongside the NEW one, because the primary key is
    (game_id, observed_at), not game_id alone. Build exactly that: an extra
    row for the SAME game_id as the fixture's played game, but with a
    different home_team. The old bug would predict (and score) this game
    TWICE, once with the wrong home team. It must instead be routed to
    `skipped_conflicting_metadata` and not predicted at all.
    """
    g = db.POINT_IN_TIME_TABLES["games"]
    con.execute(
        f"INSERT INTO {g} (game_id, season, game_date, home_team, away_team,"
        " home_points, away_points, status, observed_at, reconstructed)"
        " VALUES (?,?,?,?,?,?,?,?,?,TRUE)",
        ["0022400561", "2024-25", date(2025, 1, 15), "LAL", "NYK",
         None, None, "SCHEDULED", TIP - timedelta(days=30)],
    )

    preds, stats = replay.replay(con, always_home)

    assert stats.skipped_conflicting_metadata == 1
    assert preds == [], "the conflicting game must not be predicted at all"
    assert stats.predicted == 0
    assert stats.considered == 1


def test_conflicting_season_is_skipped_identically_with_or_without_season_filter(con):
    """FIX 19 (final review, part 3): the season predicate used to apply
    BEFORE the uniqueness check, so `--season` matching just ONE of a
    conflicting game's two season labels hid the OTHER row from the
    aggregate entirely, letting the game pass the check it should have
    failed and get predicted (and scored) -- the exact double-scoring the
    guard exists to prevent. Must be skipped identically with no
    --season, with the fixture's real season, and with the other
    (bogus, conflicting) season label.
    """
    g = db.POINT_IN_TIME_TABLES["games"]
    con.execute(
        f"INSERT INTO {g} (game_id, season, game_date, home_team, away_team,"
        " home_points, away_points, status, observed_at, reconstructed)"
        " VALUES (?,?,?,?,?,?,?,?,?,TRUE)",
        ["0022400561", "2023-24", date(2025, 1, 15), "PHI", "NYK",
         None, None, "SCHEDULED", TIP - timedelta(days=30)],
    )

    for season in (None, "2024-25", "2023-24"):
        preds, stats = replay.replay(con, always_home, season=season)
        assert preds == [], f"season={season!r} predicted a conflicting game"
        assert stats.skipped_conflicting_metadata == 1, f"season={season!r}"
        assert stats.predicted == 0, f"season={season!r}"


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


# --- FIX 13(e): the season filter is currently untested --------------------


def test_season_filter_restricts_the_scored_set(con):
    g = db.POINT_IN_TIME_TABLES["games"]
    i = db.POINT_IN_TIME_TABLES["injury_status"]
    other_tip = datetime(2023, 12, 20, 0, 0, tzinfo=UTC)
    con.execute(
        f"INSERT INTO {g} (game_id, season, game_date, home_team, away_team,"
        " home_points, away_points, status, observed_at, reconstructed)"
        " VALUES (?,?,?,?,?,?,?,?,?,TRUE)",
        ["0022300777", "2023-24", date(2023, 12, 19), "BOS", "LAL",
         100, 90, "FINAL", other_tip + timedelta(hours=3)],
    )
    con.execute(
        f"INSERT INTO {i} (report_date, game_date, matchup, team, player,"
        " status, reason, observed_at, game_time)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        [date(2023, 12, 19), date(2023, 12, 19), "LAL@BOS", "BOS", "P", "Out", "x",
         other_tip - timedelta(hours=2), "07:00 (ET)"],
    )
    insert_schedule_row(
        con, "0022300777", date(2023, 12, 19), "BOS", "LAL", other_tip, season="2023-24"
    )

    all_preds, all_stats = replay.replay(con, always_home)
    assert all_stats.considered == 2
    assert {p.season for p in all_preds} == {"2024-25", "2023-24"}

    filtered_preds, filtered_stats = replay.replay(con, always_home, season="2024-25")
    assert filtered_stats.considered == 1
    assert [p.game_id for p in filtered_preds] == ["0022400561"]
    assert {p.season for p in filtered_preds} == {"2024-25"}


# --- FIX 7: an unbounded buffer must not reach before the schedule existed -


def test_buffer_reaching_before_the_schedule_existed_is_skipped_and_counted(con):
    """The fixture's SCHEDULED row is stamped 7 days before tip-off. A
    buffer larger than that pushes the cutoff before the game was ever on
    the schedule -- the harness must not hand the predictor a game it had
    no way of knowing existed yet."""
    seen: list[str] = []

    def spy(game, view):
        seen.append(game.game_id)
        return 1.0

    preds, stats = replay.replay(con, spy, buffer_minutes=60 * 24 * 30)  # 30 days

    assert preds == []
    assert stats.skipped_buffer_too_early == 1
    assert stats.predicted == 0
    assert seen == [], "predictor must never be invoked for a too-early buffer"


def test_a_normal_buffer_does_not_trip_the_too_early_check(con):
    preds, stats = replay.replay(con, always_home, buffer_minutes=30)
    assert stats.skipped_buffer_too_early == 0
    assert stats.predicted == 1


# --- FIX 11: the no-tip-off skip carries its season -------------------------


def test_no_tipoff_skip_is_tracked_by_season(con):
    s = db.POINT_IN_TIME_TABLES["schedule"]
    con.execute(f"DELETE FROM {s}")
    _, stats = replay.replay(con, always_home)
    assert stats.skipped_no_tipoff_by_season == {"2024-25": 1}
    assert stats.considered_by_season == {"2024-25": 1}
