import json
from datetime import UTC, date, datetime, timedelta

import pytest

from model_fixtures import add_game, fixture_con
from predictor import db
from predictor.asof import AsOfView
from predictor.backtest.baselines import GameToPredict
from predictor.model.adjustments import Coefficients
from predictor.model.live import (
    START_BUFFER,
    LogError,
    grade,
    grades_path,
    in_season,
    last_capture,
    log_path,
    predict_today,
    read_log,
    results_missing,
    slate_date,
    slate_for,
)
from predictor.model.ratings import RatingParams
from predictor.model.settings import ModelSettings
from predictor.model.stage1 import Stage1Predictor
from schedule_rows import insert_schedule_row

S = ModelSettings(
    ratings=RatingParams(k=0.1, margin_cap=20.0, season_regression=0.5, hca_window=100),
    coefficients=Coefficients(back_to_back=-2.0, third_in_four=-1.0,
                              travel_per_1000km=-0.5, tz_per_hour=-0.25, altitude=1.5),
    sigma=13.0,
    half_life=None,
    tuning_games=0,
)

SEASON = "2026-27"

_SETTINGS_JSON = {"k": 0.1, "margin_cap": 20.0, "season_regression": 0.5,
                  "hca_window": 100, "sigma": 13.0, "half_life": None}


def _insert_final(con, game_id, season, game_date, home, away, home_pts, away_pts,
                   observed_at, reconstructed=False):
    table = db.POINT_IN_TIME_TABLES["games"]
    con.execute(
        f"INSERT INTO {table} (game_id, season, game_date, home_team, away_team,"
        " home_points, away_points, status, reconstructed, observed_at)"
        " VALUES (?,?,?,?,?,?,?,'FINAL',?,?)",
        [game_id, season, game_date, home, away, home_pts, away_pts, reconstructed, observed_at],
    )


def _append_raw_line(path, line):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, sort_keys=True) + "\n")


def _hand_predicted_line(game_id, home, away, tip, p_home, predicted_at, game_date="2026-11-10"):
    return {
        "predicted_at": predicted_at.isoformat(),
        "game_id": game_id,
        "season": SEASON,
        "game_date": game_date,
        "tip_off_utc": tip.isoformat(),
        "home_team": home,
        "away_team": away,
        "status": "predicted",
        "reason": None,
        "spread": 0.0,
        "p_home": p_home,
        "sentence": None,
        "terms": None,
        "settings": _SETTINGS_JSON,
        "stale_results": False,
        "last_result_capture": None,
    }


# --- 1. the slate ------------------------------------------------------

def test_slate_keeps_only_todays_eastern_games(tmp_path):
    con = fixture_con(tmp_path)
    # 2026-10-21 12:30 UTC is 08:30 ET and 18:00 IST on 2026-10-21.
    now = datetime(2026, 10, 21, 12, 30, tzinfo=UTC)

    # Dated today ET, but its tip-off is 01:00 IST on 2026-10-22 (still
    # 2026-10-21 in UTC/ET).
    insert_schedule_row(con, "0022600001", date(2026, 10, 21), "PHI", "NYK",
                         datetime(2026, 10, 21, 19, 30, tzinfo=UTC), season=SEASON)
    # Yesterday's game: excluded.
    insert_schedule_row(con, "0022600002", date(2026, 10, 20), "BOS", "MIA",
                         datetime(2026, 10, 20, 23, 0, tzinfo=UTC), season=SEASON)

    slate = slate_for(con, now)
    assert [g.game_id for g in slate] == ["0022600001"]
    g = slate[0]
    assert g.season == SEASON
    assert g.game_date == date(2026, 10, 21)
    assert g.home_team == "PHI" and g.away_team == "NYK"
    assert g.tip_off_utc == datetime(2026, 10, 21, 19, 30, tzinfo=UTC)


def test_slate_uses_et_date_not_utc_date(tmp_path):
    """2026-10-22 02:00 UTC is 22:00 ET on 2026-10-21 and 07:30 IST on
    2026-10-22 -- a case where the UTC calendar date and the ET calendar
    date disagree (unlike the 12:30 UTC case above, where ET and UTC
    happen to share a date). Using ``now.date()`` (the UTC date) instead of
    ``now.astimezone(EASTERN).date()`` would pick 2026-10-22 and miss the
    actual (ET) slate entirely."""
    con = fixture_con(tmp_path)
    now = datetime(2026, 10, 22, 2, 0, tzinfo=UTC)
    insert_schedule_row(con, "0022600011", date(2026, 10, 21), "PHI", "NYK",
                         datetime(2026, 10, 21, 23, 0, tzinfo=UTC), season=SEASON)
    insert_schedule_row(con, "0022600012", date(2026, 10, 22), "BOS", "MIA",
                         datetime(2026, 10, 22, 23, 0, tzinfo=UTC), season=SEASON)

    slate = slate_for(con, now)
    assert [g.game_id for g in slate] == ["0022600011"]


def test_slate_uses_latest_schedule_vintage_and_orders_by_tipoff_then_id(tmp_path):
    con = fixture_con(tmp_path)
    now = datetime(2026, 10, 21, 12, 30, tzinfo=UTC)
    # An earlier vintage with a TBD (NULL) tip-off, superseded by a real one.
    insert_schedule_row(con, "0022600003", date(2026, 10, 21), "BOS", "MIA", None,
                         observed_at=datetime(2026, 10, 1, tzinfo=UTC), season=SEASON)
    insert_schedule_row(con, "0022600003", date(2026, 10, 21), "BOS", "MIA",
                         datetime(2026, 10, 21, 23, 0, tzinfo=UTC),
                         observed_at=datetime(2026, 10, 20, tzinfo=UTC), season=SEASON)
    insert_schedule_row(con, "0022600001", date(2026, 10, 21), "PHI", "NYK",
                         datetime(2026, 10, 21, 19, 30, tzinfo=UTC), season=SEASON)

    slate = slate_for(con, now)
    assert [g.game_id for g in slate] == ["0022600001", "0022600003"]


def test_non_competitive_prefix_and_tbd_tipoff_are_excluded(tmp_path):
    con = fixture_con(tmp_path)
    now = datetime(2026, 10, 21, 12, 30, tzinfo=UTC)
    insert_schedule_row(con, "0012600001", date(2026, 10, 21), "PHI", "NYK",
                         datetime(2026, 10, 21, 19, 30, tzinfo=UTC), season=SEASON)
    insert_schedule_row(con, "0022600004", date(2026, 10, 21), "BOS", "MIA", None,
                         observed_at=datetime(2026, 10, 1, tzinfo=UTC), season=SEASON)
    assert slate_for(con, now) == []


# --- 1b. the slate date, back-fill of missed days, TBD tip-offs ---------

def test_slate_date_is_the_et_date_a_few_hours_ago():
    # The 18:00 IST run (08:30 EDT) handles today.
    assert slate_date(datetime(2026, 10, 21, 12, 30, tzinfo=UTC)) == date(2026, 10, 21)
    # A catch-up run just after ET midnight (00:30 EDT) still handles the
    # previous ET day.
    assert slate_date(datetime(2026, 10, 22, 4, 30, tzinfo=UTC)) == date(2026, 10, 21)
    assert slate_date(datetime(2026, 10, 22, 13, 0, tzinfo=UTC)) == date(2026, 10, 22)


def test_slate_date_of_the_scheduled_run_is_today_in_winter_too():
    # 18:00 IST is 07:30 EST once the US leaves daylight saving time (IST has
    # none) -- it must STILL handle today, not yesterday, or every game from
    # November to March would go unpredicted.
    assert slate_date(datetime(2026, 11, 5, 12, 30, tzinfo=UTC)) == date(2026, 11, 5)
    assert slate_date(datetime(2027, 1, 15, 12, 30, tzinfo=UTC)) == date(2027, 1, 15)


_MISSED_REASON = (
    "no prediction was made before tip-off (the daily prediction run did not "
    "happen in time)"
)
_TBD_REASON = "tip-off time not announced when the prediction run happened"


def test_missed_days_are_backfilled_as_not_predicted(tmp_path):
    con = fixture_con(tmp_path)
    repo_dir = tmp_path / "repo"
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)  # slate 2026-11-10

    def sched(gid, day):
        insert_schedule_row(con, gid, date(2026, 11, day), "PHI", "NYK",
                             datetime(2026, 11, day, 23, 0, tzinfo=UTC), season=SEASON)

    sched("0022600401", 6)   # 4 days before: outside the window
    sched("0022600402", 7)   # 3 days before: back-filled
    sched("0022600403", 9)   # 1 day before: back-filled
    sched("0022600404", 9)   # 1 day before, but already logged: left alone
    insert_schedule_row(con, "0012600405", date(2026, 11, 9), "BOS", "MIA",
                         datetime(2026, 11, 9, 23, 0, tzinfo=UTC), season=SEASON)  # preseason
    _append_raw_line(
        log_path(repo_dir, SEASON),
        _hand_predicted_line("0022600404", "PHI", "NYK",
                             datetime(2026, 11, 9, 23, 0, tzinfo=UTC), 0.6,
                             datetime(2026, 11, 9, 12, 30, tzinfo=UTC), game_date="2026-11-09"),
    )

    result = predict_today(con, S, repo_dir, now)

    assert result.backfilled == 2
    backfilled = [l for l in result.lines_written if l["reason"] == _MISSED_REASON]
    assert [(l["game_id"], l["game_date"]) for l in backfilled] == [
        ("0022600402", "2026-11-07"),
        ("0022600403", "2026-11-09"),
    ]
    for line in backfilled:
        assert line["status"] == "not_predicted"
        assert line["spread"] is None and line["p_home"] is None
        assert line["terms"] is None and line["sentence"] is None
        assert line["predicted_at"] == now.isoformat()

    # Idempotent: the next run adds no second back-fill line.
    again = predict_today(con, S, repo_dir, now + timedelta(minutes=5))
    assert again.backfilled == 0
    assert again.lines_written == []


def test_backfill_never_claims_a_game_that_has_not_tipped_off(tmp_path):
    # A game still dated yesterday whose (latest) tip-off is in the future
    # has not been missed -- nothing is written for it.
    con = fixture_con(tmp_path)
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    insert_schedule_row(con, "0022600410", date(2026, 11, 9), "PHI", "NYK",
                         now + timedelta(hours=2), season=SEASON)
    result = predict_today(con, S, tmp_path / "repo", now)
    assert result.backfilled == 0
    assert result.lines_written == []


def test_tbd_tipoff_on_the_slate_is_logged_but_does_not_block_a_later_prediction(tmp_path):
    con = fixture_con(tmp_path)
    repo_dir = tmp_path / "repo"
    now = datetime(2026, 11, 10, 13, 0, tzinfo=UTC)
    insert_schedule_row(con, "0022600420", date(2026, 11, 10), "PHI", "NYK", None,
                         observed_at=now - timedelta(days=2), season=SEASON)

    first = predict_today(con, S, repo_dir, now)
    assert first.not_predicted == 1 and first.predicted == 0
    (tbd,) = first.lines_written
    assert tbd["status"] == "not_predicted"
    assert tbd["reason"] == _TBD_REASON
    assert tbd["tip_off_utc"] is None
    assert tbd["spread"] is None and tbd["p_home"] is None

    # Re-running while still TBD writes nothing new.
    rerun = predict_today(con, S, repo_dir, now + timedelta(minutes=10))
    assert rerun.lines_written == []
    assert rerun.skipped_duplicates == 1

    # The tip-off gets announced; the next run predicts the game.
    insert_schedule_row(con, "0022600420", date(2026, 11, 10), "PHI", "NYK",
                         datetime(2026, 11, 11, 0, 0, tzinfo=UTC),
                         observed_at=now + timedelta(hours=1), season=SEASON)
    later = predict_today(con, S, repo_dir, now + timedelta(hours=2))
    assert later.predicted == 1
    assert later.lines_written[0]["status"] == "predicted"

    log = read_log(log_path(repo_dir, SEASON))
    assert [l["status"] for l in log] == ["not_predicted", "predicted"]


# --- 2. a predicted line matches a fresh Stage1Predictor call ----------

def test_predicted_line_matches_a_fresh_predictor_call(tmp_path):
    con = fixture_con(tmp_path)
    add_game(con, "0022600101", SEASON, date(2026, 10, 1), "PHI", "NYK", 110, 100,
             city="Philadelphia")
    add_game(con, "0022600102", SEASON, date(2026, 10, 3), "NYK", "PHI", 100, 104,
             city="New York")

    now = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)  # 07:00 ET on 2026-11-05
    insert_schedule_row(con, "0022600110", date(2026, 11, 5), "PHI", "NYK",
                         now + timedelta(hours=2), season=SEASON)

    repo_dir = tmp_path / "repo"
    result = predict_today(con, S, repo_dir, now)
    assert result.predicted == 1 and result.not_predicted == 0

    fresh = Stage1Predictor(con, S).explain(
        GameToPredict("0022600110", SEASON, date(2026, 11, 5), "PHI", "NYK"),
        AsOfView(con, now),
    )
    line = result.lines_written[0]
    assert line["status"] == "predicted"
    assert line["spread"] == pytest.approx(round(fresh.spread, 6))
    assert line["p_home"] == pytest.approx(round(fresh.p_home, 6))
    assert line["sentence"] == fresh.sentence()
    assert line["terms"] == {name: pytest.approx(round(value, 6)) for name, value in fresh.terms()}

    # The raw file text, not just the parsed dict: keys sorted, and the
    # rounded (not full-precision) spread/p_home actually on disk.
    raw_text = log_path(repo_dir, SEASON).read_text(encoding="utf-8")
    assert raw_text.endswith("\n")
    raw_line = raw_text.splitlines()[-1]
    key_order = json.loads(raw_line, object_pairs_hook=lambda pairs: [k for k, _ in pairs])
    assert key_order == sorted(key_order)
    assert f'"spread": {json.dumps(round(fresh.spread, 6))}' in raw_line
    assert f'"p_home": {json.dumps(round(fresh.p_home, 6))}' in raw_line
    # The raw spread/p_home must actually have more than 6 decimals, or
    # rounding could never be observed on the wire.
    assert round(fresh.spread, 6) != fresh.spread or round(fresh.p_home, 6) != fresh.p_home


# --- 3. the start buffer -------------------------------------------------

def test_start_buffer_boundary(tmp_path):
    con = fixture_con(tmp_path)
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    assert START_BUFFER == timedelta(minutes=30)
    insert_schedule_row(con, "0022600201", date(2026, 11, 10), "PHI", "NYK",
                         now + timedelta(minutes=20), season=SEASON)
    insert_schedule_row(con, "0022600202", date(2026, 11, 10), "BOS", "MIA",
                         now + timedelta(minutes=31), season=SEASON)
    # Exactly on the buffer boundary: `tip - START_BUFFER == now`, which is
    # `<= now` -- still "already started (or within 30 minutes)".
    insert_schedule_row(con, "0022600203", date(2026, 11, 10), "DEN", "LAL",
                         now + START_BUFFER, season=SEASON)

    result = predict_today(con, S, tmp_path / "repo", now)
    assert result.predicted == 1
    assert result.not_predicted == 2

    by_id = {line["game_id"]: line for line in result.lines_written}
    early = by_id["0022600201"]
    assert early["status"] == "not_predicted"
    assert early["reason"] == (
        "game had already started (or was within 30 minutes of tip-off) when "
        "the prediction run happened"
    )
    assert early["spread"] is None and early["p_home"] is None and early["sentence"] is None
    assert early["terms"] is None

    exactly_on_buffer = by_id["0022600203"]
    assert exactly_on_buffer["status"] == "not_predicted"

    late = by_id["0022600202"]
    assert late["status"] == "predicted"
    assert late["reason"] is None
    assert late["spread"] is not None and late["p_home"] is not None


# --- 4. duplicates and rescheduling --------------------------------------

def test_same_day_rerun_is_a_duplicate_but_reschedule_predicts_again(tmp_path):
    con = fixture_con(tmp_path)
    repo_dir = tmp_path / "repo"
    now1 = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    insert_schedule_row(con, "0022600301", date(2026, 11, 10), "PHI", "NYK",
                         now1 + timedelta(hours=2), season=SEASON)

    r1 = predict_today(con, S, repo_dir, now1)
    assert r1.predicted == 1 and r1.skipped_duplicates == 0

    r2 = predict_today(con, S, repo_dir, now1)
    assert r2.predicted == 0 and r2.not_predicted == 0 and r2.skipped_duplicates == 1
    assert r2.lines_written == []

    # Reschedule: a new, later schedule vintage moves the game two days out.
    insert_schedule_row(con, "0022600301", date(2026, 11, 12), "PHI", "NYK",
                         datetime(2026, 11, 12, 22, 0, tzinfo=UTC),
                         observed_at=now1 + timedelta(hours=1), season=SEASON)

    now2 = datetime(2026, 11, 12, 20, 0, tzinfo=UTC)
    r3 = predict_today(con, S, repo_dir, now2)
    assert r3.predicted == 1 and r3.skipped_duplicates == 0

    log = read_log(log_path(repo_dir, SEASON))
    assert [(line["game_id"], line["game_date"]) for line in log] == [
        ("0022600301", "2026-11-10"),
        ("0022600301", "2026-11-12"),
    ]


# --- 5. staleness: results_missing ----------------------------------------

def test_in_season_uses_latest_schedule_vintage(tmp_path):
    con = fixture_con(tmp_path)
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    gid = "0022600801"
    # An old vintage puts the tip-off within the +/-3 day window...
    insert_schedule_row(con, gid, date(2026, 11, 10), "PHI", "NYK",
                         now + timedelta(days=1),
                         observed_at=datetime(2026, 10, 1, tzinfo=UTC), season=SEASON)
    # ...but the LATEST vintage reschedules it far outside that window.
    insert_schedule_row(con, gid, date(2026, 12, 10), "PHI", "NYK",
                         now + timedelta(days=30),
                         observed_at=datetime(2026, 11, 1, tzinfo=UTC), season=SEASON)
    assert in_season(con, now) is False


def test_not_stale_on_opening_night_with_no_past_games(tmp_path):
    con = fixture_con(tmp_path)
    now = datetime(2026, 10, 21, 20, 0, tzinfo=UTC)
    assert results_missing(con, now) == []
    result = predict_today(con, S, tmp_path / "repo", now)
    assert result.stale is False


def test_stale_when_a_recent_game_has_no_final_result(tmp_path):
    con = fixture_con(tmp_path)
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    # Tipped 20 hours ago -- within [now-3d, now-12h] -- with no FINAL row.
    insert_schedule_row(con, "0022600811", date(2026, 11, 9), "PHI", "NYK",
                         now - timedelta(hours=20), season=SEASON)
    insert_schedule_row(con, "0022600812", date(2026, 11, 10), "BOS", "MIA",
                         now + timedelta(hours=2), season=SEASON)
    assert results_missing(con, now) == ["0022600811"]

    result = predict_today(con, S, tmp_path / "repo", now)
    assert result.stale is True
    # Yesterday's game is back-filled as not predicted; today's is predicted.
    assert result.backfilled == 1
    (line,) = [l for l in result.lines_written if l["game_id"] == "0022600812"]
    assert line["stale_results"] is True


def test_not_stale_once_the_missing_games_final_is_captured(tmp_path):
    con = fixture_con(tmp_path)
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    insert_schedule_row(con, "0022600813", date(2026, 11, 9), "PHI", "NYK",
                         now - timedelta(hours=20), season=SEASON)
    insert_schedule_row(con, "0022600814", date(2026, 11, 10), "BOS", "MIA",
                         now + timedelta(hours=2), season=SEASON)
    # A backfilled (not live-captured) FINAL row is enough to clear
    # `results_missing` -- ANY FINAL row counts -- even though it does NOT
    # count as a `last_capture()` (reconstructed = TRUE).
    _insert_final(con, "0022600813", SEASON, date(2026, 11, 9), "PHI", "NYK",
                  110, 100, observed_at=now - timedelta(hours=19), reconstructed=True)
    assert results_missing(con, now) == []
    assert last_capture(con) is None

    result = predict_today(con, S, tmp_path / "repo", now)
    assert result.stale is False
    (line,) = [l for l in result.lines_written if l["game_id"] == "0022600814"]
    assert line["stale_results"] is False
    assert line["last_result_capture"] is None


def test_old_missing_result_does_not_count_as_stale(tmp_path):
    con = fixture_con(tmp_path)
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    # Tipped 4 days ago -- older than the 3-day window.
    insert_schedule_row(con, "0022600815", date(2026, 11, 6), "PHI", "NYK",
                         now - timedelta(days=4), season=SEASON)
    assert results_missing(con, now) == []


def test_very_recent_missing_result_does_not_count_as_stale(tmp_path):
    con = fixture_con(tmp_path)
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    # Tipped only 6 hours ago -- inside the 12-hour grace period.
    insert_schedule_row(con, "0022600816", date(2026, 11, 10), "PHI", "NYK",
                         now - timedelta(hours=6), season=SEASON)
    assert results_missing(con, now) == []


# --- 6. append-only ----------------------------------------------------

def test_log_is_append_only(tmp_path):
    con = fixture_con(tmp_path)
    repo_dir = tmp_path / "repo"
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    insert_schedule_row(con, "0022600501", date(2026, 11, 10), "PHI", "NYK",
                         now + timedelta(hours=2), season=SEASON)

    predict_today(con, S, repo_dir, now)
    path = log_path(repo_dir, SEASON)
    first_run_bytes = path.read_bytes()
    assert first_run_bytes  # something was written

    # A new game appears on the same slate before the second poll.
    insert_schedule_row(con, "0022600502", date(2026, 11, 10), "BOS", "MIA",
                         now + timedelta(hours=3), season=SEASON)
    result2 = predict_today(con, S, repo_dir, now)
    assert result2.predicted == 1 and result2.skipped_duplicates == 1

    full_bytes = path.read_bytes()
    assert full_bytes.startswith(first_run_bytes)
    assert len(full_bytes) > len(first_run_bytes)


def test_read_log_rejects_a_truncated_file_and_never_appends(tmp_path):
    con = fixture_con(tmp_path)
    repo_dir = tmp_path / "repo"
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    path = log_path(repo_dir, SEASON)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"game_id": "x"}\n{"game_id": "truncated"', encoding="utf-8")
    before = path.read_bytes()

    with pytest.raises(LogError):
        read_log(path)

    insert_schedule_row(con, "0022600950", date(2026, 11, 10), "PHI", "NYK",
                         now + timedelta(hours=2), season=SEASON)
    with pytest.raises(LogError):
        predict_today(con, S, repo_dir, now)

    assert path.read_bytes() == before  # never touched, let alone appended onto


# --- 7. leak safety ------------------------------------------------------

def test_a_future_captured_result_does_not_change_the_prediction(tmp_path):
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)

    def _build(db_dir, with_future_leak):
        con = fixture_con(db_dir)
        add_game(con, "0022600601", SEASON, date(2026, 11, 1), "PHI", "NYK", 110, 100,
                 city="Philadelphia")
        insert_schedule_row(con, "0022600610", date(2026, 11, 10), "PHI", "NYK",
                             now + timedelta(hours=2), season=SEASON)
        if with_future_leak:
            # A result that only becomes visible AFTER `now` -- must not leak in.
            _insert_final(con, "0022600699", SEASON, date(2026, 11, 9), "BOS", "MIA",
                          150, 10, observed_at=now + timedelta(hours=1), reconstructed=False)
            insert_schedule_row(con, "0022600699", date(2026, 11, 9), "BOS", "MIA",
                                 datetime(2026, 11, 9, 23, 0, tzinfo=UTC), season=SEASON)
        return con

    con_without = _build(tmp_path / "without", with_future_leak=False)
    result_without = predict_today(con_without, S, tmp_path / "repo_without", now)

    con_with = _build(tmp_path / "with", with_future_leak=True)
    result_with = predict_today(con_with, S, tmp_path / "repo_with", now)

    (line_without,) = [l for l in result_without.lines_written if l["game_id"] == "0022600610"]
    (line_with,) = [l for l in result_with.lines_written if l["game_id"] == "0022600610"]
    assert line_without["spread"] == line_with["spread"]
    assert line_without["p_home"] == line_with["p_home"]


# --- 8. grading ----------------------------------------------------------

def test_grading(tmp_path):
    con = fixture_con(tmp_path)
    repo_dir = tmp_path / "repo"
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)

    # A normal game: predicted, then its result is captured after tip-off.
    insert_schedule_row(con, "0022600701", date(2026, 11, 10), "PHI", "NYK",
                         now + timedelta(hours=2), season=SEASON)
    # A game that had already started when the run happened: never predicted.
    insert_schedule_row(con, "0022600702", date(2026, 11, 10), "BOS", "MIA",
                         now - timedelta(minutes=5), season=SEASON)

    result = predict_today(con, S, repo_dir, now)
    assert result.predicted == 1 and result.not_predicted == 1
    predicted_line = next(l for l in result.lines_written if l["game_id"] == "0022600701")
    # Zero history on both teams: spread is exactly 0, so p_home is exactly
    # 0.5 (erf(0) == 0) -- a deterministic, non-flaky "favourite" baseline.
    assert predicted_line["p_home"] == 0.5

    later = now + timedelta(hours=3)
    _insert_final(con, "0022600701", SEASON, date(2026, 11, 10), "PHI", "NYK",
                  110, 100, observed_at=later, reconstructed=False)
    _insert_final(con, "0022600702", SEASON, date(2026, 11, 10), "BOS", "MIA",
                  90, 95, observed_at=later, reconstructed=False)

    # A hand-crafted line: "predicted" but logged at/after its own tip-off --
    # must never be graded even though a FINAL result exists for it.
    insert_schedule_row(con, "0022600703", date(2026, 11, 10), "DEN", "LAL",
                         now + timedelta(hours=1), season=SEASON)
    late_line = _hand_predicted_line("0022600703", "DEN", "LAL",
                                      now + timedelta(hours=1), 0.6, now + timedelta(hours=1))
    _append_raw_line(log_path(repo_dir, SEASON), late_line)
    _insert_final(con, "0022600703", SEASON, date(2026, 11, 10), "DEN", "LAL",
                  120, 100, observed_at=later, reconstructed=False)

    appended = grade(con, repo_dir, SEASON, later)
    assert appended == 1

    grades = read_log(grades_path(repo_dir, SEASON))
    assert len(grades) == 1
    g = grades[0]
    assert g["game_id"] == "0022600701"
    assert g["predicted_at"] == predicted_line["predicted_at"]
    assert g["p_home"] == predicted_line["p_home"]
    assert g["home_won"] is True
    assert g["correct"] is True  # p_home 0.5 >= 0.5 and home (PHI) actually won

    # Re-grading adds nothing.
    assert grade(con, repo_dir, SEASON, later + timedelta(hours=1)) == 0
    assert len(read_log(grades_path(repo_dir, SEASON))) == 1


def test_grade_correct_is_literal_true_or_false(tmp_path):
    con = fixture_con(tmp_path)
    repo_dir = tmp_path / "repo"
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    tip = now + timedelta(hours=2)

    insert_schedule_row(con, "0022600910", date(2026, 11, 10), "PHI", "NYK", tip, season=SEASON)
    insert_schedule_row(con, "0022600911", date(2026, 11, 10), "BOS", "MIA", tip, season=SEASON)
    _append_raw_line(log_path(repo_dir, SEASON),
                      _hand_predicted_line("0022600910", "PHI", "NYK", tip, 0.9, now))
    _append_raw_line(log_path(repo_dir, SEASON),
                      _hand_predicted_line("0022600911", "BOS", "MIA", tip, 0.2, now))

    later = tip + timedelta(hours=1)
    # Favourite (p_home 0.9) wins at home -> correct.
    _insert_final(con, "0022600910", SEASON, date(2026, 11, 10), "PHI", "NYK",
                  110, 100, observed_at=later, reconstructed=False)
    # Home underdog (p_home 0.2) wins anyway -> incorrect.
    _insert_final(con, "0022600911", SEASON, date(2026, 11, 10), "BOS", "MIA",
                  105, 100, observed_at=later, reconstructed=False)

    assert grade(con, repo_dir, SEASON, later) == 2
    grades = {g["game_id"]: g for g in read_log(grades_path(repo_dir, SEASON))}
    assert grades["0022600910"]["correct"] is True
    assert grades["0022600911"]["correct"] is False


def test_grade_uses_the_latest_predicted_line_before_tip(tmp_path):
    con = fixture_con(tmp_path)
    repo_dir = tmp_path / "repo"
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    tip = now + timedelta(hours=2)
    insert_schedule_row(con, "0022600920", date(2026, 11, 10), "PHI", "NYK", tip, season=SEASON)

    # Two predicted lines for the same game (e.g. an early run, then a
    # re-poll later), both logged before tip-off. The LATER one must win.
    _append_raw_line(log_path(repo_dir, SEASON),
                      _hand_predicted_line("0022600920", "PHI", "NYK", tip, 0.2, now))
    _append_raw_line(log_path(repo_dir, SEASON),
                      _hand_predicted_line("0022600920", "PHI", "NYK", tip, 0.8,
                                           now + timedelta(minutes=30)))

    later = tip + timedelta(hours=1)
    _insert_final(con, "0022600920", SEASON, date(2026, 11, 10), "PHI", "NYK",
                  110, 100, observed_at=later, reconstructed=False)

    assert grade(con, repo_dir, SEASON, later) == 1
    grades = read_log(grades_path(repo_dir, SEASON))
    assert len(grades) == 1
    assert grades[0]["predicted_at"] == (now + timedelta(minutes=30)).isoformat()
    assert grades[0]["p_home"] == 0.8
    assert grades[0]["correct"] is True


def test_grade_requires_the_final_rows_game_date_to_match_the_line(tmp_path):
    con = fixture_con(tmp_path)
    repo_dir = tmp_path / "repo"
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    tip = now + timedelta(hours=2)
    insert_schedule_row(con, "0022600930", date(2026, 11, 10), "PHI", "NYK", tip, season=SEASON)
    _append_raw_line(log_path(repo_dir, SEASON),
                      _hand_predicted_line("0022600930", "PHI", "NYK", tip, 0.6, now))

    later = tip + timedelta(hours=1)
    # A FINAL row for the same game_id but a DIFFERENT game_date (the game
    # itself was rescheduled, not just corrected) must not be used to grade
    # this line.
    _insert_final(con, "0022600930", SEASON, date(2026, 11, 11), "PHI", "NYK",
                  110, 100, observed_at=later, reconstructed=False)

    assert grade(con, repo_dir, SEASON, later) == 0
    assert read_log(grades_path(repo_dir, SEASON)) == []


def test_grade_cutoff_uses_the_earlier_of_logged_and_current_tip(tmp_path):
    con = fixture_con(tmp_path)
    repo_dir = tmp_path / "repo"
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    original_tip = now + timedelta(hours=2)
    insert_schedule_row(con, "0022600940", date(2026, 11, 10), "PHI", "NYK",
                         original_tip, season=SEASON)

    # Logged before the ORIGINAL tip-off, but after what the schedule will
    # later say was the real (corrected) tip-off.
    predicted_at = now + timedelta(hours=1)
    _append_raw_line(
        log_path(repo_dir, SEASON),
        _hand_predicted_line("0022600940", "PHI", "NYK", original_tip, 0.6, predicted_at),
    )

    # The schedule is corrected to an EARLIER tip-off, before `predicted_at`.
    corrected_tip = now + timedelta(minutes=30)
    insert_schedule_row(con, "0022600940", date(2026, 11, 10), "PHI", "NYK",
                         corrected_tip, observed_at=now + timedelta(minutes=45), season=SEASON)

    later = original_tip + timedelta(hours=1)
    _insert_final(con, "0022600940", SEASON, date(2026, 11, 10), "PHI", "NYK",
                  110, 100, observed_at=later, reconstructed=False)

    # The effective cutoff is min(original_tip, corrected_tip) == corrected_tip,
    # and predicted_at is AFTER that -- never graded.
    assert grade(con, repo_dir, SEASON, later) == 0
    assert read_log(grades_path(repo_dir, SEASON)) == []


# --- code version on every line (final fix wave) ---------------------------

import subprocess  # noqa: E402


def _git_repo(path, monkeypatch):
    for var, value in (("GIT_AUTHOR_NAME", "Bot"), ("GIT_AUTHOR_EMAIL", "bot@example.invalid"),
                       ("GIT_COMMITTER_NAME", "Bot"), ("GIT_COMMITTER_EMAIL", "bot@example.invalid")):
        monkeypatch.setenv(var, value)
    path.mkdir(parents=True)

    def git(*args):
        return subprocess.run(["git", *args], cwd=path, capture_output=True, text=True, timeout=30)

    assert git("init", "-b", "main").returncode == 0
    (path / "README.md").write_text("seed\n", encoding="utf-8")
    assert git("add", "README.md").returncode == 0
    assert git("commit", "-m", "seed").returncode == 0
    return git("rev-parse", "--short", "HEAD").stdout.strip()


def test_every_line_records_the_code_version(tmp_path, monkeypatch):
    con = fixture_con(tmp_path)
    repo_dir = tmp_path / "repo"
    sha = _git_repo(repo_dir, monkeypatch)
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    insert_schedule_row(con, "0022600960", date(2026, 11, 10), "PHI", "NYK",
                         now + timedelta(hours=2), season=SEASON)      # predicted
    insert_schedule_row(con, "0022600961", date(2026, 11, 10), "BOS", "MIA",
                         now - timedelta(minutes=5), season=SEASON)    # too late
    insert_schedule_row(con, "0022600962", date(2026, 11, 9), "DEN", "LAL",
                         now - timedelta(hours=20), season=SEASON)     # back-filled

    result = predict_today(con, S, repo_dir, now)

    assert len(result.lines_written) == 3
    for line in result.lines_written:
        assert line["code_version"] == sha
        assert line["code_dirty"] is False

    # The code is edited (uncommitted) before the next run.
    (repo_dir / "README.md").write_text("edited\n", encoding="utf-8")
    insert_schedule_row(con, "0022600963", date(2026, 11, 10), "UTA", "OKC",
                         now + timedelta(hours=3), season=SEASON)
    second = predict_today(con, S, repo_dir, now)
    (line,) = second.lines_written
    assert line["code_version"] == sha
    assert line["code_dirty"] is True


def test_grade_lines_record_the_code_version(tmp_path, monkeypatch):
    con = fixture_con(tmp_path)
    repo_dir = tmp_path / "repo"
    sha = _git_repo(repo_dir, monkeypatch)
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    tip = now + timedelta(hours=2)
    insert_schedule_row(con, "0022600970", date(2026, 11, 10), "PHI", "NYK", tip, season=SEASON)
    _append_raw_line(log_path(repo_dir, SEASON),
                      _hand_predicted_line("0022600970", "PHI", "NYK", tip, 0.6, now))
    later = tip + timedelta(hours=4)
    _insert_final(con, "0022600970", SEASON, date(2026, 11, 10), "PHI", "NYK",
                  110, 100, observed_at=later, reconstructed=False)

    assert grade(con, repo_dir, SEASON, later) == 1
    (g,) = read_log(grades_path(repo_dir, SEASON))
    assert g["code_version"] == sha
    assert g["code_dirty"] is False


def test_code_version_is_null_outside_a_git_checkout(tmp_path):
    con = fixture_con(tmp_path)
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    insert_schedule_row(con, "0022600980", date(2026, 11, 10), "PHI", "NYK",
                         now + timedelta(hours=2), season=SEASON)
    (line,) = predict_today(con, S, tmp_path / "repo", now).lines_written
    assert line["code_version"] is None
    assert line["code_dirty"] is False


# --- minors (final fix wave) -------------------------------------------------


def test_grade_lines_carry_the_score_and_teams(tmp_path):
    con = fixture_con(tmp_path)
    repo_dir = tmp_path / "repo"
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    tip = now + timedelta(hours=2)
    insert_schedule_row(con, "0022600990", date(2026, 11, 10), "PHI", "NYK", tip, season=SEASON)
    _append_raw_line(log_path(repo_dir, SEASON),
                      _hand_predicted_line("0022600990", "PHI", "NYK", tip, 0.4, now))
    later = tip + timedelta(hours=4)
    _insert_final(con, "0022600990", SEASON, date(2026, 11, 10), "PHI", "NYK",
                  98, 104, observed_at=later, reconstructed=False)

    assert grade(con, repo_dir, SEASON, later) == 1
    (g,) = read_log(grades_path(repo_dir, SEASON))
    assert (g["home_team"], g["away_team"]) == ("PHI", "NYK")
    assert (g["home_points"], g["away_points"]) == (98, 104)
    assert g["home_won"] is False
    assert g["correct"] is True  # p_home 0.4 picked the away team, who won


def test_negative_zero_never_reaches_the_log(tmp_path):
    # Zero history: the rest/travel terms come out as -0.0 (a negative
    # coefficient times zero). The public log must say 0.0, not -0.0.
    con = fixture_con(tmp_path)
    repo_dir = tmp_path / "repo"
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    insert_schedule_row(con, "0022600991", date(2026, 11, 10), "PHI", "NYK",
                         now + timedelta(hours=2), season=SEASON)

    predict_today(con, S, repo_dir, now)

    raw = log_path(repo_dir, SEASON).read_text(encoding="utf-8")
    assert "-0.0" not in raw
    (line,) = read_log(log_path(repo_dir, SEASON))
    for value in [line["spread"], line["p_home"], *line["terms"].values()]:
        assert not (value == 0 and str(value).startswith("-"))


# --- market line beside each prediction (market-odds task 4) ---------------

_MARKET_FIELDS = ("market_p_home", "market_spread", "market_books", "market_observed_at",
                  "market_label")


def _insert_odds(con, game_id, book, observed_at, *, home=-200, away=170, spread=-5.5,
                 source="theoddsapi"):
    table = db.POINT_IN_TIME_TABLES["odds_snapshots"]
    con.execute(
        f"INSERT INTO {table} (game_key, book, home_team, away_team, home_price,"
        " away_price, spread, total, observed_at, game_id, source, reconstructed)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [f"ev-{game_id}", book, "PHI", "NYK", home, away, spread, 220.0, observed_at,
         game_id, source, source == "kaggle_sbr"],
    )


def _market_game(db_dir, now, *, with_odds):
    con = fixture_con(db_dir)
    add_game(con, "0022601001", SEASON, date(2026, 11, 1), "PHI", "NYK", 110, 100,
             city="Philadelphia")
    add_game(con, "0022601002", SEASON, date(2026, 11, 3), "NYK", "PHI", 99, 104,
             city="New York")
    insert_schedule_row(con, "0022601010", date(2026, 11, 10), "PHI", "NYK",
                         now + timedelta(hours=2), season=SEASON)
    if with_odds:
        _insert_odds(con, "0022601010", "fanduel", now - timedelta(hours=1),
                     home=-200, away=170, spread=-5.5)
        _insert_odds(con, "0022601010", "draftkings", now - timedelta(minutes=30),
                     home=-180, away=150, spread=-4.5)
        # Observed after `now`: must not be used.
        _insert_odds(con, "0022601010", "betmgm", now + timedelta(minutes=5),
                     home=-900, away=600, spread=-15.0)
        # A historical (Kaggle) row for the same game: never a live line.
        _insert_odds(con, "0022601010", "consensus", now - timedelta(hours=2),
                     home=+300, away=-400, spread=8.0, source="kaggle_sbr")
    return con


def test_predicted_line_records_the_market_line_visible_at_now(tmp_path):
    from predictor.model import market

    now = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
    con = _market_game(tmp_path, now, with_odds=True)
    (line,) = predict_today(con, S, tmp_path / "repo", now).lines_written

    view = market.market_p_home(market.live_lines(con, "0022601010", now), S.sigma)
    assert view is not None and view.books == 2
    assert line["market_p_home"] == round(view.p_home, 6)
    assert line["market_spread"] == -5.0
    assert line["market_books"] == 2
    assert line["market_observed_at"] == (now - timedelta(minutes=30)).isoformat()
    assert line["market_label"] == "market line at 17:30 IST"


def test_predicted_line_without_odds_has_null_market_fields(tmp_path):
    now = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
    con = _market_game(tmp_path, now, with_odds=False)
    (line,) = predict_today(con, S, tmp_path / "repo", now).lines_written
    assert line["status"] == "predicted"
    for field in _MARKET_FIELDS:
        assert field in line and line[field] is None, field


def test_market_from_spread_only_uses_the_live_sigma(tmp_path):
    from predictor.model.ratings import win_probability

    now = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
    con = _market_game(tmp_path, now, with_odds=False)
    _insert_odds(con, "0022601010", "fanduel", now - timedelta(hours=1),
                 home=None, away=None, spread=-6.0)
    (line,) = predict_today(con, S, tmp_path / "repo", now).lines_written
    assert line["market_p_home"] == round(win_probability(6.0, S.sigma), 6)
    assert line["market_spread"] == -6.0
    assert line["market_books"] == 1


def test_not_predicted_lines_carry_null_market_fields(tmp_path):
    now = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
    con = fixture_con(tmp_path)
    insert_schedule_row(con, "0022601020", date(2026, 11, 10), "PHI", "NYK",
                         now + timedelta(minutes=10), season=SEASON)
    _insert_odds(con, "0022601020", "fanduel", now - timedelta(hours=1))
    (line,) = predict_today(con, S, tmp_path / "repo", now).lines_written
    assert line["status"] == "not_predicted"
    for field in _MARKET_FIELDS:
        assert line[field] is None, field


def test_model_numbers_are_byte_identical_with_and_without_odds(tmp_path):
    now = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
    with_odds = _market_game(tmp_path / "with", now, with_odds=True)
    without = _market_game(tmp_path / "without", now, with_odds=False)
    (a,) = predict_today(with_odds, S, tmp_path / "repo_with", now).lines_written
    (b,) = predict_today(without, S, tmp_path / "repo_without", now).lines_written
    assert a["market_p_home"] is not None and b["market_p_home"] is None
    model_keys = [k for k in a if not k.startswith("market_")]
    assert model_keys == [k for k in b if not k.startswith("market_")]
    dump = lambda line: json.dumps({k: line[k] for k in model_keys}, sort_keys=True)  # noqa: E731
    assert dump(a) == dump(b)


def test_explain_never_reads_the_odds_table(tmp_path):
    """Leak test: a recording AsOfView proves the model only ever reads
    tables other than odds_snapshots, even with odds rows present."""
    now = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
    con = _market_game(tmp_path, now, with_odds=True)
    seen: list[str] = []

    class RecordingView(AsOfView):
        def table(self, name):
            seen.append(name)
            return super().table(name)

        def latest(self, name, key=None):
            seen.append(name)
            return super().latest(name, key)

    Stage1Predictor(con, S).explain(
        GameToPredict("0022601010", SEASON, date(2026, 11, 10), "PHI", "NYK"),
        RecordingView(con, now),
    )
    assert seen, "the recording view saw no reads at all"
    assert "odds_snapshots" not in seen


def test_grade_records_whether_the_market_was_right(tmp_path):
    con = fixture_con(tmp_path)
    repo_dir = tmp_path / "repo"
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    tip = now + timedelta(hours=2)
    games = {"0022601030": 0.7, "0022601031": 0.3, "0022601032": None}
    for gid, market_p in games.items():
        insert_schedule_row(con, gid, date(2026, 11, 10), "PHI", "NYK", tip, season=SEASON)
        line = _hand_predicted_line(gid, "PHI", "NYK", tip, 0.6, now)
        if market_p is not None:
            line["market_p_home"] = market_p
        _append_raw_line(log_path(repo_dir, SEASON), line)
        _insert_final(con, gid, SEASON, date(2026, 11, 10), "PHI", "NYK", 110, 100,
                      observed_at=tip + timedelta(hours=4))

    assert grade(con, repo_dir, SEASON, tip + timedelta(hours=4)) == 3
    got = {g["game_id"]: g["market_correct"] for g in read_log(grades_path(repo_dir, SEASON))}
    # 0022601032's line predates market fields entirely: null, not a crash.
    assert got == {"0022601030": True, "0022601031": False, "0022601032": None}
