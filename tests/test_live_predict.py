import json
from datetime import UTC, date, datetime, timedelta

import pytest

from model_fixtures import add_game, fixture_con
from predictor import db
from predictor.asof import AsOfView
from predictor.backtest.baselines import GameToPredict
from predictor.model.adjustments import Coefficients
from predictor.model.live import (
    EASTERN,
    STALE_AFTER,
    START_BUFFER,
    grade,
    grades_path,
    in_season,
    last_capture,
    log_path,
    predict_today,
    read_log,
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

    result = predict_today(con, S, tmp_path / "repo", now)
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


# --- 3. the start buffer -------------------------------------------------

def test_start_buffer_boundary(tmp_path):
    con = fixture_con(tmp_path)
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    assert START_BUFFER == timedelta(minutes=30)
    insert_schedule_row(con, "0022600201", date(2026, 11, 10), "PHI", "NYK",
                         now + timedelta(minutes=20), season=SEASON)
    insert_schedule_row(con, "0022600202", date(2026, 11, 10), "BOS", "MIA",
                         now + timedelta(minutes=31), season=SEASON)

    result = predict_today(con, S, tmp_path / "repo", now)
    assert result.predicted == 1
    assert result.not_predicted == 1

    by_id = {line["game_id"]: line for line in result.lines_written}
    early = by_id["0022600201"]
    assert early["status"] == "not_predicted"
    assert early["reason"] == (
        "game had already started (or was within 30 minutes of tip-off) when "
        "the prediction run happened"
    )
    assert early["spread"] is None and early["p_home"] is None and early["sentence"] is None
    assert early["terms"] is None

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


# --- 5. staleness ----------------------------------------------------------

def test_stale_when_in_season_and_capture_is_old(tmp_path):
    con = fixture_con(tmp_path)
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    old_capture = now - STALE_AFTER - timedelta(hours=4)
    add_game(con, "0022600401", SEASON, date(2026, 11, 1), "BOS", "MIA", 100, 90,
             city="Boston", final_observed_at=old_capture, reconstructed=False)
    insert_schedule_row(con, "0022600402", date(2026, 11, 10), "PHI", "NYK",
                         now + timedelta(hours=2), season=SEASON)

    assert in_season(con, now) is True
    assert last_capture(con) == old_capture

    result = predict_today(con, S, tmp_path / "repo", now)
    assert result.stale is True
    assert len(result.lines_written) == 1
    line = result.lines_written[0]
    assert line["stale_results"] is True
    assert line["last_result_capture"] == old_capture.isoformat()


def test_not_stale_off_season_with_no_capture(tmp_path):
    con = fixture_con(tmp_path)
    now = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)  # off-season, nothing scheduled

    assert in_season(con, now) is False
    assert last_capture(con) is None

    result = predict_today(con, S, tmp_path / "repo", now)
    assert result.stale is False
    assert result.lines_written == []


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

    line_without = result_without.lines_written[0]
    line_with = result_with.lines_written[0]
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

    later = now + timedelta(hours=3)
    _insert_final(con, "0022600701", SEASON, date(2026, 11, 10), "PHI", "NYK",
                  110, 100, observed_at=later, reconstructed=False)
    _insert_final(con, "0022600702", SEASON, date(2026, 11, 10), "BOS", "MIA",
                  90, 95, observed_at=later, reconstructed=False)

    # A hand-crafted line: "predicted" but logged at/after its own tip-off --
    # must never be graded even though a FINAL result exists for it.
    insert_schedule_row(con, "0022600703", date(2026, 11, 10), "DEN", "LAL",
                         now + timedelta(hours=1), season=SEASON)
    late_line = {
        "predicted_at": (now + timedelta(hours=1)).isoformat(),
        "game_id": "0022600703",
        "season": SEASON,
        "game_date": "2026-11-10",
        "tip_off_utc": (now + timedelta(hours=1)).isoformat(),
        "home_team": "DEN",
        "away_team": "LAL",
        "status": "predicted",
        "reason": None,
        "spread": 1.0,
        "p_home": 0.6,
        "sentence": None,
        "terms": None,
        "settings": {"k": 0.1, "margin_cap": 20.0, "season_regression": 0.5,
                     "hca_window": 100, "sigma": 13.0, "half_life": None},
        "stale_results": False,
        "last_result_capture": None,
    }
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
    assert g["correct"] == ((predicted_line["p_home"] >= 0.5) == True)

    # Re-grading adds nothing.
    assert grade(con, repo_dir, SEASON, later + timedelta(hours=1)) == 0
    assert len(read_log(grades_path(repo_dir, SEASON))) == 1
