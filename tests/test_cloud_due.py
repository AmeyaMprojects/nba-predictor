"""Task 7: the early call and the two odds windows of the 15-minute runs."""

from datetime import UTC, date, datetime, timedelta

from model_fixtures import fixture_con
from predictor import db
from predictor.model.live import (
    EARLY_CALL_IST,
    ODDS_EARLY_WINDOW,
    ODDS_MAX_ATTEMPTS,
    ODDS_MIN_REMAINING,
    ODDS_PRE_TIP_LEAD,
    OddsWindow,
    early_call_at,
    early_call_due,
    odds_window_due,
    slate_date,
)
from schedule_rows import insert_schedule_row

SEASON = "2026-27"
DAY = date(2026, 11, 10)
TIP = datetime(2026, 11, 10, 23, 0, tzinfo=UTC)


def dt(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _slate(tmp_path, tips=(TIP,)):
    con = fixture_con(tmp_path)
    for n, tip in enumerate(tips):
        insert_schedule_row(con, f"00226009{n:02d}", DAY, "PHI", "NYK", tip, season=SEASON)
    return con


def _odds_row(con, observed_at, source="theoddsapi"):
    table = db.POINT_IN_TIME_TABLES["odds_snapshots"]
    con.execute(
        f"INSERT INTO {table} (game_key, book, home_team, away_team, home_price,"
        " away_price, spread, total, observed_at, game_id, source, reconstructed)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        ["ev-1", "fanduel", "PHI", "NYK", -200, 170, -5.5, 220.0, observed_at,
         "0022600900", source, source == "kaggle_sbr"],
    )


def test_constants():
    assert EARLY_CALL_IST.hour == 18 and EARLY_CALL_IST.minute == 0
    assert ODDS_EARLY_WINDOW == timedelta(hours=2)
    assert ODDS_PRE_TIP_LEAD == timedelta(minutes=75)
    assert ODDS_MAX_ATTEMPTS == 2 and ODDS_MIN_REMAINING == 30


def test_early_call_time_is_1230_utc():
    assert early_call_at(date(2026, 10, 22)) == datetime(2026, 10, 22, 12, 30, tzinfo=UTC)
    assert early_call_at(date(2026, 11, 3)) == datetime(2026, 11, 3, 12, 30, tzinfo=UTC)


def test_early_call_not_due_before_1800_ist_even_when_slate_rolled():
    assert slate_date(dt("2026-10-23T11:15Z")) == date(2026, 10, 23)
    assert not early_call_due(dt("2026-10-23T11:15Z"))
    assert not early_call_due(dt("2026-10-23T12:29:59Z"))
    assert early_call_due(dt("2026-10-23T12:30Z"))
    assert early_call_due(dt("2026-10-23T03:00Z"))  # slate 2026-10-22


def test_early_call_on_dst_change_day():
    assert slate_date(dt("2026-11-02T11:15Z")) == date(2026, 11, 1)
    assert early_call_due(dt("2026-11-02T11:15Z"))
    assert early_call_at(date(2026, 11, 1)) == datetime(2026, 11, 1, 12, 30, tzinfo=UTC)


def test_odds_early_window_due_once(tmp_path):
    con = _slate(tmp_path)
    w = odds_window_due(con, dt("2026-11-10T12:37Z"), [], None)
    assert w == OddsWindow("early", dt("2026-11-10T12:30Z"), dt("2026-11-10T14:30Z"))
    _odds_row(con, dt("2026-11-10T12:38Z"))
    assert odds_window_due(con, dt("2026-11-10T12:52Z"), [], None) is None


def test_odds_early_window_ignores_historical_rows(tmp_path):
    con = _slate(tmp_path)
    _odds_row(con, dt("2026-11-10T12:38Z"), source="kaggle_sbr")
    assert odds_window_due(con, dt("2026-11-10T12:52Z"), [], None).name == "early"


def test_odds_not_due_before_the_early_call(tmp_path):
    con = _slate(tmp_path)
    assert odds_window_due(con, dt("2026-11-10T12:29Z"), [], None) is None


def test_odds_pre_tip_window(tmp_path):
    con = _slate(tmp_path)
    # an early-window line does not satisfy the pre-tip window
    _odds_row(con, dt("2026-11-10T12:38Z"))
    w = odds_window_due(con, dt("2026-11-10T21:50Z"), [], None)
    assert w == OddsWindow("pre_tip", dt("2026-11-10T21:45Z"), TIP)
    assert odds_window_due(con, dt("2026-11-10T21:40Z"), [], None) is None
    assert odds_window_due(con, dt("2026-11-10T23:00Z"), [], None) is None


def test_odds_pre_tip_window_needs_a_game_still_to_predict(tmp_path):
    con = _slate(tmp_path)
    # 22:31 is within START_BUFFER of the only tip
    assert odds_window_due(con, dt("2026-11-10T22:31Z"), [], None) is None
    assert odds_window_due(con, dt("2026-11-10T22:29Z"), [], None).name == "pre_tip"


def test_early_window_closes_at_the_pre_tip_lead_for_an_early_first_tip(tmp_path):
    con = _slate(tmp_path, tips=(dt("2026-11-10T14:00Z"), TIP))
    w = odds_window_due(con, dt("2026-11-10T12:40Z"), [], None)
    assert w == OddsWindow("early", dt("2026-11-10T12:30Z"), dt("2026-11-10T12:45Z"))
    assert odds_window_due(con, dt("2026-11-10T12:50Z"), [], None).name == "pre_tip"


def test_odds_attempts_cap(tmp_path):
    con = _slate(tmp_path)
    now = dt("2026-11-10T13:10Z")
    old = {"attempted_at": "2026-11-09T21:50:00+00:00", "window": "pre_tip", "ok": False,
           "remaining": 400}
    one = {"attempted_at": "2026-11-10T12:37:00+00:00", "window": "early", "ok": False,
           "remaining": 400}
    two = {"attempted_at": dt("2026-11-10T12:52Z"), "window": "early", "ok": False,
           "remaining": 400}
    assert odds_window_due(con, now, [old, one], 400).name == "early"
    assert odds_window_due(con, now, [old, one, two], 400) is None


def test_odds_skipped_when_quota_low(tmp_path):
    con = _slate(tmp_path)
    now = dt("2026-11-10T12:37Z")
    assert odds_window_due(con, now, [], 29) is None
    assert odds_window_due(con, now, [], 30).name == "early"


def test_odds_not_due_on_empty_slate(tmp_path):
    con = fixture_con(tmp_path)
    assert odds_window_due(con, dt("2026-11-10T12:37Z"), [], None) is None
    assert odds_window_due(con, dt("2026-11-10T21:50Z"), [], None) is None


def test_late_first_run_still_makes_the_early_call_and_odds(tmp_path):
    # GitHub cron skipped every slot from 12:30 to 14:20: the first run
    # still makes the early call and fetches the early odds.
    con = _slate(tmp_path)
    now = dt("2026-11-10T14:20Z")
    assert early_call_due(now) is True
    assert odds_window_due(con, now, [], None).name == "early"
