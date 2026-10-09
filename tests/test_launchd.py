import plistlib
import subprocess

from predictor.config import PROJECT_ROOT

SCRIPTS = PROJECT_ROOT / "scripts"


def _load(name):
    return plistlib.loads((SCRIPTS / name).read_bytes())


def _slots(plist):
    return {(d["Hour"], d.get("Minute", 0)) for d in plist["StartCalendarInterval"]}


def test_schedule_job_runs_ingest_schedule_once_a_day():
    p = _load("com.predictor.schedule.plist")
    assert p["Label"] == "com.predictor.schedule"
    assert p["ProgramArguments"] == ["PROJECT_DIR/.venv/bin/predictor", "ingest-schedule"]
    assert _slots(p) == {(10, 30)}
    assert p["StandardOutPath"] == "PROJECT_DIR/data/logs/schedule.out.log"
    assert p["StandardErrorPath"] == "PROJECT_DIR/data/logs/schedule.err.log"


def test_schedule_job_does_not_run_at_load():
    # Installing reloads both jobs at once; firing both immediately would
    # race for DuckDB's write lock on the very first run.
    assert _load("com.predictor.schedule.plist")["RunAtLoad"] is False


def test_schedule_and_news_jobs_never_share_a_start_time():
    assert _slots(_load("com.predictor.schedule.plist")).isdisjoint(
        _slots(_load("com.predictor.daily.plist"))
    )


def test_install_script_installs_both_jobs_and_is_valid_bash():
    script = SCRIPTS / "install_schedule.sh"
    text = script.read_text()
    assert "install_job com.predictor.daily" in text
    assert "install_job com.predictor.schedule" in text
    subprocess.run(["bash", "-n", str(script)], check=True)


# --- Task 4: capture-results and predict-today jobs -----------------------

def test_results_job_runs_capture_results_three_times_a_day():
    p = _load("com.predictor.results.plist")
    assert p["Label"] == "com.predictor.results"
    assert p["ProgramArguments"] == ["PROJECT_DIR/.venv/bin/predictor", "capture-results"]
    # Three idempotent runs, so one run missed to a sleeping Mac (no
    # network in a dark wake) is not a lost day. 12:00 IST = 02:30 EDT:
    # most of the previous ET day has finished; 16:00 IST (06:30 EDT)
    # catches the late tips before predict-today (18:00); 22:00 IST is a
    # last same-day retry.
    assert _slots(p) == {(12, 0), (16, 0), (22, 0)}
    assert p["StandardOutPath"] == "PROJECT_DIR/data/logs/results.out.log"
    assert p["StandardErrorPath"] == "PROJECT_DIR/data/logs/results.err.log"
    assert p["RunAtLoad"] is False


def test_predict_job_runs_predict_today_once_a_day():
    p = _load("com.predictor.predict.plist")
    assert p["Label"] == "com.predictor.predict"
    assert p["ProgramArguments"] == ["PROJECT_DIR/.venv/bin/predictor", "predict-today"]
    assert _slots(p) == {(18, 0)}
    assert p["StandardOutPath"] == "PROJECT_DIR/data/logs/predict.out.log"
    assert p["StandardErrorPath"] == "PROJECT_DIR/data/logs/predict.err.log"
    assert p["RunAtLoad"] is False


ALL_JOBS = (
    "com.predictor.daily.plist",
    "com.predictor.schedule.plist",
    "com.predictor.results.plist",
    "com.predictor.predict.plist",
    "com.predictor.odds.plist",
)


def test_no_two_jobs_share_a_start_minute():
    all_slots = []
    for name in ALL_JOBS:
        all_slots.extend(_slots(_load(name)))
    assert len(all_slots) == len(set(all_slots))


def test_results_job_shares_no_start_minute_with_any_other_job():
    results_slots = _slots(_load("com.predictor.results.plist"))
    others = (
        _slots(_load("com.predictor.daily.plist"))
        | _slots(_load("com.predictor.schedule.plist"))
        | _slots(_load("com.predictor.predict.plist"))
        | _slots(_load("com.predictor.odds.plist"))
    )
    assert others == {(9, 0), (14, 0), (19, 0), (10, 30), (18, 0), (17, 30)}
    assert results_slots.isdisjoint(others)


def test_install_script_announces_the_results_time():
    text = (SCRIPTS / "install_schedule.sh").read_text()
    assert "Live results will be captured at 12:00, 16:00 and 22:00 daily." in text
    assert "11:00" not in text


def test_install_script_installs_all_four_jobs():
    script = SCRIPTS / "install_schedule.sh"
    text = script.read_text()
    assert "install_job com.predictor.results" in text
    assert "install_job com.predictor.predict" in text
    subprocess.run(["bash", "-n", str(script)], check=True)


# --- market odds: the daily ingest-odds job --------------------------------

def test_odds_job_runs_ingest_odds_once_a_day_at_1730():
    p = _load("com.predictor.odds.plist")
    assert p["Label"] == "com.predictor.odds"
    assert p["ProgramArguments"] == ["PROJECT_DIR/.venv/bin/predictor", "ingest-odds"]
    # 17:30 IST: after capture-results (16:00), before predict-today (18:00),
    # so the 18:00 log can carry the market line. One call/day stays far
    # inside the free 500-requests/month quota.
    assert _slots(p) == {(17, 30)}
    assert p["WorkingDirectory"] == "PROJECT_DIR"
    assert p["StandardOutPath"] == "PROJECT_DIR/data/logs/odds.out.log"
    assert p["StandardErrorPath"] == "PROJECT_DIR/data/logs/odds.err.log"
    assert p["RunAtLoad"] is False


def test_install_script_installs_all_five_jobs():
    script = SCRIPTS / "install_schedule.sh"
    text = script.read_text()
    for name in ALL_JOBS:
        assert f"install_job {name.removesuffix('.plist')}" in text
    assert text.count("\ninstall_job ") == 5
    assert "Market odds will be fetched at 17:30 daily." in text
    subprocess.run(["bash", "-n", str(script)], check=True)
