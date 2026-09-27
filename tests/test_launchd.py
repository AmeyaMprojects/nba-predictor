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
