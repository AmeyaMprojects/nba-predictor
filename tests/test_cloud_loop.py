"""cloud-loop: run cloud-run in-process on the :07/:22/:37/:52 marks until
the job's time budget is spent. A fake clock and sleep drive the schedule;
a fake run_once stands in for the run itself (cloud-run's own tests cover it)."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from predictor import cli, config, db, raw_store
from predictor.cloud import loop
from predictor.cloud.run import STEP_NAMES, Fetchers, RunReport, StepOutcome

START = datetime(2026, 10, 26, 7, 7, 30, tzinfo=UTC)


class FakeTime:
    """clock() and sleep() over one shared fake time; each run_once call
    advances it by ``run_minutes``."""

    def __init__(self, start: datetime = START):
        self.now = start
        self.sleeps: list[float] = []

    def clock(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        assert seconds > 0
        self.sleeps.append(seconds)
        self.now += timedelta(seconds=seconds)


def _report(started, *, raw_failed=False, secret_leak=False, unexpected=False,
            site_changed=False, publish_error=False, branch="cloud-shadow") -> RunReport:
    steps = []
    for name in STEP_NAMES:
        if name == "raw_archive":
            steps.append(StepOutcome(name, True, not raw_failed and not secret_leak,
                                     "upload failed" if raw_failed else "ok"))
        else:
            steps.append(StepOutcome(name, True, True, "ok"))
    return RunReport(
        started_at=started, finished_at=started, branch=branch, steps=steps,
        state_commit=None, predictions_commit=None, pushed=site_changed,
        site_changed=site_changed, raw_archived=not raw_failed, unexpected=unexpected,
        publish_error=publish_error, secret_leak=secret_leak,
    )


class FakeRun:
    """A run_once stand-in: records start times and the data dir in force,
    writes a raw file there, and returns (or raises) what ``plan`` says for
    each call in turn (default: a clean report)."""

    def __init__(self, t: FakeTime, plan=None, run_minutes: float = 3):
        self.t = t
        self.plan = list(plan or [])
        self.run_minutes = run_minutes
        self.starts: list[datetime] = []
        self.data_dirs: list[Path] = []

    def __call__(self, state_repo, code_dir, *, branch, fetchers, run_attempt):
        started = self.t.now
        self.starts.append(started)
        d = config.settings.data_dir
        assert db.settings.data_dir == d and raw_store.settings.data_dir == d
        self.data_dirs.append(d)
        for sub in ("espn", "odds", "odds_history"):
            (config.settings.raw_dir / sub).mkdir(parents=True, exist_ok=True)
            (config.settings.raw_dir / sub / "blob.json").write_text("{}", encoding="utf-8")
        (d / "predictor.duckdb").write_text("db", encoding="utf-8")
        self.t.now += timedelta(minutes=self.run_minutes)
        what = self.plan.pop(0) if self.plan else {}
        if isinstance(what, BaseException):
            raise what
        return _report(started, branch=branch, **what)


class FakeGh:
    def __init__(self, returncode=0):
        self.calls = []
        self.returncode = returncode

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, self.returncode, "", "boom")


def _loop(tmp_path, t, fake, *, branch="cloud-shadow", gh=None, max_minutes=60, **kw):
    texts: list[str] = []
    outputs: dict[str, str] = {}
    result = loop.run_loop(
        tmp_path / "state", tmp_path / "code", branch=branch, base_dir=tmp_path / "data",
        fetchers=Fetchers(gh_runner=gh or FakeGh()), max_minutes=max_minutes,
        clock=t.clock, sleep=t.sleep, run_once=fake, emit=texts.append,
        set_output=outputs.__setitem__, **kw,
    )
    return result, texts, outputs


# --- scheduling ---------------------------------------------------------------


def test_marks_every_15_minutes_from_minute_7():
    at = datetime(2026, 10, 26, 7, 7, tzinfo=UTC)
    assert loop.next_mark(at, 15) == datetime(2026, 10, 26, 7, 22, tzinfo=UTC)
    assert loop.next_mark(at - timedelta(seconds=1), 15) == at
    assert loop.next_mark(datetime(2026, 10, 26, 7, 53, tzinfo=UTC), 15) == datetime(
        2026, 10, 26, 8, 7, tzinfo=UTC)
    assert loop.next_mark(datetime(2026, 10, 26, 23, 59, tzinfo=UTC), 15) == datetime(
        2026, 10, 27, 0, 7, tzinfo=UTC)


def test_first_iteration_runs_at_once_then_on_the_marks(tmp_path):
    t = FakeTime()
    fake = FakeRun(t)
    result, _, _ = _loop(tmp_path, t, fake, max_minutes=60)
    minutes = [s.strftime("%H:%M:%S") for s in fake.starts]
    # Job start 07:07:30 + 60 min = 08:07:30; a start at 07:52 could end
    # 08:02 (10 min budget): allowed; 08:07 could end 08:17: not started.
    assert minutes == ["07:07:30", "07:22:00", "07:37:00", "07:52:00"]
    assert result.iterations == 4
    assert result.ok


def test_stops_when_the_next_start_would_end_past_the_budget(tmp_path):
    t = FakeTime(datetime(2026, 10, 26, 7, 0, tzinfo=UTC))
    fake = FakeRun(t)
    # Budget ends 07:31: 07:22 + 10 min = 07:32 is past it.
    _loop(tmp_path, t, fake, max_minutes=31)
    assert [s.strftime("%H:%M") for s in fake.starts] == ["07:00", "07:07"]


def test_budget_boundary_is_inclusive(tmp_path):
    t = FakeTime(datetime(2026, 10, 26, 7, 0, tzinfo=UTC))
    fake = FakeRun(t)
    _loop(tmp_path, t, fake, max_minutes=32)  # 07:22 + 10 = 07:32 == the end
    assert [s.strftime("%H:%M") for s in fake.starts] == ["07:00", "07:07", "07:22"]


def test_a_missed_mark_starts_at_once(tmp_path):
    t = FakeTime(datetime(2026, 10, 26, 7, 7, tzinfo=UTC))
    fake = FakeRun(t, run_minutes=20)  # ends 07:27, past the 07:22 mark
    _loop(tmp_path, t, fake, max_minutes=45)
    # 07:07 ends 07:27 (07:22 missed: start at once); 07:27 ends 07:47; the
    # next start (07:47) could end 07:57, past the 07:52 budget.
    assert [s.strftime("%H:%M") for s in fake.starts] == ["07:07", "07:27"]


def test_an_early_wake_never_repeats_a_mark(tmp_path):
    t = FakeTime(datetime(2026, 10, 26, 7, 7, tzinfo=UTC))
    fake = FakeRun(t)

    def early_sleep(seconds):
        t.sleeps.append(seconds)
        t.now += timedelta(seconds=seconds - 0.5)

    loop.run_loop(tmp_path / "s", tmp_path / "c", branch="cloud-shadow",
                  base_dir=tmp_path / "data", fetchers=Fetchers(gh_runner=FakeGh()),
                  max_minutes=40, clock=t.clock, sleep=early_sleep, run_once=fake,
                  emit=lambda text: None, set_output=lambda k, v: None)
    assert len(fake.starts) == 3  # 07:07, ~07:22, ~07:37 -- never 07:22 twice


def test_interval_must_divide_the_hour():
    with pytest.raises(ValueError):
        loop.next_mark(START, 25)


# --- failures -----------------------------------------------------------------


def test_an_iteration_that_raises_never_stops_the_loop(tmp_path):
    t = FakeTime()
    fake = FakeRun(t, plan=[RuntimeError("kaboom"), {"unexpected": True}, {}, {}])
    result, texts, _ = _loop(tmp_path, t, fake)
    assert len(fake.starts) == 4
    assert result.ok is False
    assert any("RuntimeError" in text and "kaboom" in text for text in texts)


def test_unexpected_error_fails_the_loop_at_the_end_only(tmp_path):
    t = FakeTime()
    fake = FakeRun(t, plan=[{"unexpected": True}])
    result, _, _ = _loop(tmp_path, t, fake)
    assert result.iterations == 4
    assert result.ok is False


def test_a_secret_leak_stops_the_loop_at_once(tmp_path):
    t = FakeTime()
    fake = FakeRun(t, plan=[{}, {"secret_leak": True, "unexpected": True}, {}])
    result, _, outputs = _loop(tmp_path, t, fake)
    assert len(fake.starts) == 2
    assert result.secret_leak is True
    assert result.ok is False
    assert outputs["secret_leak"] == "true"
    # The leaking iteration's dir is never kept.
    assert not fake.data_dirs[1].exists()
    assert not (tmp_path / "data" / "kept").exists()


def test_the_secret_is_redacted_from_an_iteration_error(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))  # never the real key file
    monkeypatch.setenv("ODDS_API_KEY", "0123456789abcdef0123456789abcdef")
    t = FakeTime()
    fake = FakeRun(t, plan=[RuntimeError("key 0123456789abcdef0123456789abcdef")])
    _, texts, _ = _loop(tmp_path, t, fake, max_minutes=5)
    joined = "".join(texts)
    assert "0123456789abcdef0123456789abcdef" not in joined
    assert "RuntimeError" in joined


# --- data dirs ----------------------------------------------------------------


def test_each_iteration_has_its_own_data_dir_removed_after_success(tmp_path):
    t = FakeTime()
    fake = FakeRun(t)
    before = config.settings
    result, _, outputs = _loop(tmp_path, t, fake)
    assert len(set(fake.data_dirs)) == 4
    for d in fake.data_dirs:
        assert d.parent == tmp_path / "data"
        assert not d.exists()
    assert config.settings is before  # restored
    assert db.settings is before and raw_store.settings is before
    assert result.raw_kept is False
    assert outputs == {"raw_kept": "false", "secret_leak": "false"}


def test_raw_kept_only_when_the_upload_failed(tmp_path):
    t = FakeTime()
    fake = FakeRun(t, plan=[{}, {"raw_failed": True}, RuntimeError("x"), {}])
    result, _, outputs = _loop(tmp_path, t, fake)
    assert result.raw_kept is True
    assert outputs["raw_kept"] == "true"
    assert outputs["secret_leak"] == "false"
    kept = tmp_path / "data" / "kept"
    (only,) = list(kept.iterdir())
    assert only.name == fake.data_dirs[1].name
    # Only the raw responses are kept; the database is not.
    assert (only / "raw" / "espn" / "blob.json").exists()
    assert not (only / "predictor.duckdb").exists()
    for d in fake.data_dirs:
        assert not d.exists()


# --- site dispatch ------------------------------------------------------------


def test_site_dispatched_after_an_iteration_that_changed_it_on_main(tmp_path):
    t = FakeTime()
    gh = FakeGh()
    fake = FakeRun(t, plan=[{"site_changed": True}, {}, {"site_changed": True}])
    result, _, _ = _loop(tmp_path, t, fake, branch="main", gh=gh)
    assert [argv for argv, _ in gh.calls] == [
        ["gh", "workflow", "run", "site.yml", "--ref", "main"]] * 2
    _, kwargs = gh.calls[0]
    assert kwargs["cwd"] == tmp_path / "state"
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert kwargs["timeout"] > 0
    assert result.ok


def test_site_never_dispatched_in_shadow_mode(tmp_path):
    t = FakeTime()
    gh = FakeGh()
    fake = FakeRun(t, plan=[{"site_changed": True}])
    _loop(tmp_path, t, fake, branch="cloud-shadow", gh=gh)
    assert gh.calls == []


def test_a_failed_dispatch_fails_the_loop_but_continues(tmp_path):
    t = FakeTime()
    gh = FakeGh(returncode=1)
    fake = FakeRun(t, plan=[{"site_changed": True}])
    result, texts, _ = _loop(tmp_path, t, fake, branch="main", gh=gh)
    assert result.iterations == 4
    assert result.ok is False
    assert any("site.yml" in text for text in texts)


# --- summaries and the CLI ----------------------------------------------------


def test_each_iteration_summary_is_headed_by_its_time(tmp_path):
    t = FakeTime()
    fake = FakeRun(t)
    _, texts, _ = _loop(tmp_path, t, fake)
    assert len(texts) == 4
    assert texts[1].startswith("# Loop iteration 2 — 2026-10-26 07:22 UTC")
    assert "## Cloud run 2026-10-26 07:22 UTC" in texts[1]


def test_cli_cloud_loop_writes_summary_and_outputs(tmp_path, monkeypatch):
    summary_file = tmp_path / "summary.md"
    out_file = tmp_path / "out.txt"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))
    monkeypatch.setenv("GITHUB_OUTPUT", str(out_file))
    monkeypatch.setenv("PREDICTOR_PUBLISH_BRANCH", "cloud-shadow")
    seen = {}

    def fake_loop(state_repo, code_dir, **kw):
        seen.update(kw, state_repo=state_repo)
        kw["emit"]("# Loop iteration 1 — x\n")
        kw["set_output"]("raw_kept", "true")
        return loop.LoopResult(iterations=1, failed=False, secret_leak=False, raw_kept=True)

    monkeypatch.setattr(loop, "run_loop", fake_loop)
    monkeypatch.setattr(cli, "_cloud_fetchers", lambda: Fetchers())
    result = CliRunner().invoke(cli.app, ["cloud-loop", "--state-repo", str(tmp_path / "s"),
                                          "--max-minutes", "20", "--interval", "15"])
    assert result.exit_code == 0, result.output
    assert seen["max_minutes"] == 20 and seen["interval"] == 15
    assert seen["branch"] == "cloud-shadow"
    assert seen["base_dir"] == config.settings.data_dir
    assert "# Loop iteration 1" in summary_file.read_text()
    assert out_file.read_text() == "raw_kept=true\n"


@pytest.mark.parametrize("failed,leak", [(True, False), (False, True)])
def test_cli_cloud_loop_exits_1_on_failure_or_leak(tmp_path, monkeypatch, failed, leak):
    monkeypatch.setenv("PREDICTOR_PUBLISH_BRANCH", "cloud-shadow")
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.setattr(loop, "run_loop", lambda *a, **k: loop.LoopResult(
        iterations=1, failed=failed, secret_leak=leak, raw_kept=False))
    monkeypatch.setattr(cli, "_cloud_fetchers", lambda: Fetchers())
    result = CliRunner().invoke(cli.app, ["cloud-loop", "--state-repo", str(tmp_path)])
    assert result.exit_code == 1


def test_cli_cloud_loop_refuses_without_a_publish_branch(tmp_path, monkeypatch):
    monkeypatch.delenv("PREDICTOR_PUBLISH_BRANCH", raising=False)
    called = []
    monkeypatch.setattr(loop, "run_loop", lambda *a, **k: called.append(1))
    result = CliRunner().invoke(cli.app, ["cloud-loop", "--state-repo", str(tmp_path)])
    assert result.exit_code == 1
    assert called == []
