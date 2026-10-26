"""cloud-loop: run `cloud-run` as a child process on the :07/:22/:37/:52
marks until the job's time budget is spent. A fake clock and sleep drive
the schedule; a fake child stands in for the cloud-run process
(cloud-run's own tests cover it; tests/test_cloud_run.py also drives the
loop with an in-process child over a real git world)."""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from predictor import cli, config
from predictor.cloud import loop
from predictor.cloud.run import Fetchers

START = datetime(2026, 10, 26, 7, 7, 30, tzinfo=UTC)
FAKE_KEY = "0123456789abcdef" * 2  # built at runtime; never a real key


class FakeTime:
    """clock() and sleep() over one shared fake time."""

    def __init__(self, start: datetime = START):
        self.now = start
        self.sleeps: list[float] = []

    def clock(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        assert seconds > 0
        self.sleeps.append(seconds)
        self.now += timedelta(seconds=seconds)


class FakeChild:
    """A cloud-run child stand-in: records each call, writes raw files in
    the data dir it was given, writes its step summary and GITHUB_OUTPUT
    file, advances the fake time, and returns its exit status. ``plan``
    gives each call in turn: a dict of flags (raw_failed, secret_leak,
    unexpected, site_changed, silent), "hang" (killed at the timeout), or an
    exception to raise."""

    def __init__(self, t: FakeTime, plan=None, run_minutes: float = 3):
        self.t = t
        self.plan = list(plan or [])
        self.run_minutes = run_minutes
        self.calls: list[dict] = []

    @property
    def starts(self) -> list[datetime]:
        return [c["start"] for c in self.calls]

    @property
    def data_dirs(self) -> list[Path]:
        return [Path(c["env"]["PREDICTOR_DATA_DIR"]) for c in self.calls]

    def __call__(self, argv, *, env, cwd, timeout):
        start = self.t.now
        self.calls.append({"argv": argv, "env": env, "cwd": cwd, "timeout": timeout,
                           "start": start})
        d = Path(env["PREDICTOR_DATA_DIR"])
        for sub in ("espn", "odds", "odds_history"):
            (d / "raw" / sub).mkdir(parents=True, exist_ok=True)
            (d / "raw" / sub / "blob.json").write_text("{}", encoding="utf-8")
        (d / "predictor.duckdb").write_text("db", encoding="utf-8")
        what = self.plan.pop(0) if self.plan else {}
        if isinstance(what, BaseException):
            raise what
        if what == "hang":
            self.t.now += timedelta(seconds=timeout)
            return None
        self.t.now += timedelta(minutes=self.run_minutes)
        if what.get("silent"):
            return 1
        Path(env["GITHUB_STEP_SUMMARY"]).write_text(
            f"## Cloud run {start:%Y-%m-%d %H:%M} UTC — branch x\n", encoding="utf-8")
        raw_failed = what.get("raw_failed", False) and not what.get("secret_leak")
        Path(env["GITHUB_OUTPUT"]).write_text(
            f"site_changed={'true' if what.get('site_changed') else 'false'}\n"
            f"raw_upload_failed={'true' if raw_failed else 'false'}\n"
            f"secret_leak={'true' if what.get('secret_leak') else 'false'}\n",
            encoding="utf-8")
        bad = what.get("unexpected") or what.get("secret_leak")
        return 1 if bad else 0


class FakeGh:
    def __init__(self, returncode=0):
        self.calls = []
        self.returncode = returncode

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, self.returncode, "", "boom")


def _loop(tmp_path, t, fake, *, branch="cloud-shadow", gh=None, max_minutes=60,
          state=None, code=None, **kw):
    texts: list[str] = []
    outputs: dict[str, str] = {}
    result = loop.run_loop(
        state or tmp_path / "state", code or tmp_path / "code", branch=branch,
        base_dir=tmp_path / "data", fetchers=Fetchers(gh_runner=gh or FakeGh()),
        max_minutes=max_minutes, clock=t.clock, sleep=t.sleep, child=fake, emit=texts.append,
        set_output=outputs.__setitem__, **kw,
    )
    return result, texts, outputs


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout


def _repo(path: Path) -> Path:
    path.mkdir(parents=True)
    _git(path, "init", "-q", "-b", "main")
    _git(path, "-c", "user.name=t", "-c", "user.email=t@example.invalid",
         "commit", "-q", "--allow-empty", "-m", "start")
    return path


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
    fake = FakeChild(t)
    result, _, _ = _loop(tmp_path, t, fake, max_minutes=60)
    minutes = [s.strftime("%H:%M:%S") for s in fake.starts]
    # Job start 07:07:30 + 60 min = 08:07:30; a start at 07:52 is killed by
    # 08:07 at the latest (15 min watchdog): allowed; 08:07 would run to 08:22.
    assert minutes == ["07:07:30", "07:22:00", "07:37:00", "07:52:00"]
    assert result.iterations == 4
    assert result.ok


def test_stops_when_the_next_start_could_run_past_the_budget(tmp_path):
    t = FakeTime(datetime(2026, 10, 26, 7, 0, tzinfo=UTC))
    fake = FakeChild(t)
    # Budget ends 07:36: 07:22 + 15 min = 07:37 is past it.
    _loop(tmp_path, t, fake, max_minutes=36)
    assert [s.strftime("%H:%M") for s in fake.starts] == ["07:00", "07:07"]


def test_budget_boundary_is_inclusive(tmp_path):
    t = FakeTime(datetime(2026, 10, 26, 7, 0, tzinfo=UTC))
    fake = FakeChild(t)
    _loop(tmp_path, t, fake, max_minutes=37)  # 07:22 + 15 = 07:37 == the end
    assert [s.strftime("%H:%M") for s in fake.starts] == ["07:00", "07:07", "07:22"]


def test_a_missed_mark_starts_at_once(tmp_path):
    t = FakeTime(datetime(2026, 10, 26, 7, 7, tzinfo=UTC))
    fake = FakeChild(t, run_minutes=14)  # ends 07:21... then 07:22 is on time
    _loop(tmp_path, t, fake, max_minutes=45)
    assert [s.strftime("%H:%M") for s in fake.starts] == ["07:07", "07:22", "07:37"]
    t = FakeTime(datetime(2026, 10, 26, 7, 7, tzinfo=UTC))
    fake = FakeChild(t, plan=["hang"])  # killed 07:22:00 -- exactly the mark
    _loop(tmp_path, t, fake, max_minutes=45)
    assert [s.strftime("%H:%M") for s in fake.starts] == ["07:07", "07:22", "07:37"]


def test_an_early_wake_never_repeats_a_mark(tmp_path):
    t = FakeTime(datetime(2026, 10, 26, 7, 7, tzinfo=UTC))
    fake = FakeChild(t)

    def early_sleep(seconds):
        t.sleeps.append(seconds)
        t.now += timedelta(seconds=seconds - 0.5)

    loop.run_loop(tmp_path / "s", tmp_path / "c", branch="cloud-shadow",
                  base_dir=tmp_path / "data", fetchers=Fetchers(gh_runner=FakeGh()),
                  max_minutes=45, clock=t.clock, sleep=early_sleep, child=fake)
    assert len(fake.starts) == 3  # 07:07, ~07:22, ~07:37 -- never 07:22 twice


def test_interval_must_divide_the_hour():
    with pytest.raises(ValueError):
        loop.next_mark(START, 25)


# --- the child ----------------------------------------------------------------


def test_each_iteration_is_a_cloud_run_child_with_its_own_env(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/somewhere")
    t = FakeTime()
    fake = FakeChild(t)
    _loop(tmp_path, t, fake, max_minutes=20, branch="main")
    (call,) = fake.calls
    assert call["argv"][:3] == [sys.executable, "-c", "from predictor.cli import app; app()"]
    assert call["argv"][3:] == ["cloud-run", "--state-repo", str(tmp_path / "state")]
    env = call["env"]
    d = Path(env["PREDICTOR_DATA_DIR"])
    assert d.parent == tmp_path / "data"
    assert env["GITHUB_OUTPUT"] == str(d / "github_output")
    assert env["GITHUB_STEP_SUMMARY"] == str(d / "step_summary.md")
    assert env["PREDICTOR_PUBLISH_BRANCH"] == "main"
    # Different code and state dirs: children run from the code checkout.
    assert call["cwd"] == tmp_path / "code"
    assert env["PYTHONPATH"] == os.pathsep.join([str(tmp_path / "code" / "src"), "/somewhere"])
    assert call["timeout"] == 15 * 60
    # Not a git checkout: no code version to pin; the child reads its own.
    assert "PREDICTOR_CODE_VERSION" not in env


def test_the_code_version_is_pinned_at_the_start(tmp_path):
    code = _repo(tmp_path / "code")
    (code / "dirty.txt").write_text("x", encoding="utf-8")
    start_sha = _git(code, "rev-parse", "--short", "HEAD").strip()
    t = FakeTime()
    fake = FakeChild(t)
    _loop(tmp_path, t, fake, max_minutes=35, code=code)
    assert len(fake.calls) == 2
    for call in fake.calls:
        assert call["env"]["PREDICTOR_CODE_VERSION"] == start_sha
        assert call["env"]["PREDICTOR_CODE_DIRTY"] == "true"
        assert call["cwd"] == code


def test_children_run_from_a_pinned_worktree_when_code_is_the_state_checkout(tmp_path):
    repo = _repo(tmp_path / "repo")
    start_sha = _git(repo, "rev-parse", "--short", "HEAD").strip()
    t = FakeTime()
    heads = []

    class MovingChild(FakeChild):
        def __call__(self, argv, *, env, cwd, timeout):
            heads.append(_git(Path(cwd), "rev-parse", "--short", "HEAD").strip())
            assert env["PYTHONPATH"].split(os.pathsep)[0] == str(Path(cwd) / "src")
            # The run fast-forwards the shared checkout to newer code.
            _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                 "commit", "-q", "--allow-empty", "-m", "newer code")
            return super().__call__(argv, env=env, cwd=cwd, timeout=timeout)

    fake = MovingChild(t)
    _loop(tmp_path, t, fake, max_minutes=35, state=repo, code=repo)
    cwds = {c["cwd"] for c in fake.calls}
    (worktree,) = cwds
    assert worktree != repo and worktree.parent == tmp_path / "data"
    assert heads == [start_sha, start_sha]
    assert all(c["env"]["PREDICTOR_CODE_VERSION"] == start_sha for c in fake.calls)
    # Removed at the end, and unregistered.
    assert not worktree.exists()
    assert str(worktree) not in _git(repo, "worktree", "list")


def test_run_child_kills_a_hung_child_and_what_it_started(tmp_path):
    pid_file = tmp_path / "grandchild.pid"
    script = (
        "import subprocess, sys, time\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(p.pid))\n"
        "time.sleep(60)\n"
    )
    began = time.monotonic()
    status = loop.run_child([sys.executable, "-c", script], env=dict(os.environ),
                            cwd=tmp_path, timeout=2)
    assert status is None
    assert time.monotonic() - began < 30
    pid = int(pid_file.read_text())
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail("the grandchild survived the kill")


def test_the_child_command_runs_the_cli(tmp_path):
    status = loop.run_child([*loop.CHILD_PREFIX, "cloud-run", "--help"],
                            env=dict(os.environ), cwd=tmp_path, timeout=120)
    assert status == 0


def test_run_child_returns_the_exit_status(tmp_path):
    assert loop.run_child([sys.executable, "-c", "raise SystemExit(3)"],
                          env=dict(os.environ), cwd=tmp_path, timeout=30) == 3


# --- failures -----------------------------------------------------------------


def test_a_hung_iteration_is_killed_and_the_loop_goes_on(tmp_path):
    state = _repo(tmp_path / "state")
    lock = state / ".git" / "index.lock"
    t = FakeTime()

    class LockingChild(FakeChild):
        def __call__(self, argv, *, env, cwd, timeout):
            if not self.calls:
                lock.write_text("", encoding="utf-8")  # killed mid-commit
            else:
                assert not lock.exists()
            return super().__call__(argv, env=env, cwd=cwd, timeout=timeout)

    fake = LockingChild(t, plan=["hang"])
    result, texts, _ = _loop(tmp_path, t, fake, state=state)
    assert [s.strftime("%H:%M:%S") for s in fake.starts] == [
        "07:07:30", "07:22:30", "07:37:00", "07:52:00"]
    assert result.ok is False
    assert "KILLED" in texts[0]
    assert not fake.data_dirs[0].exists()
    assert not (tmp_path / "data" / "kept").exists()


def test_an_iteration_that_cannot_start_never_stops_the_loop(tmp_path):
    t = FakeTime()
    fake = FakeChild(t, plan=[OSError("kaboom"), {"unexpected": True}, {}, {}])
    result, texts, _ = _loop(tmp_path, t, fake)
    assert len(fake.starts) == 4
    assert result.ok is False
    assert "OSError" in texts[0] and "kaboom" in texts[0]


def test_unexpected_error_fails_the_loop_at_the_end_only(tmp_path):
    t = FakeTime()
    fake = FakeChild(t, plan=[{"unexpected": True}])
    result, _, _ = _loop(tmp_path, t, fake)
    assert result.iterations == 4
    assert result.ok is False


def test_a_child_that_reports_nothing_fails(tmp_path):
    t = FakeTime()
    fake = FakeChild(t, plan=[{"silent": True}])
    result, texts, _ = _loop(tmp_path, t, fake)
    assert result.iterations == 4 and result.ok is False
    assert "reported nothing" in texts[0]


def test_a_secret_leak_stops_the_loop_at_once(tmp_path):
    t = FakeTime()
    fake = FakeChild(t, plan=[{}, {"secret_leak": True}, {}])
    result, _, outputs = _loop(tmp_path, t, fake)
    assert len(fake.starts) == 2
    assert result.secret_leak is True
    assert result.ok is False
    assert outputs["secret_leak"] == "true"
    assert not fake.data_dirs[1].exists()
    assert not (tmp_path / "data" / "kept").exists()


def test_the_secret_is_redacted_from_an_iteration_error(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))  # never the real key file
    monkeypatch.setenv("ODDS_API_KEY", FAKE_KEY)
    t = FakeTime()
    fake = FakeChild(t, plan=[RuntimeError(f"key {FAKE_KEY}")])
    _, texts, _ = _loop(tmp_path, t, fake, max_minutes=5)
    joined = "".join(texts)
    assert FAKE_KEY not in joined
    assert "RuntimeError" in joined


# --- data dirs ----------------------------------------------------------------


def test_each_iteration_has_its_own_data_dir_removed_after_success(tmp_path):
    t = FakeTime()
    fake = FakeChild(t)
    result, _, outputs = _loop(tmp_path, t, fake)
    assert len(set(fake.data_dirs)) == 4
    for d in fake.data_dirs:
        assert d.parent == tmp_path / "data"
        assert not d.exists()
    assert result.raw_kept is False
    assert outputs == {"raw_kept": "false", "secret_leak": "false"}


def test_raw_kept_only_when_the_upload_failed(tmp_path):
    t = FakeTime()
    fake = FakeChild(t, plan=[{}, {"raw_failed": True}, "hang", {}])
    result, _, outputs = _loop(tmp_path, t, fake)
    assert result.raw_kept is True
    assert outputs["raw_kept"] == "true"
    assert outputs["secret_leak"] == "false"
    kept = tmp_path / "data" / "kept"
    (only,) = list(kept.iterdir())
    assert only.name == fake.data_dirs[1].name
    # Only the raw responses are kept; the database and outputs are not.
    assert (only / "raw" / "espn" / "blob.json").exists()
    assert sorted(p.name for p in only.iterdir()) == ["raw"]
    for d in fake.data_dirs:
        assert not d.exists()


# --- site dispatch ------------------------------------------------------------


def test_site_dispatched_after_an_iteration_that_changed_it_on_main(tmp_path):
    t = FakeTime()
    gh = FakeGh()
    fake = FakeChild(t, plan=[{"site_changed": True}, {}, {"site_changed": True}])
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
    fake = FakeChild(t, plan=[{"site_changed": True}])
    _loop(tmp_path, t, fake, branch="cloud-shadow", gh=gh)
    assert gh.calls == []


def test_a_failed_dispatch_fails_the_loop_but_continues(tmp_path):
    t = FakeTime()
    gh = FakeGh(returncode=1)
    fake = FakeChild(t, plan=[{"site_changed": True}])
    result, texts, _ = _loop(tmp_path, t, fake, branch="main", gh=gh)
    assert result.iterations == 4
    assert result.ok is False
    assert any("site.yml" in text for text in texts)


# --- summaries and the CLI ----------------------------------------------------


def test_each_iteration_summary_is_headed_by_its_time(tmp_path):
    t = FakeTime()
    fake = FakeChild(t)
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
    real = loop.run_loop

    def fake_loop(state_repo, code_dir, **kw):
        inspect.signature(real).bind(state_repo, code_dir, **kw)  # the CLI calls it rightly
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
