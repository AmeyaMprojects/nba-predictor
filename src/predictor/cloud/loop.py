"""cloud-loop: one long GitHub job that runs ``predictor cloud-run`` again
and again, each time in a child process.

GitHub's cron for live.yml fires only every few hours in practice, so the
job loops internally instead of relying on one start per slot.

- The first iteration runs at once; later ones start on the marks
  (minute 7, then every ``interval`` minutes: :07/:22/:37/:52 for 15). When
  an iteration ran past a mark, the next one starts at once (a late slot),
  and the mark after that is counted from that late start's own mark.
- Each iteration is a ``cloud-run`` child process, killed (with everything
  it started) after ``ITERATION_TIMEOUT``: a hung fetch or push costs one
  slot, never the job. A new iteration is started only when it would be
  killed by ``max_minutes`` after the loop's start at the latest. After a
  kill, a stale ``index.lock`` left in the state checkout is removed so the
  next iteration can commit.
- Each iteration has its own data dir, ``<base_dir>/<stamp>-<n>`` (so a raw
  archive is never re-uploaded by a later iteration); the child's step
  summary and ``GITHUB_OUTPUT`` file go there too, and the loop reads
  ``site_changed``, ``raw_upload_failed`` and ``secret_leak`` from it. The
  dir is removed after the iteration, except when the raw release upload
  failed: then its ``raw/`` is moved to ``<base_dir>/kept/<same name>/raw``
  for the workflow to keep as an artifact. Nothing is kept after a secret
  was found, a kill, or a child that reported nothing (its raw files were
  never checked for the secret).
- An iteration that fails (exit status, a kill, no outputs) never stops the
  loop; the loop fails at the end. A secret leak stops the loop at once.
- After an iteration that changed the public predictions on main, site.yml
  is dispatched (``gh workflow run site.yml --ref main``); a failed dispatch
  fails the loop at the end.

Which code runs, and which code is recorded: the code checkout's (sha,
dirty) is read once, at the start, and every child records it on its
prediction and grade lines (``PREDICTOR_CODE_VERSION``/``_DIRTY``). Each
child is a new Python process that imports the package afresh, so it must
also RUN that code. On main the code checkout IS the state checkout, which
every iteration fast-forwards -- with the editable install there, children
would import code merged after the loop began while recording the start
sha. So in that case the loop adds a detached git worktree of the start sha
under ``base_dir`` (outside the checkout, never moved), runs every child
from it with its ``src/`` first on ``PYTHONPATH`` (ahead of the editable
install), and removes it at the end. When the code checkout is a different
directory (shadow mode: the state checkout is a separate worktree) nothing
moves it, and children run from it directly. Dependencies always come from
the job's venv, synced once at the start.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from predictor import config
from predictor.cloud import run
from predictor.model import publish

MARK_OFFSET_MINUTES = 7
ITERATION_TIMEOUT = timedelta(minutes=15)
KEPT_DIRNAME = "kept"
SITE_DISPATCH = ["gh", "workflow", "run", "site.yml", "--ref", "main"]
GH_TIMEOUT = 60
GIT_TIMEOUT = 120
OUTPUT_FILENAME = "github_output"
SUMMARY_FILENAME = "step_summary.md"
CODE_VERSION_ENV = "PREDICTOR_CODE_VERSION"
CODE_DIRTY_ENV = "PREDICTOR_CODE_DIRTY"

# The child: the job's interpreter, importing predictor.cli from whatever
# PYTHONPATH puts first (the pinned code).
CHILD_PREFIX = [sys.executable, "-c", "from predictor.cli import app; app()"]


@dataclass(frozen=True)
class LoopResult:
    iterations: int
    # Any iteration failed or was killed, or its site dispatch failed: the
    # loop exits 1.
    failed: bool
    secret_leak: bool
    raw_kept: bool

    @property
    def ok(self) -> bool:
        return not self.failed and not self.secret_leak


def next_mark(after: datetime, interval: int) -> datetime:
    """The first mark strictly after ``after``: minute 7 of an hour, then
    every ``interval`` minutes (which must divide the hour)."""
    if interval < 1 or 60 % interval:
        raise ValueError(f"--interval must divide 60 minutes, not {interval}")
    hour = after.replace(minute=0, second=0, microsecond=0)
    for h in range(2):
        for minute in range(MARK_OFFSET_MINUTES % interval, 60, interval):
            mark = hour + timedelta(hours=h, minutes=minute)
            if mark > after:
                return mark
    raise AssertionError("unreachable: a mark exists within two hours")


def run_child(argv: list[str], *, env: dict[str, str], cwd: Path, timeout: float) -> int | None:
    """Run ``argv`` in its own session; its exit status, or None when it ran
    past ``timeout`` and was killed together with every process it started
    (git, gh)."""
    proc = subprocess.Popen(argv, env=env, cwd=cwd, stdin=subprocess.DEVNULL,
                            start_new_session=True)
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        return None


def _redact(text: str) -> str:
    text = " ".join(str(text).split())
    try:
        key = config.odds_api_key()
    except Exception:  # noqa: BLE001 -- redaction must never fail the loop
        key = None
    return text.replace(key, "***") if key else text


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                          timeout=GIT_TIMEOUT, stdin=subprocess.DEVNULL)


def _pin_code(code_dir: Path, state_repo: Path, base_dir: Path,
              sha: str | None) -> tuple[Path, Path | None, str]:
    """(where children run from, a worktree to remove at the end, a note)."""
    if sha is None or code_dir.resolve() != state_repo.resolve():
        return code_dir, None, ""
    worktree = base_dir / f"code-{sha}"
    if worktree.exists():
        shutil.rmtree(worktree, ignore_errors=True)
        _git(code_dir, "worktree", "prune")
    result = _git(code_dir, "worktree", "add", "--detach", str(worktree), sha)
    if result.returncode != 0:
        lines = (result.stderr or result.stdout or "").strip().splitlines()
        why = _redact(lines[0])[:160] if lines else f"exit {result.returncode}"
        return code_dir, None, (f"could not pin the code at {sha} ({why}); iterations run "
                                "from the state checkout, which moves")
    return worktree, worktree, ""


def _read_outputs(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        key, sep, value = line.partition("=")
        if sep:
            out[key.strip()] = value.strip()
    return out


def _clear_index_lock(state_repo: Path) -> None:
    """A killed child can leave git's index lock behind; nothing else holds
    it now (the loop runs one child at a time)."""
    try:
        result = _git(state_repo, "rev-parse", "--git-path", "index.lock")
    except (OSError, subprocess.SubprocessError):
        return
    if result.returncode == 0 and result.stdout.strip():
        lock = Path(result.stdout.strip())
        if not lock.is_absolute():
            lock = state_repo / lock
        lock.unlink(missing_ok=True)


def _dispatch_site(gh_runner: Callable, state_repo: Path) -> str | None:
    """Dispatch site.yml; returns why it failed, or None."""
    try:
        result = gh_runner(list(SITE_DISPATCH), cwd=state_repo, capture_output=True, text=True,
                           timeout=GH_TIMEOUT, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError) as exc:
        return type(exc).__name__
    if result.returncode != 0:
        lines = (result.stderr or result.stdout or "").strip().splitlines()
        return _redact(lines[0])[:160] if lines else f"exit {result.returncode}"
    return None


def _child_env(data_dir: Path, branch: str, code_root: Path,
               sha: str | None, dirty: bool) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        PREDICTOR_DATA_DIR=str(data_dir), PREDICTOR_PUBLISH_BRANCH=branch,
        GITHUB_OUTPUT=str(data_dir / OUTPUT_FILENAME),
        GITHUB_STEP_SUMMARY=str(data_dir / SUMMARY_FILENAME),
        PYTHONPATH=os.pathsep.join(
            p for p in (str(code_root / "src"), os.environ.get("PYTHONPATH", "")) if p),
    )
    if sha is not None:
        env[CODE_VERSION_ENV] = sha
        env[CODE_DIRTY_ENV] = "true" if dirty else "false"
    else:
        env.pop(CODE_VERSION_ENV, None)
        env.pop(CODE_DIRTY_ENV, None)
    return env


def _utcnow() -> datetime:
    return datetime.now(UTC)


def run_loop(
    state_repo: Path,
    code_dir: Path,
    *,
    branch: str,
    base_dir: Path,
    fetchers: run.Fetchers = run.Fetchers(),
    max_minutes: float = 330,
    interval: int = 15,
    iteration_timeout: timedelta = ITERATION_TIMEOUT,
    clock: Callable[[], datetime] = _utcnow,
    sleep: Callable[[float], None] = time.sleep,
    child: Callable[..., int | None] = run_child,
    emit: Callable[[str], None] = lambda text: None,
    set_output: Callable[[str, str], None] = lambda key, value: None,
) -> LoopResult:
    """Run iterations until the time budget is spent; see the module docstring.
    ``emit`` gets each iteration's Markdown summary as soon as it is done;
    ``set_output`` gets ``raw_kept``/``secret_leak`` ("true"/"false");
    ``fetchers.gh_runner`` dispatches site.yml."""
    next_mark(clock(), interval)  # validates interval before anything runs
    state_repo, code_dir, base_dir = Path(state_repo), Path(code_dir), Path(base_dir)
    deadline = clock() + timedelta(minutes=max_minutes)
    base_dir.mkdir(parents=True, exist_ok=True)
    sha, dirty = publish.code_version(code_dir)
    code_root, worktree, pin_note = _pin_code(code_dir, state_repo, base_dir, sha)
    if pin_note:
        emit(f"# cloud-loop\n\n{pin_note}\n\n")
    count = 0
    failed = secret_leak = raw_kept = False
    slot: datetime | None = None
    try:
        while True:
            now = clock()
            if slot is None:
                slot = now
            else:
                slot = next_mark(slot, interval)
                if slot < now:
                    slot = now  # a mark was missed: start at once
                if slot + iteration_timeout > deadline:
                    break
                if slot > now:
                    sleep((slot - now).total_seconds())
            count += 1
            started = clock()
            name = f"{started:%Y%m%dT%H%M%SZ}-{count:02d}"
            data_dir = base_dir / name
            data_dir.mkdir(parents=True, exist_ok=True)
            text = f"# Loop iteration {count} — {started:%Y-%m-%d %H:%M} UTC\n\n"
            argv = [*CHILD_PREFIX, "cloud-run", "--state-repo", str(state_repo)]
            try:
                status = child(argv, env=_child_env(data_dir, branch, code_root, sha, dirty),
                               cwd=code_root, timeout=iteration_timeout.total_seconds())
            except Exception as exc:  # noqa: BLE001 -- one bad iteration never stops the loop
                failed = True
                shutil.rmtree(data_dir, ignore_errors=True)
                emit(text + f"iteration could not run: {type(exc).__name__}: {_redact(exc)}\n\n")
                continue
            summary_file = data_dir / SUMMARY_FILENAME
            if summary_file.exists():
                text += summary_file.read_text(encoding="utf-8", errors="replace")
            outputs = _read_outputs(data_dir / OUTPUT_FILENAME)
            if status is None:
                failed = True
                _clear_index_lock(state_repo)
                shutil.rmtree(data_dir, ignore_errors=True)
                emit(text + f"\nITERATION KILLED — still running after "
                     f"{iteration_timeout.total_seconds() / 60:g} min; the next one starts on "
                     "the next mark\n\n")
                continue
            if status != 0:
                failed = True
            if outputs.get("secret_leak") == "true":
                secret_leak = True
                shutil.rmtree(data_dir, ignore_errors=True)
                set_output("secret_leak", "true")
                emit(text + "\nthe loop stopped: a secret was found\n\n")
                break
            if not outputs:
                failed = True
                text += f"\nITERATION FAILED — cloud-run exited {status} and reported nothing\n"
            if outputs.get("raw_upload_failed") == "true":
                kept = base_dir / KEPT_DIRNAME / name
                kept.mkdir(parents=True, exist_ok=True)
                if (data_dir / "raw").exists():
                    shutil.move(str(data_dir / "raw"), str(kept / "raw"))
                raw_kept = True
                set_output("raw_kept", "true")
                text += "\nraw responses kept for the workflow artifact\n"
            shutil.rmtree(data_dir, ignore_errors=True)
            if outputs.get("site_changed") == "true" and branch == "main":
                problem = _dispatch_site(fetchers.gh_runner, state_repo)
                if problem is None:
                    text += "\nsite.yml dispatched\n"
                else:
                    failed = True
                    text += f"\nSITE DISPATCH FAILED — gh workflow run site.yml: {problem}\n"
            emit(text + "\n")
    finally:
        if worktree is not None:
            _git(code_dir, "worktree", "remove", "--force", str(worktree))
    set_output("raw_kept", "true" if raw_kept else "false")
    set_output("secret_leak", "true" if secret_leak else "false")
    return LoopResult(iterations=count, failed=failed, secret_leak=secret_leak,
                      raw_kept=raw_kept)
