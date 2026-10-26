"""cloud-loop: one long GitHub job that runs `run.run_once` again and again.

GitHub's cron for live.yml fires only every few hours in practice, so the
job loops internally instead of relying on one start per slot.

- The first iteration runs at once; later ones start on the marks
  (minute 7, then every ``interval`` minutes: :07/:22/:37/:52 for 15). When
  an iteration ran past a mark, the next one starts at once (a late slot),
  and the mark after that is counted from that late start's own mark.
- A new iteration is started only when it could finish (``ITERATION_BUDGET``)
  within ``max_minutes`` of the loop's start.
- Each iteration has its own data dir, ``<base_dir>/<stamp>-<n>`` (so a raw
  archive is never re-uploaded by a later iteration). It is removed after
  the iteration, except when the raw release upload failed: then its
  ``raw/`` is moved to ``<base_dir>/kept/<same name>/raw`` for the workflow
  to keep as an artifact. A dir is never kept after a secret was found.
- An iteration that raises, or reports an unexpected or publish error, never
  stops the loop; the loop fails at the end. A secret leak stops the loop at
  once.
- After an iteration that changed the public predictions on main, site.yml
  is dispatched (``gh workflow run site.yml --ref main``); a failed dispatch
  fails the loop at the end.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from predictor import config
from predictor.cloud import run, summary

MARK_OFFSET_MINUTES = 7
ITERATION_BUDGET = timedelta(minutes=10)
KEPT_DIRNAME = "kept"
SITE_DISPATCH = ["gh", "workflow", "run", "site.yml", "--ref", "main"]
GH_TIMEOUT = 60


@dataclass(frozen=True)
class LoopResult:
    iterations: int
    # Any iteration raised, had an unexpected or publish error, or its site
    # dispatch failed: the loop exits 1.
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


@contextmanager
def _data_dir(path: Path) -> Iterator[None]:
    """Point every module-level settings singleton at ``path`` for the
    duration of one iteration."""
    from predictor import db, raw_store

    saved = (config.settings, db.settings, raw_store.settings)
    s = config.Settings(data_dir=path)
    config.settings = db.settings = raw_store.settings = s
    try:
        yield
    finally:
        config.settings, db.settings, raw_store.settings = saved


def _redact(text: str) -> str:
    text = " ".join(str(text).split())
    try:
        key = config.odds_api_key()
    except Exception:  # noqa: BLE001 -- redaction must never fail the loop
        key = None
    return text.replace(key, "***") if key else text


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


def _utcnow() -> datetime:
    return datetime.now(UTC)


def run_loop(
    state_repo: Path,
    code_dir: Path,
    *,
    branch: str,
    base_dir: Path,
    fetchers: run.Fetchers = run.Fetchers(),
    run_attempt: int = 1,
    max_minutes: float = 330,
    interval: int = 15,
    clock: Callable[[], datetime] = _utcnow,
    sleep: Callable[[float], None] = time.sleep,
    run_once: Callable[..., run.RunReport] = run.run_once,
    emit: Callable[[str], None] = lambda text: None,
    set_output: Callable[[str, str], None] = lambda key, value: None,
) -> LoopResult:
    """Run iterations until the time budget is spent; see the module docstring.
    ``emit`` gets each iteration's Markdown summary as soon as it is done;
    ``set_output`` gets ``raw_kept``/``secret_leak`` ("true"/"false")."""
    next_mark(clock(), interval)  # validates interval before anything runs
    state_repo, base_dir = Path(state_repo), Path(base_dir)
    deadline = clock() + timedelta(minutes=max_minutes)
    count = 0
    failed = secret_leak = raw_kept = False
    slot: datetime | None = None
    while True:
        now = clock()
        if slot is None:
            slot = now
        else:
            slot = next_mark(slot, interval)
            if slot < now:
                slot = now  # a mark was missed: start at once
            if slot + ITERATION_BUDGET > deadline:
                break
            if slot > now:
                sleep((slot - now).total_seconds())
        count += 1
        started = clock()
        name = f"{started:%Y%m%dT%H%M%SZ}-{count:02d}"
        data_dir = base_dir / name
        heading = f"# Loop iteration {count} — {started:%Y-%m-%d %H:%M} UTC\n\n"
        report = None
        try:
            with _data_dir(data_dir):
                report = run_once(state_repo, code_dir, branch=branch, fetchers=fetchers,
                                  run_attempt=run_attempt)
        except Exception as exc:  # noqa: BLE001 -- one bad iteration never stops the loop
            failed = True
            emit(heading + f"iteration failed: {type(exc).__name__}: {_redact(exc)}\n\n")
        if report is None:
            shutil.rmtree(data_dir, ignore_errors=True)
            continue
        text = heading + summary.markdown(report)
        if report.unexpected or report.publish_error:
            failed = True
        if report.secret_leak:
            secret_leak = True
            shutil.rmtree(data_dir, ignore_errors=True)
            set_output("secret_leak", "true")
            emit(text + "\nthe loop stopped: a secret was found\n\n")
            break
        if summary.outputs(report)["raw_upload_failed"] == "true":
            kept = base_dir / KEPT_DIRNAME / name
            kept.mkdir(parents=True, exist_ok=True)
            raw_dir = data_dir / "raw"
            if raw_dir.exists():
                shutil.move(str(raw_dir), str(kept / "raw"))
            raw_kept = True
            set_output("raw_kept", "true")
            text += "\nraw responses kept for the workflow artifact\n"
        shutil.rmtree(data_dir, ignore_errors=True)
        if report.site_changed and branch == "main":
            problem = _dispatch_site(fetchers.gh_runner, state_repo)
            if problem is None:
                text += "\nsite.yml dispatched\n"
            else:
                failed = True
                text += f"\nSITE DISPATCH FAILED — gh workflow run site.yml: {problem}\n"
        emit(text + "\n")
    set_output("raw_kept", "true" if raw_kept else "false")
    set_output("secret_leak", "true" if secret_leak else "false")
    return LoopResult(iterations=count, failed=failed, secret_leak=secret_leak,
                      raw_kept=raw_kept)
