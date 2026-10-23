"""Commit and push the public prediction log, grades file and cloud state.

Spec: the live-operation plan, task 3 (fix round 1 after review found two
Critical safety holes: this pushes to a PUBLIC repo, unattended, so it must
NEVER touch anything but the prediction files it was given, NEVER conclude
someone else's in-progress merge/rebase/cherry-pick, and NEVER push a commit
it did not itself make).

Cloud operation, task 1: two commit kinds and a chosen branch. A
``"predictions"`` commit has the subject prefix ``"predictions: "`` and
touches only ``predictions/``; a ``"state"`` commit has ``"state: "`` and
touches only ``state/``. The branch is ``main`` by default, or the one the
caller passes (the cloud uses ``config.publish_branch()``).

Rules enforced here:
  - Every path this module is asked to commit must resolve under
    ``repo_dir/<kind dir>/`` (``predictions/`` or ``state/``), and the
    message must start with the kind's prefix -- anything else is refused
    outright, before git is touched.
  - Nothing is touched while the checkout is not on the expected branch
    (``main`` unless told otherwise), or while a
    merge/cherry-pick/revert/rebase is in progress (``MERGE_HEAD`` etc. --
    see ``_operation_in_progress``).
  - The commit is scoped with ``git commit --only -- <paths>``, after a
    pathspec-scoped ``git diff --cached --quiet -- <paths>`` check, so any
    OTHER file a human already had staged (e.g. a secret they were about to
    commit themselves) is left exactly as staged -- never swept into our
    commit, never silently discarded.
  - A push only ever happens if EVERY commit between ``origin/<branch>``
    and ``<branch>`` is one of OUR OWN safe automated commits (a
    ``"predictions: "`` commit touching only ``predictions/``, or a
    ``"state: "`` commit touching only ``state/``; a mixed commit is never
    safe) -- otherwise a human's own unpushed work on this checkout could be
    dragged along to the public remote. The very first push
    (``origin/<branch>`` not resolvable yet) is always left to a human.
  - Every git call is ``subprocess.run([...], cwd=repo_dir, capture_output=True,
    text=True, timeout=..., stdin=subprocess.DEVNULL, env=...)`` with no
    shell and no possibility of blocking on a credential/host-key prompt
    (``GIT_TERMINAL_PROMPT=0`` etc.) -- this runs unattended. A timeout or
    OS-level failure never raises out of here; it becomes a plain-message,
    non-committing/non-pushing result instead.
  - Never force-pushes. The ONE rebase allowed is narrow: only after the
    "every unpushed commit is safe" check has passed, ``git fetch origin
    <branch>``; if the remote is ahead, ``git pull --rebase --no-autostash
    origin <branch>`` (so our own safe commits replay on top, never a merge
    commit). A failed rebase is aborted at once (``git rebase --abort``),
    the local commits are kept and nothing is pushed. A push rejected as
    non-fast-forward gets one more fetch+rebase+push. When any unpushed
    commit is unsafe, no fetch or rebase ever runs.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Mirrors the attribution the live-operation brief specifies for the daily
# automated prediction commits -- unrelated to (and independent of) whatever
# trailer the person/agent implementing THIS module itself uses for their own
# commit to this repository.
_COAUTHOR_TRAILER = "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"

_GIT_TIMEOUT = 30.0
_PUSH_TIMEOUT = 60.0

_PREDICTIONS_DIRNAME = "predictions"
_STATE_DIRNAME = "state"

CommitKind = Literal["predictions", "state"]

# kind -> (the only directory it may commit, the required subject prefix)
_KINDS: dict[str, tuple[str, str]] = {
    "predictions": (_PREDICTIONS_DIRNAME, "predictions: "),
    "state": (_STATE_DIRNAME, "state: "),
}

# Never let an unattended git call block on a credential or host-key prompt,
# or read from a controlling terminal that does not exist in a cron/launchd
# context.
_GIT_ENV_OVERRIDES = {
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_ASKPASS": "/bin/false",
    "SSH_ASKPASS": "/bin/false",
    "GIT_SSH_COMMAND": "ssh -o BatchMode=yes",
}

_IN_PROGRESS_MARKERS = (
    "MERGE_HEAD",
    "CHERRY_PICK_HEAD",
    "REVERT_HEAD",
    "rebase-merge",
    "rebase-apply",
    # A multi-commit cherry-pick/revert stopped BETWEEN commits leaves only
    # this directory (no *_HEAD file) -- still an unfinished operation.
    "sequencer",
)

# `git push` stderr fragments meaning the remote has commits this checkout
# lacks -- retrying will never help until a human pulls them.
_REJECTED_MARKERS = ("rejected", "non-fast-forward", "fetch first")


def _rejected_message(branch: str) -> str:
    return (
        "GitHub has commits this checkout doesn't — the push was rejected and "
        "will keep failing until they are pulled by hand (git pull --ff-only "
        f"origin {branch}), then the next run publishes"
    )


_REJECTED_MESSAGE = _rejected_message("main")


NOTHING_TO_COMMIT = "nothing to commit"
NOTHING_TO_PUSH = "nothing to push"


@dataclass(frozen=True)
class PublishResult:
    committed: bool
    pushed: bool
    message: str
    # True only for a genuine git failure (add/commit erroring, or an
    # invalid path) -- the kind of thing `predict-today` should exit 1 for.
    # False for every benign outcome: not on main, nothing to commit, an
    # in-progress operation, a declined/failed/skipped push -- all of those
    # mean the prediction data is safely saved and nothing is lost by
    # retrying later, so they are warnings, not failures.
    error: bool = False


def _git(repo_dir: Path, args: list[str], timeout: float = _GIT_TIMEOUT):
    env = {**os.environ, **_GIT_ENV_OVERRIDES}
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            env=env,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        # Never raise out of here -- the caller always gets something with
        # a `.returncode`/`.stdout`/`.stderr` it can report in plain English.
        return subprocess.CompletedProcess(
            args=["git", *args], returncode=1, stdout="", stderr=str(exc)
        )


def current_branch(repo_dir: Path) -> str | None:
    """The checked-out branch name, or None if it cannot be determined (not
    a git repository, or a detached HEAD -- ``git symbolic-ref`` fails in
    both cases, including on a freshly-initialised, commit-less repo, where
    ``git rev-parse --abbrev-ref HEAD`` would wrongly fail too)."""
    result = _git(repo_dir, ["symbolic-ref", "--short", "HEAD"])
    if result.returncode != 0:
        return None
    branch = result.stdout.strip()
    return branch or None


def _has_remote(repo_dir: Path, name: str = "origin") -> bool:
    result = _git(repo_dir, ["remote"])
    if result.returncode != 0:
        return False
    return name in result.stdout.split()


def _operation_in_progress(repo_dir: Path) -> str | None:
    """The name of the in-progress git operation (a merge, cherry-pick,
    revert or rebase) blocking a safe commit, or None. Checked BEFORE any
    `git add`, so a run that lands mid-operation never adds to, let alone
    concludes, someone else's merge/rebase/cherry-pick."""
    result = _git(repo_dir, ["rev-parse", "--git-dir"])
    if result.returncode != 0:
        return None  # not a git repo at all -- current_branch() handles this
    git_dir = Path(result.stdout.strip())
    if not git_dir.is_absolute():
        git_dir = Path(repo_dir) / git_dir
    for marker in _IN_PROGRESS_MARKERS:
        if (git_dir / marker).exists():
            return marker
    return None


def _validate_paths(
    repo_dir: Path, paths: list[Path], dirname: str = _PREDICTIONS_DIRNAME
) -> list[Path] | str:
    """Every path resolved under ``repo_dir/<dirname>/``, or an error
    message string if any of them is not."""
    repo_root = Path(repo_dir).resolve()
    allowed_root = repo_root / dirname
    resolved: list[Path] = []
    for p in paths:
        candidate = Path(p)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        candidate = candidate.resolve()
        try:
            candidate.relative_to(allowed_root)
        except ValueError:
            return (
                f"refusing to commit {candidate}: it is outside "
                f"{allowed_root}, which is the only directory this command "
                "is ever allowed to commit"
            )
        resolved.append(candidate)
    return resolved


def _commit_subject(repo_dir: Path, sha: str) -> str:
    result = _git(repo_dir, ["log", "-1", "--format=%s", sha])
    return result.stdout.strip() if result.returncode == 0 else sha


def _is_safe_commit(repo_dir: Path, sha: str) -> bool:
    """True only for a commit THIS module could plausibly have made itself:
    subject starts with ``"predictions: "`` and it touches only paths under
    ``predictions/``, or subject starts with ``"state: "`` and it touches
    only paths under ``state/``. Anything else (a mixed commit included)
    ahead of ``origin/<branch>`` is someone else's work, and must never be
    dragged along by our push -- nor rebased by us."""
    subject = _commit_subject(repo_dir, sha)
    dirname = next(
        (d for d, prefix in _KINDS.values() if subject.startswith(prefix)), None
    )
    if dirname is None:
        return False
    files = _git(repo_dir, ["diff-tree", "--no-commit-id", "--name-only", "-r", sha])
    if files.returncode != 0:
        return False
    names = [n for n in files.stdout.splitlines() if n.strip()]
    return bool(names) and all(n.startswith(f"{dirname}/") for n in names)


def commit_and_push(
    repo_dir: Path,
    paths: list[Path],
    message: str,
    push: bool = True,
    *,
    kind: CommitKind = "predictions",
    branch: str = "main",
) -> PublishResult:
    """Commit exactly ``paths`` (nothing else) and, if ``push``, push it.

    Refuses outright (no git state touched) if ``message`` lacks the kind's
    prefix, if any path is outside ``repo_dir/<kind dir>/``, if the checkout
    is not on ``branch``, or if a merge/cherry-pick/revert/rebase is in
    progress. Only the given paths are ever staged or committed (``git
    commit --only -- <paths>``, after a pathspec-scoped staged-diff check)
    -- anything else a human already had staged is left exactly as they left
    it. A push only happens if every unpushed commit on ``branch`` is itself
    a safe automated commit (see ``_is_safe_commit``); the very first push
    (no ``origin/<branch>`` yet) is always left to a human.
    """
    if kind not in _KINDS:
        return PublishResult(
            committed=False, pushed=False, message=f"unknown commit kind {kind!r}", error=True
        )
    dirname, prefix = _KINDS[kind]
    if not message.startswith(prefix):
        return PublishResult(
            committed=False,
            pushed=False,
            message=(
                f"refusing to commit: a {kind} commit message must start with "
                f"{prefix!r} (got {message!r})"
            ),
            error=True,
        )
    validated = _validate_paths(repo_dir, paths, dirname)
    if isinstance(validated, str):
        return PublishResult(committed=False, pushed=False, message=validated, error=True)
    paths = validated

    actual = current_branch(repo_dir)
    if actual != branch:
        where = f"'{actual}'" if actual is not None else "a detached HEAD"
        what = "prediction log" if kind == "predictions" else "state"
        return PublishResult(
            committed=False,
            pushed=False,
            message=(
                f"the {what} was written but not committed because "
                f"the checkout is not on {branch} (it is on {where})"
            ),
        )

    in_progress = _operation_in_progress(repo_dir)
    if in_progress is not None:
        return PublishResult(
            committed=False,
            pushed=False,
            message=(
                f"refusing to commit: a git operation ({in_progress}) is in "
                f"progress in {repo_dir}; it was left untouched -- finish or "
                "abort it by hand first"
            ),
        )

    existing = [str(p) for p in paths if p.exists()]
    if not existing:
        # Nothing to even `git add` -- a pathspec-less `diff`/`commit` below
        # would wrongly fall back to "everything", so this must short-circuit
        # here rather than fall through with an empty pathspec list.
        return PublishResult(committed=False, pushed=False, message=NOTHING_TO_COMMIT)

    add_result = _git(repo_dir, ["add", "--", *existing])
    if add_result.returncode != 0:
        return PublishResult(
            committed=False,
            pushed=False,
            message=f"git add failed: {add_result.stderr.strip()}",
            error=True,
        )

    # Scoped to exactly the paths that exist -- whatever else is (or is not)
    # staged elsewhere in the index never affects this check, and a pathspec
    # naming a file that does not exist on disk would otherwise make
    # `git diff`/`git commit` fail outright.
    staged = _git(repo_dir, ["diff", "--cached", "--quiet", "--", *existing])
    if staged.returncode == 0:
        return PublishResult(committed=False, pushed=False, message=NOTHING_TO_COMMIT)

    full_message = f"{message}\n\n{_COAUTHOR_TRAILER}\n"
    # `--only -- <paths>`: commit ONLY these paths' content, regardless of
    # anything else sitting in the index -- a pre-staged file (e.g. a
    # secret a human was about to commit themselves) is never swept in, and
    # stays staged and uncommitted afterwards.
    commit_result = _git(repo_dir, ["commit", "--only", "-m", full_message, "--", *existing])
    if commit_result.returncode != 0:
        return PublishResult(
            committed=False,
            pushed=False,
            message=f"git commit failed: {commit_result.stderr.strip()}",
            error=True,
        )

    if not push:
        return PublishResult(
            committed=True, pushed=False, message="committed; push skipped (--no-push)"
        )

    return _push_if_safe(repo_dir, committed=True, branch=branch)


def _is_rejected(push_result) -> bool:
    stderr = push_result.stderr.lower()
    return any(marker in stderr for marker in _REJECTED_MARKERS)


def _fetch_and_rebase(repo_dir: Path, branch: str, lead: str, committed: bool) -> PublishResult | None:
    """The narrow rebase: fetch ``origin/<branch>`` and, if it is ahead,
    replay our own (already proven safe) unpushed commits on top of it.
    None when the push may go ahead; otherwise the result to return (with
    any failed rebase already aborted, and the local commits kept)."""
    fetch = _git(repo_dir, ["fetch", "origin", branch], timeout=_PUSH_TIMEOUT)
    if fetch.returncode != 0:
        return PublishResult(
            committed=committed,
            pushed=False,
            message=(
                f"{lead} fetching origin/{branch} failed "
                f"({fetch.stderr.strip()}); the commit is kept and the "
                "next run will push it"
            ),
        )
    behind = _git(repo_dir, ["rev-list", "--count", f"{branch}..origin/{branch}"])
    if behind.returncode != 0:
        return PublishResult(
            committed=committed,
            pushed=False,
            message=(
                f"{lead} the push was refused because git could not compare "
                f"{branch} with origin/{branch} ({behind.stderr.strip()})"
            ),
        )
    if behind.stdout.strip() in ("", "0"):
        return None
    pull = _git(
        repo_dir,
        ["pull", "--rebase", "--no-autostash", "origin", branch],
        timeout=_PUSH_TIMEOUT,
    )
    if pull.returncode != 0:
        # Only our own pull can have started a rebase here: the callers
        # refuse to run at all while any operation is in progress.
        if _operation_in_progress(repo_dir) is not None:
            _git(repo_dir, ["rebase", "--abort"])
        detail = (pull.stderr.strip() or pull.stdout.strip()).splitlines()
        return PublishResult(
            committed=committed,
            pushed=False,
            message=(
                f"{lead} origin/{branch} has new commits and replaying ours on "
                f"top of them failed ({detail[-1] if detail else 'conflict'}); "
                "the rebase was aborted -- kept the local commit(s); nothing pushed"
            ),
        )
    return None


def _push_if_safe(repo_dir: Path, *, committed: bool, branch: str = "main") -> PublishResult:
    """Push ``branch`` to ``origin`` -- but only if EVERY commit on
    ``branch`` not yet on ``origin/<branch>`` is one of our own safe
    automated commits, and never the very first push. Shared by
    ``commit_and_push`` (right after its own commit) and ``push_pending``
    (retrying an earlier failed push); ``committed`` only shapes the
    wording of the result."""
    lead = "committed, but" if committed else "earlier prediction commit(s) are waiting, but"

    if not _has_remote(repo_dir):
        return PublishResult(
            committed=committed,
            pushed=False,
            message=f"{lead} no GitHub remote configured yet",
        )

    if _git(repo_dir, ["rev-parse", "--verify", f"origin/{branch}"]).returncode != 0:
        return PublishResult(
            committed=committed,
            pushed=False,
            message=f"{lead} the first push must be done by hand",
        )

    shas = _unpushed_shas(repo_dir, branch)
    if isinstance(shas, str):
        # Without the list there is no way to prove every unpushed commit
        # is our own -- so nothing is pushed.
        return PublishResult(
            committed=committed,
            pushed=False,
            message=(
                f"{lead} the push was refused because git could not "
                f"list the unpushed commits ({shas}); "
                f"inspect the checkout by hand (git status, git log origin/{branch}..{branch})"
            ),
        )
    if not shas and not committed:
        return PublishResult(committed=False, pushed=False, message=NOTHING_TO_PUSH)
    offending = _unsafe_subjects(repo_dir, shas)
    if offending:
        return _refused_unsafe(lead, branch, offending, committed)

    # Every unpushed commit is our own safe commit -- only now may the
    # narrow fetch + pull --rebase run. One retry after a non-fast-forward
    # rejection (someone pushed between our fetch and our push).
    for attempt in range(2):
        problem = _fetch_and_rebase(repo_dir, branch, lead, committed)
        if problem is not None:
            return problem
        # Belt and braces: the rebased commits must still all be safe.
        shas = _unpushed_shas(repo_dir, branch)
        if isinstance(shas, str):
            return PublishResult(
                committed=committed,
                pushed=False,
                message=f"{lead} the push was refused because git could not list the unpushed commits ({shas})",
            )
        offending = _unsafe_subjects(repo_dir, shas)
        if offending:
            return _refused_unsafe(lead, branch, offending, committed)
        push_result = _git(repo_dir, ["push", "origin", branch], timeout=_PUSH_TIMEOUT)
        if push_result.returncode == 0 or not _is_rejected(push_result) or attempt == 1:
            break
    if push_result.returncode != 0:
        if _is_rejected(push_result):
            return PublishResult(
                committed=committed, pushed=False, message=_rejected_message(branch)
            )
        return PublishResult(
            committed=committed,
            pushed=False,
            message=(
                f"{lead} the push failed "
                f"({push_result.stderr.strip()}); the commit is kept and the "
                "next run will push it"
            ),
        )
    if committed:
        return PublishResult(committed=True, pushed=True, message="committed and pushed")
    return PublishResult(
        committed=False,
        pushed=True,
        message=f"pushed {len(shas)} earlier prediction commit(s)",
    )


def _unpushed_shas(repo_dir: Path, branch: str) -> list[str] | str:
    """The commits on ``branch`` not on ``origin/<branch>``, or git's error
    text if they cannot be listed."""
    result = _git(repo_dir, ["rev-list", f"origin/{branch}..{branch}"])
    if result.returncode != 0:
        return result.stderr.strip()
    return [s.strip() for s in result.stdout.splitlines() if s.strip()]


def _unsafe_subjects(repo_dir: Path, shas: list[str]) -> list[str]:
    return [_commit_subject(repo_dir, sha) for sha in shas if not _is_safe_commit(repo_dir, sha)]


def _refused_unsafe(lead: str, branch: str, offending: list[str], committed: bool) -> PublishResult:
    return PublishResult(
        committed=committed,
        pushed=False,
        message=(
            f"{lead} push was refused because {branch} has unpushed "
            "commit(s) that are not automated prediction commits "
            f"({'; '.join(offending)}); push those by hand first"
        ),
    )


def push_pending(repo_dir: Path, *, branch: str = "main") -> PublishResult:
    """Push prediction commits an earlier run made but could not push (e.g.
    no network at 18:00) -- committing nothing itself.

    Same rules as ``commit_and_push``'s push: nothing happens off
    ``branch`` or while a merge/cherry-pick/revert/rebase is in progress,
    the very first push is left to a human, and a push (or the narrow
    rebase before it) only happens when EVERY unpushed commit is a safe
    automated commit. With nothing unpushed this is a no-op whose message
    is ``NOTHING_TO_PUSH``.
    """
    actual = current_branch(repo_dir)
    if actual != branch:
        where = f"'{actual}'" if actual is not None else "a detached HEAD"
        return PublishResult(
            committed=False,
            pushed=False,
            message=f"nothing pushed because the checkout is not on {branch} (it is on {where})",
        )
    in_progress = _operation_in_progress(repo_dir)
    if in_progress is not None:
        return PublishResult(
            committed=False,
            pushed=False,
            message=(
                f"refusing to push: a git operation ({in_progress}) is in "
                f"progress in {repo_dir}; it was left untouched -- finish or "
                "abort it by hand first"
            ),
        )
    return _push_if_safe(repo_dir, committed=False, branch=branch)


def unpushed_commits(repo_dir: Path, branch: str = "main") -> int | None:
    """``git rev-list --count origin/<branch>..<branch>``, or None with no
    remote (or any other error -- e.g. ``origin/<branch>`` not fetched yet)."""
    if not _has_remote(repo_dir):
        return None
    result = _git(repo_dir, ["rev-list", "--count", f"origin/{branch}..{branch}"])
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


def unpushed_non_prediction_commits(repo_dir: Path, branch: str = "main") -> int | None:
    """How many commits on ``branch`` not yet on ``origin/<branch>`` are NOT
    safe automated commits of either kind (the same predicate
    ``commit_and_push`` uses to refuse a push), or None if it cannot be
    determined."""
    if not _has_remote(repo_dir):
        return None
    shas = _unpushed_shas(repo_dir, branch)
    if isinstance(shas, str):
        return None
    return sum(1 for sha in shas if not _is_safe_commit(repo_dir, sha))


def code_version(repo_dir: Path) -> tuple[str | None, bool]:
    """``(HEAD short sha, dirty)`` for the code that is making predictions.

    ``dirty`` is True when anything outside ``predictions/`` and ``state/``
    differs from HEAD (modified, staged or untracked-and-not-ignored) --
    i.e. the predictions were NOT made by exactly the committed code. The
    prediction and state files themselves are excluded: writing them is
    the run's own job. Never raises: outside a git repository (or on any git
    failure reading HEAD) this is ``(None, False)``; if HEAD is readable
    but the working tree is not, it is reported dirty (unverifiable).
    """
    head = _git(repo_dir, ["rev-parse", "--short", "HEAD"])
    if head.returncode != 0 or not head.stdout.strip():
        return None, False
    status = _git(
        repo_dir,
        [
            "status",
            "--porcelain",
            "--",
            ".",
            f":(exclude){_PREDICTIONS_DIRNAME}",
            f":(exclude){_STATE_DIRNAME}",
        ],
    )
    if status.returncode != 0:
        return head.stdout.strip(), True
    return head.stdout.strip(), bool(status.stdout.strip())


def head_sha(repo_dir: Path) -> str | None:
    """HEAD's short sha, or None outside a git repository (or on any git
    failure)."""
    head = _git(repo_dir, ["rev-parse", "--short", "HEAD"])
    if head.returncode != 0:
        return None
    return head.stdout.strip() or None


def prediction_files_problem(repo_dir: Path, branch: str = "main") -> str | None:
    """Plain-English reason the prediction files are NOT being published
    from this checkout, or None if nothing is wrong with them locally:
    the checkout is off ``branch``, a git operation is in progress, or
    ``predictions/`` has uncommitted changes (a failed publish)."""
    actual = current_branch(repo_dir)
    if actual != branch:
        where = f"'{actual}'" if actual is not None else "a detached HEAD (or is not a git checkout)"
        return (
            f"the checkout is on {where} — predictions written here are not "
            f"being published; switch back to {branch}"
        )
    in_progress = _operation_in_progress(repo_dir)
    if in_progress is not None:
        return (
            f"a git operation ({in_progress}) is in progress — predictions "
            "cannot be committed until it is finished or aborted by hand"
        )
    status = _git(repo_dir, ["status", "--porcelain", "--", f"{_PREDICTIONS_DIRNAME}/"])
    if status.returncode != 0:
        return f"git could not read the prediction files' state ({status.stderr.strip()})"
    if status.stdout.strip():
        return (
            "prediction files have uncommitted changes — the last publish "
            "failed; run predictor predict-today again or check the message "
            "in data/logs/predict.err.log"
        )
    return None
