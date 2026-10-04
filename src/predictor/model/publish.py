"""Commit and push the public prediction log and grades file.

Spec: the live-operation plan, task 3 (fix round 1 after review found two
Critical safety holes: this pushes to a PUBLIC repo, unattended, so it must
NEVER touch anything but the prediction files it was given, NEVER conclude
someone else's in-progress merge/rebase/cherry-pick, and NEVER push a commit
it did not itself make).

Rules enforced here:
  - Every path this module is asked to commit must resolve under
    ``repo_dir/predictions/`` -- anything else is refused outright.
  - Nothing is touched while the checkout is not on ``main``, or while a
    merge/cherry-pick/revert/rebase is in progress (``MERGE_HEAD`` etc. --
    see ``_operation_in_progress``).
  - The commit is scoped with ``git commit --only -- <paths>``, after a
    pathspec-scoped ``git diff --cached --quiet -- <paths>`` check, so any
    OTHER file a human already had staged (e.g. a secret they were about to
    commit themselves) is left exactly as staged -- never swept into our
    commit, never silently discarded.
  - A push only ever happens if EVERY commit between ``origin/main`` and
    ``main`` is one of OUR OWN automated prediction commits (subject starts
    with ``"predictions: "`` and touches only ``predictions/`` paths) --
    otherwise a human's own unpushed work on this checkout could be dragged
    along to the public remote. The very first push (``origin/main`` not
    resolvable yet) is always left to a human.
  - Every git call is ``subprocess.run([...], cwd=repo_dir, capture_output=True,
    text=True, timeout=..., stdin=subprocess.DEVNULL, env=...)`` with no
    shell and no possibility of blocking on a credential/host-key prompt
    (``GIT_TERMINAL_PROMPT=0`` etc.) -- this runs unattended. A timeout or
    OS-level failure never raises out of here; it becomes a plain-message,
    non-committing/non-pushing result instead.
  - Never force-pushes, never rebases.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Mirrors the attribution the live-operation brief specifies for the daily
# automated prediction commits -- unrelated to (and independent of) whatever
# trailer the person/agent implementing THIS module itself uses for their own
# commit to this repository.
_COAUTHOR_TRAILER = "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"

_GIT_TIMEOUT = 30.0
_PUSH_TIMEOUT = 60.0

_PREDICTIONS_DIRNAME = "predictions"

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
)


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


def _validate_paths(repo_dir: Path, paths: list[Path]) -> list[Path] | str:
    """Every path resolved under ``repo_dir/predictions/``, or an error
    message string if any of them is not."""
    repo_root = Path(repo_dir).resolve()
    allowed_root = repo_root / _PREDICTIONS_DIRNAME
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


def _is_safe_prediction_commit(repo_dir: Path, sha: str) -> bool:
    """True only for a commit THIS module could plausibly have made itself:
    subject starts with ``"predictions: "`` and it touches only paths under
    ``predictions/``. Anything else on ``main`` ahead of ``origin/main`` is
    someone else's work, and must never be dragged along by our push."""
    subject = _commit_subject(repo_dir, sha)
    if not subject.startswith("predictions: "):
        return False
    files = _git(repo_dir, ["diff-tree", "--no-commit-id", "--name-only", "-r", sha])
    if files.returncode != 0:
        return False
    names = [n for n in files.stdout.splitlines() if n.strip()]
    return bool(names) and all(n.startswith(f"{_PREDICTIONS_DIRNAME}/") for n in names)


def commit_and_push(
    repo_dir: Path, paths: list[Path], message: str, push: bool = True
) -> PublishResult:
    """Commit exactly ``paths`` (nothing else) and, if ``push``, push it.

    Refuses outright (no git state touched) if any path is outside
    ``repo_dir/predictions/``, if the checkout is not on ``main``, or if a
    merge/cherry-pick/revert/rebase is in progress. Only the given paths are
    ever staged or committed (``git commit --only -- <paths>``, after a
    pathspec-scoped staged-diff check) -- anything else a human already had
    staged is left exactly as they left it. A push only happens if every
    unpushed commit on ``main`` is itself a prior automated prediction
    commit; the very first push (no ``origin/main`` yet) is always left to a
    human.
    """
    validated = _validate_paths(repo_dir, paths)
    if isinstance(validated, str):
        return PublishResult(committed=False, pushed=False, message=validated, error=True)
    paths = validated

    branch = current_branch(repo_dir)
    if branch != "main":
        where = f"'{branch}'" if branch is not None else "a detached HEAD"
        return PublishResult(
            committed=False,
            pushed=False,
            message=(
                "the prediction log was written but not committed because "
                f"the checkout is not on main (it is on {where})"
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
        return PublishResult(committed=False, pushed=False, message="nothing to commit")

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
        return PublishResult(committed=False, pushed=False, message="nothing to commit")

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

    if not _has_remote(repo_dir):
        return PublishResult(
            committed=True,
            pushed=False,
            message="committed, but no GitHub remote configured yet",
        )

    if _git(repo_dir, ["rev-parse", "--verify", "origin/main"]).returncode != 0:
        return PublishResult(
            committed=True,
            pushed=False,
            message="committed, but the first push must be done by hand",
        )

    unpushed = _git(repo_dir, ["rev-list", "origin/main..main"])
    shas = [s.strip() for s in unpushed.stdout.splitlines() if s.strip()]
    offending = [
        _commit_subject(repo_dir, sha) for sha in shas if not _is_safe_prediction_commit(repo_dir, sha)
    ]
    if offending:
        return PublishResult(
            committed=True,
            pushed=False,
            message=(
                "committed, but push was refused because main has unpushed "
                "commit(s) that are not automated prediction commits "
                f"({'; '.join(offending)}); push those by hand first"
            ),
        )

    push_result = _git(repo_dir, ["push", "origin", "main"], timeout=_PUSH_TIMEOUT)
    if push_result.returncode != 0:
        return PublishResult(
            committed=True,
            pushed=False,
            message=(
                "committed, but the push failed "
                f"({push_result.stderr.strip()}); the commit is kept and the "
                "next run will push it"
            ),
        )
    return PublishResult(committed=True, pushed=True, message="committed and pushed")


def unpushed_commits(repo_dir: Path) -> int | None:
    """``git rev-list --count origin/main..main``, or None with no remote
    (or any other error -- e.g. ``origin/main`` not fetched yet)."""
    if not _has_remote(repo_dir):
        return None
    result = _git(repo_dir, ["rev-list", "--count", "origin/main..main"])
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None
