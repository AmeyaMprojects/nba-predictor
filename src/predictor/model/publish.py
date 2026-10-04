"""Commit and push the public prediction log and grades file.

Spec: the live-operation plan, task 3. Only ``predictions/`` files are ever
committed here, and only when the checkout is on ``main`` -- see
global-constraints.md. Every git call is ``subprocess.run([...], cwd=repo_dir,
capture_output=True, text=True, timeout=...)`` with no shell, and nothing
here ever force-pushes, rebases, or touches any path the caller did not
explicitly pass in.

A failed push is never treated as a reason to lose work or raise: the commit
already exists locally once ``committed`` is True, and the next scheduled
run's push will pick it up (or a human runs ``git push`` by hand) -- see
``commit_and_push``'s docstring.
"""

from __future__ import annotations

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


@dataclass(frozen=True)
class PublishResult:
    committed: bool
    pushed: bool
    message: str


def _git(repo_dir: Path, args: list[str], timeout: float = _GIT_TIMEOUT):
    return subprocess.run(
        ["git", *args],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        timeout=timeout,
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


def commit_and_push(
    repo_dir: Path, paths: list[Path], message: str, push: bool = True
) -> PublishResult:
    """Commit exactly ``paths`` (nothing else) and, if ``push``, push it.

    Never touches anything if the checkout is not on ``main`` -- a human
    working on a feature branch in this same repo must never have the daily
    job silently commit on top of their checkout. Only the given ``paths``
    are staged, so any other modified file in the working tree is left
    untouched and unstaged.
    """
    branch = current_branch(repo_dir)
    if branch != "main":
        where = f"'{branch}'" if branch is not None else "a non-git or detached checkout"
        return PublishResult(
            committed=False,
            pushed=False,
            message=(
                "the prediction log was written but not committed because "
                f"the checkout is on {where}, not main"
            ),
        )

    existing = [str(p) for p in paths if Path(p).exists()]
    if existing:
        add_result = _git(repo_dir, ["add", "--", *existing])
        if add_result.returncode != 0:
            return PublishResult(
                committed=False,
                pushed=False,
                message=f"git add failed: {add_result.stderr.strip()}",
            )

    staged = _git(repo_dir, ["diff", "--cached", "--name-only"])
    if not staged.stdout.strip():
        return PublishResult(committed=False, pushed=False, message="nothing to commit")

    full_message = f"{message}\n\n{_COAUTHOR_TRAILER}\n"
    commit_result = _git(repo_dir, ["commit", "-m", full_message])
    if commit_result.returncode != 0:
        return PublishResult(
            committed=False,
            pushed=False,
            message=f"git commit failed: {commit_result.stderr.strip()}",
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

    try:
        push_result = _git(repo_dir, ["push", "origin", "main"], timeout=_PUSH_TIMEOUT)
    except subprocess.TimeoutExpired:
        return PublishResult(
            committed=True,
            pushed=False,
            message=(
                "committed, but the push timed out; the commit is kept and "
                "the next run will push it"
            ),
        )
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
