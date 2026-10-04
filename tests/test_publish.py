"""Tests for `model/publish.py` -- commit and push the prediction log.

Every test works in a throwaway git repository under `tmp_path`, with a bare
temp remote standing in for GitHub (`git init --bare`). This module must
NEVER be exercised against the real predictor repository -- see
global-constraints.md.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from predictor.model.publish import (
    PublishResult,
    commit_and_push,
    current_branch,
    unpushed_commits,
)


def _git(repo_dir: Path, *args: str, timeout: float = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=repo_dir, capture_output=True, text=True, timeout=timeout
    )


@pytest.fixture(autouse=True)
def _git_identity(monkeypatch):
    # So commits succeed in CI even with no global git identity configured.
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Predictor Bot")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "bot@example.invalid")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Predictor Bot")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "bot@example.invalid")


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    result = _git(path, "init", "-b", "main")
    assert result.returncode == 0, result.stderr
    return path


def _init_bare_remote(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    result = _git(path, "init", "--bare", "-b", "main")
    assert result.returncode == 0, result.stderr
    return path


def _seed_commit(repo_dir: Path) -> None:
    (repo_dir / "README.md").write_text("seed\n", encoding="utf-8")
    assert _git(repo_dir, "add", "README.md").returncode == 0
    result = _git(repo_dir, "commit", "-m", "seed")
    assert result.returncode == 0, result.stderr


def _write_log(repo_dir: Path, text: str = '{"a": 1}\n') -> Path:
    path = repo_dir / "predictions" / "2026-27.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --- on main, with a working remote --------------------------------------

def test_commit_and_push_on_main_succeeds_and_remote_has_it(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    remote = _init_bare_remote(tmp_path / "remote.git")
    assert _git(repo, "remote", "add", "origin", str(remote)).returncode == 0
    log_path = _write_log(repo)

    result = commit_and_push(repo, [log_path], "predictions: 2026-10-21 slate")

    assert result == PublishResult(committed=True, pushed=True, message="committed and pushed")
    remote_log = _git(remote, "log", "--format=%s%n%b", "main")
    assert remote_log.returncode == 0
    assert "predictions: 2026-10-21 slate" in remote_log.stdout
    assert "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" in remote_log.stdout


# --- not on main -----------------------------------------------------------

def test_not_on_main_commits_nothing(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    assert _git(repo, "checkout", "-b", "feature").returncode == 0
    log_path = _write_log(repo)

    result = commit_and_push(repo, [log_path], "predictions: 2026-10-21 slate")

    assert result.committed is False
    assert result.pushed is False
    assert "feature" in result.message
    status = _git(repo, "status", "--porcelain")
    assert "predictions/" in status.stdout  # still unstaged, never added


# --- only the given paths -------------------------------------------------

def test_only_given_paths_are_committed(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    log_path = _write_log(repo)
    (repo / "other.txt").write_text("not part of this publish\n", encoding="utf-8")

    result = commit_and_push(repo, [log_path], "predictions: 2026-10-21 slate", push=False)

    assert result.committed is True
    status = _git(repo, "status", "--porcelain")
    assert "?? other.txt" in status.stdout
    show = _git(repo, "show", "--name-only", "--format=", "HEAD")
    assert show.stdout.split() == ["predictions/2026-27.jsonl"]


# --- no remote ---------------------------------------------------------

def test_no_remote_commits_but_skips_push(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    log_path = _write_log(repo)

    result = commit_and_push(repo, [log_path], "predictions: 2026-10-21 slate")

    assert result.committed is True
    assert result.pushed is False
    assert "no GitHub remote configured yet" in result.message


def test_unpushed_commits_is_none_with_no_remote(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    assert unpushed_commits(repo) is None


# --- --no-push ---------------------------------------------------------

def test_push_false_skips_push_even_with_a_working_remote(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    remote = _init_bare_remote(tmp_path / "remote.git")
    assert _git(repo, "remote", "add", "origin", str(remote)).returncode == 0
    log_path = _write_log(repo)

    result = commit_and_push(repo, [log_path], "predictions: 2026-10-21 slate", push=False)

    assert result.committed is True
    assert result.pushed is False
    remote_log = _git(remote, "log", "--oneline", "main")
    assert remote_log.returncode != 0 or remote_log.stdout.strip() == ""


# --- a failing remote ----------------------------------------------------

def test_failing_remote_keeps_the_commit_and_unpushed_commits_reports_it(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    remote = _init_bare_remote(tmp_path / "remote.git")
    assert _git(repo, "remote", "add", "origin", str(remote)).returncode == 0

    first_log = _write_log(repo, '{"a": 1}\n')
    first = commit_and_push(repo, [first_log], "predictions: 2026-10-21 slate")
    assert first.pushed is True

    # Point origin at a path that does not exist -- the next push must fail,
    # while origin/main (from the successful push above) still points at
    # the commit before this one.
    assert _git(
        repo, "remote", "set-url", "origin", str(tmp_path / "does-not-exist")
    ).returncode == 0
    second_log = _write_log(repo, '{"a": 1}\n{"b": 2}\n')
    second = commit_and_push(repo, [second_log], "predictions: 2026-10-22 slate")

    assert second.committed is True
    assert second.pushed is False
    assert "next run will push it" in second.message
    assert unpushed_commits(repo) == 1


# --- nothing to commit ---------------------------------------------------

def test_nothing_to_commit_when_the_path_was_never_written(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    never_written = repo / "predictions" / "2026-27.jsonl"

    result = commit_and_push(repo, [never_written], "predictions: 2026-10-21 slate")

    assert result == PublishResult(committed=False, pushed=False, message="nothing to commit")


# --- current_branch -------------------------------------------------------

def test_current_branch_reports_a_feature_branch(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    assert _git(repo, "checkout", "-b", "feature").returncode == 0
    assert current_branch(repo) == "feature"


def test_current_branch_none_outside_a_git_repo(tmp_path):
    assert current_branch(tmp_path) is None
