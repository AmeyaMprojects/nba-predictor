"""Tests for `model/publish.py` -- commit and push the prediction log.

Every test works in a throwaway git repository under `tmp_path`, with a bare
temp remote standing in for GitHub (`git init --bare`). This module must
NEVER be exercised against the real predictor repository -- see
global-constraints.md.

Fix round 1: review found two Critical safety holes (a bare `git commit`
sweeping up anything else a human had staged, e.g. a secret; and a run
landing mid-merge/rebase could conclude someone else's in-progress
operation) plus an Important one (a push could carry along a human's own
unpushed, unrelated commits to the PUBLIC remote). The tests below marked
"(fix round 1)" are new and were confirmed to FAIL against the previously
committed version of publish.py (commit 6646f97) before the fix -- see the
fix report.
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


def _establish_remote(repo_dir: Path, remote_dir: Path) -> None:
    """Add `origin` and do the FIRST push by hand -- exactly what the real
    controller does, and what `commit_and_push` itself now refuses to do
    (Important 3: the first push, with no `origin/main` to compare against,
    is always left to a human)."""
    assert _git(repo_dir, "remote", "add", "origin", str(remote_dir)).returncode == 0
    assert _git(repo_dir, "push", "origin", "main").returncode == 0


def _write_log(repo_dir: Path, text: str = '{"a": 1}\n', name: str = "2026-27.jsonl") -> Path:
    path = repo_dir / "predictions" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --- on main, with a working (already-established) remote -----------------

def test_commit_and_push_on_main_succeeds_and_remote_has_it(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    remote = _init_bare_remote(tmp_path / "remote.git")
    _establish_remote(repo, remote)
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
    assert result.error is False
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


# --- Critical 1: a pre-staged file (e.g. a secret) must never be swept in -

def test_pre_staged_file_stays_staged_while_the_prediction_commits(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    secret = repo / "secret.env"
    secret.write_text("TOKEN=super-secret\n", encoding="utf-8")
    assert _git(repo, "add", "secret.env").returncode == 0
    log_path = _write_log(repo)

    result = commit_and_push(repo, [log_path], "predictions: 2026-10-21 slate", push=False)

    assert result.committed is True, result.message
    status = _git(repo, "status", "--porcelain")
    assert "A  secret.env" in status.stdout  # still staged, untouched
    show = _git(repo, "show", "--name-only", "--format=", "HEAD")
    assert "secret.env" not in show.stdout.split()
    assert show.stdout.split() == ["predictions/2026-27.jsonl"]


def test_pre_staged_file_with_no_prediction_change_is_not_committed_and_stays_staged(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    log_path = _write_log(repo)
    first = commit_and_push(repo, [log_path], "predictions: 2026-10-21 slate", push=False)
    assert first.committed is True

    secret = repo / "secret.env"
    secret.write_text("TOKEN=super-secret\n", encoding="utf-8")
    assert _git(repo, "add", "secret.env").returncode == 0

    # The predictions file has not changed since the commit above.
    result = commit_and_push(repo, [log_path], "predictions: 2026-10-22 slate", push=False)

    assert result.committed is False
    assert result.message == "nothing to commit"
    status = _git(repo, "status", "--porcelain")
    assert "A  secret.env" in status.stdout  # still staged, never touched


# --- Critical 2: never land on top of (or conclude) an in-progress op -----

def test_merge_in_progress_is_refused_and_touches_nothing(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    (repo / ".git" / "MERGE_HEAD").write_text("deadbeef\n", encoding="utf-8")
    log_path = _write_log(repo)

    result = commit_and_push(repo, [log_path], "predictions: 2026-10-21 slate", push=False)

    assert result.committed is False
    assert result.error is False
    assert "MERGE_HEAD" in result.message or "in progress" in result.message
    status = _git(repo, "status", "--porcelain")
    assert "predictions/" in status.stdout  # never added to the index


def test_rebase_in_progress_is_refused(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    (repo / ".git" / "rebase-merge").mkdir()
    log_path = _write_log(repo)

    result = commit_and_push(repo, [log_path], "predictions: 2026-10-21 slate", push=False)

    assert result.committed is False
    status = _git(repo, "status", "--porcelain")
    assert "predictions/" in status.stdout


# --- paths must resolve under repo_dir/predictions/ ------------------------

def test_path_outside_predictions_is_refused(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    bad_path = repo / "notes.txt"
    bad_path.write_text("not a prediction file\n", encoding="utf-8")

    result = commit_and_push(repo, [bad_path], "predictions: 2026-10-21 slate", push=False)

    assert result.committed is False
    assert result.error is True
    assert "predictions" in result.message
    status = _git(repo, "status", "--porcelain")
    assert "?? notes.txt" in status.stdout  # never added


def test_path_escaping_via_dotdot_is_refused(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    outside = tmp_path / "outside.txt"
    outside.write_text("escape attempt\n", encoding="utf-8")
    sneaky = repo / "predictions" / ".." / ".." / "outside.txt"

    result = commit_and_push(repo, [sneaky], "predictions: 2026-10-21 slate", push=False)

    assert result.committed is False
    assert result.error is True


# --- no remote ---------------------------------------------------------

def test_no_remote_commits_but_skips_push(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    log_path = _write_log(repo)

    result = commit_and_push(repo, [log_path], "predictions: 2026-10-21 slate")

    assert result.committed is True
    assert result.pushed is False
    assert result.error is False
    assert "no GitHub remote configured yet" in result.message


def test_unpushed_commits_is_none_with_no_remote(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    assert unpushed_commits(repo) is None


# --- Important 3: the very first push is always left to a human -----------

def test_first_push_with_no_origin_main_is_left_to_a_human(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    remote = _init_bare_remote(tmp_path / "remote.git")
    assert _git(repo, "remote", "add", "origin", str(remote)).returncode == 0
    # Deliberately NOT pushed by hand -- origin/main does not exist yet.
    log_path = _write_log(repo)

    result = commit_and_push(repo, [log_path], "predictions: 2026-10-21 slate")

    assert result.committed is True
    assert result.pushed is False
    assert result.error is False
    assert "first push must be done by hand" in result.message
    remote_log = _git(remote, "log", "--oneline", "main")
    assert remote_log.returncode != 0 or remote_log.stdout.strip() == ""


# --- Important 3: never push along someone else's unpushed commit --------

def test_push_refused_when_a_non_prediction_commit_is_unpushed(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    remote = _init_bare_remote(tmp_path / "remote.git")
    _establish_remote(repo, remote)  # origin/main == the seed commit

    # A normal, human commit that is NOT an automated prediction commit,
    # made directly (not through commit_and_push), and never pushed.
    (repo / "README.md").write_text("seed, updated\n", encoding="utf-8")
    assert _git(repo, "add", "README.md").returncode == 0
    assert _git(repo, "commit", "-m", "chore: tweak readme").returncode == 0

    log_path = _write_log(repo)
    result = commit_and_push(repo, [log_path], "predictions: 2026-10-21 slate")

    assert result.committed is True  # our own commit still happened
    assert result.pushed is False
    assert result.error is False
    assert "chore: tweak readme" in result.message
    assert unpushed_commits(repo) == 2  # both commits stayed local
    remote_log = _git(remote, "log", "--format=%s", "main")
    assert "chore: tweak readme" not in remote_log.stdout
    assert "predictions: 2026-10-21 slate" not in remote_log.stdout


def test_push_proceeds_when_every_unpushed_commit_is_our_own(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    remote = _init_bare_remote(tmp_path / "remote.git")
    _establish_remote(repo, remote)

    first_log = _write_log(repo, '{"a": 1}\n')
    first = commit_and_push(repo, [first_log], "predictions: 2026-10-21 slate")
    assert first.pushed is True  # one prior automated commit, already pushed

    second_log = _write_log(repo, '{"a": 1}\n{"b": 2}\n')
    second = commit_and_push(repo, [second_log], "predictions: 2026-10-22 slate")
    assert second.pushed is True


# --- --no-push ---------------------------------------------------------

def test_push_false_skips_push_even_with_a_working_remote(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    remote = _init_bare_remote(tmp_path / "remote.git")
    _establish_remote(repo, remote)
    log_path = _write_log(repo)

    result = commit_and_push(repo, [log_path], "predictions: 2026-10-21 slate", push=False)

    assert result.committed is True
    assert result.pushed is False
    remote_log = _git(remote, "log", "--format=%s", "main")
    assert "predictions: 2026-10-21 slate" not in remote_log.stdout


# --- a failing remote ----------------------------------------------------

def test_failing_remote_keeps_the_commit_and_unpushed_commits_reports_it(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    remote = _init_bare_remote(tmp_path / "remote.git")
    _establish_remote(repo, remote)

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
    assert second.error is False
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


# --- Important 4: no hangs, no prompts, no terminal input -----------------

def test_git_calls_disable_prompts_and_terminal_input(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    real_run = subprocess.run
    calls: list[dict] = []

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", spy)

    assert current_branch(repo) == "main"

    assert calls, "expected at least one git subprocess call to be made"
    for kwargs in calls:
        assert kwargs.get("stdin") == subprocess.DEVNULL
        env = kwargs.get("env") or {}
        assert env.get("GIT_TERMINAL_PROMPT") == "0"
        assert env.get("GIT_ASKPASS") == "/bin/false"
        assert env.get("SSH_ASKPASS") == "/bin/false"
        assert env.get("GIT_SSH_COMMAND") == "ssh -o BatchMode=yes"


def test_git_call_timeout_becomes_a_plain_result_not_an_exception(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout", 1))

    monkeypatch.setattr(subprocess, "run", fake_run)

    # current_branch must not raise -- it degrades to None, same as any
    # other git failure.
    assert current_branch(repo) is None
