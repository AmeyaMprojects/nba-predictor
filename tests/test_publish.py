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


# --- final fix wave --------------------------------------------------------

from predictor.model import publish  # noqa: E402


def test_sequencer_in_progress_is_refused(tmp_path):
    # A multi-commit cherry-pick/revert stopped between commits leaves only
    # .git/sequencer behind (no CHERRY_PICK_HEAD) -- still someone's
    # unfinished operation.
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    (repo / ".git" / "sequencer").mkdir()
    log_path = _write_log(repo)

    result = commit_and_push(repo, [log_path], "predictions: 2026-10-21 slate")

    assert result.committed is False
    assert "sequencer" in result.message
    assert _git(repo, "rev-list", "--count", "HEAD").stdout.strip() == "1"


def test_push_refused_when_unpushed_commits_cannot_be_listed(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    remote = _init_bare_remote(tmp_path / "remote.git")
    _establish_remote(repo, remote)
    log_path = _write_log(repo)

    real_git = publish._git
    pushes = []

    def failing_rev_list(repo_dir, args, timeout=publish._GIT_TIMEOUT):
        if args[:1] == ["rev-list"]:
            return subprocess.CompletedProcess(["git", *args], 128, "", "fatal: bad revision")
        if args[:1] == ["push"]:
            pushes.append(args)
        return real_git(repo_dir, args, timeout)

    monkeypatch.setattr(publish, "_git", failing_rev_list)
    result = commit_and_push(repo, [log_path], "predictions: 2026-10-21 slate")

    assert result.committed is True
    assert result.pushed is False
    assert result.error is False
    assert "could not list" in result.message
    assert pushes == []


def test_rejected_push_says_github_has_commits_this_checkout_lacks(tmp_path, monkeypatch):
    # Cloud operation task 1: a remote that is merely ahead is now rebased
    # over (see test_pull_rebase_when_remote_is_ahead_and_all_local_commits_are_safe).
    # This message is what is left when the push is STILL rejected after
    # the one fetch+rebase+push retry.
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    remote = _init_bare_remote(tmp_path / "remote.git")
    _establish_remote(repo, remote)

    real_git = publish._git
    pushes = []

    def always_rejected(repo_dir, args, timeout=publish._GIT_TIMEOUT):
        if args[:1] == ["push"]:
            pushes.append(args)
            return subprocess.CompletedProcess(
                ["git", *args], 1, "",
                " ! [rejected]        main -> main (fetch first)\n",
            )
        return real_git(repo_dir, args, timeout)

    monkeypatch.setattr(publish, "_git", always_rejected)
    log_path = _write_log(repo)
    result = commit_and_push(repo, [log_path], "predictions: 2026-10-21 slate")

    assert result.committed is True
    assert result.pushed is False
    assert result.error is False
    assert len(pushes) == 2  # the first push plus exactly one retry
    assert result.message == (
        "GitHub has commits this checkout doesn't — the push to main was "
        "rejected twice; the commit is kept and the next run will try to "
        "rebase onto them automatically"
    )


def test_code_version_of_a_clean_checkout(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    short = _git(repo, "rev-parse", "--short", "HEAD").stdout.strip()
    assert publish.code_version(repo) == (short, False)


def test_code_version_ignores_changes_under_predictions(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    _write_log(repo)  # untracked prediction file
    assert publish.code_version(repo)[1] is False


def test_code_version_is_dirty_with_a_modified_tracked_file(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    (repo / "README.md").write_text("edited\n", encoding="utf-8")
    assert publish.code_version(repo)[1] is True


def test_code_version_is_dirty_with_an_untracked_file_outside_predictions(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    (repo / "new_module.py").write_text("x = 1\n", encoding="utf-8")
    assert publish.code_version(repo)[1] is True


def test_code_version_outside_a_git_repo(tmp_path):
    assert publish.code_version(tmp_path / "nowhere") == (None, False)


def test_unpushed_non_prediction_commits_counts_only_foreign_commits(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    remote = _init_bare_remote(tmp_path / "remote.git")
    _establish_remote(repo, remote)
    assert publish.unpushed_non_prediction_commits(repo) == 0

    commit_and_push(repo, [_write_log(repo)], "predictions: 2026-10-21 slate", push=False)
    (repo / "README.md").write_text("edited\n", encoding="utf-8")
    assert _git(repo, "commit", "-am", "chore: edit").returncode == 0

    assert publish.unpushed_non_prediction_commits(repo) == 1


def test_prediction_files_problem_is_none_on_a_clean_main(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    commit_and_push(repo, [_write_log(repo)], "predictions: 2026-10-21 slate", push=False)
    assert publish.prediction_files_problem(repo) is None


def test_prediction_files_problem_off_main(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    assert _git(repo, "checkout", "-b", "feature").returncode == 0
    assert publish.prediction_files_problem(repo) == (
        "the checkout is on 'feature' — predictions written here are not being "
        "published; switch back to main"
    )


def test_prediction_files_problem_with_uncommitted_predictions(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    _write_log(repo)
    assert publish.prediction_files_problem(repo) == (
        "prediction files have uncommitted changes — the last publish failed; "
        "run predictor predict-today again or check the message in "
        "data/logs/predict.err.log"
    )


def test_prediction_files_problem_mid_operation(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / ".git" / "MERGE_HEAD").write_text(head + "\n", encoding="utf-8")
    problem = publish.prediction_files_problem(repo)
    assert problem is not None
    assert "MERGE_HEAD" in problem


# --- push_pending: retry a failed earlier push, commit nothing --------------


def test_push_pending_pushes_unpushed_prediction_commits(tmp_path):
    from predictor.model.publish import push_pending

    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    remote = _init_bare_remote(tmp_path / "remote.git")
    _establish_remote(repo, remote)
    log_path = _write_log(repo)
    first = commit_and_push(repo, [log_path], "predictions: 2026-10-21 slate", push=False)
    assert first.committed and not first.pushed

    result = push_pending(repo)

    assert result.pushed is True and result.committed is False and result.error is False
    assert unpushed_commits(repo) == 0
    assert "predictions: 2026-10-21 slate" in _git(remote, "log", "--format=%s", "main").stdout


def test_push_pending_with_nothing_unpushed_pushes_nothing(tmp_path):
    from predictor.model.publish import NOTHING_TO_PUSH, push_pending

    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    remote = _init_bare_remote(tmp_path / "remote.git")
    _establish_remote(repo, remote)

    result = push_pending(repo)

    assert result == PublishResult(committed=False, pushed=False, message=NOTHING_TO_PUSH)


def test_push_pending_refuses_when_a_non_prediction_commit_is_unpushed(tmp_path):
    from predictor.model.publish import push_pending

    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    remote = _init_bare_remote(tmp_path / "remote.git")
    _establish_remote(repo, remote)
    log_path = _write_log(repo)
    commit_and_push(repo, [log_path], "predictions: 2026-10-21 slate", push=False)
    (repo / "README.md").write_text("seed, updated\n", encoding="utf-8")
    assert _git(repo, "add", "README.md").returncode == 0
    assert _git(repo, "commit", "-m", "chore: tweak readme").returncode == 0

    result = push_pending(repo)

    assert result.pushed is False and result.error is False
    assert "chore: tweak readme" in result.message
    assert unpushed_commits(repo) == 2


def test_push_pending_refuses_off_main(tmp_path):
    from predictor.model.publish import push_pending

    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    remote = _init_bare_remote(tmp_path / "remote.git")
    _establish_remote(repo, remote)
    log_path = _write_log(repo)
    commit_and_push(repo, [log_path], "predictions: 2026-10-21 slate", push=False)
    assert _git(repo, "checkout", "-b", "feature").returncode == 0

    result = push_pending(repo)

    assert result.pushed is False
    assert "not on main" in result.message
    assert "predictions:" not in _git(remote, "log", "--format=%s", "main").stdout


def test_push_pending_refuses_during_a_merge(tmp_path):
    from predictor.model.publish import push_pending

    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    remote = _init_bare_remote(tmp_path / "remote.git")
    _establish_remote(repo, remote)
    log_path = _write_log(repo)
    commit_and_push(repo, [log_path], "predictions: 2026-10-21 slate", push=False)
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / ".git" / "MERGE_HEAD").write_text(head + "\n", encoding="utf-8")

    result = push_pending(repo)

    assert result.pushed is False
    assert "MERGE_HEAD" in result.message
    assert unpushed_commits(repo) == 1


# --- cloud operation, task 1: state commits, a chosen branch, narrow rebase --


def _write_state(repo_dir: Path, text: str = '{"s": 1}\n', name: str = "runs.jsonl") -> Path:
    path = repo_dir / "state" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _push_from_elsewhere(tmp_path: Path, remote: Path, rel: str, text: str, subject: str,
                         branch: str = "main") -> None:
    other = tmp_path / "other"
    if not other.exists():
        assert _git(tmp_path, "clone", str(remote), str(other)).returncode == 0
    assert _git(other, "checkout", branch).returncode == 0
    assert _git(other, "pull", "--ff-only", "origin", branch).returncode == 0
    target = other / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    assert _git(other, "add", rel).returncode == 0
    assert _git(other, "commit", "-m", subject).returncode == 0
    assert _git(other, "push", "origin", branch).returncode == 0


def _subjects(repo_dir: Path, ref: str) -> list[str]:
    return _git(repo_dir, "log", "--format=%s", ref).stdout.splitlines()


def test_state_commit_touching_only_state_is_committed_and_pushed(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    remote = _init_bare_remote(tmp_path / "remote.git")
    _establish_remote(repo, remote)

    result = commit_and_push(
        repo, [_write_state(repo)], "state: cloud run 2026-10-22T12:37Z", kind="state"
    )

    assert result.committed is True and result.pushed is True and result.error is False
    assert _subjects(remote, "main")[0] == "state: cloud run 2026-10-22T12:37Z"


def test_state_kind_with_a_predictions_path_is_refused(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    before = _git(repo, "log", "--format=%H").stdout

    result = commit_and_push(repo, [_write_log(repo)], "state: x", kind="state")

    assert result.error is True and result.committed is False
    assert "outside" in result.message
    assert _git(repo, "log", "--format=%H").stdout == before


def test_message_without_the_kind_prefix_is_refused(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    before = _git(repo, "log", "--format=%H").stdout

    result = commit_and_push(repo, [_write_state(repo)], "predictions: x", kind="state")

    assert result.error is True and result.committed is False
    assert _git(repo, "log", "--format=%H").stdout == before
    assert _git(repo, "diff", "--cached", "--name-only").stdout == ""


def test_push_refused_when_a_state_commit_touches_predictions(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    remote = _init_bare_remote(tmp_path / "remote.git")
    _establish_remote(repo, remote)
    _write_log(repo, name="a.jsonl")
    assert _git(repo, "add", "predictions/a.jsonl").returncode == 0
    assert _git(repo, "commit", "-m", "state: x").returncode == 0

    result = commit_and_push(repo, [_write_state(repo)], "state: y", kind="state")

    assert result.committed is True
    assert result.pushed is False
    assert "state: x" in result.message
    assert "state: x" not in _subjects(remote, "main")
    assert publish.unpushed_non_prediction_commits(repo) == 1


def test_pull_rebase_when_remote_is_ahead_and_all_local_commits_are_safe(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    remote = _init_bare_remote(tmp_path / "remote.git")
    _establish_remote(repo, remote)
    _push_from_elsewhere(tmp_path, remote, "README.md", "readme from elsewhere\n", "docs: readme")

    result = commit_and_push(repo, [_write_log(repo)], "predictions: 2026-10-22 slate")

    assert result.committed is True and result.pushed is True and result.error is False
    assert _subjects(remote, "main")[:2] == ["predictions: 2026-10-22 slate", "docs: readme"]
    assert _git(repo, "rev-list", "--merges", "origin/main").stdout.strip() == ""
    assert publish._operation_in_progress(repo) is None


def test_no_pull_when_any_local_commit_is_unsafe(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    remote = _init_bare_remote(tmp_path / "remote.git")
    _establish_remote(repo, remote)
    _push_from_elsewhere(tmp_path, remote, "README.md", "readme from elsewhere\n", "docs: readme")
    (repo / "feature.py").write_text("x = 1\n", encoding="utf-8")
    assert _git(repo, "add", "feature.py").returncode == 0
    assert _git(repo, "commit", "-m", "feat: x").returncode == 0
    commit_and_push(repo, [_write_log(repo)], "predictions: 2026-10-22 slate", push=False)
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()

    result = publish.push_pending(repo)

    assert result.pushed is False
    assert "feat: x" in result.message
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == head
    assert not (repo / ".git" / "rebase-merge").exists()
    assert not (repo / ".git" / "rebase-apply").exists()
    assert "docs: readme" not in _subjects(repo, "main")


def test_rebase_conflict_is_aborted_and_commit_kept(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    remote = _init_bare_remote(tmp_path / "remote.git")
    _establish_remote(repo, remote)
    first = commit_and_push(repo, [_write_log(repo, '{"a": 1}\n')], "predictions: 2026-10-21 slate")
    assert first.pushed is True
    _push_from_elsewhere(
        tmp_path, remote, "predictions/2026-27.jsonl", '{"a": 9}\n', "predictions: edited elsewhere"
    )

    result = commit_and_push(repo, [_write_log(repo, '{"a": 2}\n')], "predictions: 2026-10-22 slate")

    assert result.committed is True
    assert result.pushed is False
    assert result.error is False
    assert "nothing pushed" in result.message
    assert publish._operation_in_progress(repo) is None
    assert _subjects(repo, "HEAD")[0] == "predictions: 2026-10-22 slate"
    assert _subjects(remote, "main")[0] == "predictions: edited elsewhere"


def test_rejected_push_is_retried_once_after_a_fresh_rebase(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    remote = _init_bare_remote(tmp_path / "remote.git")
    _establish_remote(repo, remote)

    real_git = publish._git
    pushes = []

    def racing_git(repo_dir, args, timeout=publish._GIT_TIMEOUT):
        if args[:1] == ["push"]:
            pushes.append(args)
            if len(pushes) == 1:
                # Someone pushes between our fetch and our push.
                _push_from_elsewhere(tmp_path, remote, "NOTES.md", "race\n", "docs: race")
        return real_git(repo_dir, args, timeout)

    monkeypatch.setattr(publish, "_git", racing_git)
    result = commit_and_push(repo, [_write_log(repo)], "predictions: 2026-10-22 slate")

    assert result.pushed is True
    assert len(pushes) == 2
    assert _subjects(remote, "main")[:2] == ["predictions: 2026-10-22 slate", "docs: race"]


def test_commit_on_cloud_shadow_pushes_only_that_branch(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    remote = _init_bare_remote(tmp_path / "remote.git")
    _establish_remote(repo, remote)
    assert _git(repo, "checkout", "-b", "cloud-shadow").returncode == 0
    assert _git(repo, "push", "origin", "cloud-shadow").returncode == 0  # first push by hand
    main_sha = _git(remote, "rev-parse", "main").stdout.strip()
    shadow_before = _git(remote, "rev-parse", "cloud-shadow").stdout.strip()

    result = commit_and_push(
        repo, [_write_state(repo)], "state: cloud run 2026-10-22T12:37Z",
        kind="state", branch="cloud-shadow",
    )

    assert result.pushed is True
    assert _git(remote, "rev-parse", "cloud-shadow").stdout.strip() != shadow_before
    assert _subjects(remote, "cloud-shadow")[0] == "state: cloud run 2026-10-22T12:37Z"
    assert _git(remote, "rev-parse", "main").stdout.strip() == main_sha
    assert publish.unpushed_commits(repo, branch="cloud-shadow") == 0


def test_branch_mismatch_commits_nothing(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)

    result = commit_and_push(
        repo, [_write_state(repo)], "state: x", kind="state", branch="cloud-shadow"
    )

    assert result.committed is False and result.pushed is False
    assert "cloud-shadow" in result.message
    assert _git(repo, "rev-list", "--count", "HEAD").stdout.strip() == "1"


def test_code_version_ignores_changes_under_state(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    _write_state(repo)
    assert publish.code_version(repo)[1] is False


def test_head_sha_is_the_short_sha(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    assert publish.head_sha(repo) == _git(repo, "rev-parse", "--short", "HEAD").stdout.strip()
    assert publish.head_sha(tmp_path / "nowhere") is None


# --- task 1 review fixes: stuck rebase, no fork-point drops, redaction -------


def test_stuck_rebase_after_a_failed_abort_is_a_visible_error(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    remote = _init_bare_remote(tmp_path / "remote.git")
    _establish_remote(repo, remote)
    first = commit_and_push(repo, [_write_log(repo, '{"a": 1}\n')], "predictions: 2026-10-21 slate")
    assert first.pushed is True
    _push_from_elsewhere(
        tmp_path, remote, "predictions/2026-27.jsonl", '{"a": 9}\n', "predictions: edited elsewhere"
    )

    real_git = publish._git
    pushes = []

    def failing_abort(repo_dir, args, timeout=publish._GIT_TIMEOUT):
        if args[:2] == ["rebase", "--abort"]:
            return subprocess.CompletedProcess(["git", *args], 1, "", "error: could not abort")
        if args[:1] == ["push"]:
            pushes.append(args)
        return real_git(repo_dir, args, timeout)

    monkeypatch.setattr(publish, "_git", failing_abort)
    result = commit_and_push(repo, [_write_log(repo, '{"a": 2}\n')], "predictions: 2026-10-22 slate")

    assert result.pushed is False
    assert result.error is True
    assert "stuck" in result.message
    assert "git rebase --abort" in result.message
    assert publish._operation_in_progress(repo) is not None
    assert pushes == []


def test_remote_force_push_does_not_drop_a_previously_pushed_local_commit(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    remote = _init_bare_remote(tmp_path / "remote.git")
    _establish_remote(repo, remote)
    seed = _git(repo, "rev-parse", "HEAD").stdout.strip()
    first = commit_and_push(
        repo, [_write_log(repo, '{"a": 1}\n')], "predictions: 2026-10-21 slate"
    )
    assert first.pushed is True

    # Someone force-pushes the remote back to the seed plus their own commit,
    # throwing away our already-pushed prediction commit.
    other = tmp_path / "other"
    assert _git(tmp_path, "clone", str(remote), str(other)).returncode == 0
    assert _git(other, "reset", "--hard", seed).returncode == 0
    (other / "NOTES.md").write_text("rewritten\n", encoding="utf-8")
    assert _git(other, "add", "NOTES.md").returncode == 0
    assert _git(other, "commit", "-m", "docs: rewritten").returncode == 0
    assert _git(other, "push", "--force", "origin", "main").returncode == 0

    result = commit_and_push(
        repo, [_write_log(repo, '{"a": 1}\n{"b": 2}\n')], "predictions: 2026-10-22 slate"
    )

    assert result.pushed is True
    assert _subjects(remote, "main")[:3] == [
        "predictions: 2026-10-22 slate",
        "predictions: 2026-10-21 slate",
        "docs: rewritten",
    ]


def test_push_refused_when_the_rebase_drops_a_local_commit(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    remote = _init_bare_remote(tmp_path / "remote.git")
    _establish_remote(repo, remote)
    commit_and_push(repo, [_write_log(repo)], "predictions: 2026-10-22 slate", push=False)
    ours = _git(repo, "rev-parse", "HEAD").stdout.strip()

    # Another commit, then the very same change, land on the remote from
    # elsewhere -- the rebase would silently drop ours as already applied.
    # (The other commit goes first so the cherry-pick has a different parent
    # and therefore a different sha from ours, whatever the clock says.)
    other = tmp_path / "other"
    assert _git(tmp_path, "clone", str(remote), str(other)).returncode == 0
    (other / "NOTES.md").write_text("x\n", encoding="utf-8")
    assert _git(other, "add", "NOTES.md").returncode == 0
    assert _git(other, "commit", "-m", "docs: notes").returncode == 0
    assert _git(other, "fetch", str(repo), "main").returncode == 0
    assert _git(other, "cherry-pick", ours).returncode == 0
    assert _git(other, "rev-parse", "HEAD").stdout.strip() != ours
    assert _git(other, "push", "origin", "main").returncode == 0
    remote_before = _git(remote, "rev-parse", "main").stdout.strip()

    result = publish.push_pending(repo)

    assert result.pushed is False
    assert result.error is True
    assert "dropped" in result.message
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == ours  # put back
    assert _git(remote, "rev-parse", "main").stdout.strip() == remote_before
    assert publish._operation_in_progress(repo) is None


_PLANTED = "ghs_PlantedTokenValue0123456789"


@pytest.mark.parametrize("failing", ["fetch", "rebase", "push"])
def test_credentials_in_git_stderr_never_reach_the_message(tmp_path, monkeypatch, failing):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    remote = _init_bare_remote(tmp_path / "remote.git")
    _establish_remote(repo, remote)
    if failing == "rebase":
        _push_from_elsewhere(tmp_path, remote, "README.md", "elsewhere\n", "docs: readme")

    real_git = publish._git
    leaky = (
        f"fatal: unable to access 'https://x-access-token:{_PLANTED}@github.com/o/r.git/': "
        f"also https://someone:{_PLANTED}@example.com/x\n"
    )

    def leaking_git(repo_dir, args, timeout=publish._GIT_TIMEOUT):
        if args[:1] == [failing]:
            # Raw, unredacted stderr -- the message layer must redact too.
            return subprocess.CompletedProcess(["git", *args], 1, "", leaky)
        return real_git(repo_dir, args, timeout)

    monkeypatch.setattr(publish, "_git", leaking_git)
    result = commit_and_push(repo, [_write_log(repo)], "predictions: 2026-10-22 slate")

    assert result.pushed is False
    assert _PLANTED not in result.message
    assert "someone" not in result.message


def test_git_helper_redacts_credentials_from_real_stderr(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _seed_commit(repo)
    # A real git call whose stderr echoes a URL with a planted token
    # (check-ref-format rejects it locally -- no network).
    url = f"https://x-access-token:{_PLANTED}@invalid.invalid/o/r.git"
    done = publish._git(repo, ["check-ref-format", "--branch", url])
    assert _PLANTED not in done.stderr and _PLANTED not in done.stdout


def test_redact_hides_userinfo_and_access_tokens():
    text = (
        f"https://x-access-token:{_PLANTED}@github.com/a "
        f"http://user:{_PLANTED}@h/x x-access-token:{_PLANTED}@github.com"
    )
    out = publish._redact(text)
    assert _PLANTED not in out
    assert "github.com/a" in out
