"""CLI-level tests for `predictor predict-today` (task 3).

Mirrors tests/test_backtest_cli.py's settings-monkeypatching pattern for the
database, and tests/test_publish.py's throwaway-git-repo pattern for
`repo_dir` -- a bare temp remote stands in for GitHub. This command must
NEVER be exercised against the real predictor repository or a real remote;
see global-constraints.md.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from typer.testing import CliRunner

from predictor import cli, config, db
from predictor.config import Settings
from predictor.model import settings as model_settings
from predictor.model.adjustments import Coefficients
from predictor.model.live import grades_path, log_path, read_log
from predictor.model.ratings import RatingParams
from predictor.model.settings import ModelSettings
from schedule_rows import insert_schedule_row

runner = CliRunner()

SEASON = "2026-27"

S = ModelSettings(
    ratings=RatingParams(k=0.1, margin_cap=20.0, season_regression=0.5, hca_window=100),
    coefficients=Coefficients(
        back_to_back=-2.0, third_in_four=-1.0, travel_per_1000km=-0.5,
        tz_per_hour=-0.25, altitude=1.5,
    ),
    sigma=13.0,
    half_life=None,
    tuning_games=0,
)


def _git(repo_dir: Path, *args: str, timeout: float = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=repo_dir, capture_output=True, text=True, timeout=timeout
    )


def _point_settings_at_tmp(tmp_path, monkeypatch) -> Settings:
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(config, "settings", s)
    monkeypatch.setattr(db, "settings", s)
    return s


def _git_repo(tmp_path: Path, monkeypatch, name: str = "repo") -> Path:
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Predictor Bot")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "bot@example.invalid")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Predictor Bot")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "bot@example.invalid")
    repo_dir = tmp_path / name
    repo_dir.mkdir()
    assert _git(repo_dir, "init", "-b", "main").returncode == 0
    (repo_dir / "README.md").write_text("seed\n", encoding="utf-8")
    assert _git(repo_dir, "add", "README.md").returncode == 0
    assert _git(repo_dir, "commit", "-m", "seed").returncode == 0
    return repo_dir


def _wire(tmp_path, monkeypatch, now, *, with_remote=True):
    """Common setup: a migrated temp DB, a throwaway git repo (on main, with
    a bare remote unless `with_remote=False`), and every indirection point
    (`cli._now`, `cli._repo_dir`, `model_settings.load`) pinned."""
    s = _point_settings_at_tmp(tmp_path, monkeypatch)
    con = db.connect(s.db_path)
    db.migrate(con)

    repo_dir = _git_repo(tmp_path, monkeypatch)
    remote_dir = None
    if with_remote:
        remote_dir = tmp_path / "remote.git"
        remote_dir.mkdir()
        assert _git(remote_dir, "init", "--bare", "-b", "main").returncode == 0
        assert _git(repo_dir, "remote", "add", "origin", str(remote_dir)).returncode == 0

    monkeypatch.setattr(cli, "_now", lambda: now)
    monkeypatch.setattr(cli, "_repo_dir", lambda: repo_dir)
    monkeypatch.setattr(model_settings, "load", lambda: S)

    return con, repo_dir, remote_dir


def test_predict_today_happy_path_predicts_commits_and_pushes(tmp_path, monkeypatch):
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    con, repo_dir, remote_dir = _wire(tmp_path, monkeypatch, now)
    insert_schedule_row(
        con, "0022600001", date(2026, 11, 10), "PHI", "NYK",
        now + timedelta(hours=2), season=SEASON,
    )
    con.close()

    result = runner.invoke(cli.app, ["predict-today"])

    assert result.exit_code == 0, result.output
    assert "1 predicted" in result.output
    assert "0 not predicted" in result.output
    assert "grades appended: 0" in result.output
    assert "committed and pushed" in result.output

    log = read_log(log_path(repo_dir, SEASON))
    assert len(log) == 1
    assert log[0]["game_id"] == "0022600001"

    remote_log = _git(remote_dir, "log", "--format=%s%n%b", "main")
    assert remote_log.returncode == 0
    assert "predictions: 2026-11-10 slate" in remote_log.stdout
    assert "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" in remote_log.stdout


def test_no_push_flag_commits_locally_but_does_not_push(tmp_path, monkeypatch):
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    con, repo_dir, remote_dir = _wire(tmp_path, monkeypatch, now)
    insert_schedule_row(
        con, "0022600002", date(2026, 11, 10), "PHI", "NYK",
        now + timedelta(hours=2), season=SEASON,
    )
    con.close()

    result = runner.invoke(cli.app, ["predict-today", "--no-push"])

    assert result.exit_code == 0, result.output
    assert "push skipped" in result.output

    local_log = _git(repo_dir, "log", "--oneline", "main")
    assert "predictions" in local_log.stdout
    remote_log = _git(remote_dir, "log", "--oneline", "main")
    assert remote_log.returncode != 0 or remote_log.stdout.strip() == ""


def test_rerun_same_day_is_a_duplicate_and_nothing_new_to_commit(tmp_path, monkeypatch):
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    con, repo_dir, remote_dir = _wire(tmp_path, monkeypatch, now)
    insert_schedule_row(
        con, "0022600003", date(2026, 11, 10), "PHI", "NYK",
        now + timedelta(hours=2), season=SEASON,
    )
    con.close()

    first = runner.invoke(cli.app, ["predict-today"])
    assert first.exit_code == 0, first.output

    second = runner.invoke(cli.app, ["predict-today"])
    assert second.exit_code == 0, second.output
    assert "1 duplicate(s) skipped" in second.output
    assert "nothing to commit" in second.output


def test_missing_settings_file_gives_a_plain_message_and_exits_1(tmp_path, monkeypatch):
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    s = _point_settings_at_tmp(tmp_path, monkeypatch)
    con = db.connect(s.db_path)
    db.migrate(con)
    con.close()
    repo_dir = _git_repo(tmp_path, monkeypatch)

    monkeypatch.setattr(cli, "_now", lambda: now)
    monkeypatch.setattr(cli, "_repo_dir", lambda: repo_dir)

    def _raise():
        raise model_settings.SettingsError("No fitted model settings found. Run: predictor fit-model")

    monkeypatch.setattr(model_settings, "load", _raise)

    result = runner.invoke(cli.app, ["predict-today"])

    assert result.exit_code == 1
    assert "No fitted model settings found" in result.output
    assert "Traceback" not in result.output
    # Nothing was written -- the database was never even opened.
    assert not (repo_dir / "predictions").exists()
