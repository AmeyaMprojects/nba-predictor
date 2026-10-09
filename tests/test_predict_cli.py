"""CLI-level tests for `predictor predict-today` (task 3).

Mirrors tests/test_backtest_cli.py's settings-monkeypatching pattern for the
database, and tests/test_publish.py's throwaway-git-repo pattern for
`repo_dir` -- a bare temp remote stands in for GitHub. This command must
NEVER be exercised against the real predictor repository or a real remote;
see global-constraints.md.

Fix round 1: `commit_and_push` now refuses the very first push (no
`origin/main` yet) and leaves it to a human -- `_wire`'s `with_remote=True`
therefore does that first push itself (`_establish_remote`), exactly as the
real controller does, so these tests exercise the AUTOMATIC push path that
runs on every subsequent `predict-today` invocation.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from typer.testing import CliRunner

from predictor import cli, config, db, raw_store
from predictor.config import Settings
from predictor.model import settings as model_settings
from predictor.model.adjustments import Coefficients
from predictor.model.live import grades_path, log_path, read_log
from predictor.model.ratings import RatingParams
from predictor.model.settings import ModelSettings
from predictor.sources import results
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
    monkeypatch.setattr(raw_store, "settings", s)
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


def _establish_remote(repo_dir: Path, remote_dir: Path) -> None:
    """Add `origin` and push the seed commit by hand -- what the real
    controller does for the very first push; `commit_and_push` itself
    refuses to do this (see test_publish.py's fix-round-1 tests)."""
    remote_dir.mkdir()
    assert _git(remote_dir, "init", "--bare", "-b", "main").returncode == 0
    assert _git(repo_dir, "remote", "add", "origin", str(remote_dir)).returncode == 0
    assert _git(repo_dir, "push", "origin", "main").returncode == 0


def _wire(tmp_path, monkeypatch, now, *, with_remote=True):
    """Common setup: a migrated temp DB, a throwaway git repo (on main, with
    a bare remote already carrying the seed commit unless
    `with_remote=False`), and every indirection point (`cli._now`,
    `cli._repo_dir`, `model_settings.load`) pinned."""
    s = _point_settings_at_tmp(tmp_path, monkeypatch)
    con = db.connect(s.db_path)
    db.migrate(con)

    repo_dir = _git_repo(tmp_path, monkeypatch)
    remote_dir = None
    if with_remote:
        remote_dir = tmp_path / "remote.git"
        _establish_remote(repo_dir, remote_dir)

    monkeypatch.setattr(cli, "_now", lambda: now)
    monkeypatch.setattr(cli, "_repo_dir", lambda: repo_dir)
    monkeypatch.setattr(model_settings, "load", lambda: S)
    # predict-today catches up missing results first; tests never touch the
    # network, so by default that download finds none (tests that want a
    # successful catch-up replace this).
    monkeypatch.setattr(results, "download", _no_network)

    return con, repo_dir, remote_dir


_REAL_DOWNLOAD = results.download


def _no_network(season):
    raise requests.ConnectionError("Failed to resolve 'stats.nba.com' (test: no network)")


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
    assert "WARNING" not in result.output  # an intentional skip, not a problem

    local_log = _git(repo_dir, "log", "--oneline", "main")
    assert "predictions" in local_log.stdout
    remote_log = _git(remote_dir, "log", "--format=%s", "main")
    assert "predictions:" not in remote_log.stdout


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
    assert "WARNING" in second.output  # benign, but still surfaced


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


def test_empty_slate_still_commits_newly_graded_predictions(tmp_path, monkeypatch):
    """No game is scheduled "today", but an earlier prediction in the SAME
    season's log just became gradable -- the grades file must still be
    written and published even though the predictions log itself gets no
    new line this run (minor 'empty slate with grades appended')."""
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    con, repo_dir, remote_dir = _wire(tmp_path, monkeypatch, now)

    tip = now - timedelta(hours=3)
    insert_schedule_row(con, "0022600777", date(2026, 11, 10), "PHI", "NYK", tip, season=SEASON)
    table = db.POINT_IN_TIME_TABLES["games"]
    con.execute(
        f"INSERT INTO {table} (game_id, season, game_date, home_team, away_team,"
        " home_points, away_points, status, reconstructed, observed_at)"
        " VALUES (?,?,?,?,?,?,?,'FINAL',FALSE,?)",
        ["0022600777", SEASON, date(2026, 11, 10), "PHI", "NYK", 110, 100, now - timedelta(hours=1)],
    )
    import json

    log_path(repo_dir, SEASON).parent.mkdir(parents=True, exist_ok=True)
    predicted_at = tip - timedelta(hours=1)
    line = {
        "predicted_at": predicted_at.isoformat(), "game_id": "0022600777", "season": SEASON,
        "game_date": "2026-11-10", "tip_off_utc": tip.isoformat(), "home_team": "PHI",
        "away_team": "NYK", "status": "predicted", "reason": None, "spread": 0.0, "p_home": 0.6,
        "sentence": None, "terms": None,
        "settings": {"k": 0.1, "margin_cap": 20.0, "season_regression": 0.5, "hca_window": 100,
                     "sigma": 13.0, "half_life": None},
        "stale_results": False, "last_result_capture": None,
    }
    with log_path(repo_dir, SEASON).open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, sort_keys=True) + "\n")
    con.close()

    result = runner.invoke(cli.app, ["predict-today"])

    assert result.exit_code == 0, result.output
    assert "0 predicted" in result.output
    assert "grades appended: 1" in result.output
    assert "committed and pushed" in result.output
    assert not grades_path(repo_dir, SEASON).read_text(encoding="utf-8") == ""
    remote_log = _git(remote_dir, "log", "--format=%s", "main")
    assert "predictions: 2026-11-10 slate" in remote_log.stdout


def test_stale_warning_is_printed_when_a_recent_result_is_missing(tmp_path, monkeypatch):
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    con, repo_dir, remote_dir = _wire(tmp_path, monkeypatch, now)
    # Tipped 20 hours ago -- inside results_missing's window -- with no
    # FINAL row recorded: triggers `stale`.
    insert_schedule_row(
        con, "0022600778", date(2026, 11, 9), "BOS", "MIA",
        now - timedelta(hours=20), season=SEASON,
    )
    insert_schedule_row(
        con, "0022600779", date(2026, 11, 10), "PHI", "NYK",
        now + timedelta(hours=2), season=SEASON,
    )
    con.close()

    result = runner.invoke(cli.app, ["predict-today"])

    assert result.exit_code == 0, result.output
    assert "WARNING: recent results are missing" in result.output


def test_failed_push_exits_zero_with_a_warning(tmp_path, monkeypatch):
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    con, repo_dir, remote_dir = _wire(tmp_path, monkeypatch, now)
    insert_schedule_row(
        con, "0022600780", date(2026, 11, 10), "PHI", "NYK",
        now + timedelta(hours=2), season=SEASON,
    )
    con.close()
    # Break the remote AFTER the first (human) push that established
    # origin/main, so the safety checks pass and only the actual `git push`
    # network call fails.
    assert _git(
        repo_dir, "remote", "set-url", "origin", str(tmp_path / "does-not-exist")
    ).returncode == 0

    result = runner.invoke(cli.app, ["predict-today"])

    assert result.exit_code == 0, result.output
    assert "WARNING" in result.output
    assert "next run will push it" in result.output


# --- final fix wave: slate date, back-fill ---------------------------------


def test_catch_up_run_after_et_midnight_handles_the_previous_et_day(tmp_path, monkeypatch):
    # 04:30 UTC on 10-22 is 00:30 EDT on 10-22: the run still belongs to the
    # 10-21 slate (whose games have all started by now -> not predicted).
    now = datetime(2026, 10, 22, 4, 30, tzinfo=UTC)
    con, repo_dir, remote_dir = _wire(tmp_path, monkeypatch, now)
    insert_schedule_row(
        con, "0022600790", date(2026, 10, 21), "PHI", "NYK",
        datetime(2026, 10, 21, 23, 0, tzinfo=UTC), season=SEASON,
    )
    con.close()

    result = runner.invoke(cli.app, ["predict-today"])

    assert result.exit_code == 0, result.output
    assert "slate 2026-10-21: 0 predicted, 1 not predicted" in result.output
    remote_log = _git(remote_dir, "log", "--format=%s", "main")
    assert "predictions: 2026-10-21 slate" in remote_log.stdout


def test_backfilled_lines_are_reported_and_published_for_their_own_season(tmp_path, monkeypatch):
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    con, repo_dir, remote_dir = _wire(tmp_path, monkeypatch, now)
    # A missed game from 2 days ago, filed under a different season's log.
    insert_schedule_row(
        con, "0022500791", date(2026, 11, 8), "PHI", "NYK",
        datetime(2026, 11, 8, 23, 0, tzinfo=UTC), season="2025-26",
    )
    con.close()

    result = runner.invoke(cli.app, ["predict-today"])

    assert result.exit_code == 0, result.output
    assert "1 back-filled for missed days" in result.output
    assert "committed and pushed" in result.output
    (line,) = read_log(log_path(repo_dir, "2025-26"))
    assert line["game_id"] == "0022500791"
    tracked = _git(repo_dir, "ls-files", "predictions").stdout.split()
    assert "predictions/2025-26.jsonl" in tracked


def test_predict_today_opens_the_database_read_only_with_lock_retry(tmp_path, monkeypatch):
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    con, repo_dir, remote_dir = _wire(tmp_path, monkeypatch, now)
    con.close()
    real = db.connect_with_retry
    calls = []

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **{**kwargs, "attempts": 1})

    monkeypatch.setattr(db, "connect_with_retry", spy)
    result = runner.invoke(cli.app, ["predict-today", "--no-push"])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0].get("read_only") is True


def test_predict_today_lock_still_held_after_retries_is_a_plain_message(tmp_path, monkeypatch):
    import duckdb

    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    con, repo_dir, remote_dir = _wire(tmp_path, monkeypatch, now)
    con.close()

    def locked(*args, **kwargs):
        raise duckdb.IOException("Could not set lock on file: Conflicting lock is held in x")

    monkeypatch.setattr(db, "connect_with_retry", locked)
    result = runner.invoke(cli.app, ["predict-today"])

    assert result.exit_code == 1
    assert "another 'predictor' command is using it" in result.output
    assert "Traceback" not in result.output


# --- CLI failure paths and previous-season grading (final fix wave) ---------


def _predicted_line(game_id, season, game_date, tip, predicted_at, p_home=0.6):
    return {
        "predicted_at": predicted_at.isoformat(), "game_id": game_id, "season": season,
        "game_date": game_date, "tip_off_utc": tip.isoformat(), "home_team": "PHI",
        "away_team": "NYK", "status": "predicted", "reason": None, "spread": 0.0,
        "p_home": p_home, "sentence": None, "terms": None,
        "settings": {"k": 0.1, "margin_cap": 20.0, "season_regression": 0.5,
                     "hca_window": 100, "sigma": 13.0, "half_life": None},
        "stale_results": False, "last_result_capture": None,
    }


def test_failed_git_commit_exits_1_with_the_git_message(tmp_path, monkeypatch):
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    con, repo_dir, remote_dir = _wire(tmp_path, monkeypatch, now)
    insert_schedule_row(con, "0022600881", date(2026, 11, 10), "PHI", "NYK",
                         now + timedelta(hours=2), season=SEASON)
    con.close()
    hook = repo_dir / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\necho 'hook says no' >&2\nexit 1\n", encoding="utf-8")
    hook.chmod(0o755)

    result = runner.invoke(cli.app, ["predict-today"])

    assert result.exit_code == 1
    assert "git commit failed" in result.output
    assert "hook says no" in result.output
    assert "Traceback" not in result.output
    # The prediction itself is safely on disk for the next run to commit.
    assert len(read_log(log_path(repo_dir, SEASON))) == 1


def test_previous_seasons_predictions_are_still_graded_and_published(tmp_path, monkeypatch):
    import json

    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)  # 2026-27 season
    con, repo_dir, remote_dir = _wire(tmp_path, monkeypatch, now)
    prev = "2025-26"
    tip = datetime(2026, 6, 10, 0, 30, tzinfo=UTC)
    insert_schedule_row(con, "0042500401", date(2026, 6, 9), "PHI", "NYK", tip, season=prev)
    table = db.POINT_IN_TIME_TABLES["games"]
    con.execute(
        f"INSERT INTO {table} (game_id, season, game_date, home_team, away_team,"
        " home_points, away_points, status, reconstructed, observed_at)"
        " VALUES (?,?,?,?,?,?,?,'FINAL',FALSE,?)",
        ["0042500401", prev, date(2026, 6, 9), "PHI", "NYK", 101, 99, tip + timedelta(hours=5)],
    )
    con.close()
    path = log_path(repo_dir, prev)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_predicted_line("0042500401", prev, "2026-06-09", tip,
                                   tip - timedelta(hours=6)), sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(cli.app, ["predict-today"])

    assert result.exit_code == 0, result.output
    assert "grades appended: 1" in result.output
    (g,) = read_log(grades_path(repo_dir, prev))
    assert g["game_id"] == "0042500401" and g["correct"] is True
    tracked = _git(repo_dir, "ls-files", "predictions").stdout.split()
    assert "predictions/2025-26-grades.jsonl" in tracked


def test_corrupt_log_line_is_a_plain_message_and_exit_1(tmp_path, monkeypatch):
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    con, repo_dir, remote_dir = _wire(tmp_path, monkeypatch, now)
    insert_schedule_row(con, "0022600882", date(2026, 11, 10), "PHI", "NYK",
                         now + timedelta(hours=2), season=SEASON)
    con.close()
    path = log_path(repo_dir, SEASON)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{this is not json\n", encoding="utf-8")
    before = path.read_bytes()

    result = runner.invoke(cli.app, ["predict-today"])

    assert result.exit_code == 1
    assert "has a corrupt line" in result.output
    assert "repair it by hand" in result.output
    assert "Traceback" not in result.output
    assert path.read_bytes() == before


def test_database_error_mid_query_is_a_plain_message_and_exit_1(tmp_path, monkeypatch):
    import duckdb

    from predictor.model import live

    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    con, repo_dir, remote_dir = _wire(tmp_path, monkeypatch, now)
    con.close()

    def broken(*args, **kwargs):
        raise duckdb.IOException("IO Error: read failed")

    monkeypatch.setattr(live, "predict_today", broken)
    result = runner.invoke(cli.app, ["predict-today"])

    assert result.exit_code == 1
    assert "The database could not be read while predicting" in result.output
    assert "Nothing was published." in result.output
    assert "Traceback" not in result.output
    assert _git(repo_dir, "rev-list", "--count", "HEAD").stdout.strip() == "1"


# --- predict-today catches up missing results first --------------------------

CATCH_UP_UNREACHABLE = (
    "predict-today: could not reach stats.nba.com (no network?) to catch up "
    "missing results -- predicting with the results already recorded."
)


def _finished_game_df(game_id, game_date, home, away, home_pts, away_pts):
    wl = ("W", "L") if home_pts > away_pts else ("L", "W")
    return pd.DataFrame(
        [
            {"GAME_ID": game_id, "GAME_DATE": game_date, "MATCHUP": f"{home} vs. {away}",
             "TEAM_ABBREVIATION": home, "PTS": home_pts, "WL": wl[0]},
            {"GAME_ID": game_id, "GAME_DATE": game_date, "MATCHUP": f"{away} @ {home}",
             "TEAM_ABBREVIATION": away, "PTS": away_pts, "WL": wl[1]},
        ]
    )


def _missing_result_and_todays_game(con, now):
    # Tipped 20 hours ago with no FINAL row: inside results_missing's window.
    insert_schedule_row(
        con, "0022600778", date(2026, 11, 9), "BOS", "MIA",
        now - timedelta(hours=20), season=SEASON,
    )
    insert_schedule_row(
        con, "0022600779", date(2026, 11, 10), "PHI", "NYK",
        now + timedelta(hours=2), season=SEASON,
    )
    con.close()


def test_missing_results_are_captured_before_predicting(tmp_path, monkeypatch):
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    con, repo_dir, remote_dir = _wire(tmp_path, monkeypatch, now)
    _missing_result_and_todays_game(con, now)
    df = _finished_game_df("0022600778", "2026-11-09", "BOS", "MIA", 112, 104)
    original_download = _REAL_DOWNLOAD
    seasons = []

    def fake_download(season):
        seasons.append(season)
        return original_download(season, fetched_at=now, fetch=lambda s: df)

    monkeypatch.setattr(results, "download", fake_download)

    result = runner.invoke(cli.app, ["predict-today", "--no-push"])

    assert result.exit_code == 0, result.output
    assert seasons == [SEASON]
    assert "WARNING: recent results are missing" not in result.output
    con = db.connect(config.settings.db_path, read_only=True)
    try:
        finals = con.execute(
            f"SELECT home_points, away_points, reconstructed FROM {db.POINT_IN_TIME_TABLES['games']} "
            "WHERE game_id = '0022600778' AND status = 'FINAL'"
        ).fetchall()
    finally:
        con.close()
    assert finals == [(112, 104, False)]
    lines = read_log(log_path(repo_dir, SEASON))
    todays = [line for line in lines if line["game_id"] == "0022600779"]
    assert todays and all(line["stale_results"] is False for line in todays)


def test_catch_up_network_failure_is_one_line_and_predictions_still_written(tmp_path, monkeypatch):
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    con, repo_dir, remote_dir = _wire(tmp_path, monkeypatch, now)
    _missing_result_and_todays_game(con, now)

    result = runner.invoke(cli.app, ["predict-today", "--no-push"])

    assert result.exit_code == 0, result.output
    assert "Traceback" not in result.output
    assert result.output.count(CATCH_UP_UNREACHABLE) == 1
    assert "stats.nba.com" not in result.output.replace(CATCH_UP_UNREACHABLE, "")
    assert "WARNING: recent results are missing" in result.output
    lines = read_log(log_path(repo_dir, SEASON))
    todays = [line for line in lines if line["game_id"] == "0022600779"]
    assert todays and all(line["stale_results"] is True for line in todays)


def test_catch_up_other_capture_failure_is_one_line_and_predict_continues(tmp_path, monkeypatch):
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    con, repo_dir, remote_dir = _wire(tmp_path, monkeypatch, now)
    _missing_result_and_todays_game(con, now)

    def conflict(season):
        raise raw_store.RawStoreConflict("key already holds different bytes")

    monkeypatch.setattr(results, "download", conflict)

    result = runner.invoke(cli.app, ["predict-today", "--no-push"])

    assert result.exit_code == 0, result.output
    assert "Traceback" not in result.output
    assert "predict-today: could not catch up missing results" in result.output
    assert read_log(log_path(repo_dir, SEASON))


def test_no_catch_up_when_no_results_are_missing(tmp_path, monkeypatch):
    now = datetime(2026, 11, 10, 20, 0, tzinfo=UTC)
    con, repo_dir, remote_dir = _wire(tmp_path, monkeypatch, now)
    insert_schedule_row(
        con, "0022600779", date(2026, 11, 10), "PHI", "NYK",
        now + timedelta(hours=2), season=SEASON,
    )
    con.close()
    calls = []
    monkeypatch.setattr(results, "download", lambda season: calls.append(season))

    result = runner.invoke(cli.app, ["predict-today", "--no-push"])

    assert result.exit_code == 0, result.output
    assert calls == []
