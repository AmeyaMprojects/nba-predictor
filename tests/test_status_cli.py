"""CLI-level test for I5: `predictor status` must exit non-zero on PROBLEMS.

Every other command in this CLI exits 1 on a problem; `status` -- the one
command whose whole purpose is health reporting -- did not, so it could
never be wired into an external monitor/cron job to alert on staleness (it
had to be read by a human every time). The output TEXT is unchanged by this
fix; only the exit code is new. Mirrors tests/test_nba_cli.py's pattern for
patching settings so the CLI command uses a temp DB, never the real one.

Task 4 adds `check_live` (live_results/prediction_log/log_published) to the
same report, which needs a `repo_dir` -- every test here points `cli._repo_dir`
at a throwaway git repo (never the real predictor checkout; see
global-constraints.md) via `_point_repo_dir_at_tmp`.
"""

import json
import subprocess
from datetime import UTC, datetime, timedelta

from typer.testing import CliRunner

from predictor import cli, config, db
from predictor.config import Settings

runner = CliRunner()


def _point_settings_at_tmp(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(config, "settings", s)
    monkeypatch.setattr(db, "settings", s)
    return s


def _git(repo_dir, *args):
    return subprocess.run(
        ["git", *args], cwd=repo_dir, capture_output=True, text=True, timeout=30
    )


def _point_repo_dir_at_tmp(tmp_path, monkeypatch):
    """A throwaway repo on `main`, with an `origin` remote and nothing
    unpushed -- so `log_published` (which has no "no games" off-season
    exemption) reports healthy regardless of what the rest of the test sets
    up. Never the real predictor repository (which has no `origin` remote
    configured and would always report `log_published` stale)."""
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Predictor Bot")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "bot@example.invalid")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Predictor Bot")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "bot@example.invalid")

    repo = tmp_path / "repo"
    repo.mkdir()
    assert _git(repo, "init", "-b", "main").returncode == 0
    (repo / "README.md").write_text("seed\n", encoding="utf-8")
    assert _git(repo, "add", "README.md").returncode == 0
    assert _git(repo, "commit", "-m", "seed").returncode == 0

    remote = tmp_path / "remote.git"
    remote.mkdir()
    assert _git(remote, "init", "--bare", "-b", "main").returncode == 0
    assert _git(repo, "remote", "add", "origin", str(remote)).returncode == 0
    assert _git(repo, "push", "origin", "main").returncode == 0

    monkeypatch.setattr(cli, "_repo_dir", lambda: repo)
    return repo


def test_status_exits_nonzero_when_a_source_is_stale(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)
    _point_repo_dir_at_tmp(tmp_path, monkeypatch)
    # A brand-new, empty temp database: every point-in-time table has no
    # rows at all, which check_sources() always reports as stale.
    result = runner.invoke(cli.app, ["status"])

    assert result.exit_code == 1
    assert "PROBLEMS" in result.stdout


def test_status_exits_zero_when_every_source_is_fresh(tmp_path, monkeypatch):
    s = _point_settings_at_tmp(tmp_path, monkeypatch)
    _point_repo_dir_at_tmp(tmp_path, monkeypatch)
    con = db.connect(s.db_path)
    db.migrate(con)

    from datetime import UTC, datetime

    now = datetime.now(UTC)
    con.execute(
        "INSERT INTO games_raw (game_id, season, game_date, home_team,"
        " away_team, status, observed_at) VALUES (?,?,?,?,?,?,?)",
        ["g1", "2024-25", now.date(), "LAL", "BOS", "SCHEDULED", now],
    )
    con.execute(
        "INSERT INTO injury_status_raw (report_date, game_date, team, player,"
        " status, observed_at) VALUES (?,?,?,?,?,?)",
        [now.date(), now.date(), "LAL", "someone", "Out", now],
    )
    con.execute(
        "INSERT INTO odds_snapshots_raw (game_key, book, home_team, away_team,"
        " observed_at) VALUES (?,?,?,?,?)",
        ["g1", "draftkings", "LAL", "BOS", now],
    )
    con.execute(
        "INSERT INTO news_items_raw (item_key, feed, observed_at) VALUES (?,?,?)",
        ["n1", "rss", now],
    )
    con.execute(
        "INSERT INTO schedule_raw (game_id, season, game_date, tip_off_utc,"
        " home_team, away_team, is_neutral_reported, is_neutral, observed_at)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        ["0022400561", "2024-25", now.date(), None, "PHI", "NYK", False, False, now],
    )
    con.execute(
        "INSERT INTO player_games_raw (game_id, season, game_date, season_type,"
        " player_id, player_name, team, source, reconstructed, observed_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?)",
        ["0022400561", "2024-25", now.date(), "regular", 1, "someone", "PHI",
         "nba_stats", True, now],
    )
    con.execute(
        "INSERT INTO injury_reports_raw (observed_at, fetched_at, source_key,"
        " row_count, content_sha) VALUES (?,?,?,?,?)",
        [now, now, "k", 1, "c" * 64],
    )
    con.execute(
        "INSERT INTO player_boxscores_raw (game_id, season, game_date, team, player_id,"
        " did_not_play, source, observed_at) VALUES (?,?,?,?,?,?,?,?)",
        ["0022400561", "2024-25", now.date(), "PHI", "1", False, "espn", now],
    )
    con.close()
    _fake_gh(monkeypatch, now - timedelta(minutes=10))

    result = runner.invoke(cli.app, ["status"])

    assert result.exit_code == 0, result.stdout
    assert "ALL OK" in result.stdout


# --- Task 11: cloud runs, and status on cloud state -------------------------


def _fake_gh(monkeypatch, created_at, conclusion="success"):
    """Points the CLI's gh runner at a fake that answers `gh run list` with
    one run -- tests never run the real gh."""
    calls = []

    def fake(args, **kwargs):
        calls.append((args, kwargs))
        runs = [{"createdAt": created_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "status": "completed", "conclusion": conclusion}]
        return subprocess.CompletedProcess(args, 0, json.dumps(runs), "")

    monkeypatch.setattr(cli, "_gh_runner", fake)
    return calls


def test_mac_status_includes_cloud_runs(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)
    repo = _point_repo_dir_at_tmp(tmp_path, monkeypatch)
    calls = _fake_gh(monkeypatch, datetime.now(UTC) - timedelta(minutes=5))

    result = runner.invoke(cli.app, ["status"])

    assert "cloud_runs: last cloud run 5m ago (success)" in result.stdout
    assert calls and calls[0][1]["cwd"] == repo
    # The Mac's own checks are all still there.
    for name in ("games", "news_items", "schedule", "live_results", "log_published"):
        assert f"] {name}:" in result.stdout


def test_mac_status_without_gh_says_could_not_check(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)
    _point_repo_dir_at_tmp(tmp_path, monkeypatch)

    def missing(args, **kwargs):
        raise FileNotFoundError("gh")

    monkeypatch.setattr(cli, "_gh_runner", missing)
    result = runner.invoke(cli.app, ["status"])

    assert "cloud_runs: could not check" in result.stdout
    assert "gh auth status" in result.stdout
    assert result.exit_code == 1


def _state_checkout(tmp_path, branch="cloud-shadow"):
    """A throwaway state checkout on `branch`, pushed to a bare remote, with
    a few cloud state files."""
    from predictor.cloud import state as cloud_state

    repo = tmp_path / "state-repo"
    repo.mkdir()
    assert _git(repo, "init", "-b", branch).returncode == 0
    state_dir = repo / "state"
    now = datetime.now(UTC)
    cloud_state.append_records(
        state_dir / "schedule" / f"far_fetches-{now:%Y-%m}.jsonl",
        [{"fetched_at": now - timedelta(minutes=30), "days": 7}],
    )
    assert _git(repo, "add", "state").returncode == 0
    assert _git(repo, "commit", "-m", "state: seed").returncode == 0
    remote = tmp_path / "state-remote.git"
    remote.mkdir()
    assert _git(remote, "init", "--bare", "-b", branch).returncode == 0
    assert _git(repo, "remote", "add", "origin", str(remote)).returncode == 0
    assert _git(repo, "push", "origin", branch).returncode == 0
    return repo


def test_status_from_state_skips_news(tmp_path, monkeypatch):
    s = _point_settings_at_tmp(tmp_path / "data", monkeypatch)
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Predictor Bot")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "bot@example.invalid")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Predictor Bot")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "bot@example.invalid")
    monkeypatch.setenv("PREDICTOR_PUBLISH_BRANCH", "cloud-shadow")
    repo = _state_checkout(tmp_path)
    calls = _fake_gh(monkeypatch, datetime.now(UTC) - timedelta(minutes=12))

    result = runner.invoke(cli.app, ["status", "--from-state", str(repo)])

    out = result.stdout
    assert "news_items: news is archived by the Mac (research only)" in out
    assert "[STALE] news_items" not in out
    assert "cloud_runs: last cloud run 12m ago (success)" in out
    assert "player_links:" in out
    assert "prediction_files: committed on cloud-shadow" in out
    assert "log_published: nothing waiting to be pushed" in out
    # Schedule freshness comes from the far-fetch record, not observed_at.
    assert "[OK   ] schedule" in out
    assert "launchctl" not in out and "com.predictor" not in out
    assert calls[0][1]["cwd"] == repo
    # The rebuilt DB is temporary: nothing written to the data dir's DB.
    assert not s.db_path.exists()
    # The empty tables are still problems, so the exit code follows the rule.
    assert result.exit_code == 1


def test_status_from_state_without_a_state_dir_is_refused(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path / "data", monkeypatch)
    _fake_gh(monkeypatch, datetime.now(UTC))
    result = runner.invoke(cli.app, ["status", "--from-state", str(tmp_path / "nowhere")])
    assert result.exit_code == 1
    assert "state" in result.output
