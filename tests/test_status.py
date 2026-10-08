import subprocess
from datetime import UTC, datetime, timedelta

import pytest

from predictor import db, status
from schedule_rows import insert_schedule_row

NOW = datetime(2025, 1, 15, 20, 0, tzinfo=UTC)
SEASON = "2024-25"


@pytest.fixture
def con(tmp_path):
    c = db.connect(tmp_path / "t.duckdb")
    db.migrate(c)
    return c


def _add_injury(con, observed_at, player="x"):
    con.execute(
        "INSERT INTO injury_status_raw (report_date, game_date, team, player, status, observed_at)"
        " VALUES (?,?,?,?,?,?)",
        [observed_at.date(), observed_at.date(), "LAL", player, "Out", observed_at],
    )


def test_empty_source_is_reported_stale_with_advice(con):
    health = {h.name: h for h in status.check_sources(con, NOW)}
    injuries = health["injury_status"]
    assert injuries.row_count == 0
    assert injuries.stale is True
    assert injuries.advice


def test_fresh_source_is_not_stale(con):
    _add_injury(con, NOW - timedelta(hours=1))
    health = {h.name: h for h in status.check_sources(con, NOW)}
    assert health["injury_status"].stale is False


def test_old_source_is_flagged_stale(con):
    _add_injury(con, NOW - timedelta(days=5))
    health = {h.name: h for h in status.check_sources(con, NOW)}
    injuries = health["injury_status"]
    assert injuries.stale is True
    assert injuries.age_hours == pytest.approx(120, abs=1)


def test_report_is_plain_english_and_names_problem_sources(con):
    _add_injury(con, NOW - timedelta(days=5))
    text = status.format_report(status.check_sources(con, NOW))
    assert "injury_status" in text
    assert "STALE" in text
    assert "OK" in text or "stale" in text.lower()


def test_report_leads_with_overall_verdict(con):
    text = status.format_report(status.check_sources(con, NOW))
    assert text.splitlines()[0].startswith(("PROBLEMS", "ALL OK"))


def test_all_four_logical_sources_are_reported(con):
    health = status.check_sources(con, NOW)
    names = {h.name for h in health}
    assert names == {"games", "injury_status", "odds_snapshots", "news_items", "schedule"}


def test_report_names_every_source(con):
    _add_injury(con, NOW - timedelta(hours=1))
    text = status.format_report(status.check_sources(con, NOW))
    for name in ("games", "injury_status", "odds_snapshots", "news_items", "schedule"):
        assert name in text


def test_odds_advice_names_the_key_file_not_an_env_export(con, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    health = {h.name: h for h in status.check_sources(con, NOW)}
    advice = health["odds_snapshots"].advice
    assert str(tmp_path / "home" / ".config" / "predictor" / "odds_api_key") in advice
    assert "export ODDS_API_KEY" not in advice
    assert "predictor ingest-odds" in advice


def test_stale_schedule_advice_names_the_command_and_the_launchd_job(con):
    health = {h.name: h for h in status.check_sources(con, NOW)}
    advice = health["schedule"].advice
    assert "predictor ingest-schedule" in advice
    assert "com.predictor.schedule" in advice


def test_schedule_fresh_within_a_day_is_not_stale(con):
    con.execute(
        "INSERT INTO schedule_raw (game_id, season, game_date, tip_off_utc,"
        " home_team, away_team, is_neutral_reported, is_neutral, observed_at)"
        " VALUES ('0022400561', '2024-25', DATE '2025-01-15', NULL, 'PHI', 'NYK',"
        " FALSE, FALSE, ?)",
        [NOW - timedelta(hours=20)],
    )
    health = {h.name: h for h in status.check_sources(con, NOW)}
    assert health["schedule"].stale is False


# --- check_live: live_results, prediction_log, log_published --------------


def _insert_final(con, game_id, game_date, observed_at, reconstructed=False):
    table = db.POINT_IN_TIME_TABLES["games"]
    con.execute(
        f"INSERT INTO {table} (game_id, season, game_date, home_team, away_team,"
        " home_points, away_points, status, reconstructed, observed_at)"
        " VALUES (?,?,?,?,?,?,?,'FINAL',?,?)",
        [game_id, SEASON, game_date, "PHI", "NYK", 100, 90, reconstructed, observed_at],
    )


def _git(repo_dir, *args, timeout=30):
    return subprocess.run(
        ["git", *args], cwd=repo_dir, capture_output=True, text=True, timeout=timeout
    )


@pytest.fixture(autouse=True)
def _git_identity(monkeypatch):
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Predictor Bot")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "bot@example.invalid")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Predictor Bot")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "bot@example.invalid")


def _repo_with_remote(path, ahead_commits=0):
    """A throwaway repo on `main` with an `origin` remote, `ahead_commits`
    commits ahead of `origin/main` (0 means fully pushed/in sync)."""
    path.mkdir(parents=True, exist_ok=True)
    assert _git(path, "init", "-b", "main").returncode == 0
    (path / "README.md").write_text("seed\n", encoding="utf-8")
    assert _git(path, "add", "README.md").returncode == 0
    assert _git(path, "commit", "-m", "seed").returncode == 0

    remote = path.parent / f"{path.name}-remote.git"
    remote.mkdir(parents=True, exist_ok=True)
    assert _git(remote, "init", "--bare", "-b", "main").returncode == 0
    assert _git(path, "remote", "add", "origin", str(remote)).returncode == 0
    assert _git(path, "push", "origin", "main").returncode == 0

    for i in range(ahead_commits):
        (path / f"extra{i}.txt").write_text("x\n", encoding="utf-8")
        assert _git(path, "add", f"extra{i}.txt").returncode == 0
        assert _git(path, "commit", "-m", f"extra {i}").returncode == 0

    return path


def _repo_without_remote(path):
    path.mkdir(parents=True, exist_ok=True)
    assert _git(path, "init", "-b", "main").returncode == 0
    (path / "README.md").write_text("seed\n", encoding="utf-8")
    assert _git(path, "add", "README.md").returncode == 0
    assert _git(path, "commit", "-m", "seed").returncode == 0
    return path


def test_off_season_is_quiet_with_no_games(con, tmp_path):
    repo = _repo_with_remote(tmp_path / "repo")
    health = {h.name: h for h in status.check_live(con, repo, NOW)}
    assert health["live_results"].stale is False
    assert health["prediction_log"].stale is False
    assert health["log_published"].stale is False


def test_live_results_stale_when_results_missing(con, tmp_path):
    repo = _repo_with_remote(tmp_path / "repo")
    insert_schedule_row(con, "0022400001", NOW.date(), "PHI", "NYK",
                         NOW - timedelta(hours=20), season=SEASON)
    health = {h.name: h for h in status.check_live(con, repo, NOW)}
    live_results = health["live_results"]
    assert live_results.stale is True
    assert "predictor capture-results" in live_results.advice
    assert "com.predictor.results" in live_results.advice


def test_live_results_not_stale_when_no_results_missing(con, tmp_path):
    repo = _repo_with_remote(tmp_path / "repo")
    insert_schedule_row(con, "0022400002", NOW.date(), "PHI", "NYK",
                         NOW - timedelta(hours=20), season=SEASON)
    _insert_final(con, "0022400002", NOW.date(), NOW - timedelta(hours=19))
    health = {h.name: h for h in status.check_live(con, repo, NOW)}
    assert health["live_results"].stale is False


def test_live_results_latest_is_last_capture_not_a_staleness_judgement(con, tmp_path):
    repo = _repo_with_remote(tmp_path / "repo")
    # A genuine live capture, long ago -- must be reported as `latest`
    # regardless of how old it is, since staleness here is driven entirely
    # by `results_missing`, not by the age of the last capture.
    old_capture = NOW - timedelta(days=10)
    _insert_final(con, "0022400003", NOW.date() - timedelta(days=10), old_capture,
                  reconstructed=False)
    # A separate, currently-missing result makes this stale.
    insert_schedule_row(con, "0022400004", NOW.date(), "PHI", "NYK",
                         NOW - timedelta(hours=20), season=SEASON)
    health = {h.name: h for h in status.check_live(con, repo, NOW)}
    live_results = health["live_results"]
    assert live_results.stale is True
    assert live_results.latest == old_capture


def test_prediction_log_stale_when_games_yesterday_and_no_recent_line(con, tmp_path):
    repo = _repo_with_remote(tmp_path / "repo")
    # Keeps `in_season` true and makes "games today or yesterday" true.
    insert_schedule_row(con, "0022400010", (NOW - timedelta(days=1)).date(), "PHI", "NYK",
                         NOW - timedelta(days=1), season=SEASON)
    from predictor.model.live import log_path

    path = log_path(repo, SEASON)
    path.parent.mkdir(parents=True, exist_ok=True)
    old_line = {
        "predicted_at": (NOW - timedelta(hours=40)).isoformat(),
        "game_id": "0022400010",
        "game_date": (NOW - timedelta(days=1)).date().isoformat(),
    }
    import json

    path.write_text(json.dumps(old_line, sort_keys=True) + "\n", encoding="utf-8")

    health = {h.name: h for h in status.check_live(con, repo, NOW)}
    prediction_log = health["prediction_log"]
    assert prediction_log.stale is True
    assert "predictor predict-today" in prediction_log.advice
    assert "com.predictor.predict" in prediction_log.advice


def test_prediction_log_not_stale_with_a_recent_line(con, tmp_path):
    repo = _repo_with_remote(tmp_path / "repo")
    insert_schedule_row(con, "0022400011", NOW.date(), "PHI", "NYK",
                         NOW + timedelta(hours=2), season=SEASON)
    from predictor.model.live import log_path

    path = log_path(repo, SEASON)
    path.parent.mkdir(parents=True, exist_ok=True)
    recent_line = {
        "predicted_at": (NOW - timedelta(hours=1)).isoformat(),
        "game_id": "0022400011",
        "game_date": NOW.date().isoformat(),
    }
    import json

    path.write_text(json.dumps(recent_line, sort_keys=True) + "\n", encoding="utf-8")

    health = {h.name: h for h in status.check_live(con, repo, NOW)}
    assert health["prediction_log"].stale is False


def _commit_prediction(repo, n):
    # Not the season log itself (prediction_log would parse it) -- any file
    # under predictions/ makes this a prediction commit.
    path = repo / "predictions" / "scratch.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f'{{"n": {n}}}\n')
    assert _git(repo, "add", "predictions/scratch.jsonl").returncode == 0
    assert _git(repo, "commit", "-m", f"predictions: day {n}").returncode == 0


def test_log_published_stale_with_unpushed_prediction_commits(con, tmp_path):
    repo = _repo_with_remote(tmp_path / "repo")
    _commit_prediction(repo, 1)
    health = {h.name: h for h in status.check_live(con, repo, NOW)}
    log_published = health["log_published"]
    assert log_published.stale is True
    assert "git push origin main" in log_published.advice
    assert "gh auth status" in log_published.advice


def test_log_published_warns_that_foreign_unpushed_commits_would_go_public(con, tmp_path):
    repo = _repo_with_remote(tmp_path / "repo", ahead_commits=2)
    _commit_prediction(repo, 1)
    health = {h.name: h for h in status.check_live(con, repo, NOW)}
    log_published = health["log_published"]
    assert log_published.stale is True
    assert log_published.advice == (
        "2 of the unpushed commits are not prediction commits — pushing would "
        "make them PUBLIC; review them before running git push origin main"
    )


def test_log_published_stale_with_no_remote(con, tmp_path):
    repo = _repo_without_remote(tmp_path / "repo")
    health = {h.name: h for h in status.check_live(con, repo, NOW)}
    log_published = health["log_published"]
    assert log_published.stale is True
    assert "no GitHub remote configured" in log_published.advice


def test_log_published_not_stale_when_fully_pushed(con, tmp_path):
    repo = _repo_with_remote(tmp_path / "repo", ahead_commits=0)
    health = {h.name: h for h in status.check_live(con, repo, NOW)}
    assert health["log_published"].stale is False


# --- format_report's log_published detail line (fix round 1) --------------
#
# `log_published` has no natural timestamp (latest is always None by
# construction), so format_report's generic "latest is None" branch used to
# swallow it into a blanket "no data at all" -- even while commits were
# genuinely waiting to be pushed, directly contradicting the advice line
# printed right under it. These three tests pin the exact wording for each
# of the three `log_published` states, via the real `format_report(check_live(...))`
# path (not the dataclass fields directly), since the defect was specifically
# in the report's rendering, not in `check_live`'s data.


def test_report_detail_says_how_many_commits_are_waiting(con, tmp_path):
    repo = _repo_with_remote(tmp_path / "repo", ahead_commits=3)
    text = status.format_report(status.check_live(con, repo, NOW))
    assert "log_published: 3 commit(s) waiting to be pushed" in text
    assert "no data at all" not in text


def test_report_detail_says_nothing_waiting_when_fully_pushed(con, tmp_path):
    repo = _repo_with_remote(tmp_path / "repo", ahead_commits=0)
    text = status.format_report(status.check_live(con, repo, NOW))
    assert "log_published: nothing waiting to be pushed" in text


def test_report_detail_says_no_remote_when_unpushed_commits_is_none(con, tmp_path):
    repo = _repo_without_remote(tmp_path / "repo")
    text = status.format_report(status.check_live(con, repo, NOW))
    assert (
        "log_published: no GitHub remote configured (or git could not be read)"
        in text
    )
    assert "no data at all" not in text


def test_check_live_returns_all_four_names(con, tmp_path):
    repo = _repo_with_remote(tmp_path / "repo")
    names = {h.name for h in status.check_live(con, repo, NOW)}
    assert names == {"live_results", "prediction_log", "prediction_files", "log_published"}


# --- prediction_files: never OK while predictions are stranded -------------


def test_prediction_files_ok_on_a_clean_main(con, tmp_path):
    repo = _repo_with_remote(tmp_path / "repo")
    health = {h.name: h for h in status.check_live(con, repo, NOW)}
    assert health["prediction_files"].stale is False
    text = status.format_report(status.check_live(con, repo, NOW))
    assert "prediction_files: committed on main" in text


def test_prediction_files_stale_off_main(con, tmp_path):
    repo = _repo_with_remote(tmp_path / "repo")
    assert _git(repo, "checkout", "-b", "experiment").returncode == 0
    health = {h.name: h for h in status.check_live(con, repo, NOW)}
    files = health["prediction_files"]
    assert files.stale is True
    assert files.advice == (
        "the checkout is on 'experiment' — predictions written here are not "
        "being published; switch back to main"
    )


def test_prediction_files_stale_with_uncommitted_predictions(con, tmp_path):
    repo = _repo_with_remote(tmp_path / "repo")
    path = repo / "predictions" / "scratch.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"a": 1}\n', encoding="utf-8")
    health = {h.name: h for h in status.check_live(con, repo, NOW)}
    files = health["prediction_files"]
    assert files.stale is True
    assert files.advice == (
        "prediction files have uncommitted changes — the last publish failed; "
        "run predictor predict-today again or check the message in "
        "data/logs/predict.err.log"
    )
    # Everything else is fine, yet the report must not say ALL OK.
    text = status.format_report(status.check_live(con, repo, NOW))
    assert not text.startswith("ALL OK")


def test_prediction_files_stale_mid_merge(con, tmp_path):
    repo = _repo_with_remote(tmp_path / "repo")
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / ".git" / "MERGE_HEAD").write_text(head + "\n", encoding="utf-8")
    files = {h.name: h for h in status.check_live(con, repo, NOW)}["prediction_files"]
    assert files.stale is True
    assert "MERGE_HEAD" in files.advice


# --- minors (final fix wave) -------------------------------------------------


def test_corrupt_prediction_log_is_reported_stale_not_a_crash(con, tmp_path):
    repo = _repo_with_remote(tmp_path / "repo")
    from predictor.model.live import log_path

    path = log_path(repo, SEASON)
    path.parent.mkdir(parents=True)
    path.write_text('{"predicted_at": "2025-01-15T10:00:00+00:00"}\n{not json\n', encoding="utf-8")

    prediction_log = {h.name: h for h in status.check_live(con, repo, NOW)}["prediction_log"]

    assert prediction_log.stale is True
    assert "corrupt line" in prediction_log.advice
    assert str(path) in prediction_log.advice


def test_prediction_log_line_missing_predicted_at_is_reported_stale(con, tmp_path):
    repo = _repo_with_remote(tmp_path / "repo")
    from predictor.model.live import log_path

    path = log_path(repo, SEASON)
    path.parent.mkdir(parents=True)
    path.write_text('{"game_id": "x"}\n', encoding="utf-8")

    prediction_log = {h.name: h for h in status.check_live(con, repo, NOW)}["prediction_log"]

    assert prediction_log.stale is True
    assert str(path) in prediction_log.advice


def test_live_results_detail_counts_games_not_rows(con, tmp_path):
    repo = _repo_with_remote(tmp_path / "repo")
    _insert_final(con, "0022400030", NOW.date(), NOW - timedelta(hours=5))
    insert_schedule_row(con, "0022400031", NOW.date(), "PHI", "NYK",
                         NOW - timedelta(hours=20), season=SEASON)
    text = status.format_report(status.check_live(con, repo, NOW))
    assert "live_results: 1 game(s) missing results, last capture 5h ago" in text


def test_live_results_detail_with_nothing_missing(con, tmp_path):
    repo = _repo_with_remote(tmp_path / "repo")
    _insert_final(con, "0022400032", NOW.date(), NOW - timedelta(hours=3))
    text = status.format_report(status.check_live(con, repo, NOW))
    assert "live_results: no games missing results, last capture 3h ago" in text


def test_live_results_detail_before_any_capture(con, tmp_path):
    repo = _repo_with_remote(tmp_path / "repo")
    text = status.format_report(status.check_live(con, repo, NOW))
    assert "live_results: no live results captured yet" in text
