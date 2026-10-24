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
    assert names == {
        "games", "injury_status", "odds_snapshots", "news_items", "schedule", "player_games",
        "injury_reports", "player_boxscores",
    }


def test_report_names_every_source(con):
    _add_injury(con, NOW - timedelta(hours=1))
    text = status.format_report(status.check_sources(con, NOW))
    for name in ("games", "injury_status", "odds_snapshots", "news_items", "schedule",
                 "player_games", "injury_reports", "player_boxscores"):
        assert name in text


def test_status_lists_player_games_with_ingest_advice(con):
    from predictor.config import season_label

    assert status.STALENESS_HOURS["player_games"] == 24 * 150
    health = {h.name: h for h in status.check_sources(con, NOW)}
    entry = health["player_games"]
    assert entry.stale is True
    assert entry.advice == f"Run: predictor ingest-player-logs --season {season_label(NOW)}"


def test_player_games_fresh_within_150_days_is_not_stale(con):
    con.execute(
        f"INSERT INTO {db.POINT_IN_TIME_TABLES['player_games']} (game_id, season, game_date,"
        " season_type, player_id, player_name, team, source, reconstructed, observed_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?)",
        ["0022400001", SEASON, NOW.date(), "regular", 1, "x", "LAL", "nba_stats", True,
         NOW - timedelta(days=149)],
    )
    health = {h.name: h for h in status.check_sources(con, NOW)}
    assert health["player_games"].stale is False


def _add_odds(con, observed_at, source="theoddsapi", book="fanduel"):
    con.execute(
        f"INSERT INTO {db.POINT_IN_TIME_TABLES['odds_snapshots']} (game_key, book,"
        " home_team, away_team, home_price, away_price, spread, total, observed_at,"
        " game_id, source, reconstructed) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [f"ev-{observed_at.isoformat()}", book, "PHI", "NYK", -150, 130, -3.5, 220.0,
         observed_at, None, source, source == "kaggle_sbr"],
    )


def _in_season(con):
    insert_schedule_row(con, "0022400900", NOW.date(), "PHI", "NYK",
                         NOW + timedelta(hours=2), season=SEASON)


def test_odds_advice_names_the_key_file_and_the_job_not_an_env_export(con, tmp_path,
                                                                       monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _in_season(con)
    repo = _repo_with_remote(tmp_path / "repo")
    health = {h.name: h for h in status.check_live(con, repo, NOW)}
    odds = health["odds"]
    assert odds.stale is True
    assert str(tmp_path / "home" / ".config" / "predictor" / "odds_api_key") in odds.advice
    assert "com.predictor.odds" in odds.advice
    assert "export ODDS_API_KEY" not in odds.advice
    assert "predictor ingest-odds" in odds.advice


def test_odds_table_entry_does_not_report_odds_a_second_time(con):
    # Live odds freshness is judged once, by check_live's `odds` entry; the
    # table entry only reports what is stored.
    health = {h.name: h for h in status.check_sources(con, NOW)}
    assert health["odds_snapshots"].stale is False
    assert health["odds_snapshots"].advice == ""
    _add_odds(con, NOW - timedelta(days=400), source="kaggle_sbr", book="consensus")
    health = {h.name: h for h in status.check_sources(con, NOW)}
    assert health["odds_snapshots"].stale is False
    assert health["odds_snapshots"].row_count == 1
    assert "odds_snapshots: 1 rows" in status.format_report(list(health.values()))


def test_live_odds_stale_in_season_without_a_recent_live_row(con, tmp_path):
    _in_season(con)
    repo = _repo_with_remote(tmp_path / "repo")
    _add_odds(con, NOW - timedelta(hours=31))
    # Historical rows, however recent, are not live odds.
    _add_odds(con, NOW - timedelta(hours=1), source="kaggle_sbr", book="consensus")
    odds = {h.name: h for h in status.check_live(con, repo, NOW)}["odds"]
    assert odds.stale is True
    assert odds.latest == NOW - timedelta(hours=31)
    assert odds.advice


def test_live_odds_fresh_within_30_hours(con, tmp_path):
    _in_season(con)
    repo = _repo_with_remote(tmp_path / "repo")
    _add_odds(con, NOW - timedelta(hours=29))
    odds = {h.name: h for h in status.check_live(con, repo, NOW)}["odds"]
    assert odds.stale is False
    assert odds.advice == ""
    text = status.format_report([odds])
    assert "odds: " in text and "STALE" not in text


def test_live_odds_quiet_off_season(con, tmp_path):
    repo = _repo_with_remote(tmp_path / "repo")
    _add_odds(con, NOW - timedelta(days=90))
    odds = {h.name: h for h in status.check_live(con, repo, NOW)}["odds"]
    assert odds.stale is False
    assert odds.advice == ""


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
    assert health["odds"].stale is False


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
    assert names == {"live_results", "prediction_log", "prediction_files", "log_published",
                     "odds"}


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


def test_new_sources_have_staleness_thresholds():
    assert status.STALENESS_HOURS["injury_reports"] == 36
    assert status.STALENESS_HOURS["player_boxscores"] == 24 * 150


def test_injury_reports_fresh_within_36_hours_is_not_stale(con):
    seen = NOW - timedelta(hours=30)
    con.execute(
        f"INSERT INTO {db.POINT_IN_TIME_TABLES['injury_reports']} (observed_at, fetched_at,"
        " source_key, row_count, content_sha) VALUES (?,?,?,?,?)",
        [seen, seen, "k", 1, "b" * 64],
    )
    health = {h.name: h for h in status.check_sources(con, NOW)}
    assert health["injury_reports"].stale is False
    assert health["player_boxscores"].stale is True


# --- Task 11 (cloud operation): status on cloud state ----------------------

import json  # noqa: E402

from predictor.cloud import state as cloud_state  # noqa: E402

CLOUD_NOW = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)


def _run(created_at, status_="completed", conclusion="success"):
    return {"createdAt": created_at, "status": status_, "conclusion": conclusion}


class _FakeGh:
    """A `subprocess.run` stand-in for `gh run list`: records every call
    and returns the given runs as JSON (or raises / fails as told)."""

    def __init__(self, runs=None, *, raises=None, returncode=0, stderr="", stdout=None):
        self.runs = runs or []
        self.raises = raises
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = stdout
        self.calls = []

    def __call__(self, args, **kwargs):
        self.calls.append((args, kwargs))
        if self.raises is not None:
            raise self.raises
        out = self.stdout if self.stdout is not None else json.dumps(self.runs)
        return subprocess.CompletedProcess(args, self.returncode, out, self.stderr)


def test_cloud_stale_after_two_hours():
    gh = _FakeGh([_run("2026-10-22T10:00:00Z")])
    fresh = status.check_cloud(CLOUD_NOW, runner=gh)
    assert fresh.name == "cloud_runs"
    assert fresh.stale is False
    assert fresh.advice == ""
    late = status.check_cloud(CLOUD_NOW + timedelta(seconds=1), runner=gh)
    assert late.stale is True
    assert "live.yml" in late.advice
    assert status.CLOUD_STALE == timedelta(hours=2)


def test_cloud_runs_gh_without_prompts_and_with_a_timeout():
    gh = _FakeGh([_run("2026-10-22T11:23:00Z")])
    status.check_cloud(CLOUD_NOW, runner=gh)
    (args, kwargs), = gh.calls
    assert args == ["gh", "run", "list", "--workflow", "live.yml", "--limit", "5",
                    "--json", "createdAt,status,conclusion"]
    assert kwargs["timeout"] == 20
    assert kwargs["env"]["GH_PROMPT_DISABLED"] == "1"
    assert kwargs["capture_output"] is True
    assert kwargs.get("stdin") == subprocess.DEVNULL


def test_cloud_detail_reads_minutes_ago():
    gh = _FakeGh([_run("2026-10-22T11:23:00Z"), _run("2026-10-22T11:08:00Z")])
    health = status.check_cloud(CLOUD_NOW, runner=gh)
    assert health.detail == "last cloud run 37m ago (success)"
    text = status.format_report([health])
    assert "[OK   ] cloud_runs: last cloud run 37m ago (success)" in text


def test_cloud_detail_for_a_run_in_progress():
    gh = _FakeGh([_run("2026-10-22T11:53:00Z", "in_progress", ""),
                  _run("2026-10-22T11:38:00Z")])
    health = status.check_cloud(CLOUD_NOW, runner=gh)
    assert health.stale is False
    assert health.detail == "last cloud run 7m ago (in_progress)"


def test_cloud_three_failures_is_stale():
    gh = _FakeGh([
        _run("2026-10-22T11:53:00Z", "in_progress", ""),
        _run("2026-10-22T11:38:00Z", conclusion="failure"),
        _run("2026-10-22T11:23:00Z", conclusion="failure"),
        _run("2026-10-22T11:08:00Z", conclusion="failure"),
        _run("2026-10-22T10:53:00Z"),
    ])
    health = status.check_cloud(CLOUD_NOW, runner=gh)
    assert health.stale is True
    assert "3" in health.detail and "fail" in health.detail
    assert "live.yml" in health.advice


def test_cloud_two_failures_is_not_stale():
    gh = _FakeGh([
        _run("2026-10-22T11:38:00Z", conclusion="failure"),
        _run("2026-10-22T11:23:00Z", conclusion="failure"),
        _run("2026-10-22T11:08:00Z"),
    ])
    assert status.check_cloud(CLOUD_NOW, runner=gh).stale is False


def test_cloud_no_runs_is_stale():
    health = status.check_cloud(CLOUD_NOW, runner=_FakeGh([]))
    assert health.stale is True
    assert "no cloud run" in health.detail


def test_cloud_gh_missing_is_reported():
    gh = _FakeGh(raises=FileNotFoundError("gh"))
    health = status.check_cloud(CLOUD_NOW, runner=gh)
    assert health.stale is True
    assert "gh auth status" in health.advice
    assert health.detail.startswith("could not check")


def test_cloud_gh_timeout_is_a_plain_could_not_check():
    gh = _FakeGh(raises=subprocess.TimeoutExpired(["gh"], 20))
    health = status.check_cloud(CLOUD_NOW, runner=gh)
    assert health.stale is True
    assert health.detail.startswith("could not check")
    assert "gh auth status" in health.advice


def test_cloud_gh_error_never_leaks_a_token():
    token = "ghs_" + "A1b2C3d4" * 5
    gh = _FakeGh(returncode=1, stderr=(
        f"error connecting to https://x-access-token:{token}@github.com/o/r "
        f"(token {token}; github_pat_{'Z9' * 20})"
    ))
    health = status.check_cloud(CLOUD_NOW, runner=gh)
    text = status.format_report([health])
    assert health.stale is True
    assert health.detail.startswith("could not check")
    assert token not in text
    assert "github_pat_Z9" not in text
    assert "gh auth status" in health.advice


def test_cloud_unreadable_gh_output_is_a_plain_could_not_check():
    health = status.check_cloud(CLOUD_NOW, runner=_FakeGh(stdout="not json"))
    assert health.stale is True
    assert health.detail.startswith("could not check")


# --- check_live on cloud state ----------------------------------------------


def test_check_live_cloud_advice_names_the_workflow_and_the_secret(con, tmp_path):
    repo = _repo_with_remote(tmp_path / "repo")
    assert _git(repo, "checkout", "-b", "cloud-shadow").returncode == 0
    assert _git(repo, "push", "origin", "cloud-shadow").returncode == 0
    _in_season(con)
    health = {h.name: h for h in status.check_live(con, repo, NOW, branch="cloud-shadow",
                                                   cloud=True)}
    assert health["prediction_files"].stale is False
    assert health["log_published"].stale is False
    odds = health["odds"]
    assert odds.stale is True
    assert "live.yml" in odds.advice
    assert "gh secret set ODDS_API_KEY" in odds.advice
    for h in health.values():
        assert "launchd" not in h.advice and "com.predictor" not in h.advice
        assert "odds_api_key" not in h.advice
    text = status.format_report(list(health.values()))
    assert "prediction_files: committed on cloud-shadow" in text


def test_check_live_cloud_results_and_log_advice_name_the_run_summary(con, tmp_path):
    repo = _repo_with_remote(tmp_path / "repo")
    insert_schedule_row(con, "0022400010", (NOW - timedelta(days=1)).date(), "PHI", "NYK",
                        NOW - timedelta(hours=20), season=SEASON)
    health = {h.name: h for h in status.check_live(con, repo, NOW, cloud=True)}
    for name in ("live_results", "prediction_log"):
        assert health[name].stale is True
        assert "live.yml" in health[name].advice
        assert "com.predictor" not in health[name].advice
        assert "launchd" not in health[name].advice


def test_log_published_advice_no_longer_says_pull_by_hand(con, tmp_path):
    repo = _repo_with_remote(tmp_path / "repo")
    _commit_prediction(repo, 1)
    advice = {h.name: h for h in status.check_live(con, repo, NOW)}["log_published"].advice
    assert "git pull" not in advice
    assert "automatically" in advice
    assert "git push origin main" in advice


def test_log_published_cloud_advice_says_the_next_run_redoes_the_work(con, tmp_path):
    repo = _repo_with_remote(tmp_path / "repo")
    _commit_prediction(repo, 1)
    advice = {h.name: h for h in status.check_live(con, repo, NOW, cloud=True)}[
        "log_published"].advice
    assert "git pull" not in advice
    assert "next" in advice and "live.yml" in advice


# --- advice for the new sources ----------------------------------------------


def test_new_sources_have_run_advice(con):
    health = {h.name: h for h in status.check_sources(con, NOW)}
    assert health["injury_reports"].stale is True
    assert health["injury_reports"].advice.startswith("Run: ")
    assert health["player_boxscores"].stale is True
    assert health["player_boxscores"].advice.startswith("Run: ")


# --- check_sources on cloud state ---------------------------------------------


def _far_fetch(state_dir, at):
    path = state_dir / "schedule" / f"far_fetches-{at:%Y-%m}.jsonl"
    cloud_state.append_records(path, [{"fetched_at": at, "days": 7}])


def test_cloud_schedule_freshness_uses_the_fetch_record_not_observed_at(con, tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    # The last schedule CHANGE was 5 days ago -- nothing moved since.
    insert_schedule_row(con, "0022400561", NOW.date(), "PHI", "NYK",
                        NOW + timedelta(hours=3), observed_at=NOW - timedelta(days=5))
    _far_fetch(state_dir, NOW - timedelta(hours=2))
    mac = {h.name: h for h in status.check_sources(con, NOW)}
    assert mac["schedule"].stale is True  # the Mac's own check is unchanged
    cloud = {h.name: h for h in status.check_sources(con, NOW, cloud_state=state_dir)}
    assert cloud["schedule"].stale is False
    assert cloud["schedule"].latest == NOW - timedelta(hours=2)
    assert "schedule: 1 change rows; ESPN scoreboard last fetched 2h ago" in status.format_report(
        list(cloud.values()))


def test_cloud_schedule_stale_when_no_fetch_is_recorded(con, tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    insert_schedule_row(con, "0022400561", NOW.date(), "PHI", "NYK",
                        NOW + timedelta(hours=3), observed_at=NOW - timedelta(days=5))
    _far_fetch(state_dir, NOW - timedelta(hours=40))
    cloud = {h.name: h for h in status.check_sources(con, NOW, cloud_state=state_dir)}
    assert cloud["schedule"].stale is True
    assert "live.yml" in cloud["schedule"].advice
    assert "launchctl" not in cloud["schedule"].advice


def test_cloud_injury_status_uses_the_latest_report_time(con, tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _add_injury(con, NOW - timedelta(days=4))  # last change
    seen = NOW - timedelta(hours=1)  # an unchanged report since
    con.execute(
        f"INSERT INTO {db.POINT_IN_TIME_TABLES['injury_reports']} (observed_at, fetched_at,"
        " source_key, row_count, content_sha) VALUES (?,?,?,?,?)",
        [seen, seen, "k", 1, "b" * 64],
    )
    assert {h.name: h for h in status.check_sources(con, NOW)}["injury_status"].stale is True
    cloud = {h.name: h for h in status.check_sources(con, NOW, cloud_state=state_dir)}
    assert cloud["injury_status"].stale is False


def test_cloud_sources_skip_news(con, tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    cloud = {h.name: h for h in status.check_sources(con, NOW, cloud_state=state_dir)}
    news = cloud["news_items"]
    assert news.stale is False
    assert news.detail == "news is archived by the Mac (research only)"


# --- player_links --------------------------------------------------------------


def _box(con, player_id, name, minutes, game_date, game_id="g1"):
    con.execute(
        f"INSERT INTO {db.POINT_IN_TIME_TABLES['player_boxscores']} (game_id, season,"
        " game_date, team, player_id, espn_player_id, player_name, did_not_play, minutes,"
        " source, observed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        [game_id, "2026-27", game_date, "PHI", player_id, player_id.removeprefix("espn:"),
         name, minutes == 0, minutes, "espn", CLOUD_NOW],
    )


def test_player_links_unmatched_warns(tmp_path):
    c = db.connect(tmp_path / "p.duckdb")
    db.migrate(c)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    recent = (CLOUD_NOW - timedelta(days=2)).date()
    _box(c, "1627732", "Linked Guy", 30.0, recent)
    _box(c, "espn:999", "Benchwarmer Zero", 0.0, recent)
    _box(c, "espn:998", "Old Timer", 25.0, (CLOUD_NOW - timedelta(days=9)).date())
    for i in range(6):
        _box(c, f"espn:90{i}", f"New Guy {i}", 5.0 + i, recent)
    health = status.check_player_links(state_dir, c, CLOUD_NOW)
    assert health.name == "player_links"
    assert health.stale is True
    assert "add them to state/players/aliases.json" in health.detail
    named = [f"New Guy {i}" for i in range(6) if f"New Guy {i}" in health.detail]
    assert len(named) == 5
    assert "Benchwarmer Zero" not in health.detail
    assert "Old Timer" not in health.detail
    assert "Linked Guy" not in health.detail


def test_player_links_ok_when_every_recent_player_is_linked(tmp_path):
    c = db.connect(tmp_path / "p.duckdb")
    db.migrate(c)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _box(c, "1627732", "Linked Guy", 30.0, (CLOUD_NOW - timedelta(days=1)).date())
    _box(c, "espn:999", "Benchwarmer Zero", 0.0, (CLOUD_NOW - timedelta(days=1)).date())
    health = status.check_player_links(state_dir, c, CLOUD_NOW)
    assert health.stale is False
    assert health.advice == ""
