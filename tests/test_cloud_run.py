"""cloud-run end to end: a bare remote plus a clone on cloud-shadow, a
fixture schedule in state/nba, fake ESPN / injury / odds / news / gh, and a
fixed clock. No network, no real remote, no real gh."""

from __future__ import annotations

import json
import subprocess
import tarfile
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
import requests
from typer.testing import CliRunner

from predictor import cli, config, db, raw_store
from predictor.cloud import run as cloud_run
from predictor.cloud import state, summary
from predictor.cloud.run import STEP_NAMES, Fetchers, run_once
from predictor.config import Settings
from predictor.model.live import read_log
from predictor.sources import news_rss, odds

FIXTURES = Path(__file__).parent / "fixtures" / "espn"
FAKE_KEY = "0123456789abcdef" * 2  # built at runtime; never a real key
SEASON = "2026-27"
NOW = datetime(2026, 10, 22, 12, 37, tzinfo=UTC)


def dt(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return result.stdout


# --- fixture world -------------------------------------------------------------

TEAMS = {
    "BOS": ("BOS", "Boston Celtics"),
    "DET": ("DET", "Detroit Pistons"),
    "NYK": ("NY", "New York Knicks"),
    "LAL": ("LAL", "Los Angeles Lakers"),
    "GSW": ("GS", "Golden State Warriors"),
}
# (game_id, espn_id, game_date, tip, home, away)
GAMES = [
    ("0022600001", "401909088", date(2026, 10, 21), "2026-10-21T23:00Z", "DET", "BOS"),
    ("0022600002", "401909101", date(2026, 10, 22), "2026-10-22T23:00Z", "BOS", "NYK"),
    ("0022600003", "401909102", date(2026, 10, 22), "2026-10-23T02:00Z", "GSW", "LAL"),
]
FINALS = {"401909088": (121, 113)}  # home, away


def _schedule_rows() -> list[dict]:
    return [
        {
            "game_id": gid, "season": SEASON, "game_date": day, "tip_off_utc": dt(tip),
            "home_team": home, "away_team": away, "arena_name": f"{home} Arena",
            "arena_city": "City", "arena_state": "ST", "is_neutral_reported": False,
            "is_neutral": False, "observed_at": dt("2026-10-01T12:00Z"),
        }
        for gid, _e, day, tip, home, away in GAMES
    ]


def _event(espn_id: str, tip: str, home: str, away: str) -> dict:
    final = FINALS.get(espn_id)

    def side(team: str, where: str, score) -> dict:
        abbr, name = TEAMS[team]
        return {"homeAway": where, "team": {"abbreviation": abbr, "displayName": name},
                "score": str(score)}

    return {
        "id": espn_id,
        "date": tip,
        "season": {"year": 2027, "type": 2},
        "status": {"type": {"completed": final is not None,
                            "name": "STATUS_FINAL" if final else "STATUS_SCHEDULED"}},
        "competitions": [{
            "timeValid": True, "neutralSite": False,
            "venue": {"fullName": f"{home} Arena", "address": {"city": "City", "state": "ST"}},
            "competitors": [side(home, "home", final[0] if final else 0),
                            side(away, "away", final[1] if final else 0)],
        }],
    }


def scoreboard(day: date) -> bytes:
    events = [_event(e, tip, home, away) for _g, e, d, tip, home, away in GAMES if d == day]
    return json.dumps({"events": events}).encode()


def summary_body(espn_id: str) -> bytes:
    assert espn_id == "401909088"
    return (FIXTURES / "summary_sample.json").read_bytes()


class _OddsResponse:
    status_code = 200
    headers = {"x-requests-remaining": "480"}
    text = ""

    def __init__(self):
        self.content = json.dumps([
            {
                "id": "oddsevent1", "commence_time": "2026-10-22T23:00:00Z",
                "home_team": "Boston Celtics", "away_team": "New York Knicks",
                "bookmakers": [{"key": "draftkings", "markets": [
                    {"key": "h2h", "outcomes": [{"name": "Boston Celtics", "price": -150},
                                                {"name": "New York Knicks", "price": 130}]},
                    {"key": "spreads", "outcomes": [{"name": "Boston Celtics", "point": -3.5},
                                                    {"name": "New York Knicks", "point": 3.5}]},
                    {"key": "totals", "outcomes": [{"name": "Over", "point": 221.5}]},
                ]}],
            }
        ]).encode()


class _OddsSession:
    def __init__(self):
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        return _OddsResponse()


RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
<item><guid>item-1</guid><title>Celtics win</title><link>https://example.invalid/1</link>
<description>Story</description></item></channel></rss>"""


class _RssResponse:
    status_code = 200
    content = RSS


class _RssSession:
    def get(self, url, timeout=None, headers=None):
        return _RssResponse()


def news(now):
    return {
        "espn": news_rss.poll_feed("espn", "https://example.invalid/espn", now,
                                   session=_RssSession()),
        "cbs": news_rss.FeedResult("cbs", False, 0, 0, 0, None, "HTTP 403"),
    }


class FakeGh:
    def __init__(self):
        self.calls: list[tuple[list[str], dict]] = []
        self.assets: list[Path] = []

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        if argv[:3] == ["gh", "release", "view"]:
            return subprocess.CompletedProcess(argv, 1, "", "release not found")
        if argv[:3] == ["gh", "release", "upload"]:
            self.assets.append(Path(argv[4]))
        return subprocess.CompletedProcess(argv, 0, "", "")


def fetchers(**over) -> Fetchers:
    base = Fetchers(
        scoreboard=scoreboard,
        summary=summary_body,
        injury_latest=lambda now: None,
        odds_download=lambda key, now: odds.download(key, now=now, session=_OddsSession()),
        gh_runner=FakeGh(),
        clock=lambda: NOW,
        news=news,
    )
    return replace(base, **over)


def point_data_dir(path: Path, monkeypatch) -> Settings:
    s = Settings(data_dir=path)
    s.ensure_dirs()
    monkeypatch.setattr(config, "settings", s)
    monkeypatch.setattr(db, "settings", s)
    monkeypatch.setattr(raw_store, "settings", s)
    return s


@pytest.fixture
def world(tmp_path, monkeypatch):
    """remote.git (main + cloud-shadow), seed (the code checkout, on main)
    and work (the state checkout, a clone on cloud-shadow)."""
    for var in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(var, "Predictor Bot")
    for var in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(var, "bot@example.invalid")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))  # never the real key file
    monkeypatch.setenv("ODDS_API_KEY", FAKE_KEY)
    for var in ("GITHUB_STEP_SUMMARY", "GITHUB_OUTPUT", "PREDICTOR_PUBLISH_BRANCH"):
        monkeypatch.delenv(var, raising=False)
    point_data_dir(tmp_path / "data", monkeypatch)

    remote = tmp_path / "remote.git"
    remote.mkdir()
    _git(remote, "init", "--bare", "-b", "main")
    seed = tmp_path / "seed"
    seed.mkdir()
    _git(seed, "init", "-b", "main")
    (seed / "README.md").write_text("seed\n", encoding="utf-8")
    state.write_snapshot(seed / "state", f"nba/schedule-{SEASON}.jsonl", _schedule_rows(),
                         state.SCHEDULE_COLUMNS)
    _git(seed, "add", ".")
    _git(seed, "commit", "-m", "seed")
    _git(seed, "remote", "add", "origin", str(remote))
    _git(seed, "push", "origin", "main")
    _git(seed, "push", "origin", "main:cloud-shadow")  # the first push is by hand
    work = tmp_path / "work"
    _git(tmp_path, "clone", "-q", "-b", "cloud-shadow", str(remote), str(work))
    return {"remote": remote, "seed": seed, "work": work, "tmp": tmp_path}


def subjects(repo: Path, n: int = 10) -> list[str]:
    return _git(repo, "log", f"-{n}", "--format=%s").splitlines()


def files_of(repo: Path, rev: str) -> list[str]:
    return _git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", rev).split()


def step(report, name):
    return next(s for s in report.steps if s.name == name)


# --- tests ----------------------------------------------------------------------


def test_first_run_after_1800_ist_commits_state_then_predictions(world):
    work, remote, seed = world["work"], world["remote"], world["seed"]
    main_before = _git(remote, "rev-parse", "main").strip()

    report = run_once(work, seed, branch="cloud-shadow", fetchers=fetchers())

    assert subjects(work, 2) == ["predictions: 2026-10-22 slate",
                                 "state: cloud run 2026-10-22T12:37Z"]
    assert all(f.startswith("predictions/") for f in files_of(work, "HEAD"))
    assert all(f.startswith("state/") for f in files_of(work, "HEAD~1"))
    assert _git(remote, "log", "-2", "--format=%s", "cloud-shadow").splitlines() == \
        subjects(work, 2)
    assert _git(remote, "rev-parse", "main").strip() == main_before
    assert report.pushed and not report.unexpected and not report.site_changed
    assert all(s.ok for s in report.steps), report.steps
    assert report.state_commit == _git(work, "rev-parse", "--short", "HEAD~1").strip()
    assert report.predictions_commit == _git(work, "rev-parse", "--short", "HEAD").strip()

    lines = read_log(work / "predictions" / f"{SEASON}.jsonl")
    predicted = [ln for ln in lines if ln["status"] == "predicted"]
    assert sorted(ln["game_id"] for ln in predicted) == ["0022600002", "0022600003"]
    # The call was made from the state commit, by the code checkout.
    assert {ln["state_version"] for ln in predicted} == {report.state_commit}
    assert {ln["code_version"] for ln in predicted} == {
        _git(seed, "rev-parse", "--short", "HEAD").strip()}
    # Never sealed at or after tip-off.
    assert all(dt(ln["predicted_at"]) < dt(ln["tip_off_utc"]) for ln in predicted)

    sd = work / "state"
    assert list(sd.glob("results/*.jsonl")) and list(sd.glob("boxscores/*/*.jsonl"))
    assert list(sd.glob("schedule/far_fetches-*.jsonl"))
    attempts = cloud_run.odds_attempts(sd)
    assert [(a["window"], a["ok"], a["remaining"]) for a in attempts] == [("early", True, 480)]
    assert not list(sd.glob("odds/*/*.jsonl"))  # odds are never republished
    assert [r["item_key"] for r in state.read_rows(sd, "news")]
    assert "far days included" in step(report, "espn_scoreboard").detail
    assert "1/1 game(s)" in step(report, "espn_boxscores").detail
    assert any(n.startswith("unmatched player:") for n in report.notes)
    assert any("news feed cbs failed: HTTP 403" in n for n in report.notes)
    assert report.raw_archived


def test_run_before_1800_ist_writes_no_prediction_lines(world):
    early = datetime(2026, 10, 22, 12, 22, tzinfo=UTC)
    report = run_once(world["work"], world["seed"], branch="cloud-shadow",
                      fetchers=fetchers(clock=lambda: early))
    assert step(report, "predict").due is False
    assert not (world["work"] / "predictions" / f"{SEASON}.jsonl").exists()
    assert subjects(world["work"], 1) == ["state: cloud run 2026-10-22T12:22Z"]
    assert report.predictions_commit is None
    assert step(report, "odds").due is False  # the early window opens at 12:30


def test_second_identical_run_commits_nothing(world):
    work = world["work"]
    run_once(work, world["seed"], branch="cloud-shadow", fetchers=fetchers())
    head = _git(work, "rev-parse", "HEAD")
    report = run_once(work, world["seed"], branch="cloud-shadow", fetchers=fetchers())
    assert _git(work, "rev-parse", "HEAD") == head
    assert report.state_commit is None and report.predictions_commit is None
    assert not report.unexpected
    assert step(report, "odds").due is False
    assert step(report, "espn_boxscores").due is False
    assert _git(work, "status", "--porcelain") == ""


def test_rerun_after_a_lost_push_redoes_the_work_without_duplicates(world, monkeypatch):
    work, remote, tmp = world["work"], world["remote"], world["tmp"]

    def network_drops(now):  # after the start-of-run fetch, before the push
        _git(work, "remote", "set-url", "origin", str(tmp / "unreachable.git"))
        return None

    first = run_once(work, world["seed"], branch="cloud-shadow",
                     fetchers=fetchers(injury_latest=network_drops))
    assert first.state_commit and first.predictions_commit and not first.pushed
    assert not first.unexpected and not first.publish_error

    fresh = tmp / "fresh"
    _git(tmp, "clone", "-q", "-b", "cloud-shadow", str(remote), str(fresh))
    point_data_dir(tmp / "data2", monkeypatch)  # a new runner, a new temp dir
    second = run_once(fresh, world["seed"], branch="cloud-shadow", fetchers=fetchers())
    assert second.pushed
    lines = read_log(fresh / "predictions" / f"{SEASON}.jsonl")
    ids = [ln["game_id"] for ln in lines if ln["status"] == "predicted"]
    assert sorted(ids) == ["0022600002", "0022600003"]
    assert _git(remote, "log", "-2", "--format=%s", "cloud-shadow").splitlines() == [
        "predictions: 2026-10-22 slate", "state: cloud run 2026-10-22T12:37Z"]


def _invoke(world, monkeypatch, fx, branch="cloud-shadow"):
    monkeypatch.setenv("PREDICTOR_PUBLISH_BRANCH", branch)
    monkeypatch.setattr(cli, "_cloud_fetchers", lambda: fx)
    monkeypatch.setattr(cli, "_repo_dir", lambda: world["seed"])
    return CliRunner().invoke(cli.app, ["cloud-run", "--state-repo", str(world["work"])])


def test_unreachable_espn_is_recorded_and_the_run_continues(world, monkeypatch):
    calls = []

    def down(day):
        raise requests.ConnectionError("Failed to resolve 'site.api.espn.com'")

    def injury(now):
        calls.append(now)
        return None

    result = _invoke(world, monkeypatch, fetchers(scoreboard=down, injury_latest=injury))
    assert result.exit_code == 0, result.output
    assert "| espn_scoreboard | yes | FAILED | could not reach the source (ConnectionError) |" \
        in result.output
    assert calls == [NOW]  # injury_report ran
    assert "| push | yes | ok |" in result.output


def test_unexpected_error_fails_the_run_after_other_steps(world, monkeypatch):
    def broken(now):
        raise ValueError("boom")

    gh = FakeGh()
    result = _invoke(world, monkeypatch, fetchers(injury_latest=broken, gh_runner=gh))
    assert result.exit_code == 1
    assert "| injury_report | yes | FAILED | ValueError: boom |" in result.output
    assert "| odds | yes | ok |" in result.output
    assert "| push | yes | ok |" in result.output  # push still attempted
    assert _git(world["remote"], "log", "-1", "--format=%s", "cloud-shadow").strip() == \
        "predictions: 2026-10-22 slate"


def test_summary_lists_every_step(world):
    report = run_once(world["work"], world["seed"], branch="cloud-shadow", fetchers=fetchers())
    assert [s.name for s in report.steps] == [
        "mac_history", "rebuild", "espn_scoreboard", "espn_boxscores", "injury_report",
        "odds", "state_commit", "predict", "grade", "raw_archive", "push",
    ] == list(STEP_NAMES)
    text = summary.markdown(report)
    assert text.startswith(
        "## Cloud run 2026-10-22 12:37 UTC (18:07 IST) — branch cloud-shadow\n")
    assert "| step | due | result | detail |" in text
    for name in STEP_NAMES:
        assert f"| {name} |" in text
    assert f"state commit: {report.state_commit}" in text
    assert f"predictions commit: {report.predictions_commit}" in text
    assert "push: pushed" in text


def test_summary_heading_format():
    report = cloud_run.RunReport(
        started_at=datetime(2026, 10, 22, 13, 7, tzinfo=UTC),
        finished_at=datetime(2026, 10, 22, 13, 8, tzinfo=UTC), branch="cloud-shadow",
        steps=[cloud_run.StepOutcome("push", True, True, "a | b")], state_commit=None,
        predictions_commit=None, pushed=False, site_changed=False, raw_archived=True,
        unexpected=False,
    )
    text = summary.markdown(report)
    assert text.splitlines()[0] == \
        "## Cloud run 2026-10-22 13:07 UTC (18:37 IST) — branch cloud-shadow"
    assert "| push | yes | ok | a \\| b |" in text
    assert summary.outputs(report) == {"site_changed": "false", "raw_upload_failed": "false",
                                       "secret_leak": "false"}


def test_odds_key_never_written_anywhere(world, monkeypatch, capsys):
    out_file = world["tmp"] / "gh_output"
    summary_file = world["tmp"] / "step_summary"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out_file))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))
    gh = FakeGh()
    seen_keys = []

    def download(key, now):
        seen_keys.append(key)
        return odds.download(key, now=now, session=_OddsSession())

    result = _invoke(world, monkeypatch, fetchers(gh_runner=gh, odds_download=download))
    assert result.exit_code == 0, result.output
    assert seen_keys == [FAKE_KEY]  # the key was used...
    key = FAKE_KEY.lower()
    work = world["work"]
    for path in (work / "state").rglob("*"):  # ...and written nowhere
        if path.is_file():
            assert key not in path.read_text(encoding="utf-8").lower(), path
    assert key not in result.output.lower()
    assert key not in summary_file.read_text().lower()
    assert out_file.read_text() == \
        "site_changed=false\nraw_upload_failed=false\nsecret_leak=false\n"
    captured = capsys.readouterr()
    assert key not in (captured.out + captured.err).lower()
    assert gh.assets, "the raw archive was uploaded"
    with tarfile.open(gh.assets[0]) as tar:
        names = tar.getnames()
        assert not any(n.startswith("raw/odds") for n in names)  # odds never archived
        for member in tar.getmembers():
            if member.isfile():
                assert key.encode() not in tar.extractfile(member).read().lower()
            assert key not in member.name.lower()
    log_all = _git(work, "log", "-p", "--all")
    assert key not in log_all.lower()
    # gh ran in the state checkout, so it finds the repository from its remote.
    assert {kw.get("cwd") for _argv, kw in gh.calls} == {work}


def test_site_changed_only_for_main_pushes(world, monkeypatch):
    shadow = run_once(world["work"], world["seed"], branch="cloud-shadow", fetchers=fetchers())
    assert shadow.pushed and shadow.predictions_commit and not shadow.site_changed
    main = world["tmp"] / "main"
    _git(world["tmp"], "clone", "-q", "-b", "main", str(world["remote"]), str(main))
    point_data_dir(world["tmp"] / "data-main", monkeypatch)
    report = run_once(main, main, branch="main", fetchers=fetchers())
    assert report.pushed and report.predictions_commit and report.site_changed
    assert summary.outputs(report)["site_changed"] == "true"
    # A run on main whose only push is state does not change the site.
    point_data_dir(world["tmp"] / "data-main2", monkeypatch)
    later = datetime(2026, 10, 22, 14, 0, tzinfo=UTC)
    again = run_once(main, main, branch="main", fetchers=fetchers(clock=lambda: later))
    assert again.predictions_commit is None and not again.site_changed


def test_wrong_branch_checkout_runs_nothing_and_is_reported(world):
    work = world["work"]
    head = _git(work, "rev-parse", "HEAD")
    report = run_once(work, world["seed"], branch="main", fetchers=fetchers())
    assert report.state_commit is None and not report.pushed and report.publish_error
    assert step(report, "rebuild").detail == \
        "not run: the checkout is on cloud-shadow, not main"
    assert all(not s.due for s in report.steps if s.name not in ("rebuild",))
    assert _git(work, "rev-parse", "HEAD") == head
    assert _git(work, "status", "--porcelain") == ""


def test_secret_in_a_state_file_stops_the_run_before_any_commit(world, monkeypatch):
    work = world["work"]

    def leaky(now):
        (work / "state" / "odds").mkdir(parents=True, exist_ok=True)
        (work / "state" / "odds" / "oops.jsonl").write_text(f"{FAKE_KEY}\n")
        return None

    head = _git(work, "rev-parse", "HEAD")
    gh = FakeGh()
    report = run_once(work, world["seed"], branch="cloud-shadow",
                      fetchers=fetchers(injury_latest=leaky, gh_runner=gh))
    assert report.unexpected
    assert _git(work, "rev-parse", "HEAD") == head
    assert step(report, "state_commit").detail.startswith("stopped:")
    assert FAKE_KEY not in step(report, "state_commit").detail
    assert step(report, "push").due is False and not report.pushed
    assert gh.calls == []
    assert summary.outputs(report)["raw_upload_failed"] == "false"


def test_far_days_only_hourly_and_only_after_a_successful_far_fetch(world):
    work = world["work"]
    asked: list[date] = []

    def board(day):
        asked.append(day)
        if day == date(2026, 10, 29):
            raise requests.Timeout("slow")
        return scoreboard(day)

    report = run_once(work, world["seed"], branch="cloud-shadow",
                      fetchers=fetchers(scoreboard=board))
    assert len(asked) == 10 and step(report, "espn_scoreboard").ok is False
    assert "2026-10-29" in step(report, "espn_scoreboard").detail
    assert not list((work / "state").glob("schedule/far_fetches-*.jsonl"))
    assert step(report, "predict").ok  # the run went on

    asked.clear()
    later = datetime(2026, 10, 22, 12, 52, tzinfo=UTC)
    run_once(work, world["seed"], branch="cloud-shadow",
             fetchers=fetchers(clock=lambda: later))
    assert len(asked) == 0  # (the default fetcher does not record)
    assert list((work / "state").glob("schedule/far_fetches-*.jsonl"))
    asked.clear()
    soon = datetime(2026, 10, 22, 13, 7, tzinfo=UTC)
    run_once(work, world["seed"], branch="cloud-shadow",
             fetchers=fetchers(clock=lambda: soon, scoreboard=board))
    assert len(asked) == 3  # near days only: the far fetch at 12:52 is < 60 min old


def test_odds_attempt_recorded_when_the_fetch_fails(world):
    def refused(key, now):
        raise odds.OddsQuotaExceeded(401, "quota", "0")

    report = run_once(world["work"], world["seed"], branch="cloud-shadow",
                      fetchers=fetchers(odds_download=refused))
    assert step(report, "odds").ok is False and not report.unexpected
    assert "HTTP 401" in step(report, "odds").detail
    attempts = cloud_run.odds_attempts(world["work"] / "state")
    assert [(a["ok"], a["remaining"]) for a in attempts] == [(False, 0)]


def test_news_is_polled_hourly(world):
    polls = []

    def counting(now):
        polls.append(now)
        return news(now)

    work = world["work"]
    run_once(work, world["seed"], branch="cloud-shadow", fetchers=fetchers(news=counting))
    later = datetime(2026, 10, 22, 13, 7, tzinfo=UTC)
    run_once(work, world["seed"], branch="cloud-shadow",
             fetchers=fetchers(news=counting, clock=lambda: later))
    hour = datetime(2026, 10, 22, 13, 37, tzinfo=UTC)
    run_once(work, world["seed"], branch="cloud-shadow",
             fetchers=fetchers(news=counting, clock=lambda: hour))
    assert polls == [NOW, hour]
    keys = [r["item_key"] for r in state.read_rows(work / "state", "news")]
    assert len(keys) == len(set(keys)) == 1  # the same item is stored once


def test_grade_lines_take_code_version_from_the_code_checkout(world):
    work, seed = world["work"], world["seed"]
    run_once(work, seed, branch="cloud-shadow", fetchers=fetchers())
    # Next day: the 10-22 games are final on ESPN.
    FINALS.update({"401909101": (110, 100), "401909102": (99, 101)})
    try:
        nxt = datetime(2026, 10, 23, 13, 0, tzinfo=UTC)
        report = run_once(work, seed, branch="cloud-shadow",
                          fetchers=fetchers(clock=lambda: nxt,
                                            summary=lambda e: b'{"boxscore": {}}'))
    finally:
        FINALS.pop("401909101")
        FINALS.pop("401909102")
    grades = read_log(work / "predictions" / f"{SEASON}-grades.jsonl")
    assert sorted(g["game_id"] for g in grades) == ["0022600002", "0022600003"]
    assert {g["code_version"] for g in grades} == {
        _git(seed, "rev-parse", "--short", "HEAD").strip()}
    assert "2 grade line(s)" in step(report, "grade").detail


def test_provisional_link_is_recorded_before_predicting(world):
    """A game first seen as espn:<id> and predicted, then linked to its NBA
    id, is not predicted again, and the link file goes in the predictions
    commit."""
    work, seed = world["work"], world["seed"]
    extra = ("0022600009", "401909109", date(2026, 10, 22), "2026-10-23T00:00Z", "DET", "LAL")
    GAMES.append(extra)
    try:
        run_once(work, seed, branch="cloud-shadow", fetchers=fetchers())
        log = read_log(work / "predictions" / f"{SEASON}.jsonl")
        assert "espn:401909109" in {ln["game_id"] for ln in log if ln["status"] == "predicted"}
        # The NBA schedule now lists the game (as the Mac snapshot would).
        rows = _schedule_rows()
        rows.append({**rows[1], "game_id": "0022600009", "tip_off_utc": dt("2026-10-23T00:00Z"),
                     "home_team": "DET", "away_team": "LAL",
                     "observed_at": dt("2026-10-22T12:40Z")})
        state.write_snapshot(work / "state", f"nba/schedule-{SEASON}.jsonl", rows,
                             state.SCHEDULE_COLUMNS)
        _git(work, "add", "state")
        _git(work, "commit", "-q", "-m", "state: mac snapshot")
        later = datetime(2026, 10, 22, 12, 52, tzinfo=UTC)
        report = run_once(work, seed, branch="cloud-shadow",
                          fetchers=fetchers(clock=lambda: later))
    finally:
        GAMES.remove(extra)
    log = read_log(work / "predictions" / f"{SEASON}.jsonl")
    predicted = [ln["game_id"] for ln in log if ln["status"] == "predicted"]
    assert "0022600009" not in predicted
    assert "skipped" in step(report, "predict").detail or "1 duplicate" in \
        step(report, "predict").detail
    assert files_of(work, "HEAD") == [f"predictions/id-links/{SEASON}.jsonl"]
    assert subjects(work, 1) == ["predictions: grades 2026-10-22"]


# --- fix round 1 --------------------------------------------------------------


def _advance_remote(world, name: str, subject: str) -> str:
    """Another writer pushes a state commit to origin/cloud-shadow."""
    other = world["tmp"] / name
    _git(world["tmp"], "clone", "-q", "-b", "cloud-shadow", str(world["remote"]), str(other))
    (other / "state" / "players").mkdir(parents=True, exist_ok=True)
    (other / "state" / "players" / "aliases.json").write_text("{}\n")
    _git(other, "add", "state")
    _git(other, "commit", "-q", "-m", subject)
    _git(other, "push", "-q", "origin", "cloud-shadow")
    return _git(other, "rev-parse", "HEAD").strip()


def test_state_commit_failure_skips_predict_but_grades(world, monkeypatch):
    real = cloud_run.publish.commit_and_push

    def failing(repo, paths, message, push=True, *, kind="predictions", branch="main"):
        if kind == "state":
            return cloud_run.publish.PublishResult(False, False, "git commit failed: disk", True)
        return real(repo, paths, message, push, kind=kind, branch=branch)

    monkeypatch.setattr(cloud_run.publish, "commit_and_push", failing)
    report = run_once(world["work"], world["seed"], branch="cloud-shadow", fetchers=fetchers())
    assert step(report, "state_commit").ok is False and report.publish_error
    assert step(report, "predict") == cloud_run.StepOutcome(
        "predict", True, False, "not run: state commit failed")
    assert step(report, "grade").ok is True
    assert not (world["work"] / "predictions" / f"{SEASON}.jsonl").exists()


def test_checkout_is_fast_forwarded_to_the_remote_before_the_run(world):
    theirs = _advance_remote(world, "other", "state: someone else")
    report = run_once(world["work"], world["seed"], branch="cloud-shadow", fetchers=fetchers())
    assert report.pushed and not report.publish_error
    work = world["work"]
    assert _git(work, "rev-parse", "HEAD~2").strip() == theirs
    lines = read_log(work / "predictions" / f"{SEASON}.jsonl")
    assert {ln["state_version"] for ln in lines if ln["status"] == "predicted"} == {
        _git(work, "rev-parse", "--short", "HEAD~1").strip()}


def test_diverged_checkout_fails_cleanly_before_writing(world):
    work = world["work"]
    (work / "state" / "x.jsonl").write_text('{"format":"lines","v":1}\n')
    _git(work, "add", "state")
    _git(work, "commit", "-q", "-m", "state: local only")
    _advance_remote(world, "other", "state: someone else")
    head = _git(work, "rev-parse", "HEAD")
    report = run_once(work, world["seed"], branch="cloud-shadow", fetchers=fetchers())
    assert report.publish_error and not report.pushed
    assert step(report, "rebuild").detail.startswith(
        "not run: cloud-shadow cannot fast-forward to origin/cloud-shadow")
    assert _git(work, "rev-parse", "HEAD") == head
    assert _git(work, "status", "--porcelain") == ""


def test_remote_moved_mid_run_pushes_nothing_and_rewrites_nothing(world):
    work = world["work"]
    start = _git(work, "rev-parse", "HEAD").strip()
    moved = []

    def someone_pushes(now):
        moved.append(_advance_remote(world, "racer", "state: racer"))
        return None

    report = run_once(work, world["seed"], branch="cloud-shadow",
                      fetchers=fetchers(injury_latest=someone_pushes))
    assert not report.pushed and not report.publish_error and not report.unexpected
    assert "remote moved" in step(report, "push").detail
    assert "next run" in step(report, "push").detail
    # No rebase: our two commits sit on the original start, untouched.
    assert _git(work, "rev-parse", "HEAD~2").strip() == start
    assert _git(world["remote"], "rev-parse", "cloud-shadow").strip() == moved[0]
    sealed = {ln["state_version"] for ln in read_log(work / "predictions" / f"{SEASON}.jsonl")
              if ln["status"] == "predicted"}
    assert sealed == {report.state_commit}
    full = _git(work, "rev-parse", report.state_commit).strip()
    assert _git(work, "rev-parse", "HEAD~1").strip() == full
    assert _git(work, "merge-base", "--is-ancestor", full, "HEAD") == ""
    assert summary.outputs(report)["site_changed"] == "false"


def test_odds_credits_counted_once_even_when_loading_fails(world, monkeypatch):
    def broken_load(con, downloaded):
        raise ValueError("bad payload")

    monkeypatch.setattr(cloud_run.odds, "load", broken_load)
    calls = []

    def download(key, now):
        calls.append(now)
        return odds.download(key, now=now, session=_OddsSession())

    work = world["work"]
    report = run_once(work, world["seed"], branch="cloud-shadow",
                      fetchers=fetchers(odds_download=download))
    assert step(report, "odds").ok is False and report.unexpected
    attempts = cloud_run.odds_attempts(work / "state")
    assert [(a["ok"], a["downloaded"], a["remaining"]) for a in attempts] == [(False, True, 480)]
    later = datetime(2026, 10, 22, 12, 52, tzinfo=UTC)
    again = run_once(work, world["seed"], branch="cloud-shadow",
                     fetchers=fetchers(odds_download=download, clock=lambda: later))
    assert step(again, "odds").due is False
    assert len(calls) == 1


def test_changed_files_ignores_rename_pairs(world):
    work = world["work"]
    src = f"state/nba/schedule-{SEASON}.jsonl"
    _git(work, "mv", src, "state/nba/renamed.jsonl")
    (work / "state" / "new.jsonl").write_text("x\n")
    found = cloud_run._changed_files(work, "state")
    assert found == sorted([work / "state" / "new.jsonl", work / "state" / "nba" / "renamed.jsonl"])
    assert all(p.exists() for p in found)


def test_secret_leak_flag_drives_the_outputs(world, monkeypatch):
    work = world["work"]

    def leaky(now):
        (work / "state" / "leak.jsonl").write_text(f"{FAKE_KEY}\n")
        return None

    report = run_once(work, world["seed"], branch="cloud-shadow",
                      fetchers=fetchers(injury_latest=leaky))
    assert report.secret_leak and report.unexpected
    assert "SECRET FOUND" in summary.markdown(report)
    assert summary.outputs(report)["raw_upload_failed"] == "false"
    assert summary.outputs(report)["secret_leak"] == "true"


@pytest.mark.parametrize("refspec", ["absent", "single-branch-main"])
def test_start_fetch_does_not_rely_on_the_remote_fetch_config(world, refspec):
    """actions/checkout's remote config is not ours to rely on: with no
    fetch refspec (or one for main only) a plain ``git fetch origin <branch>``
    would only update FETCH_HEAD and the fast-forward would miss the remote."""
    work = world["work"]
    _git(work, "config", "--unset-all", "remote.origin.fetch")
    if refspec == "single-branch-main":
        _git(work, "config", "remote.origin.fetch", "+refs/heads/main:refs/remotes/origin/main")
    theirs = _advance_remote(world, "other", "state: someone else")
    report = run_once(work, world["seed"], branch="cloud-shadow", fetchers=fetchers())
    assert not report.publish_error, step(report, "rebuild").detail
    assert report.pushed
    assert _git(work, "rev-parse", "HEAD~2").strip() == theirs
    assert _git(world["remote"], "rev-parse", "cloud-shadow").strip() == (
        _git(work, "rev-parse", "HEAD").strip())


def test_start_fetch_fast_forwards_a_shallow_single_branch_clone(world):
    """A depth-1 checkout (as in live.yml) whose remote moved by two commits
    still fast-forwards: the start fetch must not cut the history at the new
    tip (``--depth=1`` there would make the old HEAD look unrelated)."""
    shallow = world["tmp"] / "shallow"
    _git(world["tmp"], "clone", "-q", "--depth", "1", "-b", "cloud-shadow",
         f"file://{world['remote']}", str(shallow))
    assert _git(shallow, "rev-parse", "--is-shallow-repository").strip() == "true"
    first = _advance_remote(world, "one", "state: one")
    other = world["tmp"] / "one"
    (other / "state" / "players" / "aliases.json").write_text('{"a": "b"}\n')
    _git(other, "commit", "-q", "-am", "state: two")
    _git(other, "push", "-q", "origin", "cloud-shadow")
    theirs = _git(other, "rev-parse", "HEAD").strip()
    assert _git(other, "rev-parse", "HEAD~1").strip() == first
    report = run_once(shallow, world["seed"], branch="cloud-shadow", fetchers=fetchers())
    assert not report.publish_error, step(report, "rebuild").detail
    assert report.pushed
    assert _git(shallow, "rev-parse", "HEAD~2").strip() == theirs
    assert _git(world["remote"], "rev-parse", "cloud-shadow").strip() == (
        _git(shallow, "rev-parse", "HEAD").strip())


# --- pre-merge fix: cloud-run needs an explicit publish branch ---------------


def test_cloud_run_refuses_when_the_publish_branch_is_unset(world, monkeypatch):
    called = []
    monkeypatch.delenv("PREDICTOR_PUBLISH_BRANCH", raising=False)
    monkeypatch.setattr(cloud_run, "run_once", lambda *a, **k: called.append(1))
    monkeypatch.setattr(cli, "_cloud_fetchers", lambda: fetchers())
    monkeypatch.setattr(cli, "_repo_dir", lambda: world["seed"])

    result = CliRunner().invoke(cli.app, ["cloud-run", "--state-repo", str(world["work"])])

    assert result.exit_code == 1
    assert "PREDICTOR_PUBLISH_BRANCH" in result.output
    assert "cloud-shadow" in result.output and "main" in result.output
    assert called == []
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_cloud_run_refuses_an_invalid_publish_branch_plainly(world, monkeypatch):
    called = []
    monkeypatch.setenv("PREDICTOR_PUBLISH_BRANCH", "dev")
    monkeypatch.setattr(cloud_run, "run_once", lambda *a, **k: called.append(1))
    monkeypatch.setattr(cli, "_repo_dir", lambda: world["seed"])

    result = CliRunner().invoke(cli.app, ["cloud-run", "--state-repo", str(world["work"])])

    assert result.exit_code == 1
    assert "must be main or cloud-shadow" in result.output
    assert called == []



# --- pre-merge fix: odds data is never republished (licensing) ---------------

# Strings that only occur in the fake Odds API response body.
_ODDS_MARKERS = ("oddsevent1", "draftkings", "theoddsapi")


def test_no_odds_content_in_any_state_file_or_archive_member(world):
    gh = FakeGh()
    work = world["work"]

    report = run_once(work, world["seed"], branch="cloud-shadow", fetchers=fetchers(gh_runner=gh))

    assert step(report, "odds").ok, report.steps
    sd = work / "state"
    files = [p for p in sd.rglob("*") if p.is_file()]
    assert files
    for path in files:
        text = path.read_text(encoding="utf-8").lower()
        for marker in _ODDS_MARKERS:
            assert marker not in text, (path, marker)
    # Only the attempt lines (no odds content) live under state/odds.
    assert [p.name for p in sd.glob("odds/**/*") if p.is_file()] == ["attempts-2026-10.jsonl"]
    tracked = _git(work, "ls-files", "state").splitlines()
    assert not [t for t in tracked if t.startswith("state/odds/") and "attempts-" not in t]
    assert not [t for t in tracked if t.startswith("state/nba/odds")]

    assert gh.assets, "the raw archive was uploaded"
    with tarfile.open(gh.assets[0]) as tar:
        members = tar.getmembers()
        assert members
        for member in members:
            assert not member.name.startswith("raw/odds")
            if member.isfile():
                body = tar.extractfile(member).read()
                if body[:2] == b"\x1f\x8b":
                    import gzip
                    body = gzip.decompress(body)
                for marker in ("oddsevent1", "draftkings"):
                    assert marker.encode() not in body.lower(), (member.name, marker)
    # The raw odds bytes stay only in this run's temp data dir.
    assert any((config.settings.raw_dir / "odds").rglob("*"))

    # Market fields on the prediction lines are unchanged.
    lines = read_log(work / "predictions" / f"{SEASON}.jsonl")
    bos = next(ln for ln in lines if ln["game_id"] == "0022600002")
    assert bos["market_p_home"] is not None and bos["market_books"] == 1


def test_odds_window_stops_after_a_download_without_stored_odds(world):
    calls = []

    def download(key, now):
        calls.append(now)
        return odds.download(key, now=now, session=_OddsSession())

    work = world["work"]
    first = run_once(work, world["seed"], branch="cloud-shadow",
                     fetchers=fetchers(odds_download=download))
    assert step(first, "odds").ok
    later = datetime(2026, 10, 22, 12, 52, tzinfo=UTC)
    second = run_once(work, world["seed"], branch="cloud-shadow",
                      fetchers=fetchers(odds_download=download, clock=lambda: later))
    assert step(second, "odds").due is False
    assert len(calls) == 1


# --- cloud-loop: real runs in their own data dirs ------------------------------


def test_cloud_run_records_a_pinned_code_version_from_the_env(world, monkeypatch):
    monkeypatch.setenv("PREDICTOR_CODE_VERSION", "abc1234")
    monkeypatch.setenv("PREDICTOR_CODE_DIRTY", "false")
    result = _invoke(world, monkeypatch, fetchers())
    assert result.exit_code == 0, result.output
    lines = read_log(world["work"] / "predictions" / f"{SEASON}.jsonl")
    assert lines
    assert all(line["code_version"] == "abc1234" and line["code_dirty"] is False
               for line in lines)


def _in_process_child(world, monkeypatch, fx, before=None):
    """A cloud-loop child that runs `cloud-run` in this process: the env the
    loop gives the child is applied for the call (its data dir included)."""

    def child(argv, *, env, cwd, timeout):
        assert argv[3:5] == ["cloud-run", "--state-repo"]
        if before is not None:
            before()
        with monkeypatch.context() as m:
            for key in ("PREDICTOR_DATA_DIR", "PREDICTOR_PUBLISH_BRANCH", "GITHUB_OUTPUT",
                        "GITHUB_STEP_SUMMARY", "PREDICTOR_CODE_VERSION",
                        "PREDICTOR_CODE_DIRTY"):
                if key in env:
                    m.setenv(key, env[key])
                else:
                    m.delenv(key, raising=False)
            s = Settings(data_dir=Path(env["PREDICTOR_DATA_DIR"]))
            for module in (config, db, raw_store):
                m.setattr(module, "settings", s)
            m.setattr(cli, "_cloud_fetchers", lambda: fx)
            m.setattr(cli, "_repo_dir", lambda: world["seed"])
            result = CliRunner().invoke(cli.app, argv[3:])
        return result.exit_code

    return child


def test_loop_runs_real_iterations_each_in_a_fresh_data_dir(world, monkeypatch):
    from datetime import timedelta

    from predictor.cloud import loop

    work, seed = world["work"], world["seed"]
    base = world["tmp"] / "loop-data"
    now = [NOW]
    before = config.settings

    def clock():
        return now[0]

    def sleep(seconds):
        now[0] += timedelta(seconds=seconds)

    seen_dirs = []
    child = _in_process_child(world, monkeypatch, fetchers(clock=clock))

    def tracked(argv, *, env, cwd, timeout):
        seen_dirs.append(Path(env["PREDICTOR_DATA_DIR"]))
        return child(argv, env=env, cwd=cwd, timeout=timeout)

    texts = []
    result = loop.run_loop(work, seed, branch="cloud-shadow", base_dir=base,
                           fetchers=fetchers(clock=clock), max_minutes=30, clock=clock,
                           sleep=sleep, child=tracked, emit=texts.append)
    # 12:37 at once; 12:52 would run to 13:07 at the latest (15 min watchdog),
    # within 13:07; 13:07 would not be.
    assert result.iterations == 2 and result.ok, texts
    assert len(set(seen_dirs)) == 2 and all(d.parent == base for d in seen_dirs)
    assert not any(d.exists() for d in seen_dirs)
    assert config.settings is before
    assert subjects(work, 2) == ["predictions: 2026-10-22 slate",
                                 "state: cloud run 2026-10-22T12:37Z"]
    assert texts[0].startswith("# Loop iteration 1 — 2026-10-22 12:37 UTC")
    assert "## Cloud run 2026-10-22 12:37 UTC" in texts[0]


def test_every_call_in_a_loop_records_the_code_version_the_loop_started_with(world, monkeypatch):
    """When the code checkout moves mid-job, lines still name the code the
    loop started with."""
    from datetime import timedelta

    from predictor.cloud import loop

    work, seed = world["work"], world["seed"]
    start_sha = _git(seed, "rev-parse", "--short", "HEAD").strip()
    now = [NOW]

    def clock():
        return now[0]

    def sleep(seconds):
        now[0] += timedelta(seconds=seconds)

    def code_commit():
        (seed / "code.txt").write_text(str(now[0]), encoding="utf-8")
        _git(seed, "add", "code.txt")
        _git(seed, "commit", "-q", "-m", "code change")

    child = _in_process_child(world, monkeypatch, fetchers(clock=clock), before=code_commit)
    result = loop.run_loop(work, seed, branch="cloud-shadow", base_dir=world["tmp"] / "ld",
                           fetchers=fetchers(clock=clock), max_minutes=30, clock=clock,
                           sleep=sleep, child=child)
    assert result.iterations == 2 and result.ok
    assert _git(seed, "rev-parse", "--short", "HEAD").strip() != start_sha
    lines = read_log(work / "predictions" / f"{SEASON}.jsonl")
    assert lines
    assert all(line["code_version"] == start_sha and line["code_dirty"] is False
               for line in lines)
