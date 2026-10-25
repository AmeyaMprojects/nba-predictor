"""The Mac's weekly history refresh (branch mac-history) and the cloud run's
copy of it. Throwaway repos with bare remotes under tmp_path only: no
network, no real remote, no launchctl."""

from __future__ import annotations

import gzip
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from predictor import cli, config, db, raw_store
from predictor.cloud import macsync, state
from predictor.cloud.run import Fetchers, run_once
from predictor.config import Settings

SEASON = "2026-27"
REFRESH_NOW = datetime(2026, 10, 26, 5, 45, tzinfo=UTC)  # Monday 11:15 IST


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return result.stdout


def _games(n: int = 2) -> list[dict]:
    return [
        {
            "game_id": f"00226000{i:02d}", "season": SEASON, "game_date": "2026-10-21",
            "home_team": "BOS", "away_team": "DET", "home_points": 110 + i,
            "away_points": 100, "status": "FINAL", "reconstructed": False,
            "observed_at": "2026-10-22T06:00:00+00:00",
        }
        for i in range(1, n + 1)
    ]


def _news(keys: list[str]) -> list[dict]:
    return [
        {"item_key": k, "feed": "espn", "title": f"t {k}", "link": f"https://example.invalid/{k}",
         "summary": "s", "observed_at": "2026-10-20T10:00:00+00:00"}
        for k in keys
    ]


@pytest.fixture
def repos(tmp_path, monkeypatch):
    """remote.git with main, cloud-shadow and mac-history (all pushed by hand
    from the seed), a Mac clone on mac-history and a cloud clone on
    cloud-shadow."""
    for var in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(var, "Predictor Bot")
    for var in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(var, "bot@example.invalid")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    remote = tmp_path / "remote.git"
    remote.mkdir()
    _git(remote, "init", "--bare", "-b", "main")
    seed = tmp_path / "seed"
    seed.mkdir()
    _git(seed, "init", "-b", "main")
    (seed / "README.md").write_text("seed\n", encoding="utf-8")
    state.write_snapshot(seed / "state", f"nba/games-{SEASON}.jsonl", _games(1),
                         state.GAMES_COLUMNS)
    _git(seed, "add", ".")
    _git(seed, "commit", "-m", "seed")
    _git(seed, "remote", "add", "origin", str(remote))
    _git(seed, "push", "origin", "main")
    _git(seed, "push", "origin", "main:cloud-shadow")
    _git(seed, "push", "origin", "main:mac-history")
    mac = tmp_path / "mac"
    _git(tmp_path, "clone", "-q", "-b", "mac-history", str(remote), str(mac))
    cloud = tmp_path / "cloud"
    _git(tmp_path, "clone", "-q", "-b", "cloud-shadow", str(remote), str(cloud))
    return {"remote": remote, "seed": seed, "mac": mac, "cloud": cloud, "tmp": tmp_path}


def _mac_commit(mac: Path, subject: str, writes: dict[str, str | None]) -> None:
    for rel, text in writes.items():
        path = mac / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text or "", encoding="utf-8")
    _git(mac, "add", "-A")
    _git(mac, "commit", "-m", subject)
    _git(mac, "push", "-q", "origin", "mac-history")


def _snapshot_text(tmp: Path, rows: list[dict]) -> str:
    path = state.write_snapshot(tmp / "scratch-state", f"nba/games-{SEASON}.jsonl", rows,
                                state.GAMES_COLUMNS)
    return path.read_text(encoding="utf-8")


# --- the cloud side ------------------------------------------------------------


def test_mac_history_copied_when_every_commit_is_safe(repos):
    mac, cloud = repos["mac"], repos["cloud"]
    text = _snapshot_text(repos["tmp"], _games(2))
    _mac_commit(mac, "state: nba history refresh 2026-10-26",
                {f"state/nba/games-{SEASON}.jsonl": text})

    result = macsync.sync_mac_history(cloud)

    assert result.skipped_reason is None
    assert result.copied == [f"state/nba/games-{SEASON}.jsonl"]
    assert (cloud / f"state/nba/games-{SEASON}.jsonl").read_text(encoding="utf-8") == text
    # Copied into the working tree only: nothing committed, nothing merged.
    assert _git(cloud, "log", "-1", "--format=%s").strip() == "seed"
    assert _git(cloud, "rev-parse", "--abbrev-ref", "HEAD").strip() == "cloud-shadow"


def test_mac_history_with_a_commit_outside_state_nba_is_skipped(repos):
    mac, cloud = repos["mac"], repos["cloud"]
    _mac_commit(mac, "state: nba history refresh 2026-10-26",
                {f"state/nba/games-{SEASON}.jsonl": _snapshot_text(repos["tmp"], _games(2))})
    _mac_commit(mac, "tweak the readme", {"README.md": "changed\n"})
    before = (cloud / f"state/nba/games-{SEASON}.jsonl").read_text(encoding="utf-8")

    result = macsync.sync_mac_history(cloud)

    assert result.copied == []
    assert "tweak the readme" in result.skipped_reason
    assert macsync.last_synced(cloud / "state") is None
    assert (cloud / f"state/nba/games-{SEASON}.jsonl").read_text(encoding="utf-8") == before
    # Only the record of the rejected tip is written.
    assert _git(cloud, "status", "--porcelain", "-uall").strip() == (
        "?? state/mac_history/rejected.jsonl")


def test_a_state_commit_outside_state_nba_is_skipped_too(repos):
    mac, cloud = repos["mac"], repos["cloud"]
    _mac_commit(mac, "state: nba history refresh 2026-10-26",
                {f"state/nba/games-{SEASON}.jsonl": _snapshot_text(repos["tmp"], _games(2)),
                 "state/results/2026-27.jsonl": '{"format":"lines","v":1}\n'})

    result = macsync.sync_mac_history(cloud)

    assert result.copied == []
    assert "state: nba history refresh 2026-10-26" in result.skipped_reason
    assert _git(cloud, "status", "--porcelain", "-uall").strip() == (
        "?? state/mac_history/rejected.jsonl")


def test_missing_mac_history_branch_is_a_quiet_noop(repos):
    remote, cloud = repos["remote"], repos["cloud"]
    _git(remote, "branch", "-D", "mac-history")

    result = macsync.sync_mac_history(cloud)

    assert result == macsync.MacSync(copied=[], skipped_reason=None)
    assert _git(cloud, "status", "--porcelain").strip() == ""


def test_unchanged_files_are_not_copied_again(repos):
    mac, cloud = repos["mac"], repos["cloud"]
    _mac_commit(mac, "state: nba history refresh 2026-10-26",
                {f"state/nba/games-{SEASON}.jsonl": _snapshot_text(repos["tmp"], _games(2))})
    assert macsync.sync_mac_history(cloud).copied == [f"state/nba/games-{SEASON}.jsonl"]
    _git(cloud, "add", "-A")
    _git(cloud, "commit", "-qm", "state: cloud run 2026-10-26T06:07Z")

    again = macsync.sync_mac_history(cloud)

    assert again.copied == [] and again.skipped_reason is None
    assert _git(cloud, "status", "--porcelain").strip() == ""


class _Recorder:
    """subprocess.run, recording each git command line."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        return subprocess.run(argv, **kwargs)

    def git_verbs(self) -> list[str]:
        return [c[1] for c in self.calls]


def test_unchanged_mac_history_tip_is_a_quiet_noop_without_a_fetch(repos):
    mac, cloud = repos["mac"], repos["cloud"]
    _mac_commit(mac, "state: nba history refresh 2026-10-26",
                {f"state/nba/games-{SEASON}.jsonl": _snapshot_text(repos["tmp"], _games(2))})
    first = macsync.sync_mac_history(cloud)
    assert first.copied == [f"state/nba/games-{SEASON}.jsonl"]
    tip = _git(repos["remote"], "rev-parse", "mac-history").strip()
    assert macsync.last_synced(cloud / "state") == tip
    _git(cloud, "add", "-A")
    _git(cloud, "commit", "-qm", "state: cloud run 2026-10-26T06:07Z")
    runner = _Recorder()

    again = macsync.sync_mac_history(cloud, runner=runner)

    assert again == macsync.MacSync(copied=[], skipped_reason=None)
    assert runner.git_verbs() == ["ls-remote"]
    assert _git(cloud, "status", "--porcelain").strip() == ""


def test_changed_tip_on_a_shallow_checkout_unshallows_and_copies_in(repos):
    """live.yml checks out with fetch-depth 1: the full history is fetched
    once the tip moves, so main's older commits are not in the range."""
    remote, mac, seed, tmp = repos["remote"], repos["mac"], repos["seed"], repos["tmp"]
    (seed / "code.py").write_text("x = 1\n", encoding="utf-8")
    _git(seed, "add", ".")
    _git(seed, "commit", "-m", "more code")
    _git(seed, "push", "-q", "origin", "main", "main:cloud-shadow")
    _mac_commit(mac, "state: nba history refresh 2026-10-26",
                {f"state/nba/games-{SEASON}.jsonl": _snapshot_text(tmp, _games(2))})
    shallow = tmp / "shallow"
    _git(tmp, "clone", "-q", "--depth", "1", "-b", "cloud-shadow", f"file://{remote}",
         str(shallow))
    assert _git(shallow, "rev-parse", "--is-shallow-repository").strip() == "true"
    runner = _Recorder()

    result = macsync.sync_mac_history(shallow, runner=runner)

    assert result.skipped_reason is None
    assert result.copied == [f"state/nba/games-{SEASON}.jsonl"]
    fetches = [c for c in runner.calls if c[1] == "fetch"]
    assert len(fetches) == 1 and "--unshallow" in fetches[0]
    assert not any("--shallow-exclude" in a for c in runner.calls for a in c)
    assert _git(shallow, "rev-parse", "--is-shallow-repository").strip() == "false"
    assert macsync.last_synced(shallow / "state") == \
        _git(remote, "rev-parse", "mac-history").strip()


def test_shallow_checkout_with_no_mac_commits_yet_is_a_quiet_noop(repos):
    """Right after `git push origin main:mac-history` (Task 14) the branch
    holds nothing main lacks: nothing is copied from its (older) tree."""
    remote, seed, tmp = repos["remote"], repos["seed"], repos["tmp"]
    (seed / "code.py").write_text("x = 1\n", encoding="utf-8")
    _git(seed, "add", ".")
    _git(seed, "commit", "-m", "more code")
    _git(seed, "push", "-q", "origin", "main", "main:cloud-shadow")
    shallow = tmp / "shallow"
    _git(tmp, "clone", "-q", "--depth", "1", "-b", "cloud-shadow", f"file://{remote}",
         str(shallow))

    result = macsync.sync_mac_history(shallow)

    assert result == macsync.MacSync(copied=[], skipped_reason=None)
    assert _git(shallow, "status", "--porcelain", "-uall").split() == \
        ["??", f"state/{macsync.SYNCED_PATH}"]


def test_unsafe_commit_is_skipped_every_run_until_fixed(repos):
    mac, cloud = repos["mac"], repos["cloud"]
    _mac_commit(mac, "oops", {"README.md": "x\n"})

    first = macsync.sync_mac_history(cloud)
    second = macsync.sync_mac_history(cloud)

    assert "oops" in first.skipped_reason and "oops" in second.skipped_reason
    assert macsync.last_synced(cloud / "state") is None
    assert _git(cloud, "status", "--porcelain", "-uall").strip() == (
        "?? state/mac_history/rejected.jsonl")


def test_mac_news_history_is_appended_once_without_duplicates(repos):
    mac, cloud = repos["mac"], repos["cloud"]
    # The cloud already archived item "b" itself.
    state.append_rows(cloud / "state", "news", _news(["b"]))
    _git(cloud, "add", "-A")
    _git(cloud, "commit", "-qm", "state: cloud run 2026-10-25T10:07Z")
    mac_state = repos["tmp"] / "mac-news" / "state"
    files = state.append_rows(mac_state, "news", _news(["a", "b", "c"]))
    writes = {f"state/{p.relative_to(mac_state)}": p.read_text(encoding="utf-8") for p in files}
    _mac_commit(mac, "state: nba history refresh 2026-10-26", writes)

    first = macsync.sync_mac_history(cloud)

    assert first.skipped_reason is None
    assert first.news_items == 2
    keys = [r["item_key"] for r in state.read_rows(cloud / "state", "news")]
    assert sorted(keys) == ["a", "b", "c"]
    _git(cloud, "add", "-A")
    _git(cloud, "commit", "-qm", "state: cloud run 2026-10-26T06:07Z")

    second = macsync.sync_mac_history(cloud)

    assert second.news_items == 0
    assert _git(cloud, "status", "--porcelain").strip() == ""


def _point_data_dir(path: Path, monkeypatch) -> Settings:
    s = Settings(data_dir=path)
    s.ensure_dirs()
    monkeypatch.setattr(config, "settings", s)
    monkeypatch.setattr(db, "settings", s)
    monkeypatch.setattr(raw_store, "settings", s)
    return s


def test_cloud_run_puts_the_mac_files_into_its_state_commit(repos, monkeypatch):
    mac, cloud, seed = repos["mac"], repos["cloud"], repos["seed"]
    _point_data_dir(repos["tmp"] / "data", monkeypatch)
    monkeypatch.setenv("ODDS_API_KEY", "0123456789abcdef" * 2)
    _mac_commit(mac, "state: nba history refresh 2026-10-26",
                {f"state/nba/games-{SEASON}.jsonl": _snapshot_text(repos["tmp"], _games(2))})

    def offline(*_a, **_k):
        raise OSError("offline in tests")

    fetchers = Fetchers(scoreboard=offline, summary=offline, injury_latest=lambda now: None,
                        odds_download=offline, gh_runner=offline,
                        clock=lambda: datetime(2026, 10, 26, 6, 7, tzinfo=UTC),
                        news=lambda now: {})
    report = run_once(cloud, seed, branch="cloud-shadow", fetchers=fetchers)

    step = next(s for s in report.steps if s.name == "mac_history")
    assert step.due and step.ok and "1 file" in step.detail
    assert report.state_commit is not None
    changed = _git(cloud, "show", "--name-only", "--format=", report.state_commit).split()
    assert f"state/nba/games-{SEASON}.jsonl" in changed
    rebuild = next(s for s in report.steps if s.name == "rebuild")
    assert rebuild.ok


def test_cloud_run_reports_a_skipped_mac_history(repos, monkeypatch):
    mac, cloud, seed = repos["mac"], repos["cloud"], repos["seed"]
    _point_data_dir(repos["tmp"] / "data", monkeypatch)
    monkeypatch.setenv("ODDS_API_KEY", "0123456789abcdef" * 2)
    _mac_commit(mac, "oops", {"README.md": "x\n"})

    def offline(*_a, **_k):
        raise OSError("offline in tests")

    fetchers = Fetchers(scoreboard=offline, summary=offline, injury_latest=lambda now: None,
                        odds_download=offline, gh_runner=offline,
                        clock=lambda: datetime(2026, 10, 26, 6, 7, tzinfo=UTC),
                        news=lambda now: {})
    report = run_once(cloud, seed, branch="cloud-shadow", fetchers=fetchers)

    step = next(s for s in report.steps if s.name == "mac_history")
    assert step.due and not step.ok and "oops" in step.detail
    assert not report.publish_error


# --- the Mac side --------------------------------------------------------------


def test_prepare_worktree_adds_once_then_fast_forwards(repos):
    remote, seed, tmp = repos["remote"], repos["seed"], repos["tmp"]
    live = tmp / "live"
    _git(tmp, "clone", "-q", "-b", "main", str(remote), str(live))
    path = tmp / "data" / "mac-history"

    assert macsync.prepare_worktree(live, path) == path
    assert _git(path, "rev-parse", "--abbrev-ref", "HEAD").strip() == macsync.MAC_BRANCH
    assert _git(live, "rev-parse", "--abbrev-ref", "HEAD").strip() == "main"

    other = tmp / "other"
    _git(tmp, "clone", "-q", "-b", "mac-history", str(remote), str(other))
    _mac_commit(other, "state: nba history refresh 2026-10-19", {"state/nba/x.jsonl": "{}\n"})

    assert macsync.prepare_worktree(live, path) == path
    assert (path / "state/nba/x.jsonl").exists()
    assert _git(path, "log", "-1", "--format=%s").strip() == "state: nba history refresh 2026-10-19"


def test_fetch_current_players_archives_before_parsing(tmp_path, monkeypatch):
    _point_data_dir(tmp_path / "data", monkeypatch)
    frame = pd.DataFrame({
        "PERSON_ID": [1630178, 203507], "DISPLAY_FIRST_LAST": ["Tyrese Maxey", "Giannis"],
        "ROSTERSTATUS": [1, 1], "FROM_YEAR": ["2020", "2013"], "TO_YEAR": ["2026", "2026"],
        "TEAM_ID": [1610612755, 1610612749], "TEAM_ABBREVIATION": ["PHI", "MIL"],
    })
    fetched_at = datetime(2026, 10, 26, 5, 46, tzinfo=UTC)

    rows = macsync.fetch_current_players(SEASON, fetch=lambda season: frame,
                                         fetched_at=fetched_at)

    assert [r["player_id"] for r in rows] == [203507, 1630178]
    assert rows[1] == {"player_id": 1630178, "player_name": "Tyrese Maxey", "team": "PHI",
                       "team_id": 1610612755, "roster_status": 1, "from_year": "2020",
                       "to_year": "2026", "season": SEASON}
    entries = list(raw_store.iter_manifest("players"))
    assert len(entries) == 1
    payload = json.loads(gzip.decompress(raw_store.load("players", entries[0]["key"])))
    assert payload["data"][0][0] == 1630178


def _load_games_job(events):
    """A history job whose download is fake and whose load writes one games
    row and one news item into the Mac DB."""

    def download():
        events.append("download")
        return "payload"

    def load(con, payload):
        assert payload == "payload"
        con.execute(
            "INSERT OR IGNORE INTO games_raw (game_id, season, game_date, home_team, away_team,"
            " home_points, away_points, status, reconstructed, observed_at) VALUES"
            " ('0022600001', '2026-27', DATE '2026-10-21', 'BOS', 'DET', 120, 100, 'FINAL',"
            " false, TIMESTAMPTZ '2026-10-22 06:00:00+00')"
        )
        con.execute(
            "INSERT OR IGNORE INTO news_items_raw VALUES ('k1', 'espn', 't', 'l', 's',"
            " TIMESTAMPTZ '2026-10-20 10:00:00+00')"
        )
        return "games: 1 row"

    return cli._HistoryJob("games", download, load)


def test_refresh_history_commits_to_mac_history_only(repos, monkeypatch):
    remote, tmp = repos["remote"], repos["tmp"]
    live = tmp / "live"
    _git(tmp, "clone", "-q", "-b", "main", str(remote), str(live))
    settings = _point_data_dir(tmp / "data", monkeypatch)
    monkeypatch.setattr(cli, "_repo_dir", lambda: live)
    monkeypatch.setattr(cli, "_now", lambda: REFRESH_NOW)
    events: list[str] = []
    monkeypatch.setattr(cli, "_history_jobs", lambda season: [_load_games_job(events)])

    def players(season):
        events.append("players")
        return [{"player_id": 1, "player_name": "A", "team": "BOS", "team_id": 2,
                 "roster_status": 1, "from_year": "2020", "to_year": "2026", "season": season}]

    monkeypatch.setattr(macsync, "fetch_current_players", players)
    real_connect = db.connect_with_retry

    def connect(*a, **k):
        events.append("connect")
        return real_connect(*a, **k)

    monkeypatch.setattr(db, "connect_with_retry", connect)
    main_before = _git(remote, "rev-parse", "main").strip()
    shadow_before = _git(remote, "rev-parse", "cloud-shadow").strip()

    result = CliRunner().invoke(cli.app, ["refresh-history"])

    assert result.exit_code == 0, result.output
    assert events.index("connect") > max(events.index("download"), events.index("players"))
    worktree = settings.data_dir / "mac-history"
    assert _git(remote, "log", "-1", "--format=%s", "mac-history").strip() == \
        "state: nba history refresh 2026-10-26"
    assert _git(remote, "rev-parse", "main").strip() == main_before
    assert _git(remote, "rev-parse", "cloud-shadow").strip() == shadow_before
    files = _git(worktree, "show", "--name-only", "--format=", "HEAD").split()
    assert f"state/nba/games-{SEASON}.jsonl" in files
    assert f"state/nba/players-{SEASON}.jsonl" in files
    assert any(f.startswith("state/news/") for f in files)
    assert all(f.startswith(("state/nba/", "state/news/")) for f in files)
    games = list(state.read_rows(worktree / "state", "nba_games"))
    assert [g["game_id"] for g in games] == ["0022600001"]
    # The live checkout stays on main and is untouched.
    assert _git(live, "rev-parse", "--abbrev-ref", "HEAD").strip() == "main"
    assert _git(live, "status", "--porcelain").strip() == ""

    # A second refresh adds no duplicate news lines and commits nothing new.
    again = CliRunner().invoke(cli.app, ["refresh-history"])
    assert again.exit_code == 0, again.output
    news = [r["item_key"] for r in state.read_rows(worktree / "state", "news")]
    assert news == ["k1"]
    assert _git(remote, "rev-list", "--count", "mac-history").strip() == "2"

    # And the cloud copies it in.
    copied = macsync.sync_mac_history(repos["cloud"])
    assert copied.skipped_reason is None
    assert f"state/nba/games-{SEASON}.jsonl" in copied.copied
    assert copied.news_items == 1


def test_refresh_history_no_push_commits_locally(repos, monkeypatch):
    remote, tmp = repos["remote"], repos["tmp"]
    live = tmp / "live"
    _git(tmp, "clone", "-q", "-b", "main", str(remote), str(live))
    settings = _point_data_dir(tmp / "data", monkeypatch)
    monkeypatch.setattr(cli, "_repo_dir", lambda: live)
    monkeypatch.setattr(cli, "_now", lambda: REFRESH_NOW)
    monkeypatch.setattr(cli, "_history_jobs", lambda season: [_load_games_job([])])
    monkeypatch.setattr(macsync, "fetch_current_players", lambda season: [])
    before = _git(remote, "rev-parse", "mac-history").strip()

    result = CliRunner().invoke(cli.app, ["refresh-history", "--no-push"])

    assert result.exit_code == 0, result.output
    assert _git(remote, "rev-parse", "mac-history").strip() == before
    worktree = settings.data_dir / "mac-history"
    assert _git(worktree, "log", "-1", "--format=%s").strip() == \
        "state: nba history refresh 2026-10-26"


def test_refresh_history_reports_a_failed_download_and_still_exports(repos, monkeypatch):
    remote, tmp = repos["remote"], repos["tmp"]
    live = tmp / "live"
    _git(tmp, "clone", "-q", "-b", "main", str(remote), str(live))
    _point_data_dir(tmp / "data", monkeypatch)
    monkeypatch.setattr(cli, "_repo_dir", lambda: live)
    monkeypatch.setattr(cli, "_now", lambda: REFRESH_NOW)

    def broken():
        raise OSError("no route to host")

    jobs = [cli._HistoryJob("results", broken, lambda con, p: "never"),
            _load_games_job([])]
    monkeypatch.setattr(cli, "_history_jobs", lambda season: jobs)
    monkeypatch.setattr(macsync, "fetch_current_players", lambda season: [])

    result = CliRunner().invoke(cli.app, ["refresh-history"])

    assert result.exit_code == 1
    assert "results" in result.output and "no route to host" in result.output
    assert _git(remote, "log", "-1", "--format=%s", "mac-history").strip() == \
        "state: nba history refresh 2026-10-26"


# --- pre-merge fix: remember a rejected tip, do not refetch it hourly ----------


class _Spy:
    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, args, **kwargs):
        self.calls.append(list(args))
        return subprocess.run(args, **kwargs)

    def fetches(self) -> list[list[str]]:
        return [c for c in self.calls if c[1:2] == ["fetch"]]


def test_rejected_tip_is_recorded_and_not_fetched_again_until_it_moves(repos):
    mac, cloud = repos["mac"], repos["cloud"]
    _mac_commit(mac, "oops", {"README.md": "x\n"})
    tip = _git(repos["remote"], "rev-parse", "mac-history").strip()

    first_spy = _Spy()
    first = macsync.sync_mac_history(cloud, runner=first_spy)
    assert "oops" in first.skipped_reason
    assert len(first_spy.fetches()) == 1
    records = list(state.read_records(cloud / "state" / macsync.REJECTED_PATH))
    assert [r["sha"] for r in records] == [tip]
    assert "oops" in records[0]["reason"]

    # Same tip next run: still reported, but no (unshallow) fetch at all, and
    # the record is not appended again.
    second_spy = _Spy()
    second = macsync.sync_mac_history(cloud, runner=second_spy)
    assert second.copied == []
    assert "oops" in second.skipped_reason
    assert tip[:12] in second.skipped_reason
    assert second_spy.fetches() == []
    assert len(list(state.read_records(cloud / "state" / macsync.REJECTED_PATH))) == 1

    # The branch is fixed (moved): fetched and copied again.
    _git(mac, "reset", "-q", "--hard", "HEAD~1")
    text = _snapshot_text(repos["tmp"], _games(2))
    path = mac / f"state/nba/games-{SEASON}.jsonl"
    path.write_text(text, encoding="utf-8")
    _git(mac, "add", "-A")
    _git(mac, "commit", "-m", "state: nba history refresh 2026-10-26")
    _git(mac, "push", "-q", "--force", "origin", "mac-history")
    third_spy = _Spy()
    third = macsync.sync_mac_history(cloud, runner=third_spy)
    assert third.skipped_reason is None
    assert third.copied == [f"state/nba/games-{SEASON}.jsonl"]
    assert len(third_spy.fetches()) == 1


def test_cloud_run_reports_a_rejected_tip_every_run(repos, monkeypatch):
    mac, cloud, seed = repos["mac"], repos["cloud"], repos["seed"]
    _point_data_dir(repos["tmp"] / "data", monkeypatch)
    monkeypatch.setenv("ODDS_API_KEY", "0123456789abcdef" * 2)
    _mac_commit(mac, "oops", {"README.md": "x\n"})

    def offline(*_a, **_k):
        raise OSError("offline in tests")

    def fetchers(minute):
        return Fetchers(scoreboard=offline, summary=offline, injury_latest=lambda now: None,
                        odds_download=offline, gh_runner=offline,
                        clock=lambda: datetime(2026, 10, 26, 6, minute, tzinfo=UTC),
                        news=lambda now: {})

    for minute in (7, 22):
        report = run_once(cloud, seed, branch="cloud-shadow", fetchers=fetchers(minute))
        step = next(s for s in report.steps if s.name == "mac_history")
        assert step.due and not step.ok and "oops" in step.detail
    # The record went out with the run's state commit.
    assert _git(cloud, "ls-files", "state/mac_history/rejected.jsonl").strip()


def test_mac_odds_snapshot_is_never_copied_in(repos):
    # Licensing: odds are not republished, even if an old Mac export pushed
    # an nba/odds file to mac-history.
    mac, cloud = repos["mac"], repos["cloud"]
    _mac_commit(mac, "state: nba history refresh 2026-10-26",
                {f"state/nba/games-{SEASON}.jsonl": _snapshot_text(repos["tmp"], _games(2)),
                 f"state/nba/odds-{SEASON}.jsonl": '{"format":"lines","v":1}\n'})

    result = macsync.sync_mac_history(cloud)

    assert result.skipped_reason is None
    assert result.copied == [f"state/nba/games-{SEASON}.jsonl"]
    assert not (cloud / f"state/nba/odds-{SEASON}.jsonl").exists()
