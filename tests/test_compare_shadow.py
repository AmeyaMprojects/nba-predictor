"""Task 13: `compare-shadow` checks the cloud's would-be calls against the
Mac's. Pure-function tests for `compare_logs`, plus CLI tests that read
both logs out of throwaway git repos (a bare remote with `main` and
`cloud-shadow`) -- never the real repository, never the network."""

import json
import subprocess
from datetime import date, datetime, timedelta
from pathlib import Path

from typer.testing import CliRunner

from predictor import cli
from predictor.cloud.compare import (
    MAX_P_DIFF,
    MIN_SHADOW_DAYS,
    compare_logs,
    format_report,
)
from predictor.model.live import _MISSED_REASON, _NOT_PREDICTED_REASON, _TBD_REASON

SINCE = date(2026, 11, 2)


def _line(game_id, day, p=0.6, *, at="12:30", tip="23:00", status="predicted", reason=None,
          home="BOS", away="NYK", tip_day_offset=0):
    d = date.fromisoformat(day)
    tip_date = d + timedelta(days=tip_day_offset)
    return {
        "predicted_at": f"{day}T{at}:00+00:00",
        "game_id": game_id,
        "season": "2026-27",
        "game_date": day,
        "tip_off_utc": f"{tip_date.isoformat()}T{tip}:00+00:00" if tip else None,
        "home_team": home,
        "away_team": away,
        "status": status,
        "reason": reason,
        "p_home": p if status == "predicted" else None,
        "state_version": None,
    }


def _days(n, start=SINCE):
    return [(start + timedelta(days=i)).isoformat() for i in range(n)]


def _same_calls(n_days, **cloud_kw):
    mac, cloud = [], []
    for i, day in enumerate(_days(n_days)):
        gid = f"00226001{i:02d}"
        mac.append(_line(gid, day, 0.55 + i / 100))
        cloud.append(_line(gid, day, 0.55 + i / 100, at="12:37", **cloud_kw))
    return mac, cloud


def test_constants():
    assert MIN_SHADOW_DAYS == 3
    assert MAX_P_DIFF == 0.01


def test_three_identical_days_pass():
    mac, cloud = _same_calls(3)
    report = compare_logs(mac, cloud, SINCE, {})
    assert report.ok, report.reasons
    assert report.reasons == []
    assert report.slate_days == [date(2026, 11, 2), date(2026, 11, 3), date(2026, 11, 4)]
    assert len(report.pairs) == 3
    assert report.max_abs_dp == 0.0
    assert report.mac_only == [] and report.cloud_only == []
    assert all(p.cloud_before_tip for p in report.pairs)


def test_difference_over_one_point_fails():
    mac, cloud = _same_calls(3)
    cloud[1]["p_home"] = round(mac[1]["p_home"] + 0.011, 6)
    report = compare_logs(mac, cloud, SINCE, {})
    assert not report.ok
    assert report.max_abs_dp == 0.011
    assert "largest difference 0.011 > 0.01" in report.reasons


def test_difference_of_exactly_one_point_passes():
    mac, cloud = _same_calls(3)
    cloud[0]["p_home"] = round(mac[0]["p_home"] - 0.01, 6)
    report = compare_logs(mac, cloud, SINCE, {})
    assert report.ok, report.reasons
    assert report.max_abs_dp == 0.01


def test_two_days_is_not_enough():
    mac, cloud = _same_calls(2)
    report = compare_logs(mac, cloud, SINCE, {})
    assert not report.ok
    assert len(report.slate_days) == 2
    assert any("2 slate days" in r and "at least 3" in r for r in report.reasons)


def test_mac_only_game_listed_and_fails():
    mac, cloud = _same_calls(3)
    mac.append(_line("0022600999", "2026-11-03", 0.7, home="LAL", away="GSW"))
    report = compare_logs(mac, cloud, SINCE, {})
    assert not report.ok
    assert report.mac_only == ["0022600999"]
    assert any("0022600999" in r and "no cloud call" in r for r in report.reasons)
    text = format_report(report)
    assert "0022600999" in text and "LAL" in text


def test_provisional_cloud_id_matched_via_links():
    mac, cloud = _same_calls(3)
    cloud[2]["game_id"] = "espn:401705555"
    links = {"espn:401705555": mac[2]["game_id"]}
    report = compare_logs(mac, cloud, SINCE, links)
    assert report.ok, report.reasons
    assert report.mac_only == [] and report.cloud_only == []
    pair = next(p for p in report.pairs if p.game_id == mac[2]["game_id"])
    assert pair.cloud_game_id == "espn:401705555"
    assert pair.mac_p == pair.cloud_p

    unlinked = compare_logs(mac, cloud, SINCE, {})
    assert not unlinked.ok
    assert unlinked.mac_only == [mac[2]["game_id"]]
    assert unlinked.cloud_only == ["espn:401705555"]


def test_cloud_call_sealed_after_tip_fails():
    mac, cloud = _same_calls(3)
    cloud[0]["predicted_at"] = "2026-11-02T23:05:00+00:00"
    report = compare_logs(mac, cloud, SINCE, {})
    assert not report.ok
    pair = report.pairs[0]
    assert not pair.cloud_before_tip
    assert any(pair.game_id in r and "after tip-off" in r for r in report.reasons)
    assert "after tip-off" in format_report(report)


def test_tip_is_the_earliest_either_side_recorded():
    mac, cloud = _same_calls(3)
    # The cloud thinks tip is 23:30 and called at 23:10; the Mac knew 23:00.
    cloud[0]["tip_off_utc"] = "2026-11-02T23:30:00+00:00"
    cloud[0]["predicted_at"] = "2026-11-02T23:10:00+00:00"
    report = compare_logs(mac, cloud, SINCE, {})
    assert not report.pairs[0].cloud_before_tip
    assert not report.ok


def test_latest_call_sealed_before_tip_is_used():
    mac, cloud = _same_calls(3)
    # An earlier cloud line, and one after tip: the 12:37 one is used.
    cloud.append(_line(mac[0]["game_id"], mac[0]["game_date"], 0.2, at="11:00"))
    cloud.append(_line(mac[0]["game_id"], mac[0]["game_date"], 0.9, at="23:30"))
    report = compare_logs(mac, cloud, SINCE, {})
    assert report.ok, report.reasons
    pair = report.pairs[0]
    assert pair.cloud_p == mac[0]["p_home"]
    assert pair.cloud_at == datetime.fromisoformat("2026-11-02T12:37:00+00:00")


def test_cloud_only_calls_are_reported_not_failed():
    mac, cloud = _same_calls(3)
    cloud.append(_line("0022600777", "2026-11-04", 0.4))
    report = compare_logs(mac, cloud, SINCE, {})
    assert report.ok, report.reasons
    assert report.cloud_only == ["0022600777"]
    assert "0022600777" in format_report(report)


def test_lines_before_since_are_ignored():
    mac, cloud = _same_calls(3)
    mac.append(_line("0022600001", "2026-11-01", 0.7))
    report = compare_logs(mac, cloud, SINCE, {})
    assert report.ok, report.reasons
    assert all(p.game_date >= SINCE for p in report.pairs)


def test_not_predicted_lines_compared_by_reason_category():
    mac, cloud = _same_calls(3)
    day = "2026-11-03"
    # Same category, different wording: agrees.
    mac.append(_line("0022600500", day, status="not_predicted", reason=_NOT_PREDICTED_REASON))
    cloud.append(_line("0022600500", day, status="not_predicted",
                       reason=_NOT_PREDICTED_REASON.upper(), at="23:10"))
    report = compare_logs(mac, cloud, SINCE, {})
    assert report.ok, report.reasons
    # Different category: a mismatch.
    mac.append(_line("0022600501", day, status="not_predicted", reason=_MISSED_REASON))
    cloud.append(_line("0022600501", day, status="not_predicted",
                       reason=_NOT_PREDICTED_REASON, at="23:10"))
    report = compare_logs(mac, cloud, SINCE, {})
    assert not report.ok
    assert any("0022600501" in r for r in report.reasons)
    assert "0022600500" not in " ".join(report.reasons)


def test_tbd_lines_are_ignored():
    mac, cloud = _same_calls(3)
    cloud.append(_line(mac[0]["game_id"], mac[0]["game_date"], status="not_predicted",
                       reason=_TBD_REASON, at="10:00", tip=None))
    mac.append(_line("0022600600", "2026-11-03", status="not_predicted", reason=_TBD_REASON,
                     tip=None))
    report = compare_logs(mac, cloud, SINCE, {})
    assert report.ok, report.reasons
    assert "0022600600" not in [p.game_id for p in report.pairs]


def test_mac_call_with_cloud_not_predicted_fails_with_the_cloud_reason():
    mac, cloud = _same_calls(3)
    cloud[1] = _line(mac[1]["game_id"], mac[1]["game_date"], status="not_predicted",
                     reason=_MISSED_REASON, at="23:30")
    report = compare_logs(mac, cloud, SINCE, {})
    assert not report.ok
    assert report.mac_only == [mac[1]["game_id"]]
    text = format_report(report)
    assert "no prediction was made before tip-off" in text


def test_report_lists_each_mismatch_in_plain_english():
    mac, cloud = _same_calls(3)
    cloud[1]["p_home"] = round(mac[1]["p_home"] + 0.05, 6)
    report = compare_logs(mac, cloud, SINCE, {})
    text = format_report(report)
    mism = text.split("Mismatches", 1)[1]
    assert mac[1]["game_id"] in mism
    assert "0.560" in mism and "0.610" in mism          # both probabilities
    assert "12:30" in mism and "12:37" in mism            # both seal times
    assert "differ by 0.050" in mism                      # why
    assert text.rstrip().splitlines()[-len(report.reasons) - 1].startswith("FAIL")


# --- CLI ---------------------------------------------------------------

runner = CliRunner()


def _git(cwd: Path, *args: str, env=None) -> str:
    out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30,
                         env=env)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def _write_log(path: Path, lines: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(x, sort_keys=True, separators=(",", ":")) + "\n" for x in lines),
        encoding="utf-8",
    )


def _commit(repo: Path, msg: str, when: str, monkeypatch) -> None:
    monkeypatch.setenv("GIT_AUTHOR_DATE", when)
    monkeypatch.setenv("GIT_COMMITTER_DATE", when)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg)


def _repos(tmp_path, monkeypatch, mac, cloud, links=None):
    """A bare remote with `main` (the Mac's log) and `cloud-shadow` (the
    cloud's log) branched at a merge base committed 2026-11-02 14:00 UTC
    (ET date 2026-11-02), and a separate clone the CLI fetches into."""
    for k, v in {"GIT_AUTHOR_NAME": "T", "GIT_AUTHOR_EMAIL": "t@example.invalid",
                 "GIT_COMMITTER_NAME": "T", "GIT_COMMITTER_EMAIL": "t@example.invalid"}.items():
        monkeypatch.setenv(k, v)
    remote = tmp_path / "remote.git"
    _git(tmp_path, "init", "-q", "--bare", "-b", "main", str(remote))
    seed = tmp_path / "seed"
    _git(tmp_path, "clone", "-q", str(remote), str(seed))
    _git(seed, "checkout", "-q", "-b", "main")
    (seed / "README.md").write_text("seed\n", encoding="utf-8")
    # 2026-11-02 14:00 UTC = 09:00 ET on 2026-11-02.
    _commit(seed, "seed", "2026-11-02T14:00:00+00:00", monkeypatch)
    _git(seed, "push", "-q", "origin", "main")
    _git(seed, "checkout", "-q", "-b", "cloud-shadow")
    _write_log(seed / "predictions" / "2026-27.jsonl", cloud)
    if links:
        p = seed / "predictions" / "id-links" / "2026-27.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{"format":"lines","v":1}\n' + "".join(
            json.dumps({"game_id": n, "linked_at": "2026-11-03T00:00:00+00:00",
                        "provisional_id": pr}, sort_keys=True, separators=(",", ":")) + "\n"
            for pr, n in links.items()), encoding="utf-8")
    _commit(seed, "predictions: cloud", "2026-11-05T14:00:00+00:00", monkeypatch)
    _git(seed, "push", "-q", "origin", "cloud-shadow")
    _git(seed, "checkout", "-q", "main")
    _write_log(seed / "predictions" / "2026-27.jsonl", mac)
    _commit(seed, "predictions: mac", "2026-11-05T14:01:00+00:00", monkeypatch)
    _git(seed, "push", "-q", "origin", "main")

    work = tmp_path / "work"
    _git(tmp_path, "clone", "-q", "--branch", "main", str(remote), str(work))
    monkeypatch.setattr(cli, "_repo_dir", lambda: work)
    monkeypatch.setattr(cli, "_now", lambda: datetime.fromisoformat("2026-11-06T00:00:00+00:00"))
    return work, seed


def test_cli_passes_and_exits_zero(tmp_path, monkeypatch):
    mac, cloud = _same_calls(3)
    work, _ = _repos(tmp_path, monkeypatch, mac, cloud)
    result = runner.invoke(cli.app, ["compare-shadow"])
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output
    for line in mac:
        assert line["game_id"] in result.output
    # the CLI read origin refs; nothing was checked out or changed
    assert _git(work, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert _git(work, "status", "--porcelain") == ""


def test_cli_default_since_is_the_merge_base_et_date(tmp_path, monkeypatch):
    mac, cloud = _same_calls(3)
    # A Mac line from before the merge base's ET date (2026-11-02): ignored.
    mac.append(_line("0022600001", "2026-11-01", 0.7))
    _repos(tmp_path, monkeypatch, mac, cloud)
    result = runner.invoke(cli.app, ["compare-shadow"])
    assert result.exit_code == 0, result.output
    assert "since 2026-11-02" in result.output
    assert "0022600001" not in result.output
    # An explicit --since overrides it: the old Mac-only game now fails.
    result = runner.invoke(cli.app, ["compare-shadow", "--since", "2026-11-01"])
    assert result.exit_code == 1, result.output
    assert "0022600001" in result.output


def test_cli_fails_and_exits_one(tmp_path, monkeypatch):
    mac, cloud = _same_calls(3)
    cloud[0]["p_home"] = round(mac[0]["p_home"] + 0.02, 6)
    _repos(tmp_path, monkeypatch, mac, cloud)
    result = runner.invoke(cli.app, ["compare-shadow"])
    assert result.exit_code == 1, result.output
    assert "FAIL" in result.output
    assert "largest difference 0.02 > 0.01" in result.output


def test_cli_uses_cloud_id_links(tmp_path, monkeypatch):
    mac, cloud = _same_calls(3)
    cloud[2]["game_id"] = "espn:401705555"
    _repos(tmp_path, monkeypatch, mac, cloud, links={"espn:401705555": mac[2]["game_id"]})
    result = runner.invoke(cli.app, ["compare-shadow"])
    assert result.exit_code == 0, result.output
    assert "espn:401705555" in result.output


def test_cli_fetches_new_commits(tmp_path, monkeypatch):
    mac, cloud = _same_calls(2)
    work, seed = _repos(tmp_path, monkeypatch, mac, cloud)
    assert runner.invoke(cli.app, ["compare-shadow"]).exit_code == 1
    # A third day reaches both branches on the remote after the clone.
    more_mac, more_cloud = _same_calls(3)
    _write_log(seed / "predictions" / "2026-27.jsonl", more_mac)
    _commit(seed, "predictions: mac 3", "2026-11-05T15:00:00+00:00", monkeypatch)
    _git(seed, "push", "-q", "origin", "main")
    _git(seed, "checkout", "-q", "cloud-shadow")
    _write_log(seed / "predictions" / "2026-27.jsonl", more_cloud)
    _commit(seed, "predictions: cloud 3", "2026-11-05T15:00:00+00:00", monkeypatch)
    _git(seed, "push", "-q", "origin", "cloud-shadow")
    result = runner.invoke(cli.app, ["compare-shadow"])
    assert result.exit_code == 0, result.output


def test_cli_reads_logs_from_path_arguments(tmp_path, monkeypatch):
    mac, cloud = _same_calls(3)
    cloud[1]["game_id"] = "espn:401700001"
    mac_path, cloud_path = tmp_path / "m" / "2026-27.jsonl", tmp_path / "c" / "2026-27.jsonl"
    _write_log(mac_path, mac)
    _write_log(cloud_path, cloud)
    links = tmp_path / "c" / "id-links" / "2026-27.jsonl"
    links.parent.mkdir(parents=True)
    links.write_text('{"format":"lines","v":1}\n' + json.dumps(
        {"game_id": mac[1]["game_id"], "linked_at": "2026-11-03T00:00:00+00:00",
         "provisional_id": "espn:401700001"}) + "\n", encoding="utf-8")

    def no_git(*a, **k):
        raise AssertionError("git must not run when both logs and --since are given")

    monkeypatch.setattr(cli, "_compare_git", no_git)
    result = runner.invoke(cli.app, ["compare-shadow", "--since", "2026-11-02",
                                     "--mac-log", str(mac_path), "--cloud-log", str(cloud_path)])
    assert result.exit_code == 0, result.output


def test_cli_reports_a_failed_fetch(tmp_path, monkeypatch):
    mac, cloud = _same_calls(3)
    work, _ = _repos(tmp_path, monkeypatch, mac, cloud)
    _git(work, "remote", "set-url", "origin", str(tmp_path / "missing.git"))
    result = runner.invoke(cli.app, ["compare-shadow"])
    assert result.exit_code == 1
    assert "could not fetch" in result.output
