from datetime import UTC, datetime

import requests
from typer.testing import CliRunner

from predictor import cli, config, db, raw_store
from predictor.config import Settings
from predictor.sources import schedule

runner = CliRunner()


def _point_settings_at_tmp(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(config, "settings", s)
    monkeypatch.setattr(db, "settings", s)
    monkeypatch.setattr(raw_store, "settings", s)
    return s


def _result(season, written=1230, no_tipoff=(), undetermined=(), mismatches=()):
    return schedule.IngestResult(
        season=season, written=written, no_tipoff=list(no_tipoff),
        undetermined=list(undetermined), mismatches=list(mismatches),
        blob_key=f"{season}_x.json.gz",
    )


def test_clean_run_reports_and_exits_zero(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)
    monkeypatch.setattr(schedule, "ingest_season", lambda con, season: _result(season))
    out = runner.invoke(cli.app, ["ingest-schedule", "--season", "2024-25"])
    assert out.exit_code == 0, out.output
    assert "2024-25: 1,230 games saved" in out.output


def test_default_season_is_the_current_one(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)
    seen = []

    def fake(con, season):
        seen.append(season)
        return _result(season)

    monkeypatch.setattr(schedule, "ingest_season", fake)
    runner.invoke(cli.app, ["ingest-schedule"])
    assert seen == [config.season_label(datetime.now(UTC))]


def test_empty_upcoming_season_falls_back_to_the_previous_one(tmp_path, monkeypatch):
    # July-August: next season's schedule is not published yet. Keep
    # refreshing the previous one so `status` does not cry wolf for weeks.
    _point_settings_at_tmp(tmp_path, monkeypatch)
    seen = []

    def fake(con, season):
        seen.append(season)
        return _result(season, written=0 if len(seen) == 1 else 1400)

    monkeypatch.setattr(schedule, "ingest_season", fake)
    out = runner.invoke(cli.app, ["ingest-schedule"])
    current = config.season_label(datetime.now(UTC))
    assert seen == [current, config.previous_season_label(current)]
    assert out.exit_code == 0, out.output
    assert "not published yet" in out.output


def test_explicit_season_with_no_games_does_not_fall_back(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)
    seen = []

    def fake(con, season):
        seen.append(season)
        return _result(season, written=0)

    monkeypatch.setattr(schedule, "ingest_season", fake)
    runner.invoke(cli.app, ["ingest-schedule", "--season", "2030-31"])
    assert seen == ["2030-31"]


def test_tbd_and_undecided_games_are_explained_not_failed(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)
    monkeypatch.setattr(
        schedule, "ingest_season",
        lambda con, season: _result(season, no_tipoff=["a", "b"], undetermined=["c"]),
    )
    out = runner.invoke(cli.app, ["ingest-schedule", "--season", "2026-27"])
    assert out.exit_code == 0, out.output
    assert "2 game(s) have no tip-off time yet" in out.output
    assert "1 game(s) left out because their teams are not decided yet" in out.output


def test_mismatch_warns_and_exits_nonzero(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)
    monkeypatch.setattr(
        schedule, "ingest_season",
        lambda con, season: _result(season, mismatches=["0022400561: schedule says ..."]),
    )
    out = runner.invoke(cli.app, ["ingest-schedule", "--season", "2024-25"])
    assert out.exit_code == 1
    assert "WARNING" in out.output and "disagree" in out.output


def test_network_failure_is_plain_english(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)

    def down(con, season):
        raise requests.ConnectionError("Name or service not known")

    monkeypatch.setattr(schedule, "ingest_season", down)
    out = runner.invoke(cli.app, ["ingest-schedule", "--season", "2024-25"])
    assert out.exit_code == 1
    assert "Could not download the NBA schedule" in out.output
    assert "Traceback" not in out.output


def test_unreadable_payload_is_plain_english(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)

    def bad(con, season):
        raise ValueError("the NBA schedule response for 2024-25 was not in the expected shape")

    monkeypatch.setattr(schedule, "ingest_season", bad)
    out = runner.invoke(cli.app, ["ingest-schedule", "--season", "2024-25"])
    assert out.exit_code == 1
    assert "archived" in out.output and "could not be read" in out.output
