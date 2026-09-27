from datetime import UTC, datetime

import duckdb
import requests
from typer.testing import CliRunner

from predictor import cli, config, db, raw_store
from predictor.config import Settings
from predictor.sources import schedule

runner = CliRunner()

FETCHED = datetime(2026, 9, 27, 14, 30, tzinfo=UTC)


def _point_settings_at_tmp(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(config, "settings", s)
    monkeypatch.setattr(db, "settings", s)
    monkeypatch.setattr(raw_store, "settings", s)
    return s


def _fetched(season, rows=1230, no_tipoff=(), undetermined=()):
    return schedule.Fetched(
        season=season, fetched_at=FETCHED, blob_key=f"{season}_x.json.gz",
        parsed=schedule.ParseResult(
            rows=[object()] * rows, no_tipoff=list(no_tipoff),
            undetermined=list(undetermined),
        ),
    )


def _load_ok(mismatches=()):
    def load(con, fetched):
        return schedule.IngestResult(
            season=fetched.season, written=len(fetched.parsed.rows),
            no_tipoff=fetched.parsed.no_tipoff,
            undetermined=fetched.parsed.undetermined,
            mismatches=list(mismatches), blob_key=fetched.blob_key,
        )

    return load


def _patch(monkeypatch, fetch, load=None):
    monkeypatch.setattr(schedule, "fetch_and_archive", fetch)
    monkeypatch.setattr(schedule, "load", load or _load_ok())


CURRENT = config.season_label(datetime.now(UTC))
PREVIOUS = config.previous_season_label(CURRENT)


def test_clean_run_reports_and_exits_zero(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)
    _patch(monkeypatch, lambda season: _fetched(season))
    out = runner.invoke(cli.app, ["ingest-schedule", "--season", "2024-25"])
    assert out.exit_code == 0, out.output
    assert "2024-25: 1,230 games saved" in out.output


def test_default_season_is_the_current_one(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)
    seen = []

    def fake(season):
        seen.append(season)
        return _fetched(season)

    _patch(monkeypatch, fake)
    runner.invoke(cli.app, ["ingest-schedule"])
    assert seen == [CURRENT]


def test_database_is_not_opened_until_every_download_is_done(tmp_path, monkeypatch):
    # Final-review Fix 1: a slow, retried download must never hold the
    # DuckDB write lock that the unattended news job needs.
    _point_settings_at_tmp(tmp_path, monkeypatch)
    events = []
    real_connect = db.connect_with_retry

    def fake_fetch(season):
        events.append(f"fetch {season}")
        if season == CURRENT:
            raise schedule.ScheduleUnavailable("not published")
        return _fetched(season)

    def recording_connect(*args, **kwargs):
        events.append("connect")
        return real_connect(*args, **kwargs)

    def fake_load(con, fetched):
        events.append(f"load {fetched.season}")
        return _load_ok()(con, fetched)

    monkeypatch.setattr(db, "connect_with_retry", recording_connect)
    _patch(monkeypatch, fake_fetch, fake_load)
    out = runner.invoke(cli.app, ["ingest-schedule"])
    assert out.exit_code == 0, out.output
    assert events == [f"fetch {CURRENT}", f"fetch {PREVIOUS}", "connect", f"load {PREVIOUS}"]


def test_empty_upcoming_season_falls_back_to_the_previous_one(tmp_path, monkeypatch):
    # July-August: next season's schedule is not published yet. Keep
    # refreshing the previous one so `status` does not cry wolf for weeks.
    _point_settings_at_tmp(tmp_path, monkeypatch)
    seen = []

    def fake(season):
        seen.append(season)
        return _fetched(season, rows=0 if len(seen) == 1 else 1400)

    loaded = []

    def load(con, fetched):
        loaded.append(fetched.season)
        return _load_ok()(con, fetched)

    _patch(monkeypatch, fake, load)
    out = runner.invoke(cli.app, ["ingest-schedule"])
    assert seen == [CURRENT, PREVIOUS]
    assert loaded == [PREVIOUS]
    assert out.exit_code == 0, out.output
    assert f"The {CURRENT} schedule is not available yet" in out.output
    assert f"refreshing {PREVIOUS} instead" in out.output
    assert f"schedule {PREVIOUS}: 1,400 games saved" in out.output


def test_unpublished_upcoming_season_falls_back_to_the_previous_one(tmp_path, monkeypatch):
    # Measured live 2026-09-27: nba_api raises IndexError for an
    # unpublished season, which fetch_season_payload turns into this.
    _point_settings_at_tmp(tmp_path, monkeypatch)
    seen = []

    def fake(season):
        seen.append(season)
        if season == CURRENT:
            raise schedule.ScheduleUnavailable(
                f"the NBA has not published a {season} schedule"
            )
        return _fetched(season)

    _patch(monkeypatch, fake)
    out = runner.invoke(cli.app, ["ingest-schedule"])
    assert seen == [CURRENT, PREVIOUS]
    assert out.exit_code == 0, out.output
    assert f"The {CURRENT} schedule is not available yet" in out.output
    assert "has not published" in out.output
    assert "Traceback" not in out.output


def test_network_failure_on_upcoming_season_falls_back(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)
    seen = []

    def fake(season):
        seen.append(season)
        if season == CURRENT:
            raise requests.ConnectionError("boom")
        return _fetched(season)

    _patch(monkeypatch, fake)
    out = runner.invoke(cli.app, ["ingest-schedule"])
    assert seen == [CURRENT, PREVIOUS]
    assert out.exit_code == 0, out.output


def test_fallback_happens_at_most_once(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)
    seen = []

    def fake(season):
        seen.append(season)
        raise schedule.ScheduleUnavailable(f"the NBA has not published a {season} schedule")

    _patch(monkeypatch, fake)
    out = runner.invoke(cli.app, ["ingest-schedule"])
    assert seen == [CURRENT, PREVIOUS]
    assert out.exit_code == 1
    assert "Traceback" not in out.output
    assert f"has not published a {PREVIOUS} schedule" in out.output


def test_previous_season_network_failure_after_fallback_is_plain_english(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)

    def fake(season):
        if season == CURRENT:
            raise schedule.ScheduleUnavailable("not published")
        raise requests.ConnectionError("Name or service not known")

    _patch(monkeypatch, fake)
    out = runner.invoke(cli.app, ["ingest-schedule"])
    assert out.exit_code == 1
    assert f"Could not download the NBA schedule for {PREVIOUS}" in out.output
    assert "Traceback" not in out.output


def test_explicit_season_with_no_games_does_not_fall_back(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)
    seen = []

    def fake(season):
        seen.append(season)
        return _fetched(season, rows=0)

    _patch(monkeypatch, fake)
    runner.invoke(cli.app, ["ingest-schedule", "--season", "2030-31"])
    assert seen == ["2030-31"]


def test_explicit_unpublished_season_does_not_fall_back(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)
    seen = []

    def fake(season):
        seen.append(season)
        raise schedule.ScheduleUnavailable(f"the NBA has not published a {season} schedule")

    _patch(monkeypatch, fake)
    out = runner.invoke(cli.app, ["ingest-schedule", "--season", "2030-31"])
    assert seen == ["2030-31"]
    assert out.exit_code == 1
    assert "has not published a 2030-31 schedule" in out.output
    assert "Traceback" not in out.output


def test_tbd_and_undecided_games_are_explained_not_failed(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)
    _patch(
        monkeypatch,
        lambda season: _fetched(season, no_tipoff=["a", "b"], undetermined=["c"]),
    )
    out = runner.invoke(cli.app, ["ingest-schedule", "--season", "2026-27"])
    assert out.exit_code == 0, out.output
    assert "2 game(s) have no tip-off time yet" in out.output
    assert "1 game(s) left out because their teams are not decided yet" in out.output


def test_mismatch_warns_and_exits_nonzero(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)
    _patch(
        monkeypatch,
        lambda season: _fetched(season),
        _load_ok(mismatches=["0022400561: schedule says ..."]),
    )
    out = runner.invoke(cli.app, ["ingest-schedule", "--season", "2024-25"])
    assert out.exit_code == 1
    assert "WARNING" in out.output and "disagree" in out.output


def test_network_failure_is_plain_english(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)

    def down(season):
        raise requests.ConnectionError("Name or service not known")

    _patch(monkeypatch, down)
    out = runner.invoke(cli.app, ["ingest-schedule", "--season", "2024-25"])
    assert out.exit_code == 1
    assert "Could not download the NBA schedule" in out.output
    assert "Traceback" not in out.output


def test_unreadable_payload_is_plain_english(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)

    def bad(season):
        raise ValueError("the NBA schedule response for 2024-25 was not in the expected shape")

    _patch(monkeypatch, bad)
    out = runner.invoke(cli.app, ["ingest-schedule", "--season", "2024-25"])
    assert out.exit_code == 1
    assert "archived" in out.output and "could not be read" in out.output


def test_archive_conflict_is_plain_english(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)

    def conflict(season):
        raise raw_store.RawStoreConflict("key 2024-25_x.json.gz already holds different bytes")

    _patch(monkeypatch, conflict)
    out = runner.invoke(cli.app, ["ingest-schedule", "--season", "2024-25"])
    assert out.exit_code == 1
    assert "already archived" in out.output
    assert "Traceback" not in out.output


def test_database_error_while_loading_is_plain_english(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)

    def broken_load(con, fetched):
        raise duckdb.ConstraintException("NOT NULL constraint failed")

    _patch(monkeypatch, lambda season: _fetched(season), broken_load)
    out = runner.invoke(cli.app, ["ingest-schedule", "--season", "2024-25"])
    assert out.exit_code == 1
    assert "Could not save the 2024-25 schedule to the database" in out.output
    assert "Traceback" not in out.output


def test_database_error_while_migrating_is_plain_english(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)

    def broken_migrate(con):
        raise duckdb.IOException("disk full")

    monkeypatch.setattr(db, "migrate", broken_migrate)
    _patch(monkeypatch, lambda season: _fetched(season))
    out = runner.invoke(cli.app, ["ingest-schedule", "--season", "2024-25"])
    assert out.exit_code == 1
    assert "Could not save the schedule to the database" in out.output
    assert "Traceback" not in out.output


def test_locked_database_is_plain_english(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)

    def locked(*args, **kwargs):
        raise duckdb.IOException("Could not set lock on file: Conflicting lock is held")

    monkeypatch.setattr(db, "connect_with_retry", locked)
    _patch(monkeypatch, lambda season: _fetched(season))
    out = runner.invoke(cli.app, ["ingest-schedule", "--season", "2024-25"])
    assert out.exit_code == 1
    assert "Could not open the database to save the schedule" in out.output
    assert "Traceback" not in out.output


def test_unreadable_current_season_is_plain_english_not_a_fallback(tmp_path, monkeypatch):
    # Only "not published" / network / zero games fall back; a payload that
    # cannot be read is a real problem and must be reported, not papered over.
    _point_settings_at_tmp(tmp_path, monkeypatch)
    seen = []

    def bad(season):
        seen.append(season)
        raise ValueError("not in the expected shape")

    _patch(monkeypatch, bad)
    out = runner.invoke(cli.app, ["ingest-schedule"])
    assert seen == [CURRENT]
    assert out.exit_code == 1
    assert "could not be read" in out.output
    assert "Traceback" not in out.output
    assert isinstance(out.exception, SystemExit)


def test_archive_conflict_on_current_season_is_plain_english(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)

    def conflict(season):
        raise raw_store.RawStoreConflict("different bytes")

    _patch(monkeypatch, conflict)
    out = runner.invoke(cli.app, ["ingest-schedule"])
    assert out.exit_code == 1
    assert "already archived" in out.output
    assert isinstance(out.exception, SystemExit)
