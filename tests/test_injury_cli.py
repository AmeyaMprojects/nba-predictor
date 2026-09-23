"""CLI-level tests for `backfill-injuries`'s exit-code/warning logic.

These stub out `injury_report.backfill_range` entirely -- the sweep logic
itself is covered exhaustively in test_injury_backfill.py -- so these tests
only exercise the CLI's own decision of what counts as "did not fully
succeed" (Task 9 review Finding 1: `bad_content` must be included in that
decision, same as `transient`/`parse_failed`).
"""

from typer.testing import CliRunner

from predictor import cli, config, db
from predictor.config import Settings
from predictor.sources import injury_report

runner = CliRunner()


def _empty_stats(**overrides):
    stats = {
        "fetched": 0,
        "skipped": 0,
        "missing": 0,
        "ingested": 0,
        "transient": 0,
        "parse_failed": 0,
        "bad_content": 0,
    }
    stats.update(overrides)
    return stats


def _point_settings_at_tmp(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    # cli.py does `from predictor.config import settings` freshly inside the
    # command body, so patching the module attribute is enough for that
    # call site; db.connect() references its own module-level `settings`
    # name bound at db.py's import time, so that needs patching separately
    # -- same pattern the injury source tests use for raw_store.settings.
    monkeypatch.setattr(config, "settings", s)
    monkeypatch.setattr(db, "settings", s)
    return s


def test_cli_exits_nonzero_when_only_bad_content_is_nonzero(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)
    monkeypatch.setattr(
        injury_report, "backfill_range", lambda *a, **k: _empty_stats(bad_content=2)
    )

    result = runner.invoke(
        cli.app,
        ["backfill-injuries", "--start", "2025-01-15", "--end", "2025-01-15"],
    )

    assert result.exit_code == 1
    assert "bad_content" in result.stdout
    assert "RUN DID NOT FULLY SUCCEED" in result.stdout


def test_cli_exits_zero_when_all_buckets_are_clean(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)
    monkeypatch.setattr(
        injury_report,
        "backfill_range",
        lambda *a, **k: _empty_stats(fetched=1, ingested=5, missing=1),
    )

    result = runner.invoke(
        cli.app,
        ["backfill-injuries", "--start", "2025-01-15", "--end", "2025-01-15"],
    )

    assert result.exit_code == 0
    assert "RUN DID NOT FULLY SUCCEED" not in result.stdout


def test_cli_reingest_injuries_exits_nonzero_when_reports_still_fail(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)
    monkeypatch.setattr(
        injury_report,
        "reingest_archived",
        lambda *a, **k: {
            "found": 2,
            "ingested_ok": 1,
            "still_failed": 1,
            "rows_written": 50,
        },
    )

    result = runner.invoke(cli.app, ["reingest-injuries"])

    assert result.exit_code == 1
    assert "still failed to parse: 1" in result.stdout


def test_cli_reingest_injuries_exits_zero_when_all_succeed(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)
    monkeypatch.setattr(
        injury_report,
        "reingest_archived",
        lambda *a, **k: {
            "found": 2,
            "ingested_ok": 2,
            "still_failed": 0,
            "rows_written": 300,
        },
    )

    result = runner.invoke(cli.app, ["reingest-injuries"])

    assert result.exit_code == 0
