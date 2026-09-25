from typer.testing import CliRunner

from predictor import cli, config, db
from predictor.config import Settings

runner = CliRunner()


def _point_settings_at_tmp(tmp_path, monkeypatch):
    # Same pattern as tests/test_nba_cli.py / tests/test_status_cli.py: `cli.py`
    # does `from predictor.config import settings` freshly inside the command
    # body, so patching the module attribute is enough for that call site;
    # `db.connect()` references its own module-level `settings` name bound at
    # db.py's import time, so that needs patching separately.
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(config, "settings", s)
    monkeypatch.setattr(db, "settings", s)
    return s


def _migrated_db(tmp_path, monkeypatch):
    # FIX 1(c): `predictor backtest` opens the database read-only and no
    # longer migrates it, so a test that reaches db.connect() needs one to
    # already exist -- unlike the older CLI tests, which could rely on the
    # command's own db.migrate(con) call to create it on the fly.
    s = _point_settings_at_tmp(tmp_path, monkeypatch)
    con = db.connect(s.db_path)
    db.migrate(con)
    con.close()
    return s


def test_backtest_rejects_an_unknown_model_in_plain_english(tmp_path, monkeypatch):
    _migrated_db(tmp_path, monkeypatch)
    result = runner.invoke(cli.app, ["backtest", "--model", "not-a-model"])
    assert result.exit_code != 0
    assert "not-a-model" in result.stdout
    assert "always-home" in result.stdout  # tells them what IS available


def test_backtest_reports_nothing_to_score_rather_than_crashing(tmp_path, monkeypatch):
    _migrated_db(tmp_path, monkeypatch)
    from predictor.backtest import replay as replay_mod

    def empty(*args, **kwargs):
        return [], replay_mod.ReplayStats(0, 0, 0, 0, 0, 0, 0, 0)

    monkeypatch.setattr(replay_mod, "replay", empty)
    result = runner.invoke(cli.app, ["backtest", "--season", "1999-00"])
    assert result.exit_code != 0
    assert "no games" in result.stdout.lower() or "nothing to score" in result.stdout.lower()


def test_backtest_rejects_a_negative_buffer_in_plain_english(tmp_path, monkeypatch):
    _migrated_db(tmp_path, monkeypatch)
    result = runner.invoke(cli.app, ["backtest", "--buffer-minutes", "-5", "--season", "2023-24"])
    assert result.exit_code != 0
    assert "buffer_minutes must be >= 0, got -5" in result.stdout
    assert "Traceback" not in result.stdout


def test_backtest_tells_the_user_to_ingest_first_when_no_database_exists(tmp_path, monkeypatch):
    # No _migrated_db() call -- the temp data dir is empty, so there is no
    # predictor.duckdb file at all for the read-only connection to open.
    _point_settings_at_tmp(tmp_path, monkeypatch)
    result = runner.invoke(cli.app, ["backtest"])
    assert result.exit_code != 0
    assert "ingest" in result.stdout.lower()
    assert "Traceback" not in result.stdout
