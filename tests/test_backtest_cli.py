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
        return [], replay_mod.ReplayStats(
            considered=0, predicted=0, skipped_conflicting_metadata=0,
            skipped_buffer_too_early=0,
            skipped_no_tipoff=0,
            considered_by_season={},
            skipped_no_tipoff_by_season={},
            skipped_no_result=0, skipped_score_missing=0,
            skipped_result_visible=0, declined=0, failed=0,
        )

    monkeypatch.setattr(replay_mod, "replay", empty)
    result = runner.invoke(cli.app, ["backtest", "--season", "1999-00"])
    assert result.exit_code != 0
    assert "no games" in result.stdout.lower() or "nothing to score" in result.stdout.lower()


def test_backtest_hints_at_known_seasons_when_a_season_filter_matches_nothing(
    tmp_path, monkeypatch
):
    # FIX 12(a): a mistyped --season used to print all zeros with no hint
    # the season string itself was wrong. The temp db has real 2023-24 data
    # ingested (via a direct insert), so filtering on a season that does
    # not exist must name the season(s) that DO.
    s = _migrated_db(tmp_path, monkeypatch)
    from datetime import UTC, date, datetime

    from predictor import db

    con = db.connect(s.db_path)
    g = db.POINT_IN_TIME_TABLES["games"]
    con.execute(
        f"INSERT INTO {g} (game_id, season, game_date, home_team, away_team,"
        " home_points, away_points, status, observed_at, reconstructed)"
        " VALUES (?,?,?,?,?,?,?,?,?,TRUE)",
        ["0022300001", "2023-24", date(2024, 1, 1), "PHI", "NYK",
         110, 100, "FINAL", datetime(2024, 1, 2, 12, 0, tzinfo=UTC)],
    )
    con.close()

    result = runner.invoke(cli.app, ["backtest", "--season", "2024-2025"])
    assert result.exit_code != 0
    assert "2024-2025" in result.stdout
    assert "2023-24" in result.stdout


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


def test_backtest_reports_a_lock_conflict_in_plain_english_not_a_missing_database(
    tmp_path, monkeypatch
):
    """FIX 18 (final review, part 3): a lock conflict (e.g. the scheduled
    poll-news job holding the write lock) must not be reported as "No
    database found" -- that remedy ("run predictor ingest-season") needs
    the same lock, so it sends the user to a command that cannot work
    either. Reproduced cross-process: a real duckdb.IOException for a lock
    conflict looks different from one for a missing file only in its
    message text, not its exception class, so both must be checked for
    real rather than assumed.
    """
    import subprocess
    import sys
    import time

    import duckdb

    s = _migrated_db(tmp_path, monkeypatch)

    holder = subprocess.Popen(
        [
            sys.executable, "-c",
            "import duckdb, time, sys\n"
            "con = duckdb.connect(sys.argv[1])\n"
            "con.execute('CREATE TABLE IF NOT EXISTS lock_holder(x INT)')\n"
            "time.sleep(8)\n",
            str(s.db_path),
        ],
    )
    try:
        # Poll for the lock to actually be held, rather than a fixed sleep
        # -- avoids flakiness on a slow CI machine.
        deadline = time.time() + 5
        locked = False
        while time.time() < deadline:
            try:
                probe = duckdb.connect(str(s.db_path), read_only=True)
                probe.close()
            except duckdb.Error:
                locked = True
                break
            time.sleep(0.1)
        assert locked, "subprocess never acquired the database lock"

        result = runner.invoke(cli.app, ["backtest"])
        assert result.exit_code != 0
        out = result.stdout.lower()
        assert "ingest" not in out, (
            "sent the user to a command that needs the same lock"
        )
        assert "wait" in out
        assert "Traceback" not in result.stdout
    finally:
        holder.terminate()
        holder.wait(timeout=5)
