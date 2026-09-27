"""CLI-level test for I5: `predictor status` must exit non-zero on PROBLEMS.

Every other command in this CLI exits 1 on a problem; `status` -- the one
command whose whole purpose is health reporting -- did not, so it could
never be wired into an external monitor/cron job to alert on staleness (it
had to be read by a human every time). The output TEXT is unchanged by this
fix; only the exit code is new. Mirrors tests/test_nba_cli.py's pattern for
patching settings so the CLI command uses a temp DB, never the real one.
"""

from typer.testing import CliRunner

from predictor import cli, config, db
from predictor.config import Settings

runner = CliRunner()


def _point_settings_at_tmp(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(config, "settings", s)
    monkeypatch.setattr(db, "settings", s)
    return s


def test_status_exits_nonzero_when_a_source_is_stale(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)
    # A brand-new, empty temp database: every point-in-time table has no
    # rows at all, which check_sources() always reports as stale.
    result = runner.invoke(cli.app, ["status"])

    assert result.exit_code == 1
    assert "PROBLEMS" in result.stdout


def test_status_exits_zero_when_every_source_is_fresh(tmp_path, monkeypatch):
    s = _point_settings_at_tmp(tmp_path, monkeypatch)
    con = db.connect(s.db_path)
    db.migrate(con)

    from datetime import UTC, datetime

    now = datetime.now(UTC)
    con.execute(
        "INSERT INTO games_raw (game_id, season, game_date, home_team,"
        " away_team, status, observed_at) VALUES (?,?,?,?,?,?,?)",
        ["g1", "2024-25", now.date(), "LAL", "BOS", "SCHEDULED", now],
    )
    con.execute(
        "INSERT INTO injury_status_raw (report_date, game_date, team, player,"
        " status, observed_at) VALUES (?,?,?,?,?,?)",
        [now.date(), now.date(), "LAL", "someone", "Out", now],
    )
    con.execute(
        "INSERT INTO odds_snapshots_raw (game_key, book, home_team, away_team,"
        " observed_at) VALUES (?,?,?,?,?)",
        ["g1", "draftkings", "LAL", "BOS", now],
    )
    con.execute(
        "INSERT INTO news_items_raw (item_key, feed, observed_at) VALUES (?,?,?)",
        ["n1", "rss", now],
    )
    con.execute(
        "INSERT INTO schedule_raw (game_id, season, game_date, tip_off_utc,"
        " home_team, away_team, is_neutral_reported, is_neutral, observed_at)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        ["0022400561", "2024-25", now.date(), None, "PHI", "NYK", False, False, now],
    )
    con.close()

    result = runner.invoke(cli.app, ["status"])

    assert result.exit_code == 0
    assert "ALL OK" in result.stdout
