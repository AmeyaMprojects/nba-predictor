"""CLI-level tests for `ingest-season`'s exit-code/warning logic.

Mirrors tests/test_injury_cli.py's pattern for backfill-injuries: stub out
nba_stats.ingest_season entirely (the pairing/ingestion logic itself is
covered exhaustively in test_nba_stats.py), and only exercise the CLI's own
decision of when a season did NOT fully succeed -- Task 10 review Finding
"a dropped game reaches the operator only via print(), never the return
value or the exit code". ingest_season_cmd must behave like its siblings
backfill-injuries/reingest-injuries: echo an explicit WARNING and exit
non-zero whenever anything was dropped, and stay quiet with exit 0 for a
clean run.
"""

from typer.testing import CliRunner

from predictor import cli, config, db
from predictor.config import Settings
from predictor.sources import nba_stats

runner = CliRunner()


def _point_settings_at_tmp(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    # cli.py does `from predictor.config import settings` freshly inside the
    # command body, so patching the module attribute is enough for that
    # call site; db.connect() references its own module-level `settings`
    # name bound at db.py's import time, so that needs patching separately
    # -- same pattern tests/test_injury_cli.py uses.
    monkeypatch.setattr(config, "settings", s)
    monkeypatch.setattr(db, "settings", s)
    return s


def test_cli_exits_nonzero_and_warns_when_season_has_dropped_games(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)

    def fake_ingest_season(con, season, observed_at=None, dropped=None):
        if dropped is not None:
            dropped.append(
                nba_stats.DroppedGame(
                    game_id="0099999999",
                    matchup="GARBLED TEXT, GARBLED TEXT",
                    reason="could not parse MATCHUP into home/away teams",
                )
            )
        return 1400

    monkeypatch.setattr(nba_stats, "ingest_season", fake_ingest_season)

    result = runner.invoke(cli.app, ["ingest-season", "2024-25"])

    assert result.exit_code == 1
    assert "WARNING" in result.stdout
    assert "0099999999" in result.stdout
    assert "ingested 1400 games for 2024-25" in result.stdout


def test_cli_exits_zero_and_quiet_for_a_clean_season(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)

    def fake_ingest_season(con, season, observed_at=None, dropped=None):
        return 1401

    monkeypatch.setattr(nba_stats, "ingest_season", fake_ingest_season)

    result = runner.invoke(cli.app, ["ingest-season", "2024-25"])

    assert result.exit_code == 0
    assert "WARNING" not in result.stdout
    assert "ingested 1401 games for 2024-25" in result.stdout
