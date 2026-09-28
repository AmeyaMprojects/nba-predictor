"""`predictor backtest --model stage1` end to end, on a fixture archive.

Also closes harness open item 3: the BEATS / TOO CLOSE TO CALL verdict
branch had never run from the CLI, because both shipped baselines put every
game on the home side. This drives it with a predictor that disagrees.
"""

from datetime import date, timedelta

import pytest
from typer.testing import CliRunner

from model_fixtures import add_game
from predictor import cli, config, db
from predictor.backtest import replay
from predictor.config import Settings
from predictor.model import settings as ms
from predictor.model import stage1
from predictor.model.adjustments import Coefficients
from predictor.model.ratings import RatingParams
from real_archive import open_real_archive_or_skip

runner = CliRunner()

SETTINGS = ms.ModelSettings(
    ratings=RatingParams(0.1, 20.0, 0.5, 100),
    coefficients=Coefficients(-1.0, -0.5, -0.3, -0.2, 1.0),
    sigma=13.0, fit_games=1, calibrate_games=1,
)


def _archive(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(config, "settings", s)
    monkeypatch.setattr(db, "settings", s)
    con = db.connect()
    db.migrate(con)
    # warm-up game, then a test-season run where the AWAY team always wins
    add_game(con, "0021800001", "2018-19", date(2019, 3, 1), "PHI", "NYK", 120, 100,
             city="Philadelphia")
    for i in range(40):
        d = date(2023, 11, 1) + timedelta(days=2 * i)
        add_game(con, f"00223{i:05d}", "2023-24", d, "NYK", "PHI", 90, 110, city="New York")
    con.close()
    return s


def test_stage1_headline_is_test_seasons_only(tmp_path, monkeypatch):
    _archive(tmp_path, monkeypatch)
    monkeypatch.setattr(ms, "load", lambda path=None: SETTINGS)
    out = runner.invoke(cli.app, ["backtest", "--model", "stage1"])
    assert out.exit_code == 0, out.output
    assert "test seasons 2023-24, 2024-25, 2025-26 only" in out.output
    assert "By season" in out.output
    assert "2018-19  warm-up" in out.output
    assert "Example explanations" in out.output
    assert " at " in out.output and "% to win)" in out.output


def test_verdict_branch_runs_from_the_cli(tmp_path, monkeypatch):
    _archive(tmp_path, monkeypatch)
    monkeypatch.setattr(ms, "load", lambda path=None: SETTINGS)

    class AwayPicker(stage1.Stage1Predictor):
        def __call__(self, game, view):
            super().__call__(game, view)
            return 0.2  # always picks the away team, which always wins here

    monkeypatch.setattr(stage1, "Stage1Predictor", AwayPicker)
    out = runner.invoke(cli.app, ["backtest", "--model", "stage1"])
    assert out.exit_code == 0, out.output
    assert "BEATS always-pick-home" in out.output


def test_missing_settings_is_a_plain_error(tmp_path, monkeypatch):
    _archive(tmp_path, monkeypatch)

    def missing(path=None):
        raise ms.SettingsError("No fitted model settings found at x. Run: predictor fit-model")

    monkeypatch.setattr(ms, "load", missing)
    out = runner.invoke(cli.app, ["backtest", "--model", "stage1"])
    assert out.exit_code == 1
    assert "predictor fit-model" in out.output
    assert "Traceback" not in out.output


def test_no_test_season_games_is_a_plain_error(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(config, "settings", s)
    monkeypatch.setattr(db, "settings", s)
    con = db.connect()
    db.migrate(con)
    add_game(con, "0021800001", "2018-19", date(2019, 3, 1), "PHI", "NYK", 120, 100,
             city="Philadelphia")
    con.close()
    monkeypatch.setattr(ms, "load", lambda path=None: SETTINGS)
    out = runner.invoke(cli.app, ["backtest", "--model", "stage1"])
    assert out.exit_code == 1
    assert "No test-season games" in out.output


@pytest.mark.parametrize("season", ["2023-24", "2024-25", "2025-26"])
def test_every_test_season_game_is_predicted_and_every_explanation_adds_up(season):
    con = open_real_archive_or_skip()
    try:
        predictor = stage1.Stage1Predictor(con, ms.load())
        preds, stats = replay.replay(con, predictor, season=season)
        assert stats.considered == 1230
        assert stats.predicted == 1230, stats
        for p in preds:
            b = predictor.breakdowns[p.game_id]
            assert sum(v for _, v in b.terms()) == pytest.approx(b.spread, abs=1e-9)
            assert p.p_home == b.p_home
            shown = [round(v, 1) for _, v in b.terms()]
            assert f"by {abs(round(sum(shown), 1)):.1f}" in b.sentence() or "pick'em" in b.sentence()
    finally:
        con.close()
