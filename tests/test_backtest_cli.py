import pytest
from typer.testing import CliRunner

from predictor.cli import app

runner = CliRunner()


def test_backtest_rejects_an_unknown_model_in_plain_english():
    result = runner.invoke(app, ["backtest", "--model", "not-a-model"])
    assert result.exit_code != 0
    assert "not-a-model" in result.stdout
    assert "always-home" in result.stdout  # tells them what IS available


def test_backtest_reports_nothing_to_score_rather_than_crashing(monkeypatch):
    from predictor.backtest import replay as replay_mod

    def empty(*args, **kwargs):
        return [], replay_mod.ReplayStats(0, 0, 0, 0, 0, 0, 0)

    monkeypatch.setattr(replay_mod, "replay", empty)
    result = runner.invoke(app, ["backtest", "--season", "1999-00"])
    assert result.exit_code != 0
    assert "no games" in result.stdout.lower() or "nothing to score" in result.stdout.lower()
