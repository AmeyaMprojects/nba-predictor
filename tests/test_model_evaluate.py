"""Walk-forward evaluation, `evaluate-model`, and the winner-based
`fit-model` (spec 3, "Calibration redesign -- decided 2026-10-02"), Task 3
of the walk-forward recalibration.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from model_fixtures import build_history, fixture_con
from predictor import cli, config, db
from predictor.config import Settings
from predictor.model import evaluate as evaluate_mod
from predictor.model import settings as ms
from predictor.model import tuning
from real_archive import open_real_archive_or_skip


def test_fold_job_takes_every_earlier_tuning_season():
    job = evaluate_mod.fold_job("2024-25", None)
    assert job.seasons == ("2019-20", "2020-21", "2021-22", "2022-23", "2023-24")
    assert job.half_life is None


def test_fold_job_rejects_the_first_tuning_season():
    """2019-20 is the first tuning season -- it has no earlier tuning
    season to fold on, so it is not a walk-forward season at all."""
    with pytest.raises(ValueError, match="2019-20"):
        evaluate_mod.fold_job("2019-20", None)


def test_no_future_leakage_from_a_later_season(tmp_path):
    """Corrupting every 2024-25 FINAL score must not move any fold Choice
    fit on seasons strictly before 2024-25 (2020-21 through 2024-25's own
    fold, which trains only on 2019-20..2023-24), nor the predictions made
    FOR 2024-25 itself. `games_per_season=2` puts both of 2024-25's games
    on the SAME calendar day, so their pre-game numbers are fixed (from
    ratings carried out of 2023-24) before either game's result --
    corrupted or not -- is applied; corrupting them can only move ratings
    carried FORWARD into 2025-26. So the fold Choice that TRAINS on
    2024-25 (the one that predicts 2025-26) must change -- proving the
    corruption actually did something -- while every earlier fold, and
    2024-25's own predictions, must not.
    """
    con = fixture_con(tmp_path)
    build_history(con, games_per_season=2)
    before = evaluate_mod.evaluate(con)

    games_table = db.POINT_IN_TIME_TABLES["games"]
    con.execute(
        f"UPDATE {games_table} SET home_points = 200, away_points = 1 "
        "WHERE season = '2024-25' AND status = 'FINAL'"
    )
    after = evaluate_mod.evaluate(con)

    unaffected = evaluate_mod.WALK_FORWARD_SEASONS[:-1]  # 2020-21 .. 2024-25
    assert unaffected[-1] == "2024-25"
    for v_before, v_after in zip(before.variants, after.variants, strict=True):
        for season in unaffected:
            assert v_after.choices[season] == v_before.choices[season]
        before_2425 = [p.p_home for p in v_before.predictions if p.season == "2024-25"]
        after_2425 = [p.p_home for p in v_after.predictions if p.season == "2024-25"]
        assert after_2425 == before_2425
        assert v_after.choices["2025-26"] != v_before.choices["2025-26"]


def test_pick_winner_takes_the_third_variant_when_it_clears_the_tolerance():
    """0.6480 < 0.6500 - 0.001 (= 0.6490) -- the third variant wins."""
    variants = [
        evaluate_mod.VariantResult(hl, [], loss, 0.0, 0.0, {})
        for hl, loss in zip(tuning.HALF_LIVES, (0.6500, 0.6495, 0.6480), strict=True)
    ]
    assert evaluate_mod.pick_winner(variants).half_life == 1.0


def test_pick_winner_keeps_the_first_variant_when_no_later_one_clears_the_tolerance():
    """0.6492 is not below 0.6500 - 0.001 (= 0.6490) -- the first variant
    (equal weight) keeps its lead even though later variants have a lower
    raw log loss."""
    variants = [
        evaluate_mod.VariantResult(hl, [], loss, 0.0, 0.0, {})
        for hl, loss in zip(tuning.HALF_LIVES, (0.6500, 0.6495, 0.6492), strict=True)
    ]
    assert evaluate_mod.pick_winner(variants).half_life is None


def test_evaluate_is_deterministic_and_final_matches_the_winner(tmp_path):
    con = fixture_con(tmp_path)
    build_history(con)
    a = evaluate_mod.evaluate(con)
    b = evaluate_mod.evaluate(con)
    assert a == b
    assert a.final.half_life == a.winner.half_life


def test_format_evaluation_has_every_required_section(tmp_path):
    con = fixture_con(tmp_path)
    build_history(con)
    ev = evaluate_mod.evaluate(con)
    text = evaluate_mod.format_evaluation(ev)

    for v in ev.variants:
        assert evaluate_mod.variant_label(v.half_life) in text
    assert "Chosen:" in text
    for season in evaluate_mod.WALK_FORWARD_SEASONS:
        assert season in text
    assert "Final settings for 2026-27" in text
    assert "evaluated season" in text
    assert "Calibration -- when it said X%, how often did that happen?" in text


def test_evaluate_model_cli_prints_the_evaluation(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(config, "settings", s)
    monkeypatch.setattr(db, "settings", s)
    con = db.connect()
    db.migrate(con)
    build_history(con)
    con.close()

    result = CliRunner().invoke(cli.app, ["evaluate-model"])
    assert result.exit_code == 0, result.output
    assert "Walk-forward evaluation" in result.output
    assert "Chosen:" in result.output


def test_fit_model_cli_saves_the_winners_half_life(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(config, "settings", s)
    monkeypatch.setattr(db, "settings", s)
    con = db.connect()
    db.migrate(con)
    build_history(con)
    con.close()
    out_path = tmp_path / "stage1_settings.json"
    monkeypatch.setattr(ms, "SETTINGS_PATH", out_path)

    result = CliRunner().invoke(cli.app, ["fit-model"])
    assert result.exit_code == 0, result.output
    assert "Chosen:" in result.output
    saved = ms.load(out_path)

    con2 = db.connect(read_only=True)
    try:
        ev = evaluate_mod.evaluate(con2)
    finally:
        con2.close()
    assert saved.half_life == ev.winner.half_life
    assert evaluate_mod.variant_label(ev.winner.half_life) in result.output


@pytest.mark.slow
def test_evaluate_reproduces_the_committed_settings_from_the_real_archive():
    """A published number must trace to settings anyone can re-derive."""
    con = open_real_archive_or_skip()
    try:
        assert evaluate_mod.evaluate(con).final == ms.load()
    finally:
        con.close()
