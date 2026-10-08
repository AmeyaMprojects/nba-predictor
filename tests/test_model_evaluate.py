"""Walk-forward evaluation, `evaluate-model`, and the winner-based
`fit-model` (spec 3, "Calibration redesign -- decided 2026-10-02"), Task 3
of the walk-forward recalibration.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime

import pytest
from typer.testing import CliRunner

from model_fixtures import build_history, fixture_con
from predictor import cli, config, db
from predictor.backtest.replay import Prediction
from predictor.config import Settings
from predictor.model import evaluate as evaluate_mod
from predictor.model import settings as ms
from predictor.model import tuning
from predictor.model.market import MarketView
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


def test_progress_names_the_actual_grid_size(tmp_path):
    """The 'simulating N rating settings...' message must be derived from
    the grid sizes, not a hard-coded literal -- this pins today's actual
    grid size (10 x 5 x 6 x 3 = 900) so the message stays honest if any
    grid ever changes."""
    expected = (
        len(tuning.GRID_K) * len(tuning.GRID_CAP)
        * len(tuning.GRID_REGRESSION) * len(tuning.GRID_WINDOW)
    )
    assert expected == 900

    con = fixture_con(tmp_path)
    build_history(con)
    messages = []
    evaluate_mod.evaluate(con, progress=messages.append)
    assert "simulating 900 rating settings..." in messages


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
    assert "a later variant must beat it by more than 0.001" in text
    for season in evaluate_mod.WALK_FORWARD_SEASONS:
        assert season in text
    assert "Final settings for 2026-27" in text
    assert "the walk-forward above scores the method, not these exact settings" in text
    assert "evaluated season" in text
    assert "Calibration -- when it said X%, how often did that happen?" in text
    # Fix round 1 (controller ruling, 2026-10-02): the publishing-bar MET
    # verdict must never be read as the clean test -- this is walk-forward
    # on the tuning seasons; the real, held-out test is 2026-27 live.
    assert (
        "This is a walk-forward result on the tuning seasons, not a clean test: the "
        "method was chosen after a first look at 2023-26. The clean test, including "
        "any calibration claim, is 2026-27 predicted live."
    ) in text
    # The caveat must come AFTER the publishing bar's verdict, not before it
    # (a reader must see the MET/NOT MET line before the caveat that
    # qualifies it).
    assert text.index("Verdict:") < text.index("This is a walk-forward result")


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
    assert "  walk-forward: " in result.output
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


# --- Model vs market (market-odds Task 3) -------------------------------

_T = datetime(2024, 1, 1, tzinfo=UTC)


def _mg(p_model, p_market, home_won=True, *, model_spread=0.0, spread=None,
        margin=1, from_spread=False, season="2023-24", gid="g"):
    pred = Prediction(gid, season, date(2024, 1, 1), "BOS", "NYK", _T, _T,
                      p_model, home_won, True)
    view = MarketView(p_market, spread, 1, _T, from_spread)
    return evaluate_mod.MarketGame(pred, model_spread, margin, view)


def test_disagreement_zone_counts_by_hand():
    games = [
        _mg(0.60, 0.50, True),   # 0.10 apart exactly -> in; both pick home, both right
        _mg(0.55, 0.45, False),  # picks differ -> in; model wrong, market right
        _mg(0.70, 0.65, True),   # same pick, 0.05 apart -> out
        _mg(0.30, 0.45, False),  # 0.15 apart -> in; both pick away, both right
        _mg(0.52, 0.60, True),   # same pick, 0.08 apart -> out
    ]
    zone = evaluate_mod.disagreement_zone(games)
    assert zone == evaluate_mod.DisagreementZone(games=3, model_right=2, market_right=3)


def test_against_closing_spread_counts_by_hand():
    """Stored spread is the home spread (negative = home favoured); the
    margin the line expects is minus that."""
    games = [
        _mg(0.5, 0.5, model_spread=6.0, spread=-4.0, margin=5),    # model home side, home covered: right
        _mg(0.5, 0.5, model_spread=2.0, spread=-4.0, margin=10),   # model away side, home covered: wrong
        _mg(0.5, 0.5, model_spread=3.0, spread=-3.0, margin=10),   # same as the line: left out
        _mg(0.5, 0.5, model_spread=1.0, spread=2.0, margin=-2),    # push on the line: left out
        _mg(0.5, 0.5, model_spread=-5.0, spread=2.0, margin=-7),   # model away side, away covered: right
        _mg(0.5, 0.5, model_spread=9.0, spread=None, margin=-7),   # no spread: left out
    ]
    assert evaluate_mod.against_closing_spread(games) == evaluate_mod.SpreadCheck(3, 2)


def test_market_section_no_data_message():
    text = evaluate_mod.format_market_comparison([])
    assert "Model vs market (closing lines)" in text
    assert "No market data -- run predictor ingest-odds-history" in text


def test_market_section_rows_by_hand():
    games = [
        _mg(0.60, 0.50, True, season="2021-22", gid="a"),
        _mg(0.40, 0.70, True, season="2021-22", gid="b", from_spread=True),
        _mg(0.80, 0.60, False, season="2022-23", gid="c"),
    ]
    text = evaluate_mod.format_market_comparison(games)
    rows = {ln.split()[0]: ln for ln in text.splitlines() if ln.startswith("    20")}
    # 2021-22: model right on a only (50.0%); Brier (0.16 + 0.36)/2 = 0.26;
    # market right on both; Brier (0.25 + 0.09)/2 = 0.17; 1 moneyline, 1 spread.
    assert rows["2021-22"].split() == [
        "2021-22", "2", "50.0%", "0.2600", f"{(-math.log(0.6) - math.log(0.4)) / 2:.4f}",
        "100.0%", "0.1700", f"{(-math.log(0.5) - math.log(0.7)) / 2:.4f}", "1", "1",
    ]
    assert rows["2022-23"].split()[:3] == ["2022-23", "1", "0.0%"]
    pooled = next(ln for ln in text.splitlines() if ln.startswith("    all seasons"))
    assert pooled.split()[2] == "3"
    assert pooled.split()[-2:] == ["2", "1"]
    assert evaluate_mod.CLOSING_LINE_CAVEAT in text
    assert "using the model's own sigma for that season" in " ".join(text.split())


def _add_closing_line(con, game_id, *, home=None, away=None, spread=None):
    table = db.POINT_IN_TIME_TABLES["odds_snapshots"]
    con.execute(
        f"INSERT INTO {table} (game_key, book, home_team, away_team, home_price,"
        " away_price, spread, total, observed_at, game_id, source, reconstructed)"
        " VALUES (?, 'consensus', 'BOS', 'NYK', ?, ?, ?, 220, ?, ?, 'kaggle_sbr', TRUE)",
        [f"kaggle:{game_id}", home, away, spread, _T, game_id],
    )


def test_market_section_from_a_fixture_db_with_odds(tmp_path):
    con = fixture_con(tmp_path)
    build_history(con)
    # 2021-22: two games with moneylines; 2023-24: three with a spread only.
    _add_closing_line(con, "0022100001", home=-200, away=170, spread=-5.5)
    _add_closing_line(con, "0022100002", home=150, away=-170, spread=3.5)
    for n in (1, 2, 3):
        _add_closing_line(con, f"00223{n:05d}", spread=-2.5)
    # a live line for a predicted game is never used as a closing line
    table = db.POINT_IN_TIME_TABLES["odds_snapshots"]
    con.execute(
        f"INSERT INTO {table} (game_key, book, home_team, away_team, home_price,"
        " away_price, spread, total, observed_at, game_id, source, reconstructed)"
        " VALUES ('ev', 'fanduel', 'BOS', 'NYK', -500, 400, -9, 220, ?, '0022200001',"
        " 'theoddsapi', FALSE)",
        [_T],
    )
    ev = evaluate_mod.evaluate(con)
    games = evaluate_mod.market_games(con, ev.winner)
    assert sorted(g.prediction.game_id for g in games) == [
        "0022100001", "0022100002", "0022300001", "0022300002", "0022300003",
    ]
    by_id = {g.prediction.game_id: g for g in games}
    assert by_id["0022100001"].market.p_home == pytest.approx(0.642857, abs=1e-6)
    sigma_2324 = ev.winner.choices["2023-24"].sigma
    assert by_id["0022300001"].market.from_spread
    assert by_id["0022300001"].market.p_home == pytest.approx(
        0.5 * (1 + math.erf(2.5 / (sigma_2324 * math.sqrt(2))))
    )
    # margin and model spread carried through from the walk-forward
    d = ev.winner.details["0022100001"]
    assert by_id["0022100001"].margin == d.margin
    p = next(p for p in ev.winner.predictions if p.game_id == "0022100001")
    assert p.p_home == pytest.approx(
        0.5 * (1 + math.erf(d.model_spread / (d.sigma * math.sqrt(2))))
    )

    text = evaluate_mod.format_market_comparison(games)
    rows = {ln.split()[0]: ln.split() for ln in text.splitlines() if ln.startswith("    20")}
    assert set(rows) == {"2021-22", "2023-24"}
    assert rows["2021-22"][1] == "2" and rows["2021-22"][-2:] == ["2", "0"]
    assert rows["2023-24"][1] == "3" and rows["2023-24"][-2:] == ["0", "3"]
    assert "Where they disagree" in text
    assert "Against the closing spread" in text


def test_evaluate_model_cli_says_no_market_data_without_odds(tmp_path, monkeypatch):
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
    assert "Model vs market (closing lines)" in result.output
    assert "No market data -- run predictor ingest-odds-history" in result.output
