"""`evaluate-model` reports the player-model candidate against the
switch-over bar (Task 9).

The format tests use hand-built results (no database). The CLI tests build
a small fixture database in tmp_path; none touches data/ or the network.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from datetime import UTC, date, datetime

import pytest
from typer.testing import CliRunner

from model_fixtures import build_history
from predictor import cli, config, db
from predictor.backtest.replay import Prediction
from predictor.config import Settings
from predictor.model import candidate as cand
from predictor.model import evaluate as evaluate_mod
from predictor.model import tuning
from predictor.model.market import MarketView
from predictor.model.player_ids import LinkReport
from predictor.model.player_ratings import PlayerParams
from predictor.model.player_sim import SimStats
from predictor.model.ratings import RatingParams

SEASONS = evaluate_mod.WALK_FORWARD_SEASONS
TITLE = "  Player model candidate (walk-forward, final call = tip-off minus 45 minutes)"


def _pred(gid: str, season: str, p: float, won: bool) -> Prediction:
    stamp = datetime(2024, 1, 1, tzinfo=UTC)
    return Prediction(gid, season, date(2024, 1, 1), "BOS", "NYK", stamp, stamp, p, won, True)


def _season_preds(season: str, won_p: float, lost_p: float) -> list[Prediction]:
    """100 games: 65 home wins at ``won_p``, 35 home losses at ``lost_p``."""
    return [
        _pred(f"{season}-{i:03d}", season, won_p if i < 65 else lost_p, i < 65)
        for i in range(100)
    ]


def _variant(preds: list[Prediction], half_life=None) -> evaluate_mod.VariantResult:
    details = {
        p.game_id: evaluate_mod.GameDetail(3.0 if p.p_home >= 0.5 else -3.0, 12.0,
                                           5 if p.home_won else -5)
        for p in preds
    }
    return cand._variant(half_life, preds, details)


def _choice(roster: float = 0.8, injury: float = 0.9, formula: str = "box"):
    names = cand.feature_names(formula)
    coef = tuple(
        roster if n.startswith("roster_") else injury if n.startswith("injury_") else 0.1
        for n in names
    )
    return cand.CandidateChoice(
        team=RatingParams(0.05, 20.0, 0.33, 400),
        player=PlayerParams(1500.0, 0.4),
        formula=formula, w=0.5, coefficients=coef, feature_names=names,
        sigma=12.0, games=1000,
    )


def _stats() -> SimStats:
    by_season = {
        "2021-22": LinkReport(linked=50, unlinked=Counter({
            ("Morris, Marcus", "NYK"): 2, ("Smith, Jr., Dennis", "CHA"): 1})),
        "2022-23": LinkReport(linked=60, unlinked=Counter({("Morris, Marcus", "NYK"): 1})),
    }
    return SimStats(
        no_tip=2,
        fallback={"final": Counter({"2021-22": 5}), "early": Counter({"2021-22": 7})},
        no_report={"final": Counter({"2020-21": 11}), "early": Counter({"2020-21": 13})},
        unlinked=LinkReport(linked=110, unlinked=Counter({
            ("Morris, Marcus", "NYK"): 3, ("Smith, Jr., Dennis", "CHA"): 1})),
        unknown_status=Counter({"Suspended": 2}),
        unlinked_by_season=by_season,
        unknown_by_season={"2022-23": {"Suspended": 2}},
    )


def _cev(better: int = 3):
    """Current: 0.65 everywhere. Candidate: 0.68/0.62 in ``better`` seasons,
    0.64/0.65 in the rest."""
    current, candidate = [], []
    for i, season in enumerate(SEASONS):
        current += _season_preds(season, 0.65, 0.65)
        candidate += (_season_preds(season, 0.68, 0.62) if i < better
                      else _season_preds(season, 0.64, 0.65))
    cur_v = _variant(current)
    ev = evaluate_mod.Evaluation((cur_v,), cur_v, None)
    choice = _choice()
    result = cand.CandidateResult(
        half_life=None,
        final_call=_variant(candidate),
        early_call=_variant([dataclasses.replace(p, p_home=0.66) for p in candidate]),
        choices={s: choice for s in SEASONS},
        stats=_stats(),
    )
    cev = cand.CandidateEvaluation((result,), result, choice, ev)
    bar = cand.switch_over_bar(current, candidate, choice)
    return cev, bar


def _market(variant: evaluate_mod.VariantResult) -> list[evaluate_mod.MarketGame]:
    return [
        evaluate_mod.MarketGame(p, 3.0, 5 if p.home_won else -5,
                                MarketView(p_home=0.6, spread=-3.5, books=1,
                                           observed_at=datetime(2024, 1, 1, tzinfo=UTC),
                                           from_spread=False))
        for p in variant.predictions[:50]
    ]


def test_format_candidate_shows_bar_and_reasons():
    cev, bar = _cev(better=3)
    assert bar.met is False
    text = cand.format_candidate(cev, bar, None, [], [])
    assert TITLE in text
    assert "Switch-over bar: NOT MET" in text
    assert "[fail]" in text
    assert "[pass]" in text
    assert "The current model stays live." in text
    for reason in bar.reasons():
        assert reason in text.replace("\n    ", " ")
    for season in SEASONS:
        assert season in text
    assert "better" in text and "worse" in text
    assert cand.TEAM_SETTINGS_NOTE.split()[0] in text
    assert "conservative" in text
    assert "injury report" in text
    assert "identity only" in text


def test_format_candidate_met_bar_says_met():
    cev, bar = _cev(better=4)
    assert bar.met is True
    text = cand.format_candidate(cev, bar, None, [], [])
    assert "Switch-over bar: MET" in text
    assert "[fail]" not in text
    assert "The current model stays live." not in text


def test_format_candidate_shows_chosen_settings():
    cev, bar = _cev()
    text = cand.format_candidate(cev, bar, None, [], [])
    assert "formula box" in text
    assert "w 0.5" in text
    assert "half-life 1500 min" in text
    assert "regression 0.4" in text
    assert "roster box +0.800" in text
    assert "injury box +0.900" in text


def test_format_candidate_shows_dropped_correction():
    cev, bar = _cev()
    text = cand.format_candidate(cev, bar, None, [], [])
    assert "Correction model: dropped" in text
    tried = dataclasses.replace(
        cev.winner,
        final_call=_variant([dataclasses.replace(p, p_home=0.5)
                             for p in cev.winner.final_call.predictions]),
    )
    text = cand.format_candidate(cev, bar, None, [], [], tried=tried)
    assert "Correction model: dropped -- it did not improve the walk-forward (" in text
    assert "0.2500" in text  # the tried correction's Brier


def test_format_candidate_shows_included_correction():
    cev, bar = _cev()
    base = cev.winner.final_call
    corrected = dataclasses.replace(
        cev.winner,
        final_call=_variant([dataclasses.replace(p, p_home=0.9 if p.home_won else 0.1)
                             for p in base.predictions]),
    )
    text = cand.format_candidate(cev, bar, corrected, [], [])
    assert (
        f"  Correction model: included (pooled Brier {base.brier:.4f} -> 0.0100, "
        f"log loss {base.log_loss:.4f} -> "
    ) in text


def test_format_candidate_has_early_and_market_blocks():
    cev, bar = _cev()
    text = cand.format_candidate(
        cev, bar, None, _market(cev.winner.final_call), _market(cev.current.winner)
    )
    assert "early call" in text
    assert "08:30 ET (12:30 UTC)" in text
    assert "closing" in text
    assert evaluate_mod.CLOSING_LINE_CAVEAT in text
    assert "current" in text and "candidate" in text
    # Early-call candidate numbers: 0.66 everywhere -> Brier of that.
    early = cev.winner.early_call
    assert f"{early.brier:.4f}" in text


def test_format_candidate_without_market_lines():
    cev, bar = _cev()
    text = cand.format_candidate(cev, bar, None, [], [])
    assert evaluate_mod.NO_MARKET_DATA in text


def test_format_candidate_lists_unlinked_names():
    cev, bar = _cev()
    text = cand.format_candidate(cev, bar, None, [], [])
    assert "Morris, Marcus (NYK) ×3" in text
    assert "Smith, Jr., Dennis (CHA) ×1" in text
    assert text.index("Morris, Marcus (NYK) ×3") < text.index("Smith, Jr., Dennis (CHA) ×1")
    assert "Suspended ×2" in text
    assert "no known tip-off: 2" in text
    # Fallback and no-report counts appear for their seasons.
    dq = text[text.index("Data quality"):]
    line_2122 = next(ln for ln in dq.splitlines() if ln.strip().startswith("2021-22"))
    assert line_2122.split()[1:5] == ["5", "0", "7", "0"]
    assert line_2122.split()[5] == "3"  # unlinked listings that season
    line_2021 = next(ln for ln in dq.splitlines() if ln.strip().startswith("2020-21"))
    assert line_2021.split()[1:5] == ["0", "11", "0", "13"]


def test_format_candidate_top_ten_unlinked_only():
    cev, bar = _cev()
    many = Counter({(f"Name{i:02d}, X", "BOS"): 20 - i for i in range(12)})
    stats = dataclasses.replace(cev.winner.stats, unlinked=LinkReport(0, many))
    winner = dataclasses.replace(cev.winner, stats=stats)
    cev = dataclasses.replace(cev, winner=winner)
    text = cand.format_candidate(cev, bar, None, [], [])
    assert "Name09, X (BOS) ×11" in text
    assert "Name10, X" not in text


def test_evaluation_no_longer_prints_always_pick_home(tmp_path):
    from model_fixtures import fixture_con

    con = fixture_con(tmp_path)
    build_history(con)
    ev = evaluate_mod.evaluate(con)
    text = evaluate_mod.format_evaluation(ev)
    assert "always" not in text
    assert "pick home" not in text
    assert "Calibration -- when it said X%, how often did that happen?" in text


# -- the CLI -------------------------------------------------------------------

def _cli_db(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(config, "settings", s)
    monkeypatch.setattr(db, "settings", s)
    con = db.connect()
    db.migrate(con)
    return con


def test_cli_evaluate_model_without_player_logs(tmp_path, monkeypatch):
    con = _cli_db(tmp_path, monkeypatch)
    build_history(con)
    con.close()
    result = CliRunner().invoke(cli.app, ["evaluate-model"])
    assert result.exit_code == 0, result.output
    assert "Walk-forward evaluation" in result.output
    assert "  No player game logs -- run predictor ingest-player-logs" in result.output


def test_current_model_numbers_unchanged(tmp_path, monkeypatch):
    """With player logs, ``--candidate`` (the default) adds the candidate
    section after today's output; everything before it is byte-identical
    to ``--no-candidate``."""
    from test_candidate import _build, _shrink_team_grid

    _shrink_team_grid(monkeypatch)
    con = _cli_db(tmp_path, monkeypatch)
    _build(con)
    con.close()
    runner = CliRunner()
    without = runner.invoke(cli.app, ["evaluate-model", "--no-candidate"])
    assert without.exit_code == 0, without.stdout
    default = runner.invoke(cli.app, ["evaluate-model"])
    assert default.exit_code == 0, default.stdout
    with_flag = runner.invoke(cli.app, ["evaluate-model", "--candidate"])
    assert with_flag.stdout == default.stdout

    assert TITLE not in without.stdout
    assert "No player game logs" not in without.stdout
    assert TITLE in default.stdout
    assert "Switch-over bar:" in default.stdout
    assert "Correction model:" in default.stdout
    head = default.stdout[: default.stdout.index(TITLE)]
    assert head.rstrip("\n") == without.stdout.rstrip("\n")
    for key in ("log loss", "Brier", "accuracy"):
        cur = [ln for ln in without.stdout.splitlines() if key in ln]
        assert cur and all(ln in head.splitlines() for ln in cur)
    assert tuning.GRID_K == (0.05,)  # the shrunk grid was used
