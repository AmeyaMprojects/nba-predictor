"""`evaluate-model` reports the player-model candidate against the
switch-over bar (Task 9).

The format tests use hand-built results (no database). The CLI tests build
a small fixture database in tmp_path; none touches data/ or the network.
"""

from __future__ import annotations

import dataclasses
import math
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
from predictor.model.ratings import FLAT_HOME, HomeParams, RatingParams

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
        no_report={"final": Counter({"2019-20": 4, "2020-21": 11}),
                   "early": Counter({"2019-20": 6, "2020-21": 13})},
        unlinked=LinkReport(linked=110, unlinked=Counter({
            ("Morris, Marcus", "NYK"): 3, ("Smith, Jr., Dennis", "CHA"): 1})),
        unknown_status=Counter({"Suspended": 2}),
        unlinked_by_season=by_season,
        unknown_by_season={"2022-23": {"Suspended": 2}},
    )


def _flat(text: str) -> str:
    return " ".join(text.split())


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
        assert reason in _flat(text)
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
    # The terms and sigma sit on their own indented line under the settings.
    line = next(ln for ln in text.splitlines() if "roster box +0.800" in ln)
    assert line.startswith(" " * 6) and "formula" not in line and "sigma 12.00" in line
    assert "injury box +0.900" in text


def test_format_candidate_shows_dropped_correction():
    cev, bar = _cev()
    text = cand.format_candidate(cev, bar, None, [], [])
    assert (
        "Correction model: dropped -- it could not be trained (no season had two "
        "earlier seasons of errors)."
    ) in _flat(text)
    tried = dataclasses.replace(
        cev.winner,
        final_call=_variant([dataclasses.replace(p, p_home=0.5)
                             for p in cev.winner.final_call.predictions]),
    )
    text = cand.format_candidate(cev, bar, None, [], [], tried=tried)
    assert "Correction model: dropped -- it did not improve the walk-forward (" in _flat(text)
    assert "-> 0.2500" in _flat(text)  # the tried correction's Brier


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
        f"Correction model: included (pooled Brier {base.brier:.4f} -> 0.0100, "
        f"log loss {base.log_loss:.4f} -> "
    ) in _flat(text)


def test_format_candidate_has_early_and_market_blocks():
    cev, bar = _cev()
    text = cand.format_candidate(
        cev, bar, None, _market(cev.winner.final_call), _market(cev.current.winner)
    )
    assert "early call" in text
    assert "12:30 UTC on game day (08:30 EDT, 07:30 EST)" in text
    assert "same current-model numbers" in _flat(text)
    assert "does not depend on the call time" in _flat(text)
    assert "closing" in text
    assert evaluate_mod.CLOSING_LINE_CAVEAT in _flat(text)
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
    assert "warm-up" not in line_2021
    warm = next(ln for ln in dq.splitlines() if ln.strip().startswith("2019-20"))
    assert warm.split()[1:6] == ["(warm-up)", "0", "4", "0", "6"]
    assert (
        "Seasons before 2020-21 are warm-up only: they train the ratings and are not "
        "scored."
    ) in _flat(dq)


def test_no_report_line_is_wider_than_96():
    """Worst case on the hand-built fixture: formula "both" with large
    coefficients, a dropped correction with numbers, market lines, and the
    full data-quality block."""
    cev, bar = _cev()
    names = cand.feature_names("both")
    coef = tuple(-12.345678 if n.startswith(("roster_", "injury_")) else 0.1 for n in names)
    wide = dataclasses.replace(cev.final, formula="both", feature_names=names,
                               coefficients=coef, sigma=13.75)
    winner = dataclasses.replace(cev.winner, choices={s: wide for s in SEASONS})
    cev = dataclasses.replace(cev, winner=winner, final=wide)
    tried = dataclasses.replace(cev.winner, final_call=_variant(
        [dataclasses.replace(p, p_home=0.5) for p in cev.winner.final_call.predictions]))
    for kwargs in ({"tried": tried}, {}):
        text = cand.format_candidate(
            cev, bar, None, _market(cev.winner.final_call), _market(cev.current.winner),
            **kwargs,
        )
        too_wide = [ln for ln in text.splitlines() if len(ln) > 96]
        assert too_wide == []


def test_format_candidate_top_ten_unlinked_only():
    cev, bar = _cev()
    many = Counter({(f"Name{i:02d}, X", "BOS"): 20 - i for i in range(12)})
    stats = dataclasses.replace(cev.winner.stats, unlinked=LinkReport(0, many))
    winner = dataclasses.replace(cev.winner, stats=stats)
    cev = dataclasses.replace(cev, winner=winner)
    text = cand.format_candidate(cev, bar, None, [], [])
    assert "Name09, X (BOS) ×11" in text
    assert "Name10, X" not in text


# -- home court per team ---------------------------------------------------------

HOME_TITLE = "  Home court per team (walk-forward; flat = one league-wide value)"
LAG_LINE = (
    "  Reports count only from 15 minutes before each call time (live runs fetch every "
    "15 minutes)."
)


def _home_variants(per_team_home: dict[str, HomeParams], final_home: HomeParams):
    """Flat: ``_cev(3)``. Per team: ``_cev(4)`` with the given homes."""
    flat, _ = _cev(better=3)
    per_team, _ = _cev(better=4)
    choices = {
        s: dataclasses.replace(c, home=per_team_home.get(s, FLAT_HOME))
        for s, c in per_team.winner.choices.items()
    }
    winner = dataclasses.replace(per_team.winner, choices=choices)
    per_team = dataclasses.replace(
        per_team, variants=(winner,), winner=winner,
        final=dataclasses.replace(per_team.final, home=final_home),
    )
    return cand.HomeVariants(flat=flat, per_team=per_team, games=())


EDGES = [
    cand.TeamHomeEdge("DEN", 164, 164, 5.43, 3.12),
    cand.TeamHomeEdge("BOS", 120, 118, 2.0, 2.41),
    cand.TeamHomeEdge("UTA", 164, 163, -0.75, 1.98),
]


def test_format_home_court_lists_every_team_edge():
    variants = _home_variants({"2022-23": HomeParams(150.0, 0.66)}, HomeParams(150.0, 0.66))
    text = cand.format_home_court(variants, EDGES, 1.2345)
    assert "Home court per team" in text
    assert "DEN" in text
    assert "Altitude is a separate term" in text
    lines = text.splitlines()
    assert lines[0] == HOME_TITLE
    assert "  2021-22  flat" in lines
    assert "  2022-23  shrinkage 150 games" in lines
    for season in SEASONS:
        assert any(ln.startswith(f"  {season}  ") for ln in lines)
    assert "  DEN  +3.1  (own +5.4 from 164 home / 164 away games)" in lines
    assert "  BOS  +2.4  (own +2.0 from 120 home / 118 away games)" in lines
    assert "  UTA  +2.0  (own -0.8 from 164 home / 163 away games)" in lines
    assert lines.index("  DEN  +3.1  (own +5.4 from 164 home / 164 away games)") < lines.index(
        "  UTA  +2.0  (own -0.8 from 164 home / 163 away games)")
    assert (
        "  Altitude is a separate term: +1.23 points, not inside DEN's or UTA's edge."
    ) in lines
    assert LAG_LINE in lines
    assert "  Per-team home court: kept" in lines
    flat_v, per_v = variants.flat.winner.final_call, variants.per_team.winner.final_call
    flat_row = next(ln for ln in lines if ln.strip().startswith("flat "))
    team_row = next(ln for ln in lines if ln.strip().startswith("per team "))
    assert f"{flat_v.brier:.4f}" in flat_row and f"{flat_v.log_loss:.4f}" in flat_row
    assert f"{flat_v.accuracy * 100:.1f}%" in flat_row
    assert f"{per_v.brier:.4f}" in team_row and f"{per_v.log_loss:.4f}" in team_row
    assert all(len(ln) <= 96 for ln in lines)


def _worse_per_team():
    """Per-team walk-forward worse than flat (the two walk-forwards of
    ``_home_variants`` swapped), with a per-team final choice."""
    variants = _home_variants({"2022-23": HomeParams(150.0, 0.66)}, HomeParams(150.0, 0.66))
    worse = dataclasses.replace(variants.flat, final=variants.per_team.final)
    return dataclasses.replace(variants, flat=variants.per_team, per_team=worse)


def test_shipped_home_variant_needs_per_team_no_worse_on_both_scores():
    better = _home_variants({"2022-23": HomeParams(150.0, 0.66)}, HomeParams(150.0, 0.66))
    assert cand.per_team_kept(better) is True
    assert cand.shipped_home_variant(better) is better.per_team
    worse = _worse_per_team()
    assert cand.per_team_kept(worse) is False
    assert cand.shipped_home_variant(worse) is worse.flat
    # Equal scores keep per-team ("at least as well").
    same = dataclasses.replace(better, flat=better.per_team)
    assert cand.shipped_home_variant(same) is same.per_team
    # Worse on only one score drops it.
    fc = better.per_team.winner.final_call
    one = dataclasses.replace(fc, brier=fc.brier, log_loss=fc.log_loss + 1.0)
    mixed = dataclasses.replace(better, per_team=dataclasses.replace(
        better.per_team, winner=dataclasses.replace(better.per_team.winner, final_call=one)))
    assert cand.shipped_home_variant(mixed) is mixed.flat


def test_format_home_court_dropped_when_per_team_scores_worse():
    variants = _worse_per_team()
    text = cand.format_home_court(variants, EDGES, 0.0)
    f = variants.flat.winner.final_call
    t = variants.per_team.winner.final_call
    assert (
        "Per-team home court: dropped -- it scored worse than one league-wide value in the "
        f"walk-forward (Brier {t.brier:.4f} vs {f.brier:.4f}, log loss {t.log_loss:.4f} vs "
        f"{f.log_loss:.4f})"
    ) in _flat(text)
    assert "  Per-team home court: dropped -- it scored worse" in text
    assert "Per-team home court: kept" not in text
    assert "flat home court ships" in _flat(text)
    assert all(len(ln) <= 96 for ln in text.splitlines())


def test_format_home_court_dropped_when_flat_wins():
    variants = _home_variants({}, HomeParams(math.inf, 0.66))
    text = cand.format_home_court(variants, EDGES, -0.5)
    assert (
        "  Per-team home court: dropped -- the flat league value scored as well or better"
    ) in text.splitlines()
    assert "Per-team home court: kept" not in text
    for season in SEASONS:
        assert f"  {season}  flat" in text.splitlines()
    assert "Altitude is a separate term: -0.50 points" in text


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
    seen: list = []
    real_variants, real_bar = cand.evaluate_home_variants, cand.switch_over_bar

    def variants_spy(*args, **kwargs):
        seen.append(real_variants(*args, **kwargs))
        return seen[-1]

    def bar_spy(current, candidate, choice):
        seen.append((candidate, choice))
        return real_bar(current, candidate, choice)

    monkeypatch.setattr(cand, "evaluate_home_variants", variants_spy)
    monkeypatch.setattr(cand, "switch_over_bar", bar_spy)
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
    section = default.stdout[default.stdout.index(TITLE):]
    assert [ln for ln in section.splitlines() if len(ln) > 96] == []
    for key in ("log loss", "Brier", "accuracy"):
        cur = [ln for ln in without.stdout.splitlines() if key in ln]
        assert cur and all(ln in head.splitlines() for ln in cur)
    assert tuning.GRID_K == (0.05,)  # the shrunk grid was used

    # The bar is scored for the per-team variant (when no correction is
    # included, its walk-forward predictions), and the report adds the flat
    # line and the home-court section.
    variants, (candidate, choice) = seen[0], seen[1]
    assert isinstance(variants, cand.HomeVariants)
    # In this fixture per-team scores worse than flat in the walk-forward
    # (Brier 0.1042 vs 0.1038), so the flat variant ships and is scored.
    t, f = variants.per_team.winner.final_call, variants.flat.winner.final_call
    assert t.brier > f.brier
    assert cand.shipped_home_variant(variants) is variants.flat
    assert choice is variants.flat.final
    assert choice.home.flat
    if "Correction model: included" not in section:
        assert candidate is variants.flat.winner.final_call.predictions
    assert "  Per-team home court: dropped -- it scored worse" in section
    flat = variants.flat.winner.final_call
    assert (
        f"  Player model with flat home court (plan 1): Brier {flat.brier:.4f}, "
        f"log loss {flat.log_loss:.4f}, accuracy {flat.accuracy * 100:.1f}%"
    ) in section.splitlines()
    assert HOME_TITLE in section.splitlines()
    assert LAG_LINE in section.splitlines()
    for edge in cand.home_edges(variants):
        assert (
            f"  {edge.team}  {edge.edge:+.1f}  (own {edge.raw:+.1f} from "
            f"{edge.home_games} home / {edge.away_games} away games)"
        ) in section.splitlines()
    altitude = dict(zip(choice.feature_names, choice.coefficients, strict=True))["altitude"]
    assert (
        f"  Altitude is a separate term: {altitude:+.2f} points, not inside DEN's or "
        "UTA's edge."
    ) in section.splitlines()
