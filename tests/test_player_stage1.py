"""The AsOfView player-model predictor (Task 10).

Parity: for every game in a small fixture built on the Task 6 player
fixture, and for both calls, the predictor reading only an ``AsOfView`` cut
at the call time gives the same spread as the fast replay
(``player_sim.simulate`` + ``fit._simulate`` + ``candidate.spread``). That
shows the replay uses nothing the view would block. No test touches data/
or the network.
"""

from __future__ import annotations

import dataclasses
import math
from datetime import UTC, date, datetime, time

import pytest

from model_fixtures import add_game, fixture_con
from player_fixtures import (
    D1,
    D2,
    D3,
    G2,
    ROSTERS,
    SEASON,
    add_player_row,
    add_report,
    add_team_rows,
    build_player_fixture,
    final_at,
    player_index,
    report_name,
)
from predictor.asof import AsOfView
from predictor.backtest.baselines import GameToPredict
from predictor.model import candidate as cand
from predictor.model import correction as corr
from predictor.model import fit as fit_mod
from predictor.model import player_sim
from predictor.model import player_stage1 as ps1
from predictor.model.availability import CALLS, STATUS_PRIORS, InjuryHistory
from predictor.model.player_ratings import PlayerParams
from predictor.model.player_settings import PlayerModelSettings
from predictor.model.player_sim import PlayerLines, simulate
from predictor.model.ratings import RatingParams
from predictor.model.venues import VenueIndex

TEAM = RatingParams(k=0.06, margin_cap=20.0, season_regression=0.3, hca_window=400)
PLAYER = PlayerParams(half_life_minutes=1500.0, season_regression=0.4)
NEXT = "2025-26"
G4, G5 = "0022400004", "0022400005"
G6, PRE6 = "0022500001", "0012500001"
D5 = date(2024, 11, 7)
D6, DPRE = date(2025, 10, 25), date(2025, 10, 10)


def _settings(formula="both", w=0.25, correction=None) -> PlayerModelSettings:
    names = cand.feature_names(formula)
    coefficients = {n: round(0.7 - 0.15 * i, 6) for i, n in enumerate(names)}
    return PlayerModelSettings(
        team=TEAM,
        player=PLAYER,
        formula=formula,
        w=w,
        coefficients=coefficients,
        sigma=12.5,
        half_life=None,
        availability={call: dict(STATUS_PRIORS) for call in CALLS},
        correction=correction,
        tuning_games=5,
    )


def _build(con) -> None:
    """Task 6's G1..G3 (BOS v NYK), plus a BOS doubleheader on D3 (G4 v
    MIA), G5 (NYK v MIA), a NaN plus-minus and a DNP row, injury reports
    at several times (one after the final call, one with an unknown
    status), and a next-season game after a preseason game in which an NYK
    player plays for BOS."""
    build_player_fixture(con)
    add_game(con, G4, SEASON, D3, "MIA", "BOS", 99, 104)
    add_team_rows(con, G4, D3, "MIA")
    add_team_rows(con, G4, D3, "BOS", minutes=20.0)
    add_game(con, G5, SEASON, D5, "NYK", "MIA", 97, 103)
    add_team_rows(con, G5, D5, "NYK", skip=(13,))
    add_player_row(con, G5, D5, 13, "NYK", None)  # did not play
    add_team_rows(con, G5, D5, "MIA")
    # A missing plus-minus stored as NaN: the replay and the view read it the
    # same way (None).
    add_player_row(con, G2, D2, 12, "NYK", 30.0, plus_minus=float("nan"), pts=25)

    star = report_name(ROSTERS["BOS"][1])
    add_report(con, datetime.combine(D3, time(12, 0), UTC), D3, "BOS", star, "Questionable")
    add_report(con, datetime.combine(D3, time(22, 30), UTC), D3, "BOS", star, "Out")
    add_report(con, datetime.combine(D3, time(22, 30), UTC), D3, "NYK",
               report_name(ROSTERS["NYK"][12]), "Doubtful")
    add_report(con, datetime.combine(D3, time(22, 30), UTC), D3, "MIA",
               report_name(ROSTERS["MIA"][21]), "Out")
    # After G3's final call: never seen for D3.
    add_report(con, datetime.combine(D3, time(23, 16), UTC), D3, "NYK",
               report_name(ROSTERS["NYK"][11]), "Out")
    add_report(con, datetime.combine(D5, time(19, 0), UTC), D5, "MIA",
               report_name(ROSTERS["MIA"][22]), "Probable")
    add_report(con, datetime.combine(D5, time(19, 0), UTC), D5, "NYK",
               report_name(ROSTERS["NYK"][14]), "Game Time Decision")

    # Next season: NYK's player 11 plays a preseason game for BOS.
    add_player_row(con, PRE6, DPRE, 11, "BOS", 25.0, season=NEXT, season_type="preseason")
    for pid in ROSTERS["BOS"]:
        add_player_row(con, PRE6, DPRE, pid, "BOS", 25.0, season=NEXT,
                       season_type="preseason")
    add_game(con, G6, NEXT, D6, "BOS", "MIA", 108, 100)
    add_report(con, datetime.combine(D6, time(20, 0), UTC), D6, "BOS",
               report_name(ROSTERS["NYK"][11]), "Out")
    add_report(con, datetime.combine(D6, time(20, 0), UTC), D6, "MIA",
               report_name(ROSTERS["MIA"][23]), "Questionable")


@pytest.fixture
def con(tmp_path):
    c = fixture_con(tmp_path)
    _build(c)
    yield c
    c.close()


def _replay(con, settings):
    """The fast replay's spread per (game_id, call), with its call time."""
    venues = VenueIndex.from_db(con)
    games = fit_mod._load(con, venues)
    lines = PlayerLines.from_db(con)
    history = InjuryHistory.from_db(con)
    feats, _ = simulate(games, lines, history, player_index(con), settings.player, [])
    pre = fit_mod._simulate(settings.team, games)
    coef = settings.coefficient_tuple()
    out = {}
    for g, (gap, hc) in zip(games, pre, strict=True):
        calls, has_tip = player_sim._game_calls(g, lines.tip_index)
        assert has_tip
        for call in CALLS:
            row = cand.design_row(g.x, getattr(feats[g.result.game_id], call), settings.formula)
            out[(g.result.game_id, call)] = (
                g, calls[call], cand.spread(gap, hc, row, coef, settings.w), row, gap, hc,
                feats[g.result.game_id],
            )
    return out


def _game(g) -> GameToPredict:
    r = g.result
    return GameToPredict(r.game_id, r.season, r.game_date, r.home_team, r.away_team)


FITTED = {
    "early": {"Out": 0.04, "Doubtful": 0.31, "Questionable": 0.62, "Probable": 0.88,
              "Available": 0.97},
    "final": {"Out": 0.02, "Doubtful": 0.17, "Questionable": 0.55, "Probable": 0.93,
              "Available": 0.99},
}


@pytest.mark.parametrize("order", ["forward", "backward"])
def test_parity_with_simulation(con, order, monkeypatch):
    """Both paths get the same fitted (non-prior) availability table, so a
    difference in how fitted probabilities are applied would show."""
    from predictor.model.availability import AvailabilityTable

    def fitted_tables(listings, season):
        return {call: AvailabilityTable(probs=dict(FITTED[call]), samples={})
                for call in CALLS}

    monkeypatch.setattr(player_sim, "availability_tables", fitted_tables)
    settings = dataclasses.replace(
        _settings(), availability={call: dict(FITTED[call]) for call in CALLS}
    )
    expected = _replay(con, settings)
    keys = sorted(expected, key=lambda k: (expected[k][1], k))
    if order == "backward":
        keys.reverse()
    predictor = ps1.PlayerStage1Predictor(con, settings)
    injuries = roster = 0
    for key in keys:
        g, at, sim_spread, *_ = expected[key]
        b = predictor.explain(_game(g), AsOfView(con, at), key[1])
        assert b.spread == pytest.approx(sim_spread, abs=1e-9), key
        assert b.p_home == pytest.approx(
            0.5 * (1 + math.erf(b.spread / (settings.sigma * math.sqrt(2)))), abs=1e-12
        )
        injuries += b.injuries != 0.0
        roster += b.roster != 0.0
    # The comparison is not trivial: both player terms move somewhere.
    assert injuries >= 3 and roster >= 3
    assert len(keys) == 2 * 6


def test_parity_of_correction_inputs(con):
    """Every input the correction reads (the spread's pieces and the extra
    point-in-time features) matches the replay, for both calls."""
    settings = _settings(formula="pm", w=0.5)
    expected = _replay(con, settings)
    predictor = ps1.PlayerStage1Predictor(con, settings)
    for key, (g, at, s, row, gap, hc, feats) in sorted(expected.items()):
        predictor.explain(_game(g), AsOfView(con, at), key[1])
        parts = cand.GameParts(
            additive_spread=s, team_gap=gap, home_court=hc, x=tuple(g.x),
            roster=ps1._sum_terms(settings, row, "roster_"),
            injury=ps1._sum_terms(settings, row, "injury_"),
        )
        want = corr.feature_row(parts, feats, key[1])
        got = predictor.correction_inputs[key]
        assert len(got) == len(want) == len(corr.CORRECTION_FEATURES)
        for a, b in zip(got, want, strict=True):
            assert (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b, abs=1e-9), key


def test_early_and_final_calls_read_different_reports(con):
    settings = _settings(formula="box", w=0.0)
    predictor = ps1.PlayerStage1Predictor(con, settings)
    game = GameToPredict("0022400003", SEASON, D3, "BOS", "NYK")
    early = predictor.explain(game, AsOfView(con, datetime.combine(D3, time(12, 30), UTC)),
                              "early")
    final = predictor.explain(game, AsOfView(con, final_at(D3)), "final")
    assert early.absent == (("BOS", "Al Adams", "Questionable"),)
    assert final.absent == (("BOS", "Al Adams", "Out"), ("NYK", "Jo Jones", "Doubtful"))
    assert final.injuries < early.injuries < 0.0


def test_predictor_never_reads_odds(con, monkeypatch):
    names: list[str] = []
    real = AsOfView.table

    def spy(self, name):
        names.append(name)
        return real(self, name)

    monkeypatch.setattr(AsOfView, "table", spy)
    settings = _settings()
    expected = _replay(con, settings)
    predictor = ps1.PlayerStage1Predictor(con, settings)
    for key, (g, at, *_rest) in sorted(expected.items()):
        predictor.explain(_game(g), AsOfView(con, at), key[1])
    assert names
    assert "odds_snapshots" not in names
    assert set(names) <= {"games", "player_games", "injury_status"}


def test_rejects_unknown_call(con):
    predictor = ps1.PlayerStage1Predictor(con, _settings())
    game = GameToPredict("0022400003", SEASON, D3, "BOS", "NYK")
    with pytest.raises(ValueError, match="call"):
        predictor.explain(game, AsOfView(con, final_at(D3)), "late")


def test_correction_is_retrained_deterministically_on_load(con):
    correction = {"training_seasons": [SEASON], "params": dict(corr.GBM_PARAMS)}
    settings = _settings(correction=correction)
    a = ps1.PlayerStage1Predictor(con, settings)
    b = ps1.PlayerStage1Predictor(con, settings)
    assert a.correction is not None and b.correction is not None
    game = GameToPredict(G6, NEXT, D6, "BOS", "MIA")
    view = AsOfView(con, final_at(D6))
    ba, bb = a.explain(game, view, "final"), b.explain(game, view, "final")
    assert ba == bb
    rows = ps1.correction_rows(con, settings)
    assert set(rows) == {SEASON}
    model = corr.Correction.train([r for rs in rows.values() for r in rs])
    want = model.predict(a.correction_inputs[(G6, "final")])
    assert ba.adjustment == pytest.approx(want, abs=1e-12)
    no_corr = ps1.PlayerStage1Predictor(con, _settings()).explain(game, view, "final")
    assert ba.spread == pytest.approx(no_corr.spread + ba.adjustment, abs=1e-9)


def test_correction_rows_are_the_shipped_models_residuals(con):
    settings = _settings(correction={"training_seasons": [SEASON],
                                     "params": dict(corr.GBM_PARAMS)})
    rows = ps1.correction_rows(con, settings)
    expected = _replay(con, settings)
    got = rows[SEASON]
    assert len(got) == 5  # G1..G5, the 2024-25 regular-season games
    margins = {G2: -4}
    for (feat, residual) in got:
        assert len(feat) == len(corr.CORRECTION_FEATURES)
    g2 = next(v for k, v in expected.items() if k == (G2, "final"))
    assert any(
        f[0] == pytest.approx(g2[2]) and res == pytest.approx(margins[G2] - g2[2])
        for f, res in got
    )


# -- the sentence ---------------------------------------------------------------

def _breakdown(**overrides) -> ps1.PlayerBreakdown:
    values = dict(
        game_id="0022400099", home_team="DEN", away_team="LAL",
        rating=2.04, roster=1.26, home=2.31, rest=-0.03, travel=0.44, altitude=1.02,
        injuries=-3.06, adjustment=0.0, spread=0.0, p_home=0.52,
        absent=(("DEN", "Jamal Murray", "Out"), ("DEN", "Michael Porter", "Questionable")),
    )
    values.update(overrides)
    return ps1.PlayerBreakdown(**values)


def test_sentence_names_absences_and_adds_up():
    b = _breakdown()
    text = b.sentence()
    assert "injuries -3.1 (Murray out, Porter questionable)" in text
    assert "-0.0" not in text
    shown = [2.0, 1.3, 2.3, 0.0, 0.4, 1.0, -3.1]
    assert f"DEN by {sum(shown):.1f}" in text
    assert text.startswith("LAL at DEN: rating +2.0, roster +1.3, home +2.3, rest +0.0,")
    assert "(DEN 52% to win)" in text


def test_sentence_shows_the_adjustment_only_with_a_correction():
    assert "adjustment" not in _breakdown().sentence()
    text = _breakdown(adjustment=-1.44, absent=()).sentence()
    assert "adjustment -1.4" in text
    assert "injuries -3.1, adjustment -1.4 ->" in text
    total = 2.0 + 1.3 + 2.3 + 0.0 + 0.4 + 1.0 - 3.1 - 1.4
    assert f"DEN by {total:.1f}" in text


def test_sentence_pickem_and_away_favourite():
    zero = _breakdown(rating=0.0, roster=0.0, home=0.0, rest=0.0, travel=0.0,
                      altitude=0.0, injuries=-0.04, absent=())
    assert "pick'em" in zero.sentence()
    away = _breakdown(rating=-6.0, absent=())
    assert "LAL by" in away.sentence()


def test_uses_the_settings_availability_table(con, monkeypatch):
    """The shipped table (estimated on every tuning season) sets the play
    probabilities: an Out player who plays half the time costs less."""
    import dataclasses

    from predictor.model.player_ratings import PlayerBook

    def rating(self, player_id, formula, as_of=None):
        self._check_as_of(as_of)
        return 6.0 if player_id == 1 else 0.0

    monkeypatch.setattr(PlayerBook, "rating", rating)

    base = _settings(formula="box", w=0.0)
    softer = dataclasses.replace(base, availability={
        "early": dict(STATUS_PRIORS), "final": {**STATUS_PRIORS, "Out": 0.5},
    })
    game = GameToPredict("0022400003", SEASON, D3, "BOS", "NYK")
    view = AsOfView(con, final_at(D3))
    hard = ps1.PlayerStage1Predictor(con, base).explain(game, view, "final")
    soft = ps1.PlayerStage1Predictor(con, softer).explain(game, view, "final")
    # injury_box has a negative coefficient (-0.2) in these settings.
    assert hard.injuries > soft.injuries > 0.0
    assert hard.injuries - soft.injuries > 0.01
    assert hard.roster == soft.roster
