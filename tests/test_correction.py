"""Optional gradient-boosted correction (Task 8).

Most tests use a synthetic ``CandidateResult``: 300 games a walk-forward
season, with a residual that depends non-linearly on ``pace_sum`` so the
correction has something to learn. One test runs the real candidate
fixture end to end. No test touches data/ or the network.
"""

from __future__ import annotations

import dataclasses
import math
import random
from collections import Counter
from datetime import UTC, date, datetime, time, timedelta

import pytest

from model_fixtures import add_game, fixture_con
from player_fixtures import (
    D3,
    G1,
    G2,
    G3,
    SEASON,
    add_team_rows,
    build_player_fixture,
    player_index,
)
from predictor import db
from predictor.backtest import metrics
from predictor.backtest.replay import Prediction
from predictor.model import candidate as cand
from predictor.model import correction as corr
from predictor.model import evaluate as evaluate_mod
from predictor.model import fit as fit_mod
from predictor.model.adjustments import FEATURE_NAMES
from predictor.model.availability import InjuryHistory
from predictor.model.player_ids import LinkReport
from predictor.model.player_ratings import PlayerParams
from predictor.model.player_sim import (
    EXTRA_KEYS,
    CallFeatures,
    GameFeatures,
    PlayerLines,
    SimStats,
    simulate,
)
from predictor.model.ratings import win_probability
from predictor.model.venues import VenueIndex

SEASONS = evaluate_mod.WALK_FORWARD_SEASONS  # 2020-21 .. 2025-26
GAMES = 300
SIGMA = 12.0


def _call(roster: float, injury: float) -> CallFeatures:
    return CallFeatures({"box": roster}, {"box": injury}, (), None, False, False)


def _synthetic(*, corrupt: str | None = None, seed: int = 7):
    """A base CandidateResult plus its features. The true margin is the
    additive spread + 4 * sign(pace_sum - 200) + noise. ``corrupt`` scrambles
    that season's results (from its own random stream, so every other draw
    is unchanged)."""
    rng = random.Random(seed)
    scramble = random.Random(seed + 1)
    preds: dict[str, list[Prediction]] = {"final": [], "early": []}
    details: dict[str, dict[str, evaluate_mod.GameDetail]] = {"final": {}, "early": {}}
    parts: dict[str, dict[str, cand.GameParts]] = {"final": {}, "early": {}}
    features: dict[str, GameFeatures] = {}
    for si, season in enumerate(SEASONS):
        start = date(2020 + si, 11, 1)
        for n in range(GAMES):
            gid = f"002{si}{n:05d}"
            gd = start + timedelta(days=n // 5)
            gap = rng.gauss(0, 4)
            hc = 2.0
            x = tuple(rng.choice((0.0, 1.0)) for _ in FEATURE_NAMES)
            roster, injury = rng.gauss(0, 2), -abs(rng.gauss(0, 1))
            pace = rng.uniform(180, 220)
            margin_gap = gap + rng.gauss(0, 2)
            continuity = rng.uniform(-0.3, 0.3)
            additive = gap + hc + 0.5 * roster + 0.8 * injury
            effect = 4.0 if pace > 200 else -4.0
            margin = round(additive + effect + rng.gauss(0, 8))
            if season == corrupt:
                margin = scramble.choice((-40, 40))
            if margin == 0:
                margin = 1
            features[gid] = GameFeatures(
                gid, _call(roster, injury), _call(roster, injury * 0.5),
                {"recent_margin_gap": margin_gap, "pace_sum": pace,
                 "continuity_gap": continuity, "continuity_gap_early": continuity / 2},
            )
            stamp = datetime.combine(gd, time(0), UTC)
            for call, inj in (("final", injury), ("early", injury * 0.5)):
                s = gap + hc + 0.5 * roster + 0.8 * inj
                parts[call][gid] = cand.GameParts(s, gap, hc, x, 0.5 * roster, 0.8 * inj)
                details[call][gid] = evaluate_mod.GameDetail(s, SIGMA, margin)
                preds[call].append(Prediction(
                    game_id=gid, season=season, game_date=gd, home_team="BOS",
                    away_team="NYK", tipoff=stamp, cutoff=stamp,
                    p_home=win_probability(s, SIGMA), home_won=margin > 0,
                    reconstructed=True,
                ))
    stats = SimStats(0, {}, {}, LinkReport(0, Counter()), Counter())
    result = cand.CandidateResult(
        half_life=None,
        final_call=cand._variant(None, preds["final"], details["final"]),
        early_call=cand._variant(None, preds["early"], details["early"]),
        choices={},
        stats=stats,
        parts=parts,
    )
    return result, features


@pytest.fixture(scope="module")
def synthetic():
    base, features = _synthetic()
    return base, features, corr.apply_corrections(base, features)


def _spreads(variant: evaluate_mod.VariantResult, season: str) -> dict[str, float]:
    by_season = {p.game_id for p in variant.predictions if p.season == season}
    return {g: d.model_spread for g, d in variant.details.items() if g in by_season}


# -- the brief's tests -------------------------------------------------------------

def test_correction_trains_only_on_earlier_seasons(synthetic, monkeypatch):
    base, features, _ = synthetic
    calls: list[tuple[str | None, tuple[str, ...], int]] = []
    real = corr.Correction.train.__func__

    def spy(cls, rows, *, seasons=(), season=None):
        calls.append((season, tuple(seasons), len(rows)))
        return real(cls, rows, seasons=seasons, season=season)

    monkeypatch.setattr(corr.Correction, "train", classmethod(spy))
    corr.apply_corrections(base, features)

    by_season = {s: (seasons, n) for s, seasons, n in calls}
    seasons, n = by_season["2023-24"]
    assert set(seasons) <= {"2020-21", "2021-22", "2022-23"}
    assert set(seasons) == {"2020-21", "2021-22", "2022-23"}
    assert n == 3 * GAMES
    # Every training call: only strictly earlier seasons, at least two of them.
    assert set(by_season) == set(SEASONS[2:])
    for season, (seasons, _) in by_season.items():
        assert len(seasons) >= corr.MIN_TRAINING_SEASONS
        assert all(s < season for s in seasons)


def test_correction_zero_without_two_training_seasons(synthetic):
    base, _, corrected = synthetic
    for season in ("2020-21", "2021-22"):
        for call in ("final_call", "early_call"):
            b, c = getattr(base, call), getattr(corrected, call)
            assert _spreads(c, season) == _spreads(b, season)
            assert ({p.game_id: p.p_home for p in c.predictions if p.season == season}
                    == {p.game_id: p.p_home for p in b.predictions if p.season == season})
    # Later seasons do move.
    assert _spreads(corrected.final_call, "2023-24") != _spreads(base.final_call, "2023-24")


def test_correction_is_deterministic(synthetic):
    base, features, first = synthetic
    second = corr.apply_corrections(base, features)
    assert [p.p_home for p in second.final_call.predictions] == [
        p.p_home for p in first.final_call.predictions
    ]
    assert [p.p_home for p in second.early_call.predictions] == [
        p.p_home for p in first.early_call.predictions
    ]
    assert second.final_call.details == first.final_call.details


def test_correction_helps_requires_both_metrics(synthetic):
    base, _, _ = synthetic
    worse_ll = dataclasses.replace(
        base,
        final_call=dataclasses.replace(
            base.final_call, brier=base.final_call.brier - 0.01,
            log_loss=base.final_call.log_loss + 0.01,
        ),
    )
    assert corr.correction_helps(base, worse_ll) is False
    worse_brier = dataclasses.replace(
        base,
        final_call=dataclasses.replace(
            base.final_call, brier=base.final_call.brier + 0.01,
            log_loss=base.final_call.log_loss - 0.01,
        ),
    )
    assert corr.correction_helps(base, worse_brier) is False
    equal = dataclasses.replace(base)
    assert corr.correction_helps(base, equal) is False
    both = dataclasses.replace(
        base,
        final_call=dataclasses.replace(
            base.final_call, brier=base.final_call.brier - 0.01,
            log_loss=base.final_call.log_loss - 0.01,
        ),
    )
    assert corr.correction_helps(base, both) is True


# -- extra coverage ----------------------------------------------------------------

def test_correction_learns_the_signal_and_helps(synthetic):
    base, _, corrected = synthetic
    assert corr.correction_helps(base, corrected)


def test_same_game_set_and_order(synthetic):
    base, _, corrected = synthetic
    for call in ("final_call", "early_call"):
        b, c = getattr(base, call), getattr(corrected, call)
        assert [p.game_id for p in c.predictions] == [p.game_id for p in b.predictions]
        assert set(c.details) == set(b.details)
        assert c.log_loss == metrics.log_loss(c.predictions)
        assert c.brier == metrics.brier_score(c.predictions)


def test_correction_helps_refuses_different_game_sets(synthetic):
    base, _, corrected = synthetic
    fewer = dataclasses.replace(
        corrected,
        final_call=dataclasses.replace(
            corrected.final_call, predictions=corrected.final_call.predictions[1:]
        ),
    )
    with pytest.raises(ValueError):
        corr.correction_helps(base, fewer)


def test_corrupting_a_season_never_moves_earlier_or_same_season_corrections(synthetic):
    """Leakage: scrambling every 2023-24 result leaves the corrected spreads
    and sigmas for 2020-21 .. 2023-24 as they were (2023-24 is corrected by a
    model trained on earlier seasons only); 2024-25, which trains on 2023-24,
    moves."""
    _, _, clean = synthetic
    base, features = _synthetic(corrupt="2023-24")
    dirty = corr.apply_corrections(base, features)
    for season in SEASONS[:4]:
        assert _spreads(dirty.final_call, season) == _spreads(clean.final_call, season)
        assert _spreads(dirty.early_call, season) == _spreads(clean.early_call, season)
        sig = {d.sigma for g, d in dirty.final_call.details.items() if g in _spreads(
            dirty.final_call, season)}
        sig_clean = {d.sigma for g, d in clean.final_call.details.items() if g in _spreads(
            clean.final_call, season)}
        assert sig == sig_clean
    assert _spreads(dirty.final_call, "2024-25") != _spreads(clean.final_call, "2024-25")


def test_training_rows_refuse_the_predicted_season():
    base, features = _synthetic()
    rows = corr.training_rows(base, features, ("2020-21",))
    with pytest.raises(fit_mod.FitError):
        corr.Correction.train(rows, seasons=("2020-21", "2021-22"), season="2021-22")


def test_none_when_no_season_has_enough_history():
    base, features = _synthetic()
    keep = set(SEASONS[:2])
    short = dataclasses.replace(
        base,
        final_call=cand._variant(
            None, [p for p in base.final_call.predictions if p.season in keep],
            base.final_call.details,
        ),
        early_call=cand._variant(
            None, [p for p in base.early_call.predictions if p.season in keep],
            base.early_call.details,
        ),
    )
    assert corr.apply_corrections(short, features) is None


def test_correction_features_layout():
    assert corr.EXTRA_FEATURES == ("recent_form_gap", "pace_sum", "continuity_gap")
    assert corr.CORRECTION_FEATURES == (
        ("additive_spread", "team_gap", "home_court") + FEATURE_NAMES
        + ("roster_gap", "injury_gap") + corr.EXTRA_FEATURES
    )
    assert corr.GBM_PARAMS == dict(
        max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=100,
        l2_regularization=1.0, random_state=0,
    )
    assert corr.MIN_TRAINING_SEASONS == 2


def test_feature_row_derives_recent_form_from_team_gap():
    p = cand.GameParts(5.0, 3.0, 2.0, (0.0, 1.0, 0.0, 1.0, 0.0), 1.5, -0.5)
    f = GameFeatures("g", _call(0, 0), _call(0, 0), {
        "recent_margin_gap": 7.0, "pace_sum": 201.0, "continuity_gap": 0.1,
        "continuity_gap_early": 0.2,
    })
    assert corr.feature_row(p, f, "final") == (
        5.0, 3.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.5, -0.5, 4.0, 201.0, 0.1,
    )
    assert corr.feature_row(p, f, "early")[-1] == 0.2


# -- extras in player_sim --------------------------------------------------------

PARAMS = PlayerParams(half_life_minutes=1500.0, season_regression=0.4)


def _run(con):
    games = fit_mod._load(con, VenueIndex.from_db(con))
    return simulate(games, PlayerLines.from_db(con), InjuryHistory.from_db(con),
                    player_index(con), PARAMS, [])[0]


@pytest.fixture
def con(tmp_path):
    c = fixture_con(tmp_path)
    yield c
    c.close()


def test_extras_are_point_in_time(con):
    """Changing a box score (player rows and the final score) dated on or
    after the game's date leaves that game's extras unchanged."""
    build_player_fixture(con)
    later = "0022400009"
    add_game(con, later, SEASON, D3 + timedelta(days=2), "BOS", "NYK", 100, 90)
    add_team_rows(con, later, D3 + timedelta(days=2), "BOS")
    add_team_rows(con, later, D3 + timedelta(days=2), "NYK")
    before = _run(con)

    games = db.POINT_IN_TIME_TABLES["games"]
    players = db.POINT_IN_TIME_TABLES["player_games"]
    con.execute(f"UPDATE {games} SET home_points = 140, away_points = 70 "
                f"WHERE game_id IN (?, ?)", [G3, later])
    con.execute(f"UPDATE {players} SET fga = 40, tov = 9 WHERE game_date >= ?", [D3])
    con.execute(f"DELETE FROM {players} WHERE game_id = ? AND player_id IN (2, 3)", [G3])
    after = _run(con)

    assert after[G3].extras == before[G3].extras
    # Sanity: the changed G3 box score does reach the later game.
    assert after[later].extras != before[later].extras


def test_extras_hand_values(con):
    """G1 has no history (all NaN); G3 sees G1 and G2: BOS won both by
    10 and 4 (mean 7), NYK lost both (mean -7); every player row is
    12 + 0.44 * 2 - 1 + 2 = 13.88 possessions, 8 players a team."""
    build_player_fixture(con)
    feats = _run(con)
    assert all(math.isnan(v) for v in feats[G1].extras.values())
    ex = feats[G3].extras
    assert ex["recent_margin_gap"] == pytest.approx(14.0)
    assert ex["pace_sum"] == pytest.approx(2 * 8 * 13.88)
    assert ex["continuity_gap"] == pytest.approx(0.0)
    assert ex["continuity_gap_early"] == pytest.approx(0.0)
    assert set(feats[G2].extras) == set(EXTRA_KEYS)


def test_continuity_counts_only_players_from_the_last_five_games(con):
    """BOS player 8 played only in G1; five more BOS games without him push
    G1 out of the last five, so his share no longer counts as continuity."""
    build_player_fixture(con)
    d = D3
    for k in range(5):
        gid = f"00224001{k:02d}"
        d = d + timedelta(days=1)
        add_game(con, gid, SEASON, d, "BOS", "NYK", 100, 99)
        add_team_rows(con, gid, d, "BOS", skip=(8,))
        add_team_rows(con, gid, d, "NYK")
    target = "0022400199"
    add_game(con, target, SEASON, d + timedelta(days=1), "BOS", "NYK", 100, 99)
    # Drop player 8 from G2 and G3 so his last game is G1.
    players = db.POINT_IN_TIME_TABLES["player_games"]
    con.execute(f"DELETE FROM {players} WHERE player_id = 8 AND game_id IN (?, ?)", [G2, G3])
    feats = _run(con)
    ex = feats[target].extras
    assert ex["continuity_gap"] < 0.0  # BOS has a minutes share from outside the last 5


# -- end to end on the candidate fixture --------------------------------------------

def test_candidate_records_parts_and_extras(tmp_path, monkeypatch):
    from test_candidate import _build, _shrink_team_grid

    _shrink_team_grid(monkeypatch)
    con = fixture_con(tmp_path)
    _build(con)
    current = evaluate_mod.evaluate(con)
    cev = cand.evaluate_candidate(con, current)
    winner = cev.winner
    for call in ("final", "early"):
        variant = getattr(winner, f"{call}_call")
        assert set(winner.parts[call]) == set(variant.details)
        for gid, d in variant.details.items():
            assert winner.parts[call][gid].additive_spread == pytest.approx(d.model_spread)
    assert set(cev.features) >= set(winner.final_call.details)
    corrected = corr.apply_corrections(winner, cev.features)
    assert corrected is not None
    assert ([p.game_id for p in corrected.final_call.predictions]
            == [p.game_id for p in winner.final_call.predictions])
    con.close()
