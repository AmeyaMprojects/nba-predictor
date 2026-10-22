"""Candidate walk-forward and the switch-over bar (Task 7).

The fixture is small: ``model_fixtures.build_history`` (8 games a season,
four teams), five players a team at 48 minutes each, and one injury report
per game date except each season's first date. The team-rating grid is
shrunk to one combination so ``evaluate.evaluate`` is fast; the candidate's
own grid (6 player settings x 3 formulas x 5 weights) is left as is. No test
touches data/ or the network.
"""

from __future__ import annotations

import dataclasses
import math
from datetime import UTC, date, datetime, time

import pytest

from model_fixtures import build_history, fixture_con
from player_fixtures import add_player_row, add_report
from predictor import db
from predictor.backtest.replay import Prediction
from predictor.model import candidate as cand
from predictor.model import evaluate as evaluate_mod
from predictor.model import tuning
from predictor.model.adjustments import FEATURE_NAMES
from predictor.model.player_ratings import PlayerParams
from predictor.model.ratings import RatingParams

FIRST_NAMES = ("Al", "Bo", "Cy", "Di", "Ed")
TEAM_PLAYERS = {
    team: {base + i: f"{FIRST_NAMES[i]} {team.title()}" for i in range(5)}
    for team, base in (("PHI", 101), ("NYK", 201), ("BOS", 301), ("MIA", 401))
}
STATUSES = ("Out", "Questionable", "Doubtful")


def _shrink_team_grid(mp: pytest.MonkeyPatch) -> None:
    mp.setattr(tuning, "GRID_K", (0.05,))
    mp.setattr(tuning, "GRID_CAP", (20.0,))
    mp.setattr(tuning, "GRID_REGRESSION", (0.33,))
    mp.setattr(tuning, "GRID_WINDOW", (400,))


def _build(con) -> set[str]:
    """History plus player rows and injury reports. Returns the game ids
    whose date has no injury report (each season's first date)."""
    build_history(con, games_per_season=8)
    games = con.execute(
        f"SELECT game_id, season, game_date, home_team, away_team, home_points, "
        f"away_points FROM {db.POINT_IN_TIME_TABLES['games']} WHERE status = 'FINAL' "
        "ORDER BY game_date, game_id"
    ).fetchall()
    first_date: dict[str, date] = {}
    for _, season, gd, *_ in games:
        first_date.setdefault(season, gd)
    unreported: set[str] = set()
    for n, (gid, season, gd, home, away, hp, ap) in enumerate(games):
        margin = hp - ap
        for team, sign in ((home, 1), (away, -1)):
            for k, (pid, name) in enumerate(TEAM_PLAYERS[team].items()):
                add_player_row(
                    con, gid, gd, pid, team, 48.0, season=season, name=name,
                    pts=8 + 3 * k + (n % 7) + (4 if sign * margin > 0 else 0),
                    plus_minus=float(sign * margin + k - 2),
                )
        if gd == first_date[season]:
            unreported.add(gid)
            continue
        star = next(iter(TEAM_PLAYERS[home].values()))
        first, last = star.split(" ", 1)
        add_report(con, datetime.combine(gd, time(22, 30), UTC), gd, home,
                   f"{last},{first}", STATUSES[n % 3])
    return unreported


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """One fixture DB, the current model's evaluation, and one candidate
    evaluation with distinct team params per fold (so the test that team
    params come from the current fold is not trivially true)."""
    mp = pytest.MonkeyPatch()
    _shrink_team_grid(mp)
    con = fixture_con(tmp_path_factory.mktemp("candidate"))
    unreported = _build(con)
    current = evaluate_mod.evaluate(con)
    choices = {
        season: dataclasses.replace(
            c, params=dataclasses.replace(c.params, k=0.02 + 0.01 * i)
        )
        for i, (season, c) in enumerate(sorted(current.winner.choices.items()))
    }
    winner = dataclasses.replace(current.winner, choices=choices)
    current = dataclasses.replace(current, winner=winner)
    messages: list[str] = []
    cev = cand.evaluate_candidate(con, current, progress=messages.append)
    yield {"con": con, "current": current, "cev": cev, "unreported": unreported,
           "messages": messages, "mp": mp}
    mp.undo()
    con.close()


# -- the walk-forward ---------------------------------------------------------

def test_fold_never_sees_its_own_season(world, monkeypatch):
    calls: list[tuple[str | None, set[str], set[str]]] = []
    real = cand._weighted_lstsq

    def spy(games, X, y, w, *, season):
        calls.append((season, {g.result.season for g in games},
                      {g.result.game_id for g in games}))
        return real(games, X, y, w, season=season)

    monkeypatch.setattr(cand, "_weighted_lstsq", spy)
    cand.evaluate_candidate(world["con"], world["current"])

    fold_2223 = [c for c in calls if c[0] == "2022-23"]
    # 6 player settings x 3 formulas x 5 weights, for each recency variant.
    assert len(fold_2223) == 6 * 3 * 5 * len(tuning.HALF_LIVES)
    for _, seasons, _ in fold_2223:
        assert seasons and all(s < "2022-23" for s in seasons)
    # Every fold, not just 2022-23, fits only on strictly earlier seasons.
    for season, seasons, _ in calls:
        if season is not None:
            assert all(s < season for s in seasons)
    # Games with no injury report are left out of every fit.
    for _, _, ids in calls:
        assert not ids & world["unreported"]


def test_choices_unchanged_when_the_season_itself_is_corrupted(world, tmp_path):
    """Corrupting every 2022-23 result, box score and injury report leaves
    the choices for 2020-21 .. 2022-23 unchanged (fit on earlier seasons
    only); the 2023-24 choice, which trains on 2022-23, does move."""
    con = fixture_con(tmp_path)
    _build(con)
    games = db.POINT_IN_TIME_TABLES["games"]
    players = db.POINT_IN_TIME_TABLES["player_games"]
    injuries = db.POINT_IN_TIME_TABLES["injury_status"]
    con.execute(f"UPDATE {games} SET home_points = 150, away_points = 80 "
                "WHERE season = '2022-23' AND status = 'FINAL'")
    con.execute(f"UPDATE {players} SET pts = 60, plus_minus = 40 WHERE season = '2022-23'")
    con.execute(f"UPDATE {injuries} SET status = 'Out' WHERE game_date IN "
                f"(SELECT game_date FROM {games} WHERE season = '2022-23')")
    after = cand.evaluate_candidate(con, world["current"])
    before = world["cev"]
    for v_before, v_after in zip(before.variants, after.variants, strict=True):
        for season in ("2020-21", "2021-22", "2022-23"):
            assert v_after.choices[season] == v_before.choices[season]
        assert v_after.choices["2023-24"] != v_before.choices["2023-24"]
    con.close()


def test_team_params_come_from_current_fold(world):
    current, cev = world["current"], world["cev"]
    params = {current.winner.choices[s].params for s in evaluate_mod.WALK_FORWARD_SEASONS}
    assert len(params) == len(evaluate_mod.WALK_FORWARD_SEASONS)  # all distinct
    for result in cev.variants:
        assert set(result.choices) == set(evaluate_mod.WALK_FORWARD_SEASONS)
        for s in evaluate_mod.WALK_FORWARD_SEASONS:
            assert result.choices[s].team == current.winner.choices[s].params
    assert cev.final.team == current.final.ratings


def test_candidate_predicts_same_games_as_current(world):
    current, cev = world["current"], world["cev"]
    want = {p.game_id for p in current.winner.predictions}
    for result in cev.variants:
        assert {p.game_id for p in result.final_call.predictions} == want
        assert {p.game_id for p in result.early_call.predictions} == want
        # Unreported games are still predicted.
        assert world["unreported"] & want
        assert set(result.final_call.details) == want
        assert result.final_call.choices == {}
        assert result.early_call.choices == {}


def test_winner_follows_pick_winner_on_final_call_log_loss(world):
    cev = world["cev"]
    best = evaluate_mod.pick_winner([r.final_call for r in cev.variants])
    assert cev.winner.final_call is best
    assert cev.current is world["current"]


def test_choice_shape(world):
    for result in world["cev"].variants:
        for s, c in result.choices.items():
            assert c.formula in cand.FORMULAS
            assert c.w in cand.W_GRID
            assert c.player in cand.PLAYER_GRID
            assert c.feature_names == cand.feature_names(c.formula)
            assert len(c.coefficients) == len(c.feature_names)
            assert c.sigma in tuning.SIGMA_GRID
            assert c.games > 0


def test_progress_messages(world):
    msgs = world["messages"]
    assert any("player settings 6 of 6" in m for m in msgs)


def test_feature_names_by_formula():
    assert cand.feature_names("box") == FEATURE_NAMES + ("roster_box", "injury_box")
    assert cand.feature_names("pm") == FEATURE_NAMES + ("roster_pm", "injury_pm")
    assert cand.feature_names("both") == FEATURE_NAMES + (
        "roster_box", "roster_pm", "injury_box", "injury_pm")
    assert cand.FORMULAS == ("box", "pm", "both")
    assert cand.W_GRID == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert len(cand.PLAYER_GRID) == 6


def test_w_one_drops_team_gap():
    """Hand-made game: team gap 7.0, home court 2.5, features
    (b2b 1, 3-in-4 0, travel 0.5, tz 1, altitude 0, roster 3.0, injury -1.0),
    so the spread is (1 - w) * 7.0 + 2.5 + X.coef."""
    row = (1.0, 0.0, 0.5, 1.0, 0.0, 3.0, -1.0)
    coef = (-1.0, -0.5, -0.4, -0.3, 1.2, 0.8, 0.9)
    # X.coef = -1.0 + 0 - 0.2 - 0.3 + 0 + 2.4 - 0.9 = 0.0
    assert cand.spread(7.0, 2.5, row, coef, 1.0) == pytest.approx(2.5 + 0.0)
    coef2 = (-1.0, -0.5, -0.4, -0.3, 1.2, 1.0, 1.0)
    # X.coef = -1.0 - 0.2 - 0.3 + 3.0 - 1.0 = 0.5
    assert cand.spread(7.0, 2.5, row, coef2, 1.0) == pytest.approx(3.0)
    assert cand.spread(7.0, 2.5, row, coef2, 0.5) == pytest.approx(3.5 + 3.0)
    assert cand.spread(7.0, 2.5, row, coef2, 0.0) == pytest.approx(7.0 + 3.0)


# -- the switch-over bar ------------------------------------------------------

SEASONS = evaluate_mod.WALK_FORWARD_SEASONS


def _pred(gid: str, season: str, p: float, won: bool) -> Prediction:
    stamp = datetime(2024, 1, 1, tzinfo=UTC)
    return Prediction(gid, season, date(2024, 1, 1), "BOS", "NYK", stamp, stamp, p, won, True)


def _season_preds(season: str, won_p: float, lost_p: float) -> list[Prediction]:
    """100 games: 65 home wins at ``won_p``, 35 home losses at ``lost_p``."""
    return [
        _pred(f"{season}-{i:03d}", season, won_p if i < 65 else lost_p, i < 65)
        for i in range(100)
    ]


def _bar_preds(better: int) -> tuple[list[Prediction], list[Prediction]]:
    """Current: 0.65 everywhere (Brier 0.2275). Candidate: 0.68/0.62 in
    ``better`` seasons (Brier 0.2011) and 0.64/0.65 in the rest (Brier
    0.232115, slightly worse)."""
    current, candidate = [], []
    for i, season in enumerate(SEASONS):
        current += _season_preds(season, 0.65, 0.65)
        if i < better:
            candidate += _season_preds(season, 0.68, 0.62)
        else:
            candidate += _season_preds(season, 0.64, 0.65)
    return current, candidate


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


def test_switch_over_bar_met():
    """Hand values (4 better, 2 slightly worse):
    Brier current 0.2275; candidate (4*0.2011 + 2*0.232115)/6 = 0.211438.
    Log loss current 0.647447; candidate (4*0.589335 + 2*0.657524)/6 = 0.612065.
    Accuracy 0.65 both (every pick is home). Calibration: one bucket
    [0.6, 0.7) of 600 games, said (4*65.9 + 2*64.35)/6 = 65.4%, actual 65.0%."""
    current, candidate = _bar_preds(better=4)
    bar = cand.switch_over_bar(current, candidate, _choice())
    assert bar.pooled_brier[0] == pytest.approx(0.2275)
    assert bar.pooled_brier[1] == pytest.approx((4 * 0.2011 + 2 * 0.232115) / 6)
    ll_cur = -(0.65 * math.log(0.65) + 0.35 * math.log(0.35))
    assert bar.pooled_log_loss[0] == pytest.approx(ll_cur)
    assert round(ll_cur, 6) == 0.647447
    assert round(bar.pooled_log_loss[1], 6) == 0.612065
    assert bar.pooled_accuracy == (0.65, 0.65)
    assert bar.seasons_better == 4
    assert bar.seasons == 6
    assert bar.worst_bucket_gap == pytest.approx(0.4)
    assert bar.calibration_met is True
    assert bar.includes_terms is True
    assert bar.met is True
    assert bar.reasons() == []


def test_switch_over_bar_fails_on_accuracy():
    """1000 games, 654 home wins. Current says 0.654 everywhere (accuracy
    0.654). Candidate: 0.70 on 650 wins, 0.45 on 4 wins, 0.60 on 346 losses,
    accuracy 0.650; Brier 0.18427 < 0.226284, log loss 0.552 < 0.645."""
    current, candidate = [], []
    for i in range(1000):
        season = SEASONS[i % 6]
        won = i < 654
        current.append(_pred(f"g{i}", season, 0.654, won))
        p = 0.70 if i < 650 else 0.45 if won else 0.60
        candidate.append(_pred(f"g{i}", season, p, won))
    bar = cand.switch_over_bar(current, candidate, _choice())
    assert bar.pooled_accuracy == (0.654, 0.650)
    assert bar.pooled_brier[1] == pytest.approx(0.18427)
    assert bar.pooled_brier[1] < bar.pooled_brier[0]
    assert bar.pooled_log_loss[1] < bar.pooled_log_loss[0]
    assert bar.met is False
    assert any(r.startswith("Accuracy is lower") for r in bar.reasons())
    assert not any(r.startswith("Pooled Brier") for r in bar.reasons())


def test_switch_over_bar_fails_on_three_seasons():
    current, candidate = _bar_preds(better=3)
    bar = cand.switch_over_bar(current, candidate, _choice())
    assert bar.seasons_better == 3
    # Pooled metrics still pass: only the per-season condition fails.
    assert bar.pooled_brier[1] < bar.pooled_brier[0]
    assert bar.pooled_log_loss[1] < bar.pooled_log_loss[0]
    assert bar.met is False
    assert len(bar.reasons()) == 1
    assert "3 of 6" in bar.reasons()[0]


def test_switch_over_bar_fails_on_calibration():
    current, candidate = _bar_preds(better=6)
    # Push one season's candidate far off: 0.95 said on 65% wins.
    candidate = [
        dataclasses.replace(p, p_home=0.95) if p.season == SEASONS[0] else p
        for p in candidate
    ]
    bar = cand.switch_over_bar(current, candidate, _choice())
    assert bar.calibration_met is False
    assert bar.met is False
    assert any("Calibration" in r for r in bar.reasons())


def test_switch_over_bar_requires_terms():
    current, candidate = _bar_preds(better=4)
    bar = cand.switch_over_bar(current, candidate, _choice(injury=0.0, formula="both"))
    assert bar.includes_terms is False
    assert bar.met is False
    assert len(bar.reasons()) == 1
    # A coefficient that rounds to zero at 6 dp does not count either.
    bar = cand.switch_over_bar(current, candidate, _choice(roster=4e-7))
    assert bar.includes_terms is False


def test_switch_over_bar_rejects_different_game_sets():
    current, candidate = _bar_preds(better=4)
    with pytest.raises(ValueError):
        cand.switch_over_bar(current, candidate[:-1], _choice())
    swapped = candidate[:-1] + [dataclasses.replace(candidate[-1], game_id="other")]
    with pytest.raises(ValueError):
        cand.switch_over_bar(current, swapped, _choice())


def test_switch_over_bar_has_no_home_comparison():
    names = [f.name for f in dataclasses.fields(cand.SwitchOverBar)]
    assert not any("home" in n for n in names)
