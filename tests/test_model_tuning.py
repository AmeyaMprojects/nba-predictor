"""The shared settings-selection engine (spec 3, "Calibration redesign --
decided 2026-10-02"), Task 2 of the walk-forward recalibration: recency
weights and a wider grid, shared between fit-model and evaluate-model.
"""

from __future__ import annotations

from datetime import date

import pytest

from model_fixtures import _season_games, build_history, fixture_con
from predictor import db
from predictor.model import fit as fit_mod
from predictor.model import settings as ms
from predictor.model import tuning
from predictor.model.venues import VenueIndex


def test_equal_weight_is_one_everywhere():
    assert tuning.season_weights(("a", "b", "c"), None) == {"a": 1.0, "b": 1.0, "c": 1.0}


def test_half_life_three_by_hand():
    w = tuning.season_weights(("a", "b", "c", "d"), 3.0)
    # newest 'd' age 0 -> 1.0; 'a' age 3 -> 0.5; 'b' age 2 -> 0.5**(2/3)
    assert w["d"] == 1.0
    assert w["a"] == pytest.approx(0.5)
    assert w["b"] == pytest.approx(0.5 ** (2 / 3))


def test_half_life_one_by_hand():
    w = tuning.season_weights(("a", "b", "c"), 1.0)
    assert w == pytest.approx({"a": 0.25, "b": 0.5, "c": 1.0})


def test_grids_are_exact():
    assert tuning.GRID_K == (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20)
    assert tuning.GRID_CAP == (15.0, 20.0, 25.0, 30.0, 40.0)
    assert tuning.GRID_REGRESSION == (0.0, 0.1, 0.2, 0.33, 0.5, 0.66)
    assert tuning.GRID_WINDOW == (400, 800, 1230)
    assert tuning.HALF_LIVES == (None, 3.0, 1.0)
    assert tuning.TIE_TOLERANCE == 0.001


def _load(con):
    return fit_mod._load(con, VenueIndex.from_db(con))


def test_choose_is_deterministic(tmp_path):
    con = fixture_con(tmp_path)
    build_history(con)
    games = _load(con)
    job = tuning.Job(ms.TUNING_SEASONS, None)
    assert tuning.choose(games, [job]) == tuning.choose(games, [job])


def test_choose_pools_jobs_without_mixing_them(tmp_path):
    """Two jobs over disjoint seasons, run together in one `choose` call,
    must give exactly what each would get run alone -- proof the shared
    simulation pass keeps every job's grid search and weighting separate."""
    con = fixture_con(tmp_path)
    build_history(con)
    games = _load(con)
    job_a = tuning.Job(ms.TUNING_SEASONS[:3], None)
    job_b = tuning.Job(ms.TUNING_SEASONS[3:], 3.0)

    together = tuning.choose(games, [job_a, job_b])
    alone = {**tuning.choose(games, [job_a]), **tuning.choose(games, [job_b])}
    assert together == alone


def test_job_ignores_games_outside_its_seasons(tmp_path):
    """Corrupt a season that comes strictly AFTER every season in the job.
    Simulation only ever walks forward in time, so a later season's results
    can never move an earlier game's pre-game numbers regardless of whether
    the job's season filter works -- but the filter also controls which
    games enter the job's weighted MSE/lstsq/sigma loss directly, and the
    corrupted season is excluded from `job.seasons` there too. Either way,
    the Choice must be identical before and after the corruption.
    """
    con = fixture_con(tmp_path)
    build_history(con)
    job = tuning.Job(ms.TUNING_SEASONS[:3], None)
    before = tuning.choose(_load(con), [job])[job]

    later_season = ms.TUNING_SEASONS[-1]  # strictly after job.seasons
    games_table = db.POINT_IN_TIME_TABLES["games"]
    con.execute(
        f"UPDATE {games_table} SET home_points = 250, away_points = 1 WHERE season = ?",
        [later_season],
    )
    after = tuning.choose(_load(con), [job])[job]
    assert after == before


def _history_with_home_edge(con, edges, games_per_season=40):
    """Like `build_history`, but each season's home-court edge can be
    overridden via `edges` (default 3 everywhere) -- lets a test make one
    season's games look structurally unlike the rest."""
    seasons = ms.WARMUP_SEASONS[-1:] + ms.TUNING_SEASONS + ms.TEST_SEASONS
    n_days = games_per_season // 2
    for i, season in enumerate(seasons):
        _season_games(con, season, date(2015 + i, 11, 1), n_days, home_edge=edges.get(season, 3))


def test_recency_weight_changes_the_chosen_settings(tmp_path):
    """Weighting matters. Fixture: every tuning season is an ordinary
    `home_edge=3` season except the OLDEST (`TUNING_SEASONS[0]`), which is
    planted with a huge `home_edge=40` -- a blowout home margin baked into
    every one of its games. Equal weight (`None`) lets that one outlier
    season pull the weighted grid-MSE, lstsq coefficients and sigma loss
    along with the six ordinary seasons; half-life 1.0 discounts it to
    0.5**6 of a newest-season game's weight, so the fit is effectively
    chosen on the six ordinary seasons alone. The two jobs must not land on
    the same Choice (asserted directly, per the brief: comparing a derived
    summary like mean predicted spread is more fragile than just comparing
    the minimisers, which is what the recency weighting is actually for).
    """
    con = fixture_con(tmp_path)
    old_season = ms.TUNING_SEASONS[0]
    _history_with_home_edge(con, {old_season: 40.0})
    games = _load(con)

    job_equal = tuning.Job(ms.TUNING_SEASONS, None)
    job_half_life_1 = tuning.Job(ms.TUNING_SEASONS, 1.0)
    chosen = tuning.choose(games, [job_equal, job_half_life_1])
    assert chosen[job_equal] != chosen[job_half_life_1]


def test_job_with_no_matching_games_raises_fit_error():
    """A job whose seasons have no `002` games raises FitError before any
    grid search runs -- `_job_rows` filters on `game_id` and `job.seasons`
    alone, so an empty games list already exercises this without needing a
    database fixture."""
    with pytest.raises(fit_mod.FitError, match="1999-00"):
        tuning.choose([], [tuning.Job(("1999-00",), None)])
