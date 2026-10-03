"""The shared settings-selection engine (spec 3, "Calibration redesign --
decided 2026-10-02"), Task 2 of the walk-forward recalibration: recency
weights and a wider grid, shared between fit-model and evaluate-model.
"""

from __future__ import annotations

from datetime import date

import pytest

from model_fixtures import build_history, fixture_con
from predictor import db
from predictor.model import fit as fit_mod
from predictor.model import settings as ms
from predictor.model import tuning
from predictor.model.venues import VenueIndex


def _neutral_game(gid, season, d, home, away, home_pts, away_pts, x):
    """A hand-built `fit._Game`, bypassing `_load`/the database entirely, so
    a test can fix the adjustment feature vector `x` directly instead of
    deriving it from a schedule. `neutral=True` so `_simulate`'s home-court
    term is 0.0 for this game regardless of `hca_window`.
    """
    return fit_mod._Game(
        fit_mod.Result(gid, season, d, home, away, home_pts, away_pts, True),
        x,
    )


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


def test_season_weights_rejects_seasons_out_of_order():
    with pytest.raises(ValueError, match="chronological"):
        tuning.season_weights(("b", "a"), None)


def test_season_weights_rejects_duplicate_seasons():
    with pytest.raises(ValueError, match="chronological"):
        tuning.season_weights(("a", "a", "b"), None)


def test_season_weights_rejects_nonpositive_half_life():
    with pytest.raises(ValueError, match="half_life"):
        tuning.season_weights(("a", "b"), 0.0)
    with pytest.raises(ValueError, match="half_life"):
        tuning.season_weights(("a", "b"), -1.0)


def test_job_rejects_seasons_out_of_order():
    with pytest.raises(ValueError, match="chronological"):
        tuning.Job(("b", "a"), None)


def test_job_rejects_duplicate_seasons():
    with pytest.raises(ValueError, match="chronological"):
        tuning.Job(("a", "a"), None)


def test_job_rejects_nonpositive_half_life():
    with pytest.raises(ValueError, match="half_life"):
        tuning.Job(("a", "b"), 0.0)


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


def _unique_team_pairs(n):
    """`n` (home, away) team-name pairs, no name ever repeated anywhere in
    this sequence. A team that never plays more than once always has
    rating 0.0 at its one pre-game moment, for every possible RatingParams
    -- there's no history for any params to have built up."""
    for i in range(n):
        yield f"H{i}", f"A{i}"


def _recency_weighting_games():
    """OLD season: 2 back-to-back (`x[0]=1`) games, home blows out by 20.
    NEW season: 2 back-to-back games, home loses by 20 (the opposite sign).
    Every team pair is unique (see `_unique_team_pairs`) and every game is
    neutral, so `_simulate` gives gap = home_court = 0.0 for EVERY game,
    for EVERY one of the 900 grid combinations -- not just the one that
    happens to win. That makes two things exact, not just "probably true":
    (1) the whole grid ties, so the winning params must be the very first
    combination in `product`'s order (GRID_K[0], GRID_CAP[0],
    GRID_REGRESSION[0], GRID_WINDOW[0]) -- this fixture doubles as the
    tie-break fixture below; and (2) with gap=hc=0, the regression target
    for every row is just its raw margin, and since columns 1-4 of `x` are
    0 for every row, only `x[0]=1` rows constrain coefficient 0 at all (a
    row with x=(0,0,0,0,0) predicts 0 regardless of the coefficients, so it
    contributes a constant to the loss that cannot move the fit). For a
    column that is 1 on a subset and 0 elsewhere, the weighted-least-squares
    minimiser for that column is exactly the WEIGHTED MEAN of the target
    over that subset -- here, the weighted mean of the four +-20 margins.
    """
    teams = _unique_team_pairs(4)
    games = []
    for i, margin in enumerate([20, 20]):  # OLD season, 2 games
        home, away = next(teams)
        games.append(_neutral_game(
            f"0020000000{i}", "1900-01", date(1900, 1, 1 + i), home, away,
            100 + margin, 100, (1.0, 0, 0, 0, 0),
        ))
    for i, margin in enumerate([-20, -20]):  # NEW season, 2 games
        home, away = next(teams)
        games.append(_neutral_game(
            f"0020000001{i}", "1901-02", date(1901, 1, 1 + i), home, away,
            100, 100 - margin, (1.0, 0, 0, 0, 0),
        ))
    return games


def test_recency_weight_changes_the_fitted_coefficient():
    """Weighting matters in the least-squares fit itself (not just in which
    grid combination wins). By the derivation in `_recency_weighting_games`,
    the back_to_back coefficient is exactly the weighted mean of the four
    back-to-back margins (+20, +20 from OLD; -20, -20 from NEW):

    - equal weight (all four weight 1.0): (20+20-20-20)/4 = 0.0 exactly --
      the old and new seasons' opposite effects cancel.
    - half-life 1.0 (two seasons: OLD weight 0.5**((2-1-0)/1)=0.5, NEW
      weight 0.5**((2-1-1)/1)=1.0): (0.5*20+0.5*20+1.0*-20+1.0*-20) / 3
      = (10+10-20-20)/3 = -20/3 = -6.666667.

    Mutation-check (recorded in the Task 2 fix report): replacing
    `sw = np.sqrt(w)` with `sw = np.ones_like(w)` makes the half-life-1.0
    coefficient come out 0.0 too (the lstsq step silently stops being
    weighted) -- this test then fails on the second assertion.
    """
    games = _recency_weighting_games()
    job_equal = tuning.Job(("1900-01", "1901-02"), None)
    job_half_life_1 = tuning.Job(("1900-01", "1901-02"), 1.0)
    chosen = tuning.choose(games, [job_equal, job_half_life_1])

    assert chosen[job_equal].coefficients.back_to_back == pytest.approx(0.0, abs=1e-6)
    assert chosen[job_half_life_1].coefficients.back_to_back == pytest.approx(-20 / 3)


def test_ties_keep_the_first_grid_combination_in_order():
    """Every one of the 900 combinations ties exactly on this fixture (see
    `_recency_weighting_games`'s docstring), so the winner must be the
    first one `product(GRID_K, GRID_CAP, GRID_REGRESSION, GRID_WINDOW)`
    produces. Mutation-check: changing the grid loop's `mse < best[job][0]`
    to `<=` makes every later tied combination replace the current best
    too, so the LAST combination wins instead
    (`RatingParams(GRID_K[-1], GRID_CAP[-1], GRID_REGRESSION[-1],
    GRID_WINDOW[-1])`) and this assertion fails.
    """
    games = _recency_weighting_games()
    job = tuning.Job(("1900-01", "1901-02"), None)
    chosen = tuning.choose(games, [job])[job]
    assert chosen.params == tuning.RatingParams(
        tuning.GRID_K[0], tuning.GRID_CAP[0], tuning.GRID_REGRESSION[0], tuning.GRID_WINDOW[0],
    )


def _grid_selection_weighting_games():
    """Built so that recency-weighting the grid-MSE (not just the lstsq)
    changes which of the 10 GRID_K values wins -- `margin_cap`, `season_
    regression` and `hca_window` are all neutralised first:

    - every game is neutral, so `hca_window` never matters (home_court is
      never read pre-game, and the rolling-margin deque is never appended
      to either).
    - every margin below is tiny (<=10) next to the smallest GRID_CAP
      (15.0), so `margin_cap` never clips, for any grid cap value.
    - `season_regression` only fires "the first time a new season is
      seen" (`Ratings.enter_season`); OLD is simulated first ever (no
      regression event), and every NEW-season team here is brand new (so
      its rating is 0 regardless of what regression did to OLD's now-
      irrelevant teams).

    That leaves k as the only grid dimension with any effect on `pre`, so
    for a fixed k the other 90 (cap, reg, window) combinations tie and the
    first of them (GRID_CAP[0], GRID_REGRESSION[0], GRID_WINDOW[0]) is what
    represents that k against every other k's own first representative.

    Three games carry `x=(1,0,0,0,0)` (back_to_back); a fourth (`O1`, all
    zeros) only sets up a k-dependent rating gap for `O2` and otherwise
    contributes a constant to every candidate's loss (its own row predicts
    0 regardless of the fit, so it can't move the argmin over k):
    - O1: X home vs Y away, margin +1 -- first-ever game, gap=0 always.
      After this, rating(X) = +k, rating(Y) = -k (predicted was 0).
    - O2: Y home vs X away (OLD season), margin -9, x=(1,0,0,0,0). Pre-game
      gap = rating(Y)-rating(X) = -2k, so y_O2(k) = -9 - (-2k) - 0 = -9+2k.
    - O2b: fresh teams (never played), OLD season, margin -7,
      x=(1,0,0,0,0). Fresh teams -> gap=0 always -> y_O2b = -7 (constant).
    - N1: fresh teams, NEW season, margin -10, x=(1,0,0,0,0). Fresh teams
      -> gap=0 always -> y_N1 = -10 (constant).

    Job(("2000-01","2001-02"), 1.0) weights OLD=0.5, NEW=1.0 (two seasons).
    Only O2, O2b, N1 carry x[0]=1, so (as in `_recency_weighting_games`)
    the fitted back_to_back coefficient is their weighted mean:
        c0(k) = (0.5*y_O2(k) + 0.5*y_O2b + 1.0*y_N1) / 2.0
    Writing a(k) = y_O2(k) - y_N1, algebra gives y_O2(k)-c0(k) = a(k)/1.5
    and y_O2b/y_N1's residuals as fixed multiples of a(k) too, so BOTH the
    correctly-weighted grid-MSE and an unweighted mean of the same three
    squared residuals are positive multiples of a(k)**2 alone when there
    are only 2 distinct values among the 3 points -- weighting can't change
    the argmin then. With 3 genuinely distinct values (y_O2(k), y_O2b,
    y_N1) that degeneracy breaks, and a numeric sweep over GRID_K (recorded
    in the fix report) confirms: the correctly-weighted grid-MSE is
    minimised at k=GRID_K[0]=0.02 (monotonically increasing in k), while
    the SAME three residuals averaged WITHOUT the 0.5/0.5/1.0 weights is
    minimised at k=GRID_K[-1]=0.20 (monotonically decreasing in k) --
    confirmed against a live `tuning.choose()` run too, not just the
    closed-form formula.
    """
    o1 = _neutral_game("00200000001", "2000-01", date(2000, 1, 1), "X", "Y", 101, 100, (0.0, 0, 0, 0, 0))
    o2 = _neutral_game("00200000002", "2000-01", date(2000, 1, 2), "Y", "X", 100, 109, (1.0, 0, 0, 0, 0))
    o2b = _neutral_game("00200000003", "2000-01", date(2000, 1, 3), "U", "V", 100, 107, (1.0, 0, 0, 0, 0))
    n1 = _neutral_game("00200000004", "2001-02", date(2001, 1, 1), "P", "Q", 100, 110, (1.0, 0, 0, 0, 0))
    return [o1, o2, o2b, n1]


def test_recency_weight_changes_which_grid_combination_wins():
    """The recency-weighted grid-MSE (`tuning.choose`'s `mse = ...` line)
    picks a different `k` than an unweighted mean of the same residuals
    would -- see `_grid_selection_weighting_games`'s docstring for the
    derivation and the numeric sweep.

    Mutation-check (recorded in the Task 2 fix report): this test fails
    under EITHER of the two mutations findings called out:
    - `mse = float(np.mean((y - X @ coef) ** 2))` (the `w *`/`/np.sum(w)`
      weighting dropped from the grid-MSE line) selects k=GRID_K[-1]=0.20
      instead of 0.02, so the `params` assertion fails.
    - `sw = np.ones_like(w)` (the lstsq weighting dropped) leaves `params`
      at k=0.02 here, but changes the fitted coefficient from -8.99 to
      -8.653333, so the `coefficients` assertion fails instead.
    """
    games = _grid_selection_weighting_games()
    job = tuning.Job(("2000-01", "2001-02"), 1.0)
    chosen = tuning.choose(games, [job])[job]
    assert chosen.params == tuning.RatingParams(
        tuning.GRID_K[0], tuning.GRID_CAP[0], tuning.GRID_REGRESSION[0], tuning.GRID_WINDOW[0],
    )
    assert chosen.coefficients.back_to_back == pytest.approx(-8.99)


def test_spreads_and_outcomes_filters_and_computes_correctly():
    """Only `002` games in the given seasons survive, in input order, and
    each spread is exactly gap + home_court + the adjustment terms.

    Every team pair here is unique and every game neutral (see
    `_neutral_game`), so gap = home_court = 0.0 regardless of `params` --
    spread reduces to the adjustment terms alone, hand-checkable directly:
    `g1` has back_to_back=1.0 and coefficient 2.5, so its spread is exactly
    2.5; `g4` has back_to_back=0.0, so its spread is exactly 0.0.
    """
    coefficients = fit_mod.Coefficients(2.5, 0.0, 0.0, 0.0, 0.0)
    params = fit_mod.RatingParams(0.1, 20.0, 0.5, 10)

    g1 = _neutral_game("00200001", "S1", date(2050, 1, 1), "A1", "B1", 101, 100, (1.0, 0, 0, 0, 0))
    g2 = _neutral_game("00300002", "S1", date(2050, 1, 2), "A2", "B2", 110, 100, (1.0, 0, 0, 0, 0))
    g3 = _neutral_game("00200003", "S2", date(2050, 1, 3), "A3", "B3", 105, 100, (1.0, 0, 0, 0, 0))
    g4 = _neutral_game("00200004", "S1", date(2050, 1, 4), "A4", "B4", 95, 100, (0.0, 0, 0, 0, 0))

    out = tuning.spreads_and_outcomes([g1, g2, g3, g4], params, coefficients, ("S1",))

    assert [g.result.game_id for g, _, _ in out] == ["00200001", "00200004"]
    g, spread, won = out[0]
    assert spread == pytest.approx(2.5)
    assert won is True
    g, spread, won = out[1]
    assert spread == pytest.approx(0.0)
    assert won is False


def test_job_with_no_matching_games_raises_fit_error():
    """A job whose seasons have no `002` games raises FitError before any
    grid search runs -- `_job_rows` filters on `game_id` and `job.seasons`
    alone, so an empty games list already exercises this without needing a
    database fixture."""
    with pytest.raises(fit_mod.FitError, match="1999-00"):
        tuning.choose([], [tuning.Job(("1999-00",), None)])
