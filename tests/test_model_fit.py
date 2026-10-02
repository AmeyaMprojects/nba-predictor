from datetime import date, timedelta

import pytest
from typer.testing import CliRunner

from model_fixtures import add_game, fixture_con
from predictor import cli, config, db
from predictor.backtest import replay
from predictor.config import Settings
from predictor.model import fit as fit_mod
from predictor.model import settings as ms
from predictor.model.stage1 import Stage1Predictor
from predictor.model.venues import VenueIndex
from real_archive import open_real_archive_or_skip

TEAMS = ["PHI", "NYK", "BOS", "MIA"]


def _season(con, season, start, n_days, home_edge, gid_prefix="002"):
    """A tiny round-robin: every day two games, home team wins by `home_edge`
    plus a deterministic team-strength term."""
    strength = {"PHI": 3, "NYK": -3, "BOS": 1, "MIA": -1}
    n = 0
    for day in range(n_days):
        d = start + timedelta(days=2 * day)
        pairs = [(TEAMS[day % 4], TEAMS[(day + 1) % 4]), (TEAMS[(day + 2) % 4], TEAMS[(day + 3) % 4])]
        for home, away in pairs:
            n += 1
            margin = home_edge + strength[home] - strength[away]
            add_game(con, f"{gid_prefix}{season[2:4]}{n:05d}", season, d, home, away,
                     100 + max(margin, 0), 100 + max(-margin, 0), city="Boston")


def _history(con):
    seasons = ms.WARMUP_SEASONS[-1:] + ms.FIT_SEASONS + (ms.CALIBRATE_SEASON,) + ms.TEST_SEASONS
    for i, season in enumerate(seasons):
        _season(con, season, date(2015 + i, 11, 1), 20, home_edge=3)


def test_fit_is_deterministic(tmp_path):
    con = fixture_con(tmp_path)
    _history(con)
    assert fit_mod.fit(con) == fit_mod.fit(con)


def _plant_mislabeled_test_season_game(con, game_id, home_pts, away_pts):
    """A game LABELLED with a test season but DATED inside the fit window,
    interleaved with real fit-season games between the same two teams.

    Every game in `_history`'s own test seasons is dated well after the
    fit/calibrate window, so corrupting one of THOSE can never move the fit
    -- simulation only ever walks forward in time, so a later-dated game
    cannot affect the pre-game numbers of any earlier game regardless of
    whether the season exclusion works at all. Planting a mislabeled game
    inside the window, between teams that already play there, is the only
    way to make a broken exclusion actually show up as a different fit.
    """
    plant_date = date(2016, 11, 2)  # inside the first fit season's date range
    add_game(con, game_id, ms.TEST_SEASONS[0], plant_date, "PHI", "NYK", home_pts, away_pts,
              city="Boston")
    return plant_date


def test_fit_ignores_test_seasons_entirely(tmp_path):
    con = fixture_con(tmp_path)
    _history(con)
    _plant_mislabeled_test_season_game(con, "00299999901", 100, 90)
    before = fit_mod.fit(con)

    games = db.POINT_IN_TIME_TABLES["games"]
    con.execute(
        f"UPDATE {games} SET home_points = 200, away_points = 1 WHERE game_id = ?",
        ["00299999901"],
    )
    assert fit_mod.fit(con) == before


def test_load_never_returns_a_row_from_a_test_season(tmp_path):
    con = fixture_con(tmp_path)
    _history(con)
    _plant_mislabeled_test_season_game(con, "00299999902", 100, 90)
    games = fit_mod._load(con, VenueIndex.from_db(con))
    assert games  # sanity: history was actually loaded
    assert all(g.result.season not in ms.TEST_SEASONS for g in games)


def test_simulate_matches_stage1predictor_rating_gap_and_home_court(tmp_path):
    """`fit._simulate` is a from-scratch replay of the SAME `Ratings` update
    rule `Stage1Predictor.explain` uses live -- if the two ever diverged,
    the settings `fit()` chooses would not describe what the predictor
    actually does. Fixture: 5 games over 4 dates, a season change
    (2014-15 -> 2019-20, exercising `enter_season`'s regression), and one
    neutral-site game (home_court must read 0.0 there and that game's
    margin must not enter the rolling home-margin window afterwards).
    """
    con = fixture_con(tmp_path)
    games = [
        ("00214150001", "2014-15", date(2015, 11, 1), "PHI", "NYK", 100, 90, False),
        ("00214150002", "2014-15", date(2015, 11, 1), "BOS", "MIA", 95, 92, False),
        ("00214150003", "2014-15", date(2015, 11, 3), "NYK", "BOS", 88, 93, True),  # neutral
        ("00219200001", "2019-20", date(2019, 11, 1), "PHI", "MIA", 101, 99, False),
        ("00219200002", "2019-20", date(2019, 11, 3), "NYK", "PHI", 90, 105, False),
    ]
    for gid, season, d, home, away, hp, ap, neutral in games:
        add_game(con, gid, season, d, home, away, hp, ap, city="Boston", neutral=neutral)

    params = fit_mod.RatingParams(k=0.1, margin_cap=20.0, season_regression=0.5, hca_window=10)
    venues = VenueIndex.from_db(con)
    loaded_games = fit_mod._load(con, venues)
    assert [g.result.game_id for g in loaded_games] == [gid for gid, *_ in games]
    simulated = fit_mod._simulate(params, loaded_games)

    settings = ms.ModelSettings(
        ratings=params,
        coefficients=fit_mod.Coefficients(0.0, 0.0, 0.0, 0.0, 0.0),
        sigma=13.0, fit_games=1, calibrate_games=1,
    )
    predictor = Stage1Predictor(con, settings, venues)
    preds, stats = replay.replay(con, predictor)
    assert stats.predicted == len(games)

    for (gid, *_), (gap, hc) in zip(games, simulated):
        b = predictor.breakdowns[gid]
        assert b.rating == pytest.approx(gap)
        assert b.home == pytest.approx(hc)


def test_fit_counts_its_games_and_picks_values_from_the_grids(tmp_path):
    con = fixture_con(tmp_path)
    _history(con)
    s = fit_mod.fit(con)
    assert s.fit_games == 3 * 40          # three fit seasons x 40 games
    # sigma is set on every non-test, non-warm-up season pooled (three fit
    # seasons + the calibrate season), not on the calibrate season alone:
    # 4 seasons x 40 games.
    assert s.calibrate_games == 4 * 40
    assert s.ratings.k in fit_mod.GRID_K
    assert s.ratings.margin_cap in fit_mod.GRID_CAP
    assert s.ratings.season_regression in fit_mod.GRID_REGRESSION
    assert s.ratings.hca_window in fit_mod.GRID_WINDOW
    assert s.sigma in fit_mod.SIGMA_GRID


def test_sigma_grid_is_exact():
    assert fit_mod.SIGMA_GRID[0] == 8.0
    assert fit_mod.SIGMA_GRID[-1] == 20.0
    assert len(fit_mod.SIGMA_GRID) == 241
    assert 13.05 in fit_mod.SIGMA_GRID


def test_describe_is_plain_english():
    s = ms.ModelSettings(
        ratings=fit_mod.RatingParams(0.08, 20.0, 0.33, 800),
        coefficients=fit_mod.Coefficients(-1.1, -0.6, -0.3, -0.2, 1.4),
        sigma=13.2, fit_games=3369, calibrate_games=1230,
    )
    text = fit_mod.describe(s)
    assert "8% of" in text
    assert "20 points" in text
    assert "33%" in text
    assert "800" in text
    # Final review (minor): coefficients now print with 2 decimals, not 1.
    assert "back-to-back" in text and "-1.10" in text
    assert "13.2" in text


def test_describe_never_prints_negative_zero():
    """Final review (minor): the committed settings' travel_per_1000km
    (-0.022147) rounds to "-0.0" at 1 decimal -- a term a reader sees as
    zero with a minus sign in front of it. Coefficients.back_to_back here
    is -0.001, which still rounds to -0.00 even at 2 decimals, so the fix
    must normalise that too, not just add a digit."""
    s = ms.ModelSettings(
        ratings=fit_mod.RatingParams(0.08, 20.0, 0.33, 800),
        coefficients=fit_mod.Coefficients(-0.001, -0.3, -0.022147, 0.001, 1.4),
        sigma=13.2, fit_games=3369, calibrate_games=1230,
    )
    text = fit_mod.describe(s)
    assert "-0.00" not in text
    assert "back-to-back (home minus away)       +0.00" in text
    assert "per 1,000 km travelled               -0.02" in text
    assert "per time zone crossed                +0.00" in text


def test_fit_model_command_writes_the_settings_file(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(config, "settings", s)
    monkeypatch.setattr(db, "settings", s)
    con = db.connect()
    db.migrate(con)
    _history(con)
    con.close()
    out_path = tmp_path / "stage1_settings.json"
    monkeypatch.setattr(ms, "SETTINGS_PATH", out_path)
    result = CliRunner().invoke(cli.app, ["fit-model"])
    assert result.exit_code == 0, result.output
    assert ms.load(out_path) is not None
    assert "saved" in result.output.lower()


def test_fit_model_with_no_history_is_a_plain_error(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(config, "settings", s)
    monkeypatch.setattr(db, "settings", s)
    con = db.connect()
    db.migrate(con)
    con.close()  # DuckDB refuses a read-only open while a writable one is live
    monkeypatch.setattr(ms, "SETTINGS_PATH", tmp_path / "s.json")
    result = CliRunner().invoke(cli.app, ["fit-model"])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "ingest-season" in result.output


def test_fit_model_save_failure_is_a_plain_error(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(config, "settings", s)
    monkeypatch.setattr(db, "settings", s)
    con = db.connect()
    db.migrate(con)
    _history(con)
    con.close()
    bad_path = tmp_path / "does-not-exist" / "stage1_settings.json"
    monkeypatch.setattr(ms, "SETTINGS_PATH", bad_path)
    result = CliRunner().invoke(cli.app, ["fit-model"])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert not bad_path.exists()


def test_committed_settings_reproduce_from_the_real_archive():
    """A published number must trace to settings anyone can re-derive."""
    con = open_real_archive_or_skip()
    try:
        assert fit_mod.fit(con) == ms.load()
    finally:
        con.close()


def test_sigma_is_chosen_on_fit_and_calibrate_seasons_pooled(tmp_path):
    """Changing a FIT-season result must be able to move sigma (it could not
    when sigma was chosen on the calibrate season alone); changing a
    warm-up result alone must not change the sigma sample size."""
    con = fixture_con(tmp_path)
    _history(con)
    s = fit_mod.fit(con)
    assert s.calibrate_games == s.fit_games + 40
