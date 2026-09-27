from datetime import date, timedelta

import pytest
from typer.testing import CliRunner

from model_fixtures import add_game, fixture_con
from predictor import cli, config, db
from predictor.config import Settings
from predictor.model import fit as fit_mod
from predictor.model import settings as ms
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


def test_fit_ignores_test_seasons_entirely(tmp_path):
    con = fixture_con(tmp_path)
    _history(con)
    before = fit_mod.fit(con)
    games = db.POINT_IN_TIME_TABLES["games"]
    placeholders = ", ".join("?" for _ in ms.TEST_SEASONS)
    con.execute(
        f"UPDATE {games} SET home_points = 50, away_points = 150 "
        f"WHERE status = 'FINAL' AND season IN ({placeholders})",
        list(ms.TEST_SEASONS),
    )
    assert fit_mod.fit(con) == before


def test_fit_counts_its_games_and_picks_values_from_the_grids(tmp_path):
    con = fixture_con(tmp_path)
    _history(con)
    s = fit_mod.fit(con)
    assert s.fit_games == 3 * 40          # three fit seasons x 40 games
    assert s.calibrate_games == 40
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
    assert "back-to-back" in text and "-1.1" in text
    assert "13.2" in text


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


def test_committed_settings_reproduce_from_the_real_archive():
    """A published number must trace to settings anyone can re-derive."""
    con = open_real_archive_or_skip()
    try:
        assert fit_mod.fit(con) == ms.load()
    finally:
        con.close()
