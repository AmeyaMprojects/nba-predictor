import dataclasses

import pytest

from predictor.model import settings as ms
from predictor.model.adjustments import Coefficients
from predictor.model.ratings import FLAT_HOME, HomeParams, RatingParams

S = ms.ModelSettings(
    ratings=RatingParams(k=0.08, margin_cap=20.0, season_regression=0.33, hca_window=800),
    coefficients=Coefficients(-1.1, -0.6, -0.3, -0.2, 1.4),
    sigma=13.2,
    half_life=3.0,
    tuning_games=7889,
)


def test_round_trip(tmp_path):
    path = tmp_path / "s.json"
    ms.save(S, path)
    assert ms.load(path) == S


def test_round_trip_with_no_half_life(tmp_path):
    """half_life=None (equal weight) must serialise as JSON null and come
    back as None, not as a string or a number."""
    s = ms.ModelSettings(
        ratings=S.ratings, coefficients=S.coefficients, sigma=S.sigma,
        half_life=None, tuning_games=S.tuning_games,
    )
    path = tmp_path / "s.json"
    ms.save(s, path)
    assert '"half_life": null' in path.read_text()
    loaded = ms.load(path)
    assert loaded.half_life is None
    assert loaded == s


def test_json_is_stable_and_sorted():
    assert ms.to_json(S) == ms.to_json(ms.from_json(ms.to_json(S)))
    assert ms.to_json(S).endswith("\n")


def test_missing_file_is_a_plain_error(tmp_path):
    with pytest.raises(ms.SettingsError, match="predictor fit-model"):
        ms.load(tmp_path / "nope.json")


@pytest.mark.parametrize("bad", ["not json", "{}", '{"version": 2}'])
def test_corrupt_file_is_a_plain_error(tmp_path, bad):
    path = tmp_path / "s.json"
    path.write_text(bad)
    with pytest.raises(ms.SettingsError, match="predictor fit-model"):
        ms.load(path)


def test_version_1_document_is_rejected(tmp_path):
    """The old settings-file shape (fit_games/calibrate_games, no
    half_life/tuning_games) must be refused with a plain error pointing at
    the command that regenerates the file."""
    doc = ms.to_json(S).replace('"version": 2', '"version": 1')
    path = tmp_path / "s.json"
    path.write_text(doc)
    with pytest.raises(ms.SettingsError, match="predictor fit-model"):
        ms.load(path)


def test_nonpositive_sigma_is_rejected():
    text = ms.to_json(S).replace('"sigma": 13.2', '"sigma": 0.0')
    with pytest.raises(ms.SettingsError):
        ms.from_json(text)


@pytest.mark.parametrize("bad_sigma", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_sigma_is_rejected(bad_sigma):
    # Python's json module accepts NaN/Infinity/-Infinity as an extension,
    # so a corrupt file holding one of these must not slip past the
    # sigma <= 0 check (NaN and +inf both fail that comparison).
    text = ms.to_json(S).replace('"sigma": 13.2', f'"sigma": {bad_sigma}')
    with pytest.raises(ms.SettingsError):
        ms.from_json(text)


@pytest.mark.parametrize("bad_half_life", ["0.0", "-1.0", "NaN", "Infinity", "-Infinity"])
def test_nonpositive_or_non_finite_half_life_is_rejected(bad_half_life):
    text = ms.to_json(S).replace('"half_life": 3.0', f'"half_life": {bad_half_life}')
    with pytest.raises(ms.SettingsError):
        ms.from_json(text)


def test_negative_tuning_games_is_rejected():
    text = ms.to_json(S).replace('"tuning_games": 7889', '"tuning_games": -1')
    with pytest.raises(ms.SettingsError):
        ms.from_json(text)


@pytest.mark.parametrize(
    "season, role",
    [("2014-15", "warm-up"), ("2018-19", "warm-up"), ("2019-20", "tuning"),
     ("2023-24", "tuning"), ("2025-26", "tuning"), ("2026-27", "test"),
     ("2027-28", "unassigned")],
)
def test_season_roles(season, role):
    assert ms.season_role(season) == role


def test_roles_do_not_overlap():
    groups = [set(ms.WARMUP_SEASONS), set(ms.TUNING_SEASONS), set(ms.TEST_SEASONS)]
    assert sum(len(g) for g in groups) == len(set().union(*groups)) == 13


TEAM_HOME = dataclasses.replace(S, home=HomeParams(150.0, 0.658231))


def test_settings_default_home_is_flat_and_not_written():
    assert S.home == FLAT_HOME
    assert '"home"' not in ms.to_json(S)
    assert ms.from_json(ms.to_json(S)).home == FLAT_HOME


def test_settings_home_round_trip():
    text = ms.to_json(TEAM_HOME)
    assert '"home": {' in text
    assert ms.from_json(text) == TEAM_HOME
    assert ms.from_json(text).home == HomeParams(150.0, 0.658231, 164)


def test_committed_settings_file_round_trips_byte_identical():
    text = ms.SETTINGS_PATH.read_text()
    assert ms.to_json(ms.from_json(text)) == text
    assert ms.from_json(text).home == FLAT_HOME


@pytest.mark.parametrize(
    "old, new",
    [('"shrinkage": 150.0', '"shrinkage": -1.0'),
     ('"shrinkage": 150.0', '"shrinkage": NaN'),
     ('"window": 164', '"window": 0'),
     ('"altitude_allowance": 0.658231', '"altitude_allowance": NaN')],
)
def test_settings_rejects_negative_shrinkage(old, new):
    text = ms.to_json(TEAM_HOME)
    assert old in text
    with pytest.raises(ms.SettingsError):
        ms.from_json(text.replace(old, new))
