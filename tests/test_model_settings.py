import pytest

from predictor.model import settings as ms
from predictor.model.adjustments import Coefficients
from predictor.model.ratings import RatingParams

S = ms.ModelSettings(
    ratings=RatingParams(k=0.08, margin_cap=20.0, season_regression=0.33, hca_window=800),
    coefficients=Coefficients(-1.1, -0.6, -0.3, -0.2, 1.4),
    sigma=13.2,
    fit_games=3369,
    calibrate_games=1230,
)


def test_round_trip(tmp_path):
    path = tmp_path / "s.json"
    ms.save(S, path)
    assert ms.load(path) == S


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


def test_nonpositive_sigma_is_rejected():
    text = ms.to_json(S).replace('"sigma": 13.2', '"sigma": 0.0')
    with pytest.raises(ms.SettingsError):
        ms.from_json(text)


@pytest.mark.parametrize(
    "season, role",
    [("2014-15", "warm-up"), ("2018-19", "warm-up"), ("2019-20", "fit"),
     ("2021-22", "fit"), ("2022-23", "calibrate"), ("2023-24", "test"),
     ("2025-26", "test"), ("2026-27", "unassigned")],
)
def test_season_roles(season, role):
    assert ms.season_role(season) == role


def test_roles_do_not_overlap():
    groups = [set(ms.WARMUP_SEASONS), set(ms.FIT_SEASONS), {ms.CALIBRATE_SEASON}, set(ms.TEST_SEASONS)]
    assert sum(len(g) for g in groups) == len(set().union(*groups)) == 12
