"""Season roles and the committed Stage 1 settings file (spec 3).

Every fitted value lives in one JSON file tracked in git, so every
published number traces to exact settings. `predictor fit-model` writes it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from predictor.model.adjustments import Coefficients
from predictor.model.ratings import RatingParams

WARMUP_SEASONS = ("2014-15", "2015-16", "2016-17", "2017-18", "2018-19")
FIT_SEASONS = ("2019-20", "2020-21", "2021-22")
CALIBRATE_SEASON = "2022-23"
TEST_SEASONS = ("2023-24", "2024-25", "2025-26")

SETTINGS_PATH = Path(__file__).with_name("stage1_settings.json")
_VERSION = 1


class SettingsError(Exception):
    """The settings file is missing or unusable."""


@dataclass(frozen=True)
class ModelSettings:
    ratings: RatingParams
    coefficients: Coefficients
    sigma: float
    fit_games: int
    calibrate_games: int


def season_role(season: str) -> str:
    if season in WARMUP_SEASONS:
        return "warm-up"
    if season in FIT_SEASONS:
        return "fit"
    if season == CALIBRATE_SEASON:
        return "calibrate"
    if season in TEST_SEASONS:
        return "test"
    return "unassigned"


def to_json(s: ModelSettings) -> str:
    doc = {
        "version": _VERSION,
        "ratings": asdict(s.ratings),
        "coefficients": asdict(s.coefficients),
        "sigma": s.sigma,
        "fit_games": s.fit_games,
        "calibrate_games": s.calibrate_games,
    }
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"


def from_json(text: str) -> ModelSettings:
    hint = "Run: predictor fit-model"
    try:
        doc = json.loads(text)
        if doc.get("version") != _VERSION:
            raise SettingsError(
                f"the model settings file is from an unknown version "
                f"({doc.get('version')!r}). {hint}"
            )
        r = doc["ratings"]
        s = ModelSettings(
            ratings=RatingParams(
                k=float(r["k"]),
                margin_cap=float(r["margin_cap"]),
                season_regression=float(r["season_regression"]),
                hca_window=int(r["hca_window"]),
            ),
            coefficients=Coefficients(**{k: float(v) for k, v in doc["coefficients"].items()}),
            sigma=float(doc["sigma"]),
            fit_games=int(doc["fit_games"]),
            calibrate_games=int(doc["calibrate_games"]),
        )
    except SettingsError:
        raise
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise SettingsError(f"the model settings file is unreadable ({exc!r}). {hint}") from None
    if s.sigma <= 0 or s.ratings.hca_window < 1:
        raise SettingsError(f"the model settings file holds impossible values. {hint}")
    return s


def load(path: Path = SETTINGS_PATH) -> ModelSettings:
    try:
        text = path.read_text()
    except FileNotFoundError:
        raise SettingsError(
            f"No fitted model settings found at {path}. Run: predictor fit-model"
        ) from None
    return from_json(text)


def save(s: ModelSettings, path: Path = SETTINGS_PATH) -> None:
    path.write_text(to_json(s))
