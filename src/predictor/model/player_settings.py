"""The player-model candidate's settings file (plan 1, Task 10).

``predictor fit-player-model`` writes it only when the candidate passes the
switch-over bar. Nothing reads it to make live predictions in this plan: the
current model (``stage1_settings.json``) stays live, and plan 3 decides the
switch.

A shipped correction is recorded by its training seasons and its
gradient-boosting parameters only. The model itself is retrained from the
database when the predictor is built (deterministic, ``random_state=0``) and
is never pickled.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from predictor.model.availability import CALLS, STATUS_PRIORS
from predictor.model.candidate import FORMULAS, feature_names
from predictor.model.correction import GBM_PARAMS
from predictor.model.player_ratings import PlayerParams
from predictor.model.ratings import RatingParams

SETTINGS_PATH = Path(__file__).with_name("player_settings.json")
_VERSION = 1
HINT = "Run: predictor fit-player-model"


class SettingsError(Exception):
    """The player-model settings file is missing or unusable."""


@dataclass(frozen=True)
class PlayerModelSettings:
    team: RatingParams
    player: PlayerParams
    formula: str
    w: float
    # Feature name -> coefficient, in ``candidate.feature_names(formula)`` order.
    coefficients: dict[str, float]
    sigma: float
    half_life: float | None
    # "early"/"final" -> status -> probability of playing, estimated on every
    # tuning season's listings.
    availability: dict[str, dict[str, float]]
    # {"training_seasons": [...], "params": GBM_PARAMS}, or None.
    correction: dict | None
    tuning_games: int

    def coefficient_tuple(self) -> tuple[float, ...]:
        return tuple(self.coefficients[name] for name in feature_names(self.formula))


def to_json(s: PlayerModelSettings) -> str:
    doc = {
        "version": _VERSION,
        "team": asdict(s.team),
        "player": asdict(s.player),
        "formula": s.formula,
        "w": s.w,
        "coefficients": dict(s.coefficients),
        "sigma": s.sigma,
        "half_life": s.half_life,
        "availability": {call: dict(probs) for call, probs in s.availability.items()},
        "correction": None if s.correction is None else {
            "training_seasons": list(s.correction["training_seasons"]),
            "params": dict(s.correction["params"]),
        },
        "tuning_games": s.tuning_games,
    }
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"


def _finite(value) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{value!r} is not a finite number")
    return v


def from_json(text: str) -> PlayerModelSettings:
    try:
        doc = json.loads(text)
        if not isinstance(doc, dict) or doc.get("version") != _VERSION:
            version = doc.get("version") if isinstance(doc, dict) else None
            raise SettingsError(
                f"the player-model settings file is from an unknown version "
                f"({version!r}). {HINT}"
            )
        t, p = doc["team"], doc["player"]
        formula = doc["formula"]
        if formula not in FORMULAS:
            raise ValueError(f"unknown formula {formula!r}")
        names = feature_names(formula)
        raw = doc["coefficients"]
        if set(raw) != set(names):
            raise ValueError(f"coefficients must be exactly {names}, got {sorted(raw)}")
        availability = {}
        for call in CALLS:
            probs = doc["availability"][call]
            if set(probs) != set(STATUS_PRIORS):
                raise ValueError(f"availability[{call!r}] must list {tuple(STATUS_PRIORS)}")
            availability[call] = {k: _finite(probs[k]) for k in STATUS_PRIORS}
        if set(doc["availability"]) != set(CALLS):
            raise ValueError(f"availability must be keyed {CALLS}")
        correction = doc["correction"]
        if correction is not None:
            seasons = [str(v) for v in correction["training_seasons"]]
            correction = {"training_seasons": seasons, "params": dict(correction["params"])}
        half_life = doc["half_life"]
        s = PlayerModelSettings(
            team=RatingParams(
                k=float(t["k"]),
                margin_cap=float(t["margin_cap"]),
                season_regression=float(t["season_regression"]),
                hca_window=int(t["hca_window"]),
            ),
            player=PlayerParams(
                half_life_minutes=float(p["half_life_minutes"]),
                season_regression=float(p["season_regression"]),
            ),
            formula=formula,
            w=_finite(doc["w"]),
            coefficients={n: _finite(raw[n]) for n in names},
            sigma=float(doc["sigma"]),
            half_life=None if half_life is None else float(half_life),
            availability=availability,
            correction=correction,
            tuning_games=int(doc["tuning_games"]),
        )
    except SettingsError:
        raise
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise SettingsError(
            f"the player-model settings file is unreadable ({exc!r}). {HINT}"
        ) from None
    if (
        not math.isfinite(s.sigma) or s.sigma <= 0
        or not 0.0 <= s.w <= 1.0
        or s.team.hca_window < 1
        or not math.isfinite(s.player.half_life_minutes) or s.player.half_life_minutes <= 0
        or not 0.0 <= s.player.season_regression <= 1.0
        or (s.half_life is not None and (not math.isfinite(s.half_life) or s.half_life <= 0))
        or s.tuning_games < 0
        or any(not 0.0 <= v <= 1.0 for probs in s.availability.values() for v in probs.values())
    ):
        raise SettingsError(f"the player-model settings file holds impossible values. {HINT}")
    if s.correction is not None and (
        not s.correction["training_seasons"] or s.correction["params"] != GBM_PARAMS
    ):
        # The correction is retrained on load with this code's GBM_PARAMS, so
        # a file recorded with other parameters cannot be reproduced.
        raise SettingsError(
            "the player-model settings file records a correction this version "
            f"cannot retrain. {HINT}"
        )
    return s


def load(path: Path = SETTINGS_PATH) -> PlayerModelSettings:
    try:
        text = path.read_text()
    except FileNotFoundError:
        raise SettingsError(f"No player-model settings found at {path}. {HINT}") from None
    return from_json(text)


def save(s: PlayerModelSettings, path: Path = SETTINGS_PATH) -> None:
    path.write_text(to_json(s))
