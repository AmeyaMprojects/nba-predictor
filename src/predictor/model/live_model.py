"""Which model makes the live calls (plan 3, Task 6).

A committed flag file, ``state/settings/live_model.json``, picks the model:

    {"model": "player", "from_slate": "2026-11-10"}

- No file means the current model (``"team"``, logged as ``v1``) for every
  call, exactly as before this plan. The switch ships off.
- ``"player"`` (``v2``, ``PlayerStage1Predictor``) applies to slates on or
  after ``from_slate`` (required for player).
- Anything else is an error (``LiveModelError``): a broken flag never falls
  back silently to either model.

A game keeps the model of its first call (``kind_for_game``): if the flag
turns on mid-slate, games already called stay on their first model for every
later revision and final.

The two callers give one ``CallOutcome`` shape, so ``live.predict_today``
writes a line the same way for either model. Live calls read reports as of
the call time: the 15-minute report lag (``player_sim.REPLAY_REPORT_LAG``)
belongs to the replay only.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, get_args

from predictor.backtest.baselines import GameToPredict
from predictor.model import player_settings
from predictor.model.calls import CallType, GameChain
from predictor.model.player_settings import PlayerModelSettings, SettingsError
from predictor.model.player_stage1 import PlayerStage1Predictor
from predictor.model.settings import ModelSettings
from predictor.model.stage1 import Stage1Predictor

LIVE_MODEL_FILE = Path("settings/live_model.json")  # relative to the state dir
_FIX = "Fix state/settings/live_model.json."

ModelKind = Literal["team", "player"]
_KINDS: tuple[str, ...] = get_args(ModelKind)
MODEL_VERSION: dict[str, str] = {"team": "v1", "player": "v2"}
_KIND_OF_VERSION = {v: k for k, v in MODEL_VERSION.items()}
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Which availability table each call reads: revisions are made in the final
# call's information state, so they use the final table.
_TABLE_FOR: dict[str, str] = {"early": "early", "revision": "final", "final": "final"}


class LiveModelError(Exception):
    """The live-model flag (or what it needs) is unusable."""


@dataclass(frozen=True)
class LiveModel:
    kind: ModelKind
    from_slate: date | None

    @staticmethod
    def team() -> LiveModel:
        return LiveModel("team", None)


def _error(what: str) -> LiveModelError:
    return LiveModelError(f"{what} {_FIX}")


def parse_live_model(text: str) -> LiveModel:
    """The flag file's text as a ``LiveModel``; ``LiveModelError`` if it is
    not exactly ``{"model": "team"|"player", "from_slate": "YYYY-MM-DD"}``
    (``from_slate`` optional, or null, for team)."""
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _error(f"the live-model flag is not valid JSON ({exc}).") from None
    if not isinstance(doc, dict):
        raise _error("the live-model flag must be a JSON object.")
    extra = sorted(set(doc) - {"model", "from_slate"})
    if extra:
        raise _error(f"the live-model flag has unknown key(s) {extra}.")
    kind = doc.get("model")
    if kind not in _KINDS:
        raise _error(f"the live-model flag's model must be one of {list(_KINDS)}, got {kind!r}.")
    raw = doc.get("from_slate")
    if raw is None:
        if kind == "player":
            raise _error("the live-model flag needs from_slate (YYYY-MM-DD) for the player model.")
        return LiveModel(kind, None)
    if not isinstance(raw, str) or not _ISO_DATE.match(raw):
        raise _error(f"the live-model flag's from_slate must be YYYY-MM-DD, got {raw!r}.")
    try:
        from_slate = date.fromisoformat(raw)
    except ValueError:
        raise _error(f"the live-model flag's from_slate is not a date: {raw!r}.") from None
    return LiveModel(kind, from_slate)


def load_live_model(state_dir: Path) -> LiveModel:
    """The flag under ``state_dir``; a missing file is the current model."""
    path = state_dir / LIVE_MODEL_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return LiveModel.team()
    except (OSError, UnicodeDecodeError) as exc:
        raise _error(f"could not read the live-model flag at {path} ({exc}).") from None
    return parse_live_model(text)


@dataclass(frozen=True)
class Published:
    """What ``main`` publishes: the live-model flag, whether the cloud already
    has calls for the slate, and whether origin could be reached."""

    live: LiveModel
    cloud_lines: bool
    fetched: bool = True


_FETCH_TIMEOUT = 60.0
_SHOW_TIMEOUT = 30.0


def read_published(
    repo_dir: Path, slate: date, season: str, runner=None,
) -> Published:
    """The flag and the slate's cloud lines as ``origin/main`` has them (after
    a fetch). If the fetch fails the local ``HEAD`` files are read instead and
    ``fetched`` is False. Never prompts and never hangs (timeouts, no tty).
    Missing files mean the team model / no lines; a broken flag raises
    ``LiveModelError``."""
    from predictor.model.publish import _GIT_ENV_OVERRIDES

    run = runner if runner is not None else subprocess.run
    env = {**os.environ, **_GIT_ENV_OVERRIDES}

    def git(args: list[str], timeout: float):
        try:
            return run(
                ["git", *args], cwd=repo_dir, capture_output=True, text=True,
                timeout=timeout, stdin=subprocess.DEVNULL, env=env,
            )
        except (subprocess.TimeoutExpired, OSError):
            return None

    r = git(["fetch", "origin", "+refs/heads/main:refs/remotes/origin/main"], _FETCH_TIMEOUT)
    fetched = r is not None and r.returncode == 0
    ref = "origin/main" if fetched else "HEAD"

    def show(path: str) -> str | None:
        out = git(["show", f"{ref}:{path}"], _SHOW_TIMEOUT)
        return out.stdout if out is not None and out.returncode == 0 else None

    flag = show("state/settings/live_model.json")
    live = parse_live_model(flag) if flag is not None else LiveModel.team()
    log = show(f"predictions/{season}.jsonl") or ""
    day = slate.isoformat()
    cloud = False
    for raw in log.splitlines():
        try:
            line = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if (isinstance(line, dict) and line.get("game_date") == day
                and line.get("state_version") is not None):
            cloud = True
            break
    return Published(live, cloud, fetched)


def kind_for_slate(live: LiveModel, slate: date) -> ModelKind:
    if live.kind == "player" and live.from_slate is not None and live.from_slate <= slate:
        return "player"
    return "team"


def kind_for_game(live: LiveModel, slate: date, chain: GameChain | None) -> ModelKind:
    """The model of the game's first ``predicted`` line (no
    ``model_version`` means v1); the slate's model when it has none."""
    first = chain.first_predicted if chain is not None else None
    if first is None:
        return kind_for_slate(live, slate)
    version = first.get("model_version") or MODEL_VERSION["team"]
    kind = _KIND_OF_VERSION.get(version)
    if kind is None:
        raise _error(
            f"game {first.get('game_id')} was first called by model {version!r}, "
            "which this code does not know."
        )
    return kind  # type: ignore[return-value]


def load_player_settings(
    path: Path = player_settings.SETTINGS_PATH,
) -> PlayerModelSettings:
    """The player settings for live calls. A correction model is refused:
    a live correction would be trained differently from the validated one."""
    try:
        s = player_settings.load(path)
    except SettingsError as exc:
        raise _error(f"the player model is unusable: {exc}") from None
    _refuse_correction(s)
    return s


def _refuse_correction(s: PlayerModelSettings) -> None:
    if s.correction is not None:
        raise _error(
            "the player settings include a correction model, which live calls do not "
            "support; refit without it. Or set the flag back to the team model."
        )


def team_settings_dict(settings: ModelSettings) -> dict:
    """The ``settings`` field of a v1 line (unchanged keys; ``home_shrinkage``
    only when home court is per team)."""
    out = {
        "k": settings.ratings.k,
        "margin_cap": settings.ratings.margin_cap,
        "season_regression": settings.ratings.season_regression,
        "hca_window": settings.ratings.hca_window,
        "sigma": settings.sigma,
        "half_life": settings.half_life,
    }
    if not settings.home.flat:
        out["home_shrinkage"] = settings.home.shrinkage
    return out


def player_settings_dict(s: PlayerModelSettings) -> dict:
    """The ``settings`` field of a v2 line (flat home court is None, never
    ``inf``, so the log stays strict JSON)."""
    return {
        "model": "player",
        "k": s.team.k,
        "margin_cap": s.team.margin_cap,
        "season_regression": s.team.season_regression,
        "hca_window": s.team.hca_window,
        "formula": s.formula,
        "w": s.w,
        "player_half_life_minutes": s.player.half_life_minutes,
        "player_season_regression": s.player.season_regression,
        "home_shrinkage": None if s.home.flat else s.home.shrinkage,
        "sigma": s.sigma,
        "half_life": s.half_life,
    }


@dataclass(frozen=True)
class CallOutcome:
    spread: float
    p_home: float
    sentence: str
    terms: dict[str, float]
    availability: list[dict] | None
    model_version: str
    settings: dict
    sigma: float


class TeamCaller:
    """The current model (v1): the call kind does not change the numbers."""

    def __init__(self, con, settings: ModelSettings) -> None:
        self.settings = settings
        self.predictor = Stage1Predictor(con, settings)
        self._settings_dict = team_settings_dict(settings)

    def call(self, game: GameToPredict, view, kind: CallType) -> CallOutcome:
        b = self.predictor.explain(game, view)
        return CallOutcome(
            spread=b.spread,
            p_home=b.p_home,
            sentence=b.sentence(),
            terms=dict(b.terms()),
            availability=None,
            model_version=MODEL_VERSION["team"],
            settings=dict(self._settings_dict),
            sigma=self.settings.sigma,
        )


class PlayerCaller:
    """The player model (v2). Reports are read as of the view's time (no
    lag). ``id_links`` are the cloud's provisional-id links
    (``cloud.state.read_id_links(state_dir)``), so ESPN box rows stored
    under a provisional id count for their game."""

    def __init__(
        self, con, settings: PlayerModelSettings, *,
        id_links: Mapping[str, str] | None = None,
    ) -> None:
        _refuse_correction(settings)
        self.settings = settings
        self.predictor = PlayerStage1Predictor(con, settings, id_links=id_links)
        self._settings_dict = player_settings_dict(settings)

    def call(self, game: GameToPredict, view, kind: CallType) -> CallOutcome:
        table = _TABLE_FOR.get(kind)
        if table is None:
            raise ValueError(f"kind must be one of {sorted(_TABLE_FOR)}, got {kind!r}")
        b = self.predictor.explain(game, view, table)
        return CallOutcome(
            spread=b.spread,
            p_home=b.p_home,
            sentence=b.sentence(),
            terms=dict(b.terms()),
            availability=[
                {"team": team, "player": name, "status": status}
                for team, name, status in b.absent
            ],
            model_version=MODEL_VERSION["player"],
            settings=dict(self._settings_dict),
            sigma=self.settings.sigma,
        )


def make_caller(
    con,
    kind: ModelKind,
    settings: ModelSettings,
    player_settings: PlayerModelSettings | None,
    *,
    id_links: Mapping[str, str] | None = None,
) -> TeamCaller | PlayerCaller:
    if kind == "team":
        return TeamCaller(con, settings)
    if kind == "player":
        if player_settings is None:
            raise _error("the player model is selected but no player settings were loaded.")
        return PlayerCaller(con, player_settings, id_links=id_links)
    raise ValueError(f"kind must be one of {list(_KINDS)}, got {kind!r}")
