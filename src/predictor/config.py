from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    data_dir: Path

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "predictor.duckdb"

    def ensure_dirs(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)


def _default_data_dir() -> Path:
    env = os.environ.get("PREDICTOR_DATA_DIR")
    return Path(env).expanduser() if env else PROJECT_ROOT / "data"


settings = Settings(data_dir=_default_data_dir())


def season_label(now: datetime) -> str:
    """Season string (e.g. "2026-27") that `now` belongs to, for defaults.

    NBA seasons start in October and are labeled by their two years. From
    July onward the upcoming season is the relevant one; before July, the
    season already in progress (started the previous October) is.
    """
    start_year = now.year if now.month >= 7 else now.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def previous_season_label(label: str) -> str:
    """"2026-27" -> "2025-26"."""
    start_year = int(label[:4]) - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


# --- secrets ---------------------------------------------------------------
#
# Keys live in the owner's home directory, never inside the repository
# (tests/test_secrets.py fails on any key-shaped string in a tracked file).
# Paths are resolved on every call (not at import) so a test that points
# HOME at a temp directory never sees the real files. Neither reader ever
# prints, logs or raises on a missing/unreadable file -- None means "no
# usable secret", and the caller prints a plain message naming the path.


def odds_api_key_path() -> Path:
    """Where the Odds API key lives: ``~/.config/predictor/odds_api_key``."""
    return Path.home() / ".config" / "predictor" / "odds_api_key"


def kaggle_credentials_path() -> Path:
    """Kaggle's own token location: ``~/.kaggle/kaggle.json``."""
    return Path.home() / ".kaggle" / "kaggle.json"


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def odds_api_key() -> str | None:
    """The Odds API key: the key file (stripped) first, then env ``ODDS_API_KEY``.

    An empty file counts as missing. Returns None when neither is set.
    """
    text = _read_text(odds_api_key_path())
    if text and text.strip():
        return text.strip()
    env = os.environ.get("ODDS_API_KEY", "").strip()
    return env or None


def kaggle_credentials() -> tuple[str, str] | None:
    """``(username, key)`` from ``~/.kaggle/kaggle.json``, then env
    ``KAGGLE_USERNAME``/``KAGGLE_KEY`` (Kaggle's own convention).

    A missing, unreadable, malformed or incomplete file counts as missing.
    Returns None when no complete pair is found anywhere.
    """
    text = _read_text(kaggle_credentials_path())
    if text is not None:
        try:
            data = json.loads(text)
        except ValueError:
            data = None
        if isinstance(data, dict):
            username, key = data.get("username"), data.get("key")
            if isinstance(username, str) and isinstance(key, str):
                if username.strip() and key.strip():
                    return username.strip(), key.strip()
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()
    if username and key:
        return username, key
    return None
