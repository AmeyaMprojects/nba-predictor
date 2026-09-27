from __future__ import annotations

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
