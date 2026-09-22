from __future__ import annotations

import os
from dataclasses import dataclass
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
