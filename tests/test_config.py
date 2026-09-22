from pathlib import Path
from predictor.config import Settings


def test_settings_derive_paths_from_data_dir(tmp_path):
    s = Settings(data_dir=tmp_path)
    assert s.raw_dir == tmp_path / "raw"
    assert s.db_path == tmp_path / "predictor.duckdb"


def test_ensure_dirs_creates_them(tmp_path):
    s = Settings(data_dir=tmp_path / "nested")
    s.ensure_dirs()
    assert s.raw_dir.is_dir()
