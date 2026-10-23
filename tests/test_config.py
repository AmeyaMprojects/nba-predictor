from datetime import UTC, datetime
from pathlib import Path

from predictor.config import Settings, previous_season_label, season_label


def test_settings_derive_paths_from_data_dir(tmp_path):
    s = Settings(data_dir=tmp_path)
    assert s.raw_dir == tmp_path / "raw"
    assert s.db_path == tmp_path / "predictor.duckdb"


def test_ensure_dirs_creates_them(tmp_path):
    s = Settings(data_dir=tmp_path / "nested")
    s.ensure_dirs()
    assert s.raw_dir.is_dir()


def test_season_label_from_july_is_the_upcoming_season():
    assert season_label(datetime(2026, 9, 27, tzinfo=UTC)) == "2026-27"
    assert season_label(datetime(2026, 7, 1, tzinfo=UTC)) == "2026-27"


def test_season_label_before_july_is_the_season_in_progress():
    assert season_label(datetime(2027, 3, 1, tzinfo=UTC)) == "2026-27"
    assert season_label(datetime(2026, 6, 30, tzinfo=UTC)) == "2025-26"


def test_season_label_across_a_century_boundary():
    assert season_label(datetime(2099, 10, 1, tzinfo=UTC)) == "2099-00"


def test_previous_season_label():
    assert previous_season_label("2026-27") == "2025-26"
    assert previous_season_label("2000-01") == "1999-00"


def test_publish_branch_defaults_to_main(monkeypatch):
    from predictor.config import publish_branch

    monkeypatch.delenv("PREDICTOR_PUBLISH_BRANCH", raising=False)
    assert publish_branch() == "main"


def test_publish_branch_reads_cloud_shadow(monkeypatch):
    from predictor.config import publish_branch

    monkeypatch.setenv("PREDICTOR_PUBLISH_BRANCH", " cloud-shadow\n")
    assert publish_branch() == "cloud-shadow"


def test_publish_branch_rejects_other_values(monkeypatch):
    import pytest

    from predictor.config import publish_branch

    monkeypatch.setenv("PREDICTOR_PUBLISH_BRANCH", "dev")
    with pytest.raises(ValueError, match="PREDICTOR_PUBLISH_BRANCH must be main or cloud-shadow, not 'dev'"):
        publish_branch()
