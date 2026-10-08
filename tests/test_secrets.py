"""Secret readers and the tracked-file secret scan.

Every test here points HOME at a tmp directory, so the owner's real key
files are never read, created or printed.
"""

import json
import re
import subprocess
from pathlib import Path

import pytest

from predictor import config
from predictor.config import PROJECT_ROOT

FAKE_KEY = "0123456789abcdef0123456789abcdef"


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    return h


# --- odds_api_key ----------------------------------------------------------


def test_odds_key_path_is_under_the_home_config_dir(home):
    assert config.odds_api_key_path() == home / ".config" / "predictor" / "odds_api_key"


def test_odds_key_missing_everywhere_gives_none(home):
    assert config.odds_api_key() is None


def test_odds_key_read_from_file_and_stripped(home):
    path = home / ".config" / "predictor" / "odds_api_key"
    path.parent.mkdir(parents=True)
    path.write_text(f"  {FAKE_KEY}\n")
    assert config.odds_api_key() == FAKE_KEY


def test_odds_key_file_wins_over_env(home, monkeypatch):
    path = home / ".config" / "predictor" / "odds_api_key"
    path.parent.mkdir(parents=True)
    path.write_text("from-file\n")
    monkeypatch.setenv("ODDS_API_KEY", "from-env")
    assert config.odds_api_key() == "from-file"


def test_odds_key_falls_back_to_env(home, monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "from-env")
    assert config.odds_api_key() == "from-env"


def test_empty_odds_key_file_falls_back_to_env_then_none(home, monkeypatch):
    path = home / ".config" / "predictor" / "odds_api_key"
    path.parent.mkdir(parents=True)
    path.write_text("   \n")
    assert config.odds_api_key() is None
    monkeypatch.setenv("ODDS_API_KEY", "from-env")
    assert config.odds_api_key() == "from-env"


# --- kaggle_credentials ------------------------------------------------------


def test_kaggle_path_is_the_standard_location(home):
    assert config.kaggle_credentials_path() == home / ".kaggle" / "kaggle.json"


def test_kaggle_missing_gives_none(home):
    assert config.kaggle_credentials() is None


def test_kaggle_read_from_file(home):
    path = home / ".kaggle" / "kaggle.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"username": "someone", "key": "secret"}))
    assert config.kaggle_credentials() == ("someone", "secret")


def test_kaggle_file_wins_over_env(home, monkeypatch):
    path = home / ".kaggle" / "kaggle.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"username": "file-user", "key": "file-key"}))
    monkeypatch.setenv("KAGGLE_USERNAME", "env-user")
    monkeypatch.setenv("KAGGLE_KEY", "env-key")
    assert config.kaggle_credentials() == ("file-user", "file-key")


def test_kaggle_falls_back_to_env(home, monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "env-user")
    monkeypatch.setenv("KAGGLE_KEY", "env-key")
    assert config.kaggle_credentials() == ("env-user", "env-key")


@pytest.mark.parametrize("content", ["not json", "[]", '{"username": "u"}', '{"username": "", "key": "k"}'])
def test_unusable_kaggle_file_never_raises(home, content):
    path = home / ".kaggle" / "kaggle.json"
    path.parent.mkdir()
    path.write_text(content)
    assert config.kaggle_credentials() is None


# --- secret scan of tracked files ---------------------------------------------

# A 32-hex run assigned to something named like a key (`apiKey=...`,
# `key: "..."`, `ODDS_API_KEY = '...'`), and the Kaggle token shape
# `"key": "<32 hex>"`. Both Odds API keys and Kaggle keys are 32 hex chars.
_KEY_ASSIGNMENT = re.compile(
    r"""(?ix)
    (?:api_?key|\bkey)        # apiKey, api_key, ODDS_API_KEY, key
    ["']?\s*[:=]\s*["']?      # =, :, with optional quotes around
    [0-9a-f]{32}(?![0-9a-f])  # exactly a 32-hex run
    """
)
_KAGGLE_JSON = re.compile(r'"key"\s*:\s*"[0-9a-f]{32}"', re.IGNORECASE)


def scan_tracked_files(repo: Path) -> list[str]:
    """Every `path:line` in a git-tracked file holding a key-shaped secret."""
    names = subprocess.run(
        ["git", "ls-files", "-z"], cwd=repo, check=True, capture_output=True
    ).stdout.decode().split("\0")
    hits: list[str] = []
    for name in filter(None, names):
        path = repo / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if _KEY_ASSIGNMENT.search(line) or _KAGGLE_JSON.search(line):
                hits.append(f"{name}:{number}")
    return hits


def _temp_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "clean.py").write_text("API_URL = 'https://example.invalid'\n")
    return repo


def test_scan_passes_on_a_clean_temp_repo(tmp_path):
    repo = _temp_repo(tmp_path)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    assert scan_tracked_files(repo) == []


@pytest.mark.parametrize(
    "planted",
    [
        f"ODDS_API_KEY = '{FAKE_KEY}'\n",
        f"url = 'https://x.invalid/odds?apiKey={FAKE_KEY}'\n",
        json.dumps({"username": "u", "key": FAKE_KEY}) + "\n",
    ],
)
def test_scan_fails_on_a_planted_key_in_a_tracked_file(tmp_path, planted):
    repo = _temp_repo(tmp_path)
    (repo / "leak.txt").write_text(planted)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    assert scan_tracked_files(repo) == ["leak.txt:1"]


def test_scan_ignores_untracked_files(tmp_path):
    repo = _temp_repo(tmp_path)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    (repo / "untracked.txt").write_text(f"apiKey={FAKE_KEY}\n")
    assert scan_tracked_files(repo) == []


def test_no_key_shaped_secret_in_any_tracked_file_of_this_repo():
    hits = scan_tracked_files(PROJECT_ROOT)
    assert hits == [], f"key-shaped secret in tracked file(s): {hits}"
