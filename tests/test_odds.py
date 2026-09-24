from datetime import UTC, datetime

import pytest
from typer.testing import CliRunner

from predictor import cli, config, db, raw_store
from predictor.config import Settings
from predictor.sources import odds

runner = CliRunner()

NOW = datetime(2025, 1, 15, 20, 0, tzinfo=UTC)

PAYLOAD = [
    {
        "id": "abc123",
        "home_team": "Philadelphia 76ers",
        "away_team": "New York Knicks",
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Philadelphia 76ers", "price": -150},
                            {"name": "New York Knicks", "price": 130},
                        ],
                    }
                ],
            }
        ],
    }
]


def test_parse_extracts_one_row_per_bookmaker():
    rows = odds.parse_odds_payload(PAYLOAD, NOW)
    assert len(rows) == 1
    row = rows[0]
    assert row["book"] == "draftkings"
    assert row["home_price"] == -150
    assert row["away_price"] == 130
    assert row["observed_at"] == NOW


def test_parse_skips_events_without_bookmakers():
    assert odds.parse_odds_payload([{"id": "x", "home_team": "A", "away_team": "B"}], NOW) == []


def test_ingest_writes_rows(tmp_path, monkeypatch):
    # ingest_current() also archives via raw_store.store(), which reads its
    # own module-level `settings` (bound at import time via `from
    # predictor.config import settings`) rather than predictor.config's --
    # patching db.settings alone is not enough to keep this off the real
    # data/ directory. Same pattern as tests/test_injury_fetch.py and
    # tests/test_news_rss.py.
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(raw_store, "settings", s)

    con = db.connect(tmp_path / "t.duckdb")
    db.migrate(con)
    monkeypatch.setattr(odds, "fetch_current", lambda key, session=None: PAYLOAD)
    assert odds.ingest_current(con, api_key="k", now=NOW) == 1
    # odds_snapshots_raw is the physical table (see
    # predictor.db.POINT_IN_TIME_TABLES); tests are exempt from the
    # no-physical-table-name-outside-db-and-asof rule (see
    # test_no_physical_table_name_appears_outside_db_and_asof), same as
    # test_nba_stats.py and test_injury_parse.py.
    stored = con.execute("SELECT book, home_price FROM odds_snapshots_raw").fetchone()
    assert stored == ("draftkings", -150)


def test_missing_api_key_raises_a_clear_error(tmp_path, monkeypatch):
    con = db.connect(tmp_path / "t.duckdb")
    db.migrate(con)
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ODDS_API_KEY"):
        odds.ingest_current(con)


class FakeResponse:
    status_code = 401
    text = "quota"

    def json(self):
        return {}


def test_quota_exhaustion_raises_named_error():
    session = type("S", (), {"get": lambda self, *a, **k: FakeResponse()})()
    with pytest.raises(odds.OddsQuotaExceeded):
        odds.fetch_current("k", session=session)


def test_cli_missing_key_prints_plain_english_error_not_a_traceback(tmp_path, monkeypatch):
    """The end user does not write code, so a missing ODDS_API_KEY must show
    up as one readable line telling them where to get a free key -- never
    an uncaught ValueError traceback. Mirrors the settings-patching pattern
    in tests/test_nba_cli.py.
    """
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(config, "settings", s)
    monkeypatch.setattr(db, "settings", s)
    monkeypatch.delenv("ODDS_API_KEY", raising=False)

    result = runner.invoke(cli.app, ["ingest-odds"])

    assert result.exit_code == 1
    assert "ODDS_API_KEY" in result.output
    assert "the-odds-api.com" in result.output
    assert "Traceback" not in result.output
