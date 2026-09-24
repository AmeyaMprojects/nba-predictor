from datetime import UTC, datetime, timedelta

import pytest

from predictor import db, status

NOW = datetime(2025, 1, 15, 20, 0, tzinfo=UTC)


@pytest.fixture
def con(tmp_path):
    c = db.connect(tmp_path / "t.duckdb")
    db.migrate(c)
    return c


def _add_injury(con, observed_at, player="x"):
    con.execute(
        "INSERT INTO injury_status_raw (report_date, game_date, team, player, status, observed_at)"
        " VALUES (?,?,?,?,?,?)",
        [observed_at.date(), observed_at.date(), "LAL", player, "Out", observed_at],
    )


def test_empty_source_is_reported_stale_with_advice(con):
    health = {h.name: h for h in status.check_sources(con, NOW)}
    injuries = health["injury_status"]
    assert injuries.row_count == 0
    assert injuries.stale is True
    assert injuries.advice


def test_fresh_source_is_not_stale(con):
    _add_injury(con, NOW - timedelta(hours=1))
    health = {h.name: h for h in status.check_sources(con, NOW)}
    assert health["injury_status"].stale is False


def test_old_source_is_flagged_stale(con):
    _add_injury(con, NOW - timedelta(days=5))
    health = {h.name: h for h in status.check_sources(con, NOW)}
    injuries = health["injury_status"]
    assert injuries.stale is True
    assert injuries.age_hours == pytest.approx(120, abs=1)


def test_report_is_plain_english_and_names_problem_sources(con):
    _add_injury(con, NOW - timedelta(days=5))
    text = status.format_report(status.check_sources(con, NOW))
    assert "injury_status" in text
    assert "STALE" in text
    assert "OK" in text or "stale" in text.lower()


def test_report_leads_with_overall_verdict(con):
    text = status.format_report(status.check_sources(con, NOW))
    assert text.splitlines()[0].startswith(("PROBLEMS", "ALL OK"))


def test_all_four_logical_sources_are_reported(con):
    health = status.check_sources(con, NOW)
    names = {h.name for h in health}
    assert names == {"games", "injury_status", "odds_snapshots", "news_items"}


def test_report_names_every_source(con):
    _add_injury(con, NOW - timedelta(hours=1))
    text = status.format_report(status.check_sources(con, NOW))
    for name in ("games", "injury_status", "odds_snapshots", "news_items"):
        assert name in text


def test_odds_advice_mentions_api_key_when_unset(con, monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    health = {h.name: h for h in status.check_sources(con, NOW)}
    assert "ODDS_API_KEY" in health["odds_snapshots"].advice


def test_odds_advice_differs_when_key_is_set_but_still_empty(con, monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "dummy-key-for-test")
    health = {h.name: h for h in status.check_sources(con, NOW)}
    advice = health["odds_snapshots"].advice
    assert advice
    assert "get a free key" not in advice.lower()
