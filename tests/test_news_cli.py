"""CLI-level tests for `poll-news`.

Stubs out `news_rss.poll_all` and `news_rss.ingest_archived_news` entirely
-- the polling/archiving logic is covered exhaustively in test_news_rss.py
and the ingestion logic in test_news_ingest.py -- so these tests only
exercise the CLI's own output formatting and its decision of when to exit
non-zero (Task 12 state note: a dead feed must still report as FAILED, not
as "0 new", and the command must exit non-zero when any feed failed).
"""

from typer.testing import CliRunner

from predictor import cli, config, db
from predictor.config import Settings
from predictor.sources import news_rss

runner = CliRunner()


def _point_settings_at_tmp(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    # cli.py does `from predictor.config import settings` freshly inside the
    # command body, so patching the module attribute is enough for that
    # call site; db.connect() references its own module-level `settings`
    # name bound at db.py's import time, so that needs patching separately
    # -- same pattern test_injury_cli.py uses.
    monkeypatch.setattr(config, "settings", s)
    monkeypatch.setattr(db, "settings", s)
    return s


def _result(ok, new=0, skipped=0, conflicts=0, warning=None, error=None):
    return news_rss.FeedResult(
        feed="x", ok=ok, new=new, skipped=skipped, conflicts=conflicts,
        warning=warning, error=error,
    )


def test_poll_news_reports_failed_feed_and_exits_nonzero(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)
    monkeypatch.setattr(
        news_rss,
        "poll_all",
        lambda: {
            "espn": _result(True, new=3),
            "deadfeed": _result(False, error="HTTP 404"),
        },
    )
    monkeypatch.setattr(news_rss, "ingest_archived_news", lambda con: 3)

    result = runner.invoke(cli.app, ["poll-news"])

    assert "espn: 3 new" in result.stdout
    assert "deadfeed: FAILED - HTTP 404" in result.stdout
    assert "deadfeed: 0 new" not in result.stdout
    assert "news_items rows: 3" in result.stdout
    assert result.exit_code == 1


def test_poll_news_exits_zero_when_all_feeds_succeed(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)
    monkeypatch.setattr(
        news_rss,
        "poll_all",
        lambda: {"espn": _result(True, new=1), "cbs": _result(True, new=0)},
    )
    monkeypatch.setattr(news_rss, "ingest_archived_news", lambda con: 1)

    result = runner.invoke(cli.app, ["poll-news"])

    assert result.exit_code == 0
    assert "espn: 1 new" in result.stdout
    assert "cbs: 0 new" in result.stdout
    assert "news_items rows: 1" in result.stdout


def test_poll_news_loads_the_real_archive_into_the_database(tmp_path, monkeypatch):
    """End-to-end (no network): archives two items, then confirms poll-news
    actually loads them into news_items_raw via the real ingest_archived_news.
    """
    from datetime import UTC, datetime
    from types import SimpleNamespace

    s = _point_settings_at_tmp(tmp_path, monkeypatch)
    from predictor import raw_store

    monkeypatch.setattr(raw_store, "settings", s)

    now = datetime(2026, 1, 2, 3, 4, tzinfo=UTC)
    entries = SimpleNamespace(
        entries=[
            SimpleNamespace(id="a", title="t a", link="http://x/a", summary="s"),
            SimpleNamespace(id="b", title="t b", link="http://x/b", summary="s"),
        ]
    )

    def _fake_poll_all():
        # Archives directly (no network) and reports it the way poll_all
        # normally would, so poll-news's own ingest_archived_news call has
        # real archived data to load.
        stats = news_rss.archive_entries("espn", entries, now)
        return {"espn": _result(True, new=stats.new)}

    monkeypatch.setattr(news_rss, "poll_all", _fake_poll_all)

    result = runner.invoke(cli.app, ["poll-news"])

    assert result.exit_code == 0
    assert "news_items rows: 2" in result.stdout

    con = db.connect(s.db_path)
    table = db.POINT_IN_TIME_TABLES["news_items"]
    count = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    assert count == 2
