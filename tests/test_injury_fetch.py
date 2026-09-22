from datetime import UTC, date, datetime

import pytest

from predictor import raw_store
from predictor.config import Settings
from predictor.sources import injury_report


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(raw_store, "settings", s)
    return s


class FakeResponse:
    def __init__(self, status_code, content=b""):
        self.status_code = status_code
        self.content = content


class FakeSession:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        return self.mapping.get(url, FakeResponse(403))


def test_report_url_matches_verified_pattern():
    url = injury_report.report_url(date(2025, 1, 15), "05PM")
    assert url == (
        "https://ak-static.cms.nba.com/referee/injury/"
        "Injury-Report_2025-01-15_05PM.pdf"
    )


def test_hour_labels_cover_all_24_slots():
    assert len(injury_report.HOUR_LABELS) == 24
    assert "12AM" in injury_report.HOUR_LABELS
    assert "05PM" in injury_report.HOUR_LABELS


def test_fetch_returns_none_on_403():
    session = FakeSession({})
    assert injury_report.fetch_report(date(2018, 12, 11), "05PM", session) is None


def test_fetch_returns_bytes_on_200():
    day, hour = date(2025, 1, 15), "05PM"
    url = injury_report.report_url(day, hour)
    session = FakeSession({url: FakeResponse(200, b"%PDF-1.4 data")})
    assert injury_report.fetch_report(day, hour, session) == b"%PDF-1.4 data"


def test_archive_report_stores_once_and_skips_refetch(store):
    day, hour = date(2025, 1, 15), "05PM"
    url = injury_report.report_url(day, hour)
    session = FakeSession({url: FakeResponse(200, b"%PDF-1.4 data")})

    assert injury_report.archive_report(day, hour, session=session) is True
    assert injury_report.archive_report(day, hour, session=session) is False
    assert len(session.calls) == 1, "already-archived report was re-fetched"


def test_archive_rejects_non_pdf_payload(store):
    day, hour = date(2025, 1, 15), "05PM"
    url = injury_report.report_url(day, hour)
    session = FakeSession({url: FakeResponse(200, b"<html>error</html>")})
    assert injury_report.archive_report(day, hour, session=session) is False
