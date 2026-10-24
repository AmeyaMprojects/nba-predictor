"""The latest 15-minute injury report: slot names, step-back, archive."""

from datetime import UTC, date, datetime

import pytest
import requests

from predictor import raw_store
from predictor.config import Settings
from predictor.sources import injury_report
from predictor.sources.injury_report import (
    BASE_URL,
    BROWSER_USER_AGENT,
    LatestReport,
    TransientFetchError,
    fetch_latest,
    quarter_slot,
    raw_key_15,
    report_url_15,
)

PDF = b"%PDF-1.7 fake report body"


def dt(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(raw_store, "settings", s)
    monkeypatch.setattr(injury_report, "_sleep", lambda seconds: None)
    return s


class FakeResponse:
    def __init__(self, status_code, content=b""):
        self.status_code = status_code
        self.content = content
        self.headers = {}


class FakeSession:
    def __init__(self, mapping=None, default=403):
        self.mapping = mapping or {}
        self.default = default
        self.calls: list[str] = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        if url in self.mapping:
            return FakeResponse(*self.mapping[url])
        return FakeResponse(self.default)


def test_quarter_slot_names_the_15_minute_file():
    assert quarter_slot(dt("2026-10-22T21:37Z")) == (date(2026, 10, 22), "05_30PM")  # EDT
    assert quarter_slot(dt("2026-11-03T22:37Z")) == (date(2026, 11, 3), "05_30PM")  # EST
    assert quarter_slot(dt("2026-10-23T04:05Z")) == (date(2026, 10, 23), "12_00AM")
    assert quarter_slot(dt("2026-10-22T16:00Z")) == (date(2026, 10, 22), "12_00PM")


def test_quarter_slot_on_the_dst_change_day():
    # 2026-11-01 06:10Z is 01:10 EST (after the fall-back) -> 01_00AM
    assert quarter_slot(dt("2026-11-01T06:10Z")) == (date(2026, 11, 1), "01_00AM")
    # 05:50Z is 01:50 EDT (before it)
    assert quarter_slot(dt("2026-11-01T05:50Z")) == (date(2026, 11, 1), "01_45AM")


def test_quarter_slot_rejects_naive_time():
    with pytest.raises(ValueError):
        quarter_slot(datetime(2026, 10, 22, 21, 37))


def test_url_and_key_for_15_minute_slot():
    assert report_url_15(date(2026, 10, 22), "05_30PM") == (
        f"{BASE_URL}/Injury-Report_2026-10-22_05_30PM.pdf"
    )
    assert raw_key_15(date(2026, 10, 22), "05_30PM") == "Injury-Report_2026-10-22_05_30PM.pdf"


def test_fetch_latest_steps_back_over_unpublished_slots():
    now = dt("2026-10-22T21:37Z")
    session = FakeSession(
        {
            report_url_15(date(2026, 10, 22), "05_30PM"): (403,),
            report_url_15(date(2026, 10, 22), "05_15PM"): (200, PDF),
        }
    )
    latest = fetch_latest(now, session=session)
    assert isinstance(latest, LatestReport)
    assert latest.key == "Injury-Report_2026-10-22_05_15PM.pdf"
    assert latest.body == PDF
    assert latest.fetched_at == now
    assert session.calls == [
        report_url_15(date(2026, 10, 22), "05_30PM"),
        report_url_15(date(2026, 10, 22), "05_15PM"),
    ]
    # archived before it is returned
    assert raw_store.load("injury", latest.key) == PDF


def test_fetch_latest_treats_404_as_unpublished():
    now = dt("2026-10-22T21:37Z")
    session = FakeSession({report_url_15(date(2026, 10, 22), "05_00PM"): (200, PDF)}, default=404)
    latest = fetch_latest(now, session=session)
    assert latest.key == "Injury-Report_2026-10-22_05_00PM.pdf"
    assert len(session.calls) == 3


def test_fetch_latest_sends_a_browser_user_agent(monkeypatch):
    seen = []

    def fake_get(self, url, timeout=None, **kwargs):
        seen.append(self.headers.get("User-Agent"))
        return FakeResponse(200, PDF)

    monkeypatch.setattr(requests.Session, "get", fake_get)
    latest = fetch_latest(dt("2026-10-22T21:37Z"))
    assert latest is not None
    assert seen == [BROWSER_USER_AGENT]
    assert BROWSER_USER_AGENT == (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
    )
    assert injury_report.new_session().headers["User-Agent"] == BROWSER_USER_AGENT


def test_fetch_latest_gives_up_after_five_slots():
    session = FakeSession()
    assert fetch_latest(dt("2026-10-22T21:37Z"), session=session) is None
    assert len(session.calls) == 5
    assert session.calls[-1] == report_url_15(date(2026, 10, 22), "04_30PM")


def test_fetch_latest_steps_back_over_a_non_pdf_body(capsys):
    session = FakeSession(
        {
            report_url_15(date(2026, 10, 22), "05_30PM"): (200, b"<html>blocked</html>"),
            report_url_15(date(2026, 10, 22), "05_15PM"): (200, PDF),
        }
    )
    latest = fetch_latest(dt("2026-10-22T21:37Z"), session=session)
    assert latest.key == "Injury-Report_2026-10-22_05_15PM.pdf"
    assert "non-PDF" in capsys.readouterr().out
    assert not raw_store.exists("injury", "Injury-Report_2026-10-22_05_30PM.pdf")


def test_fetch_latest_lets_transient_errors_propagate():
    session = FakeSession(default=503)
    with pytest.raises(TransientFetchError):
        fetch_latest(dt("2026-10-22T21:37Z"), session=session)


def test_fetch_latest_crosses_midnight_into_the_previous_et_day():
    now = dt("2026-10-23T04:05Z")  # 00:05 EDT
    session = FakeSession({report_url_15(date(2026, 10, 22), "11_45PM"): (200, PDF)})
    latest = fetch_latest(now, session=session)
    assert latest.key == "Injury-Report_2026-10-22_11_45PM.pdf"
