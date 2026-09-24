import time
from datetime import UTC, date, datetime

import pytest
import requests

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
    def __init__(self, status_code, content=b"", headers=None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}


class FakeSession:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        return self.mapping.get(url, FakeResponse(403))


class ConstantStatusSession:
    """Always answers with the same status, regardless of URL -- for
    exercising the retry path, where the same slot is fetched repeatedly.
    """

    def __init__(self, status_code, headers=None):
        self.status_code = status_code
        self.headers = headers or {}
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        return FakeResponse(self.status_code, headers=self.headers)


class RaisingSession:
    """Simulates a network-level failure (not an HTTP status)."""

    def __init__(self, exc):
        self.exc = exc
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        raise self.exc


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    """Every test in this module must run fast -- stub the sleep
    indirection instead of letting exponential backoff actually block.
    """
    monkeypatch.setattr(injury_report, "_sleep", lambda seconds: None)


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


def test_fetch_returns_none_on_404():
    session = ConstantStatusSession(404)
    assert injury_report.fetch_report(date(2025, 1, 15), "05PM", session) is None
    assert len(session.calls) == 1, "403/404 must not be retried"


def test_fetch_raises_transient_error_on_429_after_retries():
    session = ConstantStatusSession(429)
    with pytest.raises(injury_report.TransientFetchError) as exc_info:
        injury_report.fetch_report(date(2025, 1, 15), "05PM", session)
    assert exc_info.value.status_code == 429
    assert len(session.calls) == injury_report._MAX_ATTEMPTS, (
        "should retry up to the configured attempt limit, not once"
    )


def test_fetch_raises_transient_error_on_503_after_retries():
    session = ConstantStatusSession(503, headers={"Retry-After": "1"})
    with pytest.raises(injury_report.TransientFetchError) as exc_info:
        injury_report.fetch_report(date(2025, 1, 15), "05PM", session)
    assert exc_info.value.status_code == 503
    assert exc_info.value.retry_after == 1.0
    assert len(session.calls) == injury_report._MAX_ATTEMPTS


def test_fetch_propagates_unrelated_exceptions_immediately():
    # Python's builtin ConnectionError is a coincidentally similarly-named
    # but UNRELATED exception to requests.exceptions.ConnectionError (the
    # one _RETRYABLE_NETWORK_EXCEPTIONS actually catches -- see the
    # requests-specific tests below). This guards against a future change
    # widening the retry net by matching exception NAMES rather than the
    # specific requests.exceptions types it's scoped to: any exception
    # outside that set (this one included) must still propagate on the
    # first attempt, uncaught.
    session = RaisingSession(ConnectionError("boom"))
    with pytest.raises(ConnectionError):
        injury_report.fetch_report(date(2025, 1, 15), "05PM", session)
    assert len(session.calls) == 1, "an unrelated exception must not be retried"


def test_fetch_raises_transient_error_after_retrying_a_connection_error():
    # Regression guard (Task 8 review Finding 2): a dropped/reset TCP
    # connection previously propagated immediately and killed an
    # unattended ~2,500-slot backfill about 40% through. It must now be
    # retried the same way a transient HTTP status is, and, once retries
    # are exhausted, converted to TransientFetchError -- not left as a
    # raw requests exception -- so callers keep the documented two-outcome
    # contract ("confirmed absent" vs. "could not check").
    session = RaisingSession(
        requests.exceptions.ConnectionError(
            "('Connection aborted.', ConnectionResetError(54, 'Connection reset by peer'))"
        )
    )
    with pytest.raises(injury_report.TransientFetchError) as exc_info:
        injury_report.fetch_report(date(2025, 1, 15), "05PM", session)
    assert "ConnectionError" in str(exc_info.value)
    assert len(session.calls) == injury_report._MAX_ATTEMPTS, (
        "a dropped connection should retry up to the attempt limit, not fail immediately"
    )


def test_fetch_raises_transient_error_after_retrying_a_timeout():
    session = RaisingSession(requests.exceptions.Timeout("timed out"))
    with pytest.raises(injury_report.TransientFetchError) as exc_info:
        injury_report.fetch_report(date(2025, 1, 15), "05PM", session)
    assert "Timeout" in str(exc_info.value)
    assert len(session.calls) == injury_report._MAX_ATTEMPTS


def test_fetch_raises_transient_error_after_retrying_a_chunked_encoding_error():
    session = RaisingSession(requests.exceptions.ChunkedEncodingError("truncated response"))
    with pytest.raises(injury_report.TransientFetchError):
        injury_report.fetch_report(date(2025, 1, 15), "05PM", session)
    assert len(session.calls) == injury_report._MAX_ATTEMPTS


def test_fetch_returns_none_on_403_or_404_even_with_network_retry_enabled():
    # Regression guard: adding connection-error retry must not touch the
    # confirmed-absent 403/404 path -- it must still short-circuit on the
    # FIRST attempt. Burning _MAX_ATTEMPTS on every genuinely-absent
    # offseason slot would slow the backfill for no reason.
    for status in (403, 404):
        session = ConstantStatusSession(status)
        assert injury_report.fetch_report(date(2025, 1, 15), "05PM", session) is None
        assert len(session.calls) == 1


def test_archive_report_skips_before_archive_start_without_request(store):
    day = date(2018, 12, 11)
    session = FakeSession({})
    assert injury_report.archive_report(day, "05PM", session=session) is False
    assert session.calls == [], "a pre-ARCHIVE_START date must not make a request"


def test_retry_after_clamped_to_backoff_ceiling():
    response = FakeResponse(503, headers={"Retry-After": "999999999999"})
    assert injury_report._retry_after_seconds(response) == injury_report._MAX_WAIT_SECONDS


def test_retry_after_rejects_negative_value():
    response = FakeResponse(503, headers={"Retry-After": "-5"})
    assert injury_report._retry_after_seconds(response) is None


def test_retry_after_rejects_non_numeric_value():
    response = FakeResponse(503, headers={"Retry-After": "garbage"})
    assert injury_report._retry_after_seconds(response) is None


def test_retry_after_rejects_http_date_form():
    # RFC 7231 permits Retry-After as an HTTP-date; deliberately not
    # parsed -- falls back to None (exponential backoff), same as any
    # other unparseable value. Not a bug, just documenting the boundary.
    response = FakeResponse(503, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"})
    assert injury_report._retry_after_seconds(response) is None


def test_retry_after_passes_through_well_formed_value():
    response = FakeResponse(503, headers={"Retry-After": "1"})
    assert injury_report._retry_after_seconds(response) == 1.0


def test_transient_retry_survives_real_sleep_with_zero_retry_after(monkeypatch):
    """Undoes this module's autouse sleep stub for one test and lets a
    real `time.sleep(0)` run end-to-end through the retry path. Retry-After
    is set to 0, which is both a well-formed value that must pass straight
    through _retry_after_seconds and safe to actually sleep for -- this is
    the regression guard for a bad (negative/huge/non-numeric) value
    reaching the real time.sleep() and raising the wrong exception type or
    hanging, a failure mode the autouse stub would otherwise mask.
    """
    monkeypatch.setattr(injury_report, "_sleep", time.sleep)
    session = ConstantStatusSession(503, headers={"Retry-After": "0"})
    with pytest.raises(injury_report.TransientFetchError) as exc_info:
        injury_report.fetch_report(date(2025, 1, 15), "05PM", session)
    assert exc_info.value.retry_after == 0.0
    assert len(session.calls) == injury_report._MAX_ATTEMPTS
