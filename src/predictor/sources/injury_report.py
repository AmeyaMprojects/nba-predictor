from __future__ import annotations

import time
from datetime import UTC, date, datetime

import requests
import tenacity

from predictor import raw_store

BASE_URL = "https://ak-static.cms.nba.com/referee/injury"

HOUR_LABELS: tuple[str, ...] = tuple(
    f"{h:02d}{suffix}" for suffix in ("AM", "PM") for h in list(range(1, 13))
)

# Archive verified to begin here; earlier dates return 403. Enforced in
# archive_report(), which short-circuits before making a request.
ARCHIVE_START = date(2019, 12, 1)

# Statuses that unambiguously mean "this slot was never published" -- safe
# to report as a quiet `None`. Everything else (429, 5xx, any other
# unexpected status) is ambiguous: it means "could not check" rather than
# "confirmed absent", and must never be conflated with a real absence, so
# it is raised as TransientFetchError instead. A ~2,500-slot backfill
# (Task 9) hitting a rate limit or a CDN hiccup mid-run must never have
# that silently turn into a permanent hole in injury history.
_NOT_PUBLISHED_STATUSES = (403, 404)

_MAX_ATTEMPTS = 4


class TransientFetchError(Exception):
    """A fetch attempt failed in a way that does not mean "not published".

    Raised for 429, 5xx, or any other unexpected HTTP status, after
    retries with exponential backoff are exhausted. Distinct from the
    quiet `None` that `fetch_report` returns for a genuine 403/404 "not
    published" response, and distinct from a network-level exception
    (connection error, timeout), which propagates immediately without
    retry, exactly as before.
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


def report_url(day: date, hour_label: str) -> str:
    return f"{BASE_URL}/Injury-Report_{day.isoformat()}_{hour_label}.pdf"


def raw_key(day: date, hour_label: str) -> str:
    return f"Injury-Report_{day.isoformat()}_{hour_label}.pdf"


def _sleep(seconds: float) -> None:
    """Real sleep, indirected so tests can stub it without real delay.

    Referenced by name (not bound) from the retry decorator below, so
    `monkeypatch.setattr(injury_report, "_sleep", ...)` takes effect even
    though the decorator itself is evaluated once at import time.
    """
    time.sleep(seconds)


def _retry_after_seconds(response) -> float | None:
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("Retry-After")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


_BACKOFF = tenacity.wait_exponential(multiplier=1, max=30)


def _wait_seconds(retry_state: tenacity.RetryCallState) -> float:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exc, TransientFetchError) and exc.retry_after is not None:
        return exc.retry_after
    return _BACKOFF(retry_state)


def _classify_response(response, day: date, hour_label: str) -> bytes | None:
    """Interpret one HTTP response for a single fetch attempt.

    Returns the response body on 200, `None` on a genuine 403/404 "not
    published", or raises `TransientFetchError` for anything else so the
    retry wrapper around this function -- and ultimately the caller -- can
    tell "confirmed absent" apart from "could not check".
    """
    key = raw_key(day, hour_label)
    status = response.status_code
    if status == 200:
        return response.content
    if status in _NOT_PUBLISHED_STATUSES:
        print(f"injury: not published -- {key} -> HTTP {status}")
        return None
    retry_after = _retry_after_seconds(response)
    suffix = f" (retry-after {retry_after}s)" if retry_after is not None else ""
    print(f"injury: transient failure -- {key} -> HTTP {status}{suffix}")
    raise TransientFetchError(
        f"transient HTTP {status} fetching {key}",
        status_code=status,
        retry_after=retry_after,
    )


@tenacity.retry(
    retry=tenacity.retry_if_exception_type(TransientFetchError),
    wait=_wait_seconds,
    stop=tenacity.stop_after_attempt(_MAX_ATTEMPTS),
    sleep=lambda seconds: _sleep(seconds),
    reraise=True,
)
def _fetch_attempt(session, day: date, hour_label: str) -> bytes | None:
    response = session.get(report_url(day, hour_label), timeout=30)
    return _classify_response(response, day, hour_label)


def fetch_report(day: date, hour_label: str, session=None) -> bytes | None:
    """Fetch one report slot.

    Returns the raw bytes on success, or `None` if the slot is confirmed
    to never have been published (403/404). A network-level exception
    (connection error, timeout, ...) propagates immediately, uncaught.
    A 429/5xx/unexpected status is retried with exponential backoff
    (honoring a `Retry-After` header when the response provides one) and,
    once retries are exhausted, raises `TransientFetchError` -- this must
    NOT be treated as "not published" by callers.
    """
    session = session or requests.Session()
    return _fetch_attempt(session, day, hour_label)


def archive_report(
    day: date,
    hour_label: str,
    now: datetime | None = None,
    session=None,
) -> bool:
    if day < ARCHIVE_START:
        print(
            f"injury: skipping {raw_key(day, hour_label)} -- before archive "
            f"start {ARCHIVE_START.isoformat()}"
        )
        return False

    key = raw_key(day, hour_label)
    if raw_store.exists("injury", key):
        return False

    # TransientFetchError intentionally propagates uncaught here: it means
    # "could not check", which the caller (a Task 9 backfill loop) must be
    # able to tell apart from a real, confirmed absence and count/retry
    # separately -- swallowing it here would silently reproduce the exact
    # failure mode this exception exists to prevent.
    content = fetch_report(day, hour_label, session)
    if content is None:
        return False

    if not content.startswith(b"%PDF"):
        # A fixed key per slot (not content-addressed like the news feeds)
        # means archiving a garbage body would either block the real PDF
        # forever via the exists() short-circuit above, or collide with it
        # later as a RawStoreConflict. Discarding it is correct; just make
        # sure it doesn't happen silently.
        print(f"injury: non-PDF body discarded -- {key} ({len(content)} bytes)")
        return False

    raw_store.store(
        "injury",
        key,
        content,
        now or datetime.now(UTC),
        meta={"day": day.isoformat(), "hour_label": hour_label},
    )
    return True
