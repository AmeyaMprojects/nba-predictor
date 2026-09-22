from __future__ import annotations

import io
import math
import re
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

import pdfplumber
import requests
import tenacity

from predictor import db, raw_store

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

# Ceiling for any single wait between retry attempts, whether it comes from
# our own exponential backoff or from a server-supplied Retry-After header.
# A response header is untrusted input; without this cap, a huge value
# would drive an effectively unbounded sleep in an unattended backfill.
_MAX_WAIT_SECONDS = 30


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
    """Parse and sanitize a Retry-After response header.

    Only the numeric ("delay-seconds") form is handled. RFC 7231 also
    permits an HTTP-date form; that is deliberately NOT parsed here --
    `float(value)` raises `ValueError` on it, which is caught below and
    folds it into the same `None` fallback (exponential backoff takes
    over) as any other unparseable value, rather than being a separate
    concern to handle.

    A response header is untrusted input, so the result is sanitized
    before ever reaching a caller: non-numeric, non-finite (inf/nan), and
    negative values all return `None` -- a negative value must never reach
    the real `time.sleep()`, which raises `ValueError` for one, and that
    would be an uncaught exception of the wrong type escaping this module's
    documented contract (only `TransientFetchError` or a genuine network
    exception should ever propagate out of a fetch). A well-formed but
    huge value is clamped to `_MAX_WAIT_SECONDS`, the same ceiling the
    exponential backoff path already respects, so this header can never
    drive a longer, effectively unbounded hang in an unattended backfill.
    """
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("Retry-After")
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return min(seconds, _MAX_WAIT_SECONDS)


_BACKOFF = tenacity.wait_exponential(multiplier=1, max=_MAX_WAIT_SECONDS)


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


EASTERN = ZoneInfo("America/New_York")

COLS = ["GameDate", "GameTime", "Matchup", "Team", "PlayerName", "CurrentStatus", "Reason"]
GAME_DATE, GAME_TIME, MATCHUP, TEAM, PLAYER, STATUS, REASON = range(7)

_STAMP = re.compile(r"(\d{2}/\d{2}/\d{2})\s+(\d{1,2}:\d{2})\s*(AM|PM)")


@dataclass(frozen=True)
class InjuryRow:
    game_date: date | None
    game_time: str
    matchup: str
    team: str
    player: str
    status: str
    reason: str


@dataclass(frozen=True)
class ParsedReport:
    published_at: datetime
    rows: list[InjuryRow]


class InjuryReportParseError(Exception):
    """Raised when a report's rows could not be extracted at all.

    A genuinely empty-but-well-formed injury report is not a thing this
    source produces -- even the smallest real report sampled had 3 rows.
    Zero rows out of a parsed PDF means the column headers were never
    located (format drift, an unexpected layout, a corrupted file), not
    that nobody was injured. Silently returning an empty ParsedReport
    here would let ingest_report write 0 rows and return 0 as though that
    were legitimate, producing an invisible hole in a Task 9 backfill of
    roughly 2,500 reports -- exactly what "never silently lose data --
    surface loudly" forbids.
    """


def _column_bounds(page) -> list[float] | None:
    header: dict[str, float] = {}
    for word in page.extract_words():
        if word["text"] in COLS and word["text"] not in header:
            header[word["text"]] = word["x0"]
    return [header[c] for c in COLS] if len(header) == len(COLS) else None


def _column_index(x0: float, bounds: list[float]) -> int:
    index = 0
    for i, edge in enumerate(bounds):
        if x0 >= edge - 2:
            index = i
    return index


def _lines(page, bounds):
    groups: dict[int, list] = {}
    for word in page.extract_words():
        if word["text"] in COLS:
            continue
        groups.setdefault(round(word["top"] / 3), []).append(word)

    out = []
    for key in sorted(groups):
        words = groups[key]
        cells = [""] * len(COLS)
        for word in sorted(words, key=lambda w: w["x0"]):
            i = _column_index(word["x0"], bounds)
            cells[i] = (cells[i] + " " + word["text"]).strip()
        if "InjuryReport:" in "".join(cells).replace(" ", ""):
            continue
        out.append((min(w["top"] for w in words), cells))
    return out


def _parse_page(page, bounds, carry):
    lines = _lines(page, bounds)
    anchors = [(t, c) for t, c in lines if c[PLAYER] and c[STATUS]]
    fragments = [
        (t, c[REASON]) for t, c in lines if not (c[PLAYER] and c[STATUS]) and c[REASON]
    ]

    rows = [
        {"top": t, "cells": c, "pieces": [(t, c[REASON])] if c[REASON] else []}
        for t, c in anchors
    ]

    # A wrapped Reason can sit above or below its player line, so each
    # fragment attaches to the vertically nearest anchor.
    for top, text in fragments:
        if rows:
            nearest = min(rows, key=lambda r: abs(r["top"] - top))
            nearest["pieces"].append((top, text))

    parsed = []
    for row in rows:
        cells = row["cells"]
        for i in (GAME_DATE, GAME_TIME, MATCHUP, TEAM):
            if cells[i]:
                carry[i] = cells[i]
            else:
                cells[i] = carry.get(i, "")
        reason = "".join(text for _, text in sorted(row["pieces"]))
        parsed.append(
            InjuryRow(
                game_date=_to_date(cells[GAME_DATE]),
                game_time=cells[GAME_TIME],
                matchup=cells[MATCHUP],
                team=cells[TEAM],
                player=cells[PLAYER],
                status=cells[STATUS],
                reason=reason,
            )
        )
    return parsed, carry


def _to_date(text: str) -> date | None:
    try:
        return datetime.strptime(text, "%m/%d/%Y").date()
    except ValueError:
        return None


def _published_at(first_line: str) -> datetime:
    match = _STAMP.search(first_line)
    if not match:
        raise ValueError(f"no publication timestamp in header: {first_line!r}")
    day, clock, meridiem = match.groups()
    naive = datetime.strptime(f"{day} {clock} {meridiem}", "%m/%d/%y %I:%M %p")
    return naive.replace(tzinfo=EASTERN).astimezone(UTC)


def parse_report(pdf_bytes: bytes) -> ParsedReport:
    rows: list[InjuryRow] = []
    carry: dict[int, str] = {}
    bounds = None

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        published_at = _published_at(pdf.pages[0].extract_text().split("\n")[0])
        for page in pdf.pages:
            # Only page 1 carries the header; reuse its bounds thereafter.
            bounds = _column_bounds(page) or bounds
            if bounds is None:
                continue
            page_rows, carry = _parse_page(page, bounds, carry)
            rows.extend(page_rows)

    if bounds is None or not rows:
        raise InjuryReportParseError(
            "parsed zero rows -- column headers were never located on any "
            "page (format drift, an unexpected layout, or a corrupted "
            "PDF); a well-formed report is never actually empty"
        )

    return ParsedReport(published_at=published_at, rows=rows)


def ingest_report(con, pdf_bytes: bytes) -> int:
    report = parse_report(pdf_bytes)
    observed_at = db.require_utc(report.published_at, "observed_at")
    # report_date is a calendar date meant to match the report's real
    # publication day (and the Eastern-based `day` metadata archive_report
    # already records for the same file), not whatever day UTC happens to
    # land on. Hour labels run through 11PM, so any report published from
    # roughly 7PM ET onward crosses midnight UTC -- observed_at.date()
    # would silently store the NEXT calendar day for every one of those
    # reports. Converting back to Eastern before taking .date() is what
    # keeps this aligned with the filename/header day. observed_at itself
    # stays UTC; only this calendar-date column is computed in Eastern.
    report_date = observed_at.astimezone(EASTERN).date()
    # Resolved through db.POINT_IN_TIME_TABLES rather than spelled as a
    # literal here -- the physical "_raw" table names are only allowed to
    # appear as string literals in db.py/asof.py (see
    # test_no_physical_table_name_appears_outside_db_and_asof); this
    # ingestion module must not name the physical table directly either.
    table = db.POINT_IN_TIME_TABLES["injury_status"]
    insert_sql = (
        f"INSERT OR REPLACE INTO {table} (report_date, game_date, matchup,"
        " team, player, status, reason, reconstructed, observed_at)"
        " VALUES (?,?,?,?,?,?,?,FALSE,?)"
    )
    for row in report.rows:
        # game_date is NOT NULL and part of the primary key (see schema
        # comment in predictor.db); if a row's game date failed to parse,
        # substitute the report's own publication date rather than
        # dropping the row or inserting NULL -- losing an injury row is
        # worse than an imperfect date.
        game_date = row.game_date if row.game_date is not None else report_date
        con.execute(
            insert_sql,
            [
                report_date,
                game_date,
                row.matchup,
                row.team,
                row.player,
                row.status,
                row.reason,
                observed_at,
            ],
        )
    return len(report.rows)
