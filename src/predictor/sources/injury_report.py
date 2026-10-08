from __future__ import annotations

import io
import math
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import Enum
from zoneinfo import ZoneInfo

import pdfplumber
import requests
import tenacity

from predictor import db, raw_store
from predictor.teams import team_abbr

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

    Raised for 429, 5xx, or any other unexpected HTTP status, OR for a
    connection-level failure (`requests.exceptions.ConnectionError`/
    `Timeout`/`ChunkedEncodingError` -- a dropped/reset connection, a
    timeout, or the response cutting off mid-download), after retries
    with exponential backoff are exhausted. Distinct from the quiet
    `None` that `fetch_report` returns for a genuine 403/404 "not
    published" response, which is checked without any retry.
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


# Network-level failures that mean "the request never got a response at
# all" -- a dropped/reset connection, a connect/read timeout, or the
# connection closing mid-download. Distinct from an HTTP error status
# (handled by `_classify_response`/`TransientFetchError` below): these
# never reach `session.get`'s return value, they raise instead. For a
# single call letting them propagate is correct and loud; for an
# unattended ~2,500-request backfill it is fatal -- a single reset
# killed a live run (Task 8 review Finding 2) roughly 40% through.
# Retried the same way as a transient HTTP status (same backoff, same
# attempt cap) by converting them to TransientFetchError below, which
# both makes them retryable (the retry condition here only matches
# TransientFetchError) and keeps the exhausted-retries exception type
# consistent with fetch_report's documented two-outcome contract.
_RETRYABLE_NETWORK_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


@tenacity.retry(
    retry=tenacity.retry_if_exception_type(TransientFetchError),
    wait=_wait_seconds,
    stop=tenacity.stop_after_attempt(_MAX_ATTEMPTS),
    sleep=lambda seconds: _sleep(seconds),
    reraise=True,
)
def _fetch_attempt(session, day: date, hour_label: str) -> bytes | None:
    key = raw_key(day, hour_label)
    try:
        response = session.get(report_url(day, hour_label), timeout=30)
    except _RETRYABLE_NETWORK_EXCEPTIONS as exc:
        # No response, no status code, no Retry-After header to honor --
        # this falls back to the same plain exponential backoff a
        # Retry-After-less transient HTTP status already uses (see
        # `_wait_seconds`). 403/404 never reach this branch (they're a
        # completed HTTP response, not an exception), so they still skip
        # the retry loop entirely -- burning four attempts on every
        # confirmed-absent offseason slot would slow the backfill for no
        # reason.
        print(f"injury: network error -- {key} -> {type(exc).__name__}: {exc}")
        raise TransientFetchError(
            f"network error fetching {key}: {type(exc).__name__}: {exc}"
        ) from exc
    return _classify_response(response, day, hour_label)


def fetch_report(day: date, hour_label: str, session=None) -> bytes | None:
    """Fetch one report slot.

    Returns the raw bytes on success, or `None` if the slot is confirmed
    to never have been published (403/404, checked without any retry).
    Both a 429/5xx/unexpected status AND a connection-level failure
    (`requests.exceptions.ConnectionError`/`Timeout`/`ChunkedEncodingError`
    -- a dropped/reset connection, a timeout, or the response cutting off
    mid-download) are retried with exponential backoff (honoring a
    `Retry-After` header when an HTTP response provides one) and, once
    retries are exhausted, both raise `TransientFetchError` -- this must
    NOT be treated as "not published" by callers. Any OTHER exception
    (e.g. a `requests` exception not in that retryable set) propagates
    immediately, uncaught.
    """
    session = session or requests.Session()
    return _fetch_attempt(session, day, hour_label)


class ArchiveOutcome(Enum):
    """Fine-grained result of one archive attempt for a single slot.

    `archive_report` (the pre-existing, tested public entry point) collapses
    this down to a bool -- True only for ARCHIVED, False for everything
    else -- so every existing caller/test of `archive_report` is unaffected.
    `backfill_range` calls the internal `_archive` helper directly instead,
    because it needs to tell BAD_CONTENT apart from NOT_PUBLISHED (Task 9
    review Finding 1): a 200 response with a non-PDF body is NOT the same
    thing as a confirmed 403/404 "never published" response, and conflating
    them would let a CDN block/challenge page masquerade as a clean sweep
    of "nothing published this week".
    """

    ARCHIVED = "archived"
    ALREADY_ARCHIVED = "already_archived"
    BEFORE_ARCHIVE_START = "before_archive_start"
    NOT_PUBLISHED = "not_published"
    BAD_CONTENT = "bad_content"

    @property
    def request_made(self) -> bool:
        """Whether this outcome involved an actual network request.

        ALREADY_ARCHIVED and BEFORE_ARCHIVE_START both short-circuit before
        any call to `fetch_report`; a caller sleeping a fixed "politeness"
        delay between slots that touched the network must skip that delay
        for these two outcomes (Task 9 review Finding 3).
        """
        return self not in (
            ArchiveOutcome.ALREADY_ARCHIVED,
            ArchiveOutcome.BEFORE_ARCHIVE_START,
        )


def _archive(
    day: date,
    hour_label: str,
    now: datetime | None = None,
    session=None,
) -> ArchiveOutcome:
    if day < ARCHIVE_START:
        print(
            f"injury: skipping {raw_key(day, hour_label)} -- before archive "
            f"start {ARCHIVE_START.isoformat()}"
        )
        return ArchiveOutcome.BEFORE_ARCHIVE_START

    key = raw_key(day, hour_label)
    if raw_store.exists("injury", key):
        return ArchiveOutcome.ALREADY_ARCHIVED

    # TransientFetchError intentionally propagates uncaught here: it means
    # "could not check", which the caller (a Task 9 backfill loop) must be
    # able to tell apart from a real, confirmed absence and count/retry
    # separately -- swallowing it here would silently reproduce the exact
    # failure mode this exception exists to prevent.
    content = fetch_report(day, hour_label, session)
    if content is None:
        return ArchiveOutcome.NOT_PUBLISHED

    if not content.startswith(b"%PDF"):
        # A fixed key per slot (not content-addressed like the news feeds)
        # means archiving a garbage body would either block the real PDF
        # forever via the exists() short-circuit above, or collide with it
        # later as a RawStoreConflict. Discarding it is correct; just make
        # sure it doesn't happen silently. Distinct from NOT_PUBLISHED: this
        # was a 200, i.e. the server answered, it just didn't answer with a
        # PDF -- a confirmed-absent 403/404 never reaches this branch.
        print(f"injury: non-PDF body discarded -- {key} ({len(content)} bytes)")
        return ArchiveOutcome.BAD_CONTENT

    raw_store.store(
        "injury",
        key,
        content,
        now or datetime.now(UTC),
        meta={"day": day.isoformat(), "hour_label": hour_label},
    )
    return ArchiveOutcome.ARCHIVED


def archive_report(
    day: date,
    hour_label: str,
    now: datetime | None = None,
    session=None,
) -> bool:
    """Fetch and archive one slot; True iff a new PDF was archived.

    False covers every other outcome (already archived, before the archive
    window, confirmed absent, or a 200 with a non-PDF body) -- this is the
    pre-existing public contract, unchanged. Callers that need to tell those
    "False" cases apart (Task 9's `backfill_range`) use `_archive` directly.
    """
    return _archive(day, hour_label, now, session) is ArchiveOutcome.ARCHIVED


EASTERN = ZoneInfo("America/New_York")

COLS = ["GameDate", "GameTime", "Matchup", "Team", "PlayerName", "CurrentStatus", "Reason"]
GAME_DATE, GAME_TIME, MATCHUP, TEAM, PLAYER, STATUS, REASON = range(7)

_STAMP = re.compile(r"(\d{2}/\d{2}/\d{2})\s+(\d{1,2}:\d{2})\s*(AM|PM)")

# Matches a "Page X of Y" footer line once its cell text has been
# whitespace-stripped and concatenated -- e.g. "Page1of10". See the note
# where this is used in `_lines()` for why it must be filtered out before
# forward-fill ever sees it.
_PAGE_FOOTER = re.compile(r"^Page\d+of\d+$")


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


class InjuryReportEmptyError(InjuryReportParseError):
    """Raised when headers WERE located but the report has no real filings.

    I3: distinct from the base `InjuryReportParseError` (which means
    "column headers were never located at all" -- format drift, an
    unexpected layout, a corrupted file). This subclass means the OPPOSITE
    of that: parsing worked fine, but every team on the slate either filed
    nothing or explicitly said "NOT YET SUBMITTED" -- Summer League, the
    All-Star break, and playoff off-days all genuinely produce reports
    like this (all 107 real archived parse failures sampled were exactly
    this case, none were actual format drift). `reingest_archived` buckets
    this separately (`empty_no_filings`, not `still_failed`) so the
    documented recovery command (`predictor reingest-injuries`) can
    finally report success: previously EVERY run exited 1 for these 107
    genuinely-fine reports, so an operator could never tell "fixed and
    fully recovered" apart from "still broken".
    """


# The old ("spaced") layout keeps each header label as separate words
# instead of concatenating them -- e.g. "Game" "Date" rather than
# "GameDate" -- and "Game" appears twice (Game Date, then Game Time), so
# this can only be matched as an ORDERED run, not a set of tokens.
_SPACED_HEADER_RUN: tuple[str, ...] = (
    "Game",
    "Date",
    "Game",
    "Time",
    "Matchup",
    "Team",
    "Player",
    "Name",
    "Current",
    "Status",
    "Reason",
)
# Index into _SPACED_HEADER_RUN of the FIRST word of each COLS entry, in
# COLS order -- e.g. PlayerName's x0 comes from "Player" (index 6), the
# GameTime's from the SECOND "Game" (index 2), not the first.
_SPACED_HEADER_STARTS: tuple[int, ...] = (0, 2, 4, 5, 6, 8, 10)


def _find_token_run(words, tokens: tuple[str, ...]) -> list | None:
    """First contiguous run of `words` whose text matches `tokens`, in order."""
    n = len(tokens)
    texts = [w["text"] for w in words]
    for i in range(len(texts) - n + 1):
        if tuple(texts[i : i + n]) == tokens:
            return words[i : i + n]
    return None


def _row_end(words, header_top: float, reason_x0: float) -> float | None:
    """x0 of the first header word to the right of Reason, if any.

    Both eras occasionally carry an extra "Previous Status" (and, once,
    "Previous Status Previous Reason") column after Reason -- see the
    2019-12-01..17 archived reports, where this column disappears
    partway through December 2019. Nothing downstream models those
    columns, so their words must be capped out of the Reason band rather
    than silently absorbed into it, which would corrupt Reason text with
    a trailing previous-status value or a bare dash.
    """
    extras = [
        w["x0"]
        for w in words
        if abs(w["top"] - header_top) < 1.0 and w["x0"] > reason_x0 + 1
    ]
    return min(extras) if extras else None


def _detect_header(words) -> tuple[list[float], float, float | None] | None:
    """Locate the column header line among `words`, in either layout.

    Returns `(edges, header_top, row_end)`, or None if no header line is
    present at all. `edges` is the x0 of each COLS entry's first word, in
    COLS order. `header_top` is the y-coordinate ("top") shared by every
    header word -- used elsewhere to mask the header line back out of
    data rows, which matters because, unlike the new/concatenated layout
    (header on page 1 only), the old/spaced layout repeats the header on
    every page.
    """
    # New layout: header labels already concatenated into single words
    # exactly matching COLS, findable anywhere on the page.
    header: dict[str, float] = {}
    header_top: float | None = None
    for word in words:
        if word["text"] in COLS and word["text"] not in header:
            header[word["text"]] = word["x0"]
            if header_top is None:
                header_top = word["top"]
    if len(header) == len(COLS):
        edges = [header[c] for c in COLS]
        return edges, header_top, _row_end(words, header_top, edges[-1])

    # Old layout: header labels are split across multiple words.
    run = _find_token_run(words, _SPACED_HEADER_RUN)
    if run is None:
        return None
    edges = [run[i]["x0"] for i in _SPACED_HEADER_STARTS]
    header_top = run[0]["top"]
    return edges, header_top, _row_end(words, header_top, edges[-1])


def _column_bounds(page) -> tuple[list[float], float | None] | None:
    detected = _detect_header(page.extract_words())
    if detected is None:
        return None
    edges, _header_top, row_end = detected
    return edges, row_end


def _column_index(x0: float, edges: list[float]) -> int:
    index = 0
    for i, edge in enumerate(edges):
        if x0 >= edge - 2:
            index = i
    return index


def _lines(page, bounds):
    edges, row_end = bounds
    words = page.extract_words()

    # The old/spaced layout repeats the header line on every page (the
    # new/concatenated layout does not); re-detect it on THIS page (fresh,
    # not the carried-over `bounds`) so its words are masked out of the
    # data rows rather than parsed as a bogus row -- otherwise "Player
    # Name" / "Current Status" would land in the PLAYER/STATUS cells and
    # get treated as a real anchor row, corrupting the forward-filled
    # game date/time/matchup/team for every row after it on the page.
    detected = _detect_header(words)
    header_top = detected[1] if detected is not None else None

    groups: dict[int, list] = {}
    for word in words:
        # NOTE: deliberately NOT filtering by `word["text"] in COLS` here.
        # That was the previous approach, and it is unsafe: in the old/
        # spaced layout, reason text can literally contain the word
        # "Team" (as in "Not With Team"), which collides with the COLS
        # entry "Team" and would silently drop that word from the row.
        # The new/concatenated layout never collides this way -- there
        # "NotWithTeam" is a single word -- which is why this bug stayed
        # hidden. header_top-based masking below is strictly more precise
        # (it only removes words actually on the header line) and
        # supersedes this check for both layouts.
        if header_top is not None and abs(word["top"] - header_top) < 1.0:
            continue
        if row_end is not None and word["x0"] >= row_end - 2:
            continue
        groups.setdefault(round(word["top"] / 3), []).append(word)

    out = []
    for key in sorted(groups):
        line_words = groups[key]
        cells = [""] * len(COLS)
        for word in sorted(line_words, key=lambda w: w["x0"]):
            i = _column_index(word["x0"], edges)
            cells[i] = (cells[i] + " " + word["text"]).strip()
        joined = "".join(cells).replace(" ", "")
        if "InjuryReport:" in joined:
            continue
        if _PAGE_FOOTER.match(joined):
            # A "Page X of Y" footer's text happens to fall inside the
            # TEAM column's x-range on the page, so `_column_index` files
            # it there. Harmless historically (this line was never an
            # anchor or a kept fragment, so it was invisible to
            # forward-fill), but the forward-fill loop in `_parse_page`
            # now walks EVERY line (see that function's docstring), which
            # would otherwise carry the literal text "Page1of10" into
            # `carry[TEAM]` and corrupt every row until a real TEAM value
            # appeared again. Must be dropped here, before forward-fill
            # ever sees it -- same treatment as the "InjuryReport:" title
            # line above.
            continue
        out.append((min(w["top"] for w in line_words), cells))
    return out


def _is_not_yet_submitted(reason: str) -> bool:
    """True iff `reason` is exactly the "team hasn't filed yet" placeholder.

    A team with no injury report filed for a slate gets its own row --
    team name, no player, no status, reason "NOT YET SUBMITTED" (or,
    concatenated-format, "NOTYETSUBMITTED") -- meaning the OPPOSITE of an
    injury: nobody has said anything yet. That row has no player/status,
    so it looks exactly like a wrapped-Reason continuation fragment to
    the nearest-anchor attachment logic below, and without this filter
    it gets glued onto some nearby player's real reason (observed
    concatenating repeatedly when several such teams are stacked near
    the same anchor, e.g. "...Bone bruiseNOT YET SUBMITTEDNOT YET
    SUBMITTED...").

    Matched by exact content (normalized: whitespace stripped,
    case-insensitive), not by row shape (e.g. "has a Team but no
    Player/Status"): a genuine wrapped-reason fragment can legitimately
    share a physical line with an unrelated team-section-header label
    that happens to render at the same y-coordinate (observed in the
    2019-12-01..17 archived reports, which have a narrower Reason
    column and more line-wrapping) -- excluding by shape would silently
    drop that real reason text instead of just the placeholder.
    Checked against the archive: no other wording for this placeholder
    was found; every fragment matching "not (player and status)" that
    isn't this exact string turned out to be genuine reason text.
    """
    return reason.replace(" ", "").upper() == "NOTYETSUBMITTED"


def _parse_page(page, bounds, carry):
    lines = _lines(page, bounds)

    # GAME_DATE/GAME_TIME/MATCHUP/TEAM must be forward-filled from EVERY
    # line on the page, in top-to-bottom order -- not just from lines that
    # go on to become an InjuryRow (an "anchor": has both PLAYER and
    # STATUS). A team that filed no report at all for a slate gets a
    # placeholder line (team + matchup, reason "NOT YET SUBMITTED", no
    # player/status) that is never emitted as a row -- but it can be the
    # ONLY line in the whole page announcing that a new game/team section
    # has begun. Forward-filling only from anchors (the pre-fix code)
    # silently threw that section-boundary information away: the very
    # next anchor line (an unrelated team/matchup) would then blank-fill
    # MATCHUP or TEAM from the PREVIOUS section instead, corrupting the
    # row with an impossible team/matchup pairing.
    #
    # Confirmed against the real archive: Injury-Report_2025-01-16_05PM.pdf
    # has a "CHA@CHI / CharlotteHornets ... NOT YET SUBMITTED" placeholder
    # line immediately followed by a "ChicagoBulls" anchor line whose own
    # MATCHUP cell is blank (correctly relying on forward-fill, since only
    # the first line of a matchup section prints it). Before this fix, the
    # placeholder line was invisible to forward-fill, so MATCHUP stayed on
    # the stale PREVIOUS matchup ("MIN@NYK") instead of updating to
    # "CHA@CHI" -- producing rows like team=ChicagoBulls,
    # matchup=MIN@NYK, an internally impossible pairing that
    # `_validate_team_matches_matchup` below now catches. This single
    # change (processing all lines, not just anchors, for forward-fill)
    # fixed 100% of the sampled real corruption -- see module-level notes.
    anchors: list[tuple[float, list[str]]] = []
    fragments: list[tuple[float, str]] = []
    for top, cells in lines:
        for i in (GAME_DATE, GAME_TIME, MATCHUP, TEAM):
            if cells[i]:
                carry[i] = cells[i]
            else:
                cells[i] = carry.get(i, "")
        if cells[PLAYER] and cells[STATUS]:
            anchors.append((top, cells))
        elif cells[REASON] and not _is_not_yet_submitted(cells[REASON]):
            fragments.append((top, cells[REASON]))

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
            # New/concatenated-layout reports carry the header on page 1
            # only, so later pages reuse its bounds. Old/spaced-layout
            # reports repeat the header on every page (verified across the
            # archive); _column_bounds/_detect_header re-find it there too,
            # which is harmless -- it just recomputes the same numeric
            # bounds -- and _lines() masks each page's own header line back
            # out of its data rows regardless of which case applies.
            bounds = _column_bounds(page) or bounds
            if bounds is None:
                continue
            page_rows, carry = _parse_page(page, bounds, carry)
            rows.extend(page_rows)

    if bounds is None:
        # M1: this is the ONLY branch where "column headers were never
        # located" is actually true. All 107 real archived failures take
        # the OTHER branch below (headers WERE found, on some page, but no
        # anchor row was ever extracted -- e.g. a slate consisting
        # entirely of "NOT YET SUBMITTED" placeholders, such as Summer
        # League, the All-Star break, or a playoff off-day) -- the old
        # single shared message claimed headers were never found in 100%
        # of those real cases, which is simply false and actively
        # misleads whoever reads the log.
        raise InjuryReportParseError(
            "parsed zero rows -- column headers were never located on any "
            "page (format drift, an unexpected layout, or a corrupted "
            "PDF); a well-formed report is never actually empty"
        )
    if not rows:
        raise InjuryReportEmptyError(
            "parsed zero rows -- column headers WERE located, but no row "
            "with both a player and a status was ever extracted (every "
            "line was either a header, a placeholder like 'NOT YET "
            "SUBMITTED', or unparseable) -- this is a genuinely empty "
            "slate (Summer League, All-Star break, a playoff off-day), "
            "not a parser defect"
        )

    _validate_team_matches_matchup(rows)

    return ParsedReport(published_at=published_at, rows=rows)


class InjuryTeamMismatchError(Exception):
    """Raised when a row's TEAM does not belong to either side of its own MATCHUP.

    This is internally impossible for a real report -- a player's team is
    always one of the two teams playing in their own game -- so it means
    the forward-fill carried a value across a game-section boundary it
    should not have (the exact defect this validation exists to catch;
    see the note above `_parse_page`'s forward-fill loop). Raised loudly
    rather than silently dropped or corrected, per the module's "never
    silently lose data" rule: `backfill_range`/`reingest_archived` bucket
    this together with any other parse-time exception as `parse_failed`,
    so it surfaces in the operator-facing counts rather than silently
    writing a wrong game_date (game_date is a PRIMARY KEY column) for the
    affected rows.
    """


# `team_abbr` lives in predictor.teams (shared with the odds sources) and is
# re-exported here: `injury_report.team_abbr` is part of this module's API.


def _validate_team_matches_matchup(rows: list[InjuryRow]) -> None:
    """Assert every row's TEAM is one of the two teams in its own MATCHUP.

    A Celtics player filed under a Hornets-at-Knicks game is internally
    impossible -- this single assertion catches a corrupted forward-fill
    (see `_parse_page`) the moment it happens, rather than letting it
    silently reach the database, where `game_date` is a PRIMARY KEY
    column and a wrong carry also misattributes the injury to the wrong
    game.
    """
    for row in rows:
        if "@" not in row.matchup:
            # Malformed/unparsed matchup text -- nothing to validate
            # against; not this function's job to also invent a matchup.
            continue
        away, home = (part.strip() for part in row.matchup.split("@", 1))
        abbr = team_abbr(row.team)
        if abbr is None:
            raise InjuryTeamMismatchError(
                f"row team {row.team!r} (player {row.player!r}, matchup "
                f"{row.matchup!r}) is not a recognized NBA team name -- "
                "cannot validate it against its own matchup"
            )
        if abbr not in (away, home):
            raise InjuryTeamMismatchError(
                f"row team {row.team!r} (abbr {abbr}, player "
                f"{row.player!r}) does not appear in its own matchup "
                f"{row.matchup!r} -- forward-fill likely carried a value "
                "across a game-section boundary"
            )


def player_key(player: str) -> str:
    """Canonical, whitespace-stripped identity key for a raw PLAYER cell.

    C2: the two PDF layouts render the same player differently -- the old
    (spaced) layout keeps a space after the comma ("Curry, Stephen"), the
    new (concatenated) layout does not ("Curry,Stephen") -- so a naive
    `player` column silently splits every player's history in two across
    the 2023 layout boundary (measured: 1,591 raw -> 1,125 normalized
    identities, 466 duplicates). Stripping ALL whitespace makes both
    spellings collapse onto the same key without needing to know which
    layout produced them.
    """
    return "".join(player.split())


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
    # C2: `team`/`player` store the CANONICAL key form -- a 3-letter team
    # abbreviation (also the join key against the games table's
    # home_team/away_team columns, which nba_stats already writes as
    # TEAM_ABBREVIATION; there was previously NO join key at all between
    # the two point-in-time tables) and a whitespace-stripped player name.
    # `team_display`/`player_display` keep the original, human-readable
    # text exactly as parsed, so a
    # person browsing the table still sees "Golden State Warriors" /
    # "Curry, Stephen" rather than only "GSW" / "Curry,Stephen". Both
    # columns are added by the idempotent `_add_injury_normalization_columns`
    # migration in predictor.db (see that function's docstring for why
    # columns, not an in-place rewrite, was chosen).
    insert_sql = (
        f"INSERT OR REPLACE INTO {table} (report_date, game_date, matchup,"
        " team, team_display, player, player_display, status, reason,"
        " game_time, reconstructed, observed_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,FALSE,?)"
    )
    for row in report.rows:
        # game_date is NOT NULL and part of the primary key (see schema
        # comment in predictor.db); if a row's game date failed to parse,
        # substitute the report's own publication date rather than
        # dropping the row or inserting NULL -- losing an injury row is
        # worse than an imperfect date.
        game_date = row.game_date if row.game_date is not None else report_date
        # team_abbr() returns None only for a team name that isn't a
        # recognized NBA team at all -- `_validate_team_matches_matchup`
        # (called from parse_report) already raises loudly for that case
        # UNLESS the row's matchup itself was unparseable ("@" missing),
        # which skips validation entirely. This fallback exists only for
        # that narrow, already-tolerated edge case: keep the raw text
        # rather than drop the row, same rule as the game_date fallback
        # above.
        team_key = team_abbr(row.team) or row.team
        con.execute(
            insert_sql,
            [
                report_date,
                game_date,
                row.matchup,
                team_key,
                row.team,
                player_key(row.player),
                row.player,
                row.status,
                row.reason,
                row.game_time,
                observed_at,
            ],
        )
    return len(report.rows)


def backfill_range(
    con,
    start: date,
    end: date,
    hours: Sequence[str] = ("05PM",),
    delay: float = 0.4,
    session=None,
) -> dict[str, int]:
    """Sweep archived injury reports for every day/hour slot in [start, end].

    Resumable: a slot already present in raw_store is counted as `skipped`
    and never re-fetched. Rate-limited: `delay` seconds are slept between
    slots that actually touched the network. A single shared `session` is
    used for the whole sweep so a multi-thousand-slot run does not open a
    fresh TCP/TLS connection per request.

    Four failure/exception modes are counted SEPARATELY from `missing` (a
    confirmed 403/404 "never published" slot), and none of them abort the
    run -- a single bad slot must not cost the operator a multi-thousand-
    request restart:

    - `transient`: `fetch_report` raised `TransientFetchError` (429/5xx/
      unexpected status, after its own internal retries were exhausted).
      This means "could not check", not "absent" -- conflating the two was
      the exact defect fixed in Task 7. The slot is NOT archived, so a
      re-run of this same range will retry it automatically (raw_store
      still reports it as not existing).
    - `bad_content`: the response was a 200, but the body was not a PDF
      (e.g. a CDN block/challenge page). This is deliberately NOT folded
      into `missing`: a confirmed 403/404 and a 200-with-garbage-body look
      identical to a naive caller, but they mean very different things --
      one is a confirmed "never published", the other is "the server
      answered, just not with a report". A CDN serving a 200 block page
      across a stretch of the backfill must never masquerade as "the NBA
      published nothing those weeks". The slot is NOT archived (see
      `_archive`), so a re-run of this same range will retry it.
    - `parse_failed`: the PDF was fetched (and archived to disk) but could
      not be turned into rows, either via the documented
      `InjuryReportParseError` (zero rows extracted) or via an unforeseen
      exception from parsing -- notably, a zero-page PDF currently raises
      a plain `IndexError` from pdfplumber rather than
      `InjuryReportParseError` (a pre-existing gap, not fixed here). Both
      are bucketed together because the practical meaning is identical:
      bytes were fetched, nothing was ingested. IMPORTANT ASYMMETRY: unlike
      `transient` and `bad_content`, the raw PDF IS already archived at
      this point, so a re-run of this same range will SKIP this slot
      (raw_store.exists() is now true) rather than retry ingestion.
      Recovering a `parse_failed` slot after a parser fix means re-running
      `reingest_archived`, which ingests already-archived PDFs with no
      network fetch at all.

    A connection-level failure (dropped/reset connection, timeout,
    truncated response) is now retried inside `fetch_report` itself (Task
    8 review Finding 2 -- a single reset previously killed an unattended
    ~40%-through run) and, once its own retries are exhausted, surfaces
    here as `TransientFetchError`, same as a 429/5xx -- so it is counted
    as `transient`, not treated as an abort, same as any other transient
    failure. Any OTHER raw exception (not in that retryable set) is
    deliberately NOT caught here and still aborts the run, same as
    before; resumability (via `skipped`) means a re-run only repeats work
    from that point forward, not from the start.
    """
    session = session or requests.Session()
    stats = {
        "fetched": 0,
        "skipped": 0,
        "missing": 0,
        "ingested": 0,
        "transient": 0,
        "parse_failed": 0,
        "bad_content": 0,
    }

    day = start
    while day <= end:
        for hour in hours:
            key = raw_key(day, hour)
            if raw_store.exists("injury", key):
                stats["skipped"] += 1
                continue

            try:
                outcome = _archive(day, hour, session=session)
            except TransientFetchError as exc:
                stats["transient"] += 1
                print(
                    f"injury backfill: TRANSIENT FAILURE, could not check "
                    f"{day.isoformat()} {hour} -- {exc}. Not counted as "
                    "absent; will retry on the next run of this range."
                )
                if delay:
                    time.sleep(delay)
                continue

            if outcome is ArchiveOutcome.ARCHIVED:
                stats["fetched"] += 1
                try:
                    stats["ingested"] += ingest_report(con, raw_store.load("injury", key))
                except InjuryReportParseError as exc:
                    stats["parse_failed"] += 1
                    print(
                        f"injury backfill: PARSE FAILED for "
                        f"{day.isoformat()} {hour} -- {exc}. PDF is archived "
                        "on disk but was NOT ingested; a re-run will skip "
                        "(not retry) this slot -- recover it with "
                        "reingest_archived() after fixing the parser."
                    )
                except Exception as exc:  # noqa: BLE001 -- see docstring
                    # Guards against an unforeseen parse-time exception --
                    # concretely, a zero-page PDF currently raises IndexError
                    # rather than InjuryReportParseError (pre-existing, not
                    # fixed here). Bucketed with parse_failed: same practical
                    # outcome (fetched, not ingested), and logging the real
                    # exception type keeps it diagnosable rather than hidden.
                    stats["parse_failed"] += 1
                    print(
                        f"injury backfill: UNEXPECTED ERROR parsing/ingesting "
                        f"{day.isoformat()} {hour} -- {type(exc).__name__}: "
                        f"{exc}. PDF is archived on disk but was NOT ingested; "
                        "a re-run will skip (not retry) this slot -- recover "
                        "it with reingest_archived() after fixing the parser."
                    )
            elif outcome is ArchiveOutcome.BAD_CONTENT:
                stats["bad_content"] += 1
                print(
                    f"injury backfill: BAD CONTENT (200 response, non-PDF "
                    f"body) for {day.isoformat()} {hour} -- not counted as "
                    "absent; will retry on the next run of this range."
                )
            elif outcome in (ArchiveOutcome.NOT_PUBLISHED, ArchiveOutcome.BEFORE_ARCHIVE_START):
                # Same bucket as before this fix: a pre-ARCHIVE_START date
                # was already folded into `missing` (archive_report used to
                # return False for it, same as a confirmed 403/404), and
                # that categorization is unchanged here -- only whether the
                # politeness delay below is paid for it is (Finding 3).
                # ALREADY_ARCHIVED never reaches here: the `skipped` check
                # at the top of this loop intercepts that case before
                # `_archive` is ever called.
                stats["missing"] += 1

            if delay and outcome.request_made:
                time.sleep(delay)
        day += timedelta(days=1)

    return stats


def reingest_archived(
    con,
    start: date | None = None,
    end: date | None = None,
) -> dict[str, int]:
    """Re-ingest already-archived injury report PDFs, with NO network fetch.

    This is the recovery path `backfill_range`'s own `parse_failed` warning
    promises but does not implement (Task 9 review Finding 2): once a
    `parse_failed` slot's raw PDF is on disk, `backfill_range` will SKIP it
    on every future run (raw_store.exists() is true), so the only way to
    recover it -- for example after a parser bug fix -- is to re-ingest
    directly from the archive. This walks every blob already recorded in
    raw_store's manifest for the "injury" source and calls `ingest_report`
    on each one.

    Safe to re-run at any time, on any subset of the archive:
    `ingest_report` uses INSERT OR REPLACE, so re-ingesting an already-
    successfully-ingested report is a no-op change to the database, not a
    duplicate. A report that still fails to parse is logged and skipped --
    it must never abort the sweep over the rest of the archive.

    `start`/`end` (both inclusive, by the report's own `day` metadata
    recorded at archive time) optionally narrow the sweep; omitted, the
    entire archive is walked.

    I3: `InjuryReportEmptyError` (a genuinely empty "NOT YET SUBMITTED"-
    only slate -- Summer League, the All-Star break, a playoff off-day)
    is counted separately as `empty_no_filings`, NOT `still_failed`. All
    107 remaining real archive parse failures are exactly this case, with
    no data actually lost -- but because the CLI's exit code used to key
    off `still_failed` alone including these, `predictor reingest-injuries`
    exited 1 on EVERY run regardless of whether anything was actually
    broken, so an operator could never tell "fixed and fully recovered"
    apart from "still broken". `still_failed` is now reserved for a
    genuine, still-unresolved parse defect.
    """
    stats = {
        "found": 0,
        "ingested_ok": 0,
        "still_failed": 0,
        "empty_no_filings": 0,
        "rows_written": 0,
    }

    for entry in raw_store.iter_manifest("injury"):
        meta = entry.get("meta") or {}
        day_str = meta.get("day")
        if start is not None or end is not None:
            if day_str is None:
                continue
            day = date.fromisoformat(day_str)
            if start is not None and day < start:
                continue
            if end is not None and day > end:
                continue

        stats["found"] += 1
        key = entry["key"]
        try:
            content = raw_store.load("injury", key)
            rows = ingest_report(con, content)
        except InjuryReportEmptyError as exc:
            # Not a failure: a legitimately empty slate. Logged for
            # visibility, but deliberately NOT counted in `still_failed`
            # (see I3 docstring note above) -- this must never drive the
            # CLI's exit code.
            stats["empty_no_filings"] += 1
            print(f"injury reingest: EMPTY SLATE (no real filings) -- {key} -- {exc}")
            continue
        except Exception as exc:  # noqa: BLE001 -- mirrors backfill_range's
            # blanket ingest-time catch: a still-unparseable archived report
            # must be logged and skipped, never abort the sweep over the
            # rest of the archive.
            stats["still_failed"] += 1
            print(
                f"injury reingest: STILL FAILING -- {key} -- "
                f"{type(exc).__name__}: {exc}"
            )
            continue

        stats["ingested_ok"] += 1
        stats["rows_written"] += rows

    return stats
