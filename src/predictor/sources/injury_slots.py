"""Backfill the injury-report slots the replay needs.

The stored injury archive holds one report a day (the 5 PM ET slot). The
replay makes two calls per game, and each needs the latest report published
at or before its call time:

- the early call, at 12:30 UTC on the game date (the live 18:00 IST rule),
  clipped so it is never after the final call;
- the final call, at tip-off minus 45 minutes.

For each distinct call time this walks the report slots backwards from the
call time, for up to three hours, and stops at the first report whose
header time is at or before the call time. Every report fetched on the way
is archived and ingested, because each one is useful history.

Two filename patterns exist:

- before 2025-12-22, hourly: ``Injury-Report_2024-01-15_07PM.pdf``;
- from 2025-12-22, every 15 minutes: ``Injury-Report_2025-12-22_07_15PM.pdf``.

The time of a report is always its parsed header time, never the time in
its filename. Many slots were never published or are gone (403/404); those
holes are normal and are counted, not treated as errors.
"""

from __future__ import annotations

import io
import time as _time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

import pdfplumber

from predictor import db, raw_store
from predictor.backtest import tipoff as tipoff_mod
from predictor.model.venues import competitive_sql
from predictor.sources import injury_report
from predictor.sources.injury_report import ArchiveOutcome, TransientFetchError

EASTERN = ZoneInfo("America/New_York")

NEW_PATTERN_FROM = date(2025, 12, 22)
MAX_WALKBACK = timedelta(hours=3)
EARLY_CALL_UTC = time(12, 30)
FINAL_CALL_LEAD = timedelta(minutes=45)

_QUARTER = timedelta(minutes=15)

COUNT_KEYS = (
    "targets",
    "satisfied",
    "unsatisfied",
    "fetched",
    "not_published",
    "transient",
    "bad_content",
    "parse_failed",
    "already_archived",
    "empty_slate",
)


def _sleep(seconds: float) -> None:
    """Politeness delay, indirected so tests can stub it."""
    _time.sleep(seconds)


def call_times(game_date: date, tip_off: datetime) -> tuple[datetime, datetime]:
    """(early, final) call times in UTC for a game.

    final = tip-off - 45 min; early = 12:30 UTC on the game date, clipped
    so it is never after final.
    """
    tip = db.require_utc(tip_off, "tip_off")
    final = tip - FINAL_CALL_LEAD
    early = min(datetime.combine(game_date, EARLY_CALL_UTC, UTC), final)
    return early, final


def slot_label(slot_et: datetime) -> str:
    """Filename label for a slot at the given Eastern wall-clock time.

    Hourly ("07PM") before NEW_PATTERN_FROM, quarter-hourly ("07_15PM")
    from it on.
    """
    if slot_et.date() >= NEW_PATTERN_FROM:
        return slot_et.strftime("%I_%M%p")
    return slot_et.strftime("%I%p")


def _is_slot(slot_et: datetime) -> bool:
    if slot_et.minute % 15:
        return False
    return slot_et.date() >= NEW_PATTERN_FROM or slot_et.minute == 0


def candidate_slots(target: datetime) -> list[tuple[date, str]]:
    """Slots whose label time is <= target, newest first, back to and
    including target - MAX_WALKBACK.

    Each slot is (Eastern date, label). Hourly slots before
    NEW_PATTERN_FROM, quarter-hour slots from it on; a walk that crosses
    the switch uses each day's own pattern.
    """
    target = db.require_utc(target, "target")
    bound = target - MAX_WALKBACK
    # Eastern offsets are whole hours, so flooring to the quarter hour in
    # UTC floors the Eastern wall clock too. Stepping in UTC keeps DST
    # transitions correct.
    t = target.replace(second=0, microsecond=0)
    t -= timedelta(minutes=t.minute % 15)
    slots: list[tuple[date, str]] = []
    while t >= bound:
        et = t.astimezone(EASTERN)
        if _is_slot(et):
            slots.append((et.date(), slot_label(et)))
        t -= _QUARTER
    return slots


@dataclass(frozen=True)
class SlotTarget:
    game_id: str
    kind: Literal["early", "final"]
    at: datetime


def replay_targets(con, start: date, end: date) -> list[SlotTarget]:
    """Call times for every competitive game dated in [start, end] that has
    a tip-off, deduplicated by `at` across games (the first game, in
    (game_date, game_id) order, keeps the target). Sorted by `at`.
    """
    games = db.POINT_IN_TIME_TABLES["games"]
    rows = con.execute(
        f"SELECT game_id, MIN(game_date) AS game_date, MIN(home_team), "
        f"MIN(away_team) FROM {games} "
        f"WHERE {competitive_sql()} "
        f"GROUP BY game_id HAVING MIN(game_date) BETWEEN ? AND ? "
        f"ORDER BY game_date, game_id",
        [start, end],
    ).fetchall()
    index = tipoff_mod.tipoff_index(con)

    seen: set[datetime] = set()
    targets: list[SlotTarget] = []
    for game_id, game_date, home, away in rows:
        tip = tipoff_mod.resolve_tipoff(index, game_date, home, away)
        if tip is None:
            continue
        early, final = call_times(game_date, tip.astimezone(UTC))
        for kind, at in (("early", early), ("final", final)):
            if at in seen:
                continue
            seen.add(at)
            targets.append(SlotTarget(game_id=game_id, kind=kind, at=at))
    targets.sort(key=lambda t: t.at)
    return targets


def _header_time(pdf_bytes: bytes) -> datetime:
    """The report's publication time, from its page-1 header only."""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        first_line = pdf.pages[0].extract_text().split("\n")[0]
    return injury_report._published_at(first_line)


def backfill_slots(
    con,
    targets: Iterable[SlotTarget],
    delay: float = 0.4,
    session=None,
) -> dict[str, int]:
    """Fetch, archive and ingest the report slots each target needs.

    For each distinct `at`, walk `candidate_slots(at)` and stop at the first
    archived report whose header time is <= at. A report published after
    `at` is kept, but does not satisfy the target. Counts:

    - targets / satisfied / unsatisfied: distinct call times, and whether
      a report at or before each was found within the walk-back. An
      unsatisfied target is expected (holes); the replay falls back.
    - fetched: newly archived PDFs. already_archived: slots already in the
      raw store (not re-fetched or re-ingested).
    - not_published: confirmed 403/404 (or before the archive start).
    - transient / bad_content: could not check; not archived, so a re-run
      retries them.
    - parse_failed: archived but not ingested; recover with
      `predictor reingest-injuries` after a parser fix.
    - empty_slate: archived, no real filings yet (every team "NOT YET
      SUBMITTED"). Not a failure; it still has a header time.
    """
    session = session or injury_report.new_session()
    stats = {key: 0 for key in COUNT_KEYS}

    for at in sorted({t.at for t in targets}):
        stats["targets"] += 1
        satisfied = False
        for day, label in candidate_slots(at):
            key = injury_report.raw_key(day, label)
            if raw_store.exists("injury", key):
                stats["already_archived"] += 1
                try:
                    published = _header_time(raw_store.load("injury", key))
                except Exception as exc:  # noqa: BLE001 -- a bad archived file must not abort the run
                    print(
                        f"injury slots: could not read the header time of "
                        f"archived {key} -- {type(exc).__name__}: {exc}"
                    )
                    continue
                if published <= at:
                    satisfied = True
                    break
                continue

            try:
                outcome = injury_report._archive(day, label, session=session)
            except TransientFetchError as exc:
                stats["transient"] += 1
                print(
                    f"injury slots: TRANSIENT FAILURE, could not check {key} "
                    f"-- {exc}. Not counted as absent; a re-run retries it."
                )
                if delay:
                    _sleep(delay)
                continue
            if delay and outcome.request_made:
                _sleep(delay)

            if outcome in (ArchiveOutcome.NOT_PUBLISHED, ArchiveOutcome.BEFORE_ARCHIVE_START):
                stats["not_published"] += 1
                continue
            if outcome is ArchiveOutcome.BAD_CONTENT:
                stats["bad_content"] += 1
                print(
                    f"injury slots: BAD CONTENT (200 response, non-PDF body) "
                    f"for {key} -- not archived; a re-run retries it."
                )
                continue

            # ARCHIVED (ALREADY_ARCHIVED cannot happen: checked above).
            stats["fetched"] += 1
            content = raw_store.load("injury", key)
            try:
                injury_report.ingest_report(con, content)
            except injury_report.InjuryReportEmptyError:
                stats["empty_slate"] += 1
            except Exception as exc:  # noqa: BLE001 -- see backfill_range
                stats["parse_failed"] += 1
                print(
                    f"injury slots: PARSE FAILED for {key} -- "
                    f"{type(exc).__name__}: {exc}. The PDF is archived but was "
                    "NOT ingested; recover it with 'predictor "
                    "reingest-injuries' after fixing the parser."
                )
                continue
            try:
                published = _header_time(content)
            except Exception:  # noqa: BLE001
                continue
            if published <= at:
                satisfied = True
                break

        stats["satisfied" if satisfied else "unsatisfied"] += 1

    return stats
