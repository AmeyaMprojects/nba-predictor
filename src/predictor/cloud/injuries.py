"""The latest injury report, kept in state as per-report deltas.

Each cloud run fetches the newest 15-minute report
(`injury_report.fetch_latest`) and records it here. Its time is the PDF
header, never the filename. A report newer than every recorded one appends,
to ``state/injuries/<season>/<YYYY-MM>.jsonl`` (season and month of the ET
date of the report time):

- one ``report`` line: ``{"kind":"report","v":1,"report_at","fetched_at",
  "source_key","rows","content_sha"}``;
- ``set`` lines (the full row) for keys ``(game_date, team, player)`` that
  are new or changed since the previous report of the same season;
- ``drop`` lines (the key only) for keys that are gone.

``set``/``drop`` lines have the same shape as the Mac's injury history
(`state.encode_injury_history`), and `expand_into` replays them through
`state.expand_injury_history`. Unlike a Mac history file, a cloud file is
not self-contained: a season's files are replayed together, in order,
starting from empty.

Every report becomes an ``injury_reports`` row. Only a report with at
least one change gets full ``injury_status`` rows (at ``observed_at`` =
report time); an unchanged report is just its ``injury_reports`` row, and
readers take the rows of the latest earlier changed report. An empty
report (no filings yet) is a report with 0 rows that drops every row.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from predictor import db
from predictor.cloud import state
from predictor.sources import injury_report
from predictor.sources.injury_report import LatestReport, ParsedReport

INJURY_DIR = "injuries"

_KEY = ("game_date", "team", "player")
# Row fields stored per player; report_date and reconstructed are implied by
# the report (ET date of report_at, False), as `ingest_report` writes them.
ROW_FIELDS = _KEY + (
    "game_time", "matchup", "team_display", "player_display", "status", "reason",
)


@dataclass(frozen=True)
class InjuryOutcome:
    new: bool
    report_at: datetime | None
    rows: int
    changes: int


# --- rows and deltas ---------------------------------------------------------


def canonical_rows(parsed: ParsedReport) -> list[dict]:
    """The rows `injury_report.ingest_report` stores for this report, as
    plain JSON values, one per key (a later duplicate wins, as the insert's
    OR REPLACE does), sorted by key."""
    report_date = parsed.published_at.astimezone(injury_report.EASTERN).date()
    out: dict[tuple, dict] = {}
    for r in parsed.rows:
        game_date = r.game_date if r.game_date is not None else report_date
        row = {
            "game_date": game_date.isoformat(),
            "team": injury_report.team_abbr(r.team) or r.team,
            "player": injury_report.player_key(r.player),
            "game_time": r.game_time,
            "matchup": r.matchup,
            "team_display": r.team,
            "player_display": r.player,
            "status": r.status,
            "reason": r.reason,
        }
        out[_key(row)] = row
    return [out[k] for k in sorted(out)]


def _key(row: dict) -> tuple:
    return tuple(str(row[k]) for k in _KEY)


def _plain(row: dict) -> dict:
    return {f: state._encode_value(row.get(f), f) for f in ROW_FIELDS}


def content_sha(rows) -> str:
    """Full 64-hex sha256 of the rows (order does not matter)."""
    plain = sorted((_plain(r) for r in rows), key=_key)
    text = json.dumps(plain, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def delta(prev: list[dict], cur: list[dict], report_at: datetime) -> list[dict]:
    """``set`` lines (full row) for keys new or changed in ``cur``, then
    ``drop`` lines (key only) for keys of ``prev`` not in ``cur``; each group
    sorted by key."""
    db.require_utc(report_at, "report_at")
    before = {_key(r): _plain(r) for r in prev}
    after = {_key(r): _plain(r) for r in cur}
    lines = [{"kind": "set", **after[k]} for k in sorted(after) if before.get(k) != after[k]]
    lines += [
        {"kind": "drop", **{f: before[k][f] for f in _KEY}}
        for k in sorted(before.keys() - after.keys())
    ]
    return lines


# --- recording -----------------------------------------------------------------


def _season(at: datetime) -> str:
    return state._season_of_day(state._et_date(at))


def _path(state_dir: Path, at: datetime) -> Path:
    day = state._et_date(at)
    return state_dir / INJURY_DIR / _season(at) / f"{day.year:04d}-{day.month:02d}.jsonl"


def _previous_rows(con, report_at: datetime) -> list[dict]:
    """Rows of the previous recorded report of the same season, or [].

    Walks recorded reports newest first: an empty one means no rows; the
    first one with ``injury_status`` rows at its own time is the snapshot
    every later unchanged report repeats."""
    reports = db.POINT_IN_TIME_TABLES["injury_reports"]
    status = db.POINT_IN_TIME_TABLES["injury_status"]
    season = _season(report_at)
    recorded = con.execute(
        f"SELECT observed_at, row_count FROM {reports} WHERE observed_at < ?"
        " ORDER BY observed_at DESC",
        [report_at],
    ).fetchall()
    cols = ", ".join(ROW_FIELDS)
    for observed_at, row_count in recorded:
        if _season(observed_at) != season or row_count == 0:
            return []
        found = con.execute(
            f"SELECT {cols} FROM {status} WHERE observed_at = ?", [observed_at]
        ).fetchall()
        if found:
            return [dict(zip(ROW_FIELDS, r)) for r in found]
    return []


def _parse(body: bytes) -> tuple[datetime, list[dict]]:
    """(report time from the header, canonical rows). A report with no
    filings is 0 rows; any other parse failure propagates."""
    try:
        parsed = injury_report.parse_report(body)
    except injury_report.InjuryReportEmptyError as exc:
        published = exc.published_at
        if published is None:
            from predictor.sources.injury_slots import _header_time

            published = _header_time(body)
        return db.require_utc(published, "report_at"), []
    return db.require_utc(parsed.published_at, "report_at"), canonical_rows(parsed)


def record_report(con, state_dir: Path, latest: LatestReport) -> InjuryOutcome:
    """Record ``latest`` if it is newer than every recorded report.

    Appends the report and delta lines to state, inserts its
    ``injury_reports`` row and, when anything changed, its full
    ``injury_status`` snapshot (through `injury_report.ingest_report`).
    A parse failure other than an empty report raises (the PDF is already
    archived) and records nothing."""
    report_at, rows = _parse(latest.body)
    fetched_at = db.require_utc(latest.fetched_at, "fetched_at")
    reports = db.POINT_IN_TIME_TABLES["injury_reports"]
    (last,) = con.execute(f"SELECT max(observed_at) FROM {reports}").fetchone()
    if last is not None and report_at <= last:
        return InjuryOutcome(new=False, report_at=report_at, rows=len(rows), changes=0)

    sha = content_sha(rows)
    lines = delta(_previous_rows(con, report_at), rows, report_at)
    head = {
        "kind": "report",
        "v": state.STATE_VERSION,
        "report_at": report_at,
        "fetched_at": fetched_at,
        "source_key": latest.key,
        "rows": len(rows),
        "content_sha": sha,
    }
    state.append_records(_path(state_dir, report_at), [head, *lines])
    con.execute(
        f"INSERT INTO {reports} (observed_at, fetched_at, source_key, row_count, content_sha)"
        " VALUES (?, ?, ?, ?, ?)",
        [report_at, fetched_at, latest.key, len(rows), sha],
    )
    if lines and rows:
        injury_report.ingest_report(con, latest.body)
    return InjuryOutcome(new=True, report_at=report_at, rows=len(rows), changes=len(lines))


# --- expansion -------------------------------------------------------------------


def _season_lines(state_dir: Path) -> dict[str, list[dict]]:
    by_season: dict[str, list[dict]] = {}
    for path in sorted((state_dir / INJURY_DIR).glob("*/*.jsonl")):
        by_season.setdefault(path.parent.name, []).extend(state.read_records(path))
    return by_season


def _replay(lines: list[dict], where: str) -> tuple[list[dict], list[dict]]:
    """One season's lines -> (injury_reports rows, injury_status rows).

    Report lines without changes are left out of the stream handed to
    `state.expand_injury_history`, so it emits rows only for reports with at
    least one change."""
    reports: list[dict] = []
    stream: list[dict] = []
    pending: dict | None = None
    last_at: datetime | None = None
    for line in lines:
        if line["kind"] == "report":
            if line.get("v") != state.STATE_VERSION:
                raise state.StateVersionError(
                    f"{where}: injury report line version {line.get('v')} is not supported "
                    f"(this code reads version {state.STATE_VERSION})"
                )
            at = state._as_utc(line["report_at"])
            if last_at is not None and at <= last_at:
                raise ValueError(f"{where}: injury report {at} is not after {last_at}")
            last_at = at
            reports.append({
                "observed_at": at,
                "fetched_at": line["fetched_at"],
                "source_key": line["source_key"],
                "row_count": line["rows"],
                "content_sha": line["content_sha"],
            })
            pending = {
                "kind": "report",
                "v": state.STATE_VERSION,
                "observed_at": at,
                "report_date": at.astimezone(state.EASTERN).date().isoformat(),
                "reconstructed": False,
            }
            continue
        if last_at is None:
            raise ValueError(f"{where}: injury line before any report line")
        if pending is not None:
            stream.append(pending)
            pending = None
        stream.append(line)
    return reports, state.expand_injury_history(stream)


def expand_into(con, state_dir: Path) -> int:
    """Load every cloud injury report into the DB (INSERT OR IGNORE, so a
    report the Mac history also holds keeps the Mac rows). Returns the
    ``injury_status`` rows inserted."""
    total = 0
    for season, lines in sorted(_season_lines(state_dir).items()):
        reports, rows = _replay(lines, f"{INJURY_DIR}/{season}")
        state._insert(con, "injury_reports",
                      ("observed_at", "fetched_at", "source_key", "row_count", "content_sha"),
                      reports)
        total += state._insert(con, "injury_status", state.INJURY_COLUMNS, rows)
    return total
