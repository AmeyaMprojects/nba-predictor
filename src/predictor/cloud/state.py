"""The state directory: compact, append-only JSONL files in the public repo,
and the throwaway DuckDB rebuilt from them on every cloud run.

Layout (paths relative to ``state/``):

- ``results/<season>.jsonl`` -- games rows written by the cloud (``source``
  is kept in state only, never loaded).
- ``schedule/<season>/<YYYY-MM>.jsonl`` -- schedule change rows; plus
  ``schedule/espn_links.jsonl`` and ``schedule/id_links.jsonl``.
- ``odds/<season>/<YYYY-MM>.jsonl`` -- live odds rows (month and season from
  the ET date of ``observed_at``); attempts in ``odds/attempts-<YYYY-MM>.jsonl``.
- ``boxscores/<season>/<YYYY-MM>.jsonl`` -- ESPN box-score rows.
- ``injuries/<season>/<YYYY-MM>.jsonl`` -- cloud injury report deltas (Task 6).
- ``nba/`` -- whole-file snapshots written only by the Mac:
  ``games-<season>.jsonl``, ``schedule-<season>.jsonl`` (changed vintages
  only), ``odds-<season>.jsonl``, ``player_games/<season>/<YYYY-MM>.jsonl``
  and ``injuries/<season>/<YYYY-MM>.jsonl`` (the injury_status history as
  per-report deltas; each month file starts from an empty report, so it
  rebuilds on its own). The flat ``nba/`` tables are in table form (a
  ``{"columns":[...],"format":"table"}`` header, then one array per row).
- ``settings/stage1_settings.json`` -- optional settings override.

Every line is ``json.dumps(obj, sort_keys=True, separators=(",", ":"))``
plus ``\\n``. Dates are ISO dates, timestamps ISO UTC with ``+00:00``.
Code only appends to cloud files; only ``write_snapshot`` (``nba/`` only)
replaces a whole file.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from predictor import db
from predictor.config import season_label

STATE_DIRNAME = "state"
EASTERN = ZoneInfo("America/New_York")

GAMES_COLUMNS = (
    "game_id", "season", "game_date", "home_team", "away_team", "home_points",
    "away_points", "status", "reconstructed", "observed_at",
)
SCHEDULE_COLUMNS = (
    "game_id", "season", "game_date", "tip_off_utc", "home_team", "away_team",
    "arena_name", "arena_city", "arena_state", "is_neutral_reported", "is_neutral",
    "observed_at",
)
# A schedule vintage is kept by the Mac export only when one of these differs
# from the previous kept vintage of the same game.
SCHEDULE_CHANGE_FIELDS = (
    "game_date", "tip_off_utc", "home_team", "away_team", "arena_name", "arena_city",
    "arena_state", "is_neutral_reported", "is_neutral",
)
ODDS_COLUMNS = (
    "game_key", "book", "home_team", "away_team", "home_price", "away_price", "spread",
    "total", "observed_at", "game_id", "source", "reconstructed",
)
PLAYER_GAMES_COLUMNS = (
    "game_id", "season", "game_date", "season_type", "player_id", "player_name", "team",
    "minutes", "pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "ast",
    "stl", "blk", "tov", "pf", "plus_minus", "source", "reconstructed", "observed_at",
)
BOXSCORE_COLUMNS = (
    "game_id", "season", "game_date", "team", "espn_player_id", "player_name", "starter",
    "did_not_play", "minutes", "points", "rebounds", "assists", "plus_minus", "source",
    "observed_at",
)
INJURY_COLUMNS = (
    "report_date", "game_date", "game_time", "matchup", "team", "team_display", "player",
    "player_display", "status", "reason", "reconstructed", "observed_at",
)
_INJURY_KEY = ("game_date", "team", "player")
# Uniform within one report (checked per row; a differing row carries its own).
_INJURY_REPORT_FIELDS = ("report_date", "reconstructed")

INJURY_HISTORY_DIR = "nba/injuries"
_TABLE_FORMAT = "table"
ID_LINKS_PATH = "schedule/id_links.jsonl"
SETTINGS_OVERRIDE = "settings/stage1_settings.json"

# Columns that may hold a provisional "espn:<event id>" game id.
_GAME_ID_COLUMN = "game_id"


# --- encoding -------------------------------------------------------------


def _encode_value(value, field: str):
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{field} must be timezone-aware")
        return value.astimezone(UTC).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def encode_line(obj: dict) -> str:
    """One state line: sorted keys, no spaces, trailing newline."""
    plain = {k: _encode_value(v, k) for k, v in obj.items()}
    return json.dumps(plain, sort_keys=True, separators=(",", ":")) + "\n"


def _as_date(value) -> date:
    if isinstance(value, datetime):
        return value.astimezone(EASTERN).date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _as_utc(value) -> datetime:
    if isinstance(value, datetime):
        moment = value
    else:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if moment.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return moment.astimezone(UTC)


def _et_date(value) -> date:
    return _as_utc(value).astimezone(EASTERN).date()


def _season_of_day(day: date) -> str:
    return season_label(datetime(day.year, day.month, day.day, tzinfo=UTC))


# --- families --------------------------------------------------------------


@dataclass(frozen=True)
class Family:
    name: str
    table: str  # logical table name in db.POINT_IN_TIME_TABLES
    columns: tuple[str, ...]
    path_for: Callable[[dict], str]
    # Glob (relative to state/) matching every file of the family.
    glob: str = ""


def _month_path(prefix: str, season_key: Callable[[dict], str], day_key: Callable[[dict], date]):
    def path_for(row: dict) -> str:
        day = day_key(row)
        return f"{prefix}/{season_key(row)}/{day.year:04d}-{day.month:02d}.jsonl"

    return path_for


def _row_season(row: dict) -> str:
    return str(row["season"])


def _row_game_day(row: dict) -> date:
    return _as_date(row["game_date"])


def _observed_season(row: dict) -> str:
    return _season_of_day(_et_date(row["observed_at"]))


def _observed_day(row: dict) -> date:
    return _et_date(row["observed_at"])


FAMILIES: dict[str, Family] = {
    f.name: f
    for f in (
        Family("results", "games", GAMES_COLUMNS + ("source",),
               lambda r: f"results/{_row_season(r)}.jsonl", "results/*.jsonl"),
        Family("schedule", "schedule", SCHEDULE_COLUMNS,
               _month_path("schedule", _row_season, _row_game_day), "schedule/*/*.jsonl"),
        Family("odds", "odds_snapshots", ODDS_COLUMNS,
               _month_path("odds", _observed_season, _observed_day), "odds/*/*.jsonl"),
        Family("boxscores", "player_boxscores", BOXSCORE_COLUMNS,
               _month_path("boxscores", _row_season, _row_game_day), "boxscores/*/*.jsonl"),
        Family("nba_games", "games", GAMES_COLUMNS,
               lambda r: f"nba/games-{_row_season(r)}.jsonl", "nba/games-*.jsonl"),
        Family("nba_schedule", "schedule", SCHEDULE_COLUMNS,
               lambda r: f"nba/schedule-{_row_season(r)}.jsonl", "nba/schedule-*.jsonl"),
        Family("nba_odds", "odds_snapshots", ODDS_COLUMNS,
               lambda r: f"nba/odds-{_observed_season(r)}.jsonl", "nba/odds-*.jsonl"),
        Family("nba_player_games", "player_games", PLAYER_GAMES_COLUMNS,
               _month_path("nba/player_games", _row_season, _row_game_day),
               "nba/player_games/*/*.jsonl"),
    )
}

# Load order for a rebuild: Mac snapshots first, then cloud files.
_NBA_FAMILIES = ("nba_games", "nba_schedule", "nba_odds", "nba_player_games")
_CLOUD_FAMILIES = ("results", "schedule", "odds", "boxscores")


def _family(name: str) -> Family:
    try:
        return FAMILIES[name]
    except KeyError:
        raise ValueError(f"unknown state family {name!r}") from None


# --- writing ----------------------------------------------------------------


def append_rows(state_dir: Path, family: str, rows: list[dict]) -> list[Path]:
    """Append rows to their partition files; never rewrites a line.

    Returns the touched files in first-touched order."""
    fam = _family(family)
    by_path: dict[Path, list[str]] = {}
    for row in rows:
        line = encode_line({c: row.get(c) for c in fam.columns if c in row})
        by_path.setdefault(state_dir / fam.path_for(row), []).append(line)
    for path, lines in by_path.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write("".join(lines))
    return list(by_path)


def write_snapshot(
    state_dir: Path, rel_path: str, rows: list[dict], columns: tuple[str, ...] | None = None
) -> Path:
    """Replace one whole ``nba/`` snapshot file (Mac only).

    With ``columns``, the file is in table form: a header line
    ``{"columns":[...],"format":"table"}`` then one JSON array per row, in
    column order. That keeps the large history tables about a third of the
    size of keyed lines; ``read_rows`` turns them back into dicts."""
    parts = Path(rel_path).parts
    if not parts or parts[0] != "nba" or ".." in parts or Path(rel_path).is_absolute():
        raise ValueError(f"snapshots live under nba/ only, not {rel_path!r}")
    path = state_dir / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        if columns is None:
            for row in rows:
                fh.write(encode_line(row))
        else:
            fh.write(encode_line({"columns": list(columns), "format": _TABLE_FORMAT}))
            for row in rows:
                values = [_encode_value(row.get(c), c) for c in columns]
                fh.write(json.dumps(values, separators=(",", ":")) + "\n")
    os.replace(tmp, path)
    return path


# --- reading ----------------------------------------------------------------


def _read_lines(paths: Iterable[Path]) -> Iterator[dict]:
    """Every line of each file as a dict (table-form files are zipped back)."""
    for path in paths:
        columns: list[str] | None = None
        with path.open(encoding="utf-8") as fh:
            for n, line in enumerate(fh):
                if not line.strip():
                    continue
                obj = json.loads(line)
                if n == 0 and isinstance(obj, dict) and obj.get("format") == _TABLE_FORMAT:
                    columns = obj["columns"]
                    continue
                if columns is not None:
                    yield dict(zip(columns, obj, strict=True))
                else:
                    yield obj


def read_rows(state_dir: Path, family: str) -> Iterator[dict]:
    fam = _family(family)
    return _read_lines(sorted(state_dir.glob(fam.glob)))


def read_id_links(state_dir: Path) -> dict[str, str]:
    """provisional ``espn:`` id -> NBA game id; a later line wins."""
    path = state_dir / ID_LINKS_PATH
    if not path.exists():
        return {}
    return {r["provisional_id"]: r["game_id"] for r in _read_lines([path])}


def settings_path(state_dir: Path) -> Path:
    from predictor.model import settings as model_settings

    override = state_dir / SETTINGS_OVERRIDE
    return override if override.exists() else model_settings.SETTINGS_PATH


# --- loading ----------------------------------------------------------------


def _column_types(con, physical: str) -> dict[str, str]:
    return dict(
        con.execute(
            "SELECT column_name, data_type FROM information_schema.columns"
            " WHERE table_name = ? AND table_schema = 'main'",
            [physical],
        ).fetchall()
    )


def _as_text(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(_encode_value(value, "value"))


def _insert(con, logical: str, columns: tuple[str, ...], rows: list[dict]) -> int:
    """INSERT OR IGNORE ``rows`` into the logical table; returns rows inserted.

    Values travel as text and are cast by DuckDB to each column's type, which
    keeps every value exact (floats as their shortest repr, timestamps with
    microseconds)."""
    if not rows:
        return 0
    import pandas as pd

    physical = db.POINT_IN_TIME_TABLES[logical]
    types = _column_types(con, physical)
    frame = pd.DataFrame(
        {c: pd.Series([_as_text(r.get(c)) for r in rows], dtype=object) for c in columns}
    )
    before = con.execute(f"SELECT count(*) FROM {physical}").fetchone()[0]
    name = "_state_rows"
    con.register(name, frame)
    try:
        select = ", ".join(f'CAST("{c}" AS {types[c]})' for c in columns)
        cols = ", ".join(f'"{c}"' for c in columns)
        con.execute(f"INSERT OR IGNORE INTO {physical} ({cols}) SELECT {select} FROM {name}")
    finally:
        con.unregister(name)
    return con.execute(f"SELECT count(*) FROM {physical}").fetchone()[0] - before


def load_rows(con, family: str, rows: list[dict], id_links: dict[str, str]) -> int:
    """INSERT OR IGNORE one family's rows, mapping linked ``espn:`` game ids
    to their NBA ids. Returns the number of rows inserted."""
    fam = _family(family)
    columns = tuple(c for c in fam.columns if c != "source" or fam.table != "games")
    prepared = []
    for row in rows:
        row = dict(row)
        gid = row.get(_GAME_ID_COLUMN)
        if isinstance(gid, str) and gid in id_links:
            row[_GAME_ID_COLUMN] = id_links[gid]
        prepared.append(row)
    if fam.table == "player_boxscores":
        # State keeps the ESPN athlete id only; the NBA id is resolved at
        # rebuild time (Task 5 adds the player links).
        for row in prepared:
            row["player_id"] = row.get("player_id") or f"espn:{row['espn_player_id']}"
        columns = columns + ("player_id",)
    return _insert(con, fam.table, columns, prepared)


# --- injury_status history (Mac snapshot, per-report deltas) ---------------


def _injury_payload(row: dict, report: dict) -> dict:
    payload = {
        c: row[c] for c in INJURY_COLUMNS
        if c not in _INJURY_KEY and c != "observed_at" and c not in _INJURY_REPORT_FIELDS
    }
    for f in _INJURY_REPORT_FIELDS:
        if row[f] != report[f]:
            payload[f] = row[f]
    return payload


def encode_injury_history(rows: list[dict]) -> list[dict]:
    """injury_status rows (one month, any order) -> delta lines.

    Each report (distinct ``observed_at``) becomes a ``report`` line, then
    ``set`` lines (full row) for keys new or changed since the previous
    report and ``drop`` lines (key only) for keys gone. The first report
    starts from empty."""
    reports: dict[datetime, list[dict]] = {}
    for row in rows:
        reports.setdefault(_as_utc(row["observed_at"]), []).append(row)
    lines: list[dict] = []
    prev: dict[tuple, dict] = {}
    for observed_at in sorted(reports):
        members = sorted(reports[observed_at], key=lambda r: tuple(str(r[k]) for k in _INJURY_KEY))
        head = {f: members[0][f] for f in _INJURY_REPORT_FIELDS}
        lines.append({"kind": "report", "observed_at": observed_at, **head})
        cur = {
            tuple(_encode_value(r[k], k) for k in _INJURY_KEY): {
                k: _encode_value(v, k) for k, v in _injury_payload(r, head).items()
            }
            for r in members
        }
        for key in sorted(cur):
            if prev.get(key) != cur[key]:
                lines.append({"kind": "set", **dict(zip(_INJURY_KEY, key)), **cur[key]})
        for key in sorted(prev.keys() - cur.keys()):
            lines.append({"kind": "drop", **dict(zip(_INJURY_KEY, key))})
        prev = cur
    return lines


def expand_injury_history(lines: Iterable[dict]) -> list[dict]:
    """Delta lines of one file -> the full injury_status rows."""
    out: list[dict] = []
    current: dict[tuple, dict] = {}
    report: dict | None = None

    def flush():
        if report is not None:
            for key in sorted(current):
                row = {f: report[f] for f in _INJURY_REPORT_FIELDS}
                row.update(dict(zip(_INJURY_KEY, key)))
                row.update(current[key])
                row["observed_at"] = report["observed_at"]
                out.append(row)

    for line in lines:
        kind = line["kind"]
        if kind == "report":
            flush()
            report = line
            continue
        if report is None:
            raise ValueError("injury history line before any report line")
        key = tuple(line[k] for k in _INJURY_KEY)
        if kind == "set":
            current[key] = {k: v for k, v in line.items() if k != "kind" and k not in _INJURY_KEY}
        elif kind == "drop":
            current.pop(key, None)
        else:
            raise ValueError(f"unknown injury history line kind {kind!r}")
    flush()
    return out


def _load_injury_history(con, state_dir: Path) -> int:
    total = 0
    for path in sorted(state_dir.glob(f"{INJURY_HISTORY_DIR}/*/*.jsonl")):
        total += _insert(con, "injury_status", INJURY_COLUMNS,
                         expand_injury_history(_read_lines([path])))
    return total


# --- rebuild ----------------------------------------------------------------


@dataclass(frozen=True)
class RebuildReport:
    rows: dict[str, int]
    seconds: float


def rebuild_db(state_dir: Path, db_path: Path) -> RebuildReport:
    """Build a fresh DuckDB at ``db_path`` from the state directory."""
    started = time.perf_counter()
    for stale in (db_path, db_path.with_name(db_path.name + ".wal")):
        if stale.exists():
            stale.unlink()
    con = db.connect(db_path)
    try:
        db.migrate(con)
        id_links = read_id_links(state_dir)
        counts: dict[str, int] = {name: 0 for name in db.POINT_IN_TIME_TABLES}
        for name in _NBA_FAMILIES + _CLOUD_FAMILIES:
            fam = FAMILIES[name]
            counts[fam.table] += load_rows(con, name, list(read_rows(state_dir, name)), id_links)
        counts["injury_status"] += _load_injury_history(con, state_dir)
        con.execute("CHECKPOINT")
    finally:
        con.close()
    return RebuildReport(rows=counts, seconds=time.perf_counter() - started)


# --- export (Mac) -----------------------------------------------------------


def _fetch(con, logical: str, columns: tuple[str, ...], where: str = "", params=()) -> list[dict]:
    physical = db.POINT_IN_TIME_TABLES[logical]
    cols = ", ".join(columns)
    result = con.execute(f"SELECT {cols} FROM {physical} {where}", list(params)).fetchall()
    return [dict(zip(columns, r)) for r in result]


def seasons_in_db(con) -> list[str]:
    """Every season with rows in a table the export covers."""
    found: set[str] = set()
    for logical in ("games", "schedule", "player_games"):
        physical = db.POINT_IN_TIME_TABLES[logical]
        found.update(r[0] for r in con.execute(f"SELECT DISTINCT season FROM {physical}").fetchall())
    for logical in ("odds_snapshots", "injury_status"):
        physical = db.POINT_IN_TIME_TABLES[logical]
        for (observed_at,) in con.execute(f"SELECT DISTINCT observed_at FROM {physical}").fetchall():
            found.add(_season_of_day(_et_date(observed_at)))
    return sorted(found)


def _sort_key(row: dict, fields: tuple[str, ...]) -> tuple:
    return tuple((row[f] is None, row[f]) for f in fields)


def _changed_vintages(rows: list[dict]) -> list[dict]:
    kept: list[dict] = []
    last: dict[str, tuple] = {}
    for row in sorted(rows, key=lambda r: (r["game_id"], _as_utc(r["observed_at"]))):
        sig = tuple(row[f] for f in SCHEDULE_CHANGE_FIELDS)
        if last.get(row["game_id"]) != sig:
            kept.append(row)
            last[row["game_id"]] = sig
    return kept


def _group(rows: Iterable[dict], path_for: Callable[[dict], str]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in rows:
        out.setdefault(path_for(row), []).append(row)
    return out


def export_from_db(con, state_dir: Path, seasons: list[str]) -> list[Path]:
    """Write the Mac's ``nba/`` snapshots for ``seasons``; returns the files.

    Rows are sorted so an unchanged DB re-exports byte-identical files."""
    wanted = set(seasons)
    written: list[Path] = []
    marks = ", ".join("?" for _ in wanted) or "NULL"
    in_seasons = f"WHERE season IN ({marks})"

    games = _fetch(con, "games", GAMES_COLUMNS, in_seasons, sorted(wanted))
    sched = _changed_vintages(_fetch(con, "schedule", SCHEDULE_COLUMNS, in_seasons, sorted(wanted)))
    players = _fetch(con, "player_games", PLAYER_GAMES_COLUMNS, in_seasons, sorted(wanted))
    odds = [r for r in _fetch(con, "odds_snapshots", ODDS_COLUMNS) if _observed_season(r) in wanted]
    for family, rows, order in (
        ("nba_games", games, ("game_id", "observed_at")),
        ("nba_schedule", sched, ("game_id", "observed_at")),
        ("nba_odds", odds, ("observed_at", "game_key", "book")),
        ("nba_player_games", players, ("game_id", "player_id", "observed_at")),
    ):
        fam = FAMILIES[family]
        for rel, part in sorted(_group(rows, fam.path_for).items()):
            part.sort(key=lambda r: _sort_key(r, order))
            written.append(write_snapshot(state_dir, rel, part, fam.columns))

    injuries = [
        r for r in _fetch(con, "injury_status", INJURY_COLUMNS) if _observed_season(r) in wanted
    ]
    for rel, part in sorted(_group(injuries, _injury_history_path).items()):
        written.append(write_snapshot(state_dir, rel, encode_injury_history(part)))
    return written


def _injury_history_path(row: dict) -> str:
    day = _observed_day(row)
    return f"{INJURY_HISTORY_DIR}/{_observed_season(row)}/{day.year:04d}-{day.month:02d}.jsonl"
