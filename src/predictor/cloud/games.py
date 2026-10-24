"""ESPN scoreboard events -> schedule change rows, finals and game links.

`sync_games` only computes rows; `apply_sync` appends them to the state
directory and loads them into the run's DuckDB.

Game ids: an ESPN event keeps the NBA id it was linked to before
(``schedule/espn_links.jsonl``, latest line wins); otherwise it is linked
by ET date and teams against the latest NBA schedule (exact date, or one
day either side when exactly one game fits among NBA ids not already
claimed by a known link or an exact match); otherwise it gets the
provisional id ``espn:<event id>``. When a provisional game later links to
an NBA id, ``schedule/id_links.jsonl`` records it, and every rebuild maps
the provisional rows onto the NBA id.

A TBD time keeps the known tip only on the same date. A postponed,
cancelled or suspended game gets a TBD (None) tip, no final, and a line in
``GameSync.postponed``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from predictor import db
from predictor.cloud import state
from predictor.model.live import slate_date
from predictor.model.venues import PROVISIONAL_PREFIX
from predictor.sources import odds
from predictor.sources.espn import EspnGame
from predictor.sources.results import FINISHED_AFTER_TIP

COMPETITIVE_SEASON_TYPES = frozenset({2, 3, 5})
ESPN_LINKS_PATH = "schedule/espn_links.jsonl"
SOURCE = "espn"

# Near days (slate -1..+1) are fetched every run; far days (slate -2 and
# +2..+7) only when the previous far fetch is FAR_INTERVAL old (`far_due`).
NEAR_OFFSETS = (-1, 0, 1)
FAR_OFFSETS = (-2, 2, 3, 4, 5, 6, 7)
FAR_INTERVAL = timedelta(minutes=60)
# Fields whose change makes a new schedule row.
_CHANGE_FIELDS = ("game_date", "tip_off_utc", "home_team", "away_team", "is_neutral")


@dataclass
class GameSync:
    schedule_rows: list[dict] = field(default_factory=list)
    finals: list[dict] = field(default_factory=list)
    espn_links: list[dict] = field(default_factory=list)
    id_links: list[dict] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    # "<AWAY>@<HOME> <date> (<status name>)" per postponed/cancelled game;
    # its schedule row has no tip (TBD), so no call and no final follow.
    postponed: list[str] = field(default_factory=list)


def scoreboard_days(now: datetime, include_far: bool) -> list[date]:
    """ET dates to fetch this run, in order: the near days always, plus the
    far days when ``include_far`` (together slate - 2 .. slate + 7)."""
    slate = slate_date(now)
    offsets = NEAR_OFFSETS + (FAR_OFFSETS if include_far else ())
    return sorted(slate + timedelta(days=n) for n in offsets)


def far_due(last_far_at: datetime | None, now: datetime) -> bool:
    """True when the far days should be fetched: never fetched, or the last
    far fetch is at least FAR_INTERVAL old."""
    db.require_utc(now, "now")
    return last_far_at is None or now - last_far_at >= FAR_INTERVAL


def provisional_id(espn_id: str) -> str:
    return f"{PROVISIONAL_PREFIX}{espn_id}"


def _is_provisional(game_id: str | None) -> bool:
    return bool(game_id) and game_id.startswith(PROVISIONAL_PREFIX)


def _label(game: EspnGame) -> str:
    away = game.away_team or game.away_name or "?"
    home = game.home_team or game.home_name or "?"
    return f"{away}@{home} {game.game_date.isoformat()}"


def _skip_reason(game: EspnGame) -> str | None:
    if game.season_type not in COMPETITIVE_SEASON_TYPES:
        return f"season type {game.season_type}"
    unknown = [
        name or "?"
        for team, name in ((game.away_team, game.away_name), (game.home_team, game.home_name))
        if team is None
    ]
    if unknown:
        return "unknown team " + ", ".join(repr(n) for n in unknown)
    return None


def _latest_schedule(con, game_ids: set[str]) -> dict[str, dict]:
    if not game_ids:
        return {}
    table = db.POINT_IN_TIME_TABLES["schedule"]
    cols = state.SCHEDULE_COLUMNS
    rows = con.execute(
        f"""
        SELECT {", ".join(cols)} FROM (
            SELECT *, row_number() OVER (
                PARTITION BY game_id ORDER BY observed_at DESC) AS rn
            FROM {table} WHERE game_id IN (SELECT unnest(?))
        ) WHERE rn = 1
        """,
        [sorted(game_ids)],
    ).fetchall()
    return {r[0]: dict(zip(cols, r, strict=True)) for r in rows}


def _final_ids(con, game_ids: set[str]) -> set[str]:
    if not game_ids:
        return set()
    table = db.POINT_IN_TIME_TABLES["games"]
    rows = con.execute(
        f"SELECT DISTINCT game_id FROM {table} "
        "WHERE status = 'FINAL' AND game_id IN (SELECT unnest(?))",
        [sorted(game_ids)],
    ).fetchall()
    return {r[0] for r in rows}


def _nba_link_index(con, games: list[EspnGame]):
    """`odds.build_link_index` over NBA-id games near the events' dates.
    Provisional games are left out, so an event never links to itself."""
    if not games:
        return {}
    days = [g.game_date for g in games]
    listed = odds.latest_schedule_games(
        con, min(days) - timedelta(days=1), max(days) + timedelta(days=1)
    )
    return odds.build_link_index(g for g in listed if not _is_provisional(g.game_id))


def _exact_match(index, game: EspnGame) -> str | None:
    exact = index.get((game.game_date, game.home_team, game.away_team), [])
    return exact[0] if len(exact) == 1 else None


def _without(index, claimed: set[str]):
    return {k: [g for g in ids if g not in claimed] for k, ids in index.items()}


def _resolve_ids(games: list[EspnGame], known_links: dict[str, str], index) -> list[str]:
    """Game id per event. Known NBA links and exact-date matches claim their
    NBA ids first; the one-day fallback (`odds.link_game`) then only sees
    unclaimed ids, so a back-to-back pair never shares one NBA game."""
    claimed = {g for g in known_links.values() if g and not _is_provisional(g)}
    chosen: list[str | None] = []
    for game in games:
        known = known_links.get(game.espn_id)
        gid = known if known and not _is_provisional(known) else _exact_match(index, game)
        chosen.append(gid)
        if gid:
            claimed.add(gid)
    out: list[str] = []
    for game, gid in zip(games, chosen, strict=True):
        if gid is None:
            free = _without(index, claimed)
            gid, _ = odds.link_game(free, game.game_date, game.home_team, game.away_team)
            if gid:
                claimed.add(gid)
        out.append(gid or provisional_id(game.espn_id))
    return out


def _schedule_row(game: EspnGame, game_id: str, latest: dict | None,
                  fetched_at: datetime) -> dict:
    tip = game.tip_off_utc
    if game.postponed:
        tip = None  # TBD until ESPN lists a new time
    elif tip is None and latest is not None and latest["game_date"] == game.game_date:
        # A TBD time keeps the known tip only on the same date.
        tip = latest["tip_off_utc"]
    row = {
        "game_id": game_id,
        "season": game.season,
        "game_date": game.game_date,
        "tip_off_utc": tip,
        "home_team": game.home_team,
        "away_team": game.away_team,
        "observed_at": fetched_at,
    }
    if latest is not None and not _is_provisional(game_id):
        # A linked game keeps the NBA's arena and neutral-site fields.
        row["season"] = latest["season"]
        for f in ("arena_name", "arena_city", "arena_state", "is_neutral_reported",
                  "is_neutral"):
            row[f] = latest[f]
    else:
        row.update(
            arena_name=game.venue_name, arena_city=game.venue_city,
            arena_state=game.venue_state, is_neutral_reported=game.neutral,
            is_neutral=game.neutral,
        )
    return row


def _changed(row: dict, latest: dict | None) -> bool:
    if latest is None:
        return True
    return any(row[f] != latest[f] for f in _CHANGE_FIELDS)


def sync_games(
    con, espn_games: list[EspnGame], fetched_at: datetime, known_links: dict[str, str],
    malformed: list[str] | None = None,
) -> GameSync:
    """Schedule change rows, new finals and link rows for one scoreboard.
    ``malformed`` (from `espn.load_scoreboards`) is carried into ``skipped``."""
    db.require_utc(fetched_at, "fetched_at")
    stamp = fetched_at.isoformat()
    out = GameSync(skipped=list(malformed or []))
    kept: list[EspnGame] = []
    for game in espn_games:
        reason = _skip_reason(game)
        if reason:
            out.skipped.append(f"{_label(game)} ({reason})")
        else:
            kept.append(game)

    index = _nba_link_index(con, kept)
    resolved: list[tuple[EspnGame, str, str | None]] = []
    for game, game_id in zip(kept, _resolve_ids(kept, known_links, index), strict=True):
        known = known_links.get(game.espn_id)
        if known != game_id:
            out.espn_links.append(
                {"espn_id": game.espn_id, "game_id": game_id, "linked_at": stamp}
            )
            if _is_provisional(known) and not _is_provisional(game_id):
                out.id_links.append(
                    {"provisional_id": known, "game_id": game_id, "linked_at": stamp}
                )
        resolved.append((game, game_id, known if _is_provisional(known) else None))

    ids = {gid for _, gid, _ in resolved}
    latest = _latest_schedule(con, ids)
    finals = _final_ids(con, ids | {p for _, _, p in resolved if p})
    for game, game_id, provisional in resolved:
        row = _schedule_row(game, game_id, latest.get(game_id), fetched_at)
        if _changed(row, latest.get(game_id)):
            out.schedule_rows.append(row)
        if game.postponed:
            out.postponed.append(f"{_label(game)} ({game.status_name})")
            continue
        tip = row["tip_off_utc"]
        if (
            game.completed
            and game.home_points is not None
            and game.away_points is not None
            and tip is not None
            and fetched_at >= tip + FINISHED_AFTER_TIP
            and game_id not in finals
            and provisional not in finals
        ):
            out.finals.append({
                "game_id": game_id,
                "season": row["season"],
                "game_date": game.game_date,
                "home_team": game.home_team,
                "away_team": game.away_team,
                "home_points": game.home_points,
                "away_points": game.away_points,
                "status": "FINAL",
                "reconstructed": False,
                "observed_at": fetched_at,
                "source": SOURCE,
            })
            finals.add(game_id)
    return out


def read_espn_links(state_dir: Path) -> dict[str, str]:
    """ESPN event id -> game id; a later line wins."""
    path = state_dir / ESPN_LINKS_PATH
    if not path.exists():
        return {}
    return {r["espn_id"]: r["game_id"] for r in state.read_records(path)}


def apply_sync(con, state_dir: Path, sync: GameSync) -> list[Path]:
    """Append the sync's rows to the state directory (links first, so a
    rebuild never sees a row without its link) and load schedule and finals
    into ``con``. Returns the touched files."""
    touched: list[Path] = []
    if sync.espn_links:
        touched.append(state.append_records(state_dir / ESPN_LINKS_PATH, sync.espn_links))
    if sync.id_links:
        touched.append(state.append_records(state_dir / state.ID_LINKS_PATH, sync.id_links))
    id_links = state.read_id_links(state_dir)
    if sync.schedule_rows:
        touched += state.append_rows(state_dir, "schedule", sync.schedule_rows)
        state.load_rows(con, "schedule", sync.schedule_rows, id_links)
    if sync.finals:
        touched += state.append_rows(state_dir, "results", sync.finals)
        state.load_rows(con, "results", sync.finals, id_links)
    return touched
