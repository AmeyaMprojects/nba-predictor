"""ESPN box-score players -> NBA player ids, box-score state rows, and which
game summaries are due.

Matching reuses plan 1's linker (`predictor.model.player_ids`): names are
compared by `name_key`, against an index built only from the NBA game logs
in state (``state/nba/player_games``, the last LINK_SEASONS seasons), so
a retired namesake or a player the logs have not seen yet is never guessed.
Order, for one ESPN athlete:

1. a hand alias by ESPN id (``players/aliases.json`` ``by_espn_id``) -- it
   also corrects an existing link;
2. an existing non-null link (``players/links.jsonl``, latest line wins);
3. a hand alias by name (``by_name``, keys compared after `normalize_name`);
4. a unique name match;
5. several name matches narrowed to one by team (any indexed season);
6. otherwise ``unmatched``.

New or changed links are appended; an unmatched athlete is stored once and
re-tried on every run, appending again only when a link is found (e.g.
after the Mac's weekly history refresh brings in his game logs).
Box-score state rows keep only the ESPN athlete id; ``player_id`` is
resolved on load (`resolved_player_ids`): the NBA id as a string, else
``espn:<athlete id>``, so a later alias fixes old rows on the next rebuild.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal

from predictor import db
from predictor.cloud import state
from predictor.config import season_label
from predictor.model import player_ids
from predictor.model.live import slate_date
from predictor.model.venues import PROVISIONAL_PREFIX
from predictor.sources.espn import EspnPlayerLine

LINKS_PATH = "players/links.jsonl"
ALIASES_PATH = "players/aliases.json"
# Seasons of NBA game logs a name may link against (now's season and the two before).
LINK_SEASONS = 3
SOURCE = "espn"
MAX_SUMMARIES_PER_RUN = 15
# Final games dated slate - DUE_DAYS .. slate get their summary fetched.
DUE_DAYS = 3

Method = Literal["existing", "alias", "name", "name+team", "unmatched"]

_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv"})


def normalize_name(name: str) -> str:
    """Readable name key: accents stripped, lowercase, apostrophes and
    periods removed, hyphens as spaces, Jr/Sr/II/III/IV dropped."""
    decomposed = unicodedata.normalize("NFKD", name)
    text = "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()
    text = re.sub(r"['’.]", "", text).replace("-", " ")
    return " ".join(t for t in text.split() if t not in _SUFFIXES)


class PlayerIndex(player_ids.PlayerIndex):
    """Plan 1's identity index over the NBA game logs in state."""

    @classmethod
    def build(cls, state_dir: Path, now: datetime) -> PlayerIndex:
        """Index every (player_id, player_name, team, season) in the Mac's
        ``nba/player_games`` snapshots for the last LINK_SEASONS seasons up to
        ``now``'s season. No other player list is used: a player without
        recent game-log rows stays unmatched unless an alias names him."""
        db.require_utc(now, "now")
        first_year = int(season_label(now)[:4]) - (LINK_SEASONS - 1)
        rows: set[tuple[int, str, str, str]] = set()
        for row in state.read_rows(state_dir, "nba_player_games"):
            season = str(row["season"])
            if int(season[:4]) < first_year or row.get("player_id") is None:
                continue
            rows.add((int(row["player_id"]), row["player_name"] or "", row["team"] or "",
                      season))
        return cls.from_rows(sorted(rows))


@dataclass(frozen=True)
class PlayerLink:
    espn_player_id: str
    nba_player_id: int | None
    name: str
    team: str
    method: Method
    linked_at: datetime


def _record(link: PlayerLink) -> dict:
    return {
        "espn_player_id": link.espn_player_id, "nba_player_id": link.nba_player_id,
        "name": link.name, "team": link.team, "method": link.method,
        "linked_at": link.linked_at,
    }


def read_links(state_dir: Path) -> dict[str, PlayerLink]:
    """ESPN athlete id -> its latest stored link."""
    path = state_dir / LINKS_PATH
    if not path.exists():
        return {}
    links: dict[str, PlayerLink] = {}
    for r in state.read_records(path):
        nba = r.get("nba_player_id")
        links[str(r["espn_player_id"])] = PlayerLink(
            espn_player_id=str(r["espn_player_id"]),
            nba_player_id=None if nba is None else int(nba),
            name=r["name"], team=r["team"], method=r["method"],
            linked_at=datetime.fromisoformat(r["linked_at"]),
        )
    return links


def read_aliases(state_dir: Path) -> dict:
    path = state_dir / ALIASES_PATH
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _alias_by_id(aliases: dict, espn_id: str) -> int | None:
    value = (aliases.get("by_espn_id") or {}).get(str(espn_id))
    return None if value is None else int(value)


def _alias_by_name(aliases: dict, name: str) -> int | None:
    wanted = normalize_name(name)
    for key, value in (aliases.get("by_name") or {}).items():
        if normalize_name(key) == wanted and value is not None:
            return int(value)
    return None


def link_player(
    line: EspnPlayerLine, index: player_ids.PlayerIndex, aliases: dict,
    existing: dict[str, PlayerLink], now: datetime,
) -> PlayerLink:
    """The link for one ESPN athlete (see the module docstring for the order).
    A reused link comes back with method ``existing`` and its own time."""
    db.require_utc(now, "now")
    espn_id = line.espn_player_id
    old = existing.get(espn_id)

    def made(nba_id: int | None, method: Method) -> PlayerLink:
        return PlayerLink(espn_id, nba_id, line.name, line.team, method, now)

    by_id = _alias_by_id(aliases, espn_id)
    if by_id is not None and (old is None or old.nba_player_id != by_id):
        return made(by_id, "alias")
    if old is not None and old.nba_player_id is not None:
        return replace(old, method="existing")
    by_name = _alias_by_name(aliases, line.name)
    if by_name is not None:
        return made(by_name, "alias")
    ids = index.ids_for(line.name)
    if len(ids) == 1:
        return made(next(iter(ids)), "name")
    if len(ids) > 1:
        on_team = ids & index.ids_on_team(line.name, line.team)
        if len(on_team) == 1:
            return made(next(iter(on_team)), "name+team")
    return made(None, "unmatched")


@dataclass
class LinkResult:
    links: dict[str, PlayerLink] = field(default_factory=dict)  # every line's link
    appended: list[PlayerLink] = field(default_factory=list)
    unmatched: list[PlayerLink] = field(default_factory=list)


def link_lines(
    state_dir: Path, lines: list[EspnPlayerLine], index: player_ids.PlayerIndex,
    now: datetime,
) -> LinkResult:
    """Link every line and append new or changed links to ``players/links.jsonl``.

    A link is appended when the athlete has no stored link or its NBA id
    changed; an athlete still unmatched is reported but not re-appended."""
    existing = read_links(state_dir)
    aliases = read_aliases(state_dir)
    result = LinkResult()
    for line in lines:
        if line.espn_player_id in result.links:
            continue
        link = link_player(line, index, aliases, existing, now)
        result.links[line.espn_player_id] = link
        old = existing.get(line.espn_player_id)
        if old is None or old.nba_player_id != link.nba_player_id:
            result.appended.append(link)
        if link.nba_player_id is None:
            result.unmatched.append(link)
    if result.appended:
        state.append_records(state_dir / LINKS_PATH, [_record(lk) for lk in result.appended])
    return result


def resolved_player_ids(state_dir: Path) -> dict[str, str]:
    """ESPN athlete id -> NBA player id (as a string) for every athlete that
    resolves: an alias by ESPN id first, then the latest link's NBA id, then
    (for a still-unmatched link) an alias by name. Others are absent, so
    their rows load as ``espn:<athlete id>``."""
    aliases = read_aliases(state_dir)
    out: dict[str, str] = {}
    for espn_id, link in read_links(state_dir).items():
        nba = link.nba_player_id
        if nba is None:
            nba = _alias_by_name(aliases, link.name)
        if nba is not None:
            out[espn_id] = str(nba)
    for espn_id, nba in (aliases.get("by_espn_id") or {}).items():
        if nba is not None:
            out[str(espn_id)] = str(int(nba))
    return out


# --- box-score rows -----------------------------------------------------------


def box_rows(
    game: dict, lines: list[EspnPlayerLine], links: dict[str, PlayerLink],
    observed_at: datetime,
) -> list[dict]:
    """State rows (``state.BOXSCORE_COLUMNS``) for one game's lines.

    ``game`` carries ``game_id``, ``season`` and ``game_date`` (`game_info`).
    ``observed_at`` is the summary fetch time. Every line must already have
    its link (``links``, from `link_lines`) so a stored row never lacks one;
    the rows themselves keep the ESPN athlete id only."""
    db.require_utc(observed_at)
    missing = [ln.espn_player_id for ln in lines if ln.espn_player_id not in links]
    if missing:
        raise ValueError(f"box-score lines without a player link: {', '.join(missing)}")
    return [
        {
            "game_id": game["game_id"], "season": game["season"],
            "game_date": game["game_date"], "team": ln.team,
            "espn_player_id": ln.espn_player_id, "player_name": ln.name,
            "starter": ln.starter, "did_not_play": ln.did_not_play, "minutes": ln.minutes,
            "points": ln.points, "rebounds": ln.rebounds, "assists": ln.assists,
            "plus_minus": ln.plus_minus, "source": SOURCE, "observed_at": observed_at,
        }
        for ln in lines
    ]


def apply_boxscores(con, state_dir: Path, rows: list[dict]) -> list[Path]:
    """Append box-score rows to the state directory and load them into
    ``con`` with player ids resolved from the current links. Call after
    `link_lines`. Returns the touched files."""
    if not rows:
        return []
    touched = state.append_rows(state_dir, "boxscores", rows)
    state.load_rows(con, "boxscores", rows, state.read_id_links(state_dir),
                    resolved_player_ids(state_dir))
    return touched


def game_info(con, game_id: str) -> dict:
    """``game_id``, ``season`` and ``game_date`` from the game's latest row."""
    table = db.POINT_IN_TIME_TABLES["games"]
    row = con.execute(
        f"SELECT season, game_date FROM {table} WHERE game_id = ? "
        "ORDER BY observed_at DESC LIMIT 1",
        [game_id],
    ).fetchone()
    if row is None:
        raise KeyError(f"no games row for {game_id}")
    return {"game_id": game_id, "season": row[0], "game_date": row[1]}


def boxscores_due(
    con, now: datetime, espn_links: dict[str, str],
    id_links: dict[str, str] | None = None,
) -> list[tuple[str, str]]:
    """(game_id, espn_id) of FINAL games dated slate - DUE_DAYS .. slate that
    have an ESPN event and no box-score rows yet; oldest first, at most
    MAX_SUMMARIES_PER_RUN.

    ``espn_links`` is ESPN event id -> game id (`games.read_espn_links`); a
    provisional game id maps through ``id_links`` to its NBA id, and a
    provisional ``espn:<id>`` game in the DB is its own link."""
    slate = slate_date(now)
    first: date = slate - timedelta(days=DUE_DAYS)
    id_links = id_links or {}
    event_of: dict[str, str] = {}
    for espn_id, game_id in sorted(espn_links.items()):
        event_of.setdefault(id_links.get(game_id, game_id), espn_id)
    games = db.POINT_IN_TIME_TABLES["games"]
    boxes = db.POINT_IN_TIME_TABLES["player_boxscores"]
    rows = con.execute(
        f"""
        SELECT game_id, max(game_date) AS day FROM {games} g
        WHERE status = 'FINAL' AND game_date BETWEEN ? AND ?
          AND NOT EXISTS (SELECT 1 FROM {boxes} b WHERE b.game_id = g.game_id)
        GROUP BY game_id ORDER BY day, game_id
        """,
        [first, slate],
    ).fetchall()
    due: list[tuple[str, str]] = []
    for game_id, _day in rows:
        espn_id = event_of.get(game_id)
        if espn_id is None and game_id.startswith(PROVISIONAL_PREFIX):
            espn_id = game_id.removeprefix(PROVISIONAL_PREFIX)
        if espn_id is not None:
            due.append((game_id, espn_id))
        if len(due) == MAX_SUMMARIES_PER_RUN:
            break
    return due
