"""Point-in-time player features for every game, at the early and final call.

One chronological pass, in the order of ``fit._simulate``: on each date D the
features of every game on D are computed first, and only then are D's player
rows applied. Player logs are stamped D+1 12:00 UTC, so this sees exactly what
an ``AsOfView`` at either call time on D sees. Every read of ratings, minutes
and rosters for a game on D passes ``as_of=D``, so the books raise if a game
on or after D was applied first.

Two running books:

- ``PlayerBook`` (Task 4): ratings, fed only competitive rows
  (``player_games.rating_rows_sql``; never preseason).
- ``Rosters`` (here): who is on which team, from every row including
  preseason (controller ruling: preseason counts for roster membership
  only), plus each player's recent minutes from competitive rows only.

Per game and call:

- Rotation of team T: players whose last team is T and who appear in one of
  T's last 10 games. A player listed in the injury snapshot under a team U
  other than his last team leaves his old team's rotation at once; he joins
  U's only when his recent minutes include minutes for U (Review Focus 1).
- Normal minutes: ``normalize_minutes({p: mean(recent minutes)})``.
- Play probability: ``q = table[season, call].prob(status)``, the table
  estimated on earlier seasons' listings only. Unlinked names count as
  available (q = 1) and are counted per season.
- ``roster_gap[f] = S_normal(home) - S_normal(away)`` and
  ``injury_gap[f] = (S_avail - S_normal)(home) - (S_avail - S_normal)(away)``.
- The snapshot is the latest report at or before the call that lists the
  game's date. It is a fallback when it is more than an hour older than the
  call, or when a later report (at or before the call) did not list the date.
  With no report at all the injury gap is 0 and the call is "no report".
- Games with no tip-off get both calls at 23:00 UTC on the game date.

``GameFeatures.extras`` (for the optional correction, Task 8), all
point-in-time like the rest, NaN when a team has no history yet:

- ``recent_margin_gap``: home minus away of each team's mean margin over its
  last 10 games. The correction subtracts the team-rating gap from it to get
  ``recent_form_gap`` (the rating depends on the fold's team settings, which
  this pass does not know).
- ``pace_sum``: both teams' mean possessions over their last 10 games, where
  a game's possessions are the sum over its competitive player rows of
  FGA + 0.44 FTA - OREB + TOV.
- ``continuity_gap`` / ``continuity_gap_early``: home minus away of the
  share of available expected minutes (final / early call) from players who
  played in at least one of the team's last 5 competitive games (preseason
  lineups never count here; they only set roster membership).
"""

from __future__ import annotations

import dataclasses
import math
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta

from predictor import db
from predictor.backtest import tipoff as tipoff_mod
from predictor.model import minutes as minutes_mod
from predictor.model.availability import (
    CALLS,
    AvailabilityTable,
    InjuryHistory,
    Listing,
    Snapshot,
    UnknownStatusCounter,
    estimate,
)
from predictor.model.fit import FitError, _Game
from predictor.model.player_ids import LinkReport, PlayerIndex
from predictor.model.player_ratings import (
    FORMULAS,
    MIN_MINUTES,
    RECENT_GAMES,
    PlayerBook,
    PlayerParams,
)
from predictor.model.settings import TEST_SEASONS
from predictor.sources import injury_slots
from predictor.sources.player_games import RATING_SEASON_TYPES, PlayerGameRow

NO_TIP_CALL_UTC = time(23, 0)
FALLBACK_AFTER = timedelta(hours=1)
FORM_GAMES = 10
CONTINUITY_GAMES = 5
EXTRA_KEYS = ("recent_margin_gap", "pace_sum", "continuity_gap", "continuity_gap_early")

_STAT_FIELDS = (
    "pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "ast",
    "stl", "blk", "tov", "pf",
)


# -- outputs ---------------------------------------------------------------

@dataclass(frozen=True)
class CallFeatures:
    roster_gap: dict[str, float]
    injury_gap: dict[str, float]
    absent: tuple[tuple[str, str, str], ...]
    report_at: datetime | None
    fallback: bool
    no_report: bool


@dataclass(frozen=True)
class GameFeatures:
    game_id: str
    final: CallFeatures
    early: CallFeatures
    extras: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SimStats:
    no_tip: int
    fallback: dict[str, Counter[str]]  # call -> season -> games
    no_report: dict[str, Counter[str]]  # call -> season -> games
    unlinked: LinkReport  # all seasons
    unknown_status: Counter[str]  # all seasons
    unlinked_by_season: dict[str, LinkReport] = field(default_factory=dict)
    unknown_by_season: dict[str, dict[str, int]] = field(default_factory=dict)


# -- player lines ----------------------------------------------------------

def _num(value):
    """None for NULL/NaN, else the value."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _int(value) -> int | None:
    value = _num(value)
    return None if value is None else int(value)


def _float(value) -> float | None:
    value = _num(value)
    return None if value is None else float(value)


def _played(row: PlayerGameRow) -> bool:
    return row.minutes is not None and row.minutes >= MIN_MINUTES


DayRows = list[tuple[PlayerGameRow, bool]]  # (row, feeds ratings and minutes)


class PlayerLines:
    """Every player game row (latest per game and player), by date.

    Each row carries whether it may feed ratings and minutes (competitive
    season types) or only rosters (preseason). Test seasons are excluded.
    ``tip_index`` is the schedule tip-off index (``tipoff.tipoff_index``) the
    replay derives call times from.
    """

    def __init__(
        self,
        rows: Iterable[tuple[PlayerGameRow, str]],
        tip_index: dict[tuple[date, str], datetime] | None = None,
    ) -> None:
        by_date: dict[date, DayRows] = defaultdict(list)
        played: dict[str, set[int]] = defaultdict(set)
        names: dict[int, tuple[date, str]] = {}
        for row, season_type in rows:
            if row.season in TEST_SEASONS:
                continue
            by_date[row.game_date].append((row, season_type in RATING_SEASON_TYPES))
            if _played(row):
                played[row.game_id].add(row.player_id)
            seen = names.get(row.player_id)
            if seen is None or row.game_date >= seen[0]:
                names[row.player_id] = (row.game_date, row.player_name)
        self.by_date: dict[date, DayRows] = {
            d: sorted(day, key=lambda r: (r[0].game_id, r[0].player_id))
            for d, day in sorted(by_date.items())
        }
        self.played: dict[str, frozenset[int]] = {g: frozenset(p) for g, p in played.items()}
        self.names: dict[int, str] = {p: n for p, (_, n) in names.items()}
        self.tip_index: dict[tuple[date, str], datetime] = dict(tip_index or {})

    @classmethod
    def from_db(cls, con) -> PlayerLines:
        """The latest row per (game_id, player_id), excluding test seasons,
        plus the schedule's tip-off index."""
        table = db.POINT_IN_TIME_TABLES["player_games"]
        tests = ", ".join("?" for _ in TEST_SEASONS)
        cols = ", ".join(_STAT_FIELDS)
        rows = con.execute(
            f"SELECT game_id, season, game_date, player_id, player_name, team, minutes, "
            f"{cols}, plus_minus, season_type FROM {table} "
            f"WHERE season NOT IN ({tests}) "
            "QUALIFY row_number() OVER (PARTITION BY game_id, player_id "
            "ORDER BY observed_at DESC) = 1 "
            "ORDER BY game_date, game_id, player_id",
            list(TEST_SEASONS),
        ).fetchall()
        parsed = []
        for gid, season, gd, pid, name, team, mins, *rest in rows:
            stats, pm, season_type = rest[:-2], rest[-2], rest[-1]
            parsed.append((
                PlayerGameRow(
                    gid, season, gd, int(pid), name, team, _float(mins),
                    *(_int(v) for v in stats), _float(pm),
                ),
                season_type,
            ))
        return cls(parsed, tipoff_mod.tipoff_index(con))


# -- rosters ---------------------------------------------------------------

class Rosters:
    """Team membership (every row, preseason included) and recent minutes
    (competitive rows only). Reads take ``as_of`` like ``PlayerBook``."""

    def __init__(self) -> None:
        self._last_team: dict[int, str] = {}
        self._team_games: dict[str, deque[frozenset[int]]] = {}
        # Competitive games only (never preseason), for continuity.
        self._competitive_games: dict[str, deque[frozenset[int]]] = {}
        self._minutes: dict[int, deque[tuple[str, float]]] = {}
        self._last_date: date | None = None

    def apply_day(self, day: DayRows) -> None:
        if not day:
            return
        game_date = max(row.game_date for row, _ in day)
        if self._last_date is not None and game_date < self._last_date:
            raise ValueError(f"days must be applied in order: {game_date} after {self._last_date}")
        self._last_date = game_date
        by_game: dict[str, list[tuple[PlayerGameRow, bool]]] = defaultdict(list)
        for row, rates in day:
            by_game[row.game_id].append((row, rates))
        for gid in sorted(by_game):
            lineups: dict[str, set[int]] = defaultdict(set)
            competitive: set[str] = set()
            for row, rates in by_game[gid]:
                if not _played(row):
                    continue
                self._last_team[row.player_id] = row.team
                lineups[row.team].add(row.player_id)
                if rates:
                    competitive.add(row.team)
                if rates:
                    self._minutes.setdefault(
                        row.player_id, deque(maxlen=RECENT_GAMES)
                    ).append((row.team, float(row.minutes)))  # type: ignore[arg-type]
            for team, players in sorted(lineups.items()):
                self._team_games.setdefault(
                    team, deque(maxlen=minutes_mod.ROTATION_GAMES)
                ).append(frozenset(players))
                if team in competitive:
                    self._competitive_games.setdefault(
                        team, deque(maxlen=CONTINUITY_GAMES)
                    ).append(frozenset(players))

    def _check(self, as_of: date) -> None:
        if self._last_date is not None and self._last_date >= as_of:
            raise ValueError(f"read as of {as_of} after a game dated {self._last_date} was applied")

    def rotation(self, team: str, as_of: date) -> set[int]:
        self._check(as_of)
        seen: set[int] = set()
        for players in self._team_games.get(team, ()):
            seen |= players
        return {p for p in seen if self._last_team.get(p) == team}

    def last_team(self, player_id: int, as_of: date) -> str | None:
        self._check(as_of)
        return self._last_team.get(player_id)

    def recent_minutes(self, player_id: int, as_of: date) -> list[float]:
        self._check(as_of)
        return [m for _, m in self._minutes.get(player_id, ())]

    def has_minutes_for(self, player_id: int, team: str, as_of: date) -> bool:
        self._check(as_of)
        return any(t == team for t, _ in self._minutes.get(player_id, ()))

    def recent_players(self, team: str, games: int, as_of: date) -> set[int]:
        """Players who played in at least one of ``team``'s last ``games``
        competitive games (preseason never counts; at most CONTINUITY_GAMES)."""
        self._check(as_of)
        if games > CONTINUITY_GAMES:
            raise ValueError(f"only the last {CONTINUITY_GAMES} games are kept, not {games}")
        out: set[int] = set()
        for players in list(self._competitive_games.get(team, ()))[-games:]:
            out |= players
        return out


def _possessions(row: PlayerGameRow) -> float | None:
    if row.fga is None or row.fta is None or row.oreb is None or row.tov is None:
        return None
    return row.fga + 0.44 * row.fta - row.oreb + row.tov


class TeamForm:
    """Each team's last ``FORM_GAMES`` margins (from final scores) and
    possessions (from competitive player rows). Reads take ``as_of``."""

    def __init__(self) -> None:
        self._margins: dict[str, deque[float]] = {}
        self._poss: dict[str, deque[float]] = {}
        self._last_date: date | None = None

    def apply_day(self, day: date, games: Sequence[_Game], rows: DayRows) -> None:
        if self._last_date is not None and day < self._last_date:
            raise ValueError(f"days must be applied in order: {day} after {self._last_date}")
        self._last_date = day
        for g in sorted(games, key=lambda g: g.result.game_id):
            r = g.result
            for team, m in ((r.home_team, r.home_points - r.away_points),
                            (r.away_team, r.away_points - r.home_points)):
                self._margins.setdefault(team, deque(maxlen=FORM_GAMES)).append(float(m))
        poss: dict[tuple[str, str], float] = defaultdict(float)
        for row, rates in rows:
            if not rates:
                continue
            p = _possessions(row)
            if p is not None:
                poss[(row.game_id, row.team)] += p
        for (_, team), p in sorted(poss.items()):
            self._poss.setdefault(team, deque(maxlen=FORM_GAMES)).append(p)

    def _check(self, as_of: date) -> None:
        if self._last_date is not None and self._last_date >= as_of:
            raise ValueError(f"read as of {as_of} after a game dated {self._last_date} was applied")

    @staticmethod
    def _mean(values) -> float | None:
        return sum(values) / len(values) if values else None

    def margin(self, team: str, as_of: date) -> float | None:
        self._check(as_of)
        return self._mean(self._margins.get(team, ()))

    def pace(self, team: str, as_of: date) -> float | None:
        self._check(as_of)
        return self._mean(self._poss.get(team, ()))


def _diff(a: float | None, b: float | None) -> float:
    return math.nan if a is None or b is None else a - b


def _total(a: float | None, b: float | None) -> float:
    return math.nan if a is None or b is None else a + b


# -- linking ---------------------------------------------------------------

class _Linker:
    """Resolves report names without touching the caller's index counters.

    Resolution goes through a silent per-season copy (memoised); counting
    goes through a separate per-season copy whose ``report()`` is the
    season's unlinked list.
    """

    def __init__(self, index: PlayerIndex) -> None:
        self._index = index
        self._silent: dict[str, PlayerIndex] = {}
        self._memo: dict[tuple[str, str, str], int | None] = {}
        self.counting: dict[str, PlayerIndex] = {}

    def _copy(self) -> PlayerIndex:
        return dataclasses.replace(self._index, _linked=0, _unlinked=Counter())

    def resolve(self, name: str, team: str, season: str) -> int | None:
        key = (name, team, season)
        if key not in self._memo:
            if season not in self._silent:
                self._silent[season] = self._copy()
            self._memo[key] = self._silent[season].link(name, team, season)
        return self._memo[key]

    def count(self, name: str, team: str, season: str) -> None:
        if season not in self.counting:
            self.counting[season] = self._copy()
        self.counting[season].link(name, team, season)


# -- one game, one call ----------------------------------------------------

@dataclass(frozen=True)
class _Lineup:
    """A team's rotation for one call: normal minutes and listed statuses."""

    team: str
    normal: dict[int, float]
    status: dict[int, str]


def _snapshot_by_player(
    snap: Snapshot | None, season: str, linker: _Linker
) -> dict[int, list[tuple[str, str]]]:
    """Linked snapshot entries: player id -> [(team, status)]."""
    out: dict[int, list[tuple[str, str]]] = defaultdict(list)
    if snap is None:
        return out
    for (team, name), status in sorted(snap.statuses.items()):
        pid = linker.resolve(name, team, season)
        if pid is not None:
            out[pid].append((team, status))
    return out


def _lineups(
    teams: Sequence[str],
    listed: dict[int, list[tuple[str, str]]],
    rosters: Rosters,
    as_of: date,
) -> list[_Lineup]:
    out = []
    for team in teams:
        members = rosters.rotation(team, as_of=as_of)
        for pid, entries in listed.items():
            entry_teams = {t for t, _ in entries}
            last = rosters.last_team(pid, as_of=as_of)
            if pid in members and last not in entry_teams:
                members.discard(pid)  # listed only by another team: traded away
            if (
                team in entry_teams
                and pid not in members
                and last != team
                and rosters.has_minutes_for(pid, team, as_of=as_of)
            ):
                members.add(pid)
        raw: dict[int, float] = {}
        for pid in sorted(members):
            mins = rosters.recent_minutes(pid, as_of=as_of)
            if mins:
                raw[pid] = sum(mins) / len(mins)
        normal = minutes_mod.normalize_minutes(raw)
        status = {
            pid: s for pid in normal for t, s in listed.get(pid, ()) if t == team
        }
        out.append(_Lineup(team, normal, status))
    return out


def _call_times(game: _Game, tip_index) -> tuple[datetime, datetime] | None:
    r = game.result
    tip = tipoff_mod.resolve_tipoff(tip_index, r.game_date, r.home_team, r.away_team)
    if tip is None:
        return None
    return injury_slots.call_times(r.game_date, tip.astimezone(UTC))


def _game_calls(game: _Game, tip_index) -> tuple[dict[str, datetime], bool]:
    times = _call_times(game, tip_index)
    if times is None:
        at = datetime.combine(game.result.game_date, NO_TIP_CALL_UTC, UTC)
        return {"early": at, "final": at}, False
    return {"early": times[0], "final": times[1]}, True


def availability_tables(listings: Iterable[Listing], season: str) -> dict[str, AvailabilityTable]:
    """Per-call availability tables for ``season``, from earlier seasons only."""
    listings = list(listings)
    return {call: estimate(listings, season, call) for call in CALLS}


# -- the pass --------------------------------------------------------------

def _replay(
    games: Sequence[_Game],
    lines: PlayerLines,
    visit: Callable[[_Game, Rosters], None],
    book: PlayerBook | None = None,
    after_day: Callable[[date, list[_Game], DayRows], None] | None = None,
) -> None:
    """Visit every game on D (before D's rows), then apply D's rows."""
    for g in games:
        if g.result.season in TEST_SEASONS:
            raise FitError(
                f"internal error: the player simulation was given a test-season game "
                f"({g.result.game_id}, season {g.result.season}); refusing to use it"
            )
    by_date: dict[date, list[_Game]] = defaultdict(list)
    for g in games:
        by_date[g.result.game_date].append(g)
    game_dates = {g.result.game_id: g.result.game_date for g in games}
    # A row dated before its game would be applied before that game's
    # features are computed -- a leak. Refuse any date mismatch.
    for rows in lines.by_date.values():
        for row, _ in rows:
            expected = game_dates.get(row.game_id)
            if expected is not None and expected != row.game_date:
                raise FitError(
                    f"player rows for game {row.game_id} are dated {row.game_date} "
                    f"but the game is dated {expected}; refusing to simulate"
                )
    margins_by_game = {
        g.result.game_id: {
            g.result.home_team: g.result.home_points - g.result.away_points,
            g.result.away_team: g.result.away_points - g.result.home_points,
        }
        for g in games
    }
    rosters = Rosters()
    for day in sorted(set(by_date) | set(lines.by_date)):
        for g in by_date.get(day, ()):
            visit(g, rosters)
        rows = lines.by_date.get(day, [])
        rosters.apply_day(rows)
        if after_day is not None:
            after_day(day, by_date.get(day, []), rows)
        if book is not None:
            rated: dict[str, list[PlayerGameRow]] = defaultdict(list)
            for row, rates in rows:
                if rates:
                    rated[row.game_id].append(row)
            for gid in sorted(rated):
                book.apply_game(rated[gid], margins_by_game.get(gid, {}))


def collect_listings(
    con, games: list[_Game], history: InjuryHistory, index: PlayerIndex
) -> list[Listing]:
    """Every linked rotation player in each call's snapshot, and whether he
    then played (a row for that game with at least one minute)."""
    lines = PlayerLines.from_db(con)
    linker = _Linker(index)
    out: list[Listing] = []

    def visit(g: _Game, rosters: Rosters) -> None:
        r = g.result
        calls, _ = _game_calls(g, lines.tip_index)
        played = lines.played.get(r.game_id, frozenset())
        for call in CALLS:
            snap = history.snapshot_at(r.game_date, calls[call])
            listed = _snapshot_by_player(snap, r.season, linker)
            for lineup in _lineups((r.home_team, r.away_team), listed, rosters, r.game_date):
                for pid, status in sorted(lineup.status.items()):
                    out.append(Listing(r.season, call, status, pid in played))

    _replay(games, lines, visit)
    return out


def simulate(
    games: list[_Game],
    lines: PlayerLines,
    history: InjuryHistory,
    index: PlayerIndex,
    params: PlayerParams,
    listings: list[Listing],
) -> tuple[dict[str, GameFeatures], SimStats]:
    book = PlayerBook(params)
    linker = _Linker(index)
    unknown = UnknownStatusCounter()
    tables: dict[str, dict[str, AvailabilityTable]] = {}
    fallback: dict[str, Counter[str]] = {c: Counter() for c in CALLS}
    no_report: dict[str, Counter[str]] = {c: Counter() for c in CALLS}
    no_tip = 0
    out: dict[str, GameFeatures] = {}
    form = TeamForm()
    continuity: dict[str, list[float | None]] = {}

    def call_features(g: _Game, call: str, at: datetime, rosters: Rosters,
                      seen_names: set[tuple[str, str]]) -> CallFeatures:
        r = g.result
        d = r.game_date
        teams = (r.home_team, r.away_team)
        snap = history.snapshot_at(d, at)
        if snap is not None:
            for (team, name) in snap.statuses:
                if team in teams:
                    seen_names.add((name, team))
        listed = _snapshot_by_player(snap, r.season, linker)
        if r.season not in tables:
            tables[r.season] = availability_tables(listings, r.season)
        table = tables[r.season][call]
        roster_gap = {f: 0.0 for f in FORMULAS}
        injury_gap = {f: 0.0 for f in FORMULAS}
        absent: list[tuple[int, float, str, str, str]] = []
        shares: list[float | None] = []
        for side, lineup in enumerate(_lineups(teams, listed, rosters, d)):
            sign = 1.0 if side == 0 else -1.0
            q: dict[int, float] = {}
            for pid, status in lineup.status.items():
                unknown.record(r.season, status)
                q[pid] = table.prob(status)
            avail = minutes_mod.available_minutes(lineup.normal, q)
            recent = rosters.recent_players(lineup.team, CONTINUITY_GAMES, as_of=d)
            total = sum(avail.values())
            shares.append(
                sum(m for p, m in avail.items() if p in recent) / total if total > 0 else None
            )
            for f in FORMULAS:
                ratings = {p: book.rating(p, f, as_of=d) for p in lineup.normal}
                s_normal = minutes_mod.strength(lineup.normal, ratings)
                s_avail = minutes_mod.strength(avail, ratings)
                roster_gap[f] += sign * s_normal
                if snap is not None:
                    injury_gap[f] += sign * (s_avail - s_normal)
            for pid, p in q.items():
                if p < 1.0:
                    name = lines.names.get(pid, str(pid))
                    absent.append((side, -lineup.normal[pid], name, lineup.team,
                                   lineup.status[pid]))
        continuity[call] = shares
        if snap is None:
            no_report[call][r.season] += 1
            return CallFeatures(roster_gap, injury_gap, (), None, False, True)
        report_at = snap.observed_at
        latest = history.latest_report_at(at)
        fell_back = report_at < at - FALLBACK_AFTER or (latest is not None and latest > report_at)
        if fell_back:
            fallback[call][r.season] += 1
        absent.sort()
        return CallFeatures(
            roster_gap,
            injury_gap,
            tuple((team, name, status) for _, _, name, team, status in absent),
            report_at,
            fell_back,
            False,
        )

    def visit(g: _Game, rosters: Rosters) -> None:
        nonlocal no_tip
        r = g.result
        # Like fit._simulate: the season is entered before its first read.
        book.enter_season(r.season)
        calls, has_tip = _game_calls(g, lines.tip_index)
        if not has_tip:
            no_tip += 1
        seen_names: set[tuple[str, str]] = set()
        feats = {
            call: call_features(g, call, calls[call], rosters, seen_names) for call in CALLS
        }
        # Unlinked names are counted once per game (whichever calls list them).
        for name, team in sorted(seen_names):
            linker.count(name, team, r.season)
        home, away = r.home_team, r.away_team
        extras = {
            "recent_margin_gap": _diff(form.margin(home, as_of=r.game_date),
                                       form.margin(away, as_of=r.game_date)),
            "pace_sum": _total(form.pace(home, as_of=r.game_date),
                               form.pace(away, as_of=r.game_date)),
            "continuity_gap": _diff(*continuity["final"]),
            "continuity_gap_early": _diff(*continuity["early"]),
        }
        out[r.game_id] = GameFeatures(r.game_id, feats["final"], feats["early"], extras)

    _replay(games, lines, visit, book, form.apply_day)

    by_season = {s: idx.report() for s, idx in sorted(linker.counting.items())}
    total_unlinked: Counter[tuple[str, str]] = Counter()
    for rep in by_season.values():
        total_unlinked.update(rep.unlinked)
    unknown_report = unknown.report()
    unknown_total: Counter[str] = Counter()
    for counts in unknown_report.values():
        unknown_total.update(counts)
    stats = SimStats(
        no_tip=no_tip,
        fallback=fallback,
        no_report=no_report,
        unlinked=LinkReport(
            linked=sum(rep.linked for rep in by_season.values()), unlinked=total_unlinked
        ),
        unknown_status=unknown_total,
        unlinked_by_season=by_season,
        unknown_by_season=unknown_report,
    )
    return out, stats
