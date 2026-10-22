"""Link injury-report player names to NBA player ids.

Game logs spell a player "First Last"; the injury report spells him
"Last,First" (or "Last, First"). Both are reduced to the same name key, and
the key is resolved against an identity index built from player game logs:
team plus season first, then season, then all seasons. When two different
ids share a key at the level that decides, the name is left unlinked rather
than guessed. Every lookup is counted, and unlinked names are kept per
(name, team) so the report can list them instead of dropping them silently.
"""

from __future__ import annotations

import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv"})

# Name key -> player_id for known spelling differences between the injury
# report and the game logs. Filled in from the unlinked report (Task 11).
PLAYER_ALIASES: dict[str, int] = {}


def _letters(text: str) -> str:
    return "".join(ch for ch in text if ch.isalpha())


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def _drop_suffix(tokens: list[str]) -> list[str]:
    if len(tokens) > 1 and _letters(tokens[-1]) in _SUFFIXES:
        return tokens[:-1]
    return tokens


def name_key(name: str) -> str:
    """Normalise a player name to letters-only "firstlast".

    A comma means "Last,First" (injury report); otherwise "First Last"
    (game logs). Accents are stripped, case is folded, and a trailing
    Jr./Sr./II/III/IV on the last-name part is dropped.
    """
    folded = _fold(name)
    if "," in folded:
        last, first = folded.split(",", 1)
        last_tokens = _drop_suffix(last.split())
        return _letters(first) + _letters("".join(last_tokens))
    tokens = folded.split()
    if not tokens:
        return ""
    # Last-name part is everything after the first token; a suffix is only
    # dropped when something else is left to be the last name.
    return _letters(tokens[0]) + _letters("".join(_drop_suffix(tokens[1:])))


@dataclass(frozen=True)
class LinkReport:
    linked: int
    unlinked: Counter[tuple[str, str]]


@dataclass
class PlayerIndex:
    """Identity-only index: which ids carry which name key, by team and season."""

    _by_team_season: dict[tuple[str, str, str], set[int]] = field(default_factory=dict)
    _by_season: dict[tuple[str, str], set[int]] = field(default_factory=dict)
    _by_key: dict[str, set[int]] = field(default_factory=dict)
    _linked: int = 0
    _unlinked: Counter[tuple[str, str]] = field(default_factory=Counter)

    @classmethod
    def from_rows(cls, rows: Iterable[tuple[int, str, str, str]]) -> PlayerIndex:
        by_team_season: dict[tuple[str, str, str], set[int]] = defaultdict(set)
        by_season: dict[tuple[str, str], set[int]] = defaultdict(set)
        by_key: dict[str, set[int]] = defaultdict(set)
        for player_id, player_name, team, season in rows:
            key = name_key(player_name)
            if not key:
                continue
            pid = int(player_id)
            by_team_season[(key, team, season)].add(pid)
            by_season[(key, season)].add(pid)
            by_key[key].add(pid)
        return cls(dict(by_team_season), dict(by_season), dict(by_key))

    def _resolve(self, name: str, team: str, season: str) -> int | None:
        key = name_key(name)
        if key in PLAYER_ALIASES:
            return PLAYER_ALIASES[key]
        for ids in (
            self._by_team_season.get((key, team, season)),
            self._by_season.get((key, season)),
            self._by_key.get(key),
        ):
            if not ids:
                continue
            # Ids exist at this level: it decides. Two or more -> never guess.
            return next(iter(ids)) if len(ids) == 1 else None
        return None

    def link(self, name: str, team: str, season: str) -> int | None:
        player_id = self._resolve(name, team, season)
        if player_id is None:
            self._unlinked[(name, team)] += 1
        else:
            self._linked += 1
        return player_id

    def report(self) -> LinkReport:
        return LinkReport(linked=self._linked, unlinked=Counter(self._unlinked))
