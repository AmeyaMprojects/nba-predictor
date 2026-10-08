"""Team names -> canonical 3-letter abbreviations, shared by every source.

The one copy of this table: the injury report, the live odds feed and the
historical odds file all map their team text through `team_abbr`. The
abbreviations are the same ones the schedule and games tables use
(nba_api's), so the result is directly a join key.
"""

from __future__ import annotations

# Cached lazily (not at import time) so a test that never touches this path
# never pays for the nba_api static-data load, and so a monkeypatch of
# `nba_api.stats.static.teams` in a test still takes effect.
_TEAM_ABBR_BY_NAME: dict[str, str] | None = None

# The official NBA_API full_name for the LA Clippers is "Los Angeles
# Clippers", but the injury report itself (both layouts, every era sampled)
# renders it as "LA Clippers" / "LAClippers" -- never the full "Los
# Angeles" form. Without this alias, every Clippers row would fail to
# resolve to an abbreviation at all.
_TEAM_NAME_ALIASES: dict[str, str] = {
    "LACLIPPERS": "LAC",
}


def _team_abbreviations() -> dict[str, str]:
    global _TEAM_ABBR_BY_NAME
    if _TEAM_ABBR_BY_NAME is None:
        from nba_api.stats.static import teams as nba_teams

        by_name: dict[str, str] = {}
        for team in nba_teams.get_teams():
            # A current abbreviation maps to itself, so text that is
            # already canonical ("PHX") passes straight through.
            by_name[team["abbreviation"].upper()] = team["abbreviation"]
            by_name[team["full_name"].replace(" ", "").upper()] = team["abbreviation"]
        by_name.update(_TEAM_NAME_ALIASES)
        _TEAM_ABBR_BY_NAME = by_name
    return _TEAM_ABBR_BY_NAME


def team_abbr(team_name: str | None) -> str | None:
    """Canonical 3-letter abbreviation for a team's name, or None.

    Tolerates spacing and case differences ("Miami Heat", "MiamiHeat",
    "miami heat") by normalizing both away before lookup, and accepts a
    current abbreviation itself. Returns None for a name the official
    nba_api roster (plus the LA Clippers alias) does not recognize --
    callers must treat that as "cannot canonicalize", never guess.
    """
    if not team_name:
        return None
    key = team_name.replace(" ", "").upper()
    return _team_abbreviations().get(key)
