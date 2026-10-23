"""Linking injury-report names to NBA player ids (Task 3)."""

from __future__ import annotations

from predictor.model import player_ids
from predictor.model.player_ids import LinkReport, PlayerIndex, name_key


def test_name_key_handles_both_layouts():
    assert (
        name_key("Curry,Stephen")
        == name_key("Curry, Stephen")
        == name_key("Stephen Curry")
        == "stephencurry"
    )


def test_name_key_strips_suffix_and_accents():
    assert name_key("Jackson Jr.,Jaren") == name_key("Jaren Jackson Jr.") == "jarenjackson"
    assert name_key("Dončić,Luka") == "lukadoncic"


def test_link_prefers_team_season_match():
    index = PlayerIndex.from_rows(
        [
            (1, "Marcus Morris", "LAC", "2022-23"),
            (2, "Marcus Morris", "NYK", "2022-23"),
        ]
    )
    assert index.link("Morris, Marcus", "NYK", "2022-23") == 2
    assert index.report().linked == 1


def test_duplicate_name_on_same_team_is_unlinked():
    index = PlayerIndex.from_rows(
        [
            (1, "Marcus Morris", "NYK", "2022-23"),
            (2, "Marcus Morris Sr.", "NYK", "2022-23"),
        ]
    )
    assert index.link("Morris, Marcus", "NYK", "2022-23") is None
    report = index.report()
    assert report.unlinked[("Morris, Marcus", "NYK")] == 1
    assert report.linked == 0


def test_unlinked_injury_name_is_counted_not_dropped():
    index = PlayerIndex.from_rows([(1, "Stephen Curry", "GSW", "2022-23")])
    assert index.link("Nobody, Some", "GSW", "2022-23") is None
    report = index.report()
    assert isinstance(report, LinkReport)
    assert report.linked == 0
    assert report.unlinked[("Nobody, Some", "GSW")] == 1
    assert sum(report.unlinked.values()) == 1


def test_alias_wins(monkeypatch):
    monkeypatch.setattr(player_ids, "PLAYER_ALIASES", {"nene": 7})
    index = PlayerIndex.from_rows([(3, "Nene Hilario", "HOU", "2019-20")])
    assert index.link("Nene", "HOU", "2019-20") == 7
    assert index.report().linked == 1


def test_link_falls_back_to_season_then_all_seasons():
    index = PlayerIndex.from_rows(
        [
            (1, "Stephen Curry", "GSW", "2021-22"),
            (2, "Kevin Durant", "PHX", "2023-24"),
        ]
    )
    # Not on the listed team that season, but unique in the season.
    assert index.link("Durant, Kevin", "BKN", "2023-24") == 2
    # No row at all in that season: unique key across all seasons.
    assert index.link("Curry, Stephen", "GSW", "2024-25") == 1
    assert index.report().linked == 2


def test_ambiguous_across_teams_without_team_match_is_unlinked():
    index = PlayerIndex.from_rows(
        [
            (1, "Marcus Morris", "LAC", "2022-23"),
            (2, "Marcus Morris", "NYK", "2022-23"),
        ]
    )
    assert index.link("Morris, Marcus", "BOS", "2022-23") is None


def test_suffix_glued_to_the_surname_in_the_injury_report():
    from predictor.model.player_ids import name_key

    assert name_key("WilliamsIII,Robert") == name_key("Robert Williams III")
    assert name_key("PorterJr.,Michael") == name_key("Michael Porter Jr.")
    assert name_key("LivelyII,Dereck") == name_key("Dereck Lively II")
    assert name_key("HolmesII,DaRon") == name_key("DaRon Holmes II")
    # A surname that merely ends in "ii" letters is not a suffix.
    assert name_key("Kalaii,Test") == "testkalaii"
