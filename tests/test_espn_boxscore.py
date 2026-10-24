"""ESPN game summary -> box-score lines (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from predictor.sources import espn

FIXTURE = Path(__file__).parent / "fixtures" / "espn" / "summary_sample.json"

KEYS = [
    "minutes", "points", "fieldGoalsMade-fieldGoalsAttempted",
    "threePointFieldGoalsMade-threePointFieldGoalsAttempted",
    "freeThrowsMade-freeThrowsAttempted", "rebounds", "assists", "turnovers", "steals",
    "blocks", "offensiveRebounds", "defensiveRebounds", "fouls", "plusMinus",
]


def athlete(pid, name, stats, starter=False, dnp=False):
    return {
        "active": False, "starter": starter, "didNotPlay": dnp, "reason": "",
        "ejected": False, "stats": stats,
        "athlete": {"id": pid, "displayName": name, "shortName": name},
    }


def team_block(abbr, display, athletes, keys=KEYS):
    return {
        "team": {"abbreviation": abbr, "displayName": display},
        "statistics": [{"keys": list(keys), "labels": ["?"] * len(keys),
                        "athletes": athletes, "totals": []}],
    }


def summary(*blocks) -> bytes:
    return json.dumps({"boxscore": {"players": list(blocks)}}).encode()


STARTER = athlete(
    "3945274", "Luka Dončić",
    ["34", "27", "10-20", "3-8", "4-5", "8", "9", "3", "1", "0", "1", "7", "2", "+12"],
    starter=True,
)
BENCH = athlete(
    "4277905", "Jaren Jackson Jr.",
    ["21", "9", "4-9", "1-3", "0-0", "5", "1", "1", "0", "2", "2", "3", "4", "-3"],
)
DNP = athlete("5142621", "Chris Cenac Jr.", [], dnp=True)


def test_parse_boxscore_reads_by_keys():
    # Keys in a shuffled order: values must be read by key, not position.
    order = list(reversed(KEYS))
    stats = dict(zip(KEYS, STARTER["stats"], strict=True))
    shuffled = athlete("3945274", "Luka Dončić", [stats[k] for k in order], starter=True)
    (line,) = espn.parse_boxscore(summary(team_block("LAL", "Los Angeles Lakers",
                                                     [shuffled], keys=order)))
    assert line == espn.EspnPlayerLine(
        espn_player_id="3945274", name="Luka Dončić", team="LAL", starter=True,
        did_not_play=False, minutes=34.0, points=27, rebounds=8, assists=9, plus_minus=12,
    )


def test_minus_and_zero_plus_minus():
    zero = athlete("1", "Zero Guy",
                   ["13", "0", "0-2", "0-1", "0-0", "6", "1", "0", "1", "1", "3", "3", "0", "0"])
    lines = espn.parse_boxscore(summary(team_block("MEM", "Memphis Grizzlies", [BENCH, zero])))
    assert [ln.plus_minus for ln in lines] == [-3, 0]
    assert lines[0].starter is False and lines[0].team == "MEM"


def test_dnp_player():
    (line,) = espn.parse_boxscore(summary(team_block("BOS", "Boston Celtics", [DNP])))
    assert line.did_not_play is True
    assert line.minutes is None
    assert line.points is None and line.plus_minus is None


def test_empty_stats_without_dnp_flag_counts_as_not_played():
    odd = athlete("9", "No Stats", [], dnp=False)
    (line,) = espn.parse_boxscore(summary(team_block("BOS", "Boston Celtics", [odd])))
    assert line.did_not_play is True and line.minutes is None


def test_minutes_in_clock_form_and_dashes():
    clock = athlete("2", "Clock", ["31:30", "--", "", "", "", "", "", "", "", "", "", "", "", "+1"])
    (line,) = espn.parse_boxscore(summary(team_block("BOS", "Boston Celtics", [clock])))
    assert line.minutes == 31.5
    assert line.points is None


def test_espn_team_abbreviations_map_to_nba():
    (line,) = espn.parse_boxscore(summary(team_block("NY", "New York Knicks", [BENCH])))
    assert line.team == "NYK"


def test_missing_stat_key_raises():
    keys = [k for k in KEYS if k != "plusMinus"]
    block = team_block("LAL", "Los Angeles Lakers",
                       [athlete("1", "A", ["1"] * len(keys))], keys=keys)
    with pytest.raises(espn.EspnFormatError) as err:
        espn.parse_boxscore(summary(block))
    assert "plusMinus" in str(err.value)


def test_unknown_team_raises():
    with pytest.raises(espn.EspnFormatError, match="team"):
        espn.parse_boxscore(summary(team_block("LEB", "Team LeBron", [BENCH])))


def test_no_boxscore_raises_and_no_players_gives_nothing():
    with pytest.raises(espn.EspnFormatError, match="boxscore"):
        espn.parse_boxscore(b'{"header": {}}')
    assert espn.parse_boxscore(b'{"boxscore": {"teams": []}}') == []


def test_real_summary_sample_parses():
    lines = espn.parse_boxscore(FIXTURE.read_bytes())
    by_name = {ln.name: ln for ln in lines}
    assert len(lines) == 28
    assert {ln.team for ln in lines} == {"BOS", "DET"}
    george = by_name["Paul George"]
    assert george == espn.EspnPlayerLine(
        espn_player_id="4251", name="Paul George", team="BOS", starter=True,
        did_not_play=False, minutes=31.0, points=38, rebounds=5, assists=3, plus_minus=9,
    )
    assert by_name["Ausar Thompson"].plus_minus == 12
    assert by_name["Jayson Tatum"].plus_minus == -5
    assert by_name["Jordan Walsh"].plus_minus == 0
    conley = by_name["Mike Conley"]
    assert conley.did_not_play and conley.minutes is None and not conley.starter
    assert sum(ln.starter for ln in lines) == 10
    assert sum(ln.did_not_play for ln in lines) == 6
    assert sum(ln.points or 0 for ln in lines if ln.team == "DET") == 121


# --- fetch --------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status=200, content=b'{"boxscore": {}}'):
        self.status_code = status
        self.content = content


class FakeSession:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params, timeout))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(espn, "_sleep", lambda s: None)


def test_fetch_summary_asks_for_the_event_and_returns_exact_bytes(no_sleep):
    raw = b'{ "boxscore" : {"players": []} }'
    session = FakeSession(FakeResponse(content=raw))
    assert espn.fetch_summary("401909088", session=session) == raw
    assert session.calls == [(espn.SUMMARY_URL, {"event": "401909088"}, 20)]
    assert espn.SUMMARY_URL.endswith("/basketball/nba/summary")


def test_fetch_summary_retries_then_fails(no_sleep):
    session = FakeSession(FakeResponse(503), requests.ConnectionError("x"), FakeResponse(500))
    with pytest.raises(espn.EspnFetchError, match="401909088"):
        espn.fetch_summary("401909088", session=session)
    assert len(session.calls) == 3


def test_fetch_summary_4xx_is_an_error_without_retry(no_sleep):
    session = FakeSession(FakeResponse(404))
    with pytest.raises(espn.EspnFetchError, match="404"):
        espn.fetch_summary("1", session=session)
    assert len(session.calls) == 1
