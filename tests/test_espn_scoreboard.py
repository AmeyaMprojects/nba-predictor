"""ESPN scoreboard -> schedule rows, finals and game links (no network)."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from predictor import db, raw_store
from predictor.cloud import games, state
from predictor.sources import espn
from predictor.teams import espn_team_abbr

FIXTURE = Path(__file__).parent / "fixtures" / "espn" / "scoreboard_sample.json"


def dt(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)


_NAMES = {
    "GS": "Golden State Warriors", "LAL": "Los Angeles Lakers", "SAC": "Sacramento Kings",
    "BOS": "Boston Celtics", "NY": "New York Knicks", "LAC": "LA Clippers",
    "UTAH": "Utah Jazz", "LEB": "Team LeBron", "DUR": "Team Durant", "MIA": "Miami Heat",
}


def espn_event(
    event_id="401705555", when="2026-10-23T02:00Z", home="GS", away="LAL",
    home_score="0", away_score="0", completed=False, season_year=2027, season_type=2,
    time_valid=True, neutral=False, venue=("Chase Center", "San Francisco", "CA"),
):
    def competitor(side, abbr, score):
        return {
            "id": abbr, "homeAway": side, "score": score,
            "team": {"abbreviation": abbr, "displayName": _NAMES.get(abbr, abbr)},
        }

    status = {"type": {"completed": completed, "state": "post" if completed else "pre",
                       "name": "STATUS_FINAL" if completed else "STATUS_SCHEDULED"}}
    return {
        "id": event_id,
        "date": when,
        "season": {"year": season_year, "type": season_type},
        "status": status,
        "competitions": [{
            "id": event_id, "date": when, "timeValid": time_valid, "neutralSite": neutral,
            "venue": {"fullName": venue[0], "address": {"city": venue[1], "state": venue[2]}},
            "competitors": [
                competitor("home", home, home_score), competitor("away", away, away_score),
            ],
            "status": status,
        }],
    }


def body(*events) -> bytes:
    return json.dumps({"events": list(events)}).encode()


def parse_one(**kw) -> espn.EspnGame:
    (game,) = espn.parse_scoreboard(body(espn_event(**kw)))
    return game


@pytest.fixture
def con(tmp_path):
    c = db.connect(tmp_path / "t.duckdb")
    db.migrate(c)
    yield c
    c.close()


def add_schedule(con, game_id, day, home, away, tip=None, observed="2026-10-01T12:00Z",
                 arena=("NBA Arena", "NBA City", "NC"), neutral=False, season="2026-27"):
    table = db.POINT_IN_TIME_TABLES["schedule"]
    con.execute(
        f"INSERT INTO {table} VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [game_id, season, day, dt(tip) if tip else None, home, away, *arena,
         neutral, neutral, dt(observed)],
    )


def add_final(con, game_id, day, home, away, hp=100, ap=90, observed="2026-10-23T12:00Z"):
    table = db.POINT_IN_TIME_TABLES["games"]
    con.execute(
        f"INSERT INTO {table} VALUES (?,?,?,?,?,?,?,?,?,?)",
        [game_id, "2026-27", day, home, away, hp, ap, "FINAL", True, dt(observed)],
    )


FETCHED = dt("2026-10-22T15:00Z")


# --- parse ------------------------------------------------------------------


def test_parse_scoreboard_reads_teams_tip_and_score():
    g = parse_one(when="2026-10-23T02:00Z", home="GS", away="LAL",
                  home_score="118", away_score="112", completed=True)
    assert g.game_date == date(2026, 10, 22)
    assert g.season == "2026-27"
    assert g.season_type == 2
    assert g.home_team == "GSW" and g.away_team == "LAL"
    assert g.home_name == "Golden State Warriors"
    assert g.home_points == 118 and g.away_points == 112
    assert g.completed is True
    assert g.tip_off_utc == dt("2026-10-23T02:00Z")
    assert g.venue_name == "Chase Center" and g.venue_city == "San Francisco"
    assert g.venue_state == "CA" and g.neutral is False


def test_west_coast_late_tip_keeps_et_date():
    assert parse_one(when="2026-10-23T02:30Z").game_date == date(2026, 10, 22)


def test_time_not_valid_gives_no_tip():
    g = parse_one(when="2026-10-23T04:00Z", time_valid=False)
    assert g.tip_off_utc is None
    assert g.game_date == date(2026, 10, 23)


def test_points_are_none_until_completed():
    g = parse_one(home_score="0", away_score="0", completed=False)
    assert g.home_points is None and g.away_points is None


def test_score_as_object_is_read():
    event = espn_event(completed=True)
    for c in event["competitions"][0]["competitors"]:
        c["score"] = {"value": 101.0, "displayValue": "101"}
    (g,) = espn.parse_scoreboard(body(event))
    assert g.home_points == 101 and g.away_points == 101


def test_espn_abbreviations_map():
    assert espn_team_abbr("GS", "Golden State Warriors") == "GSW"
    assert espn_team_abbr("UTAH", "Utah Jazz") == "UTA"
    assert espn_team_abbr("LAC", "LA Clippers") == "LAC"
    assert espn_team_abbr("NY", "") == "NYK"
    assert espn_team_abbr("LEB", "Team LeBron") is None
    for short, nba in (("SA", "SAS"), ("NO", "NOP"), ("WSH", "WAS"), ("PHX", "PHX")):
        assert espn_team_abbr(short, None) == nba
    assert espn_team_abbr(None, None) is None


def test_real_scoreboard_sample_parses():
    parsed = {g.espn_id: g for g in espn.parse_scoreboard(FIXTURE.read_bytes())}
    assert len(parsed) == 6
    det = parsed["401909088"]
    assert (det.home_team, det.away_team) == ("DET", "BOS")
    assert (det.home_points, det.away_points) == (121, 113)
    assert det.completed and det.season == "2026-27" and det.season_type == 2
    assert det.venue_name == "Little Caesars Arena" and det.venue_state == "MI"
    nyk = parsed["401909089"]
    assert (nyk.home_team, nyk.away_team, nyk.home_points) == ("NYK", "PHI", 100)
    # 01:50Z on Oct 21 is the evening of Oct 20 in ET.
    sas = parsed["401909090"]
    assert sas.home_team == "SAS" and sas.game_date == date(2026, 10, 20)
    assert sas.tip_off_utc == dt("2026-10-21T01:50Z")
    uta = parsed["401909861"]
    assert uta.home_team == "UTA" and not uta.completed and uta.home_points is None
    assert parsed["401909865"].away_team == "LAC"
    sac = parsed["401909867"]
    assert sac.home_team == "SAC" and sac.game_date == date(2026, 10, 25)


def test_per_day_fallback_body_parses():
    payload = {"days": {"20261022": {"events": [espn_event("1")]},
                        "20261023": {"events": [espn_event("2", when="2026-10-24T00:00Z")]}}}
    parsed = espn.parse_scoreboard(json.dumps(payload).encode())
    assert [g.espn_id for g in parsed] == ["1", "2"]


# --- fetch and archive --------------------------------------------------------


class FakeResponse:
    def __init__(self, status, content=b'{"events":[]}'):
        self.status_code = status
        self.content = content


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {}), timeout))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_fetch_scoreboard_asks_for_the_date_range(monkeypatch):
    monkeypatch.setattr(espn, "_sleep", lambda s: None)
    session = FakeSession([FakeResponse(200, b'{"events":[1]}')])
    out = espn.fetch_scoreboard(date(2026, 10, 20), date(2026, 10, 29), session=session)
    assert out == b'{"events":[1]}'
    assert session.calls == [
        (espn.SCOREBOARD_URL, {"dates": "20261020-20261029", "limit": 1000}, 20)
    ]


def test_fetch_scoreboard_retries_twice_on_5xx_and_network(monkeypatch):
    import requests

    monkeypatch.setattr(espn, "_sleep", lambda s: None)
    session = FakeSession([requests.ConnectionError("x"), FakeResponse(503), FakeResponse(200)])
    espn.fetch_scoreboard(date(2026, 10, 20), date(2026, 10, 20), session=session)
    assert len(session.calls) == 3
    session = FakeSession([FakeResponse(500)] * 3)
    with pytest.raises(espn.EspnFetchError):
        espn.fetch_scoreboard(date(2026, 10, 20), date(2026, 10, 20), session=session)
    assert len(session.calls) == 3


def test_range_refused_falls_back_to_one_request_per_day(monkeypatch):
    monkeypatch.setattr(espn, "_sleep", lambda s: None)
    session = FakeSession([
        FakeResponse(400, b'{"code":400}'),
        FakeResponse(200, json.dumps({"events": [espn_event("1")]}).encode()),
        FakeResponse(200, json.dumps({"events": [espn_event("2")]}).encode()),
    ])
    out = espn.fetch_scoreboard(date(2026, 10, 20), date(2026, 10, 21), session=session)
    assert [c[1]["dates"] for c in session.calls] == ["20261020-20261021", "20261020", "20261021"]
    assert sorted(json.loads(out)["days"]) == ["20261020", "20261021"]
    assert [g.espn_id for g in espn.parse_scoreboard(out)] == ["1", "2"]


def test_other_4xx_is_an_error(monkeypatch):
    monkeypatch.setattr(espn, "_sleep", lambda s: None)
    session = FakeSession([FakeResponse(404)])
    with pytest.raises(espn.EspnFetchError):
        espn.fetch_scoreboard(date(2026, 10, 20), date(2026, 10, 20), session=session)
    assert len(session.calls) == 1


def test_archive_stores_raw_bytes_under_kind_ident_time():
    key = espn.archive("scoreboard", b"{}", dt("2026-10-22T15:04:05Z"), "20261020-20261029")
    assert key == "scoreboard_20261020-20261029_20261022T150405Z.json"
    assert raw_store.load(espn.RAW_SOURCE, key) == b"{}"


def test_scoreboard_window_is_slate_minus_two_to_plus_seven():
    # 15:00 UTC Oct 22 -> slate date Oct 22 (ET of now - 7h).
    assert games.scoreboard_window(dt("2026-10-22T15:00Z")) == (
        date(2026, 10, 20), date(2026, 10, 29)
    )


# --- sync_games -------------------------------------------------------------


def test_all_star_and_preseason_are_skipped_and_reported(con):
    star = parse_one(event_id="1", home="LEB", away="DUR", when="2027-02-15T01:00Z")
    pre = parse_one(event_id="2", season_type=1, home="BOS", away="NY")
    sync = games.sync_games(con, [star, pre], FETCHED, {})
    assert len(sync.skipped) == 2
    assert sync.schedule_rows == [] and sync.finals == []
    assert sync.espn_links == [] and sync.id_links == []
    assert any("Team Durant@Team LeBron 2027-02-14" in s for s in sync.skipped)
    assert any(s.startswith("NYK@BOS ") and "season type 1" in s for s in sync.skipped)


def test_links_to_nba_id_by_et_date_and_teams(con):
    add_schedule(con, "0022600010", date(2026, 10, 22), "GSW", "LAL", tip="2026-10-23T02:00Z")
    sync = games.sync_games(con, [parse_one()], FETCHED, {})
    assert sync.espn_links == [
        {"espn_id": "401705555", "game_id": "0022600010", "linked_at": FETCHED.isoformat()}
    ]
    assert sync.id_links == []
    assert sync.schedule_rows == []  # nothing differs from the NBA row


def test_unlinked_competitive_game_gets_provisional_id(con):
    sync = games.sync_games(con, [parse_one(season_type=3)], FETCHED, {})
    assert [r["game_id"] for r in sync.schedule_rows] == ["espn:401705555"]
    row = sync.schedule_rows[0]
    assert row["arena_name"] == "Chase Center" and row["arena_city"] == "San Francisco"
    assert row["is_neutral"] is False and row["is_neutral_reported"] is False
    assert row["observed_at"] == FETCHED and row["season"] == "2026-27"
    assert sync.espn_links[0]["game_id"] == "espn:401705555"


def test_provisional_reconciled_when_nba_schedule_appears(con):
    add_schedule(con, "espn:401705555", date(2026, 10, 22), "GSW", "LAL",
                 tip="2026-10-23T02:00Z", observed="2026-10-20T12:00Z")
    add_schedule(con, "0042600101", date(2026, 10, 22), "GSW", "LAL",
                 tip="2026-10-23T02:00Z", observed="2026-10-21T12:00Z")
    sync = games.sync_games(con, [parse_one()], FETCHED, {"401705555": "espn:401705555"})
    assert sync.id_links == [{"provisional_id": "espn:401705555", "game_id": "0042600101",
                              "linked_at": FETCHED.isoformat()}]
    assert sync.espn_links == [{"espn_id": "401705555", "game_id": "0042600101",
                                "linked_at": FETCHED.isoformat()}]


def test_known_nba_link_is_used_without_a_new_link_row(con):
    add_schedule(con, "0022600010", date(2026, 10, 22), "GSW", "LAL", tip="2026-10-23T02:00Z")
    sync = games.sync_games(con, [parse_one()], FETCHED, {"401705555": "0022600010"})
    assert sync.espn_links == [] and sync.schedule_rows == []


def test_two_game_series_links_each_date_exactly(con):
    d = date(2026, 12, 1)
    add_schedule(con, "0022600301", d, "LAL", "SAC", tip="2026-12-02T03:00Z")
    add_schedule(con, "0022600302", d + timedelta(days=1), "LAL", "SAC",
                 tip="2026-12-03T03:00Z")
    first = parse_one(event_id="11", home="LAL", away="SAC", when="2026-12-02T03:00Z")
    second = parse_one(event_id="12", home="LAL", away="SAC", when="2026-12-03T03:00Z")
    sync = games.sync_games(con, [first, second], FETCHED, {})
    assert {r["espn_id"]: r["game_id"] for r in sync.espn_links} == {
        "11": "0022600301", "12": "0022600302",
    }


def test_schedule_row_only_on_change(con, tmp_path):
    event = parse_one()
    first = games.sync_games(con, [event], FETCHED, {})
    assert len(first.schedule_rows) == 1
    games.apply_sync(con, tmp_path / "state", first)
    known = games.read_espn_links(tmp_path / "state")
    again = games.sync_games(con, [event], FETCHED + timedelta(hours=1), known)
    assert again.schedule_rows == [] and again.espn_links == []
    moved = parse_one(when="2026-10-23T02:30Z")
    later = games.sync_games(con, [moved], FETCHED + timedelta(hours=2), known)
    assert [r["tip_off_utc"] for r in later.schedule_rows] == [dt("2026-10-23T02:30Z")]


def test_tbd_never_erases_known_tip(con):
    add_schedule(con, "0022600010", date(2026, 10, 22), "GSW", "LAL", tip="2026-10-23T02:00Z")
    tbd = parse_one(when="2026-10-22T04:00Z", time_valid=False)
    assert tbd.game_date == date(2026, 10, 22)
    sync = games.sync_games(con, [tbd], FETCHED, {"401705555": "0022600010"})
    assert sync.schedule_rows == []
    neutral = parse_one(when="2026-10-22T04:00Z", time_valid=False, neutral=True)
    sync = games.sync_games(con, [neutral], FETCHED, {"401705555": "espn:401705555"})
    # Linked games take neutrality from the NBA row, so still no change.
    assert sync.schedule_rows == []


def test_tbd_keeps_known_tip_on_a_provisional_game(con):
    add_schedule(con, "espn:401705555", date(2026, 10, 22), "GSW", "LAL",
                 tip="2026-10-23T02:00Z")
    tbd = parse_one(when="2026-10-22T04:00Z", time_valid=False, neutral=True)
    sync = games.sync_games(con, [tbd], FETCHED, {"401705555": "espn:401705555"})
    (row,) = sync.schedule_rows
    assert row["tip_off_utc"] == dt("2026-10-23T02:00Z") and row["is_neutral"] is True


def test_final_only_after_finished_after_tip(con):
    done = parse_one(when="2026-10-22T23:30Z", home_score="118", away_score="112",
                     completed=True)
    early = games.sync_games(con, [done], dt("2026-10-23T02:59Z"), {})
    assert early.finals == []
    ok = games.sync_games(con, [done], dt("2026-10-23T03:00Z"), {})
    assert ok.finals == [{
        "game_id": "espn:401705555", "season": "2026-27", "game_date": date(2026, 10, 22),
        "home_team": "GSW", "away_team": "LAL", "home_points": 118, "away_points": 112,
        "status": "FINAL", "reconstructed": False, "observed_at": dt("2026-10-23T03:00Z"),
        "source": "espn",
    }]


def test_not_completed_gives_no_final(con):
    live = parse_one(when="2026-10-22T23:30Z", home_score="50", away_score="40")
    assert games.sync_games(con, [live], dt("2026-10-23T05:00Z"), {}).finals == []


def test_existing_final_not_duplicated(con):
    add_schedule(con, "0022600010", date(2026, 10, 22), "GSW", "LAL", tip="2026-10-22T23:30Z")
    add_final(con, "0022600010", date(2026, 10, 22), "GSW", "LAL")
    done = parse_one(when="2026-10-22T23:30Z", home_score="118", away_score="112",
                     completed=True)
    sync = games.sync_games(con, [done], dt("2026-10-23T05:00Z"), {})
    assert sync.finals == []


def test_final_under_provisional_id_counts_after_linking(con):
    add_schedule(con, "0022600010", date(2026, 10, 22), "GSW", "LAL", tip="2026-10-22T23:30Z")
    add_final(con, "espn:401705555", date(2026, 10, 22), "GSW", "LAL")
    done = parse_one(when="2026-10-22T23:30Z", home_score="118", away_score="112",
                     completed=True)
    sync = games.sync_games(con, [done], dt("2026-10-23T05:00Z"),
                            {"401705555": "espn:401705555"})
    assert sync.finals == [] and len(sync.id_links) == 1


def test_linked_game_keeps_nba_arena_fields(con):
    add_schedule(con, "0022600010", date(2026, 10, 22), "GSW", "LAL",
                 tip="2026-10-23T02:00Z", arena=("Mexico Arena", "Mexico City", None),
                 neutral=True)
    moved = parse_one(when="2026-10-23T03:00Z", neutral=False)
    (row,) = games.sync_games(con, [moved], FETCHED, {}).schedule_rows
    assert row["game_id"] == "0022600010"
    assert row["arena_name"] == "Mexico Arena" and row["arena_city"] == "Mexico City"
    assert row["arena_state"] is None
    assert row["is_neutral"] is True and row["is_neutral_reported"] is True
    assert row["tip_off_utc"] == dt("2026-10-23T03:00Z")
    assert row["game_date"] == date(2026, 10, 22)


def test_apply_sync_writes_state_with_headers_and_loads_db(con, tmp_path):
    sdir = tmp_path / "state"
    add_schedule(con, "0042600101", date(2026, 10, 22), "GSW", "LAL",
                 tip="2026-10-22T23:30Z")
    done = parse_one(when="2026-10-22T23:30Z", home_score="118", away_score="112",
                     completed=True)
    sync = games.sync_games(con, [done], dt("2026-10-23T05:00Z"),
                            {"401705555": "espn:401705555"})
    games.apply_sync(con, sdir, sync)
    links = (sdir / games.ESPN_LINKS_PATH).read_text().splitlines()
    assert json.loads(links[0]) == {"format": "lines", "v": 1}
    assert state.read_id_links(sdir) == {"espn:401705555": "0042600101"}
    assert games.read_espn_links(sdir) == {"401705555": "0042600101"}
    results = list(state.read_rows(sdir, "results"))
    assert results[0]["source"] == "espn" and results[0]["game_id"] == "0042600101"
    table = db.POINT_IN_TIME_TABLES["games"]
    assert con.execute(
        f"SELECT game_id, home_points, status FROM {table}"
    ).fetchall() == [("0042600101", 118, "FINAL")]
