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
    status_name=None,
):
    def competitor(side, abbr, score):
        return {
            "id": abbr, "homeAway": side, "score": score,
            "team": {"abbreviation": abbr, "displayName": _NAMES.get(abbr, abbr)},
        }

    status = {"type": {"completed": completed, "state": "post" if completed else "pre",
                       "name": status_name
                       or ("STATUS_FINAL" if completed else "STATUS_SCHEDULED")}}
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


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(espn, "_sleep", lambda s: None)


def test_fetch_scoreboard_day_asks_for_one_date(no_sleep):
    session = FakeSession([FakeResponse(200, b'{"events":[1]}')])
    out = espn.fetch_scoreboard_day(date(2026, 10, 20), session=session)
    assert out == b'{"events":[1]}'
    assert session.calls == [(espn.SCOREBOARD_URL, {"dates": "20261020", "limit": 1000}, 20)]


def test_fetch_scoreboard_day_retries_twice_on_5xx_and_network(no_sleep):
    import requests

    session = FakeSession([requests.ConnectionError("x"), FakeResponse(503), FakeResponse(200)])
    espn.fetch_scoreboard_day(date(2026, 10, 20), session=session)
    assert len(session.calls) == 3
    session = FakeSession([FakeResponse(500)] * 3)
    with pytest.raises(espn.EspnFetchError):
        espn.fetch_scoreboard_day(date(2026, 10, 20), session=session)
    assert len(session.calls) == 3


def test_4xx_is_an_error_without_retry(no_sleep):
    for status in (400, 404):
        session = FakeSession([FakeResponse(status)])
        with pytest.raises(espn.EspnFetchError):
            espn.fetch_scoreboard_day(date(2026, 10, 20), session=session)
        assert len(session.calls) == 1


def test_archive_stores_raw_bytes_under_kind_ident_time():
    key = espn.archive("scoreboard", b"{}", dt("2026-10-22T15:04:05Z"), "20261020")
    assert key == "scoreboard_20261020_20261022T150405Z.json"
    assert raw_store.load(espn.RAW_SOURCE, key) == b"{}"


def test_download_archives_each_day_exact_bytes_and_parses_from_the_archive(no_sleep):
    # Odd spacing on purpose: the archive must hold ESPN's bytes untouched.
    day1 = b'{ "events" : [' + json.dumps(espn_event("1")).encode() + b"] }"
    day2 = json.dumps({"events": [espn_event("2", when="2026-10-24T00:00Z"),
                                  espn_event("1")]}, indent=3).encode()
    session = FakeSession([FakeResponse(200, day1), FakeResponse(200, day2)])
    fetched = dt("2026-10-22T15:00:00Z")
    keys = espn.download_scoreboards([date(2026, 10, 22), date(2026, 10, 23)], fetched,
                                     session=session)
    assert keys == ["scoreboard_20261022_20261022T150000Z.json",
                    "scoreboard_20261023_20261022T150000Z.json"]
    assert raw_store.load(espn.RAW_SOURCE, keys[0]) == day1
    assert raw_store.load(espn.RAW_SOURCE, keys[1]) == day2
    assert [g.espn_id for g in espn.load_scoreboards(keys)] == ["1", "2"]


def test_download_stops_on_a_failed_day_but_keeps_earlier_archives(no_sleep):
    session = FakeSession([FakeResponse(200, b'{"events":[]}'), FakeResponse(404)])
    fetched = dt("2026-10-22T16:00:00Z")
    with pytest.raises(espn.EspnFetchError) as err:
        espn.download_scoreboards([date(2026, 10, 22), date(2026, 10, 23)], fetched,
                                  session=session)
    assert "20261023" in str(err.value)
    assert raw_store.exists(espn.RAW_SOURCE, "scoreboard_20261022_20261022T160000Z.json")


# 15:00 UTC Oct 22 -> slate date Oct 22 (ET of now - 7h).
NOW = dt("2026-10-22T15:00Z")


def test_near_days_every_run():
    assert games.scoreboard_days(NOW, include_far=False) == [
        date(2026, 10, 21), date(2026, 10, 22), date(2026, 10, 23),
    ]


def test_far_days_complete_the_window_minus_two_to_plus_seven():
    days = games.scoreboard_days(NOW, include_far=True)
    assert days == [date(2026, 10, 20) + timedelta(days=n) for n in range(10)]
    assert len(days) <= espn.MAX_SCOREBOARD_REQUESTS


def test_far_due_after_sixty_minutes():
    assert games.far_due(None, NOW) is True
    assert games.far_due(NOW - timedelta(minutes=59, seconds=59), NOW) is False
    assert games.far_due(NOW - timedelta(minutes=60), NOW) is True


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


# --- fix round 2: reschedules, postponements, duplicate links, malformed ------


def test_parse_reads_status_name():
    assert parse_one().status_name == "STATUS_SCHEDULED"
    assert parse_one(status_name="STATUS_POSTPONED").status_name == "STATUS_POSTPONED"


def test_rescheduled_to_tbd_on_a_new_date(con):
    add_schedule(con, "0022600010", date(2026, 10, 22), "GSW", "LAL", tip="2026-10-23T02:00Z")
    moved = parse_one(when="2026-11-05T05:00Z", time_valid=False)
    (row,) = games.sync_games(con, [moved], FETCHED, {"401705555": "0022600010"}).schedule_rows
    assert row["game_date"] == date(2026, 11, 5) and row["tip_off_utc"] is None


@pytest.mark.parametrize("status_name", ["STATUS_POSTPONED", "STATUS_CANCELED",
                                         "STATUS_SUSPENDED", "STATUS_CANCELLED"])
def test_postponed_or_cancelled_game_becomes_tbd_and_is_reported(con, status_name):
    add_schedule(con, "0022600010", date(2026, 10, 22), "GSW", "LAL", tip="2026-10-22T23:30Z")
    off = parse_one(when="2026-10-22T23:30Z", status_name=status_name, completed=True,
                    home_score="0", away_score="0")
    sync = games.sync_games(con, [off], dt("2026-10-23T05:00Z"), {"401705555": "0022600010"})
    (row,) = sync.schedule_rows
    assert row["tip_off_utc"] is None and row["game_date"] == date(2026, 10, 22)
    assert sync.finals == []
    assert sync.postponed == [f"LAL@GSW 2026-10-22 ({status_name})"]
    assert sync.skipped == []


def test_postponed_provisional_game_without_known_row(con):
    off = parse_one(status_name="STATUS_POSTPONED")
    sync = games.sync_games(con, [off], FETCHED, {})
    assert sync.schedule_rows[0]["tip_off_utc"] is None
    assert len(sync.postponed) == 1


def test_back_to_back_does_not_claim_a_known_nba_id(con):
    d = date(2026, 12, 1)
    add_schedule(con, "0022600301", d, "LAL", "SAC", tip="2026-12-02T03:00Z")
    nxt = parse_one(event_id="12", home="LAL", away="SAC", when="2026-12-03T03:00Z")
    sync = games.sync_games(con, [nxt], FETCHED, {"11": "0022600301"})
    assert sync.espn_links[0]["game_id"] == "espn:12"


def test_back_to_back_does_not_claim_an_exact_match_of_this_run(con):
    d = date(2026, 12, 1)
    add_schedule(con, "0022600301", d, "LAL", "SAC", tip="2026-12-02T03:00Z")
    # The D+1 event comes first, so order cannot decide it.
    nxt = parse_one(event_id="12", home="LAL", away="SAC", when="2026-12-03T03:00Z")
    first = parse_one(event_id="11", home="LAL", away="SAC", when="2026-12-02T03:00Z")
    sync = games.sync_games(con, [nxt, first], FETCHED, {})
    assert {r["espn_id"]: r["game_id"] for r in sync.espn_links} == {
        "11": "0022600301", "12": "espn:12",
    }


def test_one_day_fallback_still_links_an_unclaimed_game(con):
    add_schedule(con, "0022600301", date(2026, 12, 1), "LAL", "SAC")
    shifted = parse_one(event_id="12", home="LAL", away="SAC", when="2026-12-03T03:00Z")
    sync = games.sync_games(con, [shifted], FETCHED, {})
    assert sync.espn_links[0]["game_id"] == "0022600301"


def test_malformed_event_is_skipped_and_the_rest_of_the_day_kept():
    bad = espn_event("9", home="BOS", away="NY")
    del bad["season"]["type"]
    problems: list[str] = []
    parsed = espn.parse_scoreboard(body(bad, espn_event("1")), problems)
    assert [g.espn_id for g in parsed] == ["1"]
    (problem,) = problems
    assert "malformed event 9" in problem and "type" in problem


def test_malformed_events_reach_the_sync_skipped_list(con):
    sync = games.sync_games(con, [parse_one()], FETCHED, {},
                            malformed=["NY @ BOS 2026-10-22 (malformed event 9: KeyError 'type')"])
    assert sync.skipped == ["NY @ BOS 2026-10-22 (malformed event 9: KeyError 'type')"]
    assert len(sync.schedule_rows) == 1


def test_load_scoreboards_collects_malformed_events():
    bad = espn_event("9")
    del bad["season"]
    key = espn.archive("scoreboard", body(bad, espn_event("1")), dt("2026-10-22T17:00Z"),
                       "20261022")
    problems: list[str] = []
    assert [g.espn_id for g in espn.load_scoreboards([key], problems)] == ["1"]
    assert len(problems) == 1


@pytest.mark.parametrize(("when", "et_day"), [
    ("2026-11-01T02:30Z", date(2026, 10, 31)),  # Sat 10:30 pm EDT, night DST ends
    ("2026-11-02T03:30Z", date(2026, 11, 1)),   # Sun 10:30 pm EST, first Sunday of Nov
    ("2027-03-14T03:30Z", date(2027, 3, 13)),   # Sat 10:30 pm EST, night DST starts
    ("2027-03-15T02:30Z", date(2027, 3, 14)),   # Sun 10:30 pm EDT, second Sunday of March
])
def test_late_games_around_dst_changes_keep_their_et_date(when, et_day):
    g = parse_one(when=when)
    assert g.game_date == et_day
    assert g.tip_off_utc == dt(when)
    assert g.tip_off_utc.astimezone(espn.EASTERN).strftime("%H:%M") == "22:30"


def test_provisional_game_at_a_neutral_site(con):
    g = parse_one(neutral=True, venue=("Arena CDMX", "Mexico City", None))
    (row,) = games.sync_games(con, [g], FETCHED, {}).schedule_rows
    assert row["game_id"] == "espn:401705555"
    assert row["is_neutral"] is True and row["is_neutral_reported"] is True
    assert (row["arena_name"], row["arena_city"], row["arena_state"]) == (
        "Arena CDMX", "Mexico City", None)


def test_tip_change_to_a_new_time_writes_a_new_row(con):
    add_schedule(con, "0022600010", date(2026, 10, 22), "GSW", "LAL", tip="2026-10-23T02:00Z")
    later = parse_one(when="2026-10-23T03:00Z")
    (row,) = games.sync_games(con, [later], FETCHED, {"401705555": "0022600010"}).schedule_rows
    assert row["tip_off_utc"] == dt("2026-10-23T03:00Z")
    assert row["game_date"] == date(2026, 10, 22) and row["arena_name"] == "NBA Arena"
