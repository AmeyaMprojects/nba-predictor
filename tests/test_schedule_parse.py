import dataclasses
import json
from datetime import UTC, date, datetime

import pytest

from predictor.sources import schedule


def _game(
    game_id,
    date_est,
    utc,
    home="PHI",
    away="NYK",
    *,
    city="Philadelphia",
    state="PA",
    arena="Wells Fargo Center",
    neutral=False,
    status_text="7:00 pm ET",
):
    # Shaped like a real ScheduleLeagueV2 game, INCLUDING the outcome
    # fields the parser must ignore (score, wins, losses, pointsLeaders).
    return {
        "gameId": game_id,
        "gameStatus": 3,
        "gameStatusText": status_text,
        "gameDateEst": f"{date_est}T00:00:00Z",
        "gameDateTimeUTC": utc,
        "arenaName": arena,
        "arenaCity": city,
        "arenaState": state,
        "isNeutral": neutral,
        "homeTeam": {"teamTricode": home, "score": 119, "wins": 30, "losses": 10},
        "awayTeam": {"teamTricode": away, "score": 110, "wins": 20, "losses": 20},
        "pointsLeaders": [{"points": 40.0}],
    }


def _payload(games, season="2024-25"):
    return json.dumps(
        {
            "meta": {},
            "leagueSchedule": {
                "seasonYear": season,
                "gameDates": [{"gameDate": "01/15/2025 00:00:00", "games": games}],
            },
        }
    ).encode("utf-8")


def test_parses_a_regular_game():
    result = schedule.parse_schedule(
        _payload([_game("0022400561", "2025-01-15", "2025-01-16T00:00:00Z")]), "2024-25"
    )
    assert result.no_tipoff == [] and result.undetermined == []
    (row,) = result.rows
    assert row == schedule.ScheduleRow(
        game_id="0022400561",
        season="2024-25",
        game_date=date(2025, 1, 15),
        tip_off_utc=datetime(2025, 1, 16, 0, 0, tzinfo=UTC),
        home_team="PHI",
        away_team="NYK",
        arena_name="Wells Fargo Center",
        arena_city="Philadelphia",
        arena_state="PA",
        is_neutral_reported=False,
        is_neutral=False,
    )


def test_rows_carry_no_outcome_fields():
    names = {f.name for f in dataclasses.fields(schedule.ScheduleRow)}
    for forbidden in ("score", "points", "status", "wins", "losses", "leader"):
        assert not any(forbidden in n for n in names), forbidden


def test_tbd_time_is_no_tipoff_not_a_midnight_placeholder():
    result = schedule.parse_schedule(
        _payload([
            _game("0022400561", "2025-01-15", "2025-01-15T05:00:00Z", status_text="TBD")
        ]),
        "2024-25",
    )
    assert result.rows[0].tip_off_utc is None
    assert result.no_tipoff == ["0022400561"]


def test_tipoff_on_a_different_eastern_date_is_rejected():
    # 2025-01-17T00:00Z is 7pm ET on 01-16, not the listed 01-15.
    result = schedule.parse_schedule(
        _payload([_game("0022400561", "2025-01-15", "2025-01-17T00:00:00Z")]), "2024-25"
    )
    assert result.rows[0].tip_off_utc is None
    assert result.no_tipoff == ["0022400561"]


def test_game_with_undecided_teams_is_left_out_and_reported():
    result = schedule.parse_schedule(
        _payload([
            _game("0062400001", "2024-12-17", "2024-12-17T05:00:00Z",
                  home=None, away=None, status_text="TBD"),
            _game("0022400561", "2025-01-15", "2025-01-16T00:00:00Z"),
        ]),
        "2024-25",
    )
    assert [r.game_id for r in result.rows] == ["0022400561"]
    assert result.undetermined == ["0062400001"]


def test_reported_neutral_flag_is_kept():
    result = schedule.parse_schedule(
        _payload([
            _game("0022400621", "2025-01-23", "2025-01-23T19:00:00Z", home="IND",
                  away="SAS", city="Paris", state="", arena="Accor Arena", neutral=True)
        ]),
        "2024-25",
    )
    row = result.rows[0]
    assert row.is_neutral_reported is True and row.is_neutral is True
    assert row.arena_state is None  # empty string becomes None


def test_game_away_from_the_home_teams_usual_arena_is_derived_neutral():
    # Before 2024-25 the league's flag is false even in Paris.
    games = [
        _game("0022200001", "2022-10-20", "2022-10-20T23:00:00Z"),
        _game("0022200002", "2022-10-22", "2022-10-22T23:00:00Z"),
        _game("0022200678", "2023-01-19", "2023-01-19T19:00:00Z", city="Paris",
              state="", arena="Accor Arena"),
    ]
    result = schedule.parse_schedule(_payload(games, season="2022-23"), "2022-23")
    by_id = {r.game_id: r for r in result.rows}
    assert by_id["0022200678"].is_neutral is True
    assert by_id["0022200678"].is_neutral_reported is False
    assert by_id["0022200001"].is_neutral is False


def test_postseason_bubble_game_is_derived_neutral():
    games = [
        _game("0021900001", "2019-10-22", "2019-10-23T02:00:00Z", home="LAL",
              away="LAC", city="Los Angeles", state="CA", arena="Staples Center"),
        _game("0041900101", "2020-08-18", "2020-08-19T01:00:00Z", home="LAL",
              away="POR", city="Orlando", state="FL", arena="ESPN Wide World"),
    ]
    result = schedule.parse_schedule(_payload(games, season="2019-20"), "2019-20")
    by_id = {r.game_id: r for r in result.rows}
    assert by_id["0041900101"].is_neutral is True
    assert by_id["0021900001"].is_neutral is False


def test_wrong_season_in_payload_is_an_error():
    with pytest.raises(ValueError, match="2023-24"):
        schedule.parse_schedule(_payload([], season="2023-24"), "2024-25")


@pytest.mark.parametrize("bad", [b"not json", b"{}", b'{"leagueSchedule": {"seasonYear": "2024-25"}}'])
def test_malformed_payload_is_a_plain_error(bad):
    with pytest.raises(ValueError, match="expected shape"):
        schedule.parse_schedule(bad, "2024-25")
