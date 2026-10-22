"""Point-in-time player ratings (Task 4)."""

from __future__ import annotations

from datetime import date

import pytest

from predictor.model.player_ratings import (
    HALF_LIFE_GRID,
    OBS_CLIP,
    REGRESSION_GRID,
    ROOKIE_PRIOR,
    PlayerBook,
    PlayerParams,
    ewma_update,
    game_score,
)
from predictor.sources.player_games import PlayerGameRow

PARAMS = PlayerParams(half_life_minutes=1500.0, season_regression=0.4)


def _row(
    player_id: int = 1,
    team: str = "BOS",
    minutes: float | None = 30.0,
    *,
    game_id: str = "0022400001",
    season: str = "2024-25",
    game_date: date = date(2024, 11, 1),
    pts: int = 30,
    fgm: int = 10,
    fga: int = 20,
    ftm: int = 8,
    fta: int = 10,
    oreb: int = 2,
    dreb: int = 8,
    ast: int = 5,
    stl: int = 2,
    blk: int = 1,
    pf: int = 3,
    tov: int = 4,
    plus_minus: float | None = 10.0,
) -> PlayerGameRow:
    return PlayerGameRow(
        game_id=game_id,
        season=season,
        game_date=game_date,
        player_id=player_id,
        player_name=f"Player {player_id}",
        team=team,
        minutes=minutes,
        pts=pts,
        fgm=fgm,
        fga=fga,
        fg3m=0,
        fg3a=0,
        ftm=ftm,
        fta=fta,
        oreb=oreb,
        dreb=dreb,
        ast=ast,
        stl=stl,
        blk=blk,
        tov=tov,
        pf=pf,
        plus_minus=plus_minus,
    )


def test_constants():
    assert HALF_LIFE_GRID == (500.0, 1500.0, 4000.0)
    assert REGRESSION_GRID == (0.2, 0.4)
    assert ROOKIE_PRIOR == -2.0
    assert OBS_CLIP == 60.0


def test_game_score_hollinger():
    assert game_score(_row()) == pytest.approx(24.0)


def test_ewma_update():
    assert ewma_update(0.0, 10.0, 1500.0, 1500.0) == pytest.approx(5.0)
    assert ewma_update(4.0, 10.0, 0.0, 1500.0) == 4.0
    assert ewma_update(0.0, 100.0, 1500.0, 1500.0) == pytest.approx(30.0)
    assert ewma_update(0.0, -100.0, 1500.0, 1500.0) == pytest.approx(-30.0)


def test_unseen_player_has_rookie_prior():
    book = PlayerBook(PARAMS)
    assert book.rating(99, "box") == ROOKIE_PRIOR
    assert book.rating(99, "pm") == ROOKIE_PRIOR
    assert book.recent_minutes(99) == []
    assert book.last_team(99) is None
    assert book.team_recent_games("BOS") == []


def test_first_game_observations():
    # First game ever: league_box_mean is 0 before it.
    # box obs = 48*24/30 = 38.4; pm obs = 48*10/30 - 5 = 11.0.
    # lambda = 0.5 ** (30/1500).
    book = PlayerBook(PARAMS)
    book.apply_game([_row()], {"BOS": 5})
    lam = 0.5 ** (30.0 / 1500.0)
    assert book.rating(1, "box") == pytest.approx(lam * -2.0 + (1 - lam) * 38.4)
    assert book.rating(1, "pm") == pytest.approx(lam * -2.0 + (1 - lam) * 11.0)


def test_league_box_mean_uses_only_earlier_games():
    book = PlayerBook(PARAMS)
    # Game 1: player 1, 30 min, gs 24 -> 48*24/30 = 38.4 per 48.
    book.apply_game([_row()], {"BOS": 5})
    # Game 2: player 2, 24 min, same box (gs 24) -> 48 per 48.
    # Mean before game 2 is 38.4 (game 1 only), so obs = 48 - 38.4 = 9.6.
    book.apply_game(
        [_row(player_id=2, minutes=24.0, game_id="0022400002", game_date=date(2024, 11, 2))],
        {"BOS": 0},
    )
    lam = 0.5 ** (24.0 / 1500.0)
    assert book.rating(2, "box") == pytest.approx(lam * -2.0 + (1 - lam) * 9.6)


def test_season_regression():
    book = PlayerBook(PARAMS)
    book.enter_season("2024-25")
    book._ratings[1] = {"box": 5.0, "pm": 5.0}
    book.enter_season("2025-26")
    assert book.rating(1, "box") == pytest.approx(3.0)
    assert book.rating(1, "pm") == pytest.approx(3.0)
    book.enter_season("2025-26")
    assert book.rating(1, "box") == pytest.approx(3.0)


def test_apply_game_enters_the_rows_season():
    book = PlayerBook(PARAMS)
    book.apply_game([_row()], {"BOS": 5})
    before = book.rating(1, "box")
    book.enter_season("2024-25")  # already seen: no change
    assert book.rating(1, "box") == before
    book.apply_game(
        [
            _row(
                player_id=2,
                season="2025-26",
                game_id="0022500001",
                game_date=date(2025, 10, 25),
            )
        ],
        {"BOS": 0},
    )
    assert book.rating(1, "box") == pytest.approx(0.6 * before)


def test_rating_follows_player_across_trade():
    book = PlayerBook(PARAMS)
    book.apply_game([_row(team="BOS")], {"BOS": 5})
    bos_box = book.rating(1, "box")
    bos_pm = book.rating(1, "pm")
    book.apply_game(
        [_row(team="NYK", game_id="0022400002", game_date=date(2024, 11, 3))],
        {"NYK": -2},
    )
    lam = 0.5 ** (30.0 / 1500.0)
    # League mean before game 2 is 38.4, so box obs = 38.4 - 38.4 = 0.
    assert book.rating(1, "box") == pytest.approx(ewma_update(bos_box, 0.0, 30.0, 1500.0))
    assert book.rating(1, "box") == pytest.approx(lam * bos_box)
    # pm obs = 48*10/30 - (-2) = 18.
    assert book.rating(1, "pm") == pytest.approx(ewma_update(bos_pm, 18.0, 30.0, 1500.0))
    assert book.last_team(1) == "NYK"
    assert book.recent_minutes(1) == [30.0, 30.0]


def test_zero_minute_row_ignored():
    book = PlayerBook(PARAMS)
    book.apply_game([_row()], {"BOS": 5})
    box, pm = book.rating(1, "box"), book.rating(1, "pm")
    for minutes in (0.0, 0.5, None):
        book.apply_game(
            [_row(minutes=minutes, game_id="0022400002", game_date=date(2024, 11, 3))],
            {"BOS": 5},
        )
        assert book.rating(1, "box") == box
        assert book.rating(1, "pm") == pm
        assert book.recent_minutes(1) == [30.0]
    # A player whose only row is a zero-minute row stays unseen.
    book.apply_game(
        [_row(player_id=7, minutes=0.0, game_id="0022400003", game_date=date(2024, 11, 4))],
        {"BOS": 5},
    )
    assert book.rating(7, "box") == ROOKIE_PRIOR
    assert book.last_team(7) is None


def test_missing_plus_minus_skips_only_pm():
    book = PlayerBook(PARAMS)
    book.apply_game([_row(plus_minus=None)], {"BOS": 5})
    assert book.rating(1, "pm") == ROOKIE_PRIOR
    assert book.rating(1, "box") != ROOKIE_PRIOR


def test_recent_minutes_and_team_games_keep_last_ten():
    book = PlayerBook(PARAMS)
    for i in range(12):
        book.apply_game(
            [
                _row(player_id=1, minutes=float(20 + i), game_id=f"g{i}", game_date=date(2024, 11, 1 + i)),
                _row(player_id=100 + i, minutes=10.0, game_id=f"g{i}", game_date=date(2024, 11, 1 + i)),
                _row(player_id=50, team="NYK", minutes=30.0, game_id=f"g{i}", game_date=date(2024, 11, 1 + i)),
            ],
            {"BOS": 1, "NYK": -1},
        )
    assert book.recent_minutes(1) == [float(20 + i) for i in range(2, 12)]
    games = book.team_recent_games("BOS")
    assert len(games) == 10
    assert games[0] == {1, 102}
    assert games[-1] == {1, 111}
    assert book.team_recent_games("NYK") == [{50}] * 10


def test_apply_game_rejects_out_of_order_dates():
    book = PlayerBook(PARAMS)
    book.apply_game([_row(game_date=date(2024, 11, 5))], {"BOS": 5})
    with pytest.raises(ValueError):
        book.apply_game([_row(game_id="x", game_date=date(2024, 11, 4))], {"BOS": 5})
