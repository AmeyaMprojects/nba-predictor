from datetime import UTC, date, datetime

import pytest

from predictor.backtest.baselines import (
    GameToPredict,
    always_home,
    fixed_probability,
)

GAME = GameToPredict(
    game_id="0022400561",
    season="2024-25",
    game_date=date(2025, 1, 15),
    home_team="PHI",
    away_team="NYK",
    tipoff=datetime(2025, 1, 16, 0, 0, tzinfo=UTC),
)


def test_always_home_returns_certainty():
    assert always_home(GAME, None) == 1.0


def test_fixed_probability_returns_its_value():
    assert fixed_probability(0.62)(GAME, None) == 0.62


def test_fixed_probability_rejects_values_outside_zero_one():
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            fixed_probability(bad)


def test_game_is_immutable():
    with pytest.raises(Exception):
        GAME.home_team = "BOS"
