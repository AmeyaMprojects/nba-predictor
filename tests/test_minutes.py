"""Expected minutes and redistribution (Task 4)."""

from __future__ import annotations

import pytest

from predictor.model.minutes import (
    MINUTES_CAP,
    ROTATION_GAMES,
    TEAM_MINUTES,
    available_minutes,
    normalize_minutes,
    strength,
)


def test_constants():
    assert TEAM_MINUTES == 240.0
    assert MINUTES_CAP == 44.0
    assert ROTATION_GAMES == 10


def test_normalize_minutes_scales_to_240():
    out = normalize_minutes({1: 36, 2: 36, 3: 24, 4: 24, 5: 24, 6: 24, 7: 24, 8: 12})
    assert round(out[1], 4) == 42.3529
    assert sum(out.values()) == pytest.approx(240.0)


def test_normalize_minutes_caps_and_respreads():
    out = normalize_minutes({1: 40, **{p: 20 for p in range(2, 8)}})
    assert out[1] == 44.0
    assert round(out[2], 4) == 32.6667
    assert sum(out.values()) == pytest.approx(240.0)


def test_normalize_minutes_every_player_capped():
    out = normalize_minutes({1: 30, 2: 30, 3: 30, 4: 30, 5: 30})
    assert out == {p: 44.0 for p in range(1, 6)}


def test_normalize_minutes_respread_can_cap_again():
    # Sum is already 240. Capping 1 (60 -> 44) spreads 16 over 180 minutes,
    # which lifts 2..5 from 42 to 45.73, so they are capped in a second round
    # and player 6 takes the last 240 - 5*44 = 20.
    out = normalize_minutes({1: 60, 2: 42, 3: 42, 4: 42, 5: 42, 6: 12})
    assert [out[p] for p in range(1, 6)] == [44.0] * 5
    assert out[6] == pytest.approx(20.0)
    assert sum(out.values()) == pytest.approx(240.0)


def test_normalize_minutes_empty_and_zero():
    assert normalize_minutes({}) == {}
    assert normalize_minutes({1: 0.0, 2: 0.0}) == {1: 0.0, 2: 0.0}


def _eight():
    normal = {p: 30.0 for p in range(1, 9)}
    ratings = {p: 0.0 for p in range(1, 9)}
    ratings[1] = 6.0
    ratings[2] = 2.0
    return normal, ratings


def test_injury_gap_out():
    normal, ratings = _eight()
    assert strength(normal, ratings) == pytest.approx(5.0)
    available = available_minutes(normal, {1: 0.0})
    assert available[1] == 0.0
    assert sum(available.values()) == pytest.approx(240.0)
    gap = strength(available, ratings) - strength(normal, ratings)
    assert round(gap, 6) == -3.571429


def test_injury_gap_questionable():
    normal, ratings = _eight()
    available = available_minutes(normal, {1: 0.5})
    gap = strength(available, ratings) - strength(normal, ratings)
    assert round(gap, 6) == -1.666667


def test_available_minutes_missing_q_is_one():
    normal, _ = _eight()
    assert available_minutes(normal, {}) == pytest.approx(normal)


def test_available_minutes_everyone_out_is_all_zero():
    normal, _ = _eight()
    assert available_minutes(normal, {p: 0.0 for p in normal}) == {p: 0.0 for p in normal}


def test_normalize_minutes_cannot_respread_onto_zero_minute_players():
    # Player 2 has no minutes to scale, so player 1's excess has nowhere to go
    # and the team total stays below 240.
    out = normalize_minutes({1: 300, 2: 0})
    assert out == {1: 44.0, 2: 0.0}
    assert sum(out.values()) < 240.0


def test_strength_uses_default_for_unrated_players():
    assert strength({1: 48.0, 2: 24.0}, {1: 3.0}, default=-2.0) == pytest.approx(2.0)
    # The default is the rookie prior.
    assert strength({2: 24.0}, {}) == pytest.approx(-1.0)
