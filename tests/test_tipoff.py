from datetime import UTC, date, datetime

import pytest

from predictor.backtest import tipoff


def test_evening_game_is_pm():
    # '07:00 (ET)' on a January date means 7pm EST = 00:00 UTC next day
    got = tipoff.parse_game_time("07:00 (ET)", date(2025, 1, 15))
    assert got == datetime(2025, 1, 16, 0, 0, tzinfo=UTC)


def test_noon_game_is_not_shifted_to_midnight():
    # '12:00 (ET)' means noon, not midnight
    got = tipoff.parse_game_time("12:00 (ET)", date(2025, 1, 15))
    assert got == datetime(2025, 1, 15, 17, 0, tzinfo=UTC)


def test_afternoon_game():
    got = tipoff.parse_game_time("03:30 (ET)", date(2025, 1, 15))
    assert got == datetime(2025, 1, 15, 20, 30, tzinfo=UTC)


def test_layout_without_space_before_paren_parses_identically():
    with_space = tipoff.parse_game_time("08:00 (ET)", date(2025, 1, 15))
    without = tipoff.parse_game_time("08:00(ET)", date(2025, 1, 15))
    assert with_space == without


def test_daylight_saving_is_honoured():
    # June is EDT (UTC-4); January is EST (UTC-5)
    summer = tipoff.parse_game_time("07:00 (ET)", date(2025, 6, 10))
    winter = tipoff.parse_game_time("07:00 (ET)", date(2025, 1, 10))
    assert summer.hour == 23
    assert winter.hour == 0  # rolled into the next UTC day


def test_unparseable_returns_none():
    for bad in ["", "TBD", "not a time", "25:00 (ET)"]:
        assert tipoff.parse_game_time(bad, date(2025, 1, 15)) is None
