from datetime import UTC, date, datetime

import pandas as pd
import pytest

from predictor import db
from predictor.sources import nba_stats

OBSERVED = datetime(2025, 6, 23, 0, 0, tzinfo=UTC)


def _frame(rows):
    return pd.DataFrame(
        rows,
        columns=["GAME_ID", "GAME_DATE", "MATCHUP", "WL", "PTS", "TEAM_ABBREVIATION"],
    )


def test_pair_team_rows_collapses_two_rows_into_one_game():
    df = _frame(
        [
            ["0042400407", "2025-06-22", "IND @ OKC", "L", 91, "IND"],
            ["0042400407", "2025-06-22", "OKC vs. IND", "W", 103, "OKC"],
        ]
    )
    games = nba_stats.pair_team_rows(df)
    assert len(games) == 1
    game = games[0]
    assert game.home_team == "OKC"
    assert game.away_team == "IND"
    assert game.home_points == 103
    assert game.away_points == 91
    assert game.game_date == date(2025, 6, 22)
    assert game.status == "FINAL"


def test_unplayed_game_is_marked_scheduled_with_no_score():
    df = _frame(
        [
            ["0022500001", "2026-10-21", "LAL @ GSW", None, None, "LAL"],
            ["0022500001", "2026-10-21", "GSW vs. LAL", None, None, "GSW"],
        ]
    )
    game = nba_stats.pair_team_rows(df)[0]
    assert game.status == "SCHEDULED"
    assert game.home_points is None


def test_unpaired_row_is_dropped_rather_than_guessed():
    df = _frame([["0042400407", "2025-06-22", "IND @ OKC", "L", 91, "IND"]])
    assert nba_stats.pair_team_rows(df) == []


def test_neutral_site_game_both_rows_at_sign_form_is_paired_correctly():
    # Real example from the 2024-25 season (NBA Cup group play, Las Vegas):
    # neither team is the true home team, so BOTH rows carry the identical
    # "AWAY @ HOME" text -- neither contains "vs." -- unlike a normal game
    # where the home team's own row says "HOME vs. AWAY".
    df = _frame(
        [
            ["0012400001", "2024-10-04", "BOS @ DEN", "L", 103, "DEN"],
            ["0012400001", "2024-10-04", "BOS @ DEN", "W", 107, "BOS"],
        ]
    )
    games = nba_stats.pair_team_rows(df)
    assert len(games) == 1
    game = games[0]
    assert game.home_team == "DEN"
    assert game.away_team == "BOS"
    assert game.home_points == 103
    assert game.away_points == 107
    assert game.status == "FINAL"


def test_vs_form_game_home_away_assignment_unchanged():
    # Pins that the "vs." form still resolves to the identical home/away
    # assignment as before the matchup-parsing rewrite -- an inversion here
    # would silently corrupt home-court advantage estimation dataset-wide.
    df = _frame(
        [
            ["0042400407", "2025-06-22", "IND @ OKC", "L", 91, "IND"],
            ["0042400407", "2025-06-22", "OKC vs. IND", "W", 103, "OKC"],
        ]
    )
    game = nba_stats.pair_team_rows(df)[0]
    assert game.home_team == "OKC"
    assert game.away_team == "IND"
    assert game.home_points == 103
    assert game.away_points == 91


def test_unresolvable_group_is_counted_and_logged_not_dropped_silently(capsys):
    # Neither "vs." nor "@" appears in either MATCHUP string, so the
    # home/away pairing cannot be parsed at all -- a genuinely malformed
    # group, distinct from the two well-formed real-world shapes above.
    df = _frame(
        [
            ["0099999999", "2025-01-01", "GARBLED TEXT", "W", 100, "AAA"],
            ["0099999999", "2025-01-01", "GARBLED TEXT", "L", 90, "BBB"],
        ]
    )
    dropped: list[nba_stats.DroppedGame] = []
    games = nba_stats.pair_team_rows(df, dropped=dropped)

    assert games == []
    assert len(dropped) == 1
    assert dropped[0].game_id == "0099999999"

    out = capsys.readouterr().out
    assert "DROPPED" in out
    assert "0099999999" in out


def test_ingest_writes_rows_with_supplied_observed_at(tmp_path, monkeypatch):
    con = db.connect(tmp_path / "t.duckdb")
    db.migrate(con)
    df = _frame(
        [
            ["0042400407", "2025-06-22", "IND @ OKC", "L", 91, "IND"],
            ["0042400407", "2025-06-22", "OKC vs. IND", "W", 103, "OKC"],
        ]
    )
    monkeypatch.setattr(nba_stats, "fetch_season", lambda season: df)

    count = nba_stats.ingest_season(con, "2024-25", observed_at=OBSERVED)
    assert count == 1
    # games_raw is the physical table (see predictor.db.POINT_IN_TIME_TABLES);
    # tests are exempt from the repo-wide "_raw" literal ban enforced by
    # tests/test_leakage.py, which explicitly excludes tests/.
    row = con.execute(
        "SELECT home_team, away_team, home_points, observed_at FROM games_raw"
    ).fetchone()
    assert row == ("OKC", "IND", 103, OBSERVED)


def test_ingest_rejects_naive_observed_at(tmp_path, monkeypatch):
    con = db.connect(tmp_path / "t2.duckdb")
    db.migrate(con)
    df = _frame(
        [
            ["0042400407", "2025-06-22", "IND @ OKC", "L", 91, "IND"],
            ["0042400407", "2025-06-22", "OKC vs. IND", "W", 103, "OKC"],
        ]
    )
    monkeypatch.setattr(nba_stats, "fetch_season", lambda season: df)

    with pytest.raises(ValueError):
        nba_stats.ingest_season(con, "2024-25", observed_at=datetime(2025, 6, 23, 0, 0))
