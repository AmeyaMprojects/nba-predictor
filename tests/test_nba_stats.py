from datetime import UTC, date, datetime, time, timedelta

import pandas as pd
import pytest

from predictor import db
from predictor.asof import AsOfView
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


def test_ingest_season_populates_caller_supplied_dropped_list(tmp_path, monkeypatch):
    # Mirrors pair_team_rows: a caller-supplied `dropped` list must be
    # populated with any game that could not be written to the database,
    # and the pinned `int` return contract (count of games ACTUALLY
    # ingested) must still hold even when some games were dropped.
    con = db.connect(tmp_path / "t3.duckdb")
    db.migrate(con)
    df = _frame(
        [
            # One good, pairable game...
            ["0042400407", "2025-06-22", "IND @ OKC", "L", 91, "IND"],
            ["0042400407", "2025-06-22", "OKC vs. IND", "W", 103, "OKC"],
            # ...and one genuinely unresolvable group.
            ["0099999999", "2025-01-01", "GARBLED TEXT", "W", 100, "AAA"],
            ["0099999999", "2025-01-01", "GARBLED TEXT", "L", 90, "BBB"],
        ]
    )
    monkeypatch.setattr(nba_stats, "fetch_season", lambda season: df)

    dropped: list[nba_stats.DroppedGame] = []
    count = nba_stats.ingest_season(con, "2024-25", observed_at=OBSERVED, dropped=dropped)

    assert count == 1
    assert len(dropped) == 1
    assert dropped[0].game_id == "0099999999"
    written = con.execute("SELECT count(*) FROM games_raw").fetchone()[0]
    assert written == 1


# --- Historical-backfill observed_at derivation (no explicit observed_at) ---

PLAYED_GAME_DATE = date(2025, 1, 15)
_EXPECTED_SCHEDULED_AT = datetime.combine(
    PLAYED_GAME_DATE - timedelta(days=7), time(12, 0), tzinfo=UTC
)
_EXPECTED_FINAL_AT = datetime.combine(
    PLAYED_GAME_DATE + timedelta(days=1), time(8, 0), tzinfo=UTC
)


def _played_game_frame():
    return _frame(
        [
            ["0042400500", "2025-01-15", "IND @ OKC", "L", 91, "IND"],
            ["0042400500", "2025-01-15", "OKC vs. IND", "W", 103, "OKC"],
        ]
    )


def _future_game_frame():
    future_date = "2026-11-20"
    return _frame(
        [
            [
                "0022500999",
                future_date,
                "LAL @ GSW",
                None,
                None,
                "LAL",
            ],
            [
                "0022500999",
                future_date,
                "GSW vs. LAL",
                None,
                None,
                "GSW",
            ],
        ]
    )


def test_played_game_without_explicit_observed_at_produces_scheduled_and_final_rows(
    tmp_path, monkeypatch
):
    con = db.connect(tmp_path / "derived.duckdb")
    db.migrate(con)
    monkeypatch.setattr(nba_stats, "fetch_season", lambda season: _played_game_frame())

    count = nba_stats.ingest_season(con, "2024-25")
    assert count == 1

    rows = con.execute(
        "SELECT status, home_points, away_points, reconstructed, observed_at "
        "FROM games_raw WHERE game_id = '0042400500' ORDER BY observed_at"
    ).fetchall()
    assert len(rows) == 2

    scheduled, final = rows
    assert scheduled == ("SCHEDULED", None, None, True, _EXPECTED_SCHEDULED_AT)
    assert final == ("FINAL", 103, 91, True, _EXPECTED_FINAL_AT)


def test_future_game_without_explicit_observed_at_produces_only_scheduled_row(
    tmp_path, monkeypatch
):
    con = db.connect(tmp_path / "derived_future.duckdb")
    db.migrate(con)
    monkeypatch.setattr(nba_stats, "fetch_season", lambda season: _future_game_frame())

    count = nba_stats.ingest_season(con, "2025-26")
    assert count == 1

    rows = con.execute(
        "SELECT status, home_points, away_points FROM games_raw "
        "WHERE game_id = '0022500999'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0] == ("SCHEDULED", None, None)


def test_asof_before_tipoff_sees_fixture_with_no_leaked_score(tmp_path, monkeypatch):
    # This is the property the entire backtest rests on: at a cutoff before
    # the FINAL observation, AsOfView must show the game with NO score --
    # not a null-checked-later score, a genuinely absent one.
    con = db.connect(tmp_path / "derived_leak.duckdb")
    db.migrate(con)
    monkeypatch.setattr(nba_stats, "fetch_season", lambda season: _played_game_frame())
    nba_stats.ingest_season(con, "2024-25")

    pre_tipoff_cutoff = _EXPECTED_SCHEDULED_AT + timedelta(hours=1)
    assert pre_tipoff_cutoff < _EXPECTED_FINAL_AT

    view = AsOfView(con, pre_tipoff_cutoff)
    rows = view.table("games").filter("game_id = '0042400500'").fetchall()
    assert len(rows) == 1
    columns = view.table("games").columns
    row = dict(zip(columns, rows[0]))
    assert row["status"] == "SCHEDULED"
    assert row["home_points"] is None
    assert row["away_points"] is None

    post_final_cutoff = _EXPECTED_FINAL_AT + timedelta(hours=1)
    view_after = AsOfView(con, post_final_cutoff)
    rows_after = view_after.table("games").filter("game_id = '0042400500'").fetchall()
    columns_after = view_after.table("games").columns
    statuses = {dict(zip(columns_after, r))["status"] for r in rows_after}
    assert "FINAL" in statuses
    final_row = next(
        dict(zip(columns_after, r))
        for r in rows_after
        if dict(zip(columns_after, r))["status"] == "FINAL"
    )
    assert final_row["home_points"] == 103
    assert final_row["away_points"] == 91


def test_asof_latest_before_tipoff_returns_scheduled_not_final(tmp_path, monkeypatch):
    con = db.connect(tmp_path / "derived_latest.duckdb")
    db.migrate(con)
    monkeypatch.setattr(nba_stats, "fetch_season", lambda season: _played_game_frame())
    nba_stats.ingest_season(con, "2024-25")

    pre_tipoff_cutoff = _EXPECTED_SCHEDULED_AT + timedelta(hours=1)
    view = AsOfView(con, pre_tipoff_cutoff)
    rows = view.latest("games").fetchall()
    columns = view.latest("games").columns
    matching = [dict(zip(columns, r)) for r in rows if r[columns.index("game_id")] == "0042400500"]
    assert len(matching) == 1
    assert matching[0]["status"] == "SCHEDULED"
    assert matching[0]["home_points"] is None


def test_reconstructed_flag_true_on_derived_rows_false_on_explicit(tmp_path, monkeypatch):
    con = db.connect(tmp_path / "derived_flag.duckdb")
    db.migrate(con)
    monkeypatch.setattr(nba_stats, "fetch_season", lambda season: _played_game_frame())

    nba_stats.ingest_season(con, "2024-25")
    derived_flags = {
        r[0]
        for r in con.execute(
            "SELECT reconstructed FROM games_raw WHERE game_id = '0042400500'"
        ).fetchall()
    }
    assert derived_flags == {True}

    con2 = db.connect(tmp_path / "explicit_flag.duckdb")
    db.migrate(con2)
    nba_stats.ingest_season(con2, "2024-25", observed_at=OBSERVED)
    explicit_flag = con2.execute(
        "SELECT reconstructed FROM games_raw WHERE game_id = '0042400500'"
    ).fetchone()[0]
    assert explicit_flag is False


def test_migrate_adds_reconstructed_column_idempotently_without_data_loss(tmp_path):
    # Simulates upgrading a pre-existing database created before the
    # `reconstructed` column existed on games_raw: build the table by hand
    # without the column, insert a row, then run migrate() twice and prove
    # neither run errors nor loses the row.
    con = db.connect(tmp_path / "upgrade.duckdb")
    con.execute(
        """
        CREATE TABLE games_raw (
            game_id       VARCHAR NOT NULL,
            season        VARCHAR NOT NULL,
            game_date     DATE NOT NULL,
            home_team     VARCHAR NOT NULL,
            away_team     VARCHAR NOT NULL,
            home_points   INTEGER,
            away_points   INTEGER,
            status        VARCHAR NOT NULL,
            observed_at   TIMESTAMP WITH TIME ZONE NOT NULL,
            PRIMARY KEY (game_id, observed_at)
        )
        """
    )
    con.execute(
        "INSERT INTO games_raw (game_id, season, game_date, home_team, away_team,"
        " home_points, away_points, status, observed_at) VALUES"
        " ('pre-existing', '2024-25', '2025-01-01', 'OKC', 'IND', 103, 91,"
        " 'FINAL', '2025-01-02 08:00:00+00')"
    )

    db.migrate(con)
    db.migrate(con)

    row = con.execute(
        "SELECT game_id, reconstructed FROM games_raw WHERE game_id = 'pre-existing'"
    ).fetchone()
    assert row == ("pre-existing", False)
    count = con.execute("SELECT count(*) FROM games_raw").fetchone()[0]
    assert count == 1
