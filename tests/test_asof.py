from datetime import UTC, datetime, timedelta

import pytest

from predictor import db
from predictor.asof import AsOfError, AsOfView

CUTOFF = datetime(2025, 1, 15, 22, 0, tzinfo=UTC)


@pytest.fixture
def con(tmp_path):
    c = db.connect(tmp_path / "t.duckdb")
    db.migrate(c)
    for offset, player in [(-2, "past"), (2, "future")]:
        c.execute(
            "INSERT INTO injury_status "
            "(report_date, game_date, team, player, status, observed_at) VALUES (?,?,?,?,?,?)",
            [
                CUTOFF.date(),
                CUTOFF.date(),
                "LAL",
                player,
                "Out",
                CUTOFF + timedelta(hours=offset),
            ],
        )
    return c


def test_returns_only_rows_observed_at_or_before_cutoff(con):
    view = AsOfView(con, CUTOFF)
    players = {r[0] for r in view.table("injury_status").project("player").fetchall()}
    assert players == {"past"}


def test_boundary_row_exactly_at_cutoff_is_included(con):
    con.execute(
        "INSERT INTO injury_status "
        "(report_date, game_date, team, player, status, observed_at) VALUES (?,?,?,?,?,?)",
        [CUTOFF.date(), CUTOFF.date(), "BOS", "boundary", "Out", CUTOFF],
    )
    view = AsOfView(con, CUTOFF)
    players = {r[0] for r in view.table("injury_status").project("player").fetchall()}
    assert "boundary" in players


def test_naive_as_of_is_rejected(con):
    with pytest.raises(AsOfError, match="timezone-aware"):
        AsOfView(con, datetime(2025, 1, 15, 22, 0))


def test_unregistered_table_is_rejected(con):
    view = AsOfView(con, CUTOFF)
    with pytest.raises(AsOfError, match="not a point-in-time table"):
        view.table("ingest_runs")


def test_unknown_table_is_rejected(con):
    view = AsOfView(con, CUTOFF)
    with pytest.raises(AsOfError, match="not a point-in-time table"):
        view.table("games; DROP TABLE games")
