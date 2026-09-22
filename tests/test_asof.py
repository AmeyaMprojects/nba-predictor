from datetime import UTC, date, datetime, timedelta, timezone

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
            "INSERT INTO injury_status_raw "
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
        "INSERT INTO injury_status_raw "
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


# --- FIX 2: as_of is immutable and fully validated ------------------------


def test_as_of_has_no_setter(con):
    view = AsOfView(con, CUTOFF)
    with pytest.raises(AttributeError):
        view.as_of = datetime(2099, 1, 1, tzinfo=UTC)


def test_as_of_property_returns_the_constructed_value(con):
    view = AsOfView(con, CUTOFF)
    assert view.as_of == CUTOFF


def test_non_utc_offset_as_of_is_rejected(con):
    tz = timezone(timedelta(hours=-5))
    with pytest.raises(AsOfError, match="UTC"):
        AsOfView(con, datetime(2025, 1, 15, 22, 0, tzinfo=tz))


# --- FIX 3: wrong type raises AsOfError, not AttributeError ---------------


@pytest.mark.parametrize(
    "bad_as_of",
    [date(2025, 1, 15), "2025-01-15T22:00:00Z", 1736979600, None],
)
def test_non_datetime_as_of_raises_asof_error(con, bad_as_of):
    with pytest.raises(AsOfError):
        AsOfView(con, bad_as_of)


# --- FIX 4: latest() collapses to one row per entity -----------------------


def test_latest_returns_only_the_most_recent_status_per_entity(con):
    # The fixture already has an 'Out' row for LAL/past at CUTOFF-2h. Add a
    # later revision (still before the cutoff) for the same entity key.
    con.execute(
        "INSERT INTO injury_status_raw "
        "(report_date, game_date, team, player, status, observed_at) VALUES (?,?,?,?,?,?)",
        [CUTOFF.date(), CUTOFF.date(), "LAL", "past", "Probable", CUTOFF - timedelta(hours=1)],
    )
    view = AsOfView(con, CUTOFF)
    rows = view.latest("injury_status").project("player, status").fetchall()
    players = {r[0]: r[1] for r in rows}
    assert players["past"] == "Probable"
    assert "future" not in players


def test_latest_rejects_unknown_key_column(con):
    view = AsOfView(con, CUTOFF)
    with pytest.raises(AsOfError, match="unknown column"):
        view.latest("injury_status", key=("team; DROP TABLE injury_status_raw",))
