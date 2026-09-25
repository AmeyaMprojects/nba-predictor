"""Adversarial tests: predictors that deliberately try to see the future.

If any test here fails, every backtest number this project produces is
meaningless, and the user would be publishing a track record built on a lie.
Treat a failure as a critical defect, never as a test to relax.
"""

from datetime import UTC, date, datetime, timedelta

import duckdb
import pytest

from predictor import db
from predictor.backtest import replay
from predictor.backtest.baselines import always_home

TIP = datetime(2025, 1, 16, 0, 0, tzinfo=UTC)


@pytest.fixture
def con(tmp_path):
    c = db.connect(tmp_path / "t.duckdb")
    db.migrate(c)
    g = db.POINT_IN_TIME_TABLES["games"]
    i = db.POINT_IN_TIME_TABLES["injury_status"]
    c.execute(
        f"INSERT INTO {g} (game_id, season, game_date, home_team, away_team,"
        " home_points, away_points, status, observed_at, reconstructed)"
        " VALUES (?,?,?,?,?,?,?,?,?,TRUE)",
        ["0022400561", "2024-25", date(2025, 1, 15), "PHI", "NYK",
         None, None, "SCHEDULED", TIP - timedelta(days=7)],
    )
    c.execute(
        f"INSERT INTO {g} (game_id, season, game_date, home_team, away_team,"
        " home_points, away_points, status, observed_at, reconstructed)"
        " VALUES (?,?,?,?,?,?,?,?,?,TRUE)",
        ["0022400561", "2024-25", date(2025, 1, 15), "PHI", "NYK",
         119, 110, "FINAL", TIP + timedelta(hours=3)],
    )
    c.execute(
        f"INSERT INTO {i} (report_date, game_date, matchup, team, player,"
        " status, reason, observed_at, game_time)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        [date(2025, 1, 15), date(2025, 1, 15), "NYK@PHI", "PHI", "Embiid,Joel",
         "Out", "injury", TIP - timedelta(hours=2), "07:00 (ET)"],
    )
    return c


def test_the_view_handed_to_a_predictor_cannot_see_the_final_score(con):
    """The core guarantee. A predictor querying games sees SCHEDULED only."""
    seen = {}

    def snooping(game, view):
        rows = view.table("games").project("status, home_points").fetchall()
        seen["rows"] = rows
        return 0.5

    replay.replay(con, snooping)
    assert seen["rows"] == [("SCHEDULED", None)]
    assert all(r[1] is None for r in seen["rows"]), "final score leaked"


def test_the_game_object_carries_exactly_the_allowed_fields(con):
    """FIX 13(c): a denylist ("these specific fields must be absent") lets a
    FUTURE field leak through silently -- someone adds `home_points` back
    under a different name, or adds an unrelated result-shaped field, and
    this test keeps passing. An allowlist of the exact expected field set
    fails loudly the moment `GameToPredict` changes at all, forcing whoever
    touches it to consciously re-justify the new shape. This is the single
    contract the model sub-project will extend, so it must fail loudly.
    """
    captured = {}

    def grabby(game, view):
        captured["fields"] = vars(game)
        return 0.5

    replay.replay(con, grabby)

    assert set(captured["fields"]) == {
        "game_id", "season", "game_date", "home_team", "away_team",
    }
    blob = repr(captured["fields"])
    assert "119" not in blob and "110" not in blob


def test_a_predictor_cannot_reach_the_physical_table_through_the_view(con):
    """The renamed physical tables are unreachable from a chained query.

    This must hold for EVERY logical name in db.POINT_IN_TIME_TABLES, not
    just "games" -- a future table added there is covered automatically
    because this iterates db.point_in_time_logical_names() rather than a
    hardcoded list.
    """
    outcome = {}

    def cheater(game, view):
        for name in db.point_in_time_logical_names():
            try:
                view.table("games").project(
                    f"status, (SELECT count(*) FROM {name}) AS leak"
                ).fetchall()
                outcome[name] = "leaked"
            except duckdb.CatalogException:
                outcome[name] = "blocked"
        return 0.5

    replay.replay(con, cheater)
    for name in db.point_in_time_logical_names():
        assert outcome[name] == "blocked", f"{name} was reachable via a stray FROM clause"


def test_a_predictor_cannot_reach_a_different_table_from_within_another_tables_view(con):
    """Launching from one table's view and reaching for a DIFFERENT logical
    name in a subquery must be blocked too, not just self-reference."""
    outcome = {}

    def cheater(game, view):
        try:
            view.table("injury_status").project(
                "player, (SELECT count(*) FROM games) AS leak"
            ).fetchall()
            outcome["leaked"] = True
        except duckdb.CatalogException:
            outcome["leaked"] = False
        return 0.5

    replay.replay(con, cheater)
    assert outcome["leaked"] is False


def test_a_predictor_cannot_move_the_cutoff(con):
    outcome = {}

    def tamperer(game, view):
        try:
            view.as_of = datetime(2099, 1, 1, tzinfo=UTC)
            outcome["moved"] = True
        except AttributeError:
            outcome["moved"] = False
        return 0.5

    replay.replay(con, tamperer)
    assert outcome["moved"] is False


def test_injury_rows_published_after_the_cutoff_are_invisible(con):
    """A report filed after the cutoff must not reach the predictor."""
    i = db.POINT_IN_TIME_TABLES["injury_status"]
    con.execute(
        f"INSERT INTO {i} (report_date, game_date, matchup, team, player,"
        " status, reason, observed_at, game_time)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        [date(2025, 1, 15), date(2025, 1, 15), "NYK@PHI", "PHI", "LateScratch,Guy",
         "Out", "injury", TIP - timedelta(minutes=5), "07:00 (ET)"],
    )
    players = {}

    def looker(game, view):
        players["seen"] = {
            r[0] for r in view.table("injury_status").project("player").fetchall()
        }
        return 0.5

    replay.replay(con, looker, buffer_minutes=30)
    assert "Embiid,Joel" in players["seen"]
    assert "LateScratch,Guy" not in players["seen"], "post-cutoff report leaked"


def test_every_cutoff_precedes_its_own_tipoff(con):
    preds, _ = replay.replay(con, always_home)
    assert preds, "fixture produced no predictions"
    for p in preds:
        assert p.cutoff < p.tipoff


def test_a_zero_buffer_still_does_not_include_the_result(con):
    """Even with no safety buffer, the FINAL row is observed after tip-off."""
    seen = {}

    def snooping(game, view):
        seen["rows"] = view.table("games").project("status, home_points").fetchall()
        return 0.5

    replay.replay(con, snooping, buffer_minutes=0)
    # FIX 13(d): `all(...)` over an empty sequence is vacuously True -- if
    # the predictor were never even invoked (e.g. every game wrongly
    # skipped at buffer_minutes=0), this test would still pass despite
    # proving nothing. Assert the view actually returned rows first.
    assert seen.get("rows"), "predictor was never invoked -- nothing was actually checked"
    assert all(r[1] is None for r in seen["rows"])
