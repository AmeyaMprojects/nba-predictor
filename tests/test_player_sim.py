"""Point-in-time player feature simulation (Task 6).

Fixture: tests/player_fixtures.py. Every game on date D has its final call at
D 23:15 UTC (tip-off D+1 00:00 UTC minus 45 min) and its early call at
D 12:30 UTC. No test touches data/ or the network.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, time, timedelta
from datetime import date as _date

import pytest

from model_fixtures import add_game, fixture_con
from player_fixtures import (
    D1,
    D3,
    G1,
    G2,
    G3,
    ROSTERS,
    SEASON,
    add_player_row,
    add_report,
    add_team_rows,
    build_player_fixture,
    early_at,
    final_at,
    player_index,
    report_name,
)
from predictor import db
from predictor.asof import AsOfView
from predictor.model import fit as fit_mod
from predictor.model import player_sim
from predictor.model.availability import STATUS_PRIORS, InjuryHistory, Listing
from predictor.model.player_ratings import PlayerBook, PlayerParams
from predictor.model.player_sim import (
    EspnMergeStats,
    PlayerLines,
    collect_listings,
    espn_rows,
    simulate,
)
from predictor.model.venues import VenueIndex

PARAMS = PlayerParams(half_life_minutes=1500.0, season_regression=0.4)
STAR = 1
STAR_NAME = report_name(ROSTERS["BOS"][STAR])  # "Adams,Al"


@pytest.fixture
def con(tmp_path):
    c = fixture_con(tmp_path)
    yield c
    c.close()


def _run(con, listings=(), params=PARAMS, report_lag=timedelta(0)):
    """No report lag by default, so the tests below can place reports right
    at the call time; the lag itself is tested on its own."""
    games = fit_mod._load(con, VenueIndex.from_db(con))
    lines = PlayerLines.from_db(con)
    history = InjuryHistory.from_db(con)
    index = player_index(con)
    return simulate(games, lines, history, index, params, list(listings),
                    report_lag=report_lag)


def _run_default_lag(con):
    games = fit_mod._load(con, VenueIndex.from_db(con))
    return simulate(games, PlayerLines.from_db(con), InjuryHistory.from_db(con),
                    player_index(con), PARAMS, [])


def _force_ratings(monkeypatch, star=6.0, others=2.0 / 7.0):
    """Star 6, every other BOS player 2/7 (sum 2), everyone else 0."""
    bos = set(ROSTERS["BOS"])

    def rating(self, player_id, formula, as_of=None):
        self._check_as_of(as_of)
        if player_id == STAR:
            return star
        return others if player_id in bos else 0.0

    monkeypatch.setattr(PlayerBook, "rating", rating)


# -- the brief's tests -------------------------------------------------------

def test_same_day_box_scores_not_visible_at_call(con):
    """BOS plays twice on D3 (fixture doubleheader). The star's huge G3 line
    must not reach the second D3 game's features; it does reach D3+2."""
    build_player_fixture(con)
    g3b, g4 = "0022400004", "0022400005"
    add_game(con, g3b, SEASON, D3, "MIA", "BOS", 99, 98)
    add_game(con, g4, SEASON, D3 + timedelta(days=2), "BOS", "MIA", 101, 99)
    add_team_rows(con, g3b, D3, "BOS")
    add_team_rows(con, g3b, D3, "MIA")
    # The star is better than his teammates, so his minutes share matters too.
    add_player_row(con, G1, D1, STAR, "BOS", 30.0, pts=35, plus_minus=12.0)
    add_player_row(con, G3, D3, STAR, "BOS", 40.0, pts=60, plus_minus=30.0)
    with_row, _ = _run(con)

    table = db.POINT_IN_TIME_TABLES["player_games"]
    con.execute(f"DELETE FROM {table} WHERE game_id = ? AND player_id = ?", [G3, STAR])
    without_row, _ = _run(con)

    assert with_row[g3b] == without_row[g3b]
    assert with_row[G3] == without_row[G3]
    # Sanity: the row is used once its date has passed.
    assert with_row[g4].final.roster_gap != without_row[g4].final.roster_gap


def test_report_published_after_call_time_is_ignored(con, monkeypatch):
    build_player_fixture(con)
    _force_ratings(monkeypatch)
    add_report(con, final_at(D3) + timedelta(seconds=1), D3, "BOS", STAR_NAME, "Out")
    late, _ = _run(con)
    assert late[G3].final.injury_gap == {"box": 0.0, "pm": 0.0}
    assert late[G3].final.no_report is True

    add_report(con, final_at(D3), D3, "BOS", STAR_NAME, "Out")
    on_time, _ = _run(con)
    assert on_time[G3].final.injury_gap["box"] != 0.0
    assert on_time[G3].final.report_at == final_at(D3)


def test_early_call_uses_earlier_report_than_final(con):
    build_player_fixture(con)
    add_report(con, datetime.combine(D3, time(12, 0), UTC), D3, "BOS", STAR_NAME,
               "Questionable")
    add_report(con, datetime.combine(D3, time(22, 30), UTC), D3, "BOS", STAR_NAME, "Out")
    feats, _ = _run(con)
    g = feats[G3]
    assert g.early.absent[0][2] == "Questionable"
    assert g.final.absent[0][2] == "Out"
    assert g.final.absent[0][:2] == ("BOS", "Al Adams")
    assert g.early.report_at == datetime.combine(D3, time(12, 0), UTC)


def test_missing_slot_falls_back_to_latest_earlier_report_and_counts(con):
    build_player_fixture(con)
    add_report(con, final_at(D3) - timedelta(hours=5), D3, "BOS", STAR_NAME, "Out")
    feats, stats = _run(con)
    assert feats[G3].final.fallback is True
    assert feats[G3].final.no_report is False
    assert stats.fallback["final"][SEASON] == 1
    # The early call (12:30Z) is before the 18:15Z report: no report at all.
    assert feats[G3].early.no_report is True
    assert stats.no_report["early"][SEASON] == 3  # G1, G2, G3 have no early report


def test_no_report_at_all_gives_zero_injury_gap_and_counts(con):
    build_player_fixture(con)
    feats, stats = _run(con)
    for call in ("early", "final"):
        cf = getattr(feats[G3], call)
        assert cf.no_report is True
        assert cf.report_at is None
        assert cf.fallback is False
        assert cf.injury_gap == {"box": 0.0, "pm": 0.0}
        assert stats.no_report[call][SEASON] == 3


def test_report_without_the_game_date_counts_as_fallback(con):
    """Ruling: a later report that does not list D (it satisfied the slot by
    header time) means the snapshot fell back -- counted even within 1 h."""
    build_player_fixture(con)
    add_report(con, final_at(D3) - timedelta(minutes=30), D3, "BOS", STAR_NAME, "Out")
    add_report(con, final_at(D3) - timedelta(minutes=10), D3 + timedelta(days=1),
               "NYK", "Ivey,Ia", "Out")
    feats, stats = _run(con)
    assert feats[G3].final.report_at == final_at(D3) - timedelta(minutes=30)
    assert feats[G3].final.fallback is True
    assert stats.fallback["final"][SEASON] == 1


def test_traded_player_listed_by_new_team_leaves_old_rotation(con, monkeypatch):
    build_player_fixture(con)
    _force_ratings(monkeypatch)
    base, _ = _run(con)
    # Full BOS: 0.625 * (6 + 2) = 5; NYK all 0.
    assert round(base[G3].final.roster_gap["box"], 6) == 5.0

    add_report(con, final_at(D3), D3, "NYK", STAR_NAME, "Out")
    feats, _ = _run(con)
    f = feats[G3].final
    # BOS without A: 7 players at 240/7 min, 2/7 each -> (240/7/48) * 2 = 10/7.
    assert round(f.roster_gap["box"], 6) == round(10 / 7, 6)
    # NYK never counted A (no NYK minutes), and BOS no longer has him.
    assert f.injury_gap == {"box": 0.0, "pm": 0.0}
    assert f.absent == ()


def test_injury_gap_matches_hand_value(con, monkeypatch):
    """Normal: 8 x 30 min, S = 0.625 * (6 + 2) = 5. Star out: 7 x 34.2857 min,
    S = 34.2857/48 * 2 = 1.428571. Gap -3.571429; NYK unaffected."""
    build_player_fixture(con)
    _force_ratings(monkeypatch)
    add_report(con, final_at(D3), D3, "BOS", STAR_NAME, "Out")
    feats, _ = _run(con)
    f = feats[G3].final
    assert round(f.injury_gap["box"], 6) == -3.571429
    assert round(f.injury_gap["pm"], 6) == -3.571429
    assert round(f.roster_gap["box"], 6) == 5.0
    assert f.absent == (("BOS", "Al Adams", "Out"),)


def test_availability_table_per_season_uses_earlier_seasons_only(con, monkeypatch):
    build_player_fixture(con)
    _force_ratings(monkeypatch)
    add_report(con, final_at(D3), D3, "BOS", STAR_NAME, "Out")
    # 200 "Out" listings that all played -- but in the replayed season itself.
    planted = [Listing(SEASON, c, "Out", True) for c in ("early", "final") for _ in range(200)]
    tables = player_sim.availability_tables(planted, SEASON)
    for call in ("early", "final"):
        assert tables[call].probs == STATUS_PRIORS
    feats, _ = _run(con, planted)
    assert round(feats[G3].final.injury_gap["box"], 6) == -3.571429

    # Control: the same listings one season earlier do move the table.
    earlier = [Listing("2023-24", "final", "Out", True) for _ in range(200)]
    assert player_sim.availability_tables(earlier, SEASON)["final"].probs["Out"] == 1.0
    feats, _ = _run(con, earlier)
    assert feats[G3].final.injury_gap["box"] == 0.0


def test_simulate_refuses_test_season(con):
    build_player_fixture(con)
    add_player_row(con, "0022600001", D3, STAR, "BOS", 30.0, season="2026-27")
    lines = PlayerLines.from_db(con)
    assert all(r.season != "2026-27" for day in lines.by_date.values() for r, _ in day)
    assert "0022600001" not in lines.played

    class _Con:
        """Hands fit._load a test-season row as if the query let one through."""

        def execute(self, *_args):
            return self

        def fetchall(self):
            return [("0022600001", "2026-27", D3, "BOS", "NYK", 100, 90)]

    with pytest.raises(fit_mod.FitError):
        fit_mod._load(_Con(), VenueIndex([]))

    games = fit_mod._load(con, VenueIndex.from_db(con))
    bad = fit_mod._Game(dataclasses.replace(games[0].result, season="2026-27"), games[0].x)
    with pytest.raises(fit_mod.FitError):
        simulate([bad], lines, InjuryHistory.from_db(con), player_index(con), PARAMS, [])


# -- extra coverage ------------------------------------------------------------

def test_later_rows_and_reports_never_change_earlier_features(con):
    """Leakage: adding a later game's box score and later reports (for a later
    date, and for the same date after its calls) leaves earlier features as
    they were."""
    build_player_fixture(con, target_rows=False)
    add_report(con, final_at(D3) - timedelta(minutes=20), D3, "BOS", STAR_NAME, "Doubtful")
    before, before_stats = _run(con)

    add_team_rows(con, G3, D3, "BOS")
    add_team_rows(con, G3, D3, "NYK")
    add_player_row(con, G3, D3, 2, "BOS", 48.0, pts=80, plus_minus=40.0)
    add_report(con, final_at(D3) + timedelta(minutes=5), D3, "BOS", STAR_NAME, "Out")
    add_report(con, final_at(D3) + timedelta(hours=20), D3 + timedelta(days=1),
               "BOS", "Baker,Bo", "Out")
    after, after_stats = _run(con)

    assert after == before
    assert after_stats.fallback == before_stats.fallback
    assert after_stats.no_report == before_stats.no_report


def test_games_without_tipoff_use_23z_and_are_counted(con):
    build_player_fixture(con)
    sched = db.POINT_IN_TIME_TABLES["schedule"]
    con.execute(f"UPDATE {sched} SET tip_off_utc = NULL WHERE game_id = ?", [G3])
    add_report(con, datetime.combine(D3, time(22, 30), UTC), D3, "BOS", STAR_NAME, "Out")
    feats, stats = _run(con)
    assert stats.no_tip == 1
    # Both calls at 23:00Z see the 22:30Z report.
    assert feats[G3].early.absent == feats[G3].final.absent == (("BOS", "Al Adams", "Out"),)


def test_unlinked_names_are_available_and_counted_per_season(con):
    build_player_fixture(con)
    add_report(con, final_at(D3), D3, "BOS", "Nobody,Zed", "Out")
    feats, stats = _run(con)
    assert feats[G3].final.injury_gap == {"box": 0.0, "pm": 0.0}
    assert stats.unlinked.unlinked[("Nobody,Zed", "BOS")] == 1
    assert stats.unlinked_by_season[SEASON].unlinked[("Nobody,Zed", "BOS")] == 1


def test_unknown_status_uses_half_and_is_counted(con, monkeypatch):
    build_player_fixture(con)
    _force_ratings(monkeypatch)
    add_report(con, final_at(D3), D3, "BOS", STAR_NAME, "Not With Team")
    feats, stats = _run(con)
    assert feats[G3].final.absent == (("BOS", "Al Adams", "Not With Team"),)
    assert stats.unknown_status["Not With Team"] == 1
    assert stats.unknown_by_season == {SEASON: {"Not With Team": 1}}
    # q = 0.5: {15, 30 x 7} -> x 16/15 -> 16 and 32; S = 2 + 1.333333 vs 5.
    assert round(feats[G3].final.injury_gap["box"], 6) == -1.666667


def test_preseason_rows_set_rosters_but_not_minutes(con, monkeypatch):
    """A summer signing (player 11, NYK last season) plays a BOS preseason
    game: he leaves NYK's rotation and, having NBA minutes, joins BOS's.
    His preseason minutes never feed his minutes history."""
    build_player_fixture(con)
    pre = "0012400001"
    add_player_row(con, pre, D3 - timedelta(days=1), 11, "BOS", 40.0,
                   season_type="preseason")
    lines = PlayerLines.from_db(con)
    rosters = player_sim.Rosters()
    for day in sorted(lines.by_date):
        if day >= D3:
            break
        rosters.apply_day(lines.by_date[day])
    assert 11 in rosters.rotation("BOS", as_of=D3)
    assert 11 not in rosters.rotation("NYK", as_of=D3)
    assert rosters.recent_minutes(11, as_of=D3) == [30.0, 30.0]


def test_collect_listings_marks_played(con):
    build_player_fixture(con)
    add_report(con, final_at(D3), D3, "BOS", STAR_NAME, "Questionable")  # he played
    add_report(con, final_at(D3), D3, "NYK", "Ivey,Ia", "Out")
    table = db.POINT_IN_TIME_TABLES["player_games"]
    con.execute(f"DELETE FROM {table} WHERE game_id = ? AND player_id = 11", [G3])
    games = fit_mod._load(con, VenueIndex.from_db(con))
    listings = collect_listings(con, games, InjuryHistory.from_db(con), player_index(con),
                               report_lag=timedelta(0))
    assert sorted(listings, key=lambda li: li.status) == [
        Listing(SEASON, "final", "Out", False),
        Listing(SEASON, "final", "Questionable", True),
    ]


def test_reads_are_guarded_by_as_of(con):
    build_player_fixture(con)
    lines = PlayerLines.from_db(con)
    rosters = player_sim.Rosters()
    rosters.apply_day(lines.by_date[D1])
    with pytest.raises(ValueError):
        rosters.rotation("BOS", as_of=D1)


def test_features_cover_every_game_and_carry_extras(con):
    build_player_fixture(con)
    feats, _ = _run(con)
    assert set(feats) == {G1, G2, G3}
    assert feats[G1].final.roster_gap == {"box": 0.0, "pm": 0.0}  # nobody known yet
    assert set(feats[G2].extras) == set(player_sim.EXTRA_KEYS)
    assert feats[G2].game_id == G2


def test_player_row_dated_differently_from_its_game_is_refused(con):
    build_player_fixture(con)
    add_player_row(con, G3, D3 - timedelta(days=1), 99, "BOS", 30.0, name="Zz Zed")
    with pytest.raises(fit_mod.FitError, match=r"0022400003.*2024-11-04.*2024-11-05|"
                                               r"0022400003.*2024-11-05.*2024-11-04"):
        _run(con)


# -- the replay report lag (plan 3, Task 2) ------------------------------------

def test_report_lag_default_is_15_minutes():
    assert player_sim.REPLAY_REPORT_LAG == timedelta(minutes=15)


def test_report_lag_hides_report_inside_lag(con, monkeypatch):
    build_player_fixture(con)
    _force_ratings(monkeypatch)
    add_report(con, final_at(D3) - timedelta(minutes=10), D3, "BOS", STAR_NAME, "Out")
    lagged, _ = _run_default_lag(con)
    assert lagged[G3].final.injury_gap == {"box": 0.0, "pm": 0.0}
    assert lagged[G3].final.no_report is True
    no_lag, _ = _run(con, report_lag=timedelta(0))
    assert no_lag[G3].final.injury_gap["box"] != 0.0


def test_report_lag_hides_report_inside_lag_at_early_call(con, monkeypatch):
    """The lag applies to the early call too: a report 10 minutes before
    12:30 UTC is hidden at the early call (and seen at the final call)."""
    build_player_fixture(con)
    _force_ratings(monkeypatch)
    add_report(con, early_at(D3) - timedelta(minutes=10), D3, "BOS", STAR_NAME, "Out")
    lagged, stats = _run_default_lag(con)
    assert lagged[G3].early.no_report is True
    assert lagged[G3].early.injury_gap == {"box": 0.0, "pm": 0.0}
    assert lagged[G3].final.injury_gap["box"] != 0.0
    assert stats.no_report["early"][SEASON] == 3
    no_lag, _ = _run(con, report_lag=timedelta(0))
    assert no_lag[G3].early.injury_gap["box"] != 0.0
    assert no_lag[G3].early.report_at == early_at(D3) - timedelta(minutes=10)


def test_report_exactly_at_lag_counts(con, monkeypatch):
    build_player_fixture(con)
    _force_ratings(monkeypatch)
    add_report(con, final_at(D3) - timedelta(minutes=15), D3, "BOS", STAR_NAME, "Out")
    feats, _ = _run_default_lag(con)
    assert feats[G3].final.injury_gap["box"] != 0.0
    assert feats[G3].final.report_at == final_at(D3) - timedelta(minutes=15)
    assert feats[G3].final.fallback is False


def test_fallback_is_judged_against_lagged_time(con):
    """A report 70 minutes before the call is 55 minutes before the lagged
    time, so it is not a fallback; a later report inside the lag that does
    not list the date is not seen, so it does not make it one either."""
    build_player_fixture(con)
    add_report(con, final_at(D3) - timedelta(minutes=70), D3, "BOS", STAR_NAME, "Out")
    add_report(con, final_at(D3) - timedelta(minutes=5), D3 + timedelta(days=1),
               "NYK", "Ivey,Ia", "Out")
    feats, stats = _run_default_lag(con)
    assert feats[G3].final.fallback is False
    assert stats.fallback["final"][SEASON] == 0
    feats0, _ = _run(con)
    assert feats0[G3].final.fallback is True


def test_collect_listings_uses_report_lag(con):
    build_player_fixture(con)
    add_report(con, final_at(D3) - timedelta(minutes=10), D3, "NYK", "Ivey,Ia", "Out")
    games = fit_mod._load(con, VenueIndex.from_db(con))
    history, index = InjuryHistory.from_db(con), player_index(con)
    assert collect_listings(con, games, history, index) == []
    assert collect_listings(con, games, history, index, report_lag=timedelta(0)) == [
        Listing(SEASON, "final", "Out", True),
    ]


# -- ESPN box rows in the live view (plan 3, Task 4) -------------------------

LIVE_SEASON = "2026-27"
E1, E2 = "0022600101", "0022600102"
ED = _date(2026, 11, 2)
AFTER = _date(2026, 11, 4)
SEEN = datetime(2026, 11, 3, 5, 0, tzinfo=UTC)
LATE = datetime(2026, 11, 4, 0, 0, tzinfo=UTC)


def _espn(con, game_id, player_id, *, team="BOS", game_date=ED, minutes=30.0,
          points=12, assists=3, plus_minus=4, did_not_play=False,
          observed_at=SEEN, name="Al Adams", season=LIVE_SEASON):
    table = db.POINT_IN_TIME_TABLES["player_boxscores"]
    con.execute(
        f"INSERT INTO {table} (game_id, season, game_date, team, player_id,"
        " espn_player_id, player_name, starter, did_not_play, minutes, points,"
        " rebounds, assists, plus_minus, source, observed_at)"
        " VALUES (?,?,?,?,?,?,?,TRUE,?,?,?,5,?,?,'espn',?)",
        [game_id, season, game_date, team, str(player_id), "9" + str(player_id),
         name, did_not_play, minutes, points, assists, plus_minus, observed_at],
    )


def _lines_at(con, at, before=AFTER, **kw):
    return PlayerLines.from_view(AsOfView(con, at), before, **kw)


def _rows(lines):
    return [row for day in lines.by_date.values() for row, _ in day]


def test_espn_rows_used_only_when_nba_rows_absent(con):
    add_player_row(con, E1, ED, 1, "BOS", season=LIVE_SEASON, observed_at=SEEN)
    _espn(con, E1, 1, minutes=99.0)
    _espn(con, E2, 2, name="Bo Baker")
    lines = _lines_at(con, LATE)
    rows = _rows(lines)
    g1 = [r for r in rows if r.game_id == E1]
    g2 = [r for r in rows if r.game_id == E2]
    assert [(r.player_id, r.minutes, r.box_complete) for r in g1] == [(1, 30.0, True)]
    assert len(g2) == 1
    r = g2[0]
    assert r.box_complete is False
    assert (r.player_id, r.pts, r.ast, r.plus_minus, r.minutes) == (2, 12, 3, 4.0, 30.0)
    assert r.fga is None and r.tov is None and r.dreb is None
    assert lines.espn == EspnMergeStats(games=1, rows=1, unlinked=0)


def test_espn_rows_flag_off_reads_nba_rows_only(con):
    _espn(con, E2, 2, name="Bo Baker")
    lines = _lines_at(con, LATE, espn=False)
    assert _rows(lines) == []
    assert lines.espn == EspnMergeStats(0, 0, 0)


def test_nba_row_seen_later_replaces_espn_row(con):
    _espn(con, E2, 2, name="Bo Baker", minutes=20.0)
    add_player_row(con, E2, ED, 2, "BOS", minutes=33.0, season=LIVE_SEASON,
                   observed_at=datetime(2026, 11, 3, 12, 0, tzinfo=UTC))
    early = _rows(_lines_at(con, datetime(2026, 11, 3, 6, 0, tzinfo=UTC)))
    assert [(r.minutes, r.box_complete) for r in early] == [(20.0, False)]
    late = _rows(_lines_at(con, LATE))
    assert [(r.minutes, r.box_complete) for r in late] == [(33.0, True)]


def test_espn_rows_respect_view_cutoff(con):
    _espn(con, E2, 2, name="Bo Baker", observed_at=SEEN)
    assert _rows(_lines_at(con, SEEN - timedelta(seconds=1))) == []
    assert len(_rows(_lines_at(con, SEEN))) == 1


def test_espn_rows_respect_before_date(con):
    _espn(con, E2, 2, name="Bo Baker")
    assert _rows(_lines_at(con, LATE, before=ED)) == []


def test_espn_latest_observation_wins(con):
    _espn(con, E2, 2, name="Bo Baker", minutes=30.0, observed_at=SEEN)
    _espn(con, E2, 2, name="Bo Baker", minutes=31.0,
          observed_at=SEEN + timedelta(hours=1))
    rows = _rows(_lines_at(con, LATE))
    assert [r.minutes for r in rows] == [31.0]


def test_unlinked_espn_player_skipped_and_counted(con):
    _espn(con, E2, "espn:3945274", name="New Guy")
    _espn(con, E2, 2, name="Bo Baker")
    lines = _lines_at(con, LATE)
    assert [r.player_id for r in _rows(lines)] == [2]
    assert lines.espn.unlinked == 1
    assert lines.espn.rows == 1


def test_dnp_espn_row_ignored(con):
    _espn(con, E2, 2, name="Bo Baker", did_not_play=True, minutes=None)
    lines = _lines_at(con, LATE)
    assert _rows(lines) == []
    assert lines.espn.rows == 0


def test_espn_season_type_from_game_id(con):
    _espn(con, "0042600101", 1)
    _espn(con, "0022600101", 2, name="Bo Baker")
    _espn(con, "0052600101", 3, name="Cy Cole")
    view = AsOfView(con, LATE)
    got, stats = espn_rows(view, AFTER, set())
    types = {row.game_id: st for row, st in got}
    assert types == {"0042600101": "playoffs", "0022600101": "regular",
                     "0052600101": "playin"}
    assert stats == EspnMergeStats(games=3, rows=3, unlinked=0)
