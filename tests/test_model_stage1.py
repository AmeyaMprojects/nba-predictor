from collections import Counter
from datetime import UTC, date, datetime, time, timedelta

import pytest

from model_fixtures import add_game, fixture_con
from predictor import db
from predictor.asof import AsOfView
from predictor.backtest import replay
from predictor.backtest.baselines import GameToPredict
from predictor.model.adjustments import Coefficients
from predictor.model.ratings import RatingParams, win_probability
from predictor.model.settings import ModelSettings
from predictor.model.stage1 import Breakdown, Stage1Predictor

S = ModelSettings(
    ratings=RatingParams(k=0.1, margin_cap=20.0, season_regression=0.5, hca_window=100),
    coefficients=Coefficients(back_to_back=-2.0, third_in_four=-1.0,
                              travel_per_1000km=-0.5, tz_per_hour=-0.25, altitude=1.5),
    sigma=13.0,
    half_life=None,
    tuning_games=0,
)


def _cutoff(d):  # 30 minutes before a 7pm ET (00:00 UTC next day) tip-off
    return datetime.combine(d + timedelta(days=1), time(0), tzinfo=UTC) - timedelta(minutes=30)


def _game(gid, d, home, away, season="2024-25"):
    return GameToPredict(gid, season, d, home, away)


@pytest.fixture
def con(tmp_path):
    c = fixture_con(tmp_path)
    add_game(c, "0022400001", "2024-25", date(2025, 1, 10), "PHI", "NYK", 110, 100,
             city="Philadelphia")
    add_game(c, "0022400002", "2024-25", date(2025, 1, 12), "NYK", "PHI", 100, 104,
             city="New York")
    add_game(c, "0022400003", "2024-25", date(2025, 1, 15), "PHI", "NYK",
             city="Philadelphia")
    return c


def test_prediction_by_hand(con):
    # Visible at the 01-15 cutoff: both FINALs (stamped 01-11 and 01-13, 12:00 UTC).
    # After game 1: PHI +1, NYK -1, home margins [10].
    # Game 2 (NYK home): predicted = -1 - 1 + 10 = 8; margin -4;
    #   delta = 0.1 * (-4 - 8) = -1.2 -> NYK -2.2, PHI +2.2; margins [10, -4].
    # Game 3: rating = 2.2 - (-2.2) = 4.4; home = mean(10, -4) = 3.0.
    # PHI last played 01-12 (3 days) -> not B2B; NYK same. Travel: PHI
    # New York -> Philadelphia; NYK New York -> Philadelphia: equal, diff 0.
    p = Stage1Predictor(con, S)
    g = _game("0022400003", date(2025, 1, 15), "PHI", "NYK")
    b = p.explain(g, AsOfView(con, _cutoff(date(2025, 1, 15))))
    assert b.rating == pytest.approx(4.4)
    assert b.home == pytest.approx(3.0)
    assert b.rest == pytest.approx(0.0)
    assert b.travel == pytest.approx(0.0)
    assert b.altitude == 0.0
    assert b.spread == pytest.approx(7.4)
    assert b.p_home == pytest.approx(win_probability(7.4, 13.0))


def test_call_returns_the_breakdown_probability_and_records_it(con):
    p = Stage1Predictor(con, S)
    g = _game("0022400003", date(2025, 1, 15), "PHI", "NYK")
    prob = p(g, AsOfView(con, _cutoff(date(2025, 1, 15))))
    assert prob == p.breakdowns["0022400003"].p_home


def test_terms_sum_to_the_spread(con):
    p = Stage1Predictor(con, S)
    b = p.explain(_game("0022400003", date(2025, 1, 15), "PHI", "NYK"),
                  AsOfView(con, _cutoff(date(2025, 1, 15))))
    assert sum(v for _, v in b.terms()) == pytest.approx(b.spread, abs=1e-12)


def test_sentence_by_hand():
    b = Breakdown("g", "DEN", "LAL", rating=4.24, home=2.41, rest=0.84,
                  travel=-0.36, altitude=1.12, spread=8.25, p_home=0.7362)
    assert b.sentence() == (
        "LAL at DEN: rating +4.2, home +2.4, rest +0.8, travel -0.4, "
        "altitude +1.1 -> DEN by 8.1 (DEN 74% to win)"
    )


def test_sentence_for_an_away_favourite():
    b = Breakdown("g", "DEN", "LAL", rating=-6.0, home=2.0, rest=0.0,
                  travel=0.0, altitude=0.0, spread=-4.0, p_home=0.38)
    assert b.sentence().endswith("-> LAL by 4.0 (DEN 38% to win)")


def test_sentence_normalises_negative_zero():
    """t7-fix1 finding 4: round(-0.03, 1) is -0.0, which f"{...:+.1f}" would
    print as "-0.0" -- a term that reads as zero but carries a minus sign
    nobody can explain. Every shown term (and the total) must normalise
    -0.0 to +0.0 instead."""
    b = Breakdown("g", "DEN", "LAL", rating=-0.03, home=0.0, rest=0.0,
                  travel=0.0, altitude=0.0, spread=-0.03, p_home=0.5)
    assert "rating +0.0" in b.sentence()
    assert "-0.0" not in b.sentence()


def test_a_result_one_second_after_the_cutoff_moves_no_rating(tmp_path):
    con = fixture_con(tmp_path)
    cutoff = _cutoff(date(2025, 1, 15))
    add_game(con, "0022400001", "2024-25", date(2025, 1, 14), "PHI", "NYK", 130, 100,
             city="Philadelphia", final_observed_at=cutoff + timedelta(seconds=1))
    add_game(con, "0022400003", "2024-25", date(2025, 1, 15), "PHI", "NYK", city="Philadelphia")
    b = Stage1Predictor(con, S).explain(
        _game("0022400003", date(2025, 1, 15), "PHI", "NYK"), AsOfView(con, cutoff))
    assert b.rating == 0.0 and b.home == 0.0


def test_a_result_observed_exactly_at_the_cutoff_is_applied(tmp_path):
    """AsOfView is inclusive (observed_at <= cutoff); a result stamped
    exactly at the cutoff must already have moved the rating."""
    con = fixture_con(tmp_path)
    cutoff = _cutoff(date(2025, 1, 15))
    add_game(con, "0022400001", "2024-25", date(2025, 1, 14), "PHI", "NYK", 130, 100,
             city="Philadelphia", final_observed_at=cutoff)
    add_game(con, "0022400003", "2024-25", date(2025, 1, 15), "PHI", "NYK", city="Philadelphia")
    b = Stage1Predictor(con, S).explain(
        _game("0022400003", date(2025, 1, 15), "PHI", "NYK"), AsOfView(con, cutoff))
    # margin 30, capped to 20 for the rating update: predicted = 0 (no
    # history), delta = 0.1 * (20 - 0) = 2.0 -> PHI +2.0, NYK -2.0.
    # home_margins holds the UNCAPPED margin: [30].
    assert b.rating == pytest.approx(4.0)
    assert b.home == pytest.approx(30.0)


def test_a_correction_replaces_the_original_result(tmp_path):
    """Two FINAL rows for the same game (an original and a later
    correction) must collapse to the latest-observed one, not both, and
    not whichever the query happens to return first."""
    def with_correction(path):
        con = fixture_con(path)
        add_game(con, "0022400001", "2024-25", date(2025, 1, 10), "PHI", "NYK", 110, 100,
                 city="Philadelphia", final_observed_at=datetime(2025, 1, 11, 12, tzinfo=UTC))
        games = db.POINT_IN_TIME_TABLES["games"]
        con.execute(
            f"INSERT INTO {games} (game_id, season, game_date, home_team, away_team,"
            " home_points, away_points, status, reconstructed, observed_at)"
            " VALUES (?,?,?,?,?,?,?,'FINAL',TRUE,?)",
            ["0022400001", "2024-25", date(2025, 1, 10), "PHI", "NYK", 90, 100,
             datetime(2025, 1, 12, 12, tzinfo=UTC)],
        )
        add_game(con, "0022400003", "2024-25", date(2025, 1, 15), "PHI", "NYK",
                 city="Philadelphia")
        return con

    def only_the_correction(path):
        con = fixture_con(path)
        add_game(con, "0022400001", "2024-25", date(2025, 1, 10), "PHI", "NYK", 90, 100,
                 city="Philadelphia", final_observed_at=datetime(2025, 1, 12, 12, tzinfo=UTC))
        add_game(con, "0022400003", "2024-25", date(2025, 1, 15), "PHI", "NYK",
                 city="Philadelphia")
        return con

    g = _game("0022400003", date(2025, 1, 15), "PHI", "NYK")
    cutoff = _cutoff(date(2025, 1, 15))  # well after both the 01-11 and 01-12 observations
    corrected = with_correction(tmp_path / "a")
    plain = only_the_correction(tmp_path / "b")
    b_corrected = Stage1Predictor(corrected, S).explain(g, AsOfView(corrected, cutoff))
    b_plain = Stage1Predictor(plain, S).explain(g, AsOfView(plain, cutoff))
    assert b_corrected == b_plain


def test_incremental_calls_equal_a_fresh_predictor_when_a_result_arrives_late(tmp_path):
    """A late-arriving result for an EARLIER game_date must not be applied
    after a later game_date already has been -- the same running predictor
    queried twice must land in the same state a fresh one reaches in one
    shot at the same final cutoff."""
    con = fixture_con(tmp_path)
    add_game(con, "0022400001", "2024-25", date(2025, 1, 10), "PHI", "NYK", 110, 100,
             city="Philadelphia", final_observed_at=datetime(2025, 1, 14, 12, tzinfo=UTC))
    add_game(con, "0022400002", "2024-25", date(2025, 1, 12), "NYK", "PHI", 100, 104,
             city="New York", final_observed_at=datetime(2025, 1, 13, 12, tzinfo=UTC))
    add_game(con, "0022400003", "2024-25", date(2025, 1, 15), "PHI", "NYK", city="Philadelphia")

    g = _game("0022400003", date(2025, 1, 15), "PHI", "NYK")
    mid_cutoff = datetime(2025, 1, 13, 20, 0, tzinfo=UTC)  # sees game 2 only
    late_cutoff = _cutoff(date(2025, 1, 15))  # sees both

    incremental = Stage1Predictor(con, S)
    incremental.explain(g, AsOfView(con, mid_cutoff))
    b_incremental = incremental.explain(g, AsOfView(con, late_cutoff))

    fresh = Stage1Predictor(con, S).explain(g, AsOfView(con, late_cutoff))
    assert b_incremental == fresh


def test_same_date_lower_game_id_arriving_later_equals_fresh(tmp_path):
    """Order matters WITHIN a date too: home_court() is a rolling mean of
    every earlier apply's margin, so two games on the same date applied in
    a different relative order move ratings differently. A naive
    incremental catch-up that only compares game_date (not game_id) would
    apply the higher game_id first here (it was visible first) and the
    lower one second, instead of matching a fresh rebuild's ascending
    (game_date, game_id) order.

    Hand check (K=0.1, cap=20): fresh applies 0022400001 (PHI/NYK, margin
    30 capped to 20) before 0022400002 (BOS/MIA, margin -20) --
    predicted=0 both times since home_court starts empty; PHI/NYK apply
    first (delta 0.1*(20-0)=2.0 -> PHI +2.0), then BOS/MIA sees
    home_court=mean([30])=30 (predicted=30, delta=0.1*(-20-30)=-5.0 ->
    BOS -5.0). Applying the higher id FIRST instead flips which apply sees
    the empty vs. the populated home_court, giving BOS -2.0 and PHI +4.0
    -- a different PHI-BOS rating gap (7.0 vs 6.0) for game 3."""
    con = fixture_con(tmp_path)
    add_game(con, "0022400002", "2024-25", date(2025, 1, 10), "BOS", "MIA", 80, 100,
             city="Boston", final_observed_at=datetime(2025, 1, 10, 18, tzinfo=UTC))
    add_game(con, "0022400001", "2024-25", date(2025, 1, 10), "PHI", "NYK", 130, 100,
             city="Philadelphia", final_observed_at=datetime(2025, 1, 11, 12, tzinfo=UTC))
    add_game(con, "0022400003", "2024-25", date(2025, 1, 15), "PHI", "BOS", city="Philadelphia")

    g = _game("0022400003", date(2025, 1, 15), "PHI", "BOS")
    mid_cutoff = datetime(2025, 1, 10, 20, 0, tzinfo=UTC)  # sees only 0022400002
    late_cutoff = _cutoff(date(2025, 1, 15))  # sees both

    incremental = Stage1Predictor(con, S)
    incremental.explain(g, AsOfView(con, mid_cutoff))
    b_incremental = incremental.explain(g, AsOfView(con, late_cutoff))

    fresh = Stage1Predictor(con, S).explain(g, AsOfView(con, late_cutoff))
    assert b_incremental == fresh
    assert fresh.rating == pytest.approx(7.0)


def test_incremental_correction_to_an_already_applied_game_equals_fresh(tmp_path):
    """A correction observed in a LATER call, for a game already applied
    in an EARLIER call, must trigger a rebuild too -- not just a
    correction bundled into the very first catch-up ever made (already
    covered by test_a_correction_replaces_the_original_result)."""
    con = fixture_con(tmp_path)
    add_game(con, "0022400001", "2024-25", date(2025, 1, 10), "PHI", "NYK", 110, 100,
             city="Philadelphia", final_observed_at=datetime(2025, 1, 11, 12, tzinfo=UTC))
    add_game(con, "0022400003", "2024-25", date(2025, 1, 15), "PHI", "NYK", city="Philadelphia")

    g = _game("0022400003", date(2025, 1, 15), "PHI", "NYK")
    first_cutoff = datetime(2025, 1, 12, 0, 0, tzinfo=UTC)  # sees only the original 110-100
    late_cutoff = _cutoff(date(2025, 1, 15))  # sees the correction too

    incremental = Stage1Predictor(con, S)
    incremental.explain(g, AsOfView(con, first_cutoff))

    # a correction to the game already applied above, observed AFTER that call
    games = db.POINT_IN_TIME_TABLES["games"]
    con.execute(
        f"INSERT INTO {games} (game_id, season, game_date, home_team, away_team,"
        " home_points, away_points, status, reconstructed, observed_at)"
        " VALUES (?,?,?,?,?,?,?,'FINAL',TRUE,?)",
        ["0022400001", "2024-25", date(2025, 1, 10), "PHI", "NYK", 90, 100,
         datetime(2025, 1, 13, 12, tzinfo=UTC)],
    )

    b_incremental = incremental.explain(g, AsOfView(con, late_cutoff))
    fresh = Stage1Predictor(con, S).explain(g, AsOfView(con, late_cutoff))
    assert b_incremental == fresh


def test_output_ignores_future_fixtures_and_invisible_results(tmp_path):
    """Closes the harness's 2020 play-in item: fixture EXISTENCE and not-yet-
    visible results must not change a single prediction."""
    def build(path, extra):
        con = fixture_con(path)
        add_game(con, "0022400001", "2024-25", date(2025, 1, 10), "PHI", "NYK", 110, 100,
                 city="Philadelphia")
        add_game(con, "0022400002", "2024-25", date(2025, 1, 12), "NYK", "PHI", 100, 104,
                 city="New York")
        add_game(con, "0022400003", "2024-25", date(2025, 1, 15), "PHI", "NYK", 99, 98,
                 city="Philadelphia")
        if extra:
            # a postseason fixture that already "exists" before the season ends
            add_game(con, "0052400001", "2024-25", date(2025, 4, 15), "PHI", "BOS",
                     city="Philadelphia")
            # a result that becomes visible only far in the future
            add_game(con, "0022400009", "2024-25", date(2025, 1, 11), "BOS", "MIA", 150, 90,
                     city="Boston",
                     final_observed_at=datetime(2030, 1, 1, tzinfo=UTC))
        return con

    plain = build(tmp_path / "a", extra=False)
    noisy = build(tmp_path / "b", extra=True)
    got = []
    for con in (plain, noisy):
        preds, _ = replay.replay(con, Stage1Predictor(con, S))
        got.append([(q.game_id, q.p_home) for q in preds if q.game_id != "0022400009"])
    assert got[0] == got[1]


def test_model_reads_only_the_games_table(con, monkeypatch):
    seen = []
    real_table = AsOfView.table

    def spy(self, name):
        seen.append(name)
        return real_table(self, name)

    monkeypatch.setattr(AsOfView, "table", spy)
    p = Stage1Predictor(con, S)
    p.explain(_game("0022400003", date(2025, 1, 15), "PHI", "NYK"),
              AsOfView(con, _cutoff(date(2025, 1, 15))))
    assert seen == ["games"]


def test_non_final_rows_with_scores_are_never_applied(tmp_path):
    """The `status = 'FINAL'` filter is load-bearing: an IN_PROGRESS row
    can carry a non-NULL score too, and must still be ignored."""
    def build(path, extra):
        con = fixture_con(path)
        add_game(con, "0022400001", "2024-25", date(2025, 1, 10), "PHI", "NYK",
                 city="Philadelphia")
        if extra:
            games = db.POINT_IN_TIME_TABLES["games"]
            con.execute(
                f"INSERT INTO {games} (game_id, season, game_date, home_team,"
                " away_team, home_points, away_points, status, reconstructed,"
                " observed_at) VALUES (?,?,?,?,?,?,?,'IN_PROGRESS',TRUE,?)",
                ["0022400001", "2024-25", date(2025, 1, 10), "PHI", "NYK", 50, 48,
                 datetime(2025, 1, 10, 20, tzinfo=UTC)],
            )
        add_game(con, "0022400003", "2024-25", date(2025, 1, 15), "PHI", "NYK",
                 city="Philadelphia")
        return con

    plain = build(tmp_path / "a", extra=False)
    noisy = build(tmp_path / "b", extra=True)
    g = _game("0022400003", date(2025, 1, 15), "PHI", "NYK")
    cutoff = _cutoff(date(2025, 1, 15))
    b_plain = Stage1Predictor(plain, S).explain(g, AsOfView(plain, cutoff))
    b_noisy = Stage1Predictor(noisy, S).explain(g, AsOfView(noisy, cutoff))
    assert b_plain == b_noisy


def test_going_back_in_time_rebuilds_from_scratch(con):
    later = _cutoff(date(2025, 1, 15))
    earlier = _cutoff(date(2025, 1, 11))
    g = _game("0022400003", date(2025, 1, 15), "PHI", "NYK")
    reused = Stage1Predictor(con, S)
    reused.explain(g, AsOfView(con, later))
    after_rewind = reused.explain(g, AsOfView(con, earlier))
    fresh = Stage1Predictor(con, S).explain(g, AsOfView(con, earlier))
    assert after_rewind == fresh


def test_preseason_results_never_touch_ratings(tmp_path):
    con = fixture_con(tmp_path)
    add_game(con, "0012400001", "2024-25", date(2025, 1, 10), "PHI", "NYK", 150, 90,
             city="Philadelphia")
    add_game(con, "0022400003", "2024-25", date(2025, 1, 15), "PHI", "NYK", city="Philadelphia")
    b = Stage1Predictor(con, S).explain(
        _game("0022400003", date(2025, 1, 15), "PHI", "NYK"),
        AsOfView(con, _cutoff(date(2025, 1, 15))))
    assert b.rating == 0.0


def test_neutral_site_gets_no_home_court(tmp_path):
    con = fixture_con(tmp_path)
    add_game(con, "0022400001", "2024-25", date(2025, 1, 10), "PHI", "NYK", 110, 100,
             city="Philadelphia")
    add_game(con, "0022400003", "2024-25", date(2025, 1, 15), "BOS", "MIA",
             city="Paris", neutral=True)
    b = Stage1Predictor(con, S).explain(
        _game("0022400003", date(2025, 1, 15), "BOS", "MIA"),
        AsOfView(con, _cutoff(date(2025, 1, 15))))
    assert b.home == 0.0


def test_unknown_city_and_missing_history_are_counted(tmp_path):
    con = fixture_con(tmp_path)
    add_game(con, "0022400001", "2024-25", date(2025, 1, 14), "PHI", "NYK", 110, 100,
             city="Atlantis")
    add_game(con, "0022400003", "2024-25", date(2025, 1, 15), "PHI", "BOS", city="Philadelphia")
    p = Stage1Predictor(con, S)
    p.explain(_game("0022400003", date(2025, 1, 15), "PHI", "BOS"),
              AsOfView(con, _cutoff(date(2025, 1, 15))))
    assert p.unknown_cities == Counter({"Atlantis": 1})
    assert p.no_history == 1  # BOS has no earlier game


def test_game_missing_from_schedule_is_counted_not_guessed(tmp_path):
    con = fixture_con(tmp_path)
    games = db.POINT_IN_TIME_TABLES["games"]
    con.execute(
        f"INSERT INTO {games} (game_id, season, game_date, home_team, away_team,"
        " status, reconstructed, observed_at) VALUES ('0022400003','2024-25',"
        " DATE '2025-01-15','PHI','NYK','SCHEDULED',TRUE,?)",
        [datetime(2025, 1, 8, 12, tzinfo=UTC)],
    )
    p = Stage1Predictor(con, S)
    b = p.explain(_game("0022400003", date(2025, 1, 15), "PHI", "NYK"),
                  AsOfView(con, _cutoff(date(2025, 1, 15))))
    assert b.home == 0.0 or b.home == pytest.approx(0.0)
    assert p.unknown_cities["(game not in the schedule)"] == 1


def test_a_prior_season_result_seen_after_the_new_season_began_equals_fresh(tmp_path):
    """A 2024-25 result dated 2025-06-01 is first observed on 2025-10-25,
    after the predictor has already entered 2025-26 (explaining a game on
    2025-10-22). Its key sorts after everything applied, so the old
    trigger applied it incrementally: Ratings switched back to 2024-25 and
    then forward again, regressing every rating twice more. A fresh
    predictor at the same cutoff applies it in order and regresses once.
    They must agree."""
    con = fixture_con(tmp_path)
    add_game(con, "0022400001", "2024-25", date(2025, 4, 1), "PHI", "NYK", 110, 100,
             city="Philadelphia")
    add_game(con, "0042400001", "2024-25", date(2025, 6, 1), "BOS", "PHI", 100, 90,
             city="Boston",
             final_observed_at=datetime(2025, 10, 25, 12, tzinfo=UTC))
    add_game(con, "0022500001", "2025-26", date(2025, 10, 22), "PHI", "NYK",
             city="Philadelphia")
    add_game(con, "0022500002", "2025-26", date(2025, 10, 28), "PHI", "BOS",
             city="Philadelphia")

    first = _game("0022500001", date(2025, 10, 22), "PHI", "NYK", season="2025-26")
    later = _game("0022500002", date(2025, 10, 28), "PHI", "BOS", season="2025-26")

    incremental = Stage1Predictor(con, S)
    incremental.explain(first, AsOfView(con, _cutoff(date(2025, 10, 22))))
    got = incremental.explain(later, AsOfView(con, _cutoff(date(2025, 10, 28))))

    fresh = Stage1Predictor(con, S).explain(later, AsOfView(con, _cutoff(date(2025, 10, 28))))
    assert got == fresh
    # Hand check of the fresh path: game A moves PHI +1 / NYK -1 (k 0.1, margin
    # 10, home court 0); game B (BOS home, home court 10 from A): predicted
    # 0 - 1 + 10 = 9, margin 10, delta 0.1 -> BOS +0.1, PHI 0.9. Entering
    # 2025-26 halves every rating once: PHI 0.45, BOS 0.05.
    assert fresh.rating == pytest.approx(0.45 - 0.05)
