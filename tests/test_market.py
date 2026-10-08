"""Market probability from stored odds lines (market-odds Task 3)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from model_fixtures import fixture_con
from predictor import db
from predictor.model import market
from predictor.model.market import OddsLine

T0 = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
T1 = datetime(2025, 1, 1, 1, 0, tzinfo=UTC)
T2 = datetime(2025, 1, 1, 2, 0, tzinfo=UTC)


def line(home=None, away=None, spread=None, at=T0, book="b"):
    return OddsLine(book, home, away, spread, at)


# --- arithmetic, checked by hand ---------------------------------------

def test_american_favourite():
    # -200: stake 200 to win 100 -> 200 / 300
    assert market.american_to_prob(-200) == pytest.approx(2 / 3)


def test_american_underdog():
    # +170: stake 100 to win 170 -> 100 / 270 = 0.370370...
    assert market.american_to_prob(170) == pytest.approx(100 / 270)
    assert round(market.american_to_prob(170), 5) == 0.37037


def test_american_even_money():
    assert market.american_to_prob(100) == 0.5
    assert market.american_to_prob(-100) == 0.5


@pytest.mark.parametrize("bad", [0, 50, -50])
def test_american_rejects_impossible_prices(bad):
    with pytest.raises(ValueError):
        market.american_to_prob(bad)


def test_devig_by_hand():
    # 0.666667 / (0.666667 + 0.370370) = 0.666667 / 1.037037 = 0.642857
    p = market.devig(market.american_to_prob(-200), market.american_to_prob(170))
    assert round(p, 4) == 0.6429
    assert p == pytest.approx(0.64286, abs=5e-6)


def test_devig_sums_to_one():
    h, a = market.american_to_prob(-200), market.american_to_prob(170)
    assert market.devig(h, a) + market.devig(a, h) == pytest.approx(1.0)


# --- market_p_home -----------------------------------------------------

def test_single_moneyline_line():
    v = market.market_p_home([line(-200, 170, -5.5)], sigma=12.0)
    assert v.p_home == pytest.approx(0.642857, abs=1e-6)
    assert v.spread == -5.5
    assert v.books == 1
    assert v.observed_at == T0
    assert v.from_spread is False


def test_median_of_three_books_and_latest_time():
    # even money -> 0.5; -200/+170 -> 0.642857; -300/+250 -> 0.75/(0.75+0.285714)=0.724138
    lines = [
        line(100, -100, at=T0, book="a"),
        line(-200, 170, at=T2, book="b"),
        line(-300, 250, at=T1, book="c"),
    ]
    v = market.market_p_home(lines, sigma=12.0)
    assert v.p_home == pytest.approx(0.642857, abs=1e-6)
    assert v.books == 3
    assert v.observed_at == T2
    assert v.spread is None


def test_median_of_two_books_is_their_average():
    v = market.market_p_home(
        [line(100, -100, book="a"), line(-200, 170, book="b")], sigma=12.0
    )
    assert v.p_home == pytest.approx((0.5 + 0.642857) / 2, abs=1e-6)


def test_spread_only_lines_ignored_when_any_line_has_moneylines():
    lines = [line(-200, 170, at=T0, book="a"), line(spread=-20.0, at=T2, book="b")]
    v = market.market_p_home(lines, sigma=12.0)
    assert v.p_home == pytest.approx(0.642857, abs=1e-6)
    assert v.books == 1
    assert v.observed_at == T0  # the spread-only line was not used
    assert v.spread == -20.0  # but its spread is still reported
    assert v.from_spread is False


def test_one_sided_moneyline_falls_back_to_spread():
    v = market.market_p_home([line(-200, None, spread=0.0)], sigma=12.0)
    assert v.from_spread is True
    assert v.p_home == pytest.approx(0.5)


def test_spread_sign_home_favoured_by_hand():
    """Home -12 with sigma 12: home expected to win by 12 points,
    Phi(12/12) = Phi(1) = 0.8413 -- home favoured, NOT 0.1587."""
    v = market.market_p_home([line(spread=-12.0)], sigma=12.0)
    assert v.from_spread is True
    assert round(v.p_home, 4) == 0.8413


def test_spread_sign_away_favoured_by_hand():
    """Home +5.5 with sigma 12: Phi(-5.5/12) = Phi(-0.4583) = 1 - 0.6766 = 0.3234."""
    v = market.market_p_home([line(spread=5.5)], sigma=12.0)
    assert round(v.p_home, 3) == 0.323


def test_median_of_spreads():
    lines = [line(spread=-12.0, book="a"), line(spread=0.0, book="b"), line(spread=-24.0, book="c")]
    v = market.market_p_home(lines, sigma=12.0)
    assert round(v.p_home, 4) == 0.8413
    assert v.spread == -12.0
    assert v.books == 3


def test_nothing_usable_is_none():
    assert market.market_p_home([], sigma=12.0) is None
    assert market.market_p_home([line(), line(-110, None)], sigma=12.0) is None


# --- reading stored lines ----------------------------------------------

def _insert(con, game_key, book, observed_at, *, game_id, source, home=-200, away=170,
            spread=-5.5):
    table = db.POINT_IN_TIME_TABLES["odds_snapshots"]
    con.execute(
        f"INSERT INTO {table} (game_key, book, home_team, away_team, home_price,"
        " away_price, spread, total, observed_at, game_id, source, reconstructed)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [game_key, book, "BOS", "NYK", home, away, spread, 220.0, observed_at,
         game_id, source, source == "kaggle_sbr"],
    )


def test_historical_lines_reads_only_kaggle_rows_with_a_game(tmp_path):
    con = fixture_con(tmp_path)
    _insert(con, "kaggle:g1", "consensus", T0, game_id="g1", source="kaggle_sbr",
            home=None, away=None, spread=-3.0)
    _insert(con, "kaggle:g2", "consensus", T0, game_id="g2", source="kaggle_sbr")
    _insert(con, "ev1", "fanduel", T0, game_id="g1", source="theoddsapi")
    _insert(con, "kaggle:x", "consensus", T0, game_id=None, source="kaggle_sbr")
    out = market.historical_lines(con)
    assert set(out) == {"g1", "g2"}
    assert out["g1"] == [OddsLine("consensus", None, None, -3.0, out["g1"][0].observed_at)]
    assert out["g2"][0].home_price == -200


def test_historical_lines_empty_table(tmp_path):
    assert market.historical_lines(fixture_con(tmp_path)) == {}


def test_live_lines_latest_per_book_at_or_before_now(tmp_path):
    con = fixture_con(tmp_path)
    _insert(con, "ev1", "fanduel", T0, game_id="g1", source="theoddsapi", home=-150)
    _insert(con, "ev1", "fanduel", T1, game_id="g1", source="theoddsapi", home=-160)
    _insert(con, "ev1", "fanduel", T2, game_id="g1", source="theoddsapi", home=-999)
    _insert(con, "ev1", "draftkings", T0, game_id="g1", source="theoddsapi", home=-140)
    _insert(con, "ev2", "fanduel", T0, game_id="g2", source="theoddsapi")
    _insert(con, "kaggle:g1", "consensus", T0, game_id="g1", source="kaggle_sbr")
    got = market.live_lines(con, "g1", now=T1)
    assert [(ln.book, ln.home_price) for ln in got] == [("draftkings", -140), ("fanduel", -160)]
    assert all(ln.observed_at <= T1 for ln in got)


def test_live_lines_nothing_before_now(tmp_path):
    con = fixture_con(tmp_path)
    _insert(con, "ev1", "fanduel", T2, game_id="g1", source="theoddsapi")
    assert market.live_lines(con, "g1", now=T1) == []


def test_live_lines_game_id_is_a_bound_value_not_sql(tmp_path):
    con = fixture_con(tmp_path)
    _insert(con, "ev1", "fanduel", T0, game_id="g'1", source="theoddsapi")
    _insert(con, "ev2", "fanduel", T0, game_id="g2", source="theoddsapi")
    assert [ln.book for ln in market.live_lines(con, "g'1", now=T1)] == ["fanduel"]
    # An injection-shaped id matches nothing rather than everything.
    assert market.live_lines(con, "x' OR '1'='1", now=T1) == []


def test_the_model_never_reads_odds():
    """Market reads live only in market.py / evaluate.py / live.py; the
    model's own fitting and prediction code never mentions odds or market."""
    from pathlib import Path

    import predictor.model as pkg

    root = Path(pkg.__file__).parent
    for name in ("stage1.py", "fit.py", "tuning.py", "ratings.py", "adjustments.py",
                 "settings.py"):
        src = (root / name).read_text()
        assert "odds_snapshots" not in src, name
        assert "market" not in src, name
