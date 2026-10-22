from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from predictor import db
from predictor.asof import AsOfView
from predictor.model import availability as av
from predictor.model.availability import (
    AvailabilityTable,
    InjuryHistory,
    Listing,
    UnknownStatusCounter,
    estimate,
)
from predictor.sources import injury_report

FIXTURE = Path(__file__).parent / "fixtures" / "Injury-Report_2025-01-15_05PM.pdf"
D = date(2025, 1, 15)


def _listings(season, call, status, n, played):
    return [Listing(season, call, status, i < played) for i in range(n)]


def test_constants_match_the_brief():
    assert av.STATUS_PRIORS == {
        "Out": 0.0, "Doubtful": 0.2, "Questionable": 0.5, "Probable": 0.9, "Available": 1.0,
    }
    assert av.UNKNOWN_STATUS_PRIOR == 0.5
    assert av.MIN_SAMPLES == 100


def test_estimate_uses_only_earlier_seasons():
    rows = _listings("2021-22", "final", "Questionable", 120, 72) + _listings(
        "2022-23", "final", "Questionable", 500, 500
    )
    table = estimate(rows, "2022-23", "final")
    assert table.prob("Questionable") == pytest.approx(0.6)
    assert table.samples["Questionable"] == 120


def test_estimate_excludes_later_seasons_too():
    rows = _listings("2023-24", "final", "Questionable", 500, 500)
    table = estimate(rows, "2022-23", "final")
    assert table.prob("Questionable") == 0.5
    assert table.samples["Questionable"] == 0


def test_estimate_falls_back_to_prior_below_min_samples():
    rows = _listings("2021-22", "final", "Doubtful", 30, 30)
    table = estimate(rows, "2022-23", "final")
    assert table.prob("Doubtful") == 0.2
    assert table.samples["Doubtful"] == 30


def test_estimate_at_exactly_min_samples_uses_the_share():
    rows = _listings("2021-22", "final", "Doubtful", 100, 25)
    assert estimate(rows, "2022-23", "final").prob("Doubtful") == pytest.approx(0.25)


def test_estimate_separates_call_types():
    rows = _listings("2021-22", "early", "Out", 400, 400) + _listings(
        "2021-22", "final", "Out", 150, 3
    )
    final = estimate(rows, "2022-23", "final")
    assert final.prob("Out") == pytest.approx(0.02)
    assert final.samples["Out"] == 150
    early = estimate(rows, "2022-23", "early")
    assert early.prob("Out") == pytest.approx(1.0)
    assert early.samples["Out"] == 400


def test_estimate_ignores_unknown_statuses_in_listings():
    rows = _listings("2021-22", "final", "Not With Team", 200, 200)
    table = estimate(rows, "2022-23", "final")
    assert table.prob("Not With Team") == av.UNKNOWN_STATUS_PRIOR
    assert "Not With Team" not in table.probs


def test_estimate_rejects_a_bad_call_type():
    with pytest.raises(ValueError):
        estimate([], "2022-23", "late")


def test_listing_rejects_non_bool_played():
    with pytest.raises(TypeError):
        Listing("2021-22", "final", "Out", float("nan"))
    with pytest.raises(ValueError):
        Listing("2021-22", "noon", "Out", True)


def test_not_listed_is_available():
    table = estimate([], "2022-23", "final")
    assert table.prob(None) == 1.0
    # Every known status falls back to its prior with no history.
    for status, prior in av.STATUS_PRIORS.items():
        assert table.prob(status) == prior


def test_unknown_status_uses_questionable_prior_and_is_counted():
    table = estimate([], "2022-23", "final")
    assert table.prob("Not With Team") == 0.5
    counter = UnknownStatusCounter()
    assert counter.record("2022-23", "Not With Team") is True
    assert counter.record("2022-23", "Out") is False
    assert counter.record("2022-23", None) is False
    assert counter.report() == {"2022-23": {"Not With Team": 1}}
    assert counter.total("2022-23") == 1
    assert counter.total("2021-22") == 0


def test_nan_status_is_unknown_not_available():
    table = estimate([], "2022-23", "final")
    assert table.prob(float("nan")) == av.UNKNOWN_STATUS_PRIOR
    counter = UnknownStatusCounter()
    assert counter.record("2022-23", float("nan")) is True
    assert counter.total("2022-23") == 1


def test_table_prob_never_returns_nan():
    table = AvailabilityTable(probs={"Out": float("nan")}, samples={})
    assert table.prob("Out") == av.STATUS_PRIORS["Out"]


# --- injury history -------------------------------------------------------


@pytest.fixture
def con(tmp_path):
    c = db.connect(tmp_path / "t.duckdb")
    db.migrate(c)
    return c


def _insert(con, observed_at, game_date, team, player, status):
    table = db.POINT_IN_TIME_TABLES["injury_status"]
    con.execute(
        f"INSERT INTO {table} (report_date, game_date, team, player, status, observed_at)"
        " VALUES (?,?,?,?,?,?)",
        [observed_at.date(), game_date, team, player, status, observed_at],
    )


def test_snapshot_at_never_sees_a_later_report(con):
    t1 = datetime(2025, 1, 15, 22, 30, tzinfo=UTC)
    t2 = datetime(2025, 1, 15, 23, 30, tzinfo=UTC)
    _insert(con, t1, D, "BOS", "Tatum,Jayson", "Questionable")
    _insert(con, t2, D, "BOS", "Tatum,Jayson", "Out")
    hist = InjuryHistory.from_db(con)
    early = hist.snapshot_at(D, datetime(2025, 1, 15, 23, 29, 59, tzinfo=UTC))
    assert early.observed_at == t1
    assert early.statuses == {("BOS", "Tatum,Jayson"): "Questionable"}
    late = hist.snapshot_at(D, t2)
    assert late.observed_at == t2
    assert late.statuses == {("BOS", "Tatum,Jayson"): "Out"}
    assert hist.snapshot_at(D, datetime(2025, 1, 15, 22, 29, tzinfo=UTC)) is None


def test_snapshot_skips_reports_without_that_game_date(con):
    t1 = datetime(2025, 1, 15, 20, 0, tzinfo=UTC)
    t2 = datetime(2025, 1, 16, 4, 0, tzinfo=UTC)
    _insert(con, t1, D, "BOS", "Tatum,Jayson", "Questionable")
    _insert(con, t2, date(2025, 1, 16), "BOS", "Tatum,Jayson", "Out")
    hist = InjuryHistory.from_db(con)
    snap = hist.snapshot_at(D, datetime(2025, 1, 16, 5, 0, tzinfo=UTC))
    assert snap.observed_at == t1
    assert snap.statuses == {("BOS", "Tatum,Jayson"): "Questionable"}
    assert hist.snapshot_at(date(2025, 1, 17), datetime(2025, 1, 18, tzinfo=UTC)) is None


def test_snapshot_at_rejects_naive_time(con):
    hist = InjuryHistory.from_db(con)
    with pytest.raises(ValueError):
        hist.snapshot_at(D, datetime(2025, 1, 15, 23, 0))


def test_snapshot_filters_rows_to_game_date(con):
    injury_report.ingest_report(con, FIXTURE.read_bytes())
    table = db.POINT_IN_TIME_TABLES["injury_status"]
    dates = {r[0] for r in con.execute(f"SELECT DISTINCT game_date FROM {table}").fetchall()}
    assert D in dates and len(dates) >= 2  # a genuine two-date report
    expected = {
        (team, player): status
        for team, player, status in con.execute(
            f"SELECT team, player, status FROM {table} WHERE game_date = ?", [D]
        ).fetchall()
    }
    total = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    hist = InjuryHistory.from_db(con)
    snap = hist.snapshot_at(D, datetime(2025, 1, 16, 12, 0, tzinfo=UTC))
    assert snap is not None
    assert snap.statuses == expected
    assert 0 < len(snap.statuses) < total


def test_from_view_respects_the_view_cutoff(con):
    t1 = datetime(2025, 1, 15, 22, 30, tzinfo=UTC)
    t2 = datetime(2025, 1, 15, 23, 30, tzinfo=UTC)
    _insert(con, t1, D, "BOS", "Tatum,Jayson", "Questionable")
    _insert(con, t2, D, "BOS", "Tatum,Jayson", "Out")
    hist = InjuryHistory.from_view(AsOfView(con, datetime(2025, 1, 15, 23, 0, tzinfo=UTC)))
    snap = hist.snapshot_at(D, datetime(2025, 1, 16, tzinfo=UTC))
    assert snap.observed_at == t1
    assert snap.statuses == {("BOS", "Tatum,Jayson"): "Questionable"}
