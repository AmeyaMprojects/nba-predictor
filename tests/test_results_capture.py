"""Tests for `sources/results.py` (live result capture) and the
`capture-results` CLI command.

Mirrors tests/test_schedule_ingest.py / tests/test_schedule_cli.py's
settings-monkeypatching pattern: `config.settings`, `db.settings` and
`raw_store.settings` are all pointed at a throwaway `tmp_path` so no test
ever touches the real, irreplaceable archive, and no real network call is
made -- `fetch=` is always a stub returning an in-test DataFrame.
"""

from __future__ import annotations

import gzip
from datetime import UTC, date, datetime, timedelta

import duckdb
import pandas as pd
import pytest
import requests
from typer.testing import CliRunner

from predictor import cli, config, db, raw_store
from predictor.asof import AsOfView
from predictor.backtest import replay
from predictor.backtest.baselines import fixed_probability
from predictor.config import Settings
from predictor.sources import nba_stats, results
from schedule_rows import insert_schedule_row

runner = CliRunner()

FETCHED = datetime(2026, 10, 4, 13, 0, tzinfo=UTC)
SEASON = "2026-27"


def _row(game_id, game_date, team, matchup, pts):
    return {
        "GAME_ID": game_id,
        "GAME_DATE": game_date,
        "MATCHUP": matchup,
        "TEAM_ABBREVIATION": team,
        "PTS": pts,
    }


def _game_df(game_id, game_date, home, away, home_pts, away_pts):
    """Two LeagueGameFinder-shaped rows (home + away) for one game."""
    return pd.DataFrame(
        [
            _row(game_id, game_date, home, f"{home} vs. {away}", home_pts),
            _row(game_id, game_date, away, f"{away} @ {home}", away_pts),
        ]
    )


def _malformed_df(game_id, game_date):
    """Three team-rows for one GAME_ID -- `pair_team_rows` cannot resolve a
    home/away pairing from this and drops the group entirely."""
    return pd.DataFrame(
        [
            _row(game_id, game_date, "PHI", "PHI vs. NYK", 119),
            _row(game_id, game_date, "NYK", "NYK @ PHI", 110),
            _row(game_id, game_date, "BOS", "BOS @ PHI", 100),
        ]
    )


@pytest.fixture
def con(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(config, "settings", s)
    monkeypatch.setattr(db, "settings", s)
    monkeypatch.setattr(raw_store, "settings", s)
    c = db.connect(tmp_path / "t.duckdb")
    db.migrate(c)
    return c


def _insert_final(con, game_id, game_date, home, away, home_pts, away_pts, observed_at):
    table = db.POINT_IN_TIME_TABLES["games"]
    con.execute(
        f"INSERT INTO {table} (game_id, season, game_date, home_team, away_team,"
        " home_points, away_points, status, reconstructed, observed_at)"
        " VALUES (?,?,?,?,?,?,?,'FINAL',TRUE,?)",
        [game_id, SEASON, game_date, home, away, home_pts, away_pts, observed_at],
    )


# --- download(): raw-first archiving and parsing -----------------------


def test_download_archives_the_exact_fetched_payload(con):
    df = _game_df("0022600001", "2026-10-03", "PHI", "NYK", 119, 110)
    downloaded = results.download(SEASON, fetched_at=FETCHED, fetch=lambda s: df)

    assert downloaded.season == SEASON
    assert downloaded.fetched_at == FETCHED
    assert downloaded.blob_key == f"{SEASON}_{FETCHED:%Y%m%dT%H%M%S}Z.json.gz"

    expected_payload = df.to_json(orient="split", date_format="iso").encode()
    archived = gzip.decompress(raw_store.load(results.SOURCE, downloaded.blob_key))
    assert archived == expected_payload


def test_download_parses_the_archived_bytes_not_the_fetch_result(con, monkeypatch):
    # Raw-first, proven: what is parsed is what raw_store.load returns, not
    # whatever `fetch` happened to return.
    real_df = _game_df("0022600001", "2026-10-03", "PHI", "NYK", 119, 110)
    other_df = _game_df("0022600099", "2026-10-03", "BOS", "MIA", 100, 90)
    real_payload = real_df.to_json(orient="split", date_format="iso").encode()
    monkeypatch.setattr(raw_store, "load", lambda source, key: gzip.compress(real_payload))

    downloaded = results.download(SEASON, fetched_at=FETCHED, fetch=lambda s: other_df)

    assert [g.game_id for g in downloaded.games] == ["0022600001"]


def test_download_preserves_full_game_id_through_the_json_round_trip(con):
    # A purely-numeric GAME_ID string must not come back as an int missing
    # its leading zeros.
    df = _game_df("0022600001", "2026-10-03", "PHI", "NYK", 119, 110)
    downloaded = results.download(SEASON, fetched_at=FETCHED, fetch=lambda s: df)
    assert downloaded.games[0].game_id == "0022600001"


# --- download(): keeps only FINAL games ---------------------------------


def test_unplayed_game_is_not_in_downloaded_games(con):
    df = _game_df("0022600002", "2026-10-05", "BOS", "MIA", None, None)
    downloaded = results.download(SEASON, fetched_at=FETCHED, fetch=lambda s: df)
    assert downloaded.games == []


# --- download(): unpairable (dropped) games are never silently lost -----


def test_dropped_games_are_recorded_and_logged(con, capsys):
    df = _malformed_df("0022600005", "2026-10-07")
    downloaded = results.download(SEASON, fetched_at=FETCHED, fetch=lambda s: df)

    assert downloaded.dropped == ["0022600005"]
    assert downloaded.games == []
    assert "nba_stats: DROPPED" in capsys.readouterr().out


def test_dropped_games_survive_into_captureresult(con):
    df = _malformed_df("0022600005", "2026-10-07")
    downloaded = results.download(SEASON, fetched_at=FETCHED, fetch=lambda s: df)

    result = results.load(con, downloaded)

    assert result.dropped == ["0022600005"]
    assert result.new_finals == 0


# --- download(): a partial (one-sided) score is a loud anomaly, not a ---
# --- silently-ignored "not yet played" game -----------------------------


def test_partial_score_prints_a_loud_warning_and_is_excluded(con, capsys):
    df = _game_df("0022600006", "2026-10-08", "BOS", "MIA", 100, None)
    downloaded = results.download(SEASON, fetched_at=FETCHED, fetch=lambda s: df)

    assert downloaded.games == []
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "0022600006" in out


# --- load(): new finished game -------------------------------------------


def test_new_finished_game_creates_exactly_one_final_row(con):
    df = _game_df("0022600001", "2026-10-03", "PHI", "NYK", 119, 110)
    downloaded = results.download(SEASON, fetched_at=FETCHED, fetch=lambda s: df)

    result = results.load(con, downloaded)

    assert result.new_finals == 1
    assert result.already_known == 0
    assert result.blob_key == downloaded.blob_key

    table = db.POINT_IN_TIME_TABLES["games"]
    rows = con.execute(
        f"SELECT observed_at, reconstructed, home_points, away_points FROM {table} "
        "WHERE game_id = ? AND status = 'FINAL'",
        ["0022600001"],
    ).fetchall()
    assert rows == [(FETCHED, False, 119, 110)]


def test_unplayed_game_adds_nothing(con):
    df = _game_df("0022600002", "2026-10-05", "BOS", "MIA", None, None)
    downloaded = results.download(SEASON, fetched_at=FETCHED, fetch=lambda s: df)

    result = results.load(con, downloaded)

    assert result.new_finals == 0
    assert result.already_known == 0
    table = db.POINT_IN_TIME_TABLES["games"]
    assert con.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0


# --- load(): idempotency ---------------------------------------------------


def test_running_twice_adds_nothing_the_second_time(con):
    df = _game_df("0022600001", "2026-10-03", "PHI", "NYK", 119, 110)
    first = results.load(con, results.download(SEASON, fetched_at=FETCHED, fetch=lambda s: df))
    assert first.new_finals == 1

    second = results.load(
        con,
        results.download(SEASON, fetched_at=FETCHED + timedelta(days=1), fetch=lambda s: df),
    )

    assert second.new_finals == 0
    assert second.already_known == 1
    table = db.POINT_IN_TIME_TABLES["games"]
    count = con.execute(
        f"SELECT count(*) FROM {table} WHERE game_id = ? AND status = 'FINAL'",
        ["0022600001"],
    ).fetchone()[0]
    assert count == 1


def test_game_already_final_from_historical_ingest_is_not_duplicated(con):
    # Simulates a FINAL row already written by nba_stats.ingest_season's
    # reconstructed historical backfill.
    _insert_final(
        con, "0022600001", date(2026, 10, 3), "PHI", "NYK", 119, 110,
        FETCHED - timedelta(days=400),
    )
    df = _game_df("0022600001", "2026-10-03", "PHI", "NYK", 119, 110)

    result = results.load(con, results.download(SEASON, fetched_at=FETCHED, fetch=lambda s: df))

    assert result.new_finals == 0
    assert result.already_known == 1
    table = db.POINT_IN_TIME_TABLES["games"]
    count = con.execute(f"SELECT count(*) FROM {table} WHERE game_id = ?", ["0022600001"]).fetchone()[0]
    assert count == 1


# --- load(): all-or-nothing -------------------------------------------------


def test_load_rolls_back_on_error(con):
    df = _game_df("0022600001", "2026-10-03", "PHI", "NYK", 119, 110)
    downloaded = results.download(SEASON, fetched_at=FETCHED, fetch=lambda s: df)
    con.close()  # force every statement inside load() to fail

    with pytest.raises(duckdb.Error):
        results.load(con, downloaded)


def test_load_rolls_back_earlier_games_when_a_later_one_fails(con):
    # A good game followed by one with a NOT-NULL violation (home_team is
    # NULL) must leave NEITHER written -- "one transaction: all or nothing".
    good = nba_stats.GameRow(
        game_id="0022600001", season=SEASON, game_date=date(2026, 10, 3),
        home_team="PHI", away_team="NYK", home_points=119, away_points=110,
        status="FINAL",
    )
    bad = nba_stats.GameRow(
        game_id="0022600002", season=SEASON, game_date=date(2026, 10, 3),
        home_team=None, away_team="MIA", home_points=100, away_points=90,
        status="FINAL",
    )
    downloaded = results.Downloaded(
        season=SEASON, fetched_at=FETCHED, blob_key="k", games=[good, bad], dropped=[]
    )

    with pytest.raises(duckdb.Error):
        results.load(con, downloaded)

    table = db.POINT_IN_TIME_TABLES["games"]
    assert con.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0


# --- leak safety: AsOfView never sees the result before its real capture time --


def test_asofview_does_not_see_the_final_before_its_capture_time(con):
    df = _game_df("0022600001", "2026-10-03", "PHI", "NYK", 119, 110)
    results.load(con, results.download(SEASON, fetched_at=FETCHED, fetch=lambda s: df))

    before = AsOfView(con, FETCHED - timedelta(seconds=1))
    finals_before = (
        before.table("games")
        .filter("game_id = '0022600001' AND status = 'FINAL'")
        .fetchall()
    )
    assert finals_before == []

    at = AsOfView(con, FETCHED)
    finals_at = (
        at.table("games")
        .filter("game_id = '0022600001' AND status = 'FINAL'")
        .fetchall()
    )
    assert len(finals_at) == 1


# --- replay integration: a captured result must actually be predicted ---
# --- (the Fix-round-1 regression: a SCHEDULED stub stamped at capture    ---
# --- time, i.e. after tip-off, made replay's earliest-SCHEDULED sanity   ---
# --- bound fire backwards and skip every live-captured game) ------------


def test_captured_result_is_predicted_by_replay(con):
    game_id = "0022600007"
    game_date = date(2026, 10, 9)
    tip = datetime(2026, 10, 9, 23, 0, tzinfo=UTC)
    insert_schedule_row(con, game_id, game_date, "PHI", "NYK", tip, season=SEASON)

    df = _game_df(game_id, "2026-10-09", "PHI", "NYK", 119, 110)
    captured_at = tip + timedelta(hours=3)  # the game ends, then capture runs
    results.load(con, results.download(SEASON, fetched_at=captured_at, fetch=lambda s: df))

    preds, stats = replay.replay(con, fixed_probability(0.6))

    assert stats.predicted == 1
    assert stats.skipped_buffer_too_early == 0
    assert stats.skipped_result_visible == 0
    assert len(preds) == 1
    assert preds[0].game_id == game_id


# --- CLI: capture-results ----------------------------------------------------


def _point_settings_at_tmp(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(config, "settings", s)
    monkeypatch.setattr(db, "settings", s)
    monkeypatch.setattr(raw_store, "settings", s)
    return s


def test_cli_success_reports_counts(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)
    df = _game_df("0022600001", "2026-10-03", "PHI", "NYK", 119, 110)
    original_download = results.download
    monkeypatch.setattr(
        results, "download",
        lambda season: original_download(season, fetched_at=FETCHED, fetch=lambda s: df),
    )

    out = runner.invoke(cli.app, ["capture-results", "--season", SEASON])

    assert out.exit_code == 0, out.output
    assert f"results {SEASON}: 1 new game result(s) recorded (0 already known)" in out.output


def test_cli_default_season_is_the_current_one(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)
    # Pinned, not the wall clock: cli._now() is the single indirection point
    # the command uses to pick a default season.
    fixed_now = datetime(2027, 3, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr(cli, "_now", lambda: fixed_now)
    seen = []

    def fake(season):
        seen.append(season)
        return results.Downloaded(
            season=season, fetched_at=FETCHED, blob_key="k", games=[], dropped=[]
        )

    monkeypatch.setattr(results, "download", fake)
    out = runner.invoke(cli.app, ["capture-results"])
    assert out.exit_code == 0, out.output
    assert seen == [config.season_label(fixed_now)]


def test_cli_download_failure_is_plain_english_and_db_is_never_opened(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)

    def fail(season):
        raise requests.ConnectionError("Name or service not known")

    monkeypatch.setattr(results, "download", fail)
    calls = []
    monkeypatch.setattr(db, "connect_with_retry", lambda *a, **k: calls.append("connect"))

    out = runner.invoke(cli.app, ["capture-results", "--season", SEASON])

    assert out.exit_code == 1
    assert "Could not download results" in out.output
    assert "nothing was saved" in out.output
    assert calls == []
    assert "Traceback" not in out.output


def test_cli_database_error_while_loading_is_plain_english(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)

    def fake(season):
        return results.Downloaded(
            season=season, fetched_at=FETCHED, blob_key="k", games=[], dropped=[]
        )

    def broken_load(con, downloaded):
        raise duckdb.ConstraintException("NOT NULL constraint failed")

    monkeypatch.setattr(results, "download", fake)
    monkeypatch.setattr(results, "load", broken_load)

    out = runner.invoke(cli.app, ["capture-results", "--season", SEASON])

    assert out.exit_code == 1
    assert f"Could not save the {SEASON} results to the database" in out.output
    assert "Traceback" not in out.output


def test_cli_raw_store_conflict_is_plain_english(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)

    def conflict(season):
        raise raw_store.RawStoreConflict(f"key {season}_x.json.gz already holds different bytes")

    monkeypatch.setattr(results, "download", conflict)
    out = runner.invoke(cli.app, ["capture-results", "--season", SEASON])

    assert out.exit_code == 1
    assert "already archived" in out.output
    assert "Traceback" not in out.output


def test_cli_exits_nonzero_and_warns_when_games_were_dropped(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)
    df = _malformed_df("0022600005", "2026-10-07")
    original_download = results.download
    monkeypatch.setattr(
        results, "download",
        lambda season: original_download(season, fetched_at=FETCHED, fetch=lambda s: df),
    )

    out = runner.invoke(cli.app, ["capture-results", "--season", SEASON])

    assert out.exit_code == 1
    assert "WARNING" in out.output
    assert "0022600005" in out.output
    # The success line (0 new results) still prints -- the WARNING is in
    # addition to it, not instead of it.
    assert f"results {SEASON}: 0 new game result(s) recorded (0 already known)" in out.output
