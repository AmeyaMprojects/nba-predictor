import gzip
import json
from datetime import UTC, date, datetime, timedelta

import pytest

from predictor import config, db, raw_store
from predictor.config import Settings
from predictor.sources import schedule

FETCHED = datetime(2026, 9, 27, 5, 0, tzinfo=UTC)


def _game(game_id, date_est, utc, home="PHI", away="NYK"):
    return {
        "gameId": game_id, "gameStatus": 3, "gameStatusText": "Final",
        "gameDateEst": f"{date_est}T00:00:00Z", "gameDateTimeUTC": utc,
        "arenaName": "Wells Fargo Center", "arenaCity": "Philadelphia",
        "arenaState": "PA", "isNeutral": False,
        "homeTeam": {"teamTricode": home, "score": 119},
        "awayTeam": {"teamTricode": away, "score": 110},
    }


def _payload(games, season="2024-25"):
    return json.dumps(
        {"leagueSchedule": {"seasonYear": season, "gameDates": [{"games": games}]}}
    ).encode("utf-8")


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


def _insert_game(con, game_id, game_date, home, away, season="2024-25"):
    g = db.POINT_IN_TIME_TABLES["games"]
    con.execute(
        f"INSERT INTO {g} (game_id, season, game_date, home_team, away_team,"
        " home_points, away_points, status, reconstructed, observed_at)"
        " VALUES (?,?,?,?,?,NULL,NULL,'SCHEDULED',TRUE,?)",
        [game_id, season, game_date, home, away, FETCHED - timedelta(days=400)],
    )


def test_ingest_archives_first_then_writes_rows_at_the_fetch_time(con):
    payload = _payload([_game("0022400561", "2025-01-15", "2025-01-16T00:00:00Z")])
    result = schedule.ingest_season(con, "2024-25", fetched_at=FETCHED, fetch=lambda s: payload)

    assert result.written == 1
    assert result.blob_key == "2024-25_20260927T050000Z.json.gz"
    archived = raw_store.load(schedule.SOURCE, result.blob_key)
    assert gzip.decompress(archived) == payload

    table = db.POINT_IN_TIME_TABLES["schedule"]
    rows = con.execute(
        f"SELECT game_id, game_date, tip_off_utc, observed_at FROM {table}"
    ).fetchall()
    assert rows == [
        ("0022400561", date(2025, 1, 15), datetime(2025, 1, 16, tzinfo=UTC), FETCHED)
    ]


def test_each_fetch_adds_a_new_vintage(con):
    first = _payload([_game("0022400561", "2025-01-15", "2025-01-16T00:00:00Z")])
    moved = _payload([_game("0022400561", "2025-01-15", "2025-01-15T22:30:00Z")])
    schedule.ingest_season(con, "2024-25", fetched_at=FETCHED, fetch=lambda s: first)
    schedule.ingest_season(
        con, "2024-25", fetched_at=FETCHED + timedelta(days=1), fetch=lambda s: moved
    )
    table = db.POINT_IN_TIME_TABLES["schedule"]
    tips = con.execute(
        f"SELECT tip_off_utc FROM {table} ORDER BY observed_at"
    ).fetchall()
    assert [t[0].hour for t in tips] == [0, 22]


def test_naive_fetch_time_is_rejected(con):
    with pytest.raises(ValueError, match="timezone-aware"):
        schedule.ingest_season(
            con, "2024-25", fetched_at=datetime(2026, 9, 27), fetch=lambda s: b""
        )


def test_unparseable_payload_is_still_archived(con):
    with pytest.raises(ValueError):
        schedule.ingest_season(con, "2024-25", fetched_at=FETCHED, fetch=lambda s: b"garbage")
    key = schedule.archive_key("2024-25", FETCHED)
    assert gzip.decompress(raw_store.load(schedule.SOURCE, key)) == b"garbage"


def test_agreeing_games_table_produces_no_mismatch(con):
    _insert_game(con, "0022400561", date(2025, 1, 15), "PHI", "NYK")
    payload = _payload([_game("0022400561", "2025-01-15", "2025-01-16T00:00:00Z")])
    result = schedule.ingest_season(con, "2024-25", fetched_at=FETCHED, fetch=lambda s: payload)
    assert result.mismatches == []


def test_date_disagreement_with_games_table_is_reported_loudly(con, capsys):
    _insert_game(con, "0022400561", date(2025, 1, 14), "PHI", "NYK")
    payload = _payload([_game("0022400561", "2025-01-15", "2025-01-16T00:00:00Z")])
    result = schedule.ingest_season(con, "2024-25", fetched_at=FETCHED, fetch=lambda s: payload)
    assert len(result.mismatches) == 1
    assert "0022400561" in result.mismatches[0]
    assert "2025-01-14" in result.mismatches[0] and "2025-01-15" in result.mismatches[0]
    assert "schedule: MISMATCH" in capsys.readouterr().out


def test_regular_season_game_missing_from_schedule_is_reported(con):
    _insert_game(con, "0022400999", date(2025, 1, 20), "BOS", "LAL")
    payload = _payload([_game("0022400561", "2025-01-15", "2025-01-16T00:00:00Z")])
    result = schedule.ingest_season(con, "2024-25", fetched_at=FETCHED, fetch=lambda s: payload)
    assert any("0022400999" in m and "not in the schedule" in m for m in result.mismatches)


# --- Final-review Fix 1: the network half never touches the database ---


def test_fetch_and_archive_needs_no_database(con):
    payload = _payload([_game("0022400561", "2025-01-15", "2025-01-16T00:30:00Z")])
    fetched = schedule.fetch_and_archive("2024-25", fetched_at=FETCHED, fetch=lambda s: payload)
    assert fetched.season == "2024-25"
    assert fetched.fetched_at == FETCHED
    assert fetched.blob_key == schedule.archive_key("2024-25", FETCHED)
    assert [r.game_id for r in fetched.parsed.rows] == ["0022400561"]
    assert gzip.decompress(raw_store.load(schedule.SOURCE, fetched.blob_key)) == payload
    # Nothing was loaded: that is load()'s job.
    table = db.POINT_IN_TIME_TABLES["schedule"]
    assert con.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0


def test_fetch_and_archive_parses_the_archived_bytes(con, monkeypatch):
    # Raw-first: what is parsed is what was archived, read back from disk.
    real_payload = _payload([_game("0022400561", "2025-01-15", "2025-01-16T00:30:00Z")])
    monkeypatch.setattr(raw_store, "load", lambda source, key: gzip.compress(real_payload))
    fetched = schedule.fetch_and_archive(
        "2024-25", fetched_at=FETCHED, fetch=lambda s: _payload([])
    )
    assert [r.game_id for r in fetched.parsed.rows] == ["0022400561"]


def test_fetch_and_archive_rejects_a_naive_fetch_time():
    with pytest.raises(ValueError, match="timezone-aware"):
        schedule.fetch_and_archive("2024-25", fetched_at=datetime(2026, 9, 27), fetch=lambda s: b"")


def test_load_writes_rows_and_reports(con):
    payload = _payload([_game("0022400561", "2025-01-15", "2025-01-16T00:30:00Z")])
    fetched = schedule.fetch_and_archive("2024-25", fetched_at=FETCHED, fetch=lambda s: payload)
    result = schedule.load(con, fetched)
    assert result.written == 1
    assert result.blob_key == fetched.blob_key
    table = db.POINT_IN_TIME_TABLES["schedule"]
    assert con.execute(f"SELECT observed_at FROM {table}").fetchone()[0] == FETCHED


def test_load_is_all_or_nothing(con):
    good = schedule.ScheduleRow(
        game_id="0022400001", season="2024-25", game_date=date(2025, 1, 15),
        tip_off_utc=None, home_team="PHI", away_team="NYK", arena_name=None,
        arena_city=None, arena_state=None, is_neutral_reported=False, is_neutral=False,
    )
    bad = schedule.ScheduleRow(
        game_id="0022400002", season="2024-25", game_date=date(2025, 1, 16),
        tip_off_utc=None, home_team=None, away_team="NYK", arena_name=None,
        arena_city=None, arena_state=None, is_neutral_reported=False, is_neutral=False,
    )
    fetched = schedule.Fetched(
        season="2024-25", fetched_at=FETCHED, blob_key="k",
        parsed=schedule.ParseResult(rows=[good, bad], no_tipoff=[], undetermined=[]),
    )
    with pytest.raises(Exception):
        schedule.load(con, fetched)
    table = db.POINT_IN_TIME_TABLES["schedule"]
    assert con.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0


# --- Final-review Fix 2: an unpublished season is a plain condition, not a crash ---


def test_unpublished_season_maps_to_schedule_unavailable_without_retry(monkeypatch):
    calls = []

    def unpublished(**kwargs):
        calls.append(kwargs)
        raise IndexError("list index out of range")

    monkeypatch.setattr(schedule.scheduleleaguev2, "ScheduleLeagueV2", unpublished)
    with pytest.raises(schedule.ScheduleUnavailable, match="has not published a 2027-28"):
        schedule.fetch_season_payload("2027-28")
    assert len(calls) == 1


def test_missing_key_in_nba_api_also_maps_to_schedule_unavailable(monkeypatch):
    calls = []

    def unpublished(**kwargs):
        calls.append(kwargs)
        raise KeyError("resultSets")

    monkeypatch.setattr(schedule.scheduleleaguev2, "ScheduleLeagueV2", unpublished)
    with pytest.raises(schedule.ScheduleUnavailable):
        schedule.fetch_season_payload("2027-28")
    assert len(calls) == 1


def test_network_errors_are_still_retried(monkeypatch):
    import requests

    calls = []

    def down(**kwargs):
        calls.append(kwargs)
        raise requests.ConnectionError("down")

    monkeypatch.setattr(schedule.scheduleleaguev2, "ScheduleLeagueV2", down)
    monkeypatch.setattr(schedule.fetch_season_payload.retry, "sleep", lambda s: None)
    with pytest.raises(requests.ConnectionError):
        schedule.fetch_season_payload("2024-25")
    assert len(calls) == 4
