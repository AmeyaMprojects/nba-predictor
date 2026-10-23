"""State directory of append-only JSONL files and the DuckDB rebuilt from it."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from predictor import cli, config, db
from predictor.cloud import state
from predictor.config import Settings


def dt(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)


def _table(con, logical: str) -> str:
    return db.POINT_IN_TIME_TABLES[logical]


def _rows(con, logical: str) -> set[tuple]:
    return set(con.execute(f"SELECT * FROM {_table(con, logical)}").fetchall())


def _game(game_id, season, day, home, away, hp, ap, status, observed, reconstructed=True):
    return (game_id, season, day, home, away, hp, ap, status, reconstructed, observed)


def _sched(game_id, season, day, tip, home, away, observed, arena="Arena", neutral=False):
    return (game_id, season, day, tip, home, away, arena, "City", "ST", neutral, neutral, observed)


def _pg(game_id, season, day, player_id, team, pts, observed, minutes=31.5, plus_minus=4.0):
    return (
        game_id, season, day, "regular", player_id, f"Player {player_id}", team,
        minutes, pts, 8, 15, 2, 5, 3, 4, 1, 6, 5, 1, 0, 2, 3, plus_minus,
        "nba_stats", True, observed,
    )


def _injury(observed, game_date, team, player, status, reason="Injury/Illness-Knee"):
    return (
        observed.astimezone(state.EASTERN).date(), game_date, "07:30(ET)", f"{team}@XXX",
        team, f"{team}Display", player, f"{player} Display", status, reason, False, observed,
    )


def _make_source_db(path: Path):
    con = db.connect(path)
    db.migrate(con)
    g = _table(con, "games")
    con.executemany(
        f"INSERT INTO {g} VALUES (?,?,?,?,?,?,?,?,?,?)",
        [
            _game("0022500001", "2025-26", date(2025, 10, 21), "BOS", "NYK", None, None,
                  "SCHEDULED", dt("2025-10-20T12:00Z")),
            _game("0022500001", "2025-26", date(2025, 10, 21), "BOS", "NYK", 110, 104,
                  "FINAL", dt("2025-10-22T12:00Z")),
            _game("0022600001", "2026-27", date(2026, 10, 21), "LAL", "GSW", 118, 112,
                  "FINAL", dt("2026-10-22T03:41:07.123456Z"), reconstructed=False),
        ],
    )
    s = _table(con, "schedule")
    con.executemany(
        f"INSERT INTO {s} VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            _sched("0022600001", "2026-27", date(2026, 10, 21), dt("2026-10-22T02:00Z"),
                   "LAL", "GSW", dt("2026-09-27T10:00Z")),
            _sched("0022600001", "2026-27", date(2026, 10, 21), dt("2026-10-22T02:00Z"),
                   "LAL", "GSW", dt("2026-09-28T10:00Z")),
            _sched("0022600001", "2026-27", date(2026, 10, 21), dt("2026-10-22T02:30Z"),
                   "LAL", "GSW", dt("2026-09-29T10:00Z")),
            _sched("0022600002", "2026-27", date(2026, 11, 3), None,
                   "BOS", "MIA", dt("2026-09-27T10:00Z"), arena=None),
            _sched("0022500001", "2025-26", date(2025, 10, 21), dt("2025-10-21T23:30Z"),
                   "BOS", "NYK", dt("2025-09-01T10:00Z"), neutral=True),
        ],
    )
    p = _table(con, "player_games")
    con.executemany(
        f"INSERT INTO {p} VALUES ({','.join('?' * 26)})",
        [
            _pg("0022500001", "2025-26", date(2025, 10, 21), 1628369, "BOS", 31,
                dt("2025-10-22T12:00Z")),
            _pg("0022500001", "2025-26", date(2025, 10, 21), 1629027, "NYK", 0,
                dt("2025-10-22T12:00Z"), minutes=None, plus_minus=None),
            _pg("0022600001", "2026-27", date(2026, 10, 21), 2544, "LAL", 27,
                dt("2026-10-22T12:00Z"), minutes=34.25, plus_minus=-3.0),
        ],
    )
    o = _table(con, "odds_snapshots")
    con.executemany(
        f"INSERT INTO {o} VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            ("k1", "sbr", "BOS", "NYK", -150, 130, -3.5, 221.5, dt("2025-10-21T23:30Z"),
             "0022500001", "kaggle_sbr", True),
            ("abc123", "draftkings", "LAL", "GSW", -120, 100, -1.5, 230.0,
             dt("2026-10-21T13:01:02.5Z"), "0022600001", "theoddsapi", False),
            ("abc999", "fanduel", "LAL", "GSW", None, None, None, None,
             dt("2026-10-21T13:01:02.5Z"), None, "theoddsapi", False),
        ],
    )
    i = _table(con, "injury_status")
    r1, r2, r3 = dt("2026-10-20T21:30Z"), dt("2026-10-21T21:30Z"), dt("2026-10-22T21:30Z")
    r_old = dt("2025-10-21T17:00Z")
    gd = date(2026, 10, 22)
    rows = [
        _injury(r_old, date(2025, 10, 21), "BOS", "Tatum,Jayson", "Out"),
        _injury(r1, gd, "LAL", "James,LeBron", "Questionable"),
        _injury(r1, gd, "LAL", "Hayes,Jaxson", "Out"),
        _injury(r1, gd, "GSW", "Curry,Stephen", "Probable"),
        # r2: James changes, Hayes drops, Curry unchanged
        _injury(r2, gd, "LAL", "James,LeBron", "Out"),
        _injury(r2, gd, "GSW", "Curry,Stephen", "Probable"),
        # r3: identical to r2 (an unchanged report still has full rows)
        _injury(r3, gd, "LAL", "James,LeBron", "Out"),
        _injury(r3, gd, "GSW", "Curry,Stephen", "Probable"),
    ]
    rows.append(
        (r3.astimezone(state.EASTERN).date(), date(2026, 10, 23), None, None, "BOS", None,
         "Brown,Jaylen", None, "Doubtful", None, False, r3)
    )
    con.executemany(f"INSERT INTO {i} VALUES ({','.join('?' * 12)})", rows)
    return con


INJURY_COLUMNS = (
    "report_date, game_date, game_time, matchup, team, team_display, player,"
    " player_display, status, reason, reconstructed, observed_at"
)


def _injury_rows(con) -> set[tuple]:
    return set(con.execute(f"SELECT {INJURY_COLUMNS} FROM {_table(con, 'injury_status')}").fetchall())


def _latest_schedule(con) -> dict[str, tuple]:
    rows = con.execute(
        "SELECT game_id, season, game_date, tip_off_utc, home_team, away_team, arena_name,"
        " arena_city, arena_state, is_neutral_reported, is_neutral"
        f" FROM {_table(con, 'schedule')}"
        " QUALIFY row_number() OVER (PARTITION BY game_id ORDER BY observed_at DESC) = 1"
    ).fetchall()
    return {r[0]: r for r in rows}


# --- append_rows / write_snapshot -------------------------------------------


def _sched_row(game_id="0022600002", day="2026-11-03", observed="2026-10-25T13:00Z"):
    return {
        "game_id": game_id, "season": "2026-27", "game_date": date.fromisoformat(day),
        "tip_off_utc": dt("2026-11-04T00:00Z"), "home_team": "BOS", "away_team": "MIA",
        "arena_name": "TD Garden", "arena_city": "Boston", "arena_state": "MA",
        "is_neutral_reported": False, "is_neutral": False, "observed_at": dt(observed),
    }


def test_append_rows_partitions_schedule_by_season_and_month(tmp_path):
    state_dir = tmp_path / "state"
    touched = state.append_rows(state_dir, "schedule", [_sched_row()])
    assert touched == [state_dir / "schedule" / "2026-27" / "2026-11.jsonl"]
    line = touched[0].read_text()
    assert line.endswith("\n") and line.count("\n") == 1
    obj = json.loads(line)
    assert line == json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n"
    assert obj["tip_off_utc"] == "2026-11-04T00:00:00+00:00"
    assert obj["game_date"] == "2026-11-03"


def test_append_never_rewrites_existing_lines(tmp_path):
    state_dir = tmp_path / "state"
    (path,) = state.append_rows(state_dir, "schedule", [_sched_row()])
    first = path.read_bytes()
    state.append_rows(state_dir, "schedule", [_sched_row(observed="2026-10-26T13:00Z")])
    after = path.read_bytes()
    assert after.startswith(first) and len(after) > len(first)
    assert list(state.read_rows(state_dir, "schedule"))[0]["observed_at"] == (
        "2026-10-25T13:00:00+00:00"
    )


def test_append_rows_rejects_naive_timestamps(tmp_path):
    row = _sched_row()
    row["observed_at"] = datetime(2026, 10, 25, 13)
    with pytest.raises(ValueError):
        state.append_rows(tmp_path / "state", "schedule", [row])


def test_write_snapshot_refuses_outside_nba(tmp_path):
    with pytest.raises(ValueError):
        state.write_snapshot(tmp_path / "state", "results/x.jsonl", [{"a": 1}])
    with pytest.raises(ValueError):
        state.write_snapshot(tmp_path / "state", "nba/../results/x.jsonl", [{"a": 1}])
    path = state.write_snapshot(tmp_path / "state", "nba/players-2026-27.jsonl", [{"b": 2, "a": 1}])
    assert path.read_text() == '{"a":1,"b":2}\n'


def test_snapshot_table_form_reads_back_as_dicts(tmp_path):
    state_dir = tmp_path / "state"
    path = state.write_snapshot(
        state_dir, "nba/games-2026-27.jsonl",
        [{"game_id": "0022600001", "season": "2026-27", "observed_at": dt("2026-10-22T12:00Z")}],
        columns=("game_id", "observed_at", "season"),
    )
    assert path.read_text() == (
        '{"columns":["game_id","observed_at","season"],"format":"table"}\n'
        '["0022600001","2026-10-22T12:00:00+00:00","2026-27"]\n'
    )
    assert list(state.read_rows(state_dir, "nba_games")) == [
        {"game_id": "0022600001", "observed_at": "2026-10-22T12:00:00+00:00", "season": "2026-27"}
    ]


def test_families_have_the_planned_keys():
    for name in ("results", "schedule", "odds", "boxscores", "nba_games", "nba_schedule"):
        assert state.FAMILIES[name].name == name


# --- export / rebuild ----------------------------------------------------------


def test_rebuild_roundtrip_equals_source_db(tmp_path):
    src = _make_source_db(tmp_path / "src.duckdb")
    state_dir = tmp_path / "state"
    state.export_from_db(src, state_dir, state.seasons_in_db(src))
    report = state.rebuild_db(state_dir, tmp_path / "work.duckdb")
    print(f"\nrebuild took {report.seconds:.3f}s for {sum(report.rows.values())} rows")
    out = db.connect(tmp_path / "work.duckdb", read_only=True)
    assert _rows(out, "games") == _rows(src, "games")
    assert _latest_schedule(out) == _latest_schedule(src)
    assert report.rows["games"] == 3


def test_roundtrip_player_games_is_exact(tmp_path):
    src = _make_source_db(tmp_path / "src.duckdb")
    state_dir = tmp_path / "state"
    state.export_from_db(src, state_dir, ["2025-26", "2026-27"])
    assert (state_dir / "nba" / "player_games" / "2026-27" / "2026-10.jsonl").exists()
    report = state.rebuild_db(state_dir, tmp_path / "work.duckdb")
    out = db.connect(tmp_path / "work.duckdb", read_only=True)
    assert _rows(out, "player_games") == _rows(src, "player_games")
    assert report.rows["player_games"] == 3


def test_roundtrip_odds_is_exact(tmp_path):
    src = _make_source_db(tmp_path / "src.duckdb")
    state_dir = tmp_path / "state"
    state.export_from_db(src, state_dir, ["2025-26", "2026-27"])
    state.rebuild_db(state_dir, tmp_path / "work.duckdb")
    out = db.connect(tmp_path / "work.duckdb", read_only=True)
    assert _rows(out, "odds_snapshots") == _rows(src, "odds_snapshots")


def test_roundtrip_injury_status_is_exact(tmp_path):
    src = _make_source_db(tmp_path / "src.duckdb")
    state_dir = tmp_path / "state"
    state.export_from_db(src, state_dir, ["2025-26", "2026-27"])
    report = state.rebuild_db(state_dir, tmp_path / "work.duckdb")
    out = db.connect(tmp_path / "work.duckdb", read_only=True)
    assert _injury_rows(out) == _injury_rows(src)
    assert report.rows["injury_status"] == 9


def test_injury_history_is_stored_as_deltas(tmp_path):
    src = _make_source_db(tmp_path / "src.duckdb")
    state_dir = tmp_path / "state"
    state.export_from_db(src, state_dir, ["2026-27"])
    lines = [
        json.loads(x)
        for x in (state_dir / "nba" / "injuries" / "2026-27" / "2026-10.jsonl").read_text().splitlines()
    ]
    kinds = [x["kind"] for x in lines]
    # r1: report + 3 sets; r2: report + set James + drop Hayes; r3: report + set Brown
    assert kinds == [
        "report", "set", "set", "set",
        "report", "set", "drop",
        "report", "set",
    ]
    assert lines[6] == {"kind": "drop", "game_date": "2026-10-22", "team": "LAL",
                        "player": "Hayes,Jaxson"}


def test_injury_month_files_are_self_contained(tmp_path):
    """A month file starts from an empty report, so one month's file can be
    re-exported without making the next month's deltas wrong."""
    con = db.connect(tmp_path / "src.duckdb")
    db.migrate(con)
    i = _table(con, "injury_status")
    a, b = dt("2026-10-31T21:30Z"), dt("2026-11-01T21:30Z")
    con.executemany(
        f"INSERT INTO {i} VALUES ({','.join('?' * 12)})",
        [_injury(a, date(2026, 11, 1), "LAL", "James,LeBron", "Out"),
         _injury(b, date(2026, 11, 1), "LAL", "James,LeBron", "Out")],
    )
    state_dir = tmp_path / "state"
    state.export_from_db(con, state_dir, ["2026-27"])
    nov = (state_dir / "nba" / "injuries" / "2026-27" / "2026-11.jsonl").read_text().splitlines()
    assert [json.loads(x)["kind"] for x in nov] == ["report", "set"]
    state.rebuild_db(state_dir, tmp_path / "work.duckdb")
    out = db.connect(tmp_path / "work.duckdb", read_only=True)
    assert _injury_rows(out) == _injury_rows(con)


def test_reexport_is_byte_identical(tmp_path):
    src = _make_source_db(tmp_path / "src.duckdb")
    first, second = tmp_path / "a", tmp_path / "b"
    paths_a = state.export_from_db(src, first, ["2025-26", "2026-27"])
    state.export_from_db(src, second, ["2025-26", "2026-27"])
    rel = sorted(p.relative_to(first) for p in paths_a)
    assert rel == sorted(p.relative_to(second) for p in second.rglob("*.jsonl"))
    for r in rel:
        assert (first / r).read_bytes() == (second / r).read_bytes()


def test_schedule_export_drops_unchanged_vintages(tmp_path):
    src = _make_source_db(tmp_path / "src.duckdb")
    state_dir = tmp_path / "state"
    state.export_from_db(src, state_dir, ["2026-27"])
    rows = list(state.read_rows(state_dir, "nba_schedule"))
    lal = [r for r in rows if r["game_id"] == "0022600001"]
    assert [r["observed_at"] for r in lal] == [
        "2026-09-27T10:00:00+00:00", "2026-09-29T10:00:00+00:00",
    ]


def test_schedule_as_of_any_cutoff_is_unchanged_by_export(tmp_path):
    src = _make_source_db(tmp_path / "src.duckdb")
    state_dir = tmp_path / "state"
    state.export_from_db(src, state_dir, ["2025-26", "2026-27"])
    state.rebuild_db(state_dir, tmp_path / "work.duckdb")
    out = db.connect(tmp_path / "work.duckdb", read_only=True)

    def as_of(con, cutoff):
        return set(con.execute(
            "SELECT game_id, tip_off_utc, arena_name FROM "
            f"{_table(con, 'schedule')} WHERE observed_at <= ?"
            " QUALIFY row_number() OVER (PARTITION BY game_id ORDER BY observed_at DESC) = 1",
            [cutoff],
        ).fetchall())

    for cutoff in ("2026-09-27T12:00Z", "2026-09-28T12:00Z", "2026-09-30T00:00Z"):
        assert as_of(out, dt(cutoff)) == as_of(src, dt(cutoff))


def test_rebuild_maps_provisional_ids(tmp_path):
    state_dir = tmp_path / "state"
    state.append_rows(state_dir, "results", [{
        "game_id": "espn:401705555", "season": "2026-27", "game_date": date(2027, 4, 18),
        "home_team": "BOS", "away_team": "MIA", "home_points": 101, "away_points": 99,
        "status": "FINAL", "reconstructed": False, "observed_at": dt("2027-04-19T02:40Z"),
        "source": "espn",
    }])
    links = state_dir / "schedule" / "id_links.jsonl"
    links.parent.mkdir(parents=True, exist_ok=True)
    links.write_text(json.dumps({
        "provisional_id": "espn:401705555", "game_id": "0042600101",
        "linked_at": "2027-04-19T03:00:00+00:00",
    }) + "\n")
    assert state.read_id_links(state_dir) == {"espn:401705555": "0042600101"}
    state.rebuild_db(state_dir, tmp_path / "work.duckdb")
    out = db.connect(tmp_path / "work.duckdb", read_only=True)
    ids = [r[0] for r in out.execute(f"SELECT game_id FROM {_table(out, 'games')}").fetchall()]
    assert ids == ["0042600101"]


def test_cloud_families_roundtrip(tmp_path):
    state_dir = tmp_path / "state"
    result = {
        "game_id": "0022600003", "season": "2026-27", "game_date": date(2026, 10, 24),
        "home_team": "NYK", "away_team": "CHI", "home_points": 99, "away_points": 101,
        "status": "FINAL", "reconstructed": False, "observed_at": dt("2026-10-25T03:00Z"),
        "source": "espn",
    }
    odds = {
        "game_key": "e1", "book": "draftkings", "home_team": "NYK", "away_team": "CHI",
        "home_price": -110, "away_price": -110, "spread": -1.0, "total": 225.5,
        "observed_at": dt("2026-10-24T13:00:01.25Z"), "game_id": "0022600003",
        "source": "theoddsapi", "reconstructed": False,
    }
    box = {
        "game_id": "espn:401700001", "season": "2026-27", "game_date": date(2026, 10, 24),
        "team": "NYK", "espn_player_id": "3945274", "player_name": "Somebody",
        "starter": True, "did_not_play": False, "minutes": 34.0, "points": 27,
        "rebounds": 5, "assists": 4, "plus_minus": 12, "source": "espn",
        "observed_at": dt("2026-10-25T04:00Z"),
    }
    state.append_rows(state_dir, "results", [result])
    state.append_rows(state_dir, "odds", [odds])
    state.append_rows(state_dir, "boxscores", [box])
    state.append_rows(state_dir, "schedule", [_sched_row()])
    report = state.rebuild_db(state_dir, tmp_path / "work.duckdb")
    assert report.rows["games"] == 1 and report.rows["odds_snapshots"] == 1
    assert report.rows["player_boxscores"] == 1 and report.rows["schedule"] == 1
    out = db.connect(tmp_path / "work.duckdb", read_only=True)
    (g,) = out.execute(f"SELECT * FROM {_table(out, 'games')}").fetchall()
    assert g[0] == "0022600003" and g[5:9] == (99, 101, "FINAL", False)
    assert g[9] == dt("2026-10-25T03:00Z")
    (o,) = out.execute(f"SELECT observed_at, spread FROM {_table(out, 'odds_snapshots')}").fetchall()
    assert o == (dt("2026-10-24T13:00:01.25Z"), -1.0)
    (b,) = out.execute(
        f"SELECT player_id, espn_player_id, plus_minus FROM {_table(out, 'player_boxscores')}"
    ).fetchall()
    assert b == ("espn:3945274", "3945274", 12)
    assert list(state.read_rows(state_dir, "results"))[0]["source"] == "espn"


def test_rebuild_starts_from_a_fresh_file(tmp_path):
    target = tmp_path / "work.duckdb"
    con = db.connect(target)
    con.execute("CREATE TABLE junk (x INTEGER)")
    con.close()
    state.rebuild_db(tmp_path / "state", target)
    out = db.connect(target, read_only=True)
    tables = {r[0] for r in out.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    assert "junk" not in tables
    assert db.POINT_IN_TIME_TABLES["games"] in tables


def test_rebuild_time_for_thirteen_seasons(tmp_path):
    """Synthetic 2014-15..2026-27 state; the time is printed, not asserted."""
    state_dir = tmp_path / "state"
    games = []
    for year in range(2014, 2027):
        season = f"{year}-{str(year + 1)[-2:]}"
        start = date(year, 10, 25)
        for n in range(1230):
            day = start + timedelta(days=n // 7)
            games.append({
                "game_id": f"002{str(year)[-2:]}{n:05d}", "season": season, "game_date": day,
                "home_team": "BOS", "away_team": "NYK", "home_points": 100,
                "away_points": 90, "status": "FINAL", "reconstructed": True,
                "observed_at": datetime(day.year, day.month, day.day, 12, tzinfo=UTC)
                + timedelta(days=1),
            })
    by_season: dict[str, list] = {}
    for g in games:
        by_season.setdefault(g["season"], []).append(g)
    for season, rows in by_season.items():
        state.write_snapshot(state_dir, f"nba/games-{season}.jsonl", rows)
    report = state.rebuild_db(state_dir, tmp_path / "work.duckdb")
    print(f"\nrebuild of {report.rows['games']} games rows took {report.seconds:.2f}s")
    assert report.rows["games"] == 13 * 1230


def test_settings_override_used_when_present(tmp_path):
    from predictor.model import settings as model_settings

    state_dir = tmp_path / "state"
    assert state.settings_path(state_dir) == model_settings.SETTINGS_PATH
    override = state_dir / "settings" / "stage1_settings.json"
    override.parent.mkdir(parents=True)
    override.write_text("{}")
    assert state.settings_path(state_dir) == override


def test_export_state_cli_prints_counts(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path / "data")
    s.data_dir.mkdir()
    monkeypatch.setattr(config, "settings", s)
    monkeypatch.setattr(db, "settings", s)
    _make_source_db(s.db_path).close()
    monkeypatch.setattr(cli, "_now", lambda: dt("2026-10-23T12:00Z"))
    out_dir = tmp_path / "out" / "state"
    result = CliRunner().invoke(cli.app, ["export-state", "--to", str(out_dir)])
    assert result.exit_code == 0, result.output
    files = sorted(out_dir.rglob("*.jsonl"))
    lines = sum(len(p.read_text().splitlines()) for p in files)
    assert f"wrote {lines} rows to {len(files)} file(s)" in result.output
    assert (out_dir / "nba" / "games-2026-27.jsonl").exists()
    assert (out_dir / "nba" / "games-2025-26.jsonl").exists()

    result = CliRunner().invoke(
        cli.app, ["export-state", "--to", str(tmp_path / "all"), "--seasons", "all"]
    )
    assert result.exit_code == 0, result.output

    result = CliRunner().invoke(
        cli.app, ["export-state", "--to", str(tmp_path / "x"), "--seasons", "bogus"]
    )
    assert result.exit_code != 0
