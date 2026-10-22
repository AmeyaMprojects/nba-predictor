"""Player game logs (nba_api LeagueGameLog, player mode), stored point-in-time.

No test touches the network (fetch is injected) or the real data directory
(settings are pointed at tmp_path).
"""

from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from predictor import cli, config, db, raw_store
from predictor.asof import AsOfView
from predictor.config import Settings
from predictor.sources import player_games

runner = CliRunner()

FETCHED = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)

COLUMNS = [
    "SEASON_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GAME_ID",
    "GAME_DATE", "MATCHUP", "MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
    "OREB", "DREB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "PLUS_MINUS",
]


def _row(game_id="0022300001", game_date="2024-01-15", player_id=201939,
         name="Stephen Curry", team="GSW", minutes=34, plus_minus=7, pts=30):
    return [
        "22023", player_id, name, team, game_id, game_date, f"{team} vs. LAL",
        minutes, 10, 20, 5, 11, 5, 5, 1, 4, 6, 2, 0, 3, 2, pts, plus_minus,
    ]


def _frame(*rows):
    return pd.DataFrame([list(r) for r in rows] or [_row()], columns=COLUMNS)


def _fetch_of(frame):
    def fetch(season, season_type):
        return frame
    return fetch


@pytest.fixture
def settings(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(config, "settings", s)
    monkeypatch.setattr(db, "settings", s)
    monkeypatch.setattr(raw_store, "settings", s)
    return s


@pytest.fixture
def con(tmp_path, settings):
    c = db.connect(tmp_path / "t.duckdb")
    db.migrate(c)
    return c


def _add_game(con, game_id, game_date=date(2024, 1, 15)):
    con.execute(
        f"INSERT INTO {db.POINT_IN_TIME_TABLES['games']} (game_id, season, game_date,"
        " home_team, away_team, status, observed_at) VALUES (?,?,?,?,?,?,?)",
        [game_id, "2023-24", game_date, "GSW", "LAL", "SCHEDULED",
         datetime(2024, 1, 8, 12, tzinfo=UTC)],
    )


def _count(con):
    table = db.POINT_IN_TIME_TABLES["player_games"]
    return con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def test_player_games_is_a_point_in_time_table(con):
    assert db.POINT_IN_TIME_TABLES["player_games"] == "player_games_raw"
    view = AsOfView(con, datetime(2024, 1, 20, tzinfo=UTC))
    assert len(view.latest("player_games").fetchall()) == 0


def test_migrate_creates_player_games_idempotently(con):
    db.migrate(con)
    db.migrate(con)
    table = db.POINT_IN_TIME_TABLES["player_games"]
    cols = {r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()}
    assert {"plus_minus", "source", "reconstructed", "season_type"} <= cols


def test_download_archives_before_parsing(settings, monkeypatch):
    calls = []
    real_load = raw_store.load

    def recording_load(source, key):
        calls.append((source, key))
        return real_load(source, key)

    monkeypatch.setattr(raw_store, "load", recording_load)
    downloaded = player_games.download(
        "2023-24", "Regular Season", FETCHED, fetch=_fetch_of(_frame())
    )
    assert downloaded.blob_key == "2023-24_regular-season_20261022T120000Z.json.gz"
    assert calls == [(player_games.SOURCE, downloaded.blob_key)]
    assert len(downloaded.rows) == 1


def test_archive_key_for_preseason_and_playin():
    assert (
        player_games.archive_key("2024-25", "Pre Season", FETCHED)
        == "2024-25_pre-season_20261022T120000Z.json.gz"
    )
    assert (
        player_games.archive_key("2024-25", "PlayIn", FETCHED)
        == "2024-25_playin_20261022T120000Z.json.gz"
    )


def test_parse_keeps_leading_zero_game_id(settings):
    downloaded = player_games.download(
        "2023-24", "Regular Season", FETCHED, fetch=_fetch_of(_frame(_row()))
    )
    row = downloaded.rows[0]
    assert row.game_id == "0022300001"
    assert row.game_date == date(2024, 1, 15)
    assert row.player_id == 201939
    assert row.team == "GSW"
    assert row.minutes == 34.0
    assert row.pts == 30
    assert row.plus_minus == 7.0


def test_parse_turns_missing_numbers_into_none(settings):
    downloaded = player_games.download(
        "2023-24", "Regular Season", FETCHED,
        fetch=_fetch_of(_frame(_row(minutes=None, plus_minus=None, pts=None))),
    )
    row = downloaded.rows[0]
    assert row.minutes is None
    assert row.plus_minus is None
    assert row.pts is None


def test_parse_reads_clock_style_minutes(settings):
    downloaded = player_games.download(
        "2023-24", "Regular Season", FETCHED,
        fetch=_fetch_of(_frame(_row(minutes="34:30"))),
    )
    assert downloaded.rows[0].minutes == pytest.approx(34.5)


def test_load_stamps_day_after_noon_utc(con, settings):
    _add_game(con, "0022300001")
    downloaded = player_games.download(
        "2023-24", "Regular Season", FETCHED, fetch=_fetch_of(_frame())
    )
    result = player_games.load(con, downloaded)
    assert result.rows_written == 1
    assert result.games == 1
    assert result.unknown_game_ids == []
    table = db.POINT_IN_TIME_TABLES["player_games"]
    observed_at, reconstructed, source, season_type = con.execute(
        f"SELECT observed_at, reconstructed, source, season_type FROM {table}"
    ).fetchone()
    assert observed_at == datetime(2024, 1, 16, 12, tzinfo=UTC)
    assert reconstructed is True
    assert source == "nba_stats"
    assert season_type == "regular"


def test_preseason_logs_are_tagged_preseason(con, settings):
    downloaded = player_games.download(
        "2024-25", "Pre Season", FETCHED,
        fetch=_fetch_of(_frame(_row(game_id="0012400001", game_date="2024-10-05"))),
    )
    player_games.load(con, downloaded)
    table = db.POINT_IN_TIME_TABLES["player_games"]
    assert con.execute(f"SELECT season_type FROM {table}").fetchone()[0] == "preseason"


def test_preseason_is_fetched_but_not_a_rating_season_type():
    assert player_games.SEASON_TYPES == ("Regular Season", "PlayIn", "Playoffs")
    assert player_games.ALL_SEASON_TYPES[0] == "Pre Season"
    assert set(player_games.SEASON_TYPES) < set(player_games.ALL_SEASON_TYPES)


def test_load_is_idempotent(con, settings):
    downloaded = player_games.download(
        "2023-24", "Regular Season", FETCHED,
        fetch=_fetch_of(_frame(_row(), _row(player_id=2544, name="LeBron James", team="LAL"))),
    )
    player_games.load(con, downloaded)
    before = _count(con)
    player_games.load(con, downloaded)
    assert before == 2
    assert _count(con) == before


def test_unknown_game_id_is_stored_and_reported(con, settings):
    _add_game(con, "0022300001")
    downloaded = player_games.download(
        "2023-24", "Regular Season", FETCHED,
        fetch=_fetch_of(_frame(_row(), _row(game_id="0062300001"))),
    )
    result = player_games.load(con, downloaded)
    assert result.unknown_game_ids == ["0062300001"]
    assert result.games == 2
    table = db.POINT_IN_TIME_TABLES["player_games"]
    assert con.execute(
        f"SELECT count(*) FROM {table} WHERE game_id = '0062300001'"
    ).fetchone()[0] == 1


def test_load_rolls_back_on_error(con, settings, monkeypatch):
    downloaded = player_games.download(
        "2023-24", "Regular Season", FETCHED,
        fetch=_fetch_of(_frame(_row(), _row(player_id=2544, name="LeBron James"))),
    )
    bad = player_games.Downloaded(
        season=downloaded.season, season_type=downloaded.season_type,
        fetched_at=downloaded.fetched_at, blob_key=downloaded.blob_key,
        rows=[downloaded.rows[0], None],
    )
    with pytest.raises(Exception):
        player_games.load(con, bad)
    assert _count(con) == 0


def test_view_hides_logs_before_observed_at(con, settings):
    downloaded = player_games.download(
        "2023-24", "Regular Season", FETCHED, fetch=_fetch_of(_frame())
    )
    player_games.load(con, downloaded)
    before = AsOfView(con, datetime(2024, 1, 16, 11, 59, 59, tzinfo=UTC))
    at = AsOfView(con, datetime(2024, 1, 16, 12, 0, 0, tzinfo=UTC))
    assert len(before.latest("player_games").fetchall()) == 0
    assert len(at.latest("player_games").fetchall()) == 1


def test_cli_download_failure_never_opens_db(settings, monkeypatch):
    connects = []
    monkeypatch.setattr(db, "connect_with_retry", lambda *a, **k: connects.append(1))

    def broken(season, season_type, fetched_at=None):
        raise RuntimeError("stats.nba.com timed out")

    monkeypatch.setattr(player_games, "download", broken)
    result = runner.invoke(cli.app, ["ingest-player-logs", "--season", "2023-24"])
    assert result.exit_code == 1
    assert result.output.startswith("Could not download player game logs")
    assert "2023-24" in result.output
    assert "nothing was saved" in result.output
    assert connects == []


def test_cli_downloads_everything_then_loads_and_reports(tmp_path, settings, monkeypatch):
    order = []
    real_download = player_games.download

    def fake_download(season, season_type, fetched_at=None):
        order.append(("download", season, season_type))
        prefix = {"Pre Season": "001", "Regular Season": "002", "PlayIn": "005",
                  "Playoffs": "004"}[season_type]
        game_id = f"{prefix}2300001"
        return real_download(
            season, season_type, FETCHED,
            fetch=_fetch_of(_frame(_row(game_id=game_id))),
        )

    real_connect = db.connect

    def recording_connect(*a, **k):
        order.append(("connect",))
        return real_connect(tmp_path / "cli.duckdb")

    monkeypatch.setattr(player_games, "download", fake_download)
    monkeypatch.setattr(db, "connect_with_retry", recording_connect)
    result = runner.invoke(cli.app, ["ingest-player-logs", "--season", "2023-24"])
    assert result.exit_code == 0, result.output
    assert order[-1] == ("connect",)
    assert [o[2] for o in order[:-1]] == list(player_games.ALL_SEASON_TYPES)
    # The competitive count excludes preseason; preseason is reported apart.
    assert "player logs 2023-24: 4 rows, 4 games (3 not in the games table)" in result.output
    assert "1 preseason game(s) not in the games table" in result.output


def test_cli_default_seasons_run_from_2014_15_to_current(settings, monkeypatch):
    seen = []

    def broken(season, season_type, fetched_at=None):
        seen.append(season)
        raise RuntimeError("stop")

    monkeypatch.setattr(player_games, "download", broken)
    monkeypatch.setattr(cli, "_now", lambda: datetime(2026, 10, 22, tzinfo=UTC))
    assert player_games.default_seasons(datetime(2026, 10, 22, tzinfo=UTC))[0] == "2014-15"
    assert player_games.default_seasons(datetime(2026, 10, 22, tzinfo=UTC))[-1] == "2026-27"
    result = runner.invoke(cli.app, ["ingest-player-logs"])
    assert result.exit_code == 1
    assert seen == ["2014-15"]


def test_no_season_total_endpoints_in_src():
    src = Path(__file__).resolve().parent.parent / "src"
    banned = (
        "PlayerEstimatedMetrics",
        "LeagueDashPlayerStats",
        "playerestimatedmetrics",
        "PlayerCareerStats",
    )
    for path in src.rglob("*.py"):
        text = path.read_text()
        for name in banned:
            assert name not in text, f"{name} appears in {path}"


def test_rating_season_types_exclude_preseason():
    assert player_games.RATING_SEASON_TYPES == ("regular", "playin", "playoffs")
    assert "preseason" not in player_games.RATING_SEASON_TYPES
    assert set(player_games.RATING_SEASON_TYPES) == {
        player_games.SEASON_TYPE_TAGS[t] for t in player_games.SEASON_TYPES
    }


def test_rating_rows_sql_keeps_competitive_rows_only(con, settings):
    for season_type, game_id in (("Pre Season", "0012300001"), ("Regular Season", "0022300001"),
                                 ("PlayIn", "0052300001"), ("Playoffs", "0042300001")):
        player_games.load(con, player_games.download(
            "2023-24", season_type, FETCHED, fetch=_fetch_of(_frame(_row(game_id=game_id))),
        ))
    table = db.POINT_IN_TIME_TABLES["player_games"]
    kept = con.execute(
        f"SELECT season_type FROM {table} WHERE {player_games.rating_rows_sql()} ORDER BY 1"
    ).fetchall()
    assert [r[0] for r in kept] == ["playin", "playoffs", "regular"]
    aliased = con.execute(
        f"SELECT count(*) FROM {table} AS p WHERE {player_games.rating_rows_sql('p')}"
    ).fetchone()[0]
    assert aliased == 3


def test_rating_rows_sql_rejects_a_non_identifier_alias():
    with pytest.raises(ValueError):
        player_games.rating_rows_sql("p; DROP TABLE x")


def test_module_docstring_restricts_preseason_to_roster_membership():
    doc = player_games.__doc__
    assert "roster membership" in doc
    assert "rating_rows_sql" in doc


def test_download_wraps_parse_failures_in_parse_error(settings):
    frame = _frame().drop(columns=["PLAYER_ID"])
    with pytest.raises(player_games.ParseError):
        player_games.download("2023-24", "Regular Season", FETCHED, fetch=_fetch_of(frame))


def test_cli_parse_error_has_its_own_message(settings, monkeypatch):
    connects = []
    monkeypatch.setattr(db, "connect_with_retry", lambda *a, **k: connects.append(1))

    def unreadable(season, season_type, fetched_at=None):
        raise player_games.ParseError("missing column 'PLAYER_ID'")

    monkeypatch.setattr(player_games, "download", unreadable)
    result = runner.invoke(cli.app, ["ingest-player-logs", "--season", "2023-24"])
    assert result.exit_code == 1
    assert result.output.startswith(
        "Downloaded and archived player game logs for 2023-24 Pre Season, but could not read them"
    )
    assert "Could not download" not in result.output
    assert connects == []


def test_cli_load_failure_names_seasons_already_saved(tmp_path, settings, monkeypatch):
    import duckdb

    real_download = player_games.download
    real_load = player_games.load

    def fake_download(season, season_type, fetched_at=None):
        return real_download(season, season_type, FETCHED,
                             fetch=_fetch_of(_frame(_row())))

    def flaky_load(con, downloaded):
        if downloaded.season == "2024-25":
            raise duckdb.Error("disk full")
        return real_load(con, downloaded)

    monkeypatch.setattr(player_games, "download", fake_download)
    monkeypatch.setattr(player_games, "load", flaky_load)
    monkeypatch.setattr(db, "connect_with_retry",
                        lambda *a, **k: db.connect(tmp_path / "cli.duckdb"))
    result = runner.invoke(
        cli.app, ["ingest-player-logs", "--season", "2023-24", "--season", "2024-25"]
    )
    assert result.exit_code == 1
    assert "Could not save player game logs for 2024-25 Pre Season" in result.output
    assert "Earlier seasons were saved: 2023-24." in result.output
