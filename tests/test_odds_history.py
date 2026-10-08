import io
import zipfile
from datetime import UTC, date, datetime

import pytest
import requests
from typer.testing import CliRunner

from predictor import cli, config, db, raw_store
from predictor.config import Settings
from predictor.sources import odds_history
from schedule_rows import insert_schedule_row

runner = CliRunner()

NOW = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)

# Copied from the real nba_2008-2026.csv (header verbatim). Rows: a pre-2014-15
# row (ignored), a home favourite, an away favourite, a blank spread, a
# 2025-26 row with blank moneylines (home favoured) and one with blank
# moneylines (away favoured, a west-coast late tip), and a playoff row.
HEADER = (
    "season,date,regular,playoffs,away,home,score_away,score_home,"
    "q1_away,q2_away,q3_away,q4_away,ot_away,q1_home,q2_home,q3_home,q4_home,ot_home,"
    "whos_favored,spread,total,moneyline_away,moneyline_home,"
    "h2_spread,h2_total,id_spread,id_total"
)
ROWS = [
    "2008,2007-10-30,True,False,por,sa,97,106,26,23,28,20,0,29,30,22,25,0,home,13,189.5,900,-1400,5,95,0,1",
    "2015,2014-10-28,True,False,dal,sa,100,101,24,29,20,27,0,26,19,31,25,0,home,3.5,203.5,140,-165,4,102,0,0",
    "2015,2014-10-28,True,False,hou,lal,108,90,31,31,23,23,0,19,26,24,21,0,away,7,207,-300,250,1.5,103,1,0",
    "2020,2019-12-19,True,False,hou,lac,122,117,27,27,36,32,0,28,41,18,30,0,away,,235.5,180,-220,2.5,115,,1",
    "2026,2025-10-21,True,False,hou,okc,124,125,30,27,22,25,20,27,24,24,29,21,home,6.5,225.5,,,,,0,1",
    "2026,2025-10-21,True,False,gs,lal,119,109,28,27,35,29,0,22,32,25,30,0,away,2.5,227.5,,,,,1,1",
    "2025,2025-04-19,False,True,lac,den,110,112,35,18,22,23,12,27,22,23,26,14,home,3.5,224.5,,,,,0,0",
]
CSV = "\n".join([HEADER, *ROWS]) + "\n"


def _zip(csv_text=CSV, name="nba_2008-2026.csv", extra=None):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, csv_text)
        for member, text in (extra or {}).items():
            zf.writestr(member, text)
    return buf.getvalue()


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(raw_store, "settings", s)
    return s


@pytest.fixture
def con(tmp_path, store):
    c = db.connect(tmp_path / "t.duckdb")
    db.migrate(c)
    return c


def _tip(y, m, d, h, mi=0):
    return datetime(y, m, d, h, mi, tzinfo=UTC)


def _seed_schedule(con):
    """Schedule games matching the fixture's in-scope rows."""
    insert_schedule_row(con, "0021400001", date(2014, 10, 28), "SAS", "DAL",
                        _tip(2014, 10, 29, 0, 0), season="2014-15")
    # 10:30pm ET tip: the UTC date is the next day, the ET date is the 28th.
    insert_schedule_row(con, "0021400002", date(2014, 10, 28), "LAL", "HOU",
                        _tip(2014, 10, 29, 2, 30), season="2014-15")
    insert_schedule_row(con, "0021900400", date(2019, 12, 19), "LAC", "HOU",
                        _tip(2019, 12, 20, 3, 30), season="2019-20")
    insert_schedule_row(con, "0022500001", date(2025, 10, 21), "OKC", "HOU",
                        _tip(2025, 10, 21, 23, 30), season="2025-26")
    insert_schedule_row(con, "0022500002", date(2025, 10, 21), "LAL", "GSW",
                        _tip(2025, 10, 22, 2, 0), season="2025-26")
    insert_schedule_row(con, "0042400111", date(2025, 4, 19), "DEN", "LAC",
                        _tip(2025, 4, 19, 19, 0), season="2024-25")


def _stored(con):
    return con.execute(
        "SELECT game_key, book, game_id, source, reconstructed, home_team, away_team,"
        " home_price, away_price, spread, total, observed_at"
        " FROM odds_snapshots_raw ORDER BY game_key"
    ).fetchall()


def _ingest(con, body=None, now=NOW):
    body = _zip() if body is None else body
    downloaded = odds_history.download(("user", "secret"), fetch=lambda u, k: body, now=now)
    return downloaded, odds_history.load(con, downloaded)


# --- parsing -----------------------------------------------------------------


def test_parse_reads_the_real_column_layout():
    rows = odds_history.parse_csv(CSV)
    assert len(rows) == 7
    old, sa, lal, lac, okc, gsw_lal, playoff = rows
    assert old.season == "2007-08"
    assert sa.season == "2014-15"
    assert sa.game_date == date(2014, 10, 28)
    assert (sa.home, sa.away) == ("SAS", "DAL")
    assert sa.regular is True and sa.playoffs is False
    assert (sa.score_home, sa.score_away) == (101, 100)
    assert sa.home_spread == -3.5  # home favoured: negative
    assert sa.total == 203.5
    assert (sa.moneyline_home, sa.moneyline_away) == (-165, 140)


def test_parse_signs_the_spread_for_an_away_favourite():
    lal = odds_history.parse_csv(CSV)[2]
    assert (lal.home, lal.away) == ("LAL", "HOU")
    assert lal.home_spread == 7.0  # away favoured: home gets +spread
    assert (lal.moneyline_home, lal.moneyline_away) == (250, -300)


def test_parse_blank_spread_and_blank_moneylines_are_none():
    rows = odds_history.parse_csv(CSV)
    assert rows[3].home_spread is None
    assert rows[3].moneyline_home == -220
    assert rows[4].moneyline_home is None and rows[4].moneyline_away is None
    assert rows[4].home_spread == -6.5
    assert rows[5].home_spread == 2.5
    assert (rows[5].home, rows[5].away) == ("LAL", "GSW")


def test_parse_playoff_flags():
    playoff = odds_history.parse_csv(CSV)[6]
    assert playoff.regular is False and playoff.playoffs is True
    assert playoff.season == "2024-25"


def test_team_codes_map_through_an_explicit_table():
    t = odds_history.KAGGLE_TEAMS
    assert len(t) == 30
    assert t["gs"] == "GSW" and t["no"] == "NOP" and t["ny"] == "NYK"
    assert t["sa"] == "SAS" and t["utah"] == "UTA" and t["wsh"] == "WAS"
    assert t["bkn"] == "BKN" and t["phx"] == "PHX" and t["cha"] == "CHA"
    assert odds_history.kaggle_team("xyz") is None


def test_unknown_team_code_is_kept_for_reporting():
    row = odds_history.parse_csv(
        "\n".join([HEADER, ROWS[1].replace(",dal,sa,", ",xyz,sa,")])
    )[0]
    assert row.away is None and row.away_code == "xyz"


def test_parse_rejects_a_file_without_the_expected_columns():
    with pytest.raises(ValueError, match="moneyline_home"):
        odds_history.parse_csv("season,date,away,home\n2015,2014-10-28,dal,sa\n")


def test_csv_member_is_chosen_by_suffix():
    body = _zip(name="nba_2008-2027.csv", extra={"README.txt": "hi"})
    assert odds_history.read_csv_member(body).startswith("season,date")


def test_zip_with_no_csv_is_an_error():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("README.txt", "hi")
    with pytest.raises(ValueError, match="csv"):
        odds_history.read_csv_member(buf.getvalue())


# --- download / raw-first ----------------------------------------------------


def test_download_archives_the_zip_under_its_hash(store):
    body = _zip()
    downloaded = odds_history.download(("u", "k"), fetch=lambda u, k: body, now=NOW)
    assert downloaded.archive_key.startswith("kaggle_") and downloaded.archive_key.endswith(".zip")
    assert len(downloaded.archive_key) == len("kaggle_") + 16 + len(".zip")
    assert raw_store.load("odds_history", downloaded.archive_key) == body


def test_same_bytes_is_a_raw_store_no_op(store):
    body = _zip()
    first = odds_history.download(("u", "k"), fetch=lambda u, k: body, now=NOW)
    second = odds_history.download(("u", "k"), fetch=lambda u, k: body, now=NOW)
    assert first.archive_key == second.archive_key
    assert len(list(raw_store.iter_manifest("odds_history"))) == 1


def test_load_parses_the_archived_bytes(con, monkeypatch):
    _seed_schedule(con)
    downloaded, _ = _ingest(con)
    con.execute("DELETE FROM odds_snapshots_raw")
    seen = []
    one_row = _zip("\n".join([HEADER, ROWS[1]]) + "\n")

    def fake_load(source, key):
        seen.append((source, key))
        return one_row

    monkeypatch.setattr(raw_store, "load", fake_load)
    summary = odds_history.load(con, downloaded)
    assert seen == [("odds_history", downloaded.archive_key)]
    assert summary.stored == 1


class FakeResponse:
    def __init__(self, status_code=200, content=b""):
        self.status_code = status_code
        self.content = content


class FakeSession:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_fetch_uses_basic_auth_and_a_timeout():
    session = FakeSession([FakeResponse(200, _zip())])
    body = odds_history.fetch_archive("user", "secret", session=session)
    assert body == _zip()
    url, kwargs = session.calls[0]
    assert url == (
        "https://www.kaggle.com/api/v1/datasets/download/"
        "cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024"
    )
    assert kwargs["auth"] == ("user", "secret")
    assert kwargs["timeout"]


def test_fetch_retries_network_errors_only(monkeypatch):
    monkeypatch.setattr(odds_history, "_sleep", lambda s: None)
    session = FakeSession([requests.ConnectionError("boom"), FakeResponse(200, _zip())])
    assert odds_history.fetch_archive("u", "k", session=session) == _zip()
    assert len(session.calls) == 2

    session = FakeSession([FakeResponse(500), FakeResponse(200, _zip())])
    with pytest.raises(odds_history.KaggleFetchError, match="HTTP 500"):
        odds_history.fetch_archive("u", "k", session=session)
    assert len(session.calls) == 1


def test_fetch_gives_up_after_repeated_network_errors(monkeypatch):
    monkeypatch.setattr(odds_history, "_sleep", lambda s: None)
    session = FakeSession([requests.Timeout("https://www.kaggle.com/x?k=secret")] * 3)
    with pytest.raises(odds_history.KaggleFetchError) as info:
        odds_history.fetch_archive("u", "secret", session=session)
    assert "secret" not in str(info.value) and "kaggle.com/" not in str(info.value)


@pytest.mark.parametrize("status", [401, 403])
def test_fetch_auth_failure(status):
    session = FakeSession([FakeResponse(status)])
    with pytest.raises(odds_history.KaggleAuthError):
        odds_history.fetch_archive("u", "k", session=session)


def test_fetch_rejects_a_body_that_is_not_a_zip():
    session = FakeSession([FakeResponse(200, b"<html>")])
    with pytest.raises(odds_history.KaggleFetchError, match="zip"):
        odds_history.fetch_archive("u", "k", session=session)


# --- linking and storing -----------------------------------------------------


def test_load_links_and_stores_one_consensus_row_per_game(con):
    _seed_schedule(con)
    _, summary = _ingest(con)

    rows = {r[0]: r for r in _stored(con)}
    assert set(rows) == {
        "kaggle:0021400001", "kaggle:0021400002", "kaggle:0021900400",
        "kaggle:0022500001", "kaggle:0022500002", "kaggle:0042400111",
    }
    sa = rows["kaggle:0021400001"]
    assert sa[1:5] == ("consensus", "0021400001", "kaggle_sbr", True)
    assert sa[5:11] == ("SAS", "DAL", -165, 140, -3.5, 203.5)
    assert sa[11] == _tip(2014, 10, 29, 0, 0)
    assert rows["kaggle:0021400002"][9] == 7.0
    assert rows["kaggle:0021900400"][9] is None
    assert rows["kaggle:0022500001"][7:9] == (None, None)
    assert summary.stored == 6
    assert summary.unmatched == []


def test_et_date_links_a_late_tip_whose_utc_date_is_the_next_day(con):
    _seed_schedule(con)
    _ingest(con)
    row = next(r for r in _stored(con) if r[0] == "kaggle:0022500002")
    assert row[11] == _tip(2025, 10, 22, 2, 0)  # observed_at is the tip-off


def test_links_against_the_latest_schedule_vintage(con):
    # The game was first listed for the 27th, then moved to the 28th.
    insert_schedule_row(con, "0021400001", date(2014, 10, 27), "SAS", "DAL",
                        _tip(2014, 10, 28, 0, 0), observed_at=_tip(2014, 9, 1, 0),
                        season="2014-15")
    insert_schedule_row(con, "0021400001", date(2014, 10, 28), "SAS", "DAL",
                        _tip(2014, 10, 29, 0, 0), observed_at=_tip(2014, 10, 1, 0),
                        season="2014-15")
    _ingest(con, _zip("\n".join([HEADER, ROWS[1]]) + "\n"))
    (row,) = _stored(con)
    assert row[2] == "0021400001" and row[11] == _tip(2014, 10, 29, 0, 0)


def test_rows_before_2014_15_are_ignored(con):
    _seed_schedule(con)
    _, summary = _ingest(con)
    assert summary.rows_read == 7
    assert summary.rows_in_scope == 6
    assert not any("2007-10-30" in u for u in summary.unmatched)


def test_unknown_team_is_reported_unmatched(con):
    _seed_schedule(con)
    body = _zip("\n".join([HEADER, ROWS[1].replace(",dal,sa,", ",xyz,sa,")]) + "\n")
    _, summary = _ingest(con, body)
    assert summary.stored == 0
    assert len(summary.unmatched) == 1
    assert "xyz" in summary.unmatched[0] and "unknown team" in summary.unmatched[0]


def test_row_with_no_scheduled_game_is_reported_unmatched(con):
    _, summary = _ingest(con)  # empty schedule
    assert summary.stored == 0
    assert len(summary.unmatched) == 6
    assert "DAL@SAS" in summary.unmatched[0] or "dal@sa" in summary.unmatched[0]


def test_score_mismatch_with_a_final_result_is_unmatched(con):
    _seed_schedule(con)
    games = db.POINT_IN_TIME_TABLES["games"]
    insert = (
        f"INSERT INTO {games} (game_id, season, game_date, home_team, away_team,"
        " home_points, away_points, status, observed_at) VALUES (?,?,?,?,?,?,?,?,?)"
    )
    con.execute(insert, ["0021400001", "2014-15", date(2014, 10, 28), "SAS", "DAL",
                         101, 100, "FINAL", _tip(2014, 10, 29, 5)])
    con.execute(insert, ["0021400002", "2014-15", date(2014, 10, 28), "LAL", "HOU",
                         91, 108, "FINAL", _tip(2014, 10, 29, 5)])
    _, summary = _ingest(con)
    keys = {r[0] for r in _stored(con)}
    assert "kaggle:0021400001" in keys  # scores agree
    assert "kaggle:0021400002" not in keys  # file says 90, we say 91
    assert any("score" in u and "HOU@LAL" in u for u in summary.unmatched)


def test_a_game_with_no_tip_off_is_skipped_and_counted(con):
    insert_schedule_row(con, "0021400001", date(2014, 10, 28), "SAS", "DAL",
                        _tip(2014, 10, 29, 0), season="2014-15")
    table = db.POINT_IN_TIME_TABLES["schedule"]
    con.execute(f"UPDATE {table} SET tip_off_utc = NULL")
    _, summary = _ingest(con, _zip("\n".join([HEADER, ROWS[1]]) + "\n"))
    assert summary.stored == 0
    assert summary.no_tip == 1
    assert _stored(con) == []


def test_rerun_is_idempotent(con):
    _seed_schedule(con)
    _, first = _ingest(con)
    before = _stored(con)
    _, second = _ingest(con)
    assert _stored(con) == before
    assert first.stored == second.stored == 6


def test_live_rows_are_untouched(con):
    _seed_schedule(con)
    table = db.POINT_IN_TIME_TABLES["odds_snapshots"]
    con.execute(
        f"INSERT INTO {table} (game_key, book, home_team, away_team, observed_at,"
        " game_id, source) VALUES ('abc', 'draftkings', 'SAS', 'DAL', ?, '0021400001',"
        " 'theoddsapi')",
        [_tip(2014, 10, 28, 20)],
    )
    _ingest(con)
    sources = con.execute(f"SELECT source, count(*) FROM {table} GROUP BY 1 ORDER BY 1").fetchall()
    assert sources == [("kaggle_sbr", 6), ("theoddsapi", 1)]


# --- coverage ----------------------------------------------------------------


def test_coverage_counts_regular_season_games_per_season(con):
    _seed_schedule(con)
    # A 2025-26 regular-season game with no line in the file.
    insert_schedule_row(con, "0022500003", date(2025, 10, 22), "BOS", "NYK",
                        _tip(2025, 10, 22, 23), season="2025-26")
    # Preseason never counts.
    insert_schedule_row(con, "0012500001", date(2025, 10, 5), "BOS", "NYK",
                        _tip(2025, 10, 5, 23), season="2025-26")
    # Seasons before 2014-15 are out of scope.
    insert_schedule_row(con, "0021300001", date(2013, 10, 29), "IND", "ORL",
                        _tip(2013, 10, 29, 23), season="2013-14")
    # A season after the file's last one (not played yet) is not reported.
    insert_schedule_row(con, "0022600001", date(2026, 10, 21), "BOS", "NYK",
                        _tip(2026, 10, 21, 23), season="2026-27")
    _, summary = _ingest(con)
    cov = {c.season: c for c in summary.seasons}
    assert "2013-14" not in cov
    assert "2026-27" not in cov
    assert (cov["2014-15"].scheduled, cov["2014-15"].linked) == (2, 2)
    assert (cov["2025-26"].scheduled, cov["2025-26"].linked) == (3, 2)
    assert cov["2025-26"].pct == pytest.approx(200 / 3)
    # The playoff game is linked but not part of regular-season coverage.
    assert (cov["2024-25"].scheduled, cov["2024-25"].linked) == (0, 0)


def test_gate_fails_on_low_or_missing_gate_season(con):
    _seed_schedule(con)
    insert_schedule_row(con, "0022500003", date(2025, 10, 22), "BOS", "NYK",
                        _tip(2025, 10, 22, 23), season="2025-26")
    _, summary = _ingest(con)
    failing = summary.failing_seasons(("2019-20", "2025-26", "2021-22"))
    assert failing == ["2021-22", "2025-26"]
    assert summary.failing_seasons(("2019-20",)) == []


# --- CLI ---------------------------------------------------------------------


@pytest.fixture
def cli_env(tmp_path, store, monkeypatch):
    monkeypatch.setattr(config, "settings", store)
    monkeypatch.setattr(db, "settings", store)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    return home


def _write_token(home):
    path = home / ".kaggle" / "kaggle.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"username": "user", "key": "not-a-real-key"}')


def test_cli_missing_credentials_names_the_token_path(cli_env):
    result = runner.invoke(cli.app, ["ingest-odds-history"])
    assert result.exit_code == 1
    assert "~/.kaggle/kaggle.json" in result.output
    assert "Create New Token" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("status", [401, 403])
def test_cli_auth_failure_is_a_plain_message(cli_env, monkeypatch, status):
    _write_token(cli_env)

    def fetch(username, key, session=None):
        raise odds_history.KaggleAuthError(status)

    monkeypatch.setattr(odds_history, "fetch_archive", fetch)
    result = runner.invoke(cli.app, ["ingest-odds-history"])
    assert result.exit_code == 1
    text = result.output.replace("\n", " ")
    assert "token" in text and "accept" in text and "dataset page" in text
    assert "not-a-real-key" not in result.output
    assert "Traceback" not in result.output


def test_cli_fetch_error_is_a_plain_message(cli_env, monkeypatch):
    _write_token(cli_env)

    def fetch(username, key, session=None):
        raise odds_history.KaggleFetchError("could not reach Kaggle (ConnectionError)")

    monkeypatch.setattr(odds_history, "fetch_archive", fetch)
    result = runner.invoke(cli.app, ["ingest-odds-history"])
    assert result.exit_code == 1
    assert "could not reach Kaggle" in result.output
    assert "Traceback" not in result.output


def _cli_with_schedule(cli_env, monkeypatch, extra_games=()):
    _write_token(cli_env)
    monkeypatch.setattr(odds_history, "fetch_archive", lambda u, k, session=None: _zip())
    con = db.connect()
    db.migrate(con)
    _seed_schedule(con)
    for args in extra_games:
        insert_schedule_row(con, *args)
    con.close()


def test_cli_prints_coverage_and_exits_0_when_gate_seasons_are_covered(cli_env, monkeypatch):
    monkeypatch.setattr(odds_history, "GATE_SEASONS", ("2019-20", "2025-26"))
    _cli_with_schedule(cli_env, monkeypatch)
    result = runner.invoke(cli.app, ["ingest-odds-history"])
    assert result.exit_code == 0, result.output
    assert "2025-26" in result.output and "100.0%" in result.output
    assert "stored 6" in result.output


def test_cli_low_coverage_exits_1_with_a_plain_message(cli_env, monkeypatch):
    monkeypatch.setattr(odds_history, "GATE_SEASONS", ("2019-20", "2025-26"))
    extra = [
        (f"00225000{n:02d}", date(2025, 11, n), "BOS", "NYK", _tip(2025, 11, n, 23),
         None, "2025-26")
        for n in range(3, 10)
    ]
    _cli_with_schedule(cli_env, monkeypatch, extra)
    result = runner.invoke(cli.app, ["ingest-odds-history"])
    assert result.exit_code == 1
    text = result.output.replace("\n", " ")
    assert "2025-26" in text and "below 90%" in text and "unreliable" in text
    assert "Traceback" not in result.output


def test_cli_reports_unmatched_rows(cli_env, monkeypatch):
    monkeypatch.setattr(odds_history, "GATE_SEASONS", ())
    _write_token(cli_env)
    body = _zip("\n".join([HEADER, ROWS[1].replace(",dal,sa,", ",xyz,sa,")]) + "\n")
    monkeypatch.setattr(odds_history, "fetch_archive", lambda u, k, session=None: body)
    result = runner.invoke(cli.app, ["ingest-odds-history"])
    assert result.exit_code == 0, result.output
    assert "1 row(s) not matched" in result.output
    assert "xyz" in result.output


def test_cli_load_failure_names_the_archive_key(cli_env, monkeypatch):
    _write_token(cli_env)
    monkeypatch.setattr(odds_history, "fetch_archive",
                        lambda u, k, session=None: _zip("garbage\n1\n"))
    result = runner.invoke(cli.app, ["ingest-odds-history"])
    assert result.exit_code == 1
    assert "kaggle_" in result.output and "odds_history" in result.output
    assert "Traceback" not in result.output
