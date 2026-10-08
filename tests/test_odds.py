import json
from datetime import UTC, date, datetime

import pytest
import requests
from typer.testing import CliRunner

from predictor import cli, config, db, raw_store
from predictor.config import Settings
from predictor.sources import odds
from predictor.teams import team_abbr
from schedule_rows import insert_schedule_row

runner = CliRunner()

NOW = datetime(2025, 1, 15, 20, 0, tzinfo=UTC)


def _event(event_id, home, away, commence, books=("draftkings",)):
    return {
        "id": event_id,
        "commence_time": commence,
        "home_team": home,
        "away_team": away,
        "bookmakers": [
            {
                "key": book,
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": home, "price": -150},
                            {"name": away, "price": 130},
                        ],
                    },
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": home, "price": -110, "point": -3.5},
                            {"name": away, "price": -110, "point": 3.5},
                        ],
                    },
                ],
            }
            for book in books
        ],
    }


# 7:00pm ET on 2025-01-15 = 00:00 UTC on the 16th.
PAYLOAD = [
    _event("abc123", "Philadelphia 76ers", "New York Knicks", "2025-01-16T00:00:00Z")
]


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None, text=""):
        self.status_code = status_code
        self.content = json.dumps(payload if payload is not None else []).encode()
        self.headers = headers or {}
        self.text = text

    def json(self):
        return json.loads(self.content)


class FakeSession:
    def __init__(self, response):
        self.response = response

    def get(self, *args, **kwargs):
        return self.response


@pytest.fixture
def store(tmp_path, monkeypatch):
    # The archive goes through raw_store's own module-level `settings`, so it
    # must be patched separately from db/config (same as test_injury_fetch).
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(raw_store, "settings", s)
    return s


@pytest.fixture
def con(tmp_path, store):
    c = db.connect(tmp_path / "t.duckdb")
    db.migrate(c)
    return c


def _fake_fetch(monkeypatch, payload, remaining=None):
    def fetch(api_key, session=None):
        return odds.OddsResponse(json.dumps(payload).encode(), remaining)

    monkeypatch.setattr(odds, "fetch_current", fetch)


def _stored(con):
    # odds_snapshots_raw is the physical table; tests are exempt from the
    # no-physical-table-name rule (see test_leakage).
    return con.execute(
        "SELECT game_key, book, game_id, source, reconstructed, home_price,"
        " spread, observed_at FROM odds_snapshots_raw ORDER BY game_key, book"
    ).fetchall()


# --- parsing -----------------------------------------------------------------


def test_parse_extracts_one_row_per_bookmaker():
    rows = odds.parse_odds_payload(PAYLOAD, NOW)
    assert len(rows) == 1
    row = rows[0]
    assert row["book"] == "draftkings"
    assert row["home_price"] == -150
    assert row["away_price"] == 130
    assert row["spread"] == -3.5
    assert row["observed_at"] == NOW


def test_parse_skips_events_without_bookmakers():
    assert odds.parse_odds_payload([{"id": "x", "home_team": "A", "away_team": "B"}], NOW) == []


def test_team_abbr_reads_odds_api_names_and_abbreviations():
    assert team_abbr("Philadelphia 76ers") == "PHI"
    assert team_abbr("Los Angeles Clippers") == "LAC"
    assert team_abbr("LA Clippers") == "LAC"
    assert team_abbr("PHX") == "PHX"
    assert team_abbr("Springfield Isotopes") is None
    assert team_abbr(None) is None


def test_eastern_date_of_commence_time():
    assert odds._eastern_date("2025-01-16T03:30:00Z") == date(2025, 1, 15)
    assert odds._eastern_date("2025-01-16T05:00:00Z") == date(2025, 1, 16)
    assert odds._eastern_date(None) is None
    assert odds._eastern_date("garbage") is None


# --- fetch -------------------------------------------------------------------


def test_fetch_returns_body_and_remaining_requests_header():
    session = FakeSession(FakeResponse(payload=PAYLOAD, headers={"x-requests-remaining": "487"}))
    response = odds.fetch_current("k", session=session)
    assert response.payload == PAYLOAD
    assert response.requests_remaining == "487"


def test_fetch_without_remaining_header_gives_none():
    response = odds.fetch_current("k", session=FakeSession(FakeResponse(payload=[])))
    assert response.requests_remaining is None


@pytest.mark.parametrize("status", [401, 429])
def test_quota_exhaustion_raises_named_error_with_remaining(status):
    session = FakeSession(
        FakeResponse(status, text="quota", headers={"x-requests-remaining": "0"})
    )
    with pytest.raises(odds.OddsQuotaExceeded) as info:
        odds.fetch_current("k", session=session)
    assert info.value.status_code == status
    assert info.value.requests_remaining == "0"


def test_other_http_errors_never_echo_the_key():
    session = FakeSession(FakeResponse(500, text="boom"))
    with pytest.raises(odds.OddsFetchError) as info:
        odds.fetch_current("SECRETKEY", session=session)
    assert "SECRETKEY" not in str(info.value)
    assert "500" in str(info.value)


def test_network_errors_never_echo_the_key():
    class Boom:
        def get(self, *a, **k):
            raise requests.ConnectionError("Max retries exceeded with url: /odds?apiKey=SECRETKEY")

    with pytest.raises(odds.OddsFetchError) as info:
        odds.fetch_current("SECRETKEY", session=Boom())
    assert "SECRETKEY" not in str(info.value)


def test_a_non_list_body_is_a_fetch_error():
    session = FakeSession(FakeResponse(payload={"message": "odd"}))
    with pytest.raises(odds.OddsFetchError):
        odds.fetch_current("k", session=session)


# --- download / ingest -------------------------------------------------------


def test_observed_at_is_stamped_after_the_fetch_returns(store, monkeypatch):
    events = []

    def fetch(api_key, session=None):
        events.append("fetched")
        return odds.OddsResponse(json.dumps(PAYLOAD).encode(), None)

    def clock():
        events.append("stamped")
        return NOW

    monkeypatch.setattr(odds, "fetch_current", fetch)
    monkeypatch.setattr(odds, "_utcnow", clock)
    downloaded = odds.download("k")
    assert events == ["fetched", "stamped"]
    assert downloaded.observed_at == NOW


def test_download_archives_the_response_bytes_and_load_parses_the_archive(con, monkeypatch):
    _fake_fetch(monkeypatch, PAYLOAD, remaining="12")
    downloaded = odds.download("k", now=NOW)
    assert downloaded.requests_remaining == "12"
    assert raw_store.load("odds", downloaded.archive_key) == json.dumps(PAYLOAD).encode()

    # Raw-first: ingest parses the archived bytes, not an in-memory copy.
    monkeypatch.setattr(raw_store, "load", lambda source, key: b"[]")
    summary = odds.ingest_current(con, fetch=downloaded)
    assert summary.events == 0
    assert summary.rows == 0


def test_ingest_writes_rows_linked_to_the_scheduled_game(con, monkeypatch):
    insert_schedule_row(
        con, "0022400555", date(2025, 1, 15), "PHI", "NYK",
        datetime(2025, 1, 16, 0, 0, tzinfo=UTC),
    )
    _fake_fetch(monkeypatch, PAYLOAD)

    summary = odds.ingest_current(con, api_key="k", now=NOW)

    assert summary.rows == 1
    assert summary.events == 1
    assert summary.linked == 1
    assert summary.unlinked == []
    assert summary.shifted == []
    assert _stored(con) == [
        ("abc123", "draftkings", "0022400555", "theoddsapi", False, -150, -3.5, NOW)
    ]


def test_a_late_pacific_tip_links_by_its_eastern_date_not_its_utc_date(con, monkeypatch):
    """7:30pm PT on Jan 15 is 03:30 UTC on Jan 16 -- the schedule files the
    game under its ET date, Jan 15, which is what must match (exactly, not
    through the +-1 day fallback)."""
    insert_schedule_row(
        con, "0022400600", date(2025, 1, 15), "LAL", "BOS",
        datetime(2025, 1, 16, 3, 30, tzinfo=UTC),
    )
    # A same-teams game on the UTC date must not be the one chosen.
    insert_schedule_row(
        con, "0022400700", date(2025, 1, 16), "LAL", "BOS",
        datetime(2025, 1, 17, 3, 30, tzinfo=UTC),
    )
    _fake_fetch(monkeypatch, [
        _event("late", "Los Angeles Lakers", "Boston Celtics", "2025-01-16T03:30:00Z")
    ])

    summary = odds.ingest_current(con, api_key="k", now=NOW)

    assert summary.linked == 1
    assert summary.shifted == []
    assert [r[2] for r in _stored(con)] == ["0022400600"]


def test_a_game_a_day_off_links_through_the_fallback_and_is_recorded(con, monkeypatch):
    insert_schedule_row(
        con, "0022400800", date(2025, 1, 16), "MIA", "CHI",
        datetime(2025, 1, 17, 0, 30, tzinfo=UTC),
    )
    _fake_fetch(monkeypatch, [
        _event("moved", "Miami Heat", "Chicago Bulls", "2025-01-16T00:30:00Z")
    ])

    summary = odds.ingest_current(con, api_key="k", now=NOW)

    assert summary.linked == 1
    assert summary.unlinked == []
    assert summary.shifted == ["CHI@MIA 2025-01-15 -> 2025-01-16"]
    assert [r[2] for r in _stored(con)] == ["0022400800"]


def test_only_the_latest_schedule_vintage_is_used(con, monkeypatch):
    # First listed on the 15th, later moved to the 18th: the line for the
    # 15th must not link to it through the stale vintage.
    insert_schedule_row(
        con, "0022400900", date(2025, 1, 15), "DEN", "UTA",
        datetime(2025, 1, 16, 2, 0, tzinfo=UTC),
        observed_at=datetime(2025, 1, 1, tzinfo=UTC),
    )
    insert_schedule_row(
        con, "0022400900", date(2025, 1, 18), "DEN", "UTA",
        datetime(2025, 1, 19, 2, 0, tzinfo=UTC),
        observed_at=datetime(2025, 1, 10, tzinfo=UTC),
    )
    _fake_fetch(monkeypatch, [
        _event("den", "Denver Nuggets", "Utah Jazz", "2025-01-16T02:00:00Z")
    ])

    summary = odds.ingest_current(con, api_key="k", now=NOW)

    assert summary.linked == 0
    assert summary.unlinked == ["UTA@DEN 2025-01-15"]


def test_unmatched_events_are_stored_unlinked_and_named(con, monkeypatch):
    _fake_fetch(monkeypatch, [
        _event("nogame", "Golden State Warriors", "Phoenix Suns", "2025-01-16T03:00:00Z"),
        _event("noteam", "Springfield Isotopes", "Phoenix Suns", "2025-01-16T03:00:00Z"),
    ])

    summary = odds.ingest_current(con, api_key="k", now=NOW)

    assert summary.events == 2
    assert summary.rows == 2
    assert summary.linked == 0
    assert summary.unlinked == [
        "PHX@GSW 2025-01-15",
        "PHX@Springfield Isotopes 2025-01-15",
    ]
    assert [(r[0], r[2], r[3]) for r in _stored(con)] == [
        ("nogame", None, "theoddsapi"),
        ("noteam", None, "theoddsapi"),
    ]


def test_preseason_games_are_not_link_targets(con, monkeypatch):
    insert_schedule_row(
        con, "0012400010", date(2025, 1, 15), "PHI", "NYK",
        datetime(2025, 1, 16, 0, 0, tzinfo=UTC),
    )
    _fake_fetch(monkeypatch, PAYLOAD)
    summary = odds.ingest_current(con, api_key="k", now=NOW)
    assert summary.linked == 0


def test_missing_api_key_raises_a_clear_error_naming_the_key_file(con, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    with pytest.raises(odds.MissingOddsKey, match="odds_api_key"):
        odds.ingest_current(con)


# --- CLI ---------------------------------------------------------------------


@pytest.fixture
def cli_env(tmp_path, store, monkeypatch):
    """Settings at a temp dir everywhere, HOME at an empty temp dir, no env key."""
    monkeypatch.setattr(config, "settings", store)
    monkeypatch.setattr(db, "settings", store)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    return home


def _write_key(home):
    path = home / ".config" / "predictor" / "odds_api_key"
    path.parent.mkdir(parents=True)
    path.write_text("test-key\n")


def test_cli_missing_key_prints_the_exact_path_to_create(cli_env):
    result = runner.invoke(cli.app, ["ingest-odds"])

    assert result.exit_code == 1
    expected = str(cli_env / ".config" / "predictor" / "odds_api_key")
    assert expected in result.output.replace("\n", "")
    assert "the-odds-api.com" in result.output
    assert "Traceback" not in result.output


def test_cli_downloads_before_opening_the_database(cli_env, monkeypatch):
    _write_key(cli_env)
    order = []

    def fetch(api_key, session=None):
        order.append(("fetch", api_key))
        return odds.OddsResponse(json.dumps(PAYLOAD).encode(), None)

    real_connect = db.connect_with_retry

    def connect(*a, **k):
        order.append(("connect", None))
        return real_connect(*a, **k)

    monkeypatch.setattr(odds, "fetch_current", fetch)
    monkeypatch.setattr(db, "connect_with_retry", connect)

    result = runner.invoke(cli.app, ["ingest-odds"])

    assert result.exit_code == 0, result.output
    assert order == [("fetch", "test-key"), ("connect", None)]
    assert "test-key" not in result.output


def test_cli_prints_rows_links_unlinked_and_remaining_requests(cli_env, monkeypatch):
    _write_key(cli_env)
    _fake_fetch(monkeypatch, PAYLOAD, remaining="487")

    result = runner.invoke(cli.app, ["ingest-odds"])

    assert result.exit_code == 0, result.output
    assert "stored 1 odds rows for 1 game(s); 0 linked to the schedule" in result.output
    assert "not matched to a scheduled game: NYK@PHI 2025-01-15" in result.output
    assert "Odds API requests remaining this month: 487" in result.output


def test_cli_quota_exceeded_is_a_plain_message_and_exit_1(cli_env, monkeypatch):
    _write_key(cli_env)

    def fetch(api_key, session=None):
        raise odds.OddsQuotaExceeded(429, "quota", requests_remaining="0")

    monkeypatch.setattr(odds, "fetch_current", fetch)

    result = runner.invoke(cli.app, ["ingest-odds"])

    assert result.exit_code == 1
    assert "The Odds API refused the request" in result.output
    assert "Odds API requests remaining this month: 0" in result.output
    assert "Traceback" not in result.output


def test_cli_fetch_error_is_a_plain_message_and_exit_1(cli_env, monkeypatch):
    _write_key(cli_env)

    def fetch(api_key, session=None):
        raise odds.OddsFetchError("could not reach The Odds API (ConnectionError)")

    monkeypatch.setattr(odds, "fetch_current", fetch)

    result = runner.invoke(cli.app, ["ingest-odds"])

    assert result.exit_code == 1
    assert "could not reach The Odds API" in result.output
    assert "Traceback" not in result.output
