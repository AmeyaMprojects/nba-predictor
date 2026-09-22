from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from predictor import db
from predictor.sources import injury_report

FIXTURE = Path(__file__).parent / "fixtures" / "Injury-Report_2025-01-15_05PM.pdf"


@pytest.fixture(scope="module")
def parsed():
    return injury_report.parse_report(FIXTURE.read_bytes())


def test_published_at_comes_from_pdf_content_not_filename(parsed):
    # Filename says 05PM; the document header says 05:30 PM Eastern.
    assert parsed.published_at == datetime(2025, 1, 15, 22, 30, tzinfo=UTC)


def test_row_count_matches_verified_baseline(parsed):
    assert len(parsed.rows) == 161


def test_all_fourteen_matchups_are_present(parsed):
    assert len({r.matchup for r in parsed.rows}) == 14


def test_group_columns_are_forward_filled(parsed):
    assert all(r.team for r in parsed.rows)
    assert all(r.matchup for r in parsed.rows)
    # This report spans two game dates (a back-to-back slate): 136 rows on
    # 2025-01-15 and 25 rows on 2025-01-16, per the "A single report can
    # span TWO game dates" fact -- this is exactly why game_date is part
    # of the widened primary key in predictor.db. Every row's game_date
    # must still parse to one of those two dates, never be missing.
    assert all(r.game_date in {date(2025, 1, 15), date(2025, 1, 16)} for r in parsed.rows)
    assert sum(r.game_date == date(2025, 1, 15) for r in parsed.rows) == 136
    assert sum(r.game_date == date(2025, 1, 16) for r in parsed.rows) == 25


def test_wrapped_reason_text_is_reassembled(parsed):
    by_player = {r.player: r for r in parsed.rows}
    assert by_player["Brunson,Jalen"].reason == "Injury/Illness-RightShoulder;Soreness"
    assert by_player["Towns,Karl-Anthony"].reason == "Injury/Illness-RightThumb;Sprained"


def test_statuses_are_from_the_known_vocabulary(parsed):
    known = {"Out", "Questionable", "Probable", "Doubtful", "Available"}
    assert {r.status for r in parsed.rows} <= known


def test_pages_after_the_first_are_parsed(parsed):
    # Page 1 alone yields 15 rows; anything near that means later pages were dropped.
    assert len(parsed.rows) > 100


def test_title_line_is_not_emitted_as_a_row(parsed):
    assert not any("InjuryReport" in r.player.replace(" ", "") for r in parsed.rows)


def test_ingest_writes_rows_with_published_at_as_observed_at(tmp_path):
    con = db.connect(tmp_path / "t.duckdb")
    db.migrate(con)
    count = injury_report.ingest_report(con, FIXTURE.read_bytes())
    assert count == 161
    distinct = con.execute("SELECT DISTINCT observed_at FROM injury_status_raw").fetchall()
    assert distinct == [(datetime(2025, 1, 15, 22, 30, tzinfo=UTC),)]


def test_ingest_is_idempotent(tmp_path):
    con = db.connect(tmp_path / "t.duckdb")
    db.migrate(con)
    injury_report.ingest_report(con, FIXTURE.read_bytes())
    injury_report.ingest_report(con, FIXTURE.read_bytes())
    total = con.execute("SELECT count(*) FROM injury_status_raw").fetchone()[0]
    assert total == 161


def test_unparseable_game_date_falls_back_to_report_publication_date(tmp_path, monkeypatch):
    # game_date is DATE NOT NULL and part of the primary key (see
    # predictor.db schema comment); losing a row is worse than an
    # imperfect date, so ingest_report must substitute the report's own
    # publication date rather than drop the row or insert NULL. The real
    # fixture happens to have zero unparseable dates, so this is
    # exercised directly against a fake ParsedReport.
    published_at = datetime(2025, 1, 15, 22, 30, tzinfo=UTC)
    fake_report = injury_report.ParsedReport(
        published_at=published_at,
        rows=[
            injury_report.InjuryRow(
                game_date=None,
                game_time="07:00(ET)",
                matchup="LAL@BOS",
                team="LosAngelesLakers",
                player="Doe,John",
                status="Out",
                reason="Injury/Illness-LeftKnee;Soreness",
            )
        ],
    )
    monkeypatch.setattr(injury_report, "parse_report", lambda pdf_bytes: fake_report)

    con = db.connect(tmp_path / "t.duckdb")
    db.migrate(con)
    count = injury_report.ingest_report(con, b"unused")
    assert count == 1
    row = con.execute(
        "SELECT game_date, report_date FROM injury_status_raw"
    ).fetchone()
    assert row == (date(2025, 1, 15), date(2025, 1, 15))


def test_ingest_rejects_naive_published_at(tmp_path, monkeypatch):
    fake_report = injury_report.ParsedReport(
        published_at=datetime(2025, 1, 15, 22, 30),  # naive, no tzinfo
        rows=[],
    )
    monkeypatch.setattr(injury_report, "parse_report", lambda pdf_bytes: fake_report)

    con = db.connect(tmp_path / "t.duckdb")
    db.migrate(con)
    with pytest.raises(ValueError):
        injury_report.ingest_report(con, b"unused")


def test_parse_raises_loudly_when_column_headers_are_never_found(monkeypatch):
    # If _column_bounds never succeeds on any page (format drift, an
    # unexpected layout, a corrupted PDF), parse_report must raise rather
    # than silently returning ParsedReport(rows=[]) -- a well-formed
    # report is never actually empty, and ingest_report writing 0 rows
    # for a real report would be an invisible hole in a Task 9 backfill.
    monkeypatch.setattr(injury_report, "_column_bounds", lambda page: None)
    with pytest.raises(injury_report.InjuryReportParseError):
        injury_report.parse_report(FIXTURE.read_bytes())


def test_report_date_uses_eastern_calendar_day_not_utc(tmp_path, monkeypatch):
    # A report published late evening Eastern crosses midnight UTC. The
    # real 2025-01-15_10PM report has published_at 2025-01-16 03:30 UTC,
    # which is 2025-01-15 22:30 ET -- the correct, filename-matching
    # publication day. report_date must reflect the Eastern day, not
    # whichever day UTC happens to land on, both for the stored
    # report_date column and for its use as the game_date fallback (a
    # row with an unparseable game_date must not land on the wrong
    # calendar day either).
    published_at = datetime(2025, 1, 16, 3, 30, tzinfo=UTC)
    fake_report = injury_report.ParsedReport(
        published_at=published_at,
        rows=[
            injury_report.InjuryRow(
                game_date=None,
                game_time="10:00(ET)",
                matchup="LAL@BOS",
                team="LosAngelesLakers",
                player="Doe,John",
                status="Out",
                reason="Injury/Illness-LeftKnee;Soreness",
            )
        ],
    )
    monkeypatch.setattr(injury_report, "parse_report", lambda pdf_bytes: fake_report)

    con = db.connect(tmp_path / "t.duckdb")
    db.migrate(con)
    injury_report.ingest_report(con, b"unused")
    row = con.execute(
        "SELECT game_date, report_date, observed_at FROM injury_status_raw"
    ).fetchone()
    assert row == (date(2025, 1, 15), date(2025, 1, 15), published_at)
