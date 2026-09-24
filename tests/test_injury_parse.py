from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from predictor import db
from predictor.sources import injury_report

FIXTURE = Path(__file__).parent / "fixtures" / "Injury-Report_2025-01-15_05PM.pdf"

# Old ("spaced") layout: header labels are separate words ("Game" "Date"
# rather than "GameDate") and data cells keep internal spaces ("Butler,
# Jimmy" rather than "Butler,Jimmy"). This is the layout every report from
# 2019-12-01 through ~2023-01 uses -- 749+ archived reports, essentially
# the entire early history of the backfill, all of which failed to parse
# before this format was supported (Task 8). Six pages, 137 rows, a
# genuine wrapped reason ("Injury Recovery + Health and Safety" /
# "Protocols" split across two physical lines), and ordinary
# multi-word team/player names, so it exercises page continuation, the
# spaced-header column matching, and reason reassembly all in one file.
OLD_FIXTURE = Path(__file__).parent / "fixtures" / "Injury-Report_2022-01-01_05PM.pdf"


@pytest.fixture(scope="module")
def parsed():
    return injury_report.parse_report(FIXTURE.read_bytes())


@pytest.fixture(scope="module")
def parsed_old():
    return injury_report.parse_report(OLD_FIXTURE.read_bytes())


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


def test_not_yet_submitted_placeholder_is_never_attached_to_a_reason(parsed):
    # Same regression as the old-layout case below, in the new layout:
    # a team with no report filed gets its own row, reason
    # "NOTYETSUBMITTED" (concatenated here), no player/status -- and
    # before this fix could attach to the nearest real anchor.
    # "Travers,Luke" (page 10 of this fixture) sits near several such
    # placeholder rows for the CLE@OKC / other late matchups.
    by_player = {r.player: r for r in parsed.rows}
    assert by_player["Travers,Luke"].reason == "GLeague-Two-Way"
    assert not any("SUBMIT" in r.reason.upper() for r in parsed.rows)


def test_pages_after_the_first_are_parsed(parsed):
    # Page 1 alone yields 15 rows; anything near that means later pages were dropped.
    assert len(parsed.rows) > 100


def test_title_line_is_not_emitted_as_a_row(parsed):
    assert not any("InjuryReport" in r.player.replace(" ", "") for r in parsed.rows)


# --- Old ("spaced") layout -- Task 8 ---------------------------------------
#
# Ground truth for the fixture (Injury-Report_2022-01-01_05PM.pdf, 6 pages)
# was established by parsing it with the fixed parser and independently
# sanity-checking the row count against the raw PDF text (grepping for
# lines that pair a "Lastname, Firstname" player cell with a known status
# word landed on the same 137, page by page). No row has a blank team,
# player, or reason.


def test_old_format_published_at_comes_from_pdf_content(parsed_old):
    assert parsed_old.published_at == datetime(2022, 1, 1, 22, 30, tzinfo=UTC)


def test_old_format_row_count_matches_verified_baseline(parsed_old):
    assert len(parsed_old.rows) == 137


def test_old_format_matchup_count_matches_verified_baseline(parsed_old):
    assert len({r.matchup for r in parsed_old.rows}) == 10


def test_old_format_statuses_are_from_the_known_vocabulary(parsed_old):
    known = {"Out", "Questionable", "Probable", "Doubtful", "Available"}
    assert {r.status for r in parsed_old.rows} == {
        "Out",
        "Questionable",
        "Probable",
        "Doubtful",
    }
    assert {r.status for r in parsed_old.rows} <= known


def test_old_format_no_row_has_a_blank_team_player_or_reason(parsed_old):
    assert all(r.team for r in parsed_old.rows)
    assert all(r.player for r in parsed_old.rows)
    assert all(r.reason for r in parsed_old.rows)
    assert all(r.matchup for r in parsed_old.rows)


def test_old_format_pages_after_the_first_are_parsed(parsed_old):
    # Page 1 alone yields 27 rows; anything near that means later pages
    # (where, unlike the new layout, the header line is repeated on every
    # page) were dropped or double counted.
    assert len(parsed_old.rows) > 100


def test_old_format_preserves_internal_spaces_in_data_cells(parsed_old):
    # Unlike the new layout ("Brunson,Jalen" / "NewYorkKnicks"), the old
    # layout keeps spaces in cell text: "Butler, Jimmy" / "Miami Heat",
    # not "Butler,Jimmy" / "MiamiHeat".
    by_player = {r.player: r for r in parsed_old.rows}
    assert "Green, Draymond" in by_player
    assert by_player["Green, Draymond"].team == "Golden State Warriors"


def test_old_format_wrapped_reason_text_is_reassembled(parsed_old):
    # Both of these wrap across two physical lines in the source PDF --
    # e.g. Wiseman's reason line reads "...Injury Recovery + Health and
    # Safety" with "Protocols" continuing on the next line -- and must
    # reassemble the same way the new layout's wrapped reasons do (pieces
    # joined with no separator).
    by_player = {r.player: r for r in parsed_old.rows}
    assert (
        by_player["Wiseman, James"].reason
        == "Injury/Illness - Right Knee; Injury Recovery + Health and SafetyProtocols"
    )
    assert (
        by_player["Fultz, Markelle"].reason
        == "Injury/Illness - Left Knee; Injury Recovery; Health & SafetyProtocols"
    )


def test_old_format_not_yet_submitted_placeholder_is_never_attached_to_a_reason(
    parsed_old,
):
    # Regression guard (post-Task-8 finding): a team with no report filed
    # yet gets its own row -- team name, no player, no status, reason
    # "NOT YET SUBMITTED" -- which looks exactly like a wrapped-Reason
    # continuation fragment to the nearest-anchor attachment logic and,
    # before this fix, got glued onto some nearby player's real reason.
    # "Freedom, Enes" on page 2 of this fixture is exactly that case: its
    # true reason is "Health and Safety Protocols", but a Boston Celtics
    # "NOT YET SUBMITTED" placeholder row sits closer to Freedom's anchor
    # line than to anything else and used to attach there.
    by_player = {r.player: r for r in parsed_old.rows}
    assert by_player["Freedom, Enes"].reason == "Health and Safety Protocols"
    assert not any("SUBMIT" in r.reason.upper() for r in parsed_old.rows)


def test_old_format_reason_can_contain_the_word_team(parsed_old):
    # Regression guard: the old layout's header row is matched by
    # multi-word text ("Team", "Matchup", "Reason", ...), not by a single
    # concatenated token, so a naive per-word filter that drops any word
    # whose TEXT happens to equal a column name would also silently drop
    # the literal word "Team" out of reason text like "Not With Team"
    # (this collision is invisible in the new layout, where the
    # equivalent reason is the single word "NotWithTeam"). Dragic's row
    # on page 4 of this fixture is exactly that case.
    by_player = {r.player: r for r in parsed_old.rows}
    assert by_player["Dragic, Goran"].reason == "Not With Team"


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
