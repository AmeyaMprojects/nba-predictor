"""Cloud injury reports kept as deltas, and their expansion on rebuild."""

import json
import re
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from predictor import db
from predictor.cloud import injuries, state
from predictor.cloud.injuries import (
    InjuryOutcome,
    canonical_rows,
    content_sha,
    delta,
    expand_into,
    record_report,
)
from predictor.sources import injury_report
from predictor.sources.injury_report import InjuryRow, LatestReport, ParsedReport

FIXTURE = Path(__file__).parent / "fixtures" / "Injury-Report_2025-01-15_05PM.pdf"
REPORTS = db.POINT_IN_TIME_TABLES["injury_reports"]
STATUS = db.POINT_IN_TIME_TABLES["injury_status"]


def dt(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)


def row(player: str, status: str, team: str = "Boston Celtics", reason: str = "Injury/Illness"):
    return InjuryRow(
        game_date=date(2026, 10, 22),
        game_time="07:30(ET)",
        matchup="BOS@NYK",
        team=team,
        player=player,
        status=status,
        reason=reason,
    )


@pytest.fixture
def con(tmp_path):
    c = db.connect(tmp_path / "live.duckdb")
    db.migrate(c)
    yield c
    c.close()


@pytest.fixture
def fake_reports(monkeypatch):
    """body -> ParsedReport (or an exception) for synthetic report bodies."""
    reports: dict[bytes, object] = {}
    real = injury_report.parse_report

    def parse(body: bytes):
        if body not in reports:
            return real(body)
        value = reports[body]
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(injury_report, "parse_report", parse)
    return reports


def latest(body: bytes, key: str = "Injury-Report_2026-10-22_05_30PM.pdf", fetched: str = "2026-10-22T21:40Z"):
    return LatestReport(key=key, body=body, fetched_at=dt(fetched))


def status_rows_at(con, at: datetime) -> list[tuple]:
    return con.execute(
        f"SELECT team, player, status FROM {STATUS} WHERE observed_at = ? ORDER BY player", [at]
    ).fetchall()


def state_lines(state_dir: Path) -> list[dict]:
    out = []
    for path in sorted(state_dir.glob("injuries/*/*.jsonl")):
        out.extend(state.read_records(path))
    return out


R1, R2, R3 = dt("2026-10-22T21:30Z"), dt("2026-10-22T21:45Z"), dt("2026-10-22T22:00Z")


def three_reports(fake_reports):
    fake_reports[b"r1"] = ParsedReport(R1, [row("Doe, A", "Out"), row("Roe, B", "Questionable"), row("Poe, C", "Out")])
    fake_reports[b"r2"] = ParsedReport(R2, [row("Doe, A", "Out"), row("Roe, B", "Out")])
    fake_reports[b"r3"] = ParsedReport(R3, [row("Doe, A", "Out"), row("Roe, B", "Out")])


# --- pure helpers ---------------------------------------------------------


def test_canonical_rows_match_what_ingest_stores():
    parsed = ParsedReport(R1, [row("Doe, A", "Out")])
    (r,) = canonical_rows(parsed)
    assert r["team"] == "BOS" and r["team_display"] == "Boston Celtics"
    assert r["player"] == "Doe,A" and r["player_display"] == "Doe, A"
    assert r["game_date"] == "2026-10-22"
    assert r["status"] == "Out" and r["matchup"] == "BOS@NYK"


def test_content_sha_is_64_hex():
    rows = canonical_rows(ParsedReport(R1, [row("Doe, A", "Out")]))
    sha = content_sha(rows)
    assert re.fullmatch(r"[0-9a-f]{64}", sha)
    assert content_sha(list(reversed(rows + rows[:0]))) == sha
    assert content_sha([]) != sha


def test_delta_records_only_changes():
    prev = canonical_rows(ParsedReport(R1, [row("A, A", "Out"), row("B, B", "Questionable"), row("C, C", "Out")]))
    cur = canonical_rows(ParsedReport(R2, [row("A, A", "Out"), row("B, B", "Out")]))
    lines = delta(prev, cur, R2)
    assert len(lines) == 2
    assert lines[0]["kind"] == "set" and lines[0]["player"] == "B,B" and lines[0]["status"] == "Out"
    assert lines[1] == {"kind": "drop", "game_date": "2026-10-22", "team": "BOS", "player": "C,C"}


# --- record_report ----------------------------------------------------------


def test_report_time_comes_from_header_not_filename(con, tmp_path):
    body = FIXTURE.read_bytes()
    fake = LatestReport(key="Injury-Report_2026-10-22_05_30PM.pdf", body=body, fetched_at=dt("2026-10-22T21:40Z"))
    out = record_report(con, tmp_path / "state", fake)
    expected = injury_report.parse_report(body).published_at
    assert out.new and out.report_at == expected
    assert out.rows > 0 and out.changes == out.rows
    got = con.execute(f"SELECT observed_at, source_key, row_count FROM {REPORTS}").fetchall()
    assert got == [(expected, fake.key, out.rows)]
    assert con.execute(f"SELECT count(*) FROM {STATUS} WHERE observed_at = ?", [expected]).fetchone()[0] == out.rows


def test_first_report_writes_report_and_set_lines(con, tmp_path, fake_reports):
    three_reports(fake_reports)
    sd = tmp_path / "state"
    out = record_report(con, sd, latest(b"r1"))
    assert out == InjuryOutcome(new=True, report_at=R1, rows=3, changes=3)
    files = sorted(sd.glob("injuries/*/*.jsonl"))
    assert [p.relative_to(sd).as_posix() for p in files] == ["injuries/2026-27/2026-10.jsonl"]
    lines = state_lines(sd)
    head = lines[0]
    assert head["kind"] == "report"
    assert head["report_at"] == R1.isoformat()
    assert head["fetched_at"] == dt("2026-10-22T21:40Z").isoformat()
    assert head["source_key"] == "Injury-Report_2026-10-22_05_30PM.pdf"
    assert head["rows"] == 3 and len(head["content_sha"]) == 64
    assert [line["kind"] for line in lines[1:]] == ["set"] * 3
    assert len(status_rows_at(con, R1)) == 3


def test_changed_report_writes_deltas_and_snapshot(con, tmp_path, fake_reports):
    three_reports(fake_reports)
    sd = tmp_path / "state"
    record_report(con, sd, latest(b"r1"))
    out = record_report(con, sd, latest(b"r2", fetched="2026-10-22T21:50Z"))
    assert out == InjuryOutcome(new=True, report_at=R2, rows=2, changes=2)
    kinds = [line["kind"] for line in state_lines(sd)]
    assert kinds == ["report", "set", "set", "set", "report", "set", "drop"]
    assert status_rows_at(con, R2) == [("BOS", "Doe,A", "Out"), ("BOS", "Roe,B", "Out")]


def test_unchanged_report_adds_only_a_report_line(con, tmp_path, fake_reports):
    three_reports(fake_reports)
    sd = tmp_path / "state"
    record_report(con, sd, latest(b"r2"))
    before = len(state_lines(sd))
    out = record_report(con, sd, latest(b"r3", fetched="2026-10-22T22:05Z"))
    assert out == InjuryOutcome(new=True, report_at=R3, rows=2, changes=0)
    lines = state_lines(sd)
    assert len(lines) == before + 1 and lines[-1]["kind"] == "report"
    assert lines[-1]["content_sha"] == lines[0]["content_sha"]
    assert status_rows_at(con, R3) == []
    assert con.execute(f"SELECT count(*) FROM {REPORTS}").fetchone()[0] == 2


def test_older_or_same_report_is_not_recorded_again(con, tmp_path, fake_reports):
    three_reports(fake_reports)
    sd = tmp_path / "state"
    record_report(con, sd, latest(b"r2"))
    snapshot = [p.read_bytes() for p in sorted(sd.glob("injuries/*/*.jsonl"))]
    same = record_report(con, sd, latest(b"r2", fetched="2026-10-22T22:10Z"))
    older = record_report(con, sd, latest(b"r1", fetched="2026-10-22T22:11Z"))
    assert same == InjuryOutcome(new=False, report_at=R2, rows=2, changes=0)
    assert older.new is False and older.report_at == R1
    assert [p.read_bytes() for p in sorted(sd.glob("injuries/*/*.jsonl"))] == snapshot
    assert con.execute(f"SELECT count(*) FROM {REPORTS}").fetchone()[0] == 1


def test_delta_follows_the_last_changed_report_after_unchanged_ones(con, tmp_path, fake_reports):
    three_reports(fake_reports)
    r4 = dt("2026-10-22T22:15Z")
    fake_reports[b"r4"] = ParsedReport(r4, [row("Doe, A", "Out"), row("Roe, B", "Out"), row("Zoe, D", "Doubtful")])
    sd = tmp_path / "state"
    for body in (b"r1", b"r2", b"r3"):
        record_report(con, sd, latest(body))
    out = record_report(con, sd, latest(b"r4"))
    assert out.changes == 1
    assert state_lines(sd)[-1]["player"] == "Zoe,D"


def test_empty_report_drops_every_previous_row(con, tmp_path, fake_reports):
    three_reports(fake_reports)
    sd = tmp_path / "state"
    record_report(con, sd, latest(b"r1"))
    empty = injury_report.InjuryReportEmptyError("no filings", published_at=R2)
    fake_reports[b"empty"] = empty
    out = record_report(con, sd, latest(b"empty"))
    assert out == InjuryOutcome(new=True, report_at=R2, rows=0, changes=3)
    assert [line["kind"] for line in state_lines(sd)[-4:]] == ["report", "drop", "drop", "drop"]
    assert con.execute(f"SELECT row_count FROM {REPORTS} WHERE observed_at = ?", [R2]).fetchone() == (0,)
    # the next report starts from nothing
    fake_reports[b"after"] = ParsedReport(R3, [row("Doe, A", "Out")])
    assert record_report(con, sd, latest(b"after")).changes == 1


def test_parse_failure_is_raised_and_nothing_is_recorded(con, tmp_path, fake_reports):
    fake_reports[b"bad"] = injury_report.InjuryReportParseError("no headers")
    sd = tmp_path / "state"
    with pytest.raises(injury_report.InjuryReportParseError):
        record_report(con, sd, latest(b"bad"))
    assert not list(sd.glob("injuries/**/*.jsonl"))
    assert con.execute(f"SELECT count(*) FROM {REPORTS}").fetchone()[0] == 0


def test_a_new_season_starts_from_empty(con, tmp_path, fake_reports):
    june = dt("2026-06-10T21:30Z")
    fake_reports[b"june"] = ParsedReport(june, [row("Doe, A", "Out")])
    fake_reports[b"oct"] = ParsedReport(R1, [row("Doe, A", "Out")])
    sd = tmp_path / "state"
    record_report(con, sd, latest(b"june"))
    out = record_report(con, sd, latest(b"oct"))
    assert out.changes == 1
    assert sorted(p.relative_to(sd).as_posix() for p in sd.glob("injuries/*/*.jsonl")) == [
        "injuries/2025-26/2026-06.jsonl",
        "injuries/2026-27/2026-10.jsonl",
    ]


def test_state_lines_are_canonical_json(con, tmp_path, fake_reports):
    three_reports(fake_reports)
    sd = tmp_path / "state"
    record_report(con, sd, latest(b"r1"))
    for line in (sd / "injuries/2026-27/2026-10.jsonl").read_text().splitlines():
        assert line == json.dumps(json.loads(line), sort_keys=True, separators=(",", ":"))


# --- expansion on rebuild ------------------------------------------------------


def test_rebuild_expands_snapshots(con, tmp_path, fake_reports):
    three_reports(fake_reports)
    sd = tmp_path / "state"
    for body in (b"r1", b"r2", b"r3"):
        record_report(con, sd, latest(body))
    report = state.rebuild_db(sd, tmp_path / "rebuilt.duckdb")
    assert report.rows["injury_status"] == 5
    assert report.rows["injury_reports"] == 3
    rebuilt = db.connect(tmp_path / "rebuilt.duckdb")
    try:
        assert len(status_rows_at(rebuilt, R1)) == 3
        assert len(status_rows_at(rebuilt, R2)) == 2
        assert status_rows_at(rebuilt, R3) == []
        assert rebuilt.execute(f"SELECT count(*) FROM {REPORTS}").fetchone()[0] == 3
        cols = ", ".join(state.INJURY_COLUMNS)
        q = f"SELECT {cols} FROM {STATUS} ORDER BY observed_at, player"
        assert rebuilt.execute(q).fetchall() == con.execute(q).fetchall()
        rq = f"SELECT * FROM {REPORTS} ORDER BY observed_at"
        assert rebuilt.execute(rq).fetchall() == con.execute(rq).fetchall()
    finally:
        rebuilt.close()


def test_expand_into_returns_rows_inserted(con, tmp_path, fake_reports):
    three_reports(fake_reports)
    sd = tmp_path / "state"
    for body in (b"r1", b"r2", b"r3"):
        record_report(con, sd, latest(body))
    fresh = db.connect(tmp_path / "fresh.duckdb")
    try:
        db.migrate(fresh)
        assert expand_into(fresh, sd) == 5
        assert expand_into(fresh, sd) == 0  # INSERT OR IGNORE
    finally:
        fresh.close()


def test_rebuild_keeps_mac_rows_for_a_report_in_both(con, tmp_path, fake_reports):
    three_reports(fake_reports)
    sd = tmp_path / "state"
    record_report(con, sd, latest(b"r1"))
    mac_rows = [
        {
            "report_date": date(2026, 10, 22), "game_date": date(2026, 10, 22),
            "game_time": "07:30(ET)", "matchup": "BOS@NYK", "team": "BOS",
            "team_display": "Boston Celtics", "player": "Doe,A", "player_display": "Doe, A",
            "status": "Out", "reason": "Mac reason", "reconstructed": False, "observed_at": R1,
        }
    ]
    state.write_snapshot(sd, "nba/injuries/2026-27/2026-10.jsonl", state.encode_injury_history(mac_rows))
    state.rebuild_db(sd, tmp_path / "rebuilt.duckdb")
    rebuilt = db.connect(tmp_path / "rebuilt.duckdb")
    try:
        reasons = rebuilt.execute(
            f"SELECT player, reason FROM {STATUS} WHERE observed_at = ? ORDER BY player", [R1]
        ).fetchall()
    finally:
        rebuilt.close()
    assert ("Doe,A", "Mac reason") in reasons
    assert len(reasons) == 3


def test_rebuild_after_an_empty_report(con, tmp_path, fake_reports):
    three_reports(fake_reports)
    sd = tmp_path / "state"
    record_report(con, sd, latest(b"r1"))
    fake_reports[b"empty"] = injury_report.InjuryReportEmptyError("none", published_at=R2)
    record_report(con, sd, latest(b"empty"))
    fake_reports[b"after"] = ParsedReport(R3, [row("Doe, A", "Out")])
    record_report(con, sd, latest(b"after"))
    state.rebuild_db(sd, tmp_path / "rebuilt.duckdb")
    rebuilt = db.connect(tmp_path / "rebuilt.duckdb")
    try:
        assert len(status_rows_at(rebuilt, R1)) == 3
        assert status_rows_at(rebuilt, R2) == []
        assert status_rows_at(rebuilt, R3) == [("BOS", "Doe,A", "Out")]
        counts = rebuilt.execute(f"SELECT row_count FROM {REPORTS} ORDER BY observed_at").fetchall()
        assert counts == [(3,), (0,), (1,)]
    finally:
        rebuilt.close()


def test_module_has_no_physical_table_names():
    source = Path(injuries.__file__).read_text()
    assert "_raw" not in source
