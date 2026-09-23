from datetime import date

import pytest

from predictor import db, raw_store
from predictor.config import Settings
from predictor.sources import injury_report

FIXTURE_BYTES = (
    __import__("pathlib").Path(__file__).parent
    / "fixtures"
    / "Injury-Report_2025-01-15_05PM.pdf"
).read_bytes()


@pytest.fixture
def env(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(raw_store, "settings", s)
    con = db.connect(tmp_path / "t.duckdb")
    db.migrate(con)
    return con


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    """No test in this module may block on a real sleep -- this stubs the
    indirection that fetch_report's internal retry/backoff uses, so a
    slot that triggers TransientFetchError's retries doesn't slow the
    suite down or (worse) actually wait on a bad Retry-After value.
    """
    monkeypatch.setattr(injury_report, "_sleep", lambda seconds: None)


class FakeResponse:
    def __init__(self, status_code, content=b""):
        self.status_code = status_code
        self.content = content
        self.headers = {}


class FakeSession:
    def __init__(self, available):
        self.available = available
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        if url in self.available:
            return FakeResponse(200, FIXTURE_BYTES)
        return FakeResponse(403)


class MappingSession:
    """Answers per-URL with an arbitrary status, for exercising 429/5xx
    (transient) responses alongside normal 200/403 responses in the same
    run.
    """

    def __init__(self, mapping, default_status=403):
        self.mapping = mapping
        self.default_status = default_status
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        if url in self.mapping:
            status, content = self.mapping[url]
            return FakeResponse(status, content)
        return FakeResponse(self.default_status)


def test_backfill_counts_missing_slots(env):
    session = FakeSession(available=set())
    stats = injury_report.backfill_range(
        env, date(2025, 1, 15), date(2025, 1, 15), ["05PM"], delay=0, session=session
    )
    assert stats["missing"] == 1
    assert stats["fetched"] == 0


def test_backfill_fetches_and_ingests_available_slots(env):
    url = injury_report.report_url(date(2025, 1, 15), "05PM")
    session = FakeSession(available={url})
    stats = injury_report.backfill_range(
        env, date(2025, 1, 15), date(2025, 1, 15), ["05PM"], delay=0, session=session
    )
    assert stats["fetched"] == 1
    assert stats["ingested"] == 161


def test_backfill_is_resumable_and_skips_archived_days(env):
    url = injury_report.report_url(date(2025, 1, 15), "05PM")
    session = FakeSession(available={url})
    args = (env, date(2025, 1, 15), date(2025, 1, 15), ["05PM"])
    injury_report.backfill_range(*args, delay=0, session=session)
    stats = injury_report.backfill_range(*args, delay=0, session=session)
    assert stats["skipped"] == 1
    assert len(session.calls) == 1, "resumed run re-fetched an archived report"


def test_backfill_returns_all_required_buckets(env):
    session = FakeSession(available=set())
    stats = injury_report.backfill_range(
        env, date(2025, 1, 15), date(2025, 1, 15), ["05PM"], delay=0, session=session
    )
    for key in ("fetched", "skipped", "missing", "ingested", "transient", "parse_failed"):
        assert key in stats


def test_backfill_counts_transient_and_continues_to_next_slot(env):
    """A TransientFetchError on one slot (429/5xx exhausting retries) must
    be counted in its own bucket, must NOT be folded into `missing`, and
    must NOT abort the sweep -- the very next slot in the range still gets
    processed normally.
    """
    day_a, day_b = date(2025, 1, 15), date(2025, 1, 16)
    url_a = injury_report.report_url(day_a, "05PM")
    url_b = injury_report.report_url(day_b, "05PM")
    session = MappingSession(
        {
            url_a: (429, b""),
            url_b: (200, FIXTURE_BYTES),
        }
    )

    stats = injury_report.backfill_range(env, day_a, day_b, ["05PM"], delay=0, session=session)

    assert stats["transient"] == 1
    assert stats["missing"] == 0, "a transient failure must never be counted as absent"
    assert stats["fetched"] == 1
    assert stats["ingested"] == 161

    # The failed slot must not have been archived -- it was never
    # confirmed present or absent.
    assert not raw_store.exists("injury", injury_report.raw_key(day_a, "05PM"))


def test_backfill_transient_slot_is_retried_on_rerun(env):
    """Unlike a parse failure, a transient slot is never archived, so a
    re-run of the same range must retry it (not silently skip it).
    """
    day = date(2025, 1, 15)
    url = injury_report.report_url(day, "05PM")

    failing_session = MappingSession({url: (503, b"")})
    stats1 = injury_report.backfill_range(
        env, day, day, ["05PM"], delay=0, session=failing_session
    )
    assert stats1["transient"] == 1

    recovered_session = FakeSession(available={url})
    stats2 = injury_report.backfill_range(
        env, day, day, ["05PM"], delay=0, session=recovered_session
    )
    assert stats2["skipped"] == 0, "a transient slot must be retried, not skipped"
    assert stats2["fetched"] == 1
    assert stats2["ingested"] == 161


def test_backfill_counts_parse_failed_and_continues(env, monkeypatch):
    """A documented InjuryReportParseError during ingestion (zero rows
    extracted) must be counted separately from `missing`/`transient`, must
    not abort the run, and must not be silently treated as a successful
    ingest of 0 rows.
    """
    url = injury_report.report_url(date(2025, 1, 15), "05PM")
    session = FakeSession(available={url})

    def raising_ingest(con, pdf_bytes):
        raise injury_report.InjuryReportParseError("simulated: zero rows extracted")

    monkeypatch.setattr(injury_report, "ingest_report", raising_ingest)

    stats = injury_report.backfill_range(
        env, date(2025, 1, 15), date(2025, 1, 15), ["05PM"], delay=0, session=session
    )

    assert stats["parse_failed"] == 1
    assert stats["fetched"] == 1, "the PDF was still fetched/archived successfully"
    assert stats["ingested"] == 0
    assert stats["missing"] == 0


def test_backfill_treats_unexpected_ingest_exception_as_parse_failed(env, monkeypatch):
    """Guards the noted pre-existing gap: a zero-page PDF currently raises
    a plain IndexError from pdfplumber, not InjuryReportParseError. That
    must still be caught per-slot and bucketed (as parse_failed) rather
    than crash the whole multi-thousand-slot backfill.
    """
    url = injury_report.report_url(date(2025, 1, 15), "05PM")
    session = FakeSession(available={url})

    def raising_ingest(con, pdf_bytes):
        raise IndexError("simulated: pdf.pages[0] on a zero-page PDF")

    monkeypatch.setattr(injury_report, "ingest_report", raising_ingest)

    stats = injury_report.backfill_range(
        env, date(2025, 1, 15), date(2025, 1, 15), ["05PM"], delay=0, session=session
    )

    assert stats["parse_failed"] == 1
    assert stats["fetched"] == 1
    assert stats["ingested"] == 0


def test_backfill_parse_failed_slot_is_skipped_not_retried_on_rerun(env, monkeypatch):
    """Documents a known asymmetry (see backfill_range's docstring): a
    parse-failed slot IS already archived to disk (unlike a transient
    slot), so a re-run of the same range reports it as `skipped`, NOT as
    a retried ingest -- even once whatever caused the parse failure is
    fixed. Recovering it needs a separate re-ingest-from-raw-store path,
    out of scope here.
    """
    url = injury_report.report_url(date(2025, 1, 15), "05PM")
    session = FakeSession(available={url})
    original_ingest = injury_report.ingest_report
    calls = {"n": 0}

    def flaky_then_fixed(con, pdf_bytes):
        calls["n"] += 1
        if calls["n"] == 1:
            raise injury_report.InjuryReportParseError("simulated transient parse bug")
        return original_ingest(con, pdf_bytes)

    monkeypatch.setattr(injury_report, "ingest_report", flaky_then_fixed)

    args = (env, date(2025, 1, 15), date(2025, 1, 15), ["05PM"])
    stats1 = injury_report.backfill_range(*args, delay=0, session=session)
    assert stats1["parse_failed"] == 1
    assert stats1["ingested"] == 0

    stats2 = injury_report.backfill_range(*args, delay=0, session=session)
    assert stats2["skipped"] == 1, "an already-archived parse-failed slot is skipped, not retried"
    assert stats2["ingested"] == 0
    assert calls["n"] == 1, "ingest was never retried for the already-archived slot"
