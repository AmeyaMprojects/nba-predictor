from datetime import UTC, date, datetime, timedelta

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


def test_backfill_bad_content_is_not_counted_as_missing(env):
    """A 200 response with a non-PDF body (e.g. a CDN block/challenge page)
    must be bucketed as `bad_content`, not folded into `missing`. Those two
    outcomes look identical over the wire but mean very different things:
    a confirmed 403/404 vs. "the server answered with something else".
    """
    day, hour = date(2025, 1, 15), "05PM"
    url = injury_report.report_url(day, hour)
    session = MappingSession({url: (200, b"<html>blocked</html>")})

    stats = injury_report.backfill_range(env, day, day, [hour], delay=0, session=session)

    assert stats["bad_content"] == 1
    assert stats["missing"] == 0, "a 200-with-garbage-body must never be counted as absent"
    assert stats["fetched"] == 0
    assert not raw_store.exists("injury", injury_report.raw_key(day, hour))


def test_backfill_genuine_404_is_still_missing(env):
    """A real 403/404 must still land in `missing`, unaffected by the new
    bad_content bucket.
    """
    day, hour = date(2025, 1, 15), "05PM"
    url = injury_report.report_url(day, hour)
    session = MappingSession({url: (404, b"")})

    stats = injury_report.backfill_range(env, day, day, [hour], delay=0, session=session)

    assert stats["missing"] == 1
    assert stats["bad_content"] == 0


def test_backfill_bad_content_does_not_abort_the_run(env):
    """A bad-content slot on one day must not stop the sweep from
    processing the next slot in the range.
    """
    day_a, day_b = date(2025, 1, 15), date(2025, 1, 16)
    url_a = injury_report.report_url(day_a, "05PM")
    url_b = injury_report.report_url(day_b, "05PM")
    session = MappingSession(
        {
            url_a: (200, b"<html>blocked</html>"),
            url_b: (200, FIXTURE_BYTES),
        }
    )

    stats = injury_report.backfill_range(env, day_a, day_b, ["05PM"], delay=0, session=session)

    assert stats["bad_content"] == 1
    assert stats["fetched"] == 1
    assert stats["ingested"] == 161


def test_backfill_buckets_reconcile_with_slots_attempted(env):
    """skipped + missing + fetched + transient + parse_failed + bad_content
    must equal the number of slots attempted. Uses one slot per distinct
    outcome (no slot is both `fetched` and `parse_failed` here, which would
    double-count a single slot against two buckets -- a documented,
    pre-existing, deliberate asymmetry, not something this test exercises).
    """
    already_archived_day = date(2025, 1, 10)
    missing_day = date(2025, 1, 11)
    fetched_day = date(2025, 1, 12)
    transient_day = date(2025, 1, 13)
    bad_content_day = date(2025, 1, 14)

    # Pre-populate one slot so it will be `skipped`.
    pre_session = FakeSession(
        available={injury_report.report_url(already_archived_day, "05PM")}
    )
    injury_report.backfill_range(
        env, already_archived_day, already_archived_day, ["05PM"], delay=0, session=pre_session
    )

    mapping = {
        injury_report.report_url(fetched_day, "05PM"): (200, FIXTURE_BYTES),
        injury_report.report_url(transient_day, "05PM"): (503, b""),
        injury_report.report_url(bad_content_day, "05PM"): (200, b"<html>blocked</html>"),
        injury_report.report_url(missing_day, "05PM"): (404, b""),
    }
    session = MappingSession(mapping)

    stats = injury_report.backfill_range(
        env, already_archived_day, bad_content_day, ["05PM"], delay=0, session=session
    )

    slots_attempted = (bad_content_day - already_archived_day).days + 1
    total = (
        stats["skipped"]
        + stats["missing"]
        + stats["fetched"]
        + stats["transient"]
        + stats["parse_failed"]
        + stats["bad_content"]
    )
    assert total == slots_attempted
    assert stats["skipped"] == 1
    assert stats["missing"] == 1
    assert stats["fetched"] == 1
    assert stats["transient"] == 1
    assert stats["bad_content"] == 1
    assert stats["parse_failed"] == 0


def test_backfill_skips_politeness_delay_when_no_request_was_made(env, monkeypatch):
    """A slot before ARCHIVE_START short-circuits inside `_archive` before
    any network call -- the caller must not pay the politeness `delay` for
    a slot that never touched the network.
    """
    sleeps: list[float] = []
    monkeypatch.setattr(injury_report.time, "sleep", lambda s: sleeps.append(s))

    day = injury_report.ARCHIVE_START - timedelta(days=1)
    session = FakeSession(available=set())

    stats = injury_report.backfill_range(
        env, day, day, ["05PM"], delay=5.0, session=session
    )

    assert stats["missing"] == 1
    assert session.calls == [], "a pre-ARCHIVE_START slot must not make a request"
    assert sleeps == [], "no request was made, so no politeness delay should be paid"


def test_reingest_archived_ingests_without_any_network_call(env):
    """`reingest_archived` must recover an already-archived report using
    only raw_store + the DB connection -- no session, no network fetch.
    """
    day, hour = date(2025, 1, 15), "05PM"
    key = injury_report.raw_key(day, hour)
    raw_store.store(
        "injury",
        key,
        FIXTURE_BYTES,
        datetime(2025, 1, 15, 22, 0, tzinfo=UTC),
        meta={"day": day.isoformat(), "hour_label": hour},
    )

    stats = injury_report.reingest_archived(env)

    assert stats["found"] == 1
    assert stats["ingested_ok"] == 1
    assert stats["still_failed"] == 0
    assert stats["rows_written"] == 161

    row_count = env.execute(
        f"SELECT COUNT(*) FROM {db.POINT_IN_TIME_TABLES['injury_status']}"
    ).fetchone()[0]
    assert row_count == 161


def test_reingest_archived_is_idempotent_on_second_run(env):
    day, hour = date(2025, 1, 15), "05PM"
    key = injury_report.raw_key(day, hour)
    raw_store.store(
        "injury",
        key,
        FIXTURE_BYTES,
        datetime(2025, 1, 15, 22, 0, tzinfo=UTC),
        meta={"day": day.isoformat(), "hour_label": hour},
    )

    injury_report.reingest_archived(env)
    stats2 = injury_report.reingest_archived(env)

    assert stats2["found"] == 1
    assert stats2["ingested_ok"] == 1
    assert stats2["still_failed"] == 0

    row_count = env.execute(
        f"SELECT COUNT(*) FROM {db.POINT_IN_TIME_TABLES['injury_status']}"
    ).fetchone()[0]
    assert row_count == 161, "re-ingesting the same report must not duplicate rows"


def test_reingest_archived_reports_still_unparseable_report_without_aborting(env):
    """A still-broken archived report must be logged and skipped, not raise
    out of `reingest_archived`, and must not stop the rest of the sweep.
    """
    good_day, bad_day = date(2025, 1, 15), date(2025, 1, 16)
    good_key = injury_report.raw_key(good_day, "05PM")
    bad_key = injury_report.raw_key(bad_day, "05PM")

    raw_store.store(
        "injury",
        bad_key,
        b"%PDF-not-actually-a-valid-pdf",
        datetime(2025, 1, 16, 22, 0, tzinfo=UTC),
        meta={"day": bad_day.isoformat(), "hour_label": "05PM"},
    )
    raw_store.store(
        "injury",
        good_key,
        FIXTURE_BYTES,
        datetime(2025, 1, 15, 22, 0, tzinfo=UTC),
        meta={"day": good_day.isoformat(), "hour_label": "05PM"},
    )

    stats = injury_report.reingest_archived(env)

    assert stats["found"] == 2
    assert stats["ingested_ok"] == 1
    assert stats["still_failed"] == 1
    assert stats["rows_written"] == 161


def test_reingest_archived_respects_date_range_filter(env):
    in_range, out_of_range = date(2025, 1, 15), date(2025, 2, 1)
    raw_store.store(
        "injury",
        injury_report.raw_key(in_range, "05PM"),
        FIXTURE_BYTES,
        datetime(2025, 1, 15, 22, 0, tzinfo=UTC),
        meta={"day": in_range.isoformat(), "hour_label": "05PM"},
    )
    raw_store.store(
        "injury",
        injury_report.raw_key(out_of_range, "05PM"),
        FIXTURE_BYTES,
        datetime(2025, 2, 1, 22, 0, tzinfo=UTC),
        meta={"day": out_of_range.isoformat(), "hour_label": "05PM"},
    )

    stats = injury_report.reingest_archived(env, start=date(2025, 1, 1), end=date(2025, 1, 31))

    assert stats["found"] == 1
    assert stats["ingested_ok"] == 1


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
