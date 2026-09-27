"""Read-only access to the real archive for archive-wide invariant tests.

Never db.connect(): its rail exists to stop a test from migrating or
writing the real database, not from reading it.
"""

import duckdb
import pytest

from predictor.config import PROJECT_ROOT

REAL_ARCHIVE = PROJECT_ROOT / "data" / "predictor.duckdb"


def open_real_archive_or_skip():
    if not REAL_ARCHIVE.exists():
        pytest.skip("real archive not present in this environment")
    try:
        con = duckdb.connect(str(REAL_ARCHIVE), read_only=True)
    except duckdb.Error as exc:  # pragma: no cover - timing-dependent
        # The scheduled news/schedule jobs hold the write lock briefly.
        pytest.skip(f"real archive is locked by another process: {exc}")
    con.execute("SET TimeZone='UTC'")
    has_schedule = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = 'schedule_raw'"
    ).fetchone()[0]
    if not has_schedule:
        con.close()
        pytest.skip("real archive has no schedule yet -- run predictor ingest-schedule")
    return con
