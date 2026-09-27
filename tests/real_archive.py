"""Read-only access to the real archive for archive-wide invariant tests.

Never db.connect(): its rail exists to stop a test from migrating or
writing the real database, not from reading it.
"""

import time

import duckdb
import pytest

from predictor.config import PROJECT_ROOT

REAL_ARCHIVE = PROJECT_ROOT / "data" / "predictor.duckdb"


LOCK_ATTEMPTS = 6
LOCK_WAIT_SECONDS = 10.0


def _connect_read_only_waiting_out_locks(sleep=time.sleep):
    """Read-only connect, retrying while a scheduled job holds the write lock.

    The news/schedule jobs hold the lock only briefly, so wait it out
    rather than silently skipping the real-archive checks. Skip only after
    LOCK_ATTEMPTS tries, LOCK_WAIT_SECONDS apart.
    """
    for attempt in range(1, LOCK_ATTEMPTS + 1):
        try:
            return duckdb.connect(str(REAL_ARCHIVE), read_only=True)
        except duckdb.Error as exc:  # pragma: no cover - timing-dependent
            if attempt == LOCK_ATTEMPTS:
                pytest.skip(
                    f"real archive is locked by another process after "
                    f"{LOCK_ATTEMPTS} tries: {exc}"
                )
            sleep(LOCK_WAIT_SECONDS)
    raise AssertionError("unreachable")


def open_real_archive_or_skip():
    if not REAL_ARCHIVE.exists():
        pytest.skip("real archive not present in this environment")
    con = _connect_read_only_waiting_out_locks()
    con.execute("SET TimeZone='UTC'")
    has_schedule = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = 'schedule_raw'"
    ).fetchone()[0]
    if not has_schedule:
        con.close()
        pytest.skip("real archive has no schedule yet -- run predictor ingest-schedule")
    return con
