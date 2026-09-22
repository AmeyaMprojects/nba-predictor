from __future__ import annotations

from datetime import UTC, date, datetime

import requests

from predictor import raw_store

BASE_URL = "https://ak-static.cms.nba.com/referee/injury"

HOUR_LABELS: tuple[str, ...] = tuple(
    f"{h:02d}{suffix}" for suffix in ("AM", "PM") for h in list(range(1, 13))
)

# Archive verified to begin here; earlier dates return 403.
ARCHIVE_START = date(2019, 12, 1)


def report_url(day: date, hour_label: str) -> str:
    return f"{BASE_URL}/Injury-Report_{day.isoformat()}_{hour_label}.pdf"


def raw_key(day: date, hour_label: str) -> str:
    return f"Injury-Report_{day.isoformat()}_{hour_label}.pdf"


def fetch_report(day: date, hour_label: str, session=None) -> bytes | None:
    session = session or requests.Session()
    response = session.get(report_url(day, hour_label), timeout=30)
    if response.status_code != 200:
        return None
    return response.content


def archive_report(
    day: date,
    hour_label: str,
    now: datetime | None = None,
    session=None,
) -> bool:
    key = raw_key(day, hour_label)
    if raw_store.exists("injury", key):
        return False

    content = fetch_report(day, hour_label, session)
    if content is None or not content.startswith(b"%PDF"):
        return False

    raw_store.store(
        "injury",
        key,
        content,
        now or datetime.now(UTC),
        meta={"day": day.isoformat(), "hour_label": hour_label},
    )
    return True
