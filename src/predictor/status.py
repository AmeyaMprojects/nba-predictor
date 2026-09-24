"""Plain-English data-health reporting.

This is the ONLY place a human ever learns something in the pipeline
broke. `poll-news` runs three times a day, unattended, via launchd
(scripts/com.predictor.daily.plist); its stdout/stderr go to
data/logs/*.log, which nobody reads. `status` is the resolution of that
gap: it surfaces staleness loudly, in one glance, instead of silently.

The governing rule for the whole project is "never silently publish stale
data -- surface loudly." A report that cries wolf on every run teaches the
user to ignore it, which defeats that rule just as surely as staying
quiet. So the thresholds and advice below are deliberately tuned against
the REAL refresh cadence of each source, not a single generic number:

- games: ingested in one bulk `ingest-season` call per season -- there is
  no scheduled job that touches this table at all. The NBA offseason
  (roughly mid-June to mid-October) is a real, expected ~120-day gap with
  nothing wrong. See STALENESS_HOURS below for the threshold chosen.
- injury_status: meant to refresh at least once a day (the 5pm backfill
  slot). As of this writing the NBA's injury-report CDN 403s everything
  after 2025-12-21, so this source IS currently, unavoidably ~9 months
  stale and CANNOT be fixed by re-running the backfill. That is left
  flagged on purpose -- hiding it would mean silently serving 9-month-old
  injury data to whatever consumes this table, which is exactly what the
  project's governing rule forbids. The advice text explains why blindly
  re-running may not help, instead of pretending it will.
- odds_snapshots: zero rows today because there is no ODDS_API_KEY, not
  because a feed broke. That is a different failure mode from "the feed
  stopped working" and gets different advice (get a key, vs. debug the
  feed) -- see `_advice`.
- news_items: the one source polled continuously (three times a day) and
  the one source that can NEVER be recovered retroactively once a poll is
  missed -- an outage here is the most urgent of the four, so it gets the
  tightest threshold relative to its cadence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from predictor.db import POINT_IN_TIME_TABLES

# Hours after which each source's newest `observed_at` counts as stale.
# See the module docstring for why each value was chosen; none of these
# are the brief's original blanket numbers except injury_status and
# odds_snapshots, which the reasoning above happens to leave unchanged.
STALENESS_HOURS: dict[str, float] = {
    # A 48h threshold (the brief's original value) would report STALE for
    # roughly a third of every year, during a completely normal offseason
    # -- that trains the user to ignore the whole report. 150 days (3600h)
    # comfortably covers a normal ~120-day offseason with headroom, while
    # still catching "a new season started and ingest-season was never
    # run" within about a month of the season opening.
    "games": 24 * 150,
    # Daily cadence (one backfill slot/day) plus one missed day of slack.
    "injury_status": 36,
    # No scheduled job exists for odds yet; kept at the same "at least
    # daily" cadence as injury_status as a placeholder for once ingest-odds
    # is actually scheduled. In practice this rarely matters: row_count is
    # 0 today, and an empty table is always reported stale regardless of
    # this threshold (see check_sources).
    "odds_snapshots": 36,
    # Polled at 09:00/14:00/19:00 local (scripts/com.predictor.daily.plist);
    # the longest normal gap between runs is ~14h (19:00 -> next 09:00).
    # 24h gives ~10h of slack for one delayed/missed run before alarming.
    "news_items": 24,
}


@dataclass(frozen=True)
class SourceHealth:
    name: str
    latest: datetime | None
    row_count: int
    age_hours: float | None
    stale: bool
    advice: str


def _next_season_label(now: datetime) -> str:
    """Best-guess season string (e.g. "2026-27") for the games advice.

    NBA seasons start in October and are labeled by their two years. From
    July onward, the upcoming season is the one worth ingesting next;
    before July, the season already in progress (started the previous
    October) is. This is a heuristic based only on the calendar -- this
    project deliberately has no schedule/calendar data source of its own
    (Basketball-Reference scraping is explicitly deferred to a later
    sub-project), so it cannot know the *actual* season boundaries.
    """
    start_year = now.year if now.month >= 7 else now.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def _advice(name: str, latest: datetime | None, now: datetime) -> str:
    """Advice shown only when a source is stale. See module docstring."""
    if name == "games":
        return f"Run: predictor ingest-season {_next_season_label(now)}"

    if name == "injury_status":
        if latest is None:
            return "Run: predictor backfill-injuries --start 2019-12-01"
        start = (latest.date() + timedelta(days=1)).isoformat()
        return (
            f"Run: predictor backfill-injuries --start {start}. If this "
            "keeps reporting the same dates as unavailable rather than "
            "'not published', the NBA's injury-report source is likely "
            "blocking recent requests again (a known recurring issue) -- "
            "check data/logs/*.log for repeated 403s before re-running "
            "many times; that is not fixable by retrying alone."
        )

    if name == "odds_snapshots":
        # A missing key and a broken feed need different advice -- see
        # module docstring.
        if os.environ.get("ODDS_API_KEY"):
            return (
                "ODDS_API_KEY is set but odds_snapshots is still "
                "empty/stale -- run: predictor ingest-odds directly and "
                "read the error (quota exceeded, invalid key, transient "
                "failure, etc.)."
            )
        return (
            "No ODDS_API_KEY set -- get a free key at "
            "https://the-odds-api.com, export ODDS_API_KEY, then run: "
            "predictor ingest-odds"
        )

    if name == "news_items":
        return (
            "Run: predictor poll-news, and confirm the launchd agent is "
            "loaded (launchctl list | grep com.predictor.daily) with an "
            "empty data/logs/daily.err.log -- news cannot be recovered "
            "retroactively once a poll is missed."
        )

    return ""


def check_sources(con, now: datetime | None = None) -> list[SourceHealth]:
    if now is None:
        now = datetime.now(UTC)

    out: list[SourceHealth] = []
    for name in sorted(POINT_IN_TIME_TABLES):
        # Resolved through POINT_IN_TIME_TABLES rather than spelled as a
        # literal here -- the physical "_raw" table names are only
        # allowed to appear as string literals in db.py/asof.py (see
        # test_no_physical_table_name_appears_outside_db_and_asof); this
        # module must not name a physical table directly either. Mirrors
        # the pattern used by predictor.sources.{injury_report,nba_stats,
        # odds,news_rss}. status.py reads the raw tables directly (not
        # through AsOfView) because it needs the TRUE current freshness,
        # not a point-in-time snapshot as of some cutoff.
        physical = POINT_IN_TIME_TABLES[name]
        count, latest = con.execute(
            f"SELECT count(*), max(observed_at) FROM {physical}"
        ).fetchone()

        if latest is None:
            out.append(
                SourceHealth(name, None, 0, None, True, _advice(name, None, now))
            )
            continue

        age = (now - latest).total_seconds() / 3600
        threshold = STALENESS_HOURS.get(name, 48)
        stale = age > threshold
        advice = _advice(name, latest, now) if stale else ""
        out.append(SourceHealth(name, latest, count, age, stale, advice))

    return out


def _format_age(hours: float) -> str:
    if hours < 0:
        # A derived/fixture observed_at can legitimately land in the
        # future (e.g. a scheduled game more than 7 days out, ingested
        # ahead of time) -- not a bug, just worth calling out plainly
        # rather than printing a confusing negative number.
        return "in the future"
    if hours < 48:
        return f"{hours:.0f}h"
    return f"{hours / 24:.0f}d"


def format_report(health: list[SourceHealth]) -> str:
    problems = [h for h in health if h.stale]
    if problems:
        lines = [
            f"PROBLEMS: {len(problems)} of {len(health)} data source(s) "
            "need attention"
        ]
    else:
        lines = ["ALL OK: every data source is fresh"]
    lines.append("")

    for h in health:
        mark = "STALE" if h.stale else "OK"
        if h.latest is None:
            detail = "no data at all"
        else:
            detail = (
                f"{h.row_count:,} rows, newest {_format_age(h.age_hours)} old "
                f"({h.latest.date()})"
            )
        lines.append(f"  [{mark:<5}] {h.name}: {detail}")
        if h.advice:
            lines.append(f"            -> {h.advice}")

    return "\n".join(lines)
