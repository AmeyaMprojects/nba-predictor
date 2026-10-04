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
  missed -- an outage here is the most urgent of the five, so it gets the
  tightest threshold relative to its cadence.
- schedule: fetched once a day by its own launchd job. A missed day loses
  that day's schedule vintage (when a game moved, and when that became
  knowable) for good, so it gets the same one-missed-run threshold as
  injury_status.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from predictor.config import season_label
from predictor.db import POINT_IN_TIME_TABLES
from predictor.model import live, publish

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
    # Fetched once a day at 10:30 local (scripts/com.predictor.schedule.plist).
    # One missed run of slack before alarming, same as injury_status.
    "schedule": 36,
}


@dataclass(frozen=True)
class SourceHealth:
    name: str
    latest: datetime | None
    row_count: int
    age_hours: float | None
    stale: bool
    advice: str


def _advice(name: str, latest: datetime | None, now: datetime) -> str:
    """Advice shown only when a source is stale. See module docstring."""
    if name == "games":
        return f"Run: predictor ingest-season {season_label(now)}"

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

    if name == "schedule":
        return (
            "Run: predictor ingest-schedule, and confirm the launchd agent "
            "is loaded (launchctl list | grep com.predictor.schedule). Each "
            "missed day is a day of schedule history that cannot be "
            "recaptured later."
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


# `prediction_log`: the longest acceptable gap between "there are games
# today or yesterday (ET)" and the newest logged prediction -- see
# `check_live`.
PREDICTION_LOG_STALE_HOURS = 30


def check_live(con, repo_dir: Path, now: datetime | None = None) -> list[SourceHealth]:
    """Health of the three live-operation pieces that `check_sources` (table
    freshness) cannot see: whether results are actually being captured,
    whether today's predictions are actually being logged, and whether the
    log is actually reaching GitHub. Appended after the table sources by the
    `status` command so a human sees the whole pipeline in one report.

    `live_results` staleness is driven entirely by
    `predictor.model.live.results_missing` (any competitive game that tipped
    off between 12h and 3 days ago with no FINAL row at all) -- NOT by the
    age of the last capture. A long-idle but otherwise healthy capture job
    (e.g. a multi-day All-Star break with no games) must not be flagged just
    because `last_capture` is old; `results_missing` already returns empty
    in that case. `latest` still reports the true last capture time (via
    `last_capture`) for visibility, completely independent of the stale
    verdict.
    """
    if now is None:
        now = datetime.now(UTC)

    out: list[SourceHealth] = []

    # --- live_results --------------------------------------------------
    cap = live.last_capture(con)
    missing = live.results_missing(con, now)
    stale_results = bool(missing)
    age = (now - cap).total_seconds() / 3600 if cap is not None else None
    advice_results = (
        "Run: predictor capture-results, and confirm the launchd agent "
        "com.predictor.results is loaded."
        if stale_results
        else ""
    )
    out.append(
        SourceHealth("live_results", cap, len(missing), age, stale_results, advice_results)
    )

    # --- prediction_log --------------------------------------------------
    season = season_label(now)
    log = live.read_log(live.log_path(repo_dir, season))
    latest_pred = max(
        (datetime.fromisoformat(line["predicted_at"]) for line in log), default=None
    )
    has_recent_games = bool(live.slate_for(con, now)) or bool(
        live.slate_for(con, now - timedelta(days=1))
    )
    stale_log = False
    if live.in_season(con, now) and has_recent_games:
        if latest_pred is None or (now - latest_pred) > timedelta(hours=PREDICTION_LOG_STALE_HOURS):
            stale_log = True
    age_log = (now - latest_pred).total_seconds() / 3600 if latest_pred is not None else None
    advice_log = (
        "Run: predictor predict-today, and confirm com.predictor.predict is loaded."
        if stale_log
        else ""
    )
    out.append(
        SourceHealth("prediction_log", latest_pred, len(log), age_log, stale_log, advice_log)
    )

    # --- log_published -----------------------------------------------------
    unpushed = publish.unpushed_commits(repo_dir)
    if unpushed is None:
        stale_pub = True
        # -1 is a sentinel, not a count: `unpushed_commits` returned None
        # (no remote configured, or any other git error), so there is no
        # count to report -- see format_report's log_published special
        # case, which reads this sentinel to pick the right detail line.
        row_count_pub = -1
        advice_pub = (
            "no GitHub remote configured for this checkout -- add one with: "
            "git remote add origin <url>, then push; if a remote IS "
            "configured, inspect the repository by hand (git status, "
            "git remote -v)."
        )
    elif unpushed > 0:
        stale_pub = True
        row_count_pub = unpushed
        advice_pub = (
            f"{unpushed} commit(s) are committed locally but not pushed -- "
            "run: git push origin main, or check the GitHub login: gh auth "
            "status."
        )
    else:
        stale_pub = False
        row_count_pub = 0
        advice_pub = ""
    out.append(
        SourceHealth("log_published", None, row_count_pub, None, stale_pub, advice_pub)
    )

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
        if h.name == "log_published":
            # log_published has no natural timestamp at all (a commit count,
            # not a freshness clock), so it never goes through the
            # `latest is None` branch below -- it gets its own wording keyed
            # off `row_count`: the actual unpushed count when known, or the
            # -1 sentinel `check_live` uses for "unpushed_commits() returned
            # None" (no remote configured, or any other git error).
            if h.row_count < 0:
                detail = "no GitHub remote configured (or git could not be read)"
            elif h.row_count > 0:
                detail = f"{h.row_count} commit(s) waiting to be pushed"
            else:
                detail = "nothing waiting to be pushed"
        elif h.latest is None:
            # Existing table sources (check_sources) always treat latest=None
            # as stale (an empty table). The only check_live entry that can
            # still reach this branch is prediction_log with an empty/missing
            # log file -- `stale` there also depends on whether there are
            # recent games at all (see check_live), so `latest is None` and
            # `stale is False` both commonly hold together in the off-season
            # ("n/a": nothing predicted yet, and nothing is wrong).
            detail = "no data at all" if h.stale else "n/a"
        else:
            detail = (
                f"{h.row_count:,} rows, newest {_format_age(h.age_hours)} old "
                f"({h.latest.date()})"
            )
        lines.append(f"  [{mark:<5}] {h.name}: {detail}")
        if h.advice:
            lines.append(f"            -> {h.advice}")

    return "\n".join(lines)
