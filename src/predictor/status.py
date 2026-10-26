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
- odds_snapshots: holds both historical closing lines (Kaggle, imported
  once) and live lines (The Odds API, fetched daily). Table age says
  nothing useful about either, so this entry only reports what is stored
  and is never stale; live-odds freshness is judged once, by
  `check_live`'s `odds` entry (no live line in 30h while in season). A
  missing API key and a broken feed need different advice -- that entry's
  advice covers both.
- news_items: the one source polled continuously (three times a day) and
  the one source that can NEVER be recovered retroactively once a poll is
  missed -- an outage here is the most urgent of the five, so it gets the
  tightest threshold relative to its cadence.
- schedule: fetched once a day by its own launchd job. A missed day loses
  that day's schedule vintage (when a game moved, and when that became
  knowable) for good, so it gets the same one-missed-run threshold as
  injury_status.
- player_games: like games, loaded in bulk (`ingest-player-logs`) with no
  scheduled job, so it shares the games threshold and offseason reasoning.
- injury_reports: one row per injury-report PDF (by header time); same
  one-missed-day threshold as injury_status.
- player_boxscores: ESPN box scores, written only on game days, so the
  games threshold and offseason reasoning apply.

Cloud operation (`status --from-state PATH`, `check_cloud`): the cloud run
(`.github/workflows/live.yml`, one long job looping every 15 minutes) keeps its data as
append-only files under `PATH/state`, and `status --from-state` rebuilds a
throwaway DuckDB from them. Some table ages mean something different there:

- schedule rows in state are CHANGE-only (a vintage is written only when a
  game moved), so `max(observed_at)` would look stale whenever nothing
  moved. Freshness comes from the far-fetch record instead
  (`state/schedule/far_fetches-*.jsonl`, written hourly after a fully
  successful far-day scoreboard fetch).
- injury_status rows are change-only too; an unchanged report still writes
  its `injury_reports` row, so the newer of the two is the report time.
- news_items is not judged: the news archive is the Mac's (research only).
- `cloud_runs` (`check_cloud`) asks GitHub, through `gh run list`, when
  live.yml last ran and how. It never prompts, times out after 20 s, and
  redacts tokens from anything gh prints; without gh or network it is one
  plain "could not check" INFO line that never fails the report.
- On the Mac's own DB, injury_reports and player_boxscores are INFO only:
  only the cloud run writes them.
- `player_links` (`check_player_links`) names ESPN players who played in
  the last 7 days but could not be linked to an NBA player id.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from predictor.config import odds_api_key_path, season_label
from predictor.db import POINT_IN_TIME_TABLES
from predictor.model import live, publish
from predictor.model.market import LIVE_SOURCE

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
    # Polled at 09:00/14:00/19:00 local (scripts/com.predictor.daily.plist);
    # the longest normal gap between runs is ~14h (19:00 -> next 09:00).
    # 24h gives ~10h of slack for one delayed/missed run before alarming.
    "news_items": 24,
    # Fetched once a day at 10:30 local (scripts/com.predictor.schedule.plist).
    # One missed run of slack before alarming, same as injury_status.
    "schedule": 36,
    # Ingested in bulk by `ingest-player-logs`, like games: no scheduled
    # job, and the same ~120-day offseason gap, so the same threshold.
    "player_games": 24 * 150,
    # One row per official injury-report PDF, several a day in season.
    # Same one-missed-day slack as injury_status.
    "injury_reports": 36,
    # ESPN box scores arrive only on game days; like games and
    # player_games, a normal offseason must not alarm.
    "player_boxscores": 24 * 150,
}


@dataclass(frozen=True)
class SourceHealth:
    name: str
    latest: datetime | None
    row_count: int
    age_hours: float | None
    stale: bool
    advice: str
    # When set, format_report prints this as the entry's detail text instead
    # of deriving one from latest/row_count (check_live's entries, whose
    # "rows" are not table rows).
    detail: str | None = None
    # Information only: shown as INFO, never stale, never affects the exit
    # code (a table the Mac never writes; a cloud check that could not run).
    info: bool = False


# The cloud's "where to look": every cloud step is a row of the run summary.
_RUN_SUMMARY = "Run: gh run list --workflow live.yml, then gh run view <id> and read the"

# Which cloud step writes each table (cloud advice, and the Mac's advice for
# the two tables only the cloud writes).
_CLOUD_STEP = {
    "games": "espn_scoreboard",
    "schedule": "espn_scoreboard",
    "injury_status": "injury_report",
    "injury_reports": "injury_report",
    "player_boxscores": "espn_boxscores",
}


def _cloud_advice(name: str, now: datetime) -> str:
    """Advice for a stale source on cloud state: the live.yml run summary,
    never launchd jobs or local commands."""
    if name == "player_games":
        return (
            "the NBA player logs in state/ come from the Mac's snapshot -- on the "
            f"Mac run: predictor ingest-player-logs --season {season_label(now)}, "
            "then predictor export-state --to <state checkout>/state, and commit the state/nba/ change"
        )
    step = _CLOUD_STEP.get(name)
    if step is None:
        return ""
    return f"{_RUN_SUMMARY} {step} step (it writes {name})."


def _advice(name: str, latest: datetime | None, now: datetime) -> str:
    """Advice shown only when a source is stale. See module docstring."""
    if name == "games":
        return f"Run: predictor ingest-season {season_label(now)}"

    if name == "player_games":
        return f"Run: predictor ingest-player-logs --season {season_label(now)}"

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


NEWS_ON_CLOUD = "news is archived by the Mac (research only)"
# Tables only the cloud run writes: information only on the Mac's own DB.
_CLOUD_ONLY = ("injury_reports", "player_boxscores")


def _far_fetched_at(state_dir: Path) -> datetime | None:
    """The newest successful far-day scoreboard fetch recorded in state."""
    from predictor.cloud import state as cloud_state
    from predictor.cloud.run import FAR_FETCHES_GLOB

    times = [
        datetime.fromisoformat(str(r["fetched_at"]).replace("Z", "+00:00")).astimezone(UTC)
        for path in sorted(state_dir.glob(FAR_FETCHES_GLOB))
        for r in cloud_state.read_records(path)
        if r.get("fetched_at")
    ]
    return max(times) if times else None


def _newest(*times: datetime | None) -> datetime | None:
    known = [t for t in times if t is not None]
    return max(known) if known else None


def check_sources(
    con, now: datetime | None = None, *, cloud_state: Path | None = None
) -> list[SourceHealth]:
    """Freshness of every point-in-time table. With ``cloud_state`` (the
    ``state/`` directory the DB was rebuilt from) the cloud rules apply:
    news is not judged, schedule freshness is the last recorded fetch, and
    injury_status counts unchanged reports (see the module docstring)."""
    if now is None:
        now = datetime.now(UTC)

    out: list[SourceHealth] = []
    for name in sorted(POINT_IN_TIME_TABLES):
        if cloud_state is not None and name == "news_items":
            out.append(SourceHealth(name, None, 0, None, False, "", detail=NEWS_ON_CLOUD))
            continue
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

        if name == "odds_snapshots":
            # Never stale here: live-odds freshness is check_live's `odds`
            # entry (see module docstring), so odds are reported once.
            age = (now - latest).total_seconds() / 3600 if latest is not None else None
            out.append(
                SourceHealth(
                    name, latest, count, age, False, "",
                    detail=(
                        "odds are not kept in cloud state (they may not be "
                        "republished); downloads are checked under 'odds'"
                        if cloud_state is not None
                        else "no odds stored yet (live odds are checked under 'odds')"
                        if latest is None
                        else None
                    ),
                )
            )
            continue

        detail = None
        if cloud_state is not None:
            if name == "schedule":
                # Change-only rows: the fetch record says when it was checked.
                fetched = _far_fetched_at(cloud_state)
                latest = _newest(latest, fetched)
                if fetched is not None:
                    ago = _format_age((now - fetched).total_seconds() / 3600)
                    when = ago if ago == "in the future" else f"{ago} ago"
                    detail = f"{count:,} change rows; ESPN scoreboard last fetched {when}"
            elif name == "injury_status":
                reports = con.execute(
                    f"SELECT max(observed_at) FROM {POINT_IN_TIME_TABLES['injury_reports']}"
                ).fetchone()[0]
                latest = _newest(latest, reports)

        if cloud_state is None and name in _CLOUD_ONLY:
            # Only the cloud run writes these: on the Mac's own DB they are
            # shown for information and never judged.
            if latest is None:
                seen = "no rows in this database"
            else:
                age = (now - latest).total_seconds() / 3600
                seen = f"{count:,} rows, newest {_format_age(age)} old ({latest.date()})"
            out.append(SourceHealth(
                name, latest, count, None if latest is None else age, False, "",
                detail=(
                    f"information only (written by the cloud run; check it with "
                    f"predictor status --from-state <checkout>): {seen}"
                ),
                info=True,
            ))
            continue

        def advice_for(when: datetime | None) -> str:
            if cloud_state is not None:
                return _cloud_advice(name, now)
            return _advice(name, when, now)

        if latest is None:
            out.append(SourceHealth(name, None, 0, None, True, advice_for(None)))
            continue

        age = (now - latest).total_seconds() / 3600
        threshold = STALENESS_HOURS.get(name, 48)
        stale = age > threshold
        advice = advice_for(latest) if stale else ""
        out.append(SourceHealth(name, latest, count, age, stale, advice, detail=detail))

    return out


# `odds`: in season, the longest acceptable gap since the newest live
# (The Odds API) line -- one daily 17:30 run plus slack for a late one.
LIVE_ODDS_STALE_HOURS = 30


def _odds_advice() -> str:
    # A missing key and a broken feed need different advice. The key file
    # is never read here (status only names it), so both cases are covered
    # in one message.
    key_path = odds_api_key_path()
    return (
        "If there is no Odds API key yet: get a free key at "
        f"https://the-odds-api.com and save it in {key_path} "
        f"(then chmod 600 {key_path}). If the key is there: run "
        "predictor ingest-odds directly and read its message (quota used "
        "up, invalid key, network failure, etc.), and confirm the launchd "
        "agent com.predictor.odds is loaded."
    )


def _cloud_odds_advice() -> str:
    return (
        f"{_RUN_SUMMARY} odds step. If it says the key is missing, set the "
        "Actions secret: gh secret set ODDS_API_KEY. Otherwise it names the "
        "problem (quota used up, invalid key, network failure)."
    )


def _cloud_odds_downloads(state_dir: Path) -> tuple[int, datetime | None]:
    """(successful odds downloads, newest one) from the cloud's attempt lines
    ``state/odds/attempts-*.jsonl`` -- the cloud keeps no odds rows in state
    (licensing), so these lines are its only odds record."""
    from predictor.cloud import state as cloud_state

    times = []
    for path in sorted(Path(state_dir).glob("odds/attempts-*.jsonl")):
        for rec in cloud_state.read_records(path):
            if rec.get("ok") or rec.get("downloaded"):
                at = datetime.fromisoformat(str(rec["attempted_at"]))
                times.append(at if at.tzinfo else at.replace(tzinfo=UTC))
    return len(times), max(times, default=None)


def _check_live_odds(
    con, now: datetime, cloud: bool = False, state_dir: Path | None = None
) -> SourceHealth:
    """Stale only while in season (`live.in_season`, as the other live
    checks use) with no live line observed in the last
    LIVE_ODDS_STALE_HOURS. Historical (Kaggle) rows never count. For the
    cloud (``state_dir`` given) the newest successful download in the
    attempt lines stands in for the newest live line."""
    if state_dir is not None:
        count, latest = _cloud_odds_downloads(state_dir)
        noun, none = "newest successful odds download", "no successful odds download recorded yet"
    else:
        count, latest = con.execute(
            f"SELECT count(*), max(observed_at) FROM {POINT_IN_TIME_TABLES['odds_snapshots']} "
            "WHERE source = ?",
            [LIVE_SOURCE],
        ).fetchone()
        noun, none = "newest live line", "no live odds collected yet"
    age = (now - latest).total_seconds() / 3600 if latest is not None else None
    season_on = live.in_season(con, now)
    stale = season_on and (age is None or age > LIVE_ODDS_STALE_HOURS)
    if latest is None:
        detail = none
    else:
        when = "in the future" if age < 0 else f"{_format_age(age)} ago"
        detail = f"{noun} {when}"
    if not season_on:
        detail += " (off-season, not checked)"
    return SourceHealth(
        "odds", latest, count, age, stale,
        (_cloud_odds_advice() if cloud else _odds_advice()) if stale else "",
        detail=detail,
    )


# `prediction_log`: the longest acceptable gap between "there are games
# today or yesterday (ET)" and the newest logged prediction -- see
# `check_live`.
PREDICTION_LOG_STALE_HOURS = 30


def check_live(
    con,
    repo_dir: Path,
    now: datetime | None = None,
    *,
    branch: str = "main",
    cloud: bool = False,
) -> list[SourceHealth]:
    """Health of the live-operation pieces that `check_sources` (table
    freshness) cannot see: whether results are actually being captured,
    whether today's predictions are actually being logged, whether the
    log is actually reaching GitHub, and whether live odds are arriving
    (`odds`). Appended after the table sources by the
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

    `branch` is the branch `repo_dir` publishes to (`main` on the Mac,
    `config.publish_branch()` for a cloud state checkout). `cloud=True`
    gives advice for the cloud run (the live.yml run summary, the
    ODDS_API_KEY Actions secret) instead of launchd jobs and key files.
    """
    if now is None:
        now = datetime.now(UTC)

    out: list[SourceHealth] = []

    # --- live_results --------------------------------------------------
    cap = live.last_capture(con)
    missing = live.results_missing(con, now)
    stale_results = bool(missing)
    age = (now - cap).total_seconds() / 3600 if cap is not None else None
    if not stale_results:
        advice_results = ""
    elif cloud:
        advice_results = f"{_RUN_SUMMARY} espn_scoreboard step (it records finals)."
    else:
        advice_results = (
            "Run: predictor capture-results, and confirm the launchd agent "
            "com.predictor.results is loaded."
        )
    if cap is None:
        detail_results = "no live results captured yet"
        if missing:
            detail_results += f", {len(missing)} game(s) missing results"
    else:
        when = "in the future" if age < 0 else f"{_format_age(age)} ago"
        missing_text = (
            f"{len(missing)} game(s) missing results" if missing else "no games missing results"
        )
        detail_results = f"{missing_text}, last capture {when}"
    out.append(
        SourceHealth(
            "live_results", cap, len(missing), age, stale_results, advice_results,
            detail=detail_results,
        )
    )

    # --- prediction_log --------------------------------------------------
    season = season_label(now)
    log_file = live.log_path(repo_dir, season)
    log_error = None
    log: list[dict] = []
    latest_pred = None
    try:
        log = live.read_log(log_file)
        latest_pred = max(
            (datetime.fromisoformat(line["predicted_at"]) for line in log), default=None
        )
    except live.LogError as exc:
        log_error = str(exc)
    except (KeyError, TypeError, ValueError) as exc:
        log_error = (
            f"the prediction log at {log_file} has a line without a valid "
            f"predicted_at ({exc!r}); inspect and repair it by hand"
        )
    if log_error is not None:
        # predict-today refuses to append to a log it cannot read, so this
        # is a stopped pipeline, not a cosmetic problem.
        out.append(
            SourceHealth(
                "prediction_log", None, 0, None, True,
                f"{log_error}. predict-today will not run until it is repaired.",
                detail="the log could not be read",
            )
        )
    else:
        has_recent_games = bool(live.slate_for(con, now)) or bool(
            live.slate_for(con, now - timedelta(days=1))
        )
        stale_log = False
        if live.in_season(con, now) and has_recent_games:
            if latest_pred is None or (now - latest_pred) > timedelta(
                hours=PREDICTION_LOG_STALE_HOURS
            ):
                stale_log = True
        age_log = (now - latest_pred).total_seconds() / 3600 if latest_pred is not None else None
        if not stale_log:
            advice_log = ""
        elif cloud:
            advice_log = f"{_RUN_SUMMARY} predict and state_commit steps."
        else:
            advice_log = (
                "Run: predictor predict-today, and confirm com.predictor.predict is loaded."
            )
        out.append(
            SourceHealth("prediction_log", latest_pred, len(log), age_log, stale_log, advice_log)
        )

    # --- prediction_files ---------------------------------------------------
    # Never OK while predictions are written but stranded locally: off main
    # (never committed), mid-operation, or left uncommitted by a failed
    # publish -- log_published alone cannot see any of these (it only
    # counts commits that exist).
    files_problem = publish.prediction_files_problem(repo_dir, branch)
    out.append(
        SourceHealth(
            "prediction_files",
            None,
            0,
            None,
            files_problem is not None,
            files_problem or "",
            detail=(
                "NOT being published from this checkout"
                if files_problem is not None
                else f"committed on {branch}, nothing left uncommitted"
            ),
        )
    )

    # --- log_published -----------------------------------------------------
    unpushed = publish.unpushed_commits(repo_dir, branch)
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
        # Same predicate predict-today's own push uses: a commit that is not
        # an automated prediction commit is never pushed automatically, and
        # a human told to "just push" would make it public.
        foreign = publish.unpushed_non_prediction_commits(repo_dir, branch)
        if foreign:
            advice_pub = (
                f"{foreign} of the unpushed commits are not prediction commits "
                "— pushing would make them PUBLIC; review them before running "
                f"git push origin {branch}"
            )
        elif cloud:
            advice_pub = (
                f"{unpushed} commit(s) in this checkout are not on origin/{branch}. "
                "Cloud runs never rebase: when GitHub moved during a run, nothing "
                "is pushed and the next live.yml run starts from GitHub and redoes "
                "the work -- nothing needs pushing by hand. If this persists, "
                "read the push step of the latest live.yml run summary."
            )
        else:
            advice_pub = (
                f"{unpushed} commit(s) are committed locally but not pushed -- "
                "the next predict-today or capture-results run pushes them, "
                "rebasing its own safe commits over anything new on GitHub "
                f"automatically. To push now: git push origin {branch}, or "
                "check the GitHub login: gh auth status."
            )
    else:
        stale_pub = False
        row_count_pub = 0
        advice_pub = ""
    out.append(
        SourceHealth("log_published", None, row_count_pub, None, stale_pub, advice_pub)
    )

    # --- odds (live lines) ------------------------------------------------
    out.append(_check_live_odds(con, now, cloud, Path(repo_dir) / "state" if cloud else None))

    return out


# --- cloud runs ------------------------------------------------------------

# The live workflow is one ~5.5 h job looping every 15 minutes (cloud-loop),
# restarted by a cron GitHub fires only every few hours. Fresh: a run still
# in progress that started under CLOUD_RUNNING ago (a loop that has not hit
# its timeout), or a run that finished within CLOUD_STALE.
CLOUD_STALE = timedelta(hours=6, minutes=15)
CLOUD_RUNNING = timedelta(hours=6)
CLOUD_WORKFLOW = "live.yml"
GH_TIMEOUT = 20
# Consecutive failed (completed) runs that count as broken.
CLOUD_FAILURES = 3

_GH_ARGS = [
    # 20: enough completed runs for the failure streak even among the
    # cancelled starts a queued loop leaves behind.
    "gh", "run", "list", "--workflow", CLOUD_WORKFLOW, "--limit", "20",
    "--json", "createdAt,startedAt,updatedAt,status,conclusion",
]
# GitHub tokens gh could echo: classic/fine-grained/app/OAuth forms.
_GH_TOKEN = re.compile(r"\b(gh[pousr]_[A-Za-z0-9]{8,}|github_pat_[A-Za-z0-9_]{8,})")
_GH_ADVICE = (
    "check gh auth status (gh must be installed and logged in, and the "
    "network reachable), then gh run list --workflow live.yml"
)


def _redact_gh(text: str) -> str:
    return _GH_TOKEN.sub("***", publish._redact(text))


def _could_not_check(reason: str) -> SourceHealth:
    # Information, not a problem: no gh or no network says nothing about the
    # cloud, and must not fail the report.
    return SourceHealth(
        "cloud_runs", None, 0, None, False, _GH_ADVICE,
        detail=f"could not check cloud runs ({reason})", info=True,
    )


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _ago(delta: timedelta) -> str:
    minutes = delta.total_seconds() / 60
    if minutes < 0:
        return "in the future"
    if minutes < 120:
        return f"{minutes:.0f}m ago"
    return f"{_format_age(minutes / 60)} ago"


def check_cloud(
    now: datetime, runner=subprocess.run, *, cwd: Path | None = None
) -> SourceHealth:
    """When live.yml last ran, from ``gh run list`` (run in ``cwd``, so gh
    finds the repository from that checkout's remote). Fresh while a run is
    in progress that started (``startedAt``, else ``createdAt``) under
    CLOUD_RUNNING ago, or when the newest completed run finished (its
    ``updatedAt``) within CLOUD_STALE; stale otherwise, and whenever the 3
    newest completed runs all failed. Cancelled runs count for neither. gh never prompts (GH_PROMPT_DISABLED, no
    stdin), is given GH_TIMEOUT seconds, and anything it prints is redacted;
    a missing gh, a timeout or an error is one plain "could not check"."""
    env = dict(os.environ, GH_PROMPT_DISABLED="1", GIT_TERMINAL_PROMPT="0")
    try:
        result = runner(
            list(_GH_ARGS), capture_output=True, text=True, timeout=GH_TIMEOUT,
            env=env, stdin=subprocess.DEVNULL, cwd=cwd,
        )
    except FileNotFoundError:
        return _could_not_check("gh is not installed")
    except subprocess.TimeoutExpired:
        return _could_not_check(f"gh did not answer within {GH_TIMEOUT} s")
    except OSError as exc:
        return _could_not_check(f"gh could not be run: {type(exc).__name__}")
    if result.returncode != 0:
        lines = (result.stderr or result.stdout or "").strip().splitlines()
        first = _redact_gh(lines[0])[:160] if lines else f"exit {result.returncode}"
        return _could_not_check(f"gh: {first}")
    try:
        runs = json.loads(result.stdout or "")
        dated = sorted(
            ((_parse_time(r["createdAt"]), r) for r in runs),
            key=lambda pair: pair[0], reverse=True,
        )
        # A completed run's age counts from when it finished (a loop job
        # finishes hours after it was created). A cancelled run says nothing
        # about the loop (a queued start replaced in the concurrency group,
        # or cancelled by hand): it is not counted as finished at all.
        finished = sorted(
            ((_parse_time(r.get("updatedAt") or r["createdAt"]), r)
             for _, r in dated
             if r.get("status") == "completed" and r.get("conclusion") != "cancelled"),
            key=lambda pair: pair[0], reverse=True,
        )
        # A run in progress counts from when it started (it may have queued
        # behind the previous loop for hours).
        running = any(
            r.get("status") == "in_progress"
            and now - _parse_time(r.get("startedAt") or r["createdAt"]) < CLOUD_RUNNING
            for _, r in dated
        )
    except (ValueError, TypeError, KeyError, AttributeError):
        return _could_not_check("gh's answer could not be read")

    if not dated:
        return SourceHealth(
            "cloud_runs", None, 0, None, True,
            "the live workflow has never run: check it is enabled with "
            "gh workflow list, or start it with gh workflow run live.yml",
            detail="no cloud run found for live.yml",
        )
    # Age is that of the last COMPLETED run; with none completed so far, the
    # newest run's. A loop still running (and younger than CLOUD_RUNNING)
    # keeps the check fresh whatever that age is.
    newest_at, newest = finished[0] if finished else dated[0]
    age = now - newest_at
    fresh = running or (bool(finished) and age <= CLOUD_STALE)
    state = newest.get("conclusion") or newest.get("status") or "unknown"
    detail = f"last cloud run {_ago(age)} ({state})"
    if finished and dated[0][1].get("status") != "completed":
        detail += f"; one {dated[0][1].get('status') or 'running'} now"
    completed = [r for _, r in finished][:CLOUD_FAILURES]
    failing = len(completed) == CLOUD_FAILURES and all(
        r.get("conclusion") == "failure" for r in completed
    )
    if not fresh:
        advice = (
            f"no live.yml run completed or running within "
            f"{CLOUD_STALE.total_seconds() / 3600:g}h: check "
            "the workflow is enabled (gh workflow list) and read the latest run "
            "(gh run list --workflow live.yml, then gh run view <id>)"
        )
    elif failing:
        detail += f"; the last {CLOUD_FAILURES} completed runs all failed"
        advice = (
            "read the failing steps of the latest live.yml run summary "
            "(gh run list --workflow live.yml, then gh run view <id>)"
        )
    else:
        advice = ""
    return SourceHealth(
        "cloud_runs", newest_at, len(dated), age.total_seconds() / 3600,
        bool(advice), advice, detail=detail,
    )


# --- player links (cloud state) --------------------------------------------

PLAYER_LINKS_DAYS = 7
PLAYER_LINKS_NAMED = 5
_EASTERN = ZoneInfo("America/New_York")


def check_player_links(state_dir: Path, con, now: datetime) -> SourceHealth:
    """Stale when an ESPN player who played (minutes > 0) in a game of the
    last PLAYER_LINKS_DAYS days (ET) is still unlinked (``espn:<id>``) --
    their minutes are invisible to the player model until linked. Links
    known in ``state_dir`` (links.jsonl, aliases.json) count as linked even
    if the DB was built before them."""
    from predictor.cloud.players import resolved_player_ids

    since = (now.astimezone(_EASTERN) - timedelta(days=PLAYER_LINKS_DAYS)).date()
    rows = con.execute(
        f"SELECT espn_player_id, player_name, max(game_date) "
        f"FROM {POINT_IN_TIME_TABLES['player_boxscores']} "
        "WHERE player_id LIKE 'espn:%' AND minutes > 0 AND game_date >= ? "
        "GROUP BY espn_player_id, player_name ORDER BY max(game_date) DESC, player_name",
        [since],
    ).fetchall()
    resolved = resolved_player_ids(state_dir)
    unmatched: dict[str, str] = {}
    for espn_id, name, _ in rows:
        if espn_id is not None and str(espn_id) in resolved:
            continue
        unmatched.setdefault(str(espn_id), f"{name or 'unnamed'} (ESPN {espn_id})")
    if not unmatched:
        return SourceHealth(
            "player_links", None, 0, None, False, "",
            detail=f"every ESPN player of the last {PLAYER_LINKS_DAYS} days is linked",
        )
    names = list(unmatched.values())
    shown = ", ".join(names[:PLAYER_LINKS_NAMED])
    more = len(names) - PLAYER_LINKS_NAMED
    if more > 0:
        shown += f" and {more} more"
    return SourceHealth(
        "player_links", None, len(names), None, True,
        'in state/players/aliases.json add "<ESPN id>": <NBA player id> under '
        '"by_espn_id" (or "<name>": <NBA player id> under "by_name") and commit '
        "it; the next run's rebuild relinks their old rows",
        detail=(
            f"{len(names)} ESPN player(s) who played in the last "
            f"{PLAYER_LINKS_DAYS} days have no NBA id: {shown} -- add them to "
            "state/players/aliases.json"
        ),
    )


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
        mark = "STALE" if h.stale else ("INFO" if h.info else "OK")
        if h.detail is not None:
            detail = h.detail
        elif h.name == "log_published":
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
            # as stale (an empty table). Of check_live's entries,
            # live_results and prediction_files always carry their own
            # `detail` (handled above) and log_published has its own branch,
            # so the only one that can reach this branch is prediction_log
            # with an empty/missing log file -- `stale` there also depends
            # on whether there are recent games at all (see check_live), so
            # `latest is None` and `stale is False` both commonly hold
            # together in the off-season ("n/a": nothing predicted yet, and
            # nothing is wrong). A prediction_log that could not be read at
            # all carries its own `detail` too.
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
