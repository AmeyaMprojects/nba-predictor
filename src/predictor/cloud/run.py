"""One cloud run: do whatever is due, commit state and predictions
separately, push once, and report every step.

Steps, always in this order and always all listed in the report (a step
that is not due has ``due=False``)::

    mac_history, rebuild, espn_scoreboard, espn_boxscores, injury_report,
    odds, state_commit, predict, grade, raw_archive, push

- Source steps append their rows to ``state/`` first, then load them into
  the run's DuckDB (rebuilt from ``state/`` at the start of every run).
- A source that fails with an expected error (network, ESPN format, Odds
  API quota or fetch error, a missing key) is recorded as a failed step and
  the run carries on. Any other exception is recorded with its type name,
  sets ``unexpected`` (the CLI then exits 1), and the run still carries on.
- A `SecretLeak` stops the run at once: nothing more is committed, and
  nothing is pushed.
- The ``state:`` commit (paths under ``state/`` only) and the
  ``predictions:`` commit (paths under ``predictions/`` only) are separate
  commits, made with ``push=False``; the ``push`` step pushes both. A run
  killed before the push loses only its own commits, and the next run
  redoes the work from ``state/``: every writer dedupes against what state
  already holds.
- The RSS news archiver runs at most hourly inside the run (it is not one
  of the listed steps). Each feed's failure is shown in the summary and
  never fails the run.

The run never seals a call at or after tip-off: `live.predict_today` only
predicts a game that tips more than ``START_BUFFER`` after ``now``, and
``now`` is re-read after the fetches.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import requests

from predictor import config, db
from predictor.cloud import archive, games, injuries, players, state
from predictor.model import live, publish
from predictor.sources import espn, injury_report, news_rss, odds

STEP_NAMES = (
    "mac_history", "rebuild", "espn_scoreboard", "espn_boxscores", "injury_report",
    "odds", "state_commit", "predict", "grade", "raw_archive", "push",
)

FAR_FETCHES_GLOB = "schedule/far_fetches-*.jsonl"
ODDS_ATTEMPTS_GLOB = "odds/attempts-*.jsonl"
NEWS_POLLS_GLOB = "news/polls-*.jsonl"
NEWS_INTERVAL = timedelta(minutes=60)
# Unmatched players named one by one in the summary; the rest are counted.
MAX_UNMATCHED_NOTES = 10

# Failures of a source that are expected now and then: recorded, not fatal.
EXPECTED_ERRORS: tuple[type[BaseException], ...] = (
    requests.RequestException,
    injury_report.TransientFetchError,
    injury_report.InjuryReportParseError,
    espn.EspnFetchError,
    espn.EspnFormatError,
    odds.OddsQuotaExceeded,
    odds.OddsFetchError,
    odds.MissingOddsKey,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _odds_download(api_key: str, now: datetime) -> odds.OddsDownload:
    return odds.download(api_key, now=now)


def _news_poll(now: datetime) -> dict[str, news_rss.FeedResult]:
    return news_rss.poll_all(now)


@dataclass(frozen=True)
class Fetchers:
    """Everything a run reaches outside the machine, injectable for tests.

    - ``scoreboard(day) -> bytes``: one ET date's ESPN scoreboard.
    - ``summary(espn_id) -> bytes``: one ESPN game summary.
    - ``injury_latest(now) -> LatestReport | None`` (archives the PDF).
    - ``odds_download(api_key, now) -> OddsDownload`` (archives the body).
    - ``gh_runner``: ``subprocess.run``-like, for ``gh`` (run in the state
      checkout, so gh finds the repository from its remote).
    - ``clock() -> datetime`` (UTC).
    - ``news(now) -> {feed: FeedResult}`` (archives each feed).
    """

    scoreboard: Callable[[date], bytes] = espn.fetch_scoreboard_day
    summary: Callable[[str], bytes] = espn.fetch_summary
    injury_latest: Callable = injury_report.fetch_latest
    odds_download: Callable = _odds_download
    gh_runner: Callable = subprocess.run
    clock: Callable[[], datetime] = _utcnow
    news: Callable = _news_poll


@dataclass(frozen=True)
class StepOutcome:
    name: str
    due: bool
    ok: bool
    detail: str


@dataclass
class RunReport:
    started_at: datetime
    finished_at: datetime
    branch: str
    steps: list[StepOutcome]
    state_commit: str | None
    predictions_commit: str | None
    pushed: bool
    site_changed: bool
    raw_archived: bool
    unexpected: bool
    # A publish result had error=True (git add/commit failed, bad path).
    publish_error: bool = False
    # Extra lines for the summary: postponed/skipped games, unmatched
    # players, news feeds, the push message.
    notes: list[str] = field(default_factory=list)
    push_message: str = ""
    # A SecretLeak stopped the run (nothing more committed or pushed).
    secret_leak: bool = False


class _Stop(Exception):
    """A SecretLeak was found: stop the run before anything else is
    committed or pushed."""


class _Halt(Exception):
    """The checkout could not be brought up to the remote: nothing is run."""


@dataclass
class _Run:
    state_repo: Path
    code_dir: Path
    branch: str
    fetchers: Fetchers
    run_attempt: int
    started: datetime
    steps: dict[str, StepOutcome] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    unexpected: bool = False
    publish_error: bool = False
    con: object = None
    state_commit: str | None = None
    predictions_commit: str | None = None
    pushed: bool = False
    push_message: str = ""
    raw_archived: bool = False
    site_changed: bool = False
    secret_leak: bool = False

    @property
    def state_dir(self) -> Path:
        return self.state_repo / state.STATE_DIRNAME

    def clock(self) -> datetime:
        return db.require_utc(self.fetchers.clock(), "clock")

    def record(self, name: str, due: bool, ok: bool, detail: str) -> None:
        self.steps[name] = StepOutcome(name, due, ok, _plain(detail))

    def step(self, name: str, fn: Callable[[], tuple[bool, bool, str]],
             *, needs_db: bool = False) -> None:
        """Run one step. ``fn`` returns (due, ok, detail)."""
        if needs_db and self.con is None:
            self.record(name, True, False, "not run: no database (the rebuild failed)")
            return
        try:
            due, ok, detail = fn()
        except archive.SecretLeak as exc:
            self.unexpected = True
            self.secret_leak = True
            self.record(name, True, False, f"stopped: {exc}")
            raise _Stop() from None
        except EXPECTED_ERRORS as exc:
            self.record(name, True, False, _expected_detail(exc))
            return
        except Exception as exc:  # noqa: BLE001 -- every step must be reported
            self.unexpected = True
            self.record(name, True, False, f"{type(exc).__name__}: {exc}")
            return
        self.record(name, due, ok, detail)


def _plain(text: str) -> str:
    """One line, with no secret in it."""
    return archive._scrub(" ".join(str(text).split()), config.odds_api_key())


def _expected_detail(exc: BaseException) -> str:
    if isinstance(exc, odds.MissingOddsKey):
        return "no Odds API key (env ODDS_API_KEY is not set)"
    if isinstance(exc, odds.OddsQuotaExceeded):
        left = f", {exc.requests_remaining} requests left" if exc.requests_remaining else ""
        return f"The Odds API refused the request ({exc}{left})"
    if isinstance(exc, requests.RequestException):
        # The text of a requests error can carry the URL; the type is enough.
        return f"could not reach the source ({type(exc).__name__})"
    return f"{type(exc).__name__}: {exc}"


# --- small state readers -----------------------------------------------------


def _records(state_dir: Path, pattern: str) -> list[dict]:
    out: list[dict] = []
    for path in sorted(state_dir.glob(pattern)):
        out.extend(state.read_records(path))
    return out


def _as_utc(value) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def _latest_time(state_dir: Path, pattern: str, key: str) -> datetime | None:
    times = [_as_utc(r[key]) for r in _records(state_dir, pattern) if r.get(key)]
    return max(times) if times else None


def _month_file(state_dir: Path, prefix: str, at: datetime) -> Path:
    return state_dir / f"{prefix}-{at:%Y-%m}.jsonl"


def odds_attempts(state_dir: Path) -> list[dict]:
    """Every recorded odds attempt (``attempted_at, window, ok, remaining``)."""
    return _records(state_dir, ODDS_ATTEMPTS_GLOB)


def last_remaining(attempts: list[dict]) -> int | None:
    """The quota left after the latest attempt that reported one."""
    known = [a for a in attempts if a.get("remaining") is not None]
    if not known:
        return None
    latest = max(known, key=lambda a: _as_utc(a["attempted_at"]))
    return int(latest["remaining"])


def _to_int(value) -> int | None:
    try:
        return None if value is None else int(str(value).strip())
    except ValueError:
        return None


# --- the steps ---------------------------------------------------------------


def _rebuild(run: _Run) -> tuple[bool, bool, str]:
    db_path = config.settings.db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    report = state.rebuild_db(run.state_dir, db_path)
    run.con = db.connect(db_path)
    total = sum(report.rows.values())
    return True, True, f"{total} rows from state in {report.seconds:.1f} s"


def _scoreboard(run: _Run) -> tuple[bool, bool, str]:
    now = run.started
    last_far = _latest_time(run.state_dir, FAR_FETCHES_GLOB, "fetched_at")
    include_far = games.far_due(last_far, now)
    days = games.scoreboard_days(now, include_far)
    far_days = set(days) - set(games.scoreboard_days(now, False))
    keys: list[str] = []
    failed: list[str] = []
    errors: list[BaseException] = []
    for day in days:
        try:
            body = run.fetchers.scoreboard(day)
        except EXPECTED_ERRORS as exc:
            failed.append(day.isoformat())
            errors.append(exc)
            continue
        keys.append(espn.archive("scoreboard", body, run.clock(), f"{day:%Y%m%d}"))
    if not keys and errors:
        raise errors[0]
    fetched_at = run.clock()
    problems: list[str] = []
    found = espn.load_scoreboards(keys, problems)
    sync = games.sync_games(run.con, found, fetched_at, games.read_espn_links(run.state_dir),
                            malformed=problems)
    games.apply_sync(run.con, run.state_dir, sync)
    if include_far and not any(d.isoformat() in failed for d in far_days):
        state.append_records(_month_file(run.state_dir, "schedule/far_fetches", fetched_at),
                             [{"fetched_at": fetched_at, "days": len(far_days)}])
    for line in sync.postponed:
        run.notes.append(f"postponed: {line}")
    for line in sync.skipped:
        run.notes.append(f"skipped ESPN event: {line}")
    detail = (
        f"{len(keys)}/{len(days)} day(s){' (far days included)' if include_far else ''}; "
        f"{len(sync.schedule_rows)} schedule row(s), {len(sync.finals)} final(s), "
        f"{len(sync.espn_links)} new link(s), {len(sync.postponed)} postponed, "
        f"{len(sync.skipped)} skipped"
    )
    if failed:
        first = _expected_detail(errors[0])
        return True, False, f"failed day(s) {', '.join(failed)} ({first}); {detail}"
    return True, True, detail


def _boxscores(run: _Run) -> tuple[bool, bool, str]:
    now = run.started
    due = players.boxscores_due(run.con, now, games.read_espn_links(run.state_dir),
                                state.read_id_links(run.state_dir))
    if not due:
        return False, True, "no finished game is missing its box score"
    index = players.PlayerIndex.build(run.state_dir, now)  # once per run
    done = rows_total = 0
    unmatched: dict[str, players.PlayerLink] = {}
    linked_now: set[str] = set()
    failed: list[str] = []
    for game_id, espn_id in due:
        try:
            body = run.fetchers.summary(espn_id)
            fetched_at = run.clock()
            espn.archive("summary", body, fetched_at, espn_id)  # raw first
            lines = espn.parse_boxscore(body)
            result = players.link_lines(run.state_dir, lines, index, fetched_at)
            rows = players.box_rows(players.game_info(run.con, game_id), lines,
                                    result.links, fetched_at)
            players.apply_boxscores(run.con, run.state_dir, rows)
        except EXPECTED_ERRORS as exc:
            failed.append(f"{game_id} ({_expected_detail(exc)})")
            continue
        done += 1
        rows_total += len(rows)
        for link in result.unmatched:
            unmatched[link.espn_player_id] = link
        linked_now |= {k for k, lk in result.links.items() if lk.nba_player_id is not None}
    listed = sorted(unmatched.values(), key=lambda lk: (lk.team, lk.name))
    for link in listed[:MAX_UNMATCHED_NOTES]:
        run.notes.append(f"unmatched player: {link.name} ({link.team}, ESPN {link.espn_player_id})")
    if len(listed) > MAX_UNMATCHED_NOTES:
        run.notes.append(f"unmatched player: and {len(listed) - MAX_UNMATCHED_NOTES} more")
    resolved = players.resolved_player_ids(run.state_dir)
    by_nba: dict[str, list[str]] = {}
    for espn_id, nba_id in resolved.items():
        by_nba.setdefault(nba_id, []).append(espn_id)
    for nba_id, espn_ids in sorted(by_nba.items()):
        if len(espn_ids) > 1 and linked_now & set(espn_ids):
            run.notes.append(
                f"WARNING: ESPN athletes {', '.join(sorted(espn_ids))} all resolve to "
                f"NBA player {nba_id}; fix with state/players/aliases.json"
            )
    detail = (f"{done}/{len(due)} game(s), {rows_total} row(s), "
              f"{len(unmatched)} unmatched player(s)")
    if failed:
        return True, False, f"{detail}; failed: {'; '.join(failed)}"
    return True, True, detail


def _injury(run: _Run) -> tuple[bool, bool, str]:
    latest = run.fetchers.injury_latest(run.started)
    if latest is None:
        return True, True, "no report published in the last five 15-minute slots"
    outcome = injuries.record_report(run.con, run.state_dir, latest)
    when = f"{outcome.report_at:%Y-%m-%d %H:%M} UTC" if outcome.report_at else "?"
    if not outcome.new:
        return True, True, f"report of {when} already recorded"
    return True, True, (f"new report of {when}: {outcome.rows} row(s), "
                        f"{outcome.changes} change line(s)")


def _odds(run: _Run) -> tuple[bool, bool, str]:
    attempts = odds_attempts(run.state_dir)
    remaining = last_remaining(attempts)
    window = live.odds_window_due(run.con, run.started, attempts, remaining)
    if window is None:
        return False, True, "no odds window due"
    attempted_at = run.clock()
    # "downloaded": the response body reached the archive, so its credits
    # are spent -- `odds_window_due` closes the window even if loading failed.
    attempt = {"attempted_at": attempted_at, "window": window.name, "ok": False,
               "downloaded": False, "remaining": None}
    try:
        key = config.odds_api_key()
        if not key:
            raise odds.MissingOddsKey("no key")
        downloaded = run.fetchers.odds_download(key, attempted_at)
        attempt["downloaded"] = True
        attempt["remaining"] = _to_int(downloaded.requests_remaining)
        summary = odds.load(run.con, downloaded)
        rows = _odds_rows(run.con, downloaded.observed_at)
        state.append_rows(run.state_dir, "odds", rows)
        attempt["ok"] = True
    except odds.OddsQuotaExceeded as exc:
        attempt["remaining"] = _to_int(exc.requests_remaining)
        raise
    finally:
        state.append_records(_month_file(run.state_dir, "odds/attempts", attempted_at),
                             [attempt])
    left = f", {attempt['remaining']} requests left" if attempt["remaining"] is not None else ""
    return True, True, (f"{window.name} window: {summary.rows} row(s), "
                        f"{summary.linked}/{summary.events} event(s) linked{left}")


def _odds_rows(con, observed_at: datetime) -> list[dict]:
    table = db.POINT_IN_TIME_TABLES["odds_snapshots"]
    cols = state.ODDS_COLUMNS
    found = con.execute(
        f"SELECT {', '.join(cols)} FROM {table} WHERE source = ? AND observed_at = ? "
        "ORDER BY game_key, book",
        [odds.SOURCE, observed_at],
    ).fetchall()
    return [dict(zip(cols, r, strict=True)) for r in found]


def _news(run: _Run) -> None:
    """The hourly RSS poll: new items go to ``state/news`` and the DB; each
    feed failure becomes a summary note. Never fails the run on a feed."""
    now = run.started
    last = _latest_time(run.state_dir, NEWS_POLLS_GLOB, "polled_at")
    if last is not None and now - last < NEWS_INTERVAL:
        return
    try:
        results = run.fetchers.news(now)
        table = db.POINT_IN_TIME_TABLES["news_items"]
        before = {r[0] for r in run.con.execute(f"SELECT item_key FROM {table}").fetchall()}
        news_rss.ingest_archived_news(run.con)
        cols = state.NEWS_COLUMNS
        fresh = [
            dict(zip(cols, r, strict=True))
            for r in run.con.execute(
                f"SELECT {', '.join(cols)} FROM {table} ORDER BY observed_at, item_key"
            ).fetchall()
            if r[0] not in before
        ]
        if fresh:
            state.append_rows(run.state_dir, "news", fresh)
        failed = {name: (r.error or "failed") for name, r in sorted(results.items()) if not r.ok}
        state.append_records(_month_file(run.state_dir, "news/polls", now),
                             [{"polled_at": now, "failed": failed}])
    except Exception as exc:  # noqa: BLE001 -- news never fails the run
        if not isinstance(exc, EXPECTED_ERRORS):
            run.unexpected = True
        run.notes.append(_plain(f"news: not polled ({type(exc).__name__}: {exc})"))
        return
    note = f"news: {len(fresh)} new item(s) from {len(results) - len(failed)}/{len(results)} feed(s)"
    run.notes.append(note)
    for name, error in failed.items():
        run.notes.append(_plain(f"news feed {name} failed: {error} (stays on the Mac)"))


def _changed_files(repo: Path, dirname: str) -> list[Path]:
    """Files under ``repo/dirname`` that differ from HEAD or are untracked."""
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "-uall", "--no-renames", "--", dirname],
        cwd=repo, capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git status failed: {result.stderr.strip()}")
    out: list[Path] = []
    for entry in result.stdout.split("\0"):
        if len(entry) > 3 and "D" not in entry[:2]:
            out.append(repo / entry[3:])
    return sorted(set(out))


def _commit(run: _Run, kind: str, paths: list[Path], message: str) -> tuple[bool, str, str | None]:
    """(ok, message, sha) of one commit with push=False."""
    if not paths:
        return True, "nothing to commit", None
    result = publish.commit_and_push(run.state_repo, paths, message, push=False,
                                     kind=kind, branch=run.branch)
    if result.error:
        run.publish_error = True
        return False, result.message, None
    if not result.committed:
        return result.message == publish.NOTHING_TO_COMMIT, result.message, None
    sha = publish.head_sha(run.state_repo)
    return True, f"committed {sha} ({len(paths)} file(s))", sha


def _state_commit(run: _Run) -> tuple[bool, bool, str]:
    # The commit sweeps up every changed file under state/, whichever step
    # wrote it -- including files a failed step or a killed earlier run left
    # half done. That is safe only because every state writer appends whole
    # lines (one write per batch, a version header on a new file), never
    # rewrites a line, and dedupes against what state already holds, so a
    # partial write is a valid prefix the next run completes without
    # duplicates. A new writer under state/ must keep that property.
    paths = _changed_files(run.state_repo, state.STATE_DIRNAME)
    archive.assert_no_secret([*paths, config.settings.raw_dir], config.odds_api_key())
    now = run.started
    ok, message, sha = _commit(run, "state", paths, f"state: cloud run {now:%Y-%m-%dT%H:%MZ}")
    run.state_commit = sha
    return True, ok, message


def _slate_season(run: _Run, now: datetime) -> str:
    slate = live.slate_for(run.con, now)
    return slate[0].season if slate else config.season_label(now)


def _id_links(run: _Run) -> list[Path]:
    path = run.state_dir / state.ID_LINKS_PATH
    records = list(state.read_records(path)) if path.exists() else []
    return live.append_id_links(run.state_repo, records)


def _state_committed(run: _Run) -> bool:
    """The state commit step succeeded and left nothing under state/
    uncommitted, so ``head_sha`` names the state the call is made from."""
    outcome = run.steps.get("state_commit")
    if outcome is None or not outcome.ok:
        return False
    return not _changed_files(run.state_repo, state.STATE_DIRNAME)


def _predict(run: _Run) -> tuple[bool, bool, str]:
    now = run.clock()  # re-read after the fetches
    if not _state_committed(run):
        return live.early_call_due(now), False, "not run: state commit failed"
    if not live.early_call_due(now):
        _id_links(run)
        return False, True, f"the early call is due from {live.early_call_at(live.slate_date(now)):%H:%M} UTC"
    from predictor.model import settings as model_settings

    _id_links(run)  # before predicting: a linked provisional call covers its game
    loaded = model_settings.load(state.settings_path(run.state_dir))
    result = live.predict_today(run.con, loaded, run.state_repo, now, code_dir=run.code_dir,
                                state_version=publish.head_sha(run.state_repo))
    detail = (f"slate {live.slate_date(now)}: {result.predicted} predicted, "
              f"{result.not_predicted} not predicted, "
              f"{result.skipped_duplicates} duplicate(s) skipped")
    if result.stale:
        detail += "; recent results missing (stale_results)"
    return True, True, detail


def _grade(run: _Run) -> tuple[bool, bool, str]:
    now = run.clock()
    _id_links(run)
    graded, _seasons = live.grade_seasons(run.con, run.state_repo, _slate_season(run, now), now,
                                          code_dir=run.code_dir)
    return True, True, f"{graded} grade line(s) appended"


def _predictions_commit(run: _Run) -> None:
    """Commit everything the predict and grade steps wrote (logs, grades,
    id-links), even after a failure in one of them."""
    now = run.clock()
    paths = _changed_files(run.state_repo, "predictions")
    if not paths:
        return
    archive.assert_no_secret(paths, config.odds_api_key())
    pred_dir = run.state_repo / "predictions"
    calls = any(p.parent == pred_dir and live._SEASON_LOG.match(p.name) for p in paths)
    slate = live.slate_date(now).isoformat()
    message = f"predictions: {slate} slate" if calls else f"predictions: grades {slate}"
    ok, text, sha = _commit(run, "predictions", paths, message)
    run.predictions_commit = sha
    if not ok:
        prev = run.steps["grade"]
        run.record("grade", prev.due, False, f"{prev.detail}; predictions commit: {text}")


def _raw_archive(run: _Run) -> tuple[bool, bool, str]:
    data_dir = config.settings.data_dir
    state_repo = run.state_repo

    def gh(argv, **kwargs):
        return run.fetchers.gh_runner(argv, cwd=state_repo, **kwargs)

    result = archive.archive_run(config.settings.raw_dir, data_dir / "archive", run.started,
                                 run.run_attempt, secret=config.odds_api_key(), runner=gh)
    run.raw_archived = result.ok
    return True, result.ok, result.message


def _unpushed_subjects(repo: Path, branch: str) -> list[str]:
    result = subprocess.run(
        ["git", "log", "--format=%s", f"origin/{branch}..{branch}"],
        cwd=repo, capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL,
    )
    return result.stdout.splitlines() if result.returncode == 0 else []


def _push(run: _Run) -> tuple[bool, bool, str]:
    predictions = any(s.startswith("predictions: ")
                      for s in _unpushed_subjects(run.state_repo, run.branch))
    # No rebase in a cloud run: rewriting our commits would orphan the
    # state_version on lines already sealed. A moved remote leaves them
    # unpushed; the next run starts from the remote and redoes the work.
    result = publish.push_pending(run.state_repo, branch=run.branch, rebase=False)
    run.pushed = result.pushed
    run.push_message = _plain(result.message)
    run.site_changed = result.pushed and predictions and run.branch == "main"
    if result.error:
        run.publish_error = True
    ok = result.pushed or result.message == publish.NOTHING_TO_PUSH
    return True, ok and not result.error, result.message


def _sync_checkout(run: _Run) -> str | None:
    """Fast-forward the checkout to ``origin/<branch>`` before anything is
    written, so the state commit (and the ``state_version`` sealed into
    calls) sits on top of what the remote has and is never rewritten later.
    Returns why that was impossible, or None."""
    repo, branch = run.state_repo, run.branch
    actual = publish.current_branch(repo)
    if actual != branch:
        return f"the checkout is on {actual or 'a detached HEAD'}, not {branch}"
    fetch = publish._git(repo, ["fetch", "origin", branch], timeout=publish._PUSH_TIMEOUT)
    if fetch.returncode != 0:
        return f"fetching origin/{branch} failed ({publish._err(fetch)})"
    merge = publish._git(repo, ["merge", "--ff-only", "--quiet", f"origin/{branch}"])
    if merge.returncode != 0:
        return (f"{branch} cannot fast-forward to origin/{branch} "
                f"({publish._err(merge) or 'the branches have diverged'})")
    return None


# --- the run -----------------------------------------------------------------


def run_once(
    state_repo: Path,
    code_dir: Path,
    *,
    branch: str,
    fetchers: Fetchers = Fetchers(),
    run_attempt: int = 1,
) -> RunReport:
    """Do whatever is due now, in `STEP_NAMES` order; see the module docstring."""
    state_repo = Path(state_repo)
    run = _Run(state_repo=state_repo, code_dir=Path(code_dir), branch=branch,
               fetchers=fetchers, run_attempt=run_attempt,
               started=db.require_utc(fetchers.clock(), "clock"))
    config.settings.ensure_dirs()
    try:
        run.record("mac_history", False, True, "not built yet (Task 12)")
        problem = _sync_checkout(run)
        if problem is not None:
            run.publish_error = True
            run.record("rebuild", True, False, f"not run: {problem}")
            run.notes.append(_plain(
                f"the run stopped before writing anything: {problem}; fix the checkout by hand"))
            raise _Halt()
        run.step("rebuild", lambda: _rebuild(run))
        run.step("espn_scoreboard", lambda: _scoreboard(run), needs_db=True)
        run.step("espn_boxscores", lambda: _boxscores(run), needs_db=True)
        run.step("injury_report", lambda: _injury(run), needs_db=True)
        run.step("odds", lambda: _odds(run), needs_db=True)
        if run.con is not None:
            _news(run)
        run.step("state_commit", lambda: _state_commit(run))
        run.step("predict", lambda: _predict(run), needs_db=True)
        run.step("grade", lambda: _grade(run), needs_db=True)
        try:
            _predictions_commit(run)
        except archive.SecretLeak as exc:
            run.unexpected = True
            run.secret_leak = True
            run.notes.append(_plain(f"stopped before the predictions commit: {exc}"))
            raise _Stop() from None
        except Exception as exc:  # noqa: BLE001
            run.unexpected = True
            run.notes.append(_plain(f"predictions commit failed: {type(exc).__name__}: {exc}"))
        run.step("raw_archive", lambda: _raw_archive(run))
        run.step("push", lambda: _push(run))
    except _Stop:
        run.notes.append("the run stopped: a secret was found; nothing more was committed or pushed")
    except _Halt:
        pass
    finally:
        if run.con is not None:
            run.con.close()
    steps = [
        run.steps.get(name) or StepOutcome(name, False, False, "not run")
        for name in STEP_NAMES
    ]
    return RunReport(
        started_at=run.started,
        finished_at=db.require_utc(fetchers.clock(), "clock"),
        branch=branch,
        steps=steps,
        state_commit=run.state_commit,
        predictions_commit=run.predictions_commit,
        pushed=run.pushed,
        site_changed=run.site_changed,
        raw_archived=run.raw_archived,
        unexpected=run.unexpected,
        publish_error=run.publish_error,
        notes=run.notes,
        push_message=run.push_message,
        secret_leak=run.secret_leak,
    )
