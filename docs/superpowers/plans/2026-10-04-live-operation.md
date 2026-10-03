# Live Operation (Result Capture + Daily Public Prediction Log) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** From opening night, capture real NBA results daily with true timestamps, and every evening predict the day's games into an append-only log that is committed and pushed to a public GitHub repo.

**Architecture:**
- `sources/results.py` captures current-season FINAL results.
- `model/live.py` builds today's slate from the schedule, predicts through `AsOfView` cut at "now", and writes log and grade lines.
- `model/publish.py` commits the log files on `main` and pushes.
- New CLI commands are `capture-results` and `predict-today`. `status` gains live checks, and two launchd jobs are added.

**Tech Stack:** Python 3.14, DuckDB, nba_api, typer, git via subprocess, launchd.

**Spec:** `docs/superpowers/specs/2026-09-22-nba-predictor-design.md`, section **7. Live operation — decided 2026-10-04**.

## Global Constraints

- **Time zones.** Local machine time is IST, and launchd uses local time. Slate dates are US-Eastern calendar dates (`America/New_York`). Every stored timestamp is UTC and goes through `db.require_utc`.
- **Testable clock.** "Now" is injectable everywhere: functions take a `now: datetime` argument, and CLI commands use `datetime.now(UTC)`. Tests never depend on the wall clock.
- **Raw-first.** Every download is archived to `raw_store` before it is parsed or written to the database. The database is opened only after all downloading is done, using `db.connect_with_retry`.
- **Captured results** are FINAL rows in the games table with `observed_at` set to the real capture time and `reconstructed = FALSE`. A game that already has any FINAL row is never inserted again.
- **The model reads results only through `AsOfView`**, the existing Stage1Predictor path.
- **The log is append-only.** Code never rewrites or deletes a line of `predictions/*.jsonl`. One JSON object per line, keys sorted, UTF-8, `\n`-terminated.
- **Git is safe:**
  - Commit only `predictions/` files, and only when the current branch is `main`.
  - Never force-push, never rebase, never touch other paths.
  - Tests use throwaway repos under `tmp_path`, never the real repo.
- **Physical `_raw` table names** appear only in db.py and asof.py.
- **No tracebacks for expected failures.** Use plain-English messages and non-zero exits.
- **Stall avoidance.** Commands over about 2 minutes run with `run_in_background` and short polling. Never run multi-minute real-archive commands needlessly.
- Baseline before this plan: `uv run pytest -q` gives **572 passed, 1 deselected** on `main` (HEAD 51d0006). Work on branch `live-operation`.

---

### Task 1: `capture-results`

**Files:**
- Create: `src/predictor/sources/results.py`
- Modify: `src/predictor/cli.py` (new command)
- Test: `tests/test_results_capture.py`

**Interfaces:**
- `results.SOURCE = "results"`
- `Downloaded(season, fetched_at, blob_key, games: list[nba_stats.GameRow])` (frozen)
- `download(season, fetched_at=None, fetch=nba_stats.fetch_season) -> Downloaded`
  - Fetches the DataFrame.
  - Archives `df.to_json(orient="split", date_format="iso").encode()`, gzipped (`mtime=0`), under key `f"{season}_{fetched_at:%Y%m%dT%H%M%S}Z.json.gz"`.
  - Re-reads the archive (`pd.read_json(..., orient="split")`) and pairs rows with `nba_stats.pair_team_rows(df, season)`.
  - Keeps only games with both points present (status FINAL).
- `CaptureResult(season, new_finals: int, already_known: int, blob_key: str)` (frozen)
- `load(con, downloaded) -> CaptureResult`
  - In one transaction: for each FINAL GameRow whose `game_id` has no FINAL row in the games table, insert a FINAL row with `observed_at=downloaded.fetched_at`, `reconstructed=False`, and the GameRow's season, date, teams and points.
  - Also insert a SCHEDULED row (NULL points) at the same `observed_at` if the game has no row at all. Without it, replay's earliest-SCHEDULED check would see nothing.
  - Rolls back and re-raises on error.

**CLI `capture-results [--season S]`:**
- The default season is `season_label(now)`.
- The download runs first. A `requests.RequestException` gives "Could not download results … nothing was saved", exit 1.
- Then `connect_with_retry`, `migrate`, `load`.
- `duckdb.Error` gives a plain message and exit 1.
- On success, print `results {season}: {new_finals} new game result(s) recorded ({already_known} already known)`.

**Tests** (fixture archive via `tests/model_fixtures.py`; the DataFrame is built in-test with the LeagueGameFinder columns that `pair_team_rows` uses: GAME_ID, GAME_DATE, MATCHUP, TEAM_ABBREVIATION, PTS):
1. New finished game → exactly one FINAL row with `observed_at == fetched_at` and `reconstructed False`.
2. Running twice → the second run adds 0 and `already_known` counts it.
3. A game already FINAL from the historical ingest (reconstructed) is not duplicated.
4. An unplayed game (PTS null) adds nothing.
5. Raw-first: the archive holds the exact payload, and parsing reads it back (monkeypatch `raw_store.load` to prove it is used).
6. The CLI download failure path prints a plain message and exits 1, and the DB is never opened (monkeypatch `db.connect_with_retry` to record calls).
7. Leak safety: an `AsOfView` cut one second before `fetched_at` does not see the new FINAL; one cut at `fetched_at` does.

Commit: `feat: capture-results records finished games with their real capture time`.

---

### Task 2: Today's slate, the prediction log, and grading

**Files:**
- Create: `src/predictor/model/live.py`
- Test: `tests/test_live_predict.py`

**Interfaces:**
- `EASTERN = ZoneInfo("America/New_York")`; `START_BUFFER = timedelta(minutes=30)`; `STALE_AFTER = timedelta(hours=36)`
- `SlateGame(game_id, season, game_date, home_team, away_team, tip_off_utc)` (frozen)
- `slate_for(con, now) -> list[SlateGame]`
  - Uses the latest schedule vintage per game (QUALIFY row_number, as in venues.py).
  - Keeps games whose `game_date == now.astimezone(EASTERN).date()`, with game-id prefix in ("002", "004", "005", "006") and non-null `tip_off_utc`.
  - Ordered by tip_off_utc, then game_id.
  - Reads the schedule table directly; schedule facts are static (spec 1.1).
- `last_capture(con) -> datetime | None`: the max `observed_at` of FINAL rows with `reconstructed = FALSE`.
- `in_season(con, now) -> bool`: true if any schedule game (competitive prefixes) has `tip_off_utc` within the last 3 days or the next 3 days of `now`.
- `log_path(repo_dir, season) -> Path`: `repo_dir / "predictions" / f"{season}.jsonl"`. `grades_path` is `…/f"{season}-grades.jsonl"`.
- `read_log(path) -> list[dict]`, returning `[]` if the file is missing.
- `predict_today(con, settings, repo_dir, now) -> RunResult`, where `RunResult(predicted: int, not_predicted: int, skipped_duplicates: int, stale: bool, lines_written: list[dict])`:
  - `stale = in_season(con, now) and (last_capture(con) is None or now - last_capture(con) > STALE_AFTER)`.
  - For each slate game, in order:
    - If the log for its season already has a line with the same `game_id` and `game_date` (any status), skip it and count it as a duplicate.
    - If `tip_off_utc - START_BUFFER <= now`, write a `not_predicted` line with reason `"game had already started (or was within 30 minutes of tip-off) when the prediction run happened"`.
    - Otherwise call `Stage1Predictor(con, settings).explain(GameToPredict(game_id, season, game_date, home, away), AsOfView(con, now))`. Build ONE predictor per run and reuse it for every game, since its catch-up is incremental.
  - Each line (append; open the file in "a" mode and write `json.dumps(line, sort_keys=True) + "\n"`):

```python
{
  "predicted_at": now.isoformat(), "game_id": ..., "season": ..., "game_date": "YYYY-MM-DD",
  "tip_off_utc": tip.isoformat(), "home_team": ..., "away_team": ...,
  "status": "predicted" | "not_predicted", "reason": None | "...",
  "spread": float|None, "p_home": float|None, "sentence": str|None,
  "terms": {"rating":..,"home":..,"rest":..,"travel":..,"altitude":..} | None,
  "settings": {"k":..,"margin_cap":..,"season_regression":..,"hca_window":..,"sigma":..,"half_life":..},
  "stale_results": bool, "last_result_capture": iso|None,
}
```

  - Floats are rounded to 6 decimals in `spread`, `p_home` and `terms`.
  - Creates `predictions/` if missing.
- `grade(con, repo_dir, season, now) -> int`
  - For each `game_id` in the season log, take the latest `predicted` line whose `predicted_at` is before that game's `tip_off_utc` (the value on the line).
  - If a FINAL result exists in the games table (latest FINAL row per game; this is a scoring read, like replay's) and the grades file has no line for that `game_id`, append `{"game_id", "predicted_at", "p_home", "home_won", "correct": (p_home >= 0.5) == home_won, "graded_at": now.isoformat()}`.
  - Returns the number appended.

**Tests** (fixture archive; `ModelSettings` built in-test as in tests/test_model_stage1.py; `repo_dir = tmp_path`):
1. The slate picks only today's US-Eastern games. A game dated today ET but 01:00 IST tomorrow is included, and yesterday's game is excluded. Hand-check with a fixed `now`, e.g. 2026-10-21 12:30 UTC, which is 18:00 IST on 10-21 and 08:30 ET on 10-21.
2. A predicted line equals the Breakdown from a fresh Stage1Predictor with `AsOfView(con, now)`: same spread, p_home and sentence.
3. A game tipping 20 minutes after `now` gets `not_predicted` with the reason. One tipping 31 minutes after gets `predicted`.
4. A same-day re-run writes 0 new lines and counts duplicates. A rescheduled game (same id, new date) is predicted again.
5. Stale: with in-season games and a last capture 40 h before `now`, every line has `stale_results True` and the right `last_result_capture`. With no capture and the off-season (no games within ±3 days), `stale` is False.
6. Append-only: write the log twice and check that the bytes of the first run's lines are a prefix of the file afterwards.
7. Leak safety: a FINAL result observed after `now` does not change the prediction (compare with and without it).
8. Grading: a predicted line plus a captured FINAL gives one grade line with the correct `correct` value, and re-grading adds nothing. A `not_predicted` line is never graded. A prediction made after tip-off is never graded.

Commit: `feat: predict today's slate into an append-only log and grade it once results arrive`.

---

### Task 3: Publishing (commit and push) and the `predict-today` command

**Files:**
- Create: `src/predictor/model/publish.py`
- Modify: `src/predictor/cli.py`
- Test: `tests/test_publish.py`, `tests/test_predict_cli.py`

**Interfaces (`publish.py`):**
- `PublishResult(committed: bool, pushed: bool, message: str)` (frozen)
- `current_branch(repo_dir) -> str | None`
- `commit_and_push(repo_dir, paths: list[Path], message: str, push: bool = True) -> PublishResult`
  - If the branch is not `main`: return `committed=False`, with a message saying the log was written but not committed because the checkout is on `<branch>`.
  - `git add -- <paths>` (only these). If nothing is staged, `committed=False`.
  - `git commit -m <message>` (the message includes the predicted date, e.g. `predictions: 2026-10-21 slate`, followed by a blank line and `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`).
  - If `push` is set: `git push origin main` with a 60 s timeout. On failure, `pushed=False` and the message says the commit is kept and the next run will push it.
  - If there is no `origin` remote: `pushed=False`, with "no GitHub remote configured yet".
  - All git calls use `subprocess.run([...], cwd=repo_dir, capture_output=True, text=True, timeout=…)` with no shell.
- `unpushed_commits(repo_dir) -> int | None`: `git rev-list --count origin/main..main`. Returns None when there is no remote or on error.

**CLI `predict-today [--no-push]`:**
1. `now = datetime.now(UTC)`; `repo_dir = PROJECT_ROOT`.
2. Load settings. On `SettingsError`, print a plain message and exit 1.
3. Open the DB **read-only** (`db.connect(read_only=True)`); prediction only reads. A lock or `duckdb.Error` gives a plain message and exit 1.
4. Season = the slate's season, or `season_label(now)` if the slate is empty.
5. Run `grade` for that season, then `predict_today`.
6. `commit_and_push` the season log and grades files. Do it even when the slate is empty but grades were appended. Skip it when nothing changed.
7. Print a summary: date, predicted / not predicted / duplicates, a stale warning if stale, grades appended, and the publish message.
8. Exit 1 only on real failures. A failed push prints a warning and exits 0, because the data was saved and the next run retries.

**Tests:**
- `test_publish.py`, using a temp git repo with a bare temp remote as `origin` (created by `git init --bare`):
  - On main, commit and push succeed, and the remote has the commit.
  - On another branch, nothing is committed.
  - Only the given paths are committed; another modified file stays unstaged.
  - With no remote, the commit happens and the push is skipped with a message.
  - With a failing remote (point `origin` to a non-existent path), it stays committed with `pushed=False`, and `unpushed_commits` returns 1.
  - Set `GIT_AUTHOR_NAME`/`EMAIL` and `GIT_COMMITTER_*` via `monkeypatch.setenv` so commits work in CI.
- `test_predict_cli.py`:
  - Patch `datetime`/now via a module-level `_now()` helper in cli that tests monkeypatch.
  - Use a fixture archive and a temp `repo_dir`; add a monkeypatchable `_repo_dir()` helper.
  - Exit 0 with a summary; a log file is created; `--no-push` doesn't push; a missing settings file gives a plain message and exit 1.

Commit: `feat: predict-today writes, commits and pushes the daily prediction log`.

---

### Task 4: Status checks and launchd jobs

**Files:**
- Modify: `src/predictor/status.py`, `src/predictor/cli.py` (status), `scripts/install_schedule.sh`
- Create: `scripts/com.predictor.results.plist` (11:00), `scripts/com.predictor.predict.plist` (18:00)
- Test: `tests/test_status.py`, `tests/test_launchd.py`

**Status additions**, as new `SourceHealth`-shaped entries appended after the table sources by a new `check_live(con, repo_dir, now) -> list[SourceHealth]`; `status` prints them in the same report:
- `live_results`:
  - Latest value is `last_capture`.
  - Stale only if `in_season(con, now)` and (no capture, or capture older than 36 h).
  - Advice: `Run: predictor capture-results, and confirm the launchd agent com.predictor.results is loaded.`
- `prediction_log`:
  - Latest value is the newest `predicted_at` in the current season's log.
  - Stale only if in season and there are games today or yesterday (ET) but no line in the last 30 h.
  - Advice: `Run: predictor predict-today, and confirm com.predictor.predict is loaded.`
- `log_published`:
  - Stale if `unpushed_commits > 0`, with advice to run `git push origin main` or check the GitHub login (`gh auth status`).
  - If there is no remote, stale with the advice "no GitHub remote configured".
  - `None` (an error) is reported, not hidden.

**Plists:** copy the structure of `com.predictor.schedule.plist`:
- ProgramArguments `[PROJECT_DIR/.venv/bin/predictor, capture-results]` at 11:00, and `[…, predict-today]` at 18:00.
- Logs go to `data/logs/results.*.log` and `data/logs/predict.*.log`.
- `RunAtLoad` is false.

`install_schedule.sh` installs all four jobs (`install_job` for each label) and echoes the times.

**Tests:**
- Status:
  - The off-season is quiet (all three live entries are not stale with no games).
  - In season with an old capture: `live_results` is stale.
  - A log line older than 30 h with games yesterday: `prediction_log` is stale.
  - Unpushed commits: `log_published` is stale (temp git repos as in test_publish).
- Launchd: both plists parse, with the right args and times. No two of the four jobs share a start minute. The install script is valid bash and names all four labels.

Commit: `feat: status reports live results and log publication; schedule both daily jobs`.

---

### Task 5: Go live (controller, with the user's confirmation for publishing)

- [ ] Full suite green.
- [ ] Merge `live-operation` into `main` (the user has approved merging this sub-project's work as part of the plan; confirm anyway per memory).
- [ ] **Confirm with the user**, then create the public repo and push:

```bash
gh repo create <name> --public --source=. --remote=origin --push
```

  Ask the user for the repo name (suggest `nba-predictor`).
- [ ] `bash scripts/install_schedule.sh`; verify all four jobs are loaded.
- [ ] Dry run: `predictor capture-results --season 2025-26` adds 0 new results (all known). `predictor predict-today --no-push` before the season reports an empty slate, then `predictor status`.
- [ ] Record in the spec and in memory.
