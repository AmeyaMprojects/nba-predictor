# Cloud Operation Implementation Plan (plan 2 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the daily live pipeline (schedule, results, injury report, odds, the single 18:00 IST call, grading, publishing, site rebuild) onto a GitHub Actions workflow that runs every 15 minutes, using the **current** model, first in shadow mode on branch `cloud-shadow`, then switched to `main` with the owner's OK.

**Architecture:**
- A state directory `state/` in the public repo holds compact, append-only JSONL files (results, schedule changes, injury report deltas, odds, ESPN box scores, player and game ID links). Each run rebuilds a throwaway DuckDB from it; DuckDB is never committed.
- New sources: ESPN scoreboard and per-game summary (`sources/espn.py`), and the latest 15-minute official injury report (`sources/injury_report.py`). ESPN games link to NBA game IDs the same way odds lines do; unlinked competitive games get a provisional `espn:<id>`.
- `cloud/run.py` decides what is due, runs it, makes a `state:` commit and a `predictions:` commit, uploads raw responses to a weekly GitHub release, and pushes through the extended `publish.py` (two commit kinds, any one configured branch, a narrow `git pull --rebase`).
- `.github/workflows/live.yml` runs it every 15 minutes and dispatches `site.yml` when predictions reach `main`. The Mac keeps a weekly history refresh that pushes `state/nba/` to branch `mac-history`. The cloud copies those files in.

**Tech Stack:** Python 3.14, DuckDB, requests, typer, git and `gh` via subprocess, GitHub Actions (`ubuntu-latest`, `astral-sh/setup-uv`), launchd (Mac history job only), PyYAML (dev only, for workflow tests).

**Spec:** `docs/superpowers/specs/2026-10-22-cloud-injuries-design.md` (Goals 1, "Cloud workflow", "State storage", "The Mac's remaining role", author's calls 2, 5 and 6). It extends `docs/superpowers/specs/2026-09-22-nba-predictor-design.md` sections 7 and 8. Plans 1 (player/injury model) and 3 (early/revision/final calls, site changes) are out of scope.

## Global Constraints

- **Time.** Store every timestamp in UTC through `db.require_utc`. "Now" is injected everywhere and tests never read the wall clock. The slate is still the ET date of `now − 7h` (`live.slate_date`). The early call is the first run at or after **18:00 IST = 12:30 UTC** on the slate date. IST has no DST.
- **One call per game per slate, as today.** This plan adds no revisions or final calls. `predict_today` dedupe is unchanged, except that provisional IDs are linked (Task 7).
- **Append-only.** Code never rewrites or deletes a line in `predictions/**.jsonl` or in any `state/` file. There are three exceptions:
  - `state/nba/*` holds whole-file snapshots written only by the Mac.
  - `state/players/aliases.json` is edited by hand.
  - `state/settings/stage1_settings.json` is an optional override.
  
  JSON lines use `json.dumps(obj, sort_keys=True, separators=(",", ":"))` and end with `\n`.
- **Commits.**
  - Subjects start with `predictions: ` (paths under `predictions/` only) or `state: ` (paths under `state/` only).
  - A commit never mixes the two.
  - Never force-push.
  - The only rebase allowed is the narrow one in Task 1.
  - The first push to any branch is made by hand.
- **Branches.**
  - `main` is public and verifiable.
  - `cloud-shadow` holds the cloud's state and would-be calls during shadow mode.
  - `mac-history` is written only by the Mac's weekly job.
  - The cloud picks its branch from env `PREDICTOR_PUBLISH_BRANCH` (`main` | `cloud-shadow`). The workflow default is `cloud-shadow`.
- **Secrets.**
  - `ODDS_API_KEY` exists only as an Actions secret, passed as env to the one `cloud-run` step.
  - It is never echoed, written to `state/`, the summary, outputs, raw archive or logs.
  - `tests/test_secrets.py` keeps passing. State files must therefore contain no standalone 32-hex run: content hashes are full 64-hex sha256.
- **Raw-first.** Every response goes to the local `raw_store` (under `$PREDICTOR_DATA_DIR`, the runner's temp dir) before it is parsed. At the end of the run the raw dir is tarred and uploaded as a release asset, never with `--clobber`.
- **Physical `_raw` table names** appear only in `db.py`/`asof.py`. Everything else uses `db.POINT_IN_TIME_TABLES`.
- **Odds budget.**
  - At most 2 windows per slate day, and at most 2 attempts per window.
  - Skip the fetch when the last recorded `x-requests-remaining` is below 30.
  - Worst case is 12 credits a day, normal is about 6.
- **ESPN politeness.** At most 1 scoreboard and 15 summary requests per run, 20 s timeout, 2 retries on network errors or 5xx.
- **No network in tests.** Inject fetchers and `gh`/git runners. Git tests use throwaway repos and bare remotes under `tmp_path`.
- **Visible failures.**
  - A source that fails is recorded as a failed step in the run summary, and the run continues.
  - Only git/publish errors or unexpected exceptions make the run exit 1.
- **Coordination with plan 1.** If plan 1 has already merged a player box-score table or 15-minute injury URL helpers, reuse its names and add the ESPN `source` column by migration instead of creating a second table. Otherwise this plan's names stand and plan 1 adopts them.
- **Baseline:** `uv run pytest -q` gives **906 passed, 1 deselected** on `main` (HEAD c5cef70). Branch: `cloud-operation`. Run the full suite with `run_in_background` (about 5 min).


## Controller rulings (2026-10-22, before execution)

- **Ruling 2026-10-23, after plan 1 merged (supersedes the "no second player table" clause below):**
  - **Two tables, not one.** Plan 1's `player_games` (`player_games_raw`) holds NBA game logs with a numeric `player_id` (BIGINT). ESPN box scores identify unlinked players as `'espn:<athlete id>'` text, which cannot share that key. So Task 2 keeps `player_boxscores_raw` as specified, for ESPN rows only.
  - **Plan 3 merges the read.** NBA rows are used when present, otherwise ESPN rows, and only rows seen before the call. That is spec call 6.
  - **Task 5** links ESPN players to NBA ids with plan 1's `predictor.model.player_ids` (`name_key`, `PlayerIndex`), not a new linker.
  - **Task 6** reuses plan 1's injury helpers: `injury_report.new_session()` for the browser User-Agent session, and the slot labels and header-time parsing in `predictor.sources.injury_slots` / `injury_report`. It must not duplicate them.
  - **`injury_reports_raw`** (header-time index) is new and complements the existing `injury_status` rows.
- **Order:** this plan runs **after** plan 1 (`2026-10-22-player-model.md`) is merged. Plan 1's table and helper names win: `player_games` for NBA player logs, and its injury-report session and slot helpers. This plan's ESPN box scores join them through a `source` column, so no second player table holds the same thing.
- **News archiver:** the owner said GitHub runs everything, so the RSS news archiver (`poll-news`, now launchd `com.predictor.daily`) moves into the cloud run once an hour, provided its feeds answer from GitHub's servers. Task 9 checks the feeds once. If a feed is blocked, that feed stays on the Mac and the run summary says so. Cost if wrong: news history keeps a Mac dependency.
- **Site trigger:** pushes made with `GITHUB_TOKEN` do not start other workflows, but `workflow_dispatch` called through the API does. Use `gh workflow run site.yml`, as the plan says.

## Review Focus

1. **Late or skipped scheduled runs.** GitHub cron starts 5–20+ minutes late and sometimes skips slots.
   - The first run after 12:30 UTC must still make the early call and fetch early odds.
   - A game that tipped before any run gets `not_predicted`.
   - Test: `test_late_first_run_still_makes_the_early_call_and_odds` (Task 7).
2. **Push races and partial runs.**
   - A human push to `main` mid-run must be rebased over, not refused, when all local commits are safe.
   - A run killed after committing but before pushing loses only its own work. The next run redoes it with no duplicate lines.
   - Tests: Task 1 rebase tests and `test_rerun_after_a_lost_push_redoes_the_work_without_duplicates` (Task 9).
3. **ESPN team and date quirks.**
   - Abbreviations `GS`, `NY`, `NO`, `SA`, `UTAH`, `WSH`, and "LA Clippers".
   - Late West-coast tips whose UTC date is the next day.
   - All-Star and exhibition teams.
   - Two-game series (same home and away on consecutive days).
   
   Each must link to the right NBA game or be skipped and reported. Tests are in Task 4.
4. **DST and injury slot naming.**
   - The 15-minute filename is built in ET (EDT→EST on 2026-11-01), and midnight is `12_00AM`.
   - The report time comes from the PDF header.
   - The early call and slate on the DST-change day are unchanged.
   
   Tests: `test_quarter_slot_names_the_15_minute_file` (Task 6), `test_early_call_on_dst_change_day` (Task 7).
5. **Provisional IDs.** A game predicted as `espn:<id>` and later linked to an NBA ID must not be predicted again, must grade against the right result, and its log line stays byte-identical. Tests are in Task 7.

---

### Task 1: Two commit kinds, any one branch, narrow pull --rebase

**Files:**
- Modify: `src/predictor/model/publish.py`, `src/predictor/config.py`, `src/predictor/cli.py` (callers pass defaults unchanged)
- Test: `tests/test_publish.py`, `tests/test_config.py`

**Interfaces:**
- Produces:
  - `CommitKind = Literal["predictions", "state"]`
  - `commit_and_push(repo_dir: Path, paths: list[Path], message: str, push: bool = True, *, kind: CommitKind = "predictions", branch: str = "main") -> PublishResult`
  - `push_pending(repo_dir: Path, *, branch: str = "main") -> PublishResult`
  - `unpushed_commits(repo_dir, branch="main") -> int | None`
  - `unpushed_non_prediction_commits(repo_dir, branch="main") -> int | None` (counts commits that are not safe of either kind)
  - `prediction_files_problem(repo_dir, branch="main") -> str | None`
  - `code_version(repo_dir) -> tuple[str | None, bool]`
  - `head_sha(repo_dir) -> str | None` (short)
  - `config.publish_branch() -> str`

**Requirements:**
1. Paths must resolve under `repo_dir/<kind dir>/`. The message must start with the kind's prefix (`"predictions: "` / `"state: "`), or the result is `error=True` and git is not touched.
2. Every `main` literal (branch check, `origin/main..main`, `push origin main`) becomes `branch`. The refusal text names the expected branch.
3. `_is_safe_commit(repo_dir, sha)` replaces `_is_safe_prediction_commit`. A commit is safe when:
   - its subject starts with `predictions: ` and every changed path is under `predictions/`, or
   - its subject starts with `state: ` and every changed path is under `state/`.
   
   Anything else, including a mixed commit, is unsafe.
4. **Narrow rebase**, inside `_push_if_safe`, after the existing "every unpushed commit is safe" check:
   - `git fetch origin <branch>` (60 s).
   - If `rev-list --count <branch>..origin/<branch>` > 0, run `git pull --rebase --no-autostash origin <branch>`.
   - On failure, run `git rebase --abort` and return `pushed=False, error=False`, with a message naming the conflict ("kept the local commit(s); nothing pushed").
   - Push. If the push is rejected as non-fast-forward, repeat fetch+rebase+push once.
   - Never run the rebase when any unpushed commit is unsafe.
5. `code_version` dirty-check excludes both `predictions` and `state`.
6. `config.publish_branch()`: env `PREDICTOR_PUBLISH_BRANCH`, stripped, default `"main"`. Allowed values are `{"main", "cloud-shadow"}`. Anything else raises `ValueError("PREDICTOR_PUBLISH_BRANCH must be main or cloud-shadow, not '<v>'")`.
7. Update the module docstring: "never rebases" becomes the narrow rule above.

**Tests** (existing tests must still pass unchanged):
```python
def test_state_commit_touching_only_state_is_committed_and_pushed(tmp_path): ...
    # kind="state", message "state: cloud run 2026-10-22T12:37Z" -> committed and pushed True
def test_state_kind_with_a_predictions_path_is_refused(tmp_path): ...
    # result.error is True and "outside" in result.message; git log unchanged
def test_message_without_the_kind_prefix_is_refused(tmp_path): ...
    # kind="state", message "predictions: x" -> error True
def test_push_refused_when_a_state_commit_touches_predictions(tmp_path): ...
    # a hand-made "state: x" commit changing predictions/a.jsonl -> pushed False
def test_pull_rebase_when_remote_is_ahead_and_all_local_commits_are_safe(tmp_path): ...
    # remote gets "docs: readme" (README.md); local "predictions: 2026-10-22 slate"
    # -> pushed True; remote subjects[:2] == ["predictions: 2026-10-22 slate", "docs: readme"];
    #    `git rev-list --merges origin/main` is empty
def test_no_pull_when_any_local_commit_is_unsafe(tmp_path): ...
    # remote ahead + local "feat: x" -> pushed False; local HEAD unchanged; no rebase-merge dir
def test_rebase_conflict_is_aborted_and_commit_kept(tmp_path): ...
    # remote edits the same predictions line -> pushed False, error False;
    # publish._operation_in_progress(repo) is None; HEAD subject is ours
def test_commit_on_cloud_shadow_pushes_only_that_branch(tmp_path): ...
    # branch="cloud-shadow": origin/cloud-shadow advanced, origin/main sha unchanged
def test_branch_mismatch_commits_nothing(tmp_path): ...
    # checkout main, branch="cloud-shadow" -> committed False, "cloud-shadow" in message
def test_code_version_ignores_changes_under_state(tmp_path): ...
# tests/test_config.py
def test_publish_branch_defaults_to_main(monkeypatch): ...
def test_publish_branch_reads_cloud_shadow(monkeypatch): ...
def test_publish_branch_rejects_other_values(monkeypatch): ...   # "dev" -> ValueError
```

- [ ] Write the tests and run `uv run pytest tests/test_publish.py tests/test_config.py -q`. The new tests fail.
- [ ] Implement, then run the same command (PASS) and the full suite.
- [ ] Commit: `feat: publish state commits, to a chosen branch, with a narrow safe rebase`

---

### Task 2: New tables and one competitive-game rule

**Files:**
- Modify: `src/predictor/db.py`, `src/predictor/model/venues.py`, `src/predictor/model/live.py`, `src/predictor/sources/odds.py`, `src/predictor/model/stage1.py`, `src/predictor/model/fit.py`, `src/predictor/status.py` (staleness entries)
- Test: `tests/test_db.py`, `tests/test_model_venues.py`, `tests/test_live_predict.py`

**Interfaces:**
- Produces:
  - `venues.is_competitive(game_id: str) -> bool`
  - `venues.competitive_sql(column: str = "game_id") -> str`, which returns `(substr(game_id, 1, 3) IN ('002','004','005','006') OR game_id LIKE 'espn:%')`
  - logical tables `"injury_reports"` → `injury_reports_raw` and `"player_boxscores"` → `player_boxscores_raw`

**Requirements:**
1. Add the two new tables to `_SCHEMA`. `migrate` stays idempotent.
   ```sql
   injury_reports_raw (observed_at TIMESTAMPTZ NOT NULL PRIMARY KEY,   -- PDF header time
                       fetched_at TIMESTAMPTZ NOT NULL, source_key VARCHAR NOT NULL,
                       row_count INTEGER NOT NULL, content_sha VARCHAR NOT NULL)
   player_boxscores_raw (game_id VARCHAR NOT NULL, season VARCHAR NOT NULL, game_date DATE NOT NULL,
                         team VARCHAR NOT NULL, player_id VARCHAR NOT NULL,   -- NBA id, or 'espn:<athlete id>'
                         espn_player_id VARCHAR, player_name VARCHAR, starter BOOLEAN,
                         did_not_play BOOLEAN NOT NULL, minutes DOUBLE, points INTEGER,
                         rebounds INTEGER, assists INTEGER, plus_minus INTEGER,
                         source VARCHAR NOT NULL, observed_at TIMESTAMPTZ NOT NULL,
                         PRIMARY KEY (game_id, player_id, source, observed_at))
   ```
2. Replace every `substr(game_id, 1, 3) IN (...)` site with `competitive_sql(...)`: live.py ×4, odds.py, stage1.py, fit.py. Replace the Python prefix check at `venues.py:40` with `is_competitive`. Provisional IDs are only ever minted for competitive games (Task 4).
3. `status.STALENESS_HOURS` gains `"injury_reports": 36` and `"player_boxscores": 24 * 150`, so `check_sources` reports them sensibly.

**Tests:**
```python
def test_is_competitive():
    assert is_competitive("0022600001") and is_competitive("espn:401705555")
    assert not is_competitive("0012600001")
def test_new_tables_exist_and_migrate_is_idempotent(tmp_path): ...  # migrate twice, both tables present
def test_slate_includes_a_provisional_espn_game(tmp_path): ...
    # schedule row game_id "espn:401705555", game_date 2026-10-22, tip 23:00Z
    # -> [g.game_id for g in slate_for(con, 2026-10-22T13:00Z)] == ["espn:401705555"]
```
- [ ] Write the tests and confirm they fail. Implement. Run `uv run pytest tests/test_db.py tests/test_model_venues.py tests/test_live_predict.py -q`, then the full suite (the leak and physical-name tests must still pass).
- [ ] Commit: `feat: tables for injury report times and player box scores; provisional ESPN games count as competitive`

---

### Task 3: State directory, DB rebuild, export from the Mac

**Files:**
- Create: `src/predictor/cloud/__init__.py`, `src/predictor/cloud/state.py`
- Modify: `src/predictor/cli.py` (`export-state`)
- Test: `tests/test_cloud_state.py`

**Interfaces:**
- Produces:
  - `STATE_DIRNAME = "state"`
  - `Family(name: str, table: str, columns: tuple[str, ...], path_for: Callable[[dict], str])` (frozen); `FAMILIES: dict[str, Family]` with keys `results`, `schedule`, `odds`, `boxscores`, `nba_games`, `nba_schedule`
  - `append_rows(state_dir: Path, family: str, rows: list[dict]) -> list[Path]` (touched files)
  - `read_rows(state_dir: Path, family: str) -> Iterator[dict]`
  - `load_rows(con, family: str, rows: list[dict], id_links: dict[str, str]) -> int` (INSERT OR IGNORE; returns inserted)
  - `write_snapshot(state_dir: Path, rel_path: str, rows: list[dict]) -> Path` (refuses unless `rel_path` starts with `nba/`)
  - `read_id_links(state_dir) -> dict[str, str]` (provisional → NBA id, from `state/schedule/id_links.jsonl`)
  - `RebuildReport(rows: dict[str, int], seconds: float)`
  - `rebuild_db(state_dir: Path, db_path: Path) -> RebuildReport`
  - `export_from_db(con, state_dir: Path, seasons: list[str]) -> list[Path]`
  - `settings_path(state_dir: Path) -> Path`
- Consumes: Task 2 tables. Injury loading is not part of this task: Task 6 adds the `cloud.injuries.expand_into(con, state_dir)` call to `rebuild_db`.

**Layout** (paths relative to `state/`; `<season>` like `2026-27`; months from the row's ET `game_date`, or from `observed_at` for odds):

| Family | Path | Writer |
|---|---|---|
| results (games rows, `source` kept in state only) | `results/<season>.jsonl` | cloud |
| schedule (change rows only) | `schedule/<season>/<YYYY-MM>.jsonl` | cloud |
| ESPN event links | `schedule/espn_links.jsonl` (`espn_id`, `game_id`, `linked_at`) | cloud |
| provisional → NBA | `schedule/id_links.jsonl` (`provisional_id`, `game_id`, `linked_at`) | cloud |
| injuries (report + set/drop lines) | `injuries/<season>/<YYYY-MM>.jsonl` | cloud (Task 6) |
| odds (live rows) | `odds/<season>/<YYYY-MM>.jsonl`; attempts `odds/attempts-<YYYY-MM>.jsonl` | cloud |
| boxscores | `boxscores/<season>/<YYYY-MM>.jsonl` | cloud (Task 5) |
| player links / aliases | `players/links.jsonl`, `players/aliases.json` | cloud / owner |
| Mac snapshots | `nba/games-<season>.jsonl`, `nba/schedule-<season>.jsonl`, `nba/players-<season>.jsonl` | Mac only |
| settings override | `settings/stage1_settings.json` | Mac/owner |

**Requirements:**
1. `rebuild_db`:
   - deletes `db_path` if present, connects, migrates
   - loads `nba_*` families, then cloud families, then injuries
   - maps every `espn:` game_id that has an id link to its NBA id while loading
   - returns row counts per logical table
   
   It must take under 30 s for 2014-15..2026-27 (print the time in the test as information; no assert).
2. `export_from_db` (Mac):
   - Writes `nba/games-<season>.jsonl` with all games rows.
   - Writes `nba/schedule-<season>.jsonl` with only the vintages where any of `game_date, tip_off_utc, home_team, away_team, arena_name, arena_city, arena_state, is_neutral_reported, is_neutral` differs from the previous kept vintage of that game.
   - Sorts by `(game_id, observed_at)` so that an unchanged DB re-exports byte-identical files.
3. `settings_path` returns `state/settings/stage1_settings.json` if it exists, else `model.settings.SETTINGS_PATH`.
4. CLI `export-state --to PATH [--seasons all|current]`:
   - The default `current` means `season_label(now)` and its previous season.
   - Opens the Mac DB read-only and prints `wrote <n> rows to <m> file(s)`.

**Tests:**
```python
def test_append_rows_partitions_schedule_by_season_and_month(tmp_path): ...
    # row game_date 2026-11-03, season "2026-27" -> touched == [state/schedule/2026-27/2026-11.jsonl]
def test_append_never_rewrites_existing_lines(tmp_path): ...  # first line bytes identical after second append
def test_write_snapshot_refuses_outside_nba(tmp_path): ...    # "results/x.jsonl" -> ValueError
def test_rebuild_roundtrip_equals_source_db(tmp_path): ...
    # fixture DB -> export_from_db(all) -> rebuild_db: games_raw rows equal as sets;
    # latest schedule vintage per game equal
def test_schedule_export_drops_unchanged_vintages(tmp_path): ...  # 3 vintages, only 3rd moves tip -> 2 rows
def test_rebuild_maps_provisional_ids(tmp_path): ...
    # results row "espn:401705555" + id link -> games_raw has "0042600101", no "espn:" rows
def test_rebuild_starts_from_a_fresh_file(tmp_path): ...       # junk table in old file is gone
def test_settings_override_used_when_present(tmp_path): ...
def test_export_state_cli_prints_counts(tmp_path, monkeypatch): ...
```
- [ ] Write the tests and confirm they fail. Implement. Run `uv run pytest tests/test_cloud_state.py -q` (PASS). Commit: `feat: state directory of append-only JSONL and a DuckDB rebuilt from it`

---

### Task 4: ESPN scoreboard → schedule, finals, game links

**Files:**
- Create: `src/predictor/sources/espn.py`, `src/predictor/cloud/games.py`
- Modify: `src/predictor/teams.py`
- Test: `tests/test_espn_scoreboard.py` (ESPN-shaped JSON built in the test by a helper `espn_event(...)`). Also save one real trimmed response, captured on the Mac, to `tests/fixtures/espn/scoreboard_sample.json` and use it in one parse test.

**Interfaces:**
- Produces:
  - `espn.SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"`
  - `espn.RAW_SOURCE = "espn"`
  - `espn.fetch_scoreboard(first: date, last: date, session=None) -> bytes` (params `{"dates": f"{first:%Y%m%d}-{last:%Y%m%d}", "limit": 1000}`)
  - `espn.archive(kind: str, body: bytes, fetched_at: datetime, ident: str) -> str` (raw_store key `f"{kind}_{ident}_{fetched_at:%Y%m%dT%H%M%SZ}.json"`)
  - `espn.EspnGame` (frozen): `espn_id, season, season_type: int, game_date: date, tip_off_utc: datetime | None, home_team: str | None, away_team: str | None, home_name, away_name, neutral: bool, venue_name, venue_city, venue_state, completed: bool, home_points: int | None, away_points: int | None`
  - `espn.parse_scoreboard(body: bytes) -> list[EspnGame]`
  - `teams.ESPN_ABBR = {"GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}`
  - `teams.espn_team_abbr(abbr: str | None, display_name: str | None) -> str | None`
  - `games.COMPETITIVE_SEASON_TYPES = frozenset({2, 3, 5})`
  - `games.GameSync(schedule_rows, finals, espn_links, id_links: list[dict], skipped: list[str])`
  - `games.sync_games(con, espn_games: list[EspnGame], fetched_at: datetime, known_links: dict[str, str]) -> GameSync` (`known_links`: espn_id → game_id from `espn_links.jsonl`, latest wins)
- Consumes: `odds.build_link_index`, `odds.link_game`, `odds.latest_schedule_games`, `results.FINISHED_AFTER_TIP`, `state.append_rows`/`load_rows`.

**Requirements:**
1. Parse:
   - `game_date` is the ET date of the event's `date`.
   - `season` comes from `season.year` (the end year): 2027 → `"2026-27"`.
   - `tip_off_utc` is None when `competitions[0].timeValid` is false.
   - Teams map through `espn_team_abbr`: `team_abbr(display_name)` first, then `ESPN_ABBR`, then the abbreviation itself if `team_abbr` accepts it, else None.
   - `completed` comes from `status.type.completed`. Points are ints, or None when not completed.
2. `sync_games`, for each event:
   - **Skip** season types outside `{2, 3, 5}`, and any event with a None team. Add `"<AWAY>@<HOME> <date> (<reason>)"` to `skipped`.
   - **game_id:**
     - If `known_links` gives an NBA id, use it.
     - Otherwise link by ET date + teams with `link_game` (exact date first, ±1 day only if unique).
     - Otherwise use `f"espn:{espn_id}"`.
     
     A new or changed link appends an `espn_links` row. A provisional link replaced by an NBA id also appends an `id_links` row.
   - **Schedule row:**
     - Emit one only when `game_date`, `tip_off_utc`, `home_team`, `away_team` or `is_neutral` differs from the latest DB row of that game.
     - A linked game copies arena and neutral fields from its latest NBA row, and takes date and tip from ESPN.
     - A provisional game uses ESPN venue fields and `is_neutral_reported = is_neutral = neutral`.
     - An ESPN tip of None never replaces a known tip.
     - `observed_at = fetched_at`.
   - **Final:** only if `completed`, both points are set, `fetched_at >= tip + FINISHED_AFTER_TIP`, and the game has no FINAL row. Write a games row with `status 'FINAL'`, `reconstructed False`, `observed_at fetched_at`, and `source "espn"` (in state only).
3. The scoreboard window is ET dates `slate_date(now) − 2` to `slate_date(now) + 7`.

**Tests:**
```python
def test_parse_scoreboard_reads_teams_tip_and_score(): ...
    # date "2026-10-23T02:00Z", GS home 118, LAL away 112, completed
    # -> game_date == date(2026,10,22), season == "2026-27", home_team == "GSW", home_points == 118
def test_west_coast_late_tip_keeps_et_date(): ...       # "2026-10-23T02:30Z" -> 2026-10-22
def test_time_not_valid_gives_no_tip(): ...
def test_espn_abbreviations_map():
    assert espn_team_abbr("GS", "Golden State Warriors") == "GSW"
    assert espn_team_abbr("UTAH", "Utah Jazz") == "UTA"
    assert espn_team_abbr("LAC", "LA Clippers") == "LAC"
    assert espn_team_abbr("NY", "") == "NYK"
    assert espn_team_abbr("LEB", "Team LeBron") is None
def test_all_star_and_preseason_are_skipped_and_reported(): ...  # len(skipped) == 2, no rows
def test_links_to_nba_id_by_et_date_and_teams(): ...
def test_unlinked_competitive_game_gets_provisional_id(): ...    # -> "espn:401705555"
def test_provisional_reconciled_when_nba_schedule_appears(): ...
    # known_links {"401705555": "espn:401705555"}; NBA row 0042600101 same date/teams
    # -> id_links == [{"provisional_id": "espn:401705555", "game_id": "0042600101", "linked_at": fetched_at.isoformat()}]
def test_two_game_series_links_each_date_exactly(): ...           # SAC@LAL on D and D+1 -> each own id
def test_schedule_row_only_on_change(): ...                       # second identical sync -> schedule_rows == []
def test_tbd_never_erases_known_tip(): ...
def test_final_only_after_finished_after_tip(): ...
    # tip 2026-10-22T23:30Z; fetched 2026-10-23T02:59Z -> finals == []; at 03:00Z -> 1 final
def test_existing_final_not_duplicated(): ...
def test_linked_game_keeps_nba_arena_fields(): ...
```
- [ ] Write the tests and confirm they fail. Implement. Run `uv run pytest tests/test_espn_scoreboard.py -q` (PASS). Commit: `feat: ESPN scoreboard feeds schedule, results and game links`

---

### Task 5: ESPN box scores and player links

**Files:**
- Modify: `src/predictor/sources/espn.py`
- Create: `src/predictor/cloud/players.py`
- Test: `tests/test_espn_boxscore.py`, `tests/test_player_links.py`

**Interfaces:**
- Produces:
  - `espn.SUMMARY_URL = ".../basketball/nba/summary"`
  - `espn.fetch_summary(espn_id: str, session=None) -> bytes` (param `event`)
  - `espn.EspnFormatError(Exception)`
  - `espn.EspnPlayerLine` (frozen): `espn_player_id, name, team, starter: bool, did_not_play: bool, minutes: float | None, points, rebounds, assists, plus_minus: int | None`
  - `espn.parse_boxscore(body: bytes) -> list[EspnPlayerLine]`
  - `players.normalize_name(name: str) -> str`
  - `players.PlayerIndex.build(state_dir: Path) -> PlayerIndex`, built from `nba_api.stats.static.players.get_players()` (offline) plus `state/nba/players-<season>.jsonl` (`player_id, name, team`)
  - `players.PlayerLink` (frozen): `espn_player_id, nba_player_id: int | None, name, team, method: Literal["existing","alias","name","name+team","unmatched"], linked_at`
  - `players.link_player(line: EspnPlayerLine, index, aliases: dict, existing: dict[str, PlayerLink], now) -> PlayerLink`
  - `players.boxscores_due(con, now) -> list[tuple[str, str]]`: (game_id, espn_id) for FINAL games dated within 3 days before the slate with an ESPN link and no box-score rows, at most `MAX_SUMMARIES_PER_RUN = 15`
  - `players.box_rows(game, lines, links, observed_at) -> list[dict]`

**Requirements:**
1. Stat indexes come from `statistics[0].keys` (`minutes`, `points`, `rebounds`, `assists`, `plusMinus`), never from position or label. A missing key raises `EspnFormatError` naming the key.
   - `"+12"` → 12, `"-3"` → −3.
   - A DNP player gets `did_not_play True` and `minutes None`.
2. `normalize_name`:
   - NFKD with accents stripped, then lowercase.
   - Apostrophes and periods removed; hyphens become spaces.
   - Suffix tokens `jr sr ii iii iv` dropped, spaces collapsed.
3. Linking order:
   - An existing non-null link is reused.
   - Then `aliases.json` `{"by_espn_id": {"<id>": nba_id}, "by_name": {"<normalized>": nba_id}}`.
   - Then a unique normalized-name match.
   - Then several matches filtered by team (Mac players file).
   - Otherwise `unmatched`.
   
   New or changed links append to `players/links.jsonl`. Every run re-tries the unmatched ones in memory and appends only when a link is newly found.
4. State box-score rows store `espn_player_id` and no NBA id. `rebuild_db` resolves `player_id` from the latest links (NBA id as a string, else `espn:<athlete id>`), so a later alias fixes old rows. `observed_at` is the summary fetch time.

**Tests:**
```python
def test_parse_boxscore_reads_by_keys(): ...   # starter minutes 34.0, points 27, plus_minus 12 from "+12"
def test_dnp_player(): ...                      # did_not_play True, minutes None
def test_missing_stat_key_raises(): ...          # no "plusMinus" -> EspnFormatError("plusMinus" in str)
def test_normalize_name():
    assert normalize_name("Luka Dončić") == "luka doncic"
    assert normalize_name("Jaren Jackson Jr.") == "jaren jackson"
    assert normalize_name("Shai Gilgeous-Alexander") == "shai gilgeous alexander"
    assert normalize_name("De'Aaron Fox") == "deaaron fox"
def test_alias_wins_over_name(): ...
def test_ambiguous_name_resolved_by_team(): ...  # two "marcus morris" -> method "name+team"
def test_unmatched_is_stored_and_reported(): ...
def test_rebuild_resolves_player_id_after_alias_added(tmp_path): ...
    # before alias: player_id == "espn:3945274"; after alias by_espn_id {"3945274": 1630178}: "1630178"
def test_boxscores_due_caps_at_15(): ...
```
- [ ] Write the tests and confirm they fail. Implement. Run `uv run pytest tests/test_espn_boxscore.py tests/test_player_links.py -q` (PASS). Commit: `feat: ESPN box scores with seen-at times, linked to NBA player IDs`

---

### Task 6: Latest 15-minute injury report, stored as deltas

**Files:**
- Modify: `src/predictor/sources/injury_report.py`, `src/predictor/cloud/state.py` (call `expand_into` from `rebuild_db`)
- Create: `src/predictor/cloud/injuries.py`
- Test: `tests/test_injury_latest.py`, `tests/test_cloud_injuries.py`

**Interfaces:**
- Produces:
  - `injury_report.BROWSER_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"`
  - `injury_report.quarter_slot(at_utc: datetime) -> tuple[date, str]` (ET date and label `"05_30PM"`)
  - `injury_report.report_url_15(day: date, label: str) -> str` (`f"{BASE_URL}/Injury-Report_{day.isoformat()}_{label}.pdf"`) and `raw_key_15` (the same file name)
  - `injury_report.LatestReport(key: str, body: bytes, fetched_at: datetime)`
  - `injury_report.fetch_latest(now: datetime, session=None, max_back: int = 4) -> LatestReport | None`
  - `injuries.canonical_rows(parsed: ParsedReport) -> list[dict]`
  - `injuries.content_sha(rows) -> str` (sha256 hex, 64 chars)
  - `injuries.delta(prev: list[dict], cur: list[dict], report_at: datetime) -> list[dict]`
  - `injuries.InjuryOutcome(new: bool, report_at: datetime | None, rows: int, changes: int)`
  - `injuries.record_report(con, state_dir: Path, latest: LatestReport) -> InjuryOutcome`
  - `injuries.expand_into(con, state_dir: Path) -> int`

**Requirements:**
1. `fetch_latest`:
   - Tries the slot at `now` floored to 15 minutes in ET, then up to `max_back` earlier slots (5 requests at most), with a session whose `User-Agent` is `BROWSER_USER_AGENT`.
   - 403/404 means step back. A 200 with a `%PDF` body wins.
   - A non-PDF 200 steps back and logs the event.
   - `TransientFetchError` propagates.
   - It archives to `raw_store("injury", key)` before returning.
2. The report time is `parse_report(body).published_at` (the header), never the filename.
3. `record_report`:
   - Returns `new=False` if `report_at` is ≤ the latest `injury_reports` observed_at.
   - Otherwise appends one `{"kind":"report","report_at","fetched_at","source_key","rows","content_sha"}` line, plus `set` lines (full row) and `drop` lines (`game_date, team, player`). Each delta is keyed by `(game_date, team, player)` against the previous distinct report's rows read from the DB.
   - Inserts the `injury_reports` row.
   - When `changes > 0`, inserts the full snapshot through the existing `injury_report.ingest_report(con, body)`.
   - `InjuryReportEmptyError` counts as a report with 0 rows; every previous row is dropped.
4. `expand_into` replays the lines in time order per season, starting from empty. Every report becomes an `injury_reports` row. Only reports with ≥ 1 change get full `injury_status` rows, at `observed_at = report_at`.

**Tests:**
```python
def test_quarter_slot_names_the_15_minute_file():
    assert quarter_slot(dt("2026-10-22T21:37Z")) == (date(2026,10,22), "05_30PM")   # EDT
    assert quarter_slot(dt("2026-11-03T22:37Z")) == (date(2026,11,3), "05_30PM")    # EST
    assert quarter_slot(dt("2026-10-23T04:05Z")) == (date(2026,10,23), "12_00AM")
    assert quarter_slot(dt("2026-10-22T16:00Z")) == (date(2026,10,22), "12_00PM")
def test_fetch_latest_steps_back_over_unpublished_slots(): ...
    # 403 for 05_30PM, PDF for 05_15PM -> key == "Injury-Report_2026-10-22_05_15PM.pdf"
def test_fetch_latest_sends_a_browser_user_agent(): ...
def test_fetch_latest_gives_up_after_five_slots(): ...      # None, exactly 5 requests
def test_report_time_comes_from_header_not_filename(tmp_path): ...
    # fixture Injury-Report_2025-01-15_05PM.pdf under a fake 15-min key -> report_at == parse_report(body).published_at
def test_delta_records_only_changes(): ...
    # prev {A Out, B Questionable, C Out}, cur {A Out, B Out} -> [set B Out, drop C] (2 lines)
def test_unchanged_report_adds_only_a_report_line(tmp_path): ...
def test_older_or_same_report_is_not_recorded_again(tmp_path): ...
def test_rebuild_expands_snapshots(tmp_path): ...
    # r1 (3 rows), r2 (changes -> 2 rows), r3 (unchanged): injury_status rows at r1 == 3, at r2 == 2,
    # none at r3; injury_reports rows == 3
def test_content_sha_is_64_hex(): ...
```
- [ ] Write the tests and confirm they fail. Implement. Run `uv run pytest tests/test_injury_latest.py tests/test_cloud_injuries.py tests/test_injury_fetch.py -q` (PASS). Commit: `feat: fetch the latest 15-minute injury report and keep it as deltas`

---

### Task 7: Due rules, provisional-ID links, and the state version on calls

**Files:**
- Modify: `src/predictor/model/live.py`, `src/predictor/cli.py` (Mac `predict-today` unchanged in behaviour)
- Test: `tests/test_live_predict.py`, `tests/test_cloud_due.py`

**Interfaces:**
- Produces (in `live.py`):
  - `EARLY_CALL_IST = time(18, 0)`
  - `early_call_at(slate: date) -> datetime` (UTC)
  - `early_call_due(now: datetime) -> bool` (`now >= early_call_at(slate_date(now))`)
  - `ODDS_EARLY_WINDOW = timedelta(hours=2)`, `ODDS_PRE_TIP_LEAD = timedelta(minutes=75)`, `ODDS_MAX_ATTEMPTS = 2`, `ODDS_MIN_REMAINING = 30`
  - `OddsWindow(name: Literal["early", "pre_tip"], opens: datetime, closes: datetime)` (frozen)
  - `odds_window_due(con, now: datetime, attempts: list[dict], remaining: int | None) -> OddsWindow | None`
  - `predict_today(con, settings, repo_dir, now, *, code_dir: Path | None = None, state_version: str | None = None) -> RunResult`
  - `id_links_path(repo_dir, season) -> Path` (`predictions/id-links/<season>.jsonl`)
  - `read_id_links(repo_dir, season) -> dict[str, str]`
  - `append_id_links(repo_dir, links: list[dict]) -> list[Path]`
  - `grade(...)` keeps its signature

**Requirements:**
1. Odds windows. `first_tip` is the minimum tip of `slate_for(con, now)`.
   - **early** opens at `early_call_at(slate)` and closes at `min(opens + 2h, first_tip − 75 min)`.
   - **pre_tip** opens at `first_tip − 75 min` and closes at `first_tip`.
   - A window is due when all of these hold:
     - `opens <= now < closes`
     - a slate game tips more than `START_BUFFER` after `now`
     - no live (`theoddsapi`) row has `observed_at >= opens`
     - fewer than 2 attempts are recorded with `attempted_at >= opens`
     - `remaining` is None or ≥ 30
   
   Attempts come from `state/odds/attempts-*.jsonl` (`attempted_at, window, ok, remaining`; never the key).
2. Each predicted and not_predicted line gains `"state_version"` (null on the Mac). `code_version` reads `code_dir or repo_dir`.
3. Provisional links. `read_id_links` maps provisional → NBA id.
   - `predict_today`'s `keys_for` treats a logged line under P as covering N.
   - `grade` looks up the FINAL row and the current tip under the line's own id, then its linked id. The grade line gains `"linked_game_id"` (null when none). `already_graded` considers both ids.
   - Logged lines are never touched.
   - `append_id_links` appends `{provisional_id, game_id, linked_at}` only for provisional IDs present in that season's log and not yet recorded.
   - The site ignores this subdirectory, because `site/src/lib/data.ts` reads only top-level `*.jsonl`. Confirm this by reading the file, and that unknown line keys are ignored.

**Tests:**
```python
def test_early_call_time_is_1230_utc():
    assert early_call_at(date(2026,10,22)) == datetime(2026,10,22,12,30,tzinfo=UTC)
    assert early_call_at(date(2026,11,3)) == datetime(2026,11,3,12,30,tzinfo=UTC)
def test_early_call_not_due_before_1800_ist_even_when_slate_rolled():
    assert slate_date(dt("2026-10-23T11:15Z")) == date(2026,10,23) and not early_call_due(dt("2026-10-23T11:15Z"))
    assert not early_call_due(dt("2026-10-23T12:29:59Z")) and early_call_due(dt("2026-10-23T12:30Z"))
    assert early_call_due(dt("2026-10-23T03:00Z"))           # slate 2026-10-22
def test_early_call_on_dst_change_day():
    assert slate_date(dt("2026-11-02T11:15Z")) == date(2026,11,1) and early_call_due(dt("2026-11-02T11:15Z"))
def test_odds_early_window_due_once(): ...   # tips 23:00Z; 12:37Z -> "early"; row observed 12:38Z -> None at 12:52Z
def test_odds_pre_tip_window(): ...          # 21:50Z -> "pre_tip"; 21:40Z -> None; 23:00Z -> None
def test_odds_attempts_cap(): ...            # two failed attempts after opens -> None
def test_odds_skipped_when_quota_low(): ...  # remaining 29 -> None; 30 -> "early"
def test_odds_not_due_on_empty_slate(): ...
def test_late_first_run_still_makes_the_early_call_and_odds(): ...
    # no earlier run; now 14:20Z, first tip 23:00Z -> early_call_due True, window "early"
def test_state_version_written_on_every_line(tmp_path): ...  # "abc1234" on each line; None by default
def test_code_version_taken_from_code_dir(tmp_path): ...
def test_provisional_line_blocks_repredicting_linked_game(tmp_path): ...
    # log: predicted "espn:401705555" 2026-12-16; link -> "0042600101"; slate has 0042600101
    # -> skipped_duplicates == 1, no new line, log bytes unchanged
def test_grade_uses_linked_result_and_keeps_provisional_id(tmp_path): ...
    # grade line game_id "espn:401705555", linked_game_id "0042600101"
def test_append_id_links_only_for_logged_provisional_ids_once(tmp_path): ...
```
- [ ] Write the tests and confirm they fail. Implement. Run `uv run pytest tests/test_cloud_due.py tests/test_live_predict.py tests/test_predict_cli.py -q`, then the full suite. Commit: `feat: early-call and odds windows for 15-minute runs; provisional game IDs linked without rewriting the log`

---

### Task 8: Raw archive as weekly release assets, with a secret guard

**Files:**
- Create: `src/predictor/cloud/archive.py`
- Test: `tests/test_cloud_archive.py`

**Decision:** raw responses go to GitHub **release assets**, not a branch.
- The injury PDFs alone are about 6 MB a day, which is over 2 GB a year of git history on a branch every runner would have to fetch to append. Release assets add nothing to clone size.
- They are append-only by name: `gh release upload` without `--clobber` fails on an existing name.
- They survive across runs.
- One release per ISO week caps assets at 96 × 7 = 672, under GitHub's 1,000-per-release limit.

**Interfaces:**
- Produces:
  - `release_tag(now: datetime) -> str` (`f"raw-{iso_year}-W{week:02d}"`)
  - `asset_name(now: datetime, run_attempt: int) -> str` (`f"run-{now:%Y%m%dT%H%MZ}-a{run_attempt}.tar.gz"`)
  - `bundle(raw_dir: Path, out: Path) -> Path | None` (tar.gz of the whole raw dir: every blob there is from this run, since the data dir is fresh per run; None when empty)
  - `class SecretLeak(Exception)`
  - `assert_no_secret(paths: Iterable[Path], secret: str | None) -> None` (byte search; no-op when secret is None)
  - `ArchiveResult(ok: bool, message: str)`
  - `upload(tag: str, asset: Path, runner=subprocess.run) -> ArchiveResult`

**Requirements:**
1. `upload`:
   - Runs `gh release view <tag>`. If it is missing, runs `gh release create <tag> --title "Raw archive <tag>" --notes "Append-only raw source responses, one asset per cloud run." --prerelease --latest=false`.
   - Then runs `gh release upload <tag> <asset>`.
   - Never passes `--clobber`. Every call has a 120 s timeout. A failure becomes `ok=False` with stderr, never an exception.
2. The orchestrator calls `assert_no_secret` on every file in the raw dir and the state files it touched, before upload and before commit.

**Tests:**
```python
def test_release_tag_is_iso_week():
    assert release_tag(dt("2026-10-22T13:07Z")) == "raw-2026-W43"
    assert release_tag(dt("2027-01-01T00:07Z")) == "raw-2026-W53"
def test_asset_name(): assert asset_name(dt("2026-10-22T13:07Z"), 1) == "run-20261022T1307Z-a1.tar.gz"
def test_upload_never_clobbers(): ...            # no recorded argv contains "--clobber"
def test_release_created_once_when_missing(): ... # view fails -> create then upload (3 calls)
def test_secret_in_raw_blob_refuses_upload(tmp_path): ...
    # planted "0123456789abcdef0123456789abcdef" -> SecretLeak; runner never called
def test_bundle_contains_manifest_and_blobs(tmp_path): ...
```
- [ ] Write the tests and confirm they fail. Implement. PASS. Commit: `feat: archive each cloud run's raw responses as a weekly release asset`

---

### Task 9: The run orchestrator, run summary, and `cloud-run`

**Files:**
- Create: `src/predictor/cloud/run.py`, `src/predictor/cloud/summary.py`
- Modify: `src/predictor/cli.py` (`cloud-run`)
- Test: `tests/test_cloud_run.py`

**Interfaces:**
- Consumes: Tasks 1–8, and `odds.download`/`odds.load`.
- Produces:
  - `StepOutcome(name: str, due: bool, ok: bool, detail: str)` (frozen)
  - `RunReport(started_at, finished_at, branch: str, steps: list[StepOutcome], state_commit: str | None, predictions_commit: str | None, pushed: bool, site_changed: bool, raw_archived: bool, unexpected: bool)`
  - `Fetchers(scoreboard, summary, injury_latest, odds_download, gh_runner, clock)` dataclass, with defaults for the real functions
  - `run_once(state_repo: Path, code_dir: Path, *, branch: str, fetchers: Fetchers = Fetchers(), run_attempt: int = 1) -> RunReport`
  - `summary.markdown(report: RunReport) -> str`
  - `summary.outputs(report) -> dict[str, str]` (`site_changed`, `raw_upload_failed`, both `"true"`/`"false"`)

**Requirements:**
1. Step order and names (exactly this list in every report; a step that is not due has `due=False`):
   `["mac_history", "rebuild", "espn_scoreboard", "espn_boxscores", "injury_report", "odds", "state_commit", "predict", "grade", "raw_archive", "push"]`
   - **mac_history:** Task 12. A no-op until then.
   - **rebuild:** `rebuild_db(state_repo/state, $PREDICTOR_DATA_DIR/predictor.duckdb)`.
   - **Source steps:** append state rows, then load them into the DB.
   - **odds:** only when `odds_window_due`. Always append an attempt line, with `ok` and `remaining`.
   - **state_commit:** `commit_and_push(..., kind="state", branch=branch, push=False)` with message `f"state: cloud run {now:%Y-%m-%dT%H:%MZ}"`.
   - **predict:** only when `early_call_due(now)`, with `now` re-read after the fetches, `state_version = head_sha(state_repo)`, and settings from `state.settings_path`.
   - **grade:** always. `_grade_seasons` logic is moved and shared, and `append_id_links` is included.
   - Prediction commit message: `predictions: <slate> slate` if lines were written, else `predictions: grades <slate>`, with `push=False`.
   - **raw_archive:** `assert_no_secret`, `bundle`, `upload`.
   - **push:** `push_pending(state_repo, branch=branch)`.
2. Expected source failures (`requests.RequestException`, `TransientFetchError`, `EspnFormatError`, `OddsQuotaExceeded`, `OddsFetchError`, a missing key) give `ok=False` with a plain detail, and the run continues.
   - Any other exception in a step also gives `ok=False` with its type name, sets `unexpected=True`, and the run continues.
   - A `SecretLeak` stops the run before any commit, and sets `unexpected=True`.
3. `site_changed` is True only when a `predictions:` commit was pushed and `branch == "main"`.
4. Summary heading: `## Cloud run 2026-10-22 13:07 UTC (18:37 IST) — branch cloud-shadow`. Then a table `| step | due | result | detail |`, then a line with the commit shas and push result.
5. CLI `cloud-run --state-repo PATH`:
   - `branch = config.publish_branch()`; the code dir is the project root.
   - Appends the summary to `$GITHUB_STEP_SUMMARY` and the outputs to `$GITHUB_OUTPUT` when those are set. Prints the summary to stdout.
   - Exits 1 when `unexpected` or a publish result has `error`.

**Tests** (bare remote plus clone on `cloud-shadow`, fixture schedule state, fake fetchers, a fixed clock):
```python
def test_first_run_after_1800_ist_commits_state_then_predictions(tmp_path): ...
    # clock 2026-10-22T12:37Z -> subjects == ["predictions: 2026-10-22 slate", "state: cloud run 2026-10-22T12:37Z"] (newest first);
    # each commit touches only its dir; origin/cloud-shadow has both; origin/main sha unchanged
def test_run_before_1800_ist_writes_no_prediction_lines(tmp_path): ...  # 12:22Z -> predict due False
def test_second_identical_run_commits_nothing(tmp_path): ...
def test_rerun_after_a_lost_push_redoes_the_work_without_duplicates(tmp_path): ...
    # run 1 push fails (remote unreachable), fresh clone, run 2 -> one predicted line per game
def test_unreachable_espn_is_recorded_and_the_run_continues(tmp_path): ...
    # scoreboard raises ConnectionError -> espn_scoreboard ok False; injury_report ran; exit 0
def test_unexpected_error_fails_the_run_after_other_steps(tmp_path): ...  # exit 1, push still attempted
def test_summary_lists_every_step(tmp_path): ...   # names == the list above
def test_odds_key_never_written_anywhere(tmp_path, monkeypatch, capsys): ...
    # ODDS_API_KEY="0123456789abcdef0123456789abcdef"; key absent from state files, summary,
    # outputs, stdout/stderr, raw tarball members, and every commit's content
def test_site_changed_only_for_main_pushes(tmp_path): ...
```
- [ ] Write the tests and confirm they fail. Implement. Run `uv run pytest tests/test_cloud_run.py -q` (PASS), then the full suite. Commit: `feat: cloud-run does whatever is due, commits state and predictions separately, and summarises the run`

---

### Task 10: `live.yml`

**Files:**
- Create: `.github/workflows/live.yml`
- Modify: `pyproject.toml` (dev extra gains `pyyaml>=6.0`)
- Test: `tests/test_workflows.py` (load with `yaml.load(text, Loader=yaml.BaseLoader)`, so `on` stays a string key and every value is a string)

**Essentials** (exact):
```yaml
name: live
on:
  schedule:
    - cron: '7,22,37,52 * * * *'   # off the top of the hour, where GitHub delays are worst
  workflow_dispatch:
permissions:
  contents: write
  actions: write                   # to dispatch site.yml (GITHUB_TOKEN pushes do not trigger it)
concurrency:
  group: live
  cancel-in-progress: false
jobs:
  run:
    runs-on: ubuntu-latest
    timeout-minutes: 12
    env:
      PREDICTOR_PUBLISH_BRANCH: ${{ vars.PREDICTOR_PUBLISH_BRANCH || 'cloud-shadow' }}
      PREDICTOR_DATA_DIR: ${{ runner.temp }}/predictor-data
      GH_TOKEN: ${{ github.token }}
    steps:
      - uses: actions/checkout@v4
        with: { ref: main, fetch-depth: 0 }
      - uses: astral-sh/setup-uv@v6
        with: { python-version: '3.14' }
      - run: uv sync --frozen
      - name: Git identity
        run: |
          git config --global user.name "nba-predictor-bot"
          git config --global user.email "41898282+github-actions[bot]@users.noreply.github.com"
      - name: State checkout
        run: |
          if [ "$PREDICTOR_PUBLISH_BRANCH" = main ]; then
            echo "STATE_REPO=$GITHUB_WORKSPACE" >> "$GITHUB_ENV"
          else
            git fetch origin "$PREDICTOR_PUBLISH_BRANCH" || { echo "::error::branch $PREDICTOR_PUBLISH_BRANCH missing - create it by hand"; exit 1; }
            git worktree add -B "$PREDICTOR_PUBLISH_BRANCH" "$RUNNER_TEMP/state-repo" "origin/$PREDICTOR_PUBLISH_BRANCH"
            echo "STATE_REPO=$RUNNER_TEMP/state-repo" >> "$GITHUB_ENV"
          fi
      - id: cloud
        name: Cloud run
        run: uv run predictor cloud-run --state-repo "$STATE_REPO"
        env:
          ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}
      - name: Rebuild site
        if: always() && steps.cloud.outputs.site_changed == 'true'
        run: gh workflow run site.yml --ref main
      - name: Keep raw responses if the release upload failed
        if: always() && steps.cloud.outputs.raw_upload_failed == 'true'
        uses: actions/upload-artifact@v4
        with: { name: 'raw-${{ github.run_id }}-${{ github.run_attempt }}', path: '${{ runner.temp }}/predictor-data/raw', retention-days: 90 }
```
The state worktree lives in `$RUNNER_TEMP` so it never makes the code checkout look dirty to `code_version`.

**Tests:**
```python
def test_live_runs_every_15_minutes_and_by_hand(): ...    # cron == "7,22,37,52 * * * *"; "workflow_dispatch" in on
def test_live_never_overlaps(): ...                        # concurrency == {"group": "live", "cancel-in-progress": "false"}
def test_live_permissions(): ...                           # {"contents": "write", "actions": "write"}
def test_default_branch_is_shadow(): ...                   # env PREDICTOR_PUBLISH_BRANCH value as above
def test_odds_key_only_in_the_cloud_run_step_env(): ...
    # exactly one step has env ODDS_API_KEY == "${{ secrets.ODDS_API_KEY }}"; no step's `run` mentions ODDS_API_KEY
def test_site_dispatched_only_when_predictions_changed(): ...
def test_site_yml_still_has_workflow_dispatch(): ...
```
- [ ] Write the tests and confirm they fail. Add the workflow. Run `uv sync --extra dev` and `uv run pytest tests/test_workflows.py -q` (PASS). Commit: `feat: live workflow every 15 minutes, shadow branch by default`

---

### Task 11: Status on cloud state, with cloud freshness

**Files:**
- Modify: `src/predictor/status.py`, `src/predictor/cli.py` (`status --from-state PATH`)
- Test: `tests/test_status.py`, `tests/test_status_cli.py`

**Interfaces:**
- Produces:
  - `CLOUD_STALE = timedelta(hours=2)`
  - `check_cloud(now: datetime, runner=subprocess.run) -> SourceHealth` (name `"cloud_runs"`)
  - `check_player_links(state_dir: Path, con, now) -> SourceHealth` (name `"player_links"`)
  - `check_live(con, repo_dir, now=None, *, branch: str = "main", cloud: bool = False)`

**Requirements:**
1. `check_cloud` runs `gh run list --workflow live.yml --limit 5 --json createdAt,status,conclusion` (20 s).
   - Stale when there is no run, when `now − newest createdAt > 2h`, or when the 3 newest completed runs all have conclusion `failure`.
   - When `gh` is missing or fails, it is stale with advice `"check gh auth status"`.
   - The detail reads `"last cloud run 37m ago (success)"`.
   - Both the Mac `status` and `--from-state` include it.
2. `status --from-state PATH`:
   - Rebuilds a temporary DB from `PATH/state`.
   - Runs `check_sources` without `news_items` (detail: "news is archived by the Mac (research only)").
   - Runs `check_live(..., repo_dir=PATH, branch=config.publish_branch(), cloud=True)`. The advice names the live.yml run summary and `gh secret set ODDS_API_KEY` instead of launchd jobs and key files.
   - Adds `check_cloud` and `check_player_links`. The latter is stale when any unmatched ESPN player played more than 0 minutes in the last 7 days. Its detail lists up to 5 names and says "add them to state/players/aliases.json".
   - The exit code follows the existing rule.

**Tests:**
```python
def test_cloud_stale_after_two_hours(): ...   # newest 2026-10-22T10:00Z: now 12:00Z -> not stale; 12:00:01Z -> stale
def test_cloud_three_failures_is_stale(): ...
def test_cloud_gh_missing_is_reported(): ...   # runner raises FileNotFoundError -> stale, "gh auth status" in advice
def test_status_from_state_skips_news(tmp_path): ...
def test_player_links_unmatched_warns(tmp_path): ...
```
- [ ] Write the tests and confirm they fail. Implement. PASS. Commit: `feat: status checks cloud runs and works on cloud state`

---

### Task 12: Mac weekly history refresh and its sync in the cloud

**Files:**
- Create: `src/predictor/cloud/macsync.py`, `scripts/com.predictor.history.plist`, `scripts/uninstall_live_jobs.sh`
- Modify: `src/predictor/cli.py` (`refresh-history`), `src/predictor/cloud/run.py` (the `mac_history` step), `scripts/install_schedule.sh`
- Test: `tests/test_macsync.py`, `tests/test_launchd.py`

**Interfaces:**
- Produces:
  - `macsync.MAC_BRANCH = "mac-history"`
  - `macsync.MacSync(copied: list[str], skipped_reason: str | None)`
  - `macsync.sync_mac_history(state_repo: Path, runner=subprocess.run) -> MacSync`
  - `macsync.prepare_worktree(repo_dir: Path, path: Path) -> Path`
  - `macsync.fetch_current_players(season: str) -> list[dict]`, via `CommonAllPlayers(is_only_current_season=1)`, archived raw-first under source `players`

**Requirements:**
1. `sync_mac_history` (cloud, before `rebuild`):
   - `git fetch origin mac-history`. A missing branch is a quiet no-op.
   - Every commit in `HEAD..origin/mac-history` must have a subject starting with `state: ` and touch only `state/nba/`. Otherwise copy nothing and set `skipped_reason`.
   - Copy every `state/nba/*` file whose content differs (`git show origin/mac-history:<path>`) into the working tree. They go into this run's `state:` commit.
2. CLI `refresh-history [--all-seasons] [--no-push]` (Mac):
   - `prepare_worktree` at `settings.data_dir / "mac-history"` (`git worktree add` once, then `pull --ff-only origin mac-history`).
   - Runs the existing schedule ingest and `_capture_results(season)` into the Mac DB, without grading or publishing.
   - Fetches the players.
   - `export_from_db` plus `write_snapshot("nba/players-<season>.jsonl")` into the worktree.
   - `commit_and_push(worktree, paths, f"state: nba history refresh {today}", kind="state", branch="mac-history")`.
3. `com.predictor.history.plist`: Weekday 1, 11:15, `refresh-history`, `RunAtLoad` false, logs `data/logs/history.*.log`.
   - `install_schedule.sh` installs it as well.
   - `install_schedule.sh --after-cutover` installs only `com.predictor.daily` (news) and `com.predictor.history`.
   - `uninstall_live_jobs.sh` unloads and removes exactly `com.predictor.predict`, `com.predictor.results`, `com.predictor.odds` and `com.predictor.schedule`.

**Tests:**
```python
def test_mac_history_copied_when_every_commit_is_safe(tmp_path): ...  # copied == ["state/nba/games-2026-27.jsonl"]
def test_mac_history_with_a_commit_outside_state_nba_is_skipped(tmp_path): ...  # copied == [], reason names the subject
def test_missing_mac_history_branch_is_a_quiet_noop(tmp_path): ...
def test_refresh_history_commits_to_mac_history_only(tmp_path, monkeypatch): ...  # subject "state: nba history refresh 2026-10-26"
# tests/test_launchd.py
def test_history_job_runs_weekly_on_monday_1115(): ...
def test_install_script_after_cutover_installs_only_news_and_history(): ...
def test_uninstall_script_names_exactly_the_four_live_jobs(): ...
# existing test_no_two_jobs_share_a_start_minute keeps passing
```
- [ ] Write the tests and confirm they fail. Implement. PASS. Commit: `feat: weekly Mac history refresh to mac-history, copied in by the cloud run`

---

### Task 13: `compare-shadow`

**Files:**
- Create: `src/predictor/cloud/compare.py`
- Modify: `src/predictor/cli.py`
- Test: `tests/test_compare_shadow.py`

**Interfaces:**
- Produces:
  - `MIN_SHADOW_DAYS = 3`, `MAX_P_DIFF = 0.01`
  - `GamePair(game_id, game_date, mac_p: float | None, cloud_p: float | None, mac_at, cloud_at, cloud_before_tip: bool)`
  - `ShadowReport(slate_days: list[date], pairs: list[GamePair], mac_only: list[str], cloud_only: list[str], max_abs_dp: float, ok: bool, reasons: list[str])`
  - `compare_logs(mac_lines: list[dict], cloud_lines: list[dict], since: date, id_links: dict[str, str]) -> ShadowReport`

**Requirements:**
1. Only `predicted` lines with `game_date >= since` count, one per game: the latest sealed before tip. Provisional cloud IDs are matched through `id_links`.
2. `ok` requires all of the following. Otherwise `reasons` names each failure:
   - at least 3 slate days where both sides predicted
   - every Mac game has a cloud call sealed before tip
   - `max_abs_dp <= 0.01`
3. CLI `compare-shadow [--since YYYY-MM-DD]`:
   - Runs `git fetch origin main cloud-shadow`.
   - Reads `git show origin/<b>:predictions/<season>.jsonl`.
   - The default `since` is the ET date of the merge-base commit's time.
   - Prints one row per game plus the verdict. Exit 0 on PASS, 1 on FAIL.

**Tests:**
```python
def test_three_identical_days_pass(): ...
def test_difference_over_one_point_fails(): ...  # dp 0.011 -> "largest difference 0.011 > 0.01" in reasons
def test_two_days_is_not_enough(): ...
def test_mac_only_game_listed_and_fails(): ...
def test_provisional_cloud_id_matched_via_links(): ...
```
- [ ] Write the tests and confirm they fail. Implement. PASS. Commit: `feat: compare-shadow checks cloud calls against the Mac's`

---

### Task 14: Shadow mode and cut-over (controller)

- [ ] Run the full suite green, run the whole-branch review, and fix what it finds. Merge `cloud-operation` to `main` and push by hand.
- [ ] Ask the owner to run `gh secret set ODDS_API_KEY` themselves; the key is never typed into chat. Verify with `gh secret list`, which shows the name only.
- [ ] Make the first pushes by hand: `git push origin main:cloud-shadow` and `git push origin main:mac-history`.
- [ ] Seed from the live copy: `predictor refresh-history --all-seasons`. Check that `mac-history` has `state/nba/games-<season>.jsonl` for 2014-15..2026-27.
- [ ] Run `gh workflow run live.yml`. In the run summary, check:
  - every step is listed
  - the `cloud-shadow` branch received one `state:` commit that holds `state/nba/` plus ESPN state
  - the `rebuild` counts include the 33,112 games rows exported (plus new ones)
  - a `raw-2026-W43` release asset exists
- [ ] Shadow for at least 3 slate days while the Mac keeps publishing. Run `predictor compare-shadow` daily, and `predictor status --from-state <cloud-shadow worktree>`. Check odds credits (about 6 a day) and the unmatched player list; add aliases if needed.
- [ ] Record the shadow results in the spec, under a new "Result" section in `2026-10-22-cloud-injuries-design.md`.
- [ ] **Cut-over, only after the owner's explicit OK given at that moment:**
  1. `bash scripts/uninstall_live_jobs.sh` (predict, results, odds, schedule).
  2. In the live copy: `git checkout origin/cloud-shadow -- state/`, then commit `state: adopt cloud state from cloud-shadow` and push by hand.
  3. `gh variable set PREDICTOR_PUBLISH_BRANCH --body main`, then `gh workflow run live.yml`. Check that a `state:` commit reaches `main`.
  4. After the next 18:00 IST run, check that a `predictions:` commit is on `main`, that site.yml was dispatched and deployed, and that the site shows the slate.
  5. `bash scripts/install_schedule.sh --after-cutover` (news and history only).
- [ ] Rollback, if needed: `gh variable set PREDICTOR_PUBLISH_BRANCH --body cloud-shadow`, then `bash scripts/install_schedule.sh`.
- [ ] Update memory.
