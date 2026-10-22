# Player Model Implementation Plan (plan 1 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a candidate Stage 1 model that knows who is playing (player ratings, expected minutes, injury-report availability), replay it walk-forward over 2020-21..2025-26 at the final-call and early-call times, and report it against the switch-over bar. The live model is not switched.

**Architecture:**
- Two new data feeds, both run on the Mac:
  - Player game logs come from `LeagueGameLog(player_or_team_abbreviation='P')` and go into a new point-in-time table, `player_games`.
  - An injury-report backfill fetches only the report slots the replay needs.
- New pure modules turn those rows into two additive Stage 1 terms, **roster** and **injuries**:
  - `player_ids` links injury-report names to player ids.
  - `player_ratings` holds the ratings.
  - `minutes` holds expected minutes and redistribution.
  - `availability` maps a status to a playing probability.
- `player_sim` runs one chronological, point-in-time pass that produces per-game features for each call time.
- `candidate` picks settings walk-forward on top of the current model's team-rating settings, then scores the switch-over bar. `evaluate-model` prints the result.
- An optional gradient-boosted correction, plus an AsOfView predictor with a settings file, are added only for plan 3 to use. The settings file is written only when the bar passes.

**Tech Stack:** Python 3.14, uv, DuckDB, nba_api, pdfplumber, requests, numpy, typer. scikit-learn is added in Task 8 only.

**Spec:** `docs/superpowers/specs/2026-10-22-cloud-injuries-design.md` (sections "The richer model", "Switch-over bar", "Calls made after the builder check" 1 and 6). It extends `docs/superpowers/specs/2026-09-22-nba-predictor-design.md` (Stage 1, walk-forward calibration).

## Global Constraints

- **Scope.** This is plan 1 of 3. No GitHub Actions, ESPN sources, state directory, `call_type`/revisions, or site changes. `predict-today`, `live.py`, `stage1.py`, `fit.py`, `tuning.py`, `settings.py` and `stage1_settings.json` are not modified, and the current model stays live.
- **No leakage** (spec anti-goal):
  - A call replayed at time T reads only `player_games` rows with `observed_at <= T` and only the latest injury report with `observed_at <= T`.
  - Availability probabilities for season S use only seasons strictly before S.
  - Settings for season S are chosen only on tuning seasons strictly before S.
  - Season-total endpoints (`PlayerEstimatedMetrics`, `LeagueDashPlayerStats`, `PlayerCareerStats`) must never appear in `src/`. A source-scan test enforces this.
- **Call times.** All are UTC datetimes.
  - `final_call_at = tip_off − 45 min`.
  - `early_call_at = datetime.combine(game_date, time(12, 30), UTC)` (the live 18:00 IST rule), clipped to `<= final_call_at`.
  - Tip-off is `backtest.tipoff.resolve_tipoff(tipoff_index(con), ...)`, the minimum across schedule vintages.
- **Injury report URLs.**
  - Before 2025-12-22: `Injury-Report_YYYY-MM-DD_HHAM|PM.pdf` (hourly).
  - From 2025-12-22: `Injury-Report_YYYY-MM-DD_HH_MM{AM|PM}.pdf`, with MM in {00, 15, 30, 45}.
  - The report time is always the parsed header time (`ParsedReport.published_at`), never the filename.
  - 403/404 means "not published". Every request sends a browser-like `User-Agent`.
- **Raw-first.** Player-log DataFrames and injury PDFs are archived to `raw_store` before parsing, and parsing reads the archived bytes.
- **Physical `_raw` table names** appear only in `db.py`/`asof.py`. Other code uses `db.POINT_IN_TIME_TABLES[...]`.
- **The model never reads odds.** The odds table is read only by the market-gap report section.
- **Test seasons are never fitted.** Any 2026-27 row reaching a candidate fit raises `fit.FitError`.
- **stats.nba.com is reachable only from the Mac.** No test touches the network: fetch functions and sessions are injected.
- Plain-English CLI errors with non-zero exit codes and no tracebacks for expected failures. Commands over about 2 minutes run with `run_in_background`.
- Baseline: `uv run pytest -q` collects **906 tests, 1 deselected** on `main` (HEAD c5cef70). Confirm it is green before Task 1. Branch: `player-model`.


## Controller rulings (2026-10-22, before execution)

- **Preseason rosters:** Task 1 also ingests preseason player logs (season type "Pre Season"), tagged `season_type='preseason'`. They are used **only** for roster membership in Task 6, so summer signings and trades count from their first preseason game. They never feed player ratings or minutes. Cost if wrong: opening-week rosters stay stale, which is the planner's assumption 3.
- **scikit-learn:** verified on 2026-10-22: scikit-learn 1.9.1 installs and imports on Python 3.14, so Task 8 proceeds.
- **Team-rating settings reused per fold** (planner's assumption 2) is accepted. The report must state that the bar comparison is conservative for that reason.

## Review Focus

1. **A traded player listed by his new team before his first game for them.** He must leave the old team's rotation at once, and he counts for the new team only once he has a minutes history. Test in Task 6.
2. **An injury-report name that links to no player id.** It is treated as available, counted per season, and listed in the report, never dropped silently. Test in Task 3.
3. **Two players whose names normalise to the same key** (e.g. a father/son pair or a Jr./Sr. pair). Team plus season decides; if the same team still has two, the name is unlinked, never guessed. Test in Task 3.
4. **A needed report slot that was never published, or a report that does not list that game's date.** The replay uses the latest earlier report that lists the date, and counts it as a fallback. With no report at all the injury term is 0 and the game is counted as "no report". Test in Task 6.
5. **A status string outside the five known ones.** It uses the Questionable prior (0.5) and is counted in the report. It never raises and is never treated as available. Test in Task 5.

---

### Task 1: Player game logs, stored point-in-time

**Files:**
- Modify: `src/predictor/db.py` (`POINT_IN_TIME_TABLES`, `_SCHEMA`), `src/predictor/asof.py` (`_DEFAULT_LATEST_KEY`), `src/predictor/status.py` (`STALENESS_HOURS`, `_advice`), `src/predictor/cli.py` (`ingest-player-logs`)
- Create: `src/predictor/sources/player_games.py`
- Test: `tests/test_player_games.py`, `tests/test_db.py`, `tests/test_status.py`

**Interfaces:**
- Produces:
  - `db.POINT_IN_TIME_TABLES["player_games"] == "player_games_raw"`, with schema:
    ```sql
    CREATE TABLE IF NOT EXISTS player_games_raw (
        game_id VARCHAR NOT NULL, season VARCHAR NOT NULL, game_date DATE NOT NULL,
        player_id BIGINT NOT NULL, player_name VARCHAR NOT NULL, team VARCHAR NOT NULL,
        minutes DOUBLE, pts INTEGER, fgm INTEGER, fga INTEGER, fg3m INTEGER, fg3a INTEGER,
        ftm INTEGER, fta INTEGER, oreb INTEGER, dreb INTEGER, ast INTEGER, stl INTEGER,
        blk INTEGER, tov INTEGER, pf INTEGER, plus_minus DOUBLE,
        source VARCHAR NOT NULL, reconstructed BOOLEAN NOT NULL DEFAULT FALSE,
        observed_at TIMESTAMP WITH TIME ZONE NOT NULL,
        PRIMARY KEY (game_id, player_id, observed_at))
    ```
    The `source` column is `'nba_stats'` here. Plan 2 adds `'espn'`, per spec decision 6.
  - `asof._DEFAULT_LATEST_KEY["player_games"] = ("game_id", "player_id")`
  - `player_games.SOURCE = "player_games"`; `SEASON_TYPES = ("Regular Season", "PlayIn", "Playoffs")`
  - `fetch_season_type(season: str, season_type: str) -> pd.DataFrame`: nba_api `leaguegamelog.LeagueGameLog(season=..., season_type_all_star=..., player_or_team_abbreviation="P", league_id="00", timeout=60)`, with tenacity retry like `nba_stats.fetch_season`.
  - `archive_key(season, season_type, fetched_at) -> str`: `f"{season}_{season_type.lower().replace(' ', '-')}_{fetched_at:%Y%m%dT%H%M%S}Z.json.gz"`
  - `PlayerGameRow` (frozen): `game_id: str, season: str, game_date: date, player_id: int, player_name: str, team: str`, the 15 box columns above, and `plus_minus: float | None`.
  - `Downloaded(season, season_type, fetched_at, blob_key, rows: list[PlayerGameRow])` (frozen)
  - `download(season, season_type, fetched_at=None, fetch=fetch_season_type) -> Downloaded`: archives gzipped `orient="split"` JSON (`mtime=0`), then parses it back with GAME_ID read as a string, the same way `results.download` does.
  - `LoadResult(season, season_type, rows_written: int, games: int, unknown_game_ids: list[str])` (frozen)
  - `load(con, downloaded) -> LoadResult`:
    - INSERT OR REPLACE in one transaction.
    - `observed_at = nba_stats._derive_observed_at(game_date, game_id)[1]` (game date + 1 day, 12:00 UTC), with `reconstructed=TRUE` and `source='nba_stats'`.
    - `unknown_game_ids` lists the game ids with no row in the games table. They are still stored.
- CLI `predictor ingest-player-logs [--season S]...`:
  - The default is every season from `2014-15` through `config.season_label(now)`.
  - Every download (season × type) runs first, then `connect_with_retry`, `migrate` and `load`.
  - A download failure prints `Could not download player game logs for {season} {season_type} ({exc}); nothing was saved.` and exits 1.
  - Per season it prints `player logs {season}: {rows} rows, {games} games ({n} not in the games table)`.

- [ ] **Step 1: Write the failing tests**
  - `test_player_games_is_a_point_in_time_table`: `"player_games" in db.POINT_IN_TIME_TABLES`; `AsOfView(con, t).latest("player_games")` works on an empty table. The existing `test_default_latest_key_covers_every_point_in_time_table` must pass.
  - `test_migrate_creates_player_games_idempotently`: `migrate` twice gives no error, and the columns include `plus_minus`, `source` and `reconstructed`.
  - `test_download_archives_before_parsing`: monkeypatch `raw_store.load` to record calls. Then `downloaded.blob_key == "2023-24_regular-season_20261022T120000Z.json.gz"`, and `load` was called with that key.
  - `test_parse_keeps_leading_zero_game_id`: a frame row with `GAME_ID "0022300001"` gives `row.game_id == "0022300001"`.
  - `test_load_stamps_day_after_noon_utc`: a row for game date 2024-01-15 gives `observed_at == datetime(2024, 1, 16, 12, tzinfo=UTC)` and `reconstructed is True`.
  - `test_load_is_idempotent`: loading twice gives `count(*)` unchanged.
  - `test_unknown_game_id_is_stored_and_reported`: `result.unknown_game_ids == ["0062300001"]`, and the row exists.
  - `test_view_hides_logs_before_observed_at`: `AsOfView` at `2024-01-16 11:59:59Z` sees 0 rows, and at `12:00:00Z` sees 1.
  - `test_cli_download_failure_never_opens_db`: monkeypatch `db.connect_with_retry` to record calls. Exit code is 1, the message starts with "Could not download player game logs", and there are no connect calls.
  - `test_status_lists_player_games_with_ingest_advice`: when stale, the advice is `"Run: predictor ingest-player-logs --season {season_label(now)}"`. Staleness is `24 * 150` hours.
  - `test_no_season_total_endpoints_in_src`: no file under `src/` contains `PlayerEstimatedMetrics`, `LeagueDashPlayerStats`, `playerestimatedmetrics` or `PlayerCareerStats`.

- [ ] **Step 2: Run them and confirm they fail.** Run `uv run pytest tests/test_player_games.py -v`. Expected: FAIL (`player_games` module missing).
- [ ] **Step 3: Implement** the interfaces above. Frame columns: `SEASON_ID, PLAYER_ID, PLAYER_NAME, TEAM_ABBREVIATION, GAME_ID, GAME_DATE, MATCHUP, MIN, FGM, FGA, FG3M, FG3A, FTM, FTA, OREB, DREB, AST, STL, BLK, TOV, PF, PTS, PLUS_MINUS`. Missing numbers become None.
- [ ] **Step 4: Run the tests and the full suite.** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** with the message `feat: player game logs stored point-in-time (ingest-player-logs)`.

---

### Task 2: Injury-report slot backfill for the replay

**Files:**
- Modify: `src/predictor/sources/injury_report.py` (browser `User-Agent` session only), `src/predictor/cli.py` (`backfill-injury-slots`)
- Create: `src/predictor/sources/injury_slots.py`
- Test: `tests/test_injury_slots.py`, `tests/test_injury_fetch.py`

**Interfaces:**
- Consumes: `injury_report._archive(day, hour_label, session=...) -> ArchiveOutcome`, `injury_report.raw_key`, `injury_report.parse_report`, `injury_report.ingest_report`, `injury_report.TransientFetchError`, `tipoff.tipoff_index`/`resolve_tipoff`.
- Produces:
  - `injury_report.BROWSER_HEADERS: dict[str, str]`, a desktop Chrome `User-Agent` plus `Accept: application/pdf,*/*`.
  - `injury_report.new_session() -> requests.Session`, with those headers. `fetch_report` and `backfill_range` use it when `session is None`.
  - `injury_slots.NEW_PATTERN_FROM = date(2025, 12, 22)`, `MAX_WALKBACK = timedelta(hours=3)`, `EARLY_CALL_UTC = time(12, 30)`, `FINAL_CALL_LEAD = timedelta(minutes=45)`
  - `call_times(game_date: date, tip_off: datetime) -> tuple[datetime, datetime]`: returns `(early, final)`.
  - `slot_label(slot_et: datetime) -> str`: hourly `"07PM"` before `NEW_PATTERN_FROM`, otherwise `"07_15PM"`.
  - `candidate_slots(target: datetime) -> list[tuple[date, str]]`:
    - Slots whose label time is `<= target` (in ET), newest first, back to and including `target − MAX_WALKBACK`.
    - The step is 60 min (old) or 15 min (new), and the ET date is part of each slot.
  - `SlotTarget(game_id: str, kind: Literal["early", "final"], at: datetime)` (frozen)
  - `replay_targets(con, start: date, end: date) -> list[SlotTarget]`: competitive games with a tip-off, deduplicated by `at` across games.
  - `backfill_slots(con, targets, delay=0.4, session=None) -> dict[str, int]`:
    - For each distinct `at`, walk `candidate_slots(at)`.
    - Archive and ingest each fetched PDF (every report is useful history).
    - Stop at the first archived report whose header `published_at <= at`. A report published after `at` is kept but does not satisfy the target.
    - Counts: `targets`, `satisfied`, `unsatisfied` (no report within the walk-back; the replay will fall back), `fetched`, `not_published`, `transient`, `bad_content`, `parse_failed`, `already_archived`.
- CLI `predictor backfill-injury-slots [--start 2020-12-22] [--end today] [--delay 0.4]`:
  - Prints each count.
  - Exits 1 only when `transient + bad_content + parse_failed > 0`, with the same plain warnings as `backfill-injuries`. `unsatisfied` alone is informational: holes are expected.

- [ ] **Step 1: Write the failing tests**
  - `test_default_session_sends_browser_user_agent`: `injury_report.new_session().headers["User-Agent"].startswith("Mozilla/5.0")`, and `fetch_report` with no session uses `new_session` (monkeypatched).
  - `test_call_times_final_is_tip_minus_45`: a tip at `2024-01-16 00:00Z` gives final `2024-01-15 23:15Z` and early `2024-01-15 12:30Z`.
  - `test_early_call_clipped_to_final`: a tip at `2024-01-15 13:00Z` gives `early == final == 12:15Z`.
  - `test_slot_label_switches_pattern_on_2025_12_22`: ET 2025-12-21 19:00 gives `"07PM"`, and ET 2025-12-22 19:15 gives `"07_15PM"`. Also `injury_report.report_url(date(2025,12,22), "07_15PM").endswith("Injury-Report_2025-12-22_07_15PM.pdf")`.
  - `test_candidate_slots_old_pattern_walks_back_hourly`: target ET 2024-01-15 18:15 gives `[(date(2024,1,15), "06PM"), (date(2024,1,15), "05PM"), (date(2024,1,15), "04PM")]` (15:00 is before the 15:15 bound).
  - `test_candidate_slots_new_pattern_walks_back_quarter_hours`: target ET 2026-01-10 18:20 gives a first slot of `"06_15PM"`, 12 slots in total, and a last slot of `"03_30PM"`.
  - `test_candidate_slots_cross_midnight_et`: target ET 2024-01-16 00:40 gives a first slot of `(2024-01-16, "12AM")` and a second of `(2024-01-15, "11PM")`.
  - `test_backfill_uses_header_time_not_filename`: the fixture `Injury-Report_2025-01-15_05PM.pdf` has header 2025-01-15 22:30Z. A target at 22:15Z is not satisfied by it (it walks on), and a target at 22:45Z is satisfied.
  - `test_backfill_tolerates_holes`: one target at ET 2024-01-15 18:00 and a session returning 403 for every slot give `not_published == 4` and `unsatisfied == 1`, with no exception.
  - `test_backfill_transient_counted_and_retried_on_rerun`: a 503 gives `transient == 1`, and a second run fetches the slot again.
  - `test_replay_targets_dedupe_shared_times`: two games tipping at the same time give one final target.
  - `test_cli_exit_codes`: unsatisfied only gives exit 0; transient gives exit 1.

- [ ] **Step 2: Run them and confirm they fail.** Run `uv run pytest tests/test_injury_slots.py tests/test_injury_fetch.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** using `injury_report._archive` and `raw_store.load` → `parse_report` to read `published_at`. Put the slot-label ET-to-label conversion in `injury_slots`; `report_url`/`raw_key` already accept any label.
- [ ] **Step 4: Run the tests and the full suite.** Expected: all pass. The existing `test_report_url_matches_verified_pattern` is unchanged.
- [ ] **Step 5: Commit** with the message `feat: backfill the injury-report slots the replay needs (both filename patterns)`.

---

### Task 3: Linking injury-report names to player ids

**Files:**
- Create: `src/predictor/model/player_ids.py`
- Test: `tests/test_player_ids.py`

**Interfaces:**
- Produces:
  - `name_key(name: str) -> str`:
    - Accepts "First Last" (game logs) or "Last,First"/"Last, First" (injury `player`/`player_display`). A comma means last-then-first.
    - NFKD-strips accents and lowercases.
    - Drops a trailing suffix token in `{"jr", "sr", "ii", "iii", "iv"}` from the last-name part.
    - Keeps letters only, as first + last.
  - `PLAYER_ALIASES: dict[str, int]`, mapping a name key to a player_id for known spelling differences. It starts empty and is filled in Task 11 from the unlinked report.
  - `PlayerIndex.from_rows(rows: Iterable[tuple[int, str, str, str]]) -> PlayerIndex`, built from `(player_id, player_name, team, season)`. This is identity only, not results.
  - `PlayerIndex.link(name: str, team: str, season: str) -> int | None`, in this order:
    1. alias;
    2. a unique key match among players with a row for (team, season);
    3. a unique key match among players with any row in that season;
    4. a unique key match across all seasons;
    5. otherwise None.
  - `LinkReport(linked: int, unlinked: Counter[tuple[str, str]])`, keyed by (name, team), and `PlayerIndex.report() -> LinkReport`. Every `link` call is counted.

- [ ] **Step 1: Write the failing tests**
  - `test_name_key_handles_both_layouts`: `name_key("Curry,Stephen") == name_key("Curry, Stephen") == name_key("Stephen Curry") == "stephencurry"`.
  - `test_name_key_strips_suffix_and_accents`: `name_key("Jackson Jr.,Jaren") == name_key("Jaren Jackson Jr.") == "jarenjackson"`, and `name_key("Dončić,Luka") == "lukadoncic"`.
  - `test_link_prefers_team_season_match`: two players keyed `"marcusmorris"` (ids 1 and 2, on LAC and NYK) give `link("Morris, Marcus", "NYK", "2022-23") == 2`.
  - `test_duplicate_name_on_same_team_is_unlinked` (Review Focus 3): two ids with the same key on the same team and season give `None`, and `report().unlinked[("Morris, Marcus", "NYK")] == 1`.
  - `test_unlinked_injury_name_is_counted_not_dropped` (Review Focus 2): an unknown name gives `None`, `linked == 0`, and the unlinked count is 1.
  - `test_alias_wins`: `PLAYER_ALIASES` monkeypatched with `{"nene": 7}` gives `link("Nene", "HOU", "2019-20") == 7`.

- [ ] **Step 2: Run them and confirm they fail.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Commit** with the message `feat: link injury-report names to NBA player ids`.

---

### Task 4: Player ratings, expected minutes and redistribution (pure)

**Files:**
- Create: `src/predictor/model/player_ratings.py`, `src/predictor/model/minutes.py`
- Test: `tests/test_player_ratings.py`, `tests/test_minutes.py`

**Interfaces:**
- Produces, in `player_ratings`:
  - `PlayerParams(half_life_minutes: float, season_regression: float)` (frozen)
  - `HALF_LIFE_GRID = (500.0, 1500.0, 4000.0)`, `REGRESSION_GRID = (0.2, 0.4)`, `ROOKIE_PRIOR = -2.0`, `OBS_CLIP = 60.0`
  - `game_score(row: PlayerGameRow) -> float`, Hollinger:
    `PTS + 0.4·FGM − 0.7·FGA − 0.4·(FTA − FTM) + 0.7·OREB + 0.3·DREB + STL + 0.7·AST + 0.7·BLK − 0.4·PF − TOV`
  - `ewma_update(r: float, obs: float, minutes: float, half_life: float) -> float`: `λ = 0.5 ** (minutes / half_life)`, returning `λ·r + (1−λ)·clip(obs, ±OBS_CLIP)`.
  - `PlayerBook(params)` holds two ratings per player id: `"box"` and `"pm"`. It is keyed by player and never by team, so trades carry ratings over.
    - `.enter_season(season)` multiplies every rating by `(1 − season_regression)` the first time a new season is seen.
    - `.apply_game(rows: list[PlayerGameRow], margins: dict[str, int])`:
      - `margins` maps team → that team's final margin.
      - Box observation: `48·game_score/min − league_box_mean`, where `league_box_mean` is the cumulative minutes-weighted mean of `48·game_score/min` over every row applied before this game.
      - Plus-minus observation: `48·plus_minus/min − margins[team]`.
      - Rows with `minutes` None or `< 1` are skipped.
      - An unseen player starts at `ROOKIE_PRIOR`.
    - `.rating(player_id, formula: Literal["box", "pm"]) -> float`, which returns `ROOKIE_PRIOR` when the player is unseen.
    - `.recent_minutes(player_id) -> list[float]`: the last 10 games played, any team.
    - `.last_team(player_id) -> str | None`
    - `.team_recent_games(team) -> list[set[int]]`: the player-id sets of the team's last 10 games.
- Produces, in `minutes`:
  - `TEAM_MINUTES = 240.0`, `MINUTES_CAP = 44.0`, `ROTATION_GAMES = 10`
  - `normalize_minutes(minutes: dict[int, float], total=TEAM_MINUTES, cap=MINUTES_CAP) -> dict[int, float]`: scales to `total`, then caps and re-spreads the excess over uncapped players until stable. If every player is capped, all stay at the cap.
  - `strength(minutes: dict[int, float], ratings: dict[int, float]) -> float`: `Σ (m/48)·r`
  - `available_minutes(normal: dict[int, float], play_prob: dict[int, float]) -> dict[int, float]`: `normalize_minutes({p: q_p·m_p})`. A missing `q` means 1.0.

- [ ] **Step 1: Write the failing tests**
  - `test_game_score_hollinger`: PTS 30, FGM 10, FGA 20, FTM 8, FTA 10, OREB 2, DREB 8, AST 5, STL 2, BLK 1, PF 3, TOV 4 gives `game_score == pytest.approx(24.0)`.
  - `test_ewma_update`: `ewma_update(0.0, 10.0, 1500.0, 1500.0) == pytest.approx(5.0)`; `ewma_update(4.0, 10.0, 0.0, 1500.0) == 4.0`; `ewma_update(0.0, 100.0, 1500.0, 1500.0) == pytest.approx(30.0)` (clipped).
  - `test_season_regression`: rating 5.0 with `season_regression=0.4` gives 3.0 after entering the next season, and entering the same season again leaves it unchanged.
  - `test_rating_follows_player_across_trade`: a player rated with team BOS, then a row for NYK, keeps the BOS-era rating as the starting point, and `last_team == "NYK"`.
  - `test_zero_minute_row_ignored`: `minutes=0` leaves the rating and `recent_minutes` unchanged.
  - `test_normalize_minutes_scales_to_240`: `{1:36, 2:36, 3:24, 4:24, 5:24, 6:24, 7:24, 8:12}` gives `round(out[1], 4) == 42.3529` and `sum == pytest.approx(240.0)`.
  - `test_normalize_minutes_caps_and_respreads`: `{1:40, 2..7:20}` gives `out[1] == 44.0` and `round(out[2], 4) == 32.6667`.
  - `test_injury_gap_out`: 8 players with 30 min each, ratings `{1: 6.0, 2: 2.0}` (others 0), player 1 with q=0. Then `strength(normal) == pytest.approx(5.0)` and `round(strength(available) − strength(normal), 6) == -3.571429`.
  - `test_injury_gap_questionable`: the same, with q=0.5 for player 1, gives `round(gap, 6) == -1.666667`.

- [ ] **Step 2: Run them and confirm they fail.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Commit** with the message `feat: point-in-time player ratings, expected minutes and redistribution`.

---

### Task 5: Availability from the injury report

**Files:**
- Create: `src/predictor/model/availability.py`
- Test: `tests/test_availability.py`

**Interfaces:**
- Produces:
  - `STATUS_PRIORS = {"Out": 0.0, "Doubtful": 0.2, "Questionable": 0.5, "Probable": 0.9, "Available": 1.0}`, `UNKNOWN_STATUS_PRIOR = 0.5`, `MIN_SAMPLES = 100`
  - `Listing(season: str, call: Literal["early", "final"], status: str, played: bool)` (frozen)
  - `AvailabilityTable(probs: dict[str, float], samples: dict[str, int])` (frozen), with method `.prob(status: str | None) -> float`:
    - `None` (not listed) returns 1.0.
    - An unknown status returns `UNKNOWN_STATUS_PRIOR`.
  - `estimate(listings: Iterable[Listing], season: str, call: str) -> AvailabilityTable`:
    - Uses only listings with `listing.season < season` and the same `call`.
    - Per status, the played share when there are at least `MIN_SAMPLES` listings; otherwise the prior.
  - `Snapshot(observed_at: datetime, statuses: dict[tuple[str, str], str])`, keyed by (team, injury `player` key).
  - `InjuryHistory.from_db(con) -> InjuryHistory`: reads the `injury_status` table directly (training path, like `fit._load`). Its `.snapshot_at(game_date, at)` returns the latest report with `observed_at <= at` that has a row for `game_date`, or `None`.
  - `InjuryHistory.from_view(view)`: the same, read through `AsOfView.table("injury_status")`, for Task 10.
  - `UnknownStatusCounter`: per-season counts of unknown status strings, which the report shows.

- [ ] **Step 1: Write the failing tests**
  - `test_estimate_uses_only_earlier_seasons`: 120 Questionable listings in 2021-22 (72 played) and 500 in 2022-23 (all played). Estimating for 2022-23 gives `prob("Questionable") == pytest.approx(0.6)`.
  - `test_estimate_falls_back_to_prior_below_min_samples`: 30 Doubtful listings give `prob("Doubtful") == 0.2` and `samples["Doubtful"] == 30`.
  - `test_estimate_separates_call_types`: early listings never affect a final table.
  - `test_not_listed_is_available`: `prob(None) == 1.0`.
  - `test_unknown_status_uses_questionable_prior_and_is_counted` (Review Focus 5): `prob("Not With Team") == 0.5`, and the counter records it once for its season.
  - `test_snapshot_at_never_sees_a_later_report`: reports at 22:30Z and 23:30Z give `snapshot_at(d, 23:29:59Z).observed_at == 22:30Z`, and `snapshot_at(d, 23:30Z).observed_at == 23:30Z`.
  - `test_snapshot_skips_reports_without_that_game_date`: the latest report lists only the next day, so the earlier report that lists `d` is returned.
  - `test_snapshot_filters_rows_to_game_date`: a two-date report (`tests/fixtures/Injury-Report_2025-01-15_05PM.pdf`, ingested) gives a 2025-01-15 snapshot with only rows whose `game_date` is 2025-01-15.

- [ ] **Step 2: Run them and confirm they fail.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Commit** with the message `feat: injury-report availability with probabilities from earlier seasons only`.

---

### Task 6: Point-in-time player feature simulation

**Files:**
- Create: `src/predictor/model/player_sim.py`
- Test: `tests/test_player_sim.py` (fixtures in `tests/player_fixtures.py`, which builds on `tests/model_fixtures.py`)

**Interfaces:**
- Consumes: Tasks 1, 3, 4 and 5; `fit._load(con, venues) -> list[fit._Game]`; `injury_slots.call_times`; `tipoff.tipoff_index`/`resolve_tipoff`.
- Produces:
  - `CallFeatures(roster_gap: dict[str, float], injury_gap: dict[str, float], absent: tuple[tuple[str, str, str], ...], report_at: datetime | None, fallback: bool, no_report: bool)` (frozen):
    - `roster_gap` and `injury_gap` are keyed `"box"`/`"pm"`.
    - Both gaps are home minus away.
    - `absent` holds `(team, display name, status)` for rotation players with `q < 1`.
    - `fallback` is True when `report_at < at − 1h`.
  - `GameFeatures(game_id, final: CallFeatures, early: CallFeatures, extras: dict[str, float])`. `extras` is for Task 8.
  - `SimStats(no_tip: int, fallback: dict[str, Counter[str]], no_report: dict[str, Counter[str]], unlinked: LinkReport, unknown_status: Counter[str])`, where the per-call counters are keyed by season.
  - `collect_listings(con, games, history, index) -> list[Listing]`: for every game, both calls and every linked rotation player in the snapshot, `played = (a player_games row exists for that game with minutes >= 1)`.
  - `simulate(games: list[fit._Game], lines: PlayerLines, history: InjuryHistory, index: PlayerIndex, params: PlayerParams, listings: list[Listing]) -> tuple[dict[str, GameFeatures], SimStats]`
  - `PlayerLines.from_db(con)`: the latest row per (game_id, player_id), excluding test seasons.
- Rules:
  - Date by date, in the order of `fit._simulate`: compute features for every game on date D first, then apply D's player rows.
  - Logs are stamped D+1 12:00 UTC, so this matches what an `AsOfView` at either call time on D sees. Task 10 checks this.
  - **Rotation of team T:** players whose `last_team` is T and who appear in one of T's last 10 games.
  - **Normal minutes:** `normalize_minutes({p: mean(recent_minutes(p))})`.
  - **Trades:** a player listed in the snapshot under team U ≠ `last_team` is removed from his old team's rotation. He is added to U's only when he has `recent_minutes` (Review Focus 1).
  - **Play probability:** `q = table[season, call].prob(status)`. Unlinked names are q = 1 and counted.
  - **Per formula f:** `roster_gap[f] = S_normal(home) − S_normal(away)` and `injury_gap[f] = (S_avail − S_normal)(home) − (S_avail − S_normal)(away)`.
  - **Games with no tip-off** get both calls at `datetime.combine(game_date, time(23, 0), UTC)` and are counted in `no_tip`.

- [ ] **Step 1: Write the failing tests**
  - `test_same_day_box_scores_not_visible_at_call`: player A scores big on D. Features for A's team's second game on D (a fixture doubleheader on one date) equal those computed without A's D row.
  - `test_report_published_after_call_time_is_ignored`: a report at `final_at + 1s` listing the star Out gives `injury_gap == 0.0` at final, and a report at `final_at` gives nonzero.
  - `test_early_call_uses_earlier_report_than_final`: a 12:00Z report has the star Questionable and a 22:30Z report has him Out. Then `early.absent[0][2] == "Questionable"` and `final.absent[0][2] == "Out"`.
  - `test_missing_slot_falls_back_to_latest_earlier_report_and_counts` (Review Focus 4): the only report is 5 h before final, so `final.fallback is True` and `stats.fallback["final"][season] == 1`. With no report at all, `no_report is True` and `injury_gap == {"box": 0.0, "pm": 0.0}`.
  - `test_traded_player_listed_by_new_team_leaves_old_rotation` (Review Focus 1): A last played for BOS and is listed Out by NYK. BOS's `S_normal` excludes A, and NYK's injury gap is 0 because A has no NYK minutes history in this fixture.
  - `test_injury_gap_matches_hand_value`: the fixture team with 8 × 30 min, star rating forced via monkeypatched `PlayerBook.rating`, and the star Out gives `round(final.injury_gap["box"], 6) == -3.571429` with an unaffected opponent.
  - `test_availability_table_per_season_uses_earlier_seasons_only`: listings planted only in the replayed season leave the table equal to `STATUS_PRIORS`.
  - `test_simulate_refuses_test_season`: a 2026-27 player row present gives `PlayerLines.from_db` excluding it, and `fit._load` raising as today if a 2026-27 game slips in.

- [ ] **Step 2: Run them and confirm they fail.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Commit** with the message `feat: point-in-time player features at early and final call times`.

---

### Task 7: Candidate walk-forward and the switch-over bar

**Files:**
- Create: `src/predictor/model/candidate.py`
- Test: `tests/test_candidate.py`

**Interfaces:**
- Consumes:
  - `evaluate.evaluate(con) -> Evaluation`, for the current model's per-fold `RatingParams` (`ev.winner.choices[season].params`), its `final.ratings`, and its predictions.
  - `tuning.season_weights`, `tuning.HALF_LIVES`, `tuning.TIE_TOLERANCE`, `tuning._best_sigma`; `fit._simulate`; Task 6 `simulate`.
- Produces:
  - `FORMULAS = ("box", "pm", "both")`, `W_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)`
  - `CandidateChoice(team: RatingParams, player: PlayerParams, formula: str, w: float, coefficients: tuple[float, ...], feature_names: tuple[str, ...], sigma: float, games: int)` (frozen)
    - `feature_names` is the five `adjustments.FEATURE_NAMES`, then `roster_box`/`roster_pm` and `injury_box`/`injury_pm` as `formula` selects.
  - The spread model is `(1 − w)·team_gap + home_court + X·coefficients`. The least-squares target is `margin − (1 − w)·team_gap − home_court`.
  - Fitting:
    - Weighted least squares on final-call features of `002` games in the job's seasons.
    - Games with `final.no_report` are left out of the fit but still predicted.
    - Within a job, the lowest weighted MSE over (player params × formula × w) wins, with the first-seen order on ties. Sigma comes from `_best_sigma` on the winner.
  - Team params come from the current model's fold choice. They are not re-gridded; this keeps the search at 6 player passes × 15 (formula, w) per job.
  - `CandidateResult(half_life, final_call: evaluate.VariantResult, early_call: evaluate.VariantResult, choices: dict[str, CandidateChoice], stats: SimStats)`:
    - Both `VariantResult`s carry `details`, so `evaluate.match_market` works on them.
    - `choices` on those `VariantResult`s is `{}`.
  - `CandidateEvaluation(variants: tuple[CandidateResult, ...], winner: CandidateResult, final: CandidateChoice, current: evaluate.Evaluation)`
  - `evaluate_candidate(con, current: evaluate.Evaluation, progress=None) -> CandidateEvaluation`. The recency variant is picked by `evaluate.pick_winner` semantics on final-call log loss.
  - `SwitchOverBar(pooled_brier: tuple[float, float], pooled_log_loss: tuple[float, float], pooled_accuracy: tuple[float, float], seasons_better: int, seasons: int, worst_bucket_gap: float | None, calibration_met: bool, includes_terms: bool)`, where each tuple is (current, candidate):
    - `.met` is True iff all five conditions hold:
      1. Candidate Brier < current Brier and candidate log loss < current log loss.
      2. Candidate accuracy >= current accuracy (raw values).
      3. `seasons_better >= 4`.
      4. `calibration_met` (`report.publishing_bar(candidate).calibration_met`: 5 points in buckets of 50+).
      5. `includes_terms`.
    - `.reasons() -> list[str]`: one plain sentence per failed condition.
  - `switch_over_bar(current: Sequence[Prediction], candidate: Sequence[Prediction], choice: CandidateChoice) -> SwitchOverBar`:
    - Raises `ValueError` if the two game_id sets differ.
    - `includes_terms` is True iff at least one roster coefficient and one injury coefficient in `choice` are nonzero after rounding to 6 dp.

- [ ] **Step 1: Write the failing tests**
  - `test_fold_never_sees_its_own_season`: a choice for 2022-23 is fitted on games whose seasons are all `< "2022-23"` (spy on the rows passed to lstsq).
  - `test_team_params_come_from_current_fold`: `result.choices[s].team == current.winner.choices[s].params` for every walk-forward season.
  - `test_w_one_drops_team_gap`: with `w=1.0`, the spread equals `home_court + X·coef` on a hand-made game.
  - `test_candidate_predicts_same_games_as_current`: the game_id set of `final_call.predictions` equals that of `current.winner.predictions`.
  - `test_switch_over_bar_met`, with constructed predictions over 6 seasons × 100 games where the candidate is better in 4 seasons: `seasons_better == 4`, `met is True`, `reasons() == []`.
  - `test_switch_over_bar_fails_on_accuracy`: the candidate has lower Brier and log loss but accuracy 0.650 against the current 0.654. Then `met is False` and `reasons()` contains a line starting with `"Accuracy is lower"`.
  - `test_switch_over_bar_fails_on_three_seasons`: `seasons_better == 3` gives `met is False`.
  - `test_switch_over_bar_requires_terms`: a choice whose injury coefficients are all 0.0 gives `includes_terms is False` and `met is False`.
  - `test_switch_over_bar_rejects_different_game_sets`: `pytest.raises(ValueError)`.
  - `test_switch_over_bar_has_no_home_comparison`: `SwitchOverBar` has no field containing `"home"`.

- [ ] **Step 2: Run them and confirm they fail.** Expected: FAIL.
- [ ] **Step 3: Implement.** Cache the six `simulate` passes, which are shared by every job and variant. Fold seasons and `fold_job` come from `evaluate`.
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Commit** with the message `feat: walk-forward the player-aware candidate and score the switch-over bar`.

---

### Task 8: Optional gradient-boosted correction

**Files:**
- Modify: `pyproject.toml` (add `"scikit-learn>=1.7"` to `dependencies`), `uv.lock`, `src/predictor/model/player_sim.py` (fill `GameFeatures.extras`)
- Create: `src/predictor/model/correction.py`
- Test: `tests/test_correction.py`

**Interfaces:**
- Produces:
  - `EXTRA_FEATURES = ("recent_form_gap", "pace_sum", "continuity_gap")`, filled in `player_sim` point-in-time:
    - `recent_form_gap`: home minus away of (mean margin over the team's last 10 games − team rating).
    - `pace_sum`: the sum over both teams of mean possessions over the last 10 games, where possessions = Σ(FGA + 0.44·FTA − OREB + TOV).
    - `continuity_gap`: home minus away of the share of available expected minutes from players who played in at least one of the team's last 5 games.
  - `CORRECTION_FEATURES`: `("additive_spread", "team_gap", "home_court")` + `FEATURE_NAMES` + `("roster_gap", "injury_gap")` + `EXTRA_FEATURES`. Here `roster_gap`/`injury_gap` are the chosen formula's roster and injury terms in points (coefficient × feature, summed over the formula's columns).
  - `GBM_PARAMS = dict(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=100, l2_regularization=1.0, random_state=0)`
  - `MIN_TRAINING_SEASONS = 2`
  - `Correction.train(rows: Sequence[tuple[tuple[float, ...], float]]) -> Correction`: fits `sklearn.ensemble.HistGradientBoostingRegressor(**GBM_PARAMS)` on (features, residual = margin − additive spread). `.predict(x) -> float`.
  - `apply_corrections(result: CandidateResult, features: dict[str, GameFeatures]) -> CandidateResult | None`:
    - For each walk-forward season S, train on the walk-forward residuals of earlier walk-forward seasons only. If there are fewer than `MIN_TRAINING_SEASONS`, the correction for S is 0.
    - Sigma is re-chosen with `_best_sigma` on the corrected training residuals.
    - Returns the corrected result for both calls.
  - `correction_helps(base: CandidateResult, corrected: CandidateResult) -> bool`: pooled final-call Brier AND log loss both strictly lower.

- [ ] **Step 1: Write the failing tests**
  - `test_correction_trains_only_on_earlier_seasons`: a spy on `Correction.train` gives, for S = 2023-24, training seasons ⊆ {2020-21, 2021-22, 2022-23}.
  - `test_correction_zero_without_two_training_seasons`: 2020-21 and 2021-22 corrected spreads equal their base spreads.
  - `test_correction_is_deterministic`: two runs give identical predictions.
  - `test_correction_helps_requires_both_metrics`: a constructed corrected result with lower Brier but higher log loss gives `False`.
  - `test_extras_are_point_in_time`: changing a box score dated on or after the game date leaves `extras` unchanged.

- [ ] **Step 2: Run them and confirm they fail.** Expected: FAIL.
- [ ] **Step 3: Implement.** Run `uv add "scikit-learn>=1.7"`. If no wheel exists for Python 3.14, stop and report it rather than pin an older Python.
- [ ] **Step 4: Run the tests and the full suite.** Expected: PASS.
- [ ] **Step 5: Commit** with the message `feat: optional gradient-boosted correction trained on earlier seasons' errors`.

---

### Task 9: `evaluate-model` reports the candidate against the bar

**Files:**
- Modify: `src/predictor/model/evaluate.py` (stop printing `format_publishing_bar` in `format_evaluation`; the calibration table stays), `src/predictor/model/candidate.py` (`format_candidate`), `src/predictor/cli.py` (`evaluate-model --candidate/--no-candidate`, default `--candidate`)
- Test: `tests/test_candidate_report.py`, `tests/test_model_evaluate.py`

**Interfaces:**
- Produces `format_candidate(cev: CandidateEvaluation, bar: SwitchOverBar, corrected: CandidateResult | None, market_final: list[evaluate.MarketGame], market_current: list[evaluate.MarketGame]) -> str`. It prints a section titled `"  Player model candidate (walk-forward, final call = tip-off minus 45 minutes)"` containing:
  1. A per-season table: season, games, then current and candidate Brier, log loss and accuracy, and `better`/`worse`.
  2. A pooled row.
  3. `"  Switch-over bar: MET"` or `"  Switch-over bar: NOT MET"`, then the five conditions, one per line, each prefixed `[pass]`/`[fail]` with its numbers.
  4. When not met, `"  The current model stays live."`, followed by `reasons()`.
  5. Chosen settings per season: formula, w, half-life minutes, regression, and roster/injury coefficients.
  6. The correction line, which is one of:
     - `"  Correction model: included (pooled Brier X -> Y, log loss X -> Y)"`
     - `"  Correction model: dropped -- it did not improve the walk-forward (...)"`
  7. An early-call block: the same per-season table at 08:30 ET (12:30 UTC) for the candidate, with the current model's numbers repeated (it has no injury input, so they are identical).
  8. A market-gap block: `evaluate._market_row` for current and candidate against closing lines, plus `CLOSING_LINE_CAVEAT`. If there are no lines, `NO_MARKET_DATA`.
  9. A data-quality block: fallback and no-report counts per season and call, unlinked injury names (total, then the top 10 as `name (TEAM) ×n`), unknown statuses, and games with no tip-off.
- `--no-candidate` prints exactly today's output, minus the always-pick-home bar. If the `player_games` table is empty, the candidate section prints `"  No player game logs -- run predictor ingest-player-logs"` and exits 0.

- [ ] **Step 1: Write the failing tests**
  - `test_format_candidate_shows_bar_and_reasons`: a constructed failing bar gives text containing `"Switch-over bar: NOT MET"`, `"[fail]"` and `"The current model stays live."`.
  - `test_format_candidate_shows_dropped_correction`: `corrected=None` gives text containing `"Correction model: dropped"`.
  - `test_format_candidate_has_early_and_market_blocks`: the text contains `"early call"` and `"closing"`.
  - `test_format_candidate_lists_unlinked_names`: `"Morris, Marcus (NYK) ×3"` appears.
  - `test_evaluation_no_longer_prints_always_pick_home`: `format_evaluation(ev)` does not contain `"always"` or `"pick home"`.
  - `test_cli_evaluate_model_without_player_logs`: a fixture DB with no player rows gives exit 0, and the output contains `"No player game logs"`.
  - `test_current_model_numbers_unchanged`: the current section's log loss, Brier and accuracy lines are byte-identical with and without `--candidate`.

- [ ] **Step 2: Run them and confirm they fail.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests and the full suite.** Expected: PASS.
- [ ] **Step 5: Commit** with the message `feat: evaluate-model reports the player-model candidate against the switch-over bar`.

---

### Task 10: AsOfView predictor, parity, and the candidate settings file

**Files:**
- Create: `src/predictor/model/player_stage1.py`, `src/predictor/model/player_settings.py`
- Modify: `src/predictor/cli.py` (`fit-player-model`)
- Test: `tests/test_player_stage1.py`, `tests/test_player_settings.py`

**Interfaces:**
- Produces:
  - `player_settings.SETTINGS_PATH = Path(__file__).with_name("player_settings.json")`; `_VERSION = 1`
  - `PlayerModelSettings(team: RatingParams, player: PlayerParams, formula: str, w: float, coefficients: dict[str, float], sigma: float, half_life: float | None, availability: dict[str, dict[str, float]], correction: dict | None, tuning_games: int)` (frozen):
    - `availability` is keyed `"early"`/`"final"` and is estimated on all tuning seasons.
    - `correction` is `{"training_seasons": [...], "params": GBM_PARAMS}`, or None. A correction is retrained deterministically on load and never pickled.
  - `to_json` / `from_json` / `load` / `save`, with `SettingsError` messages ending `"Run: predictor fit-player-model"`.
  - `PlayerBreakdown(game_id, home_team, away_team, rating, roster, home, rest, travel, altitude, injuries, adjustment, spread, p_home, absent)` (frozen):
    - `.sentence()` follows `stage1.Breakdown.sentence` rules (shown terms add up, no `-0.0`).
    - It names absences after the injuries term, as in `"injuries -3.1 (Murray out, Porter questionable)"`.
  - `PlayerStage1Predictor(con, settings: PlayerModelSettings, venues=None)` with `.explain(game: GameToPredict, view: AsOfView, call: Literal["early", "final"]) -> PlayerBreakdown`:
    - Team ratings are caught up exactly as `Stage1Predictor._catch_up` does, by composing a `Stage1Predictor` built from the settings' team params.
    - Player state comes from `view.table("player_games")` and injuries from `InjuryHistory.from_view(view)`.
  - CLI `predictor fit-player-model`:
    - Runs `evaluate.evaluate`, then `evaluate_candidate`, then `switch_over_bar`.
    - If met, it saves `player_settings.json` and prints its path.
    - If not met, it prints `"Switch-over bar not met -- nothing was saved; the current model stays live."` and the reasons, and exits 0.
    - It never touches `stage1_settings.json`.

- [ ] **Step 1: Write the failing tests**
  - `test_parity_with_simulation`: on the Task 6 fixture, for every game and both calls, `PlayerStage1Predictor.explain(..., AsOfView(con, call_at), call).spread == pytest.approx(sim_spread, abs=1e-9)`. This proves the fast replay has no leak the view would block.
  - `test_predictor_never_reads_odds`: a spy on `AsOfView.table` names never sees `"odds_snapshots"`.
  - `test_sentence_names_absences_and_adds_up`: a constructed breakdown gives a sentence containing `"injuries -3.1 (Murray out, Porter questionable)"`, and the shown terms sum to the shown margin.
  - `test_settings_round_trip`: `from_json(to_json(s)) == s`.
  - `test_settings_rejects_wrong_version`: `SettingsError`, with the message ending `"Run: predictor fit-player-model"`.
  - `test_fit_player_model_writes_nothing_when_bar_fails`: `switch_over_bar` monkeypatched to a failing bar gives exit 0, no `player_settings.json` file, and an unchanged `stage1_settings.json` mtime.
  - `test_fit_player_model_saves_when_bar_met`: a met bar gives the file present and `load()` succeeding.

- [ ] **Step 2: Run them and confirm they fail.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests and the full suite.** Expected: PASS.
- [ ] **Step 5: Commit** with the message `feat: AsOfView player-model predictor and candidate settings file`.

---

### Task 11: Real-data run (controller, on the Mac)

- [ ] Full suite green; final whole-branch review; fix wave; merge `player-model` to `main`; pull into the live copy; `uv sync`.
- [ ] `predictor ingest-player-logs` (run in background), then record rows per season and the `unknown_game_ids`. Expect about 26k rows per regular season.
- [ ] `predictor backfill-injury-slots --start 2020-12-22` (run in background; resumable). Record `targets / satisfied / unsatisfied / not_published`. Re-run until `transient == 0`.
- [ ] `predictor evaluate-model`, then read the unlinked-name list.
  - Add genuine spelling differences to `PLAYER_ALIASES`, with a test per alias added to `test_player_ids.py`, and commit.
  - Re-run until unlinked rotation names are below 1% of listings, or each remaining one is explained.
- [ ] `predictor fit-player-model`. Record the bar outcome, pooled and per-season numbers, the early-call numbers, the market gap and the correction verdict in the spec under a new heading, "Replay result (plan 1)".
  - If the bar fails, nothing is saved, the current model stays live, and plan 3 ships the cloud with the current model, as the spec says.
- [ ] Update memory.
