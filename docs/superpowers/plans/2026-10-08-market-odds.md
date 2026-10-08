# Market Odds Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Ingest historical closing odds (Kaggle) and daily live odds (The Odds API), link both to `game_id`, and compare the model with the market in `evaluate-model` and in the daily log. The model itself never reads odds.

**Architecture:**
- `odds_snapshots_raw` gains `game_id`, `source` and `reconstructed`.
- `sources/odds.py` (live) links events to `game_id` via the schedule.
- The new `sources/odds_history.py` downloads, archives, parses and links the Kaggle file.
- The new `model/market.py` turns lines into a de-vigged home-win probability.
- `evaluate.py` adds a model-vs-market section, and `live.py` adds market fields to log and grade lines.

**Spec:** `docs/superpowers/specs/2026-09-22-nba-predictor-design.md`, section **8. Market odds — decided 2026-10-08**.

## Global Constraints

- **Secrets never enter the repository.**
  - The odds key is read from `~/.config/predictor/odds_api_key` (stripped), falling back to env `ODDS_API_KEY`.
  - The Kaggle credentials come from `~/.kaggle/kaggle.json` (`{"username","key"}`).
  - Never print a key. Never write a key to the DB, logs, raw store or `meta`.
  - A test scans every git-tracked file for key-shaped strings: 32-hex runs next to `apiKey`/`key` assignments, and Kaggle `"key": "<32 hex>"`. It fails on any hit.
- **The model never reads odds.** `Stage1Predictor`, `fit` and `tuning` must not touch the odds table. Extend the existing leak tests: a predictor wrapper that records `AsOfView.table` names must never see `"odds_snapshots"` during `explain`.
- **Raw-first.** Archive before parse; parse the archived bytes.
- Physical `_raw` table names appear only in db.py/asof.py.
- Plain-English CLI errors, non-zero exit codes, no tracebacks for expected failures. Stall avoidance: long commands via `run_in_background`.
- **Team names → abbreviations:** reuse `injury_report`'s team-name table (`_TEAM_ABBR_BY_NAME` + aliases). If it is private, add a small public helper `team_abbr(name) -> str | None` in a shared place and reuse it, rather than copying the table.
- **Linking a line to a game:** ET calendar date + home abbr + away abbr against the schedule table's latest vintage (competitive prefixes). Unmatched lines are stored with `game_id = NULL`, counted and reported, never dropped silently.
- Baseline: `uv run pytest -q` gives **712 passed, 1 deselected** on `main` (HEAD 9f86688). Branch: `market-odds`.

---

### Task 1: Storage, secrets, and live odds linked to games

**Files:**
- Modify: `src/predictor/db.py` (migration), `src/predictor/config.py` (secret readers), `src/predictor/sources/odds.py`, `src/predictor/cli.py` (`ingest-odds`), `src/predictor/status.py` (odds advice text)
- Create: `scripts/com.predictor.odds.plist` (17:30, `ingest-odds`)
- Modify: `scripts/install_schedule.sh`
- Tests: `tests/test_odds.py`, `tests/test_db.py`, `tests/test_secrets.py` (new), `tests/test_launchd.py`

**Requirements:**
1. **Migration:** an idempotent `_add_odds_link_columns`:

   ```sql
   ALTER TABLE odds_snapshots_raw ADD COLUMN IF NOT EXISTS game_id VARCHAR
   ALTER TABLE odds_snapshots_raw ADD COLUMN IF NOT EXISTS source VARCHAR
   ALTER TABLE odds_snapshots_raw ADD COLUMN IF NOT EXISTS reconstructed BOOLEAN DEFAULT FALSE
   ```

   Backfill `reconstructed = FALSE` and `source = 'theoddsapi'` where NULL. The DDL in `_SCHEMA` also lists the columns, for fresh DBs. The existing PK `(game_key, book, observed_at)` is unchanged.
2. **`config.odds_api_key() -> str | None`:** the file first, then env. `config.kaggle_credentials() -> tuple[str, str] | None`. Both return None when missing and never raise for a missing file.
3. **`odds.ingest_current(con, api_key=None, now=None, session=None, fetch=None)`:**
   - The download happens before the DB is used. The CLI order is download → archive → `connect_with_retry` → load.
   - `observed_at` is stamped after the fetch returns.
   - Each event is linked to a `game_id`:
     - ET date of `commence_time` + `team_abbr(home)` + `team_abbr(away)`
     - matched against the latest schedule vintage
     - if not found on that date, try ±1 day and record the match
   - Rows are stored with `source='theoddsapi'` and `reconstructed=False`.
   - Return a summary dataclass: `rows`, `events`, `linked`, `unlinked` (list of "AWAY@HOME date").
4. **CLI `ingest-odds`:**
   - Uses `config.odds_api_key()`. With no key, prints a plain message with the exact path to create.
   - `OddsQuotaExceeded` gives a plain message and exit 1. The response's `x-requests-remaining` header is printed when present (expose it from `fetch_current`).
   - Unlinked events are printed as an informational line, not a failure.
5. **Status:** the odds advice names the key file path, not "export ODDS_API_KEY".
6. **Plist and install:** `com.predictor.odds.plist` at 17:30 with `RunAtLoad` false, logs `data/logs/odds.*.log`. The install script installs five jobs. The launchd tests are updated, including the no-shared-start-minute check.
7. **Tests:**
   - migration idempotency, plus the new columns' defaults
   - secrets readers: file wins over env; missing gives None
   - the secret scan of tracked files (`git ls-files`) fails on a planted key-shaped string in a temp repo and passes on the real tree
   - linking: exact date, a ±1 day ET-date edge (a 7:30pm PT game is the next UTC day), and an unmatched event
   - the quota message
   - the remaining-requests header printed

Commit: `feat: live odds linked to games; key read from a private file; daily odds job`.

---

### Task 2: Historical closing odds from Kaggle

**Files:**
- Create: `src/predictor/sources/odds_history.py`
- Modify: `src/predictor/cli.py` (`ingest-odds-history`)
- Test: `tests/test_odds_history.py`

**Requirements:**
1. **Dataset:** Kaggle `cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024` (the slug is historical; the dataset has since been extended to 2025-26).
   - Download via the Kaggle public API with HTTP basic auth from `config.kaggle_credentials()`: `GET https://www.kaggle.com/api/v1/datasets/download/cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024`, which returns a zip.
   - Use `requests` with a timeout, and retry only on network errors.
   - With no credentials: a plain message with the exact path `~/.kaggle/kaggle.json` and how to create it.
   - 401/403: a plain message saying the token is wrong or the dataset terms must be accepted on the dataset page.
2. **Raw-first:** archive the zip bytes to `raw_store` source `odds_history`, key `kaggle_<sha16>.zip`. Then open the archived bytes. Same bytes means a no-op.
3. **Parse.** The real file was inspected by the controller on 2026-10-08:
   - The zip holds one file, `nba_2008-2026.csv` (24,440 rows, 2007-08..2025-26). Select the member by the `.csv` suffix, not by exact name, since the name changes with each update.
   - Columns: `season,date,regular,playoffs,away,home,score_away,score_home,q1_away..ot_home,whos_favored,spread,total,moneyline_away,moneyline_home,h2_spread,h2_total,id_spread,id_total`.
   - `season` is the END year: `2026` means `2025-26`.
   - `date` is `YYYY-MM-DD`, the US local game date; treat it as the ET date.
   - Team codes are lowercase: atl bkn bos cha chi cle dal den det gs hou ind lac lal mem mia mil min no ny okc orl phi phx por sa sac tor utah wsh. Map them with an explicit table to our abbreviations (gs GSW, no NOP, ny NYK, sa SAS, utah UTA, wsh WAS, the rest upper-cased). An unknown code is reported, never skipped silently.
   - `spread` is UNSIGNED; `whos_favored` is `home` or `away`. The home spread (negative = home favoured) is `-spread` if `whos_favored == "home"`, else `+spread`. Blank spread means None (3 rows).
   - `moneyline_home` / `moneyline_away` are American odds as ints, or blank. They are blank for every game from 2023-24 onward and half of 2022-23, so those seasons fall back to the spread. This is expected, not an error.
   - `total`: float or None. Ignore `h2_*`, `id_*` and the quarter columns. Keep `regular`, `playoffs`, `score_home`, `score_away` for linking checks.
   - The module docstring records all of the above.
4. **Link and store:**
   - Link each row by ET date + home + away against the latest schedule vintage (competitive prefixes). Rows before 2014-15 are ignored.
   - **Score check:** when the linked game has a FINAL result, the file's scores must equal ours. A mismatch is reported as unmatched, not stored.
   - Store one row per game: `book='consensus'`, `source='kaggle_sbr'`, `reconstructed=True`, `home_price` = moneyline_home, `away_price` = moneyline_away, `spread` = the signed home spread, `total`.
   - `observed_at` = the game's `tip_off_utc` from the schedule; skip rows with no tip, counted.
   - `game_key = f"kaggle:{game_id}"`, written with INSERT OR REPLACE.
5. **Coverage report** (returned and printed by the CLI): per season, the regular-season games in the schedule, games linked to a line, and coverage %, plus any unmatched rows (count and the first 10). Exit 1 if any season from 2019-20 to 2025-26 has coverage below 90%, with a plain message (the comparison would be unreliable). Otherwise exit 0.
6. **Tests:**
   - the parser on a small in-test fixture shaped exactly like the real file's columns, copied from the inspected header
   - team alias mapping, including a deliberately unknown team that is reported
   - linking and the ET-date handling
   - idempotent re-run
   - raw-first proven via monkeypatched `raw_store.load`
   - CLI messages for missing credentials, 401/403 and low coverage
   - no network in tests: inject the download function

Commit: `feat: ingest historical closing odds from Kaggle and report coverage`.

---

### Task 3: Market probability and the model-vs-market comparison

**Files:**
- Create: `src/predictor/model/market.py`
- Modify: `src/predictor/model/evaluate.py` (report section), `src/predictor/cli.py` only if needed
- Tests: `tests/test_market.py`, `tests/test_model_evaluate.py`

**Requirements:**
1. `american_to_prob(price: int) -> float`: −200 → 2/3; +170 → 100/270 = 0.37037…
2. `devig(p_home_raw, p_away_raw) -> float`: returns `p_home_raw / (p_home_raw + p_away_raw)`. Example: −200/+170 → 0.6667/(0.6667+0.3704) = 0.64286 (4 dp checked by hand in the test).
3. `market_p_home(lines, sigma) -> MarketView | None`, where `MarketView(p_home: float, spread: float | None, books: int, observed_at: datetime)`:
   - For each line with both moneylines, take the de-vigged probability. If no line has moneylines but a spread exists, use `win_probability(-spread, sigma)`. The spread is quoted for the home team, with negative meaning home favoured. A test pins this sign convention by hand.
   - The median across lines.
   - The latest `observed_at` among the lines used.
   - `None` when nothing is usable.
4. **Reading lines.** For a `game_id`, read rows from the odds table:
   - Historical comparison: rows with `source='kaggle_sbr'`.
   - Live: the latest snapshot per book with `observed_at <= now`, through `AsOfView`.
   - These reads live only in `market.py` and `evaluate.py` / `live.py`, never in the model.
5. **`evaluate-model` gains a "Model vs market (closing lines)" section** for the chosen variant's walk-forward predictions with a market line:
   - per season and pooled: games compared; model and market accuracy, Brier and log loss
   - disagreement zone: games where the picks differ OR |model p − market p| ≥ 0.10; count; model right %; market right %
   - beating the closing spread: games where model spread and closing spread differ and the game isn't a push against the closing spread; the share where the actual margin landed on the model's side
   - one honest line: "Closing lines include information up to tip-off (injuries, lineups) that the model did not have."
   - if no odds are stored, the section prints "No market data -- run predictor ingest-odds-history".
6. **Tests:**
   - all arithmetic by hand
   - disagreement-zone and closing-spread counts on a tiny constructed set
   - the section appears and contains the per-season rows when a fixture DB has odds
   - the no-odds message
7. Per season, the report shows how many market probabilities came from moneylines and how many from the spread, and states plainly that spread-derived probabilities use the model's own sigma.

Commit: `feat: compare the model with closing-market probabilities in evaluate-model`.

---

### Task 4: Market fields in the daily log

**Files:**
- Modify: `src/predictor/model/live.py`, `src/predictor/status.py` (odds live check)
- Tests: `tests/test_live_predict.py`, `tests/test_status.py`

**Requirements:**
1. Each `predicted` line gains:
   - `market_p_home`, `market_spread`, `market_books`, `market_observed_at`, from live odds (`source='theoddsapi'`) visible at `now` for that `game_id`
   - `market_label: "market line at 17:30 IST"` when there is a market, otherwise all market fields are null
   - The model's numbers must be byte-identical with and without odds present (test).
2. Grade lines gain `market_correct`: `(market_p_home >= 0.5) == home_won`, or null without a market.
3. Status gains an `odds` live entry:
   - stale if `in_season` and no `theoddsapi` row was observed in the last 30 h
   - advice names the key file and the job label `com.predictor.odds`
   - off-season is quiet
   - Replace or retire the generic `odds_snapshots` table-freshness entry so status doesn't report odds twice. Keep it if the tests rely on it, but make its advice consistent.
4. Tests for all of the above, plus the leak test (the model never reads odds).

Commit: `feat: daily log records the 17:30 market line beside each prediction`.

---

### Task 5: Go live (controller)

- [ ] Confirm the owner's key files exist: `test -s ~/.config/predictor/odds_api_key && test -s ~/.kaggle/kaggle.json`. Never print their contents.
- [ ] Full suite green; final whole-branch review; fix wave.
- [ ] Merge to `main`, push, then pull into the live copy and run `uv sync` there.
- [ ] From the live copy:
  - `predictor ingest-odds-history`, and record coverage.
  - `predictor evaluate-model`, and record the model-vs-market section in the spec.
  - `predictor ingest-odds` once (live), and record the linked count and remaining requests.
- [ ] Reinstall the jobs from the live copy (`bash scripts/install_schedule.sh`) and verify five jobs are loaded.
- [ ] Update memory.
