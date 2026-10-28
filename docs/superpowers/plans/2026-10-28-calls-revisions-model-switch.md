# Calls, Revisions and the Model Switch Implementation Plan (plan 3 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Estimate home court per team (point-in-time, shrunk toward the league value, strength chosen walk-forward), re-score the player model with it against the switch-over bar, then make the cloud write early, revision and final calls, switch the live model to the player model behind an explicit flag, and show call history and revisions on the site.

**Architecture:**
- **Home court per team** lives in `Ratings`, so the current model (`Stage1Predictor`, `fit._simulate`) and the player model (which composes a `Stage1Predictor`) share one implementation. A flat setting (`shrinkage = inf`) reproduces today's numbers exactly. `candidate.py` adds the shrinkage to its walk-forward search and reports flat and per-team side by side.
- **Calls** are decided by a small pure module, `model/calls.py`: call windows, per-game call chains, injury status changes, the revision rule and the reason text. `live.predict_today` writes the lines. The model that makes them comes from `model/live_model.py`: a committed flag file picks the current model (v1) or the player model (v2) from a given slate on.
- **Cloud:** the `predict` step runs on every run, not only at the early call. Odds fetched in one loop iteration are cached for later iterations, so revision and final calls carry the latest market line. The Mac's `predict-today` refuses to write calls once v2 is live or the cloud already publishes `main`.
- **Site:** calls are grouped per game. The record uses the graded call, with a secondary early-call line, a "Model v2 from <date>" marker and the call history.

**Tech Stack:** Python 3.14, uv, DuckDB, numpy, typer, git via subprocess; Astro 7 site with `node --test` (Node ≥ 22.22, built-in type stripping) for the site's pure logic.

**Spec:** `docs/superpowers/specs/2026-10-22-cloud-injuries-design.md`, including "Calls and their timing" (Goal 2), "Prediction log and site changes", "Calls made after the builder check" 2–6, and "Owner decision 2026-10-25: home court per team". It extends `docs/superpowers/specs/2026-09-22-nba-predictor-design.md` §3 (Stage 1: "Home court advantage — estimated from data, not assumed constant") and §7 (append-only log).

## Global Constraints

- **Scope.** Plan 3 of 3. The cut-over is not in this plan: the controller does it separately. The model switch ships **off**. With no `state/settings/live_model.json`, every call is made by the current model (v1), exactly as today.
- **Nothing is sealed at or after tip-off** (spec anti-goal).
  - Early calls keep today's rule: tip-off more than `START_BUFFER` (30 min) after `now`.
  - Revision and final calls need `now < tip − CALL_CUTOFF` (15 min).
  - `predicted_at` is the `now` of the run, and the run commits within the same iteration.
- **Append-only.** A log line is never edited or removed. Several lines per game are now normal: one early call, any number of revisions, and one final call (`predicted` or `not_predicted`). Provisional `espn:` ids are never rewritten. Every later call of a game is written under the `game_id` of that game's first line.
- **The graded call is the last `predicted` line whose `predicted_at` is before the cutoff.** The cutoff is the earlier of the logged and current tip-off, as `grade` does today. When two lines share a `predicted_at`, the later line in the log wins.
- **Point-in-time.**
  - Live calls read only through `AsOfView(con, now)`.
  - `player_boxscores` is read only through `AsOfView.latest` (key `game_id, player_id, source`).
  - ESPN box rows are used for a game only when it has no NBA `player_games` row.
  - The time of the latest injury report comes from `injury_reports` (unchanged reports add no `injury_status` rows).
  - In the replay, a report counts at call time T only if its header time is `<= T − REPLAY_REPORT_LAG` (15 min).
- **Walk-forward.** The home-court shrinkage and the altitude allowance for season S are chosen only from tuning seasons strictly before S. `math.inf` (flat) is first in the grid, so per-team wins only on strictly lower weighted MSE.
- **Altitude stays a separate term.** Residuals that feed a team's home edge have the altitude allowance subtracted at altitude games. The altitude coefficient is still fitted on its own.
- **Live correction model:** none. `live_model.load_player_settings` refuses settings whose `correction` is not null (the plan-1 ruling: a live correction would be trained differently from the validated one).
- **The model never reads odds.** Market fields on calls are display-only, as today.
- **Odds budget unchanged:** 2 windows per slate day (plan 2). No extra fetches for revisions or finals.
- **Times:** stored in UTC. Copy shows IST first, then ET. `revision_reason` times are IST (`"new injury report 23:30 IST: Murray (DEN) out"`).
- **JSON lines:** `json.dumps(line, sort_keys=True)` plus `\n`, as `live._append_line` does now.
- **No network in tests.** Fake callers, fetchers and clocks; git tests use `tmp_path` repos.
- **Baseline:** `uv run pytest -q` collects **1535 tests, 1 deselected** on `main` (HEAD 08dec5c). Run it as `uv run pytest -q > log; echo exit=$?` after `uv sync --extra dev` (never piped to `tail`). Branch: `calls-and-switch`. Run the full suite with `run_in_background`.

## Calls made in this plan (for the controller to rule on)

1. **Replay report lag = 15 minutes.** The loop runs at :07/:22/:37/:52 and fetches the newest 15-minute slot, so a live call at t sees reports up to about t − 7 to t − 22 min. A 15-minute lag keeps the replay from seeing a report the live final call could not have. Cost if wrong: the replay is slightly pessimistic.
2. **`CALL_CUTOFF` = 15 minutes.** The final-call window is `[tip − 45, tip − 15)`. That is 30 minutes wide, so it always holds two loop marks (one late or skipped iteration is survivable). Revisions stop 15 minutes before tip, which leaves time to commit and push. Cost: a scratch announced inside the last 15 minutes is never called.
3. **ESPN rows are "box-incomplete".** They have no FGA/FTA/OREB/STL/BLK/TOV/PF, so they update plus-minus ratings, minutes and rosters, but never the box rating. NBA rows replace them at the Mac's weekly refresh.
4. **A game keeps the model of its first call.** If the flag turns on mid-slate, games already called stay on v1 for every later call. The flag file also carries `from_slate`, so a slate is normally all one model.
5. **The odds cache** lives in the loop's base directory, outside every iteration's data directory. It is never committed or archived and is lost when the job ends. Without it, revision and final lines copy the market fields of the game's latest call.
6. **Missed finals.** A game with a sealed call but no final once `tip − CALL_CUTOFF` has passed gets one `not_predicted` line with `call_type: "final"`. A game with no call at all keeps today's single `not_predicted` line.


## Controller rulings (2026-10-28, before execution)

- All six "Calls made in this plan" are accepted as written: the 15-minute replay lag, `CALL_CUTOFF` of 15 minutes, ESPN rows being box-incomplete, a game keeping the model of its first call, the odds cache living for one job, and the missed-final line.
- The cloud is LIVE on main since the cut-over on 2026-10-28, and the Mac's predict job is uninstalled. The Task 9 guard still ships as a safety net for manual runs.
- Task 11's flag stays OFF until the owner approves switching the live model, and the controller asks at that moment.
- Every test run reports the real pytest exit status: `uv run pytest -q > log; echo exit=$?`.

## Review Focus

1. **Loop delay against tip-off.** A late or skipped iteration must still leave a final call in `[tip − 45, tip − 15)`, and no revision or final may be written at or after `tip − 15`. Tests: `test_final_window_holds_two_loop_marks` (Task 5), `test_never_sealed_inside_the_cutoff` (Task 7).
2. **Duplicate revisions and ping-pong.** Two runs that see the same report must write one revision. A status that flips back must revise again, measured against the latest sealed call. Tests: `test_no_duplicate_revision_for_same_report`, `test_status_flipping_back_revises_again` (Task 7).
3. **Late scratch after the final call.** A change before `tip − 15` revises after the final, and that revision becomes the graded call. A change after `tip − 15` writes nothing, and the final stays graded. Tests: `test_revision_after_final_becomes_graded_call`, `test_late_scratch_after_cutoff_is_not_called` (Task 7).
4. **Unknown statuses and dropped names.** A status outside the five known ones (e.g. "Not With Team") counts as a change and is named as written. A player dropped from the report reads "available". Tests: `test_unknown_status_counts_as_a_change`, `test_status_changes_detects_new_flip_and_drop` (Task 5).
5. **Postponed games and provisional ids.** A game moved to another date gets no final or missed-final line under its old date, and a fresh early call on its new date. A chain begun under `espn:<id>` keeps that id for every later call. Tests: `test_postponed_game_gets_no_final_under_old_date`, `test_provisional_chain_keeps_its_id` (Task 7).
6. **DST.** On 2026-11-01 (EDT→EST), the final window is computed in UTC and the slate is unchanged. Test: `test_final_due_across_dst_change` (Task 5).

---

### Task 1: Home court per team in `Ratings` (shared by both models)

**Files:**
- Modify: `src/predictor/model/ratings.py`, `src/predictor/model/fit.py` (`_simulate`, new `final_ratings`), `src/predictor/model/stage1.py` (`explain`, `_apply_rows`), `src/predictor/model/settings.py` (`ModelSettings.home`, JSON)
- Test: `tests/test_model_ratings.py`, `tests/test_model_fit.py`, `tests/test_model_stage1.py`, `tests/test_model_settings.py`

**Interfaces:**
- Produces, in `ratings`:
  - `TEAM_HOME_WINDOW = 164` (each team's last 164 home and 164 away non-neutral games)
  - `SHRINKAGE_GRID = (math.inf, 400.0, 150.0, 50.0)`
  - `HomeParams(shrinkage: float, altitude_allowance: float = 0.0, window: int = TEAM_HOME_WINDOW)` (frozen), with property `.flat -> bool` (`shrinkage == math.inf`). Shrinkage must be `>= 0` or inf, otherwise `ValueError`.
  - `FLAT_HOME = HomeParams(math.inf)`
  - `shrunk_deviation(home: Sequence[float], away: Sequence[float], shrinkage: float) -> float`:
    - 0.0 if either side is empty or shrinkage is inf.
    - Otherwise `raw · n_eff / (n_eff + shrinkage)`, with `raw = mean(home) − mean(away)` and `n_eff = 1 / (1/n_home + 1/n_away)`.
  - `TeamHomeSummary(team: str, home_games: int, away_games: int, raw: float, deviation: float)` (frozen)
  - `Ratings(params: RatingParams, home: HomeParams = FLAT_HOME)`:
    - `.home_court(team: str | None = None) -> float`. With no team: the league mean, unchanged. With a team: league mean + `.deviation(team)`.
    - `.deviation(team) -> float` returns `shrunk_deviation(home residuals, away residuals, home.shrinkage)`.
    - `.team_home_summary(team) -> TeamHomeSummary`
    - `.apply(result, altitude_game: bool = False)`, in this order:
      1. `gap` = rating gap.
      2. `league` = `home_court()`.
      3. `hc` = 0 if neutral, else `league + deviation(home_team)`.
      4. Update ratings on `margin − (gap + hc)`, as now.
      5. If not neutral: `resid = margin − gap − league − altitude_allowance·altitude_game`. Append `resid` to the home team's home deque and `−resid` to the away team's away deque, then append `margin` to the league deque (as now).
- `fit._simulate(params, games, home: HomeParams = FLAT_HOME) -> list[tuple[float, float]]`:
  - The second element is `0.0` if neutral, else `ratings.home_court(home_team)`.
  - `apply(..., altitude_game=g.x[4] == 1.0)`.
- `fit.final_ratings(params, games, home) -> Ratings`: the `Ratings` after the last game.
- `ModelSettings.home: HomeParams = FLAT_HOME` (last field).
  - `to_json` writes `"home": {"shrinkage", "altitude_allowance", "window"}` only when `not home.flat`, so the committed `stage1_settings.json` round-trips byte-identical.
  - `from_json`: a missing key gives `FLAT_HOME`. A negative shrinkage raises `SettingsError`. `_VERSION` stays 2.
- `Stage1Predictor`:
  - Builds `Ratings(settings.ratings, settings.home)`.
  - `explain`: `home = 0.0 if neutral else self._ratings.home_court(game.home_team)`.
  - `_apply_rows` passes `altitude_game=adj.is_altitude_game(venue.city, neutral)`.

- [ ] **Step 1: Write the failing tests**
  - `test_shrunk_deviation_values`:
    - `shrunk_deviation([4.0]*4, [0.0]*4, 2.0) == 2.0`
    - `shrunk_deviation([3.0, 5.0], [1.0], 1.0) == pytest.approx(1.2)`
    - `shrunk_deviation([4.0], [0.0], math.inf) == 0.0`
    - `shrunk_deviation([], [1.0], 1.0) == 0.0`
  - `test_team_home_court_two_games`: `Ratings(RatingParams(0.1, 40.0, 0.0, 10), HomeParams(0.0))` applies two games:
    - Game 1: A at home beats B 110–100 (2024-01-01).
    - Game 2: B at home beats A 102–100 (2024-01-02).
    
    Then `home_court() == 6.0`, `home_court("A") == pytest.approx(10.0)`, `home_court("B") == pytest.approx(10.0)` and `rating("A") == pytest.approx(1.6)`. The same games with `HomeParams(0.5)` give `home_court("A") == pytest.approx(8.0)`.
  - `test_altitude_allowance_kept_out_of_team_edge`: the same games with `HomeParams(0.0, altitude_allowance=1.5)`, and game 1 applied with `altitude_game=True`, give `home_court("A") == pytest.approx(8.5)`.
  - `test_neutral_game_leaves_team_edges_untouched`: a neutral game gives `team_home_summary("A").home_games == 0`, and `home_court()` is unchanged.
  - `test_team_window_keeps_last_164_per_side`: after 170 home games of A, `team_home_summary("A").home_games == 164`.
  - `test_flat_home_equals_league_value`: with `FLAT_HOME`, after the fixture season, `home_court(t) == home_court()` for every team. The existing fit and stage1 tests that pin flat values pass unchanged.
  - `test_stage1_matches_simulation_with_team_home`: on the `model_fixtures` DB, with settings `home=HomeParams(50.0, 0.66)`, `Stage1Predictor.explain(...).home` for every game equals `fit._simulate(params, games, home)[i][1]` within `1e-9`.
  - `test_settings_home_round_trip`: `from_json(to_json(s)) == s` for `home=HomeParams(150.0, 0.658231)`. The committed `stage1_settings.json` text equals `to_json(from_json(text))`.
  - `test_settings_rejects_negative_shrinkage`: `SettingsError`.
- [ ] **Step 2: Run them and confirm they fail.** Run `uv run pytest tests/test_model_ratings.py tests/test_model_settings.py tests/test_model_stage1.py tests/test_model_fit.py -q`. Expected: FAIL (`HomeParams` missing).
- [ ] **Step 3: Implement** the interfaces above. Use `deque(maxlen=home.window)` per team and side. `fit.describe` adds the line `"Home court: league average plus each team's own edge, shrunk by {shrinkage:g} games"` only when not flat.
- [ ] **Step 4: Run the tests and the full suite.** Expected: all pass, and today's stage1 numbers are unchanged.
- [ ] **Step 5: Commit** with the message `feat: home court per team, shrunk toward the league value (flat by default)`.

---

### Task 2: Replay report lag and the per-team walk-forward in the candidate

**Files:**
- Modify: `src/predictor/model/player_sim.py`, `src/predictor/model/candidate.py`
- Test: `tests/test_player_sim.py`, `tests/test_candidate.py`

**Interfaces:**
- Consumes: Task 1's `HomeParams`, `FLAT_HOME`, `SHRINKAGE_GRID`, `fit._simulate(..., home)` and `fit.final_ratings`.
- Produces:
  - `player_sim.REPLAY_REPORT_LAG = timedelta(minutes=15)`.
  - `simulate(games, lines, history, index, params, listings, *, report_lag: timedelta = REPLAY_REPORT_LAG)` and `collect_listings(con, games, history, index, *, report_lag=REPLAY_REPORT_LAG)`. Every `history.snapshot_at(d, at)` becomes `snapshot_at(d, at − report_lag)`, and `fallback` is judged against `at − report_lag`. Call times themselves are unchanged.
  - `CandidateChoice.home: HomeParams = FLAT_HOME` (new last field).
  - `_World.team_sim(params: RatingParams, home: HomeParams = FLAT_HOME) -> np.ndarray`, cached per `(params, home)`.
  - `_fit_job(world, job, team, season, *, shrinkage_grid: tuple[float, ...], altitude_allowance: float) -> CandidateChoice`:
    - The loop order is shrinkage (grid order, outermost), then player, formula and w.
    - Only a strictly lower weighted MSE replaces the current best, so ties keep flat.
  - Altitude allowance: for fold S, `current.winner.choices[S].coefficients.altitude`; for the final fit, `current.final.coefficients.altitude`. It is always stored in `choice.home`, even when the choice is flat.
  - `evaluate_candidate(con, current, progress=None, *, shrinkage_grid: tuple[float, ...] = (math.inf,)) -> CandidateEvaluation`. The signature and the default (flat) are as in plan 1.
  - `HomeVariants(flat: CandidateEvaluation, per_team: CandidateEvaluation)` (frozen).
  - `evaluate_home_variants(con, current, progress=None) -> HomeVariants`: builds the world once (one `simulate` per `PLAYER_GRID` entry), then fits with grid `(math.inf,)` and with `SHRINKAGE_GRID`.
  - `TeamHomeEdge(team: str, home_games: int, away_games: int, raw: float, edge: float)` (frozen).
  - `home_edges(games: list[_Game], choice: CandidateChoice) -> list[TeamHomeEdge]`:
    - Read from `fit.final_ratings(choice.team, games, choice.home)`.
    - `edge = home_court(team)`.
    - Covers the teams with a game in `TUNING_SEASONS[-1]`, sorted by edge descending, then team.

- [ ] **Step 1: Write the failing tests**
  - `test_report_lag_hides_report_inside_lag`:
    - A report at `final_at − 10 min` listing the star Out gives `final.injury_gap == {"box": 0.0, "pm": 0.0}` with the default lag, and a nonzero gap with `report_lag=timedelta(0)`.
    - A report at `final_at − 15 min` gives a nonzero gap with the default lag.
  - `test_flat_grid_keeps_flat_choices`: `evaluate_candidate(con, current)` gives `choice.home.flat` for every season, and team sims called with a `HomeParams` whose `.flat` is True.
  - `test_per_team_needs_strictly_lower_mse`: monkeypatch `candidate._weighted_lstsq` to return equal MSE for every call. Every per-team choice is then flat.
  - `test_altitude_allowance_comes_from_current_fold`: `per_team.winner.choices[s].home.altitude_allowance == current.winner.choices[s].coefficients.altitude` for every walk-forward season.
  - `test_home_variants_share_one_world`: a spy on `player_sim.simulate` records `len(PLAYER_GRID)` calls for `evaluate_home_variants`.
  - `test_home_edges_cover_last_season_teams`: the team set equals the fixture's 2025-26 teams. With a flat choice, every `edge == pytest.approx(league)`.
  - The existing `test_fold_never_sees_its_own_season` passes for the per-team grid too.
- [ ] **Step 2: Run them and confirm they fail.** Run `uv run pytest tests/test_player_sim.py tests/test_candidate.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.** `GameParts.home_court` already carries the per-game value; it now holds the team-specific edge.
- [ ] **Step 4: Run the tests and the full suite.** Expected: all pass.
- [ ] **Step 5: Commit** with the message `feat: walk the per-team home-court shrinkage forward in the player candidate; replay reports lag 15 minutes`.

---

### Task 3: Player settings v2, predictor, report and `fit-player-model`

**Files:**
- Modify: `src/predictor/model/player_settings.py`, `src/predictor/model/player_stage1.py`, `src/predictor/model/candidate.py` (`format_home_court`), `src/predictor/cli.py` (`evaluate-model`, `fit-player-model`)
- Test: `tests/test_player_settings.py`, `tests/test_player_stage1.py`, `tests/test_candidate_report.py`

**Interfaces:**
- Consumes: Task 2's `HomeVariants`, `evaluate_home_variants`, `home_edges` and `TeamHomeEdge`.
- Produces:
  - `player_settings._VERSION = 2`; `PlayerModelSettings.home: HomeParams = FLAT_HOME` (last field).
    - JSON `"home": {"shrinkage": float | null, "altitude_allowance": float, "window": int}` is always written in v2. `null` means `math.inf`.
    - `from_json` accepts version 1 (no `home` key) as `FLAT_HOME`, so the committed v1 file still loads. Any other version raises `SettingsError`.
  - `settings_from_candidate` copies `cev.final.home`.
  - `player_stage1._stage1_settings` passes `home=s.home`.
  - `PlayerStage1Predictor(con, settings, venues=None, *, report_lag: timedelta = timedelta(0))`: the snapshot is `snapshot_at(d, view.as_of − report_lag)`. Live uses 0.
  - `format_home_court(variants: HomeVariants, edges: list[TeamHomeEdge], altitude: float) -> str`, a section titled `"  Home court per team (walk-forward; flat = one league-wide value)"` with:
    1. One line per walk-forward season: `"  2022-23  flat"` or `"  2022-23  shrinkage 150 games"`.
    2. A pooled row each for flat and per-team: Brier, log loss, accuracy.
    3. `"  Per-team home court: kept"`, or `"  Per-team home court: dropped -- the flat league value scored as well or better"`.
    4. One line per team: `f"  {team}  {edge:+.1f}  (own {raw:+.1f} from {home_games} home / {away_games} away games)"`.
    5. `f"  Altitude is a separate term: {altitude:+.2f} points, not inside DEN's or UTA's edge."`
    6. `"  Reports count only from 15 minutes before each call time (live runs fetch every 15 minutes)."`
  - `evaluate-model`:
    - Runs `evaluate_home_variants`.
    - Prints `format_candidate` for `variants.per_team`, with the bar from `switch_over_bar(current.winner.predictions, per_team.winner.final_call.predictions, per_team.final)`.
    - Prints one extra line, `"  Player model with flat home court (plan 1): Brier X, log loss Y, accuracy Z%"`, then `format_home_court`.
  - `fit-player-model` uses the same bar and saves `settings_from_candidate(con, variants.per_team, corrected)`. The rest of its behaviour is as in plan 1.

- [ ] **Step 1: Write the failing tests**
  - `test_settings_v1_file_loads_with_flat_home`: the committed `player_settings.json` loads, and `.home.flat is True`.
  - `test_settings_v2_round_trip_with_inf`: `HomeParams(math.inf, 0.66)` writes `"shrinkage": null`, and `from_json(to_json(s)) == s`.
  - `test_parity_with_simulation_with_team_home_and_lag`: settings with `home=HomeParams(150.0, 0.66)`. For every fixture game and both calls, `PlayerStage1Predictor(con, s, report_lag=REPLAY_REPORT_LAG).explain(g, AsOfView(con, call_at), call).spread` equals the simulation's spread within `1e-9`.
  - `test_format_home_court_lists_every_team_edge`: the text contains `"Home court per team"`, `"DEN"` and `"Altitude is a separate term"`.
  - `test_format_home_court_dropped_when_flat_wins`: every per-team choice flat gives `"Per-team home court: dropped"`.
  - `test_fit_player_model_bar_uses_per_team_variant`: a spy on `switch_over_bar` shows its candidate predictions are `variants.per_team.winner.final_call.predictions`.
- [ ] **Step 2: Run them and confirm they fail.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests and the full suite.** Expected: all pass.
- [ ] **Step 5: Commit** with the message `feat: player settings carry home court per team; evaluate-model reports each team's home edge`.

---

### Task 4: ESPN box rows for live player ratings

**Files:**
- Modify: `src/predictor/sources/player_games.py` (`PlayerGameRow.box_complete`), `src/predictor/model/player_ratings.py` (`apply_game`), `src/predictor/model/player_sim.py` (`PlayerLines.from_view`, `espn_rows`)
- Test: `tests/test_player_sim.py`, `tests/test_player_ratings.py`

**Interfaces:**
- Produces:
  - `PlayerGameRow.box_complete: bool = True` (new last field).
  - `PlayerBook.apply_game` skips the `"box"` update for a row with `box_complete False`. The `"pm"` rating, minutes and lineups update as before.
  - `EspnMergeStats(games: int, rows: int, unlinked: int)` (frozen).
  - `espn_rows(view, before: date, nba_games: set[str]) -> tuple[list[tuple[PlayerGameRow, str]], EspnMergeStats]`:
    - Reads `view.latest("player_boxscores")` and keeps games dated before `before` with no id in `nba_games`.
    - Skips `did_not_play` rows. Skips `player_id` values starting `"espn:"`, counting them as `unlinked`.
    - Maps `player_id=int(...)`, `pts=points`, `ast=assists`, `plus_minus=float`, `minutes`, all other box fields None, `box_complete=False`.
    - Season type: `"playoffs"` for game-id prefix `004`, `"playin"` for `005`, else `"regular"`.
  - `PlayerLines.from_view(view, before, *, espn: bool = True)`: NBA rows as now, plus `espn_rows(...)` when `espn` is True. The stats go on the new attribute `PlayerLines.espn: EspnMergeStats` (all zeros by default).

- [ ] **Step 1: Write the failing tests**
  - `test_espn_rows_used_only_when_nba_rows_absent`:
    - Game G1 has NBA and ESPN rows, so only its NBA rows are present.
    - Game G2 has only ESPN rows, so they are present with `box_complete is False`.
    - `lines.espn.games == 1`.
  - `test_espn_rows_respect_view_cutoff`: an ESPN row observed `2026-11-03T05:00Z` is absent at view `04:59:59Z` and present at `05:00Z`.
  - `test_espn_latest_observation_wins`: two observations of one (game, player), with minutes 30.0 then 31.0, give 31.0.
  - `test_unlinked_espn_player_skipped_and_counted`: `player_id "espn:3945274"` gives `lines.espn.unlinked == 1` and no row.
  - `test_dnp_espn_row_ignored`.
  - `test_espn_season_type_from_game_id`: `"0042600101"` gives `"playoffs"` and `"0022600101"` gives `"regular"`.
  - `test_box_rating_not_updated_from_incomplete_row`: a new player with one incomplete row keeps `rating(p, "box") == ROOKIE_PRIOR` and gets a changed `"pm"` rating, and `recent_minutes` gains the row.
- [ ] **Step 2: Run them and confirm they fail.** Expected: FAIL.
- [ ] **Step 3: Implement.** `from_db` (the replay) is unchanged: NBA rows only.
- [ ] **Step 4: Run the tests and the full suite.** The plan-1 parity tests must still pass.
- [ ] **Step 5: Commit** with the message `feat: live player ratings read ESPN box rows when NBA rows are absent (plus-minus and minutes only)`.

---

### Task 5: Call rules (pure)

**Files:**
- Create: `src/predictor/model/calls.py`
- Test: `tests/test_calls.py`

**Interfaces:**
- Consumes: `availability.Snapshot`.
- Produces:
  - `CallType = Literal["early", "revision", "final"]`
  - `ALL_CALLS: frozenset[str] = frozenset({"early", "revision", "final"})`; `EARLY_ONLY = frozenset({"early"})`
  - `FINAL_LEAD = timedelta(minutes=45)`, `CALL_CUTOFF = timedelta(minutes=15)`, `REVISION_MIN_MOVE = 0.02`, `MAX_NAMED_CHANGES = 3`
  - `call_type_of(line: dict) -> str`: `line.get("call_type") or "early"`
  - `GameChain(lines: tuple[dict, ...])` (frozen). Its properties:
    - `.game_id`: the first line's.
    - `.early`: the first early line whose `reason` is not `TBD_REASON`. `live._TBD_REASON` moves here as `calls.TBD_REASON` (same text) and `live` imports it, so `calls` never imports `live`.
    - `.final`: the first line with call type `"final"`.
    - `.first_predicted` and `.latest_predicted`: by log order.
    - `.predicted`: every predicted line, in log order.
  - `chains(log: list[dict], links: dict[str, str]) -> dict[tuple[str, str], GameChain]`, keyed `(links.get(game_id, game_id), game_date)`, in log order.
  - `final_due(tip: datetime, now: datetime) -> bool`: `tip − FINAL_LEAD <= now < tip − CALL_CUTOFF`
  - `still_callable(tip: datetime, now: datetime) -> bool`: `now < tip − CALL_CUTOFF`
  - `StatusChange(team: str, player: str, before: str | None, after: str | None)` (frozen):
    - `player` is the report's `"Last, First"` key.
    - `.text() -> str` gives `f"{surname} ({team}) {after.strip().lower() if after else 'available'}"`, where the surname is the part before the comma, or the last word when there is no comma.
  - `status_changes(before: Snapshot | None, after: Snapshot | None, teams: tuple[str, str]) -> list[StatusChange]`:
    - Covers every `(team, player)` key of either team in either snapshot whose status differs after `strip().lower()`. A missing key is None.
    - Ordered by the team's position in `teams`, then player.
  - `revision_due(latest: dict, new_p: float, report_at: datetime | None, changes: list[StatusChange]) -> bool`: True iff all of these hold:
    - `report_at` is not None;
    - `latest["injury_report_at"]` is None or older than `report_at`;
    - `changes` is not empty;
    - `round(abs(new_p − latest["p_home"]), 6) >= REVISION_MIN_MOVE`.
  - `revision_reason(report_at: datetime, changes: list[StatusChange]) -> str`: `f"new injury report {report_at IST %H:%M} IST: "` plus the first `MAX_NAMED_CHANGES` texts joined by `", "`, plus `f" and {n} more"` when more are left.

- [ ] **Step 1: Write the failing tests**
  - `test_final_due_window`: for a tip at `2026-11-06T00:00Z`:
    - `final_due` at `23:14:59Z` is False; at `23:15Z` and `23:44:59Z` it is True; at `23:45Z` it is False.
    - `still_callable` at `23:44:59Z` is True and at `23:45Z` is False.
  - `test_final_due_across_dst_change`: a tip at `2026-11-02T00:00Z` (19:00 EST on Nov 1) gives `final_due(tip, dt("2026-11-01T23:15Z"))`, and `live.slate_date(dt("2026-11-01T23:15Z")) == date(2026, 11, 1)`.
  - `test_final_window_holds_two_loop_marks`: for each tip minute 0..59 at hour 00Z, exactly 2 of the marks from `loop.next_mark(..., 15)` fall in `[tip − 45, tip − 15)`.
  - `test_chain_reads_old_lines_as_early`: a line without `call_type` is `.early`, and `.final is None`.
  - `test_chain_follows_provisional_link`: lines under `"espn:401705555"` with link `{"espn:401705555": "0042600101"}` give the key `("0042600101", "2026-12-16")` and `.game_id == "espn:401705555"`.
  - `test_status_changes_detects_new_flip_and_drop`:
    - Before: `{("DEN","Murray, Jamal"): "Questionable", ("DEN","Porter Jr., Michael"): "Out"}`. After: `{("DEN","Murray, Jamal"): "Out"}`.
    - Two changes, with texts `"Murray (DEN) out"` and `"Porter Jr. (DEN) available"`.
  - `test_status_changes_ignore_other_teams_and_case`: `"OUT"` against `"Out"` is no change, and a LAL row with teams `("DEN","BOS")` is ignored.
  - `test_unknown_status_counts_as_a_change`: Available to `"Not With Team"` gives `"Murray (DEN) not with team"`.
  - `test_before_snapshot_none_lists_every_listed_player`.
  - `test_revision_due_needs_newer_report_change_and_two_points`: latest `{"p_home": 0.61, "injury_report_at": "2026-11-05T22:30:00+00:00"}` and report `23:30Z`, with one change:
    - `new_p 0.59` gives True.
    - `0.5901` gives False.
    - A report at `22:30Z` gives False.
    - No changes gives False.
  - `test_revision_reason_text`:
    - `revision_reason(dt("2026-11-05T18:00Z"), [murray_out]) == "new injury report 23:30 IST: Murray (DEN) out"`.
    - With 5 changes, the text ends with `" and 2 more"`.
- [ ] **Step 2: Run them and confirm they fail.** Run `uv run pytest tests/test_calls.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.** No I/O; `IST = ZoneInfo("Asia/Kolkata")`.
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Commit** with the message `feat: call windows, call chains, injury status changes and the revision rule`.

---

### Task 6: The live-model flag and the two callers

**Files:**
- Create: `src/predictor/model/live_model.py`
- Modify: `src/predictor/model/live.py` (`_settings_dict` moves to `live_model.team_settings_dict`; `live` imports it)
- Test: `tests/test_live_model.py`

**Interfaces:**
- Consumes: `Stage1Predictor`; `PlayerStage1Predictor` (Task 3); `player_settings.load`; Task 5's `CallType` and `GameChain` (`predictor.model.calls`).
- Produces:
  - `LIVE_MODEL_FILE = Path("settings/live_model.json")`, relative to the state dir.
  - `ModelKind = Literal["team", "player"]`; `MODEL_VERSION: dict[str, str] = {"team": "v1", "player": "v2"}`.
  - `LiveModel(kind: ModelKind, from_slate: date | None)` (frozen), with `LiveModel.team() -> LiveModel` returning `LiveModel("team", None)`.
  - `LiveModelError(Exception)`. Messages end `"Fix state/settings/live_model.json."`.
  - `load_live_model(state_dir: Path) -> LiveModel`:
    - A missing file gives `team()`.
    - The file is `{"model": "player" | "team", "from_slate": "YYYY-MM-DD"}`. `from_slate` is required for player.
    - Anything else raises `LiveModelError`.
  - `parse_live_model(text: str) -> LiveModel`, the parser `load_live_model` uses. Task 9 reads the flag text from git.
  - `kind_for_slate(live: LiveModel, slate: date) -> ModelKind`: `"player"` iff `live.kind == "player"` and `live.from_slate <= slate`.
  - `kind_for_game(live: LiveModel, slate: date, chain: GameChain | None) -> ModelKind`: the kind whose version matches the `model_version` of the chain's first `predicted` line (a missing field means v1). A chain with no predicted line uses `kind_for_slate`.
  - `load_player_settings(path: Path = player_settings.SETTINGS_PATH) -> PlayerModelSettings`: raises `LiveModelError("the player settings include a correction model, which live calls do not support; refit without it. ...")` when `correction is not None`.
  - `CallOutcome(spread: float, p_home: float, sentence: str, terms: dict[str, float], availability: list[dict] | None, model_version: str, settings: dict, sigma: float)` (frozen).
  - `TeamCaller(con, settings: ModelSettings)` with `.call(game: GameToPredict, view: AsOfView, kind: CallType) -> CallOutcome`. `kind` is ignored, `availability` is None, `model_version` is `"v1"`, and `settings` is `team_settings_dict(settings)` (today's keys).
  - `PlayerCaller(con, settings: PlayerModelSettings)` with the same `.call`:
    - `kind == "early"` uses the `"early"` availability table; `"revision"` and `"final"` use `"final"`.
    - `availability = [{"team": t, "player": name, "status": status} for (t, name, status) in breakdown.absent]`.
    - `model_version` is `"v2"`, and `settings` is `player_settings_dict(settings)`.
  - `player_settings_dict(s) -> dict` with keys `model` (`"player"`), `k`, `margin_cap`, `season_regression`, `hca_window`, `formula`, `w`, `player_half_life_minutes`, `player_season_regression`, `home_shrinkage` (None when flat), `sigma`, `half_life`. `team_settings_dict` gains `home_shrinkage` only when not flat.
  - `make_caller(con, kind: ModelKind, settings: ModelSettings, player_settings: PlayerModelSettings | None) -> TeamCaller | PlayerCaller`. It raises `LiveModelError` when kind is player and `player_settings` is None. Tests monkeypatch it.

- [ ] **Step 1: Write the failing tests**
  - `test_missing_flag_file_means_team`.
  - `test_flag_switches_from_slate`: `from_slate 2026-11-10` gives `kind_for_slate(.., date(2026,11,9)) == "team"` and `kind_for_slate(.., date(2026,11,10)) == "player"`.
  - `test_game_keeps_model_of_its_first_call`: a chain whose early line has no `model_version`, on a slate after `from_slate`, gives `"team"`.
  - `test_bad_flag_file_raises`: `{"model": "plyer"}` raises `LiveModelError`, and `str(exc).endswith("Fix state/settings/live_model.json.")`. `{"model": "player"}` without `from_slate` also raises.
  - `test_live_refuses_settings_with_correction`.
  - `test_player_caller_uses_early_table_only_for_early`: a spy on `PlayerStage1Predictor.explain` sees `call == "early"` for kind early, and `"final"` for revision and final.
  - `test_team_caller_matches_stage1`: `spread` and `p_home` equal `Stage1Predictor.explain`, `availability is None` and `model_version == "v1"`.
  - `test_player_caller_lists_absences`: `availability == [{"team": "DEN", "player": "Jamal Murray", "status": "Out"}]`.
- [ ] **Step 2: Run them and confirm they fail.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests and the full suite.** Expected: all pass, and `live.py`'s line `settings` dict is unchanged for v1.
- [ ] **Step 5: Commit** with the message `feat: a committed flag picks the live model from a given slate; callers for both models`.

---

### Task 7: `predict_today` writes early, revision and final calls; `grade` names the graded call

**Files:**
- Modify: `src/predictor/model/live.py`
- Test: `tests/test_live_calls.py` (new), `tests/test_live_predict.py`

**Interfaces:**
- Consumes: Task 5 (all); Task 6 (`LiveModel`, `kind_for_game`, `make_caller`, `CallOutcome`).
- Produces:
  - `predict_today(con, settings, repo_dir, now, *, code_dir=None, state_version=None, code_version=None, calls: frozenset[str] = EARLY_ONLY, live_model: LiveModel = LiveModel.team(), player_settings: PlayerModelSettings | None = None) -> RunResult`
  - `RunResult` gains `revisions: int = 0`, `finals: int = 0`, `missed_finals: int = 0`, `model_versions: frozenset[str] = frozenset()`. `predicted` keeps counting early `predicted` lines only.
  - `latest_report_at(con, now) -> datetime | None`: `max(observed_at)` of `AsOfView(con, now).table("injury_reports")`.
  - `_MISSED_FINAL_REASON = "no final call was made in the 45 minutes before tip-off (no run reached it in time)"`
  - `calls_due(con, now) -> bool`: `early_call_due(now)`, or a game dated in `[slate − BACKFILL_DAYS, slate]` whose current tip is after `now − timedelta(hours=1)`.
- **Line fields.** Every line, Mac and cloud, gains these:
  - `call_type`
  - `supersedes`: the `predicted_at` of the chain's latest predicted line, or null
  - `revision_reason` (null unless set)
  - `injury_report_at`: ISO time of `latest_report_at` for a v2 call, null for v1 and for `not_predicted` lines
  - `availability`: from `CallOutcome`, null for v1
  - `model_version`: `"v1"` or `"v2"`; on a `not_predicted` line, the kind for that game
  - `settings`, from the `CallOutcome`; the existing fields stay.
- **Rules, per competitive game of today's slate with a known tip:**
  1. **Early:** today's rules, unchanged (`START_BUFFER`, TBD lines, backfill, blocked by `chain.early`), with `call_type: "early"`.
  2. **Final:** if `"final" in calls`, the chain has a predicted line, `chain.final is None` and `final_due(tip, now)`, write a `predicted` final.
     - Its `revision_reason` is set when `revision_due` would hold against `chain.latest_predicted`; otherwise null.
     - This may be written in the same run as the early call, after it.
  3. **Revision:** otherwise, if `"revision" in calls`, the game's kind is `"player"`, `chain.latest_predicted` exists, `still_callable(tip, now)` and `revision_due(...)` hold, write a revision.
     - `changes` is `status_changes(history.snapshot_at(date, latest.injury_report_at), history.snapshot_at(date, now), (away, home))`, with `history = InjuryHistory.from_view(AsOfView(con, now))`. A null `injury_report_at` means a None `before`.
     - `new_p` comes from `caller.call(..., "revision")`.
  4. **Missed final:** if `"final" in calls`, then for every game dated in `[slate − BACKFILL_DAYS, slate]` whose chain has a predicted line, has no final, and `now >= current_tip − CALL_CUTOFF`, write one `not_predicted` line with `call_type: "final"` and `_MISSED_FINAL_REASON`.
  - Each game's model is `kind_for_game(live_model, slate, chain)`. A caller is built at most once per kind per run.
  - Later lines of a chain use `chain.game_id`.
  - Market fields:
    - Early: `_market_fields` as today.
    - Revision and final: `_market_fields`. When `market_p_home` is null, the five market fields are copied from `chain.latest_predicted`.
  - Guard: `assert now < tip − CALL_CUTOFF` before any revision or final line is appended.
- **`grade`:**
  - Ties on `predicted_at` go to the later log line (`>=` in place of `>`).
  - Grade lines gain `call_type` (that of the graded line, via `call_type_of`), `early_predicted_at`, `early_p_home` and `early_correct`. These three come from `chain.early` and are null when the chain has no predicted early line. `early_correct` uses the same home-on-0.5 tie-break.
  - Everything else is unchanged.

- [ ] **Step 1: Write the failing tests** in `tests/test_live_calls.py`:
  - **Setup:** a fixture repo plus the DB from `tests/model_fixtures.py`, and `make_caller` monkeypatched to a fake whose p comes from a dict keyed by `(report_at, call kind)`. Injury rows go into `injury_status` and `injury_reports`. All times are on slate 2026-11-05.
  - `test_early_line_has_new_fields`: `call_type == "early"`, `supersedes is None`, `revision_reason is None`, `model_version == "v1"`, `injury_report_at is None` and `availability is None`.
  - `test_final_written_once_in_window_even_if_unchanged`: tip `2026-11-06T00:00Z`, early at `12:37Z`, run at `23:22Z`.
    - One final line, with `p_home` equal to the early line's, `supersedes == early["predicted_at"]` and `revision_reason is None`.
    - A run at `23:37Z` writes nothing.
  - `test_final_not_written_outside_window`: a run at `23:14Z` writes no final.
  - `test_missed_final_logged_after_cutoff`: no run in the window, then a run at `23:52Z`.
    - One line: `status == "not_predicted"`, `call_type == "final"` and `reason.startswith("no final call")`.
    - The next run writes nothing.
  - `test_revision_written_on_new_report_with_two_point_move` (v2, tip `2026-11-06T02:00Z`): early p 0.61 with report `22:30Z`; a new report at `23:30Z` has Murray (DEN) Out; fake p is 0.55; run at `23:37Z`.
    - One line with `call_type == "revision"` and `supersedes == early["predicted_at"]`.
    - `revision_reason == "new injury report 05:00 IST: Murray (DEN) out"`.
    - `injury_report_at == "2026-11-05T23:30:00+00:00"`, and `availability` lists Murray Out.
  - `test_no_duplicate_revision_for_same_report`: a second run at `23:52Z` writes nothing.
  - `test_status_flipping_back_revises_again`: a report at `23:45Z` drops Murray, the fake p returns to 0.61, and a run at `23:52Z` writes a second revision. It has `supersedes ==` the first revision's `predicted_at` and a reason ending `"Murray (DEN) available"`.
  - `test_small_move_no_revision`: a move of `0.019` writes no line.
  - `test_team_model_never_revises`: a v1 game with a changed report and a moved p writes no revision.
  - `test_revision_after_final_becomes_graded_call`: final at `01:22Z`, then a scratch report at `01:30Z` and a run at `01:37Z` (tip `02:00Z`). A revision is written, and `grade` (home won, final p 0.61, revision p 0.45) gives `call_type == "revision"`, `correct is False` and `early_correct is True`.
  - `test_late_scratch_after_cutoff_is_not_called`: report at `01:50Z`, run at `01:52Z`, so no line. The grade names the final.
  - `test_final_carries_revision_reason_when_rule_fires`.
  - `test_postponed_game_gets_no_final_under_old_date`: early on 2026-11-05, then the schedule moves the game to 2026-11-20. Runs on the evening of 11-05 write nothing for it, and the slate run on 2026-11-20 at `12:37Z` writes a new early line.
  - `test_provisional_chain_keeps_its_id`: early under `"espn:401705555"`, then a link to `"0042600101"` and a slate row under the NBA id. The final line has `game_id == "espn:401705555"`.
  - `test_market_carried_forward_when_no_fresh_odds`: the final line's five market fields equal the early line's.
  - `test_early_and_final_in_one_run_grade_the_final`: tip `13:00Z`, run `12:37Z`. Two lines with equal `predicted_at`, and `grade` gives `call_type == "final"`.
  - `test_never_sealed_inside_the_cutoff`:
    - Runs every 15 min from `12:07Z` to `03:52Z` over 4 games tipping 23:00, 23:30, 00:00 and 02:30Z, with the report changing every hour.
    - Every revision or final line has `predicted_at < tip − 15 min`, and every game has exactly one final line.
  - `test_mac_default_calls_only_early`: `calls` defaulted gives no line with `call_type != "early"`.
  - In `tests/test_live_predict.py`, the existing tests pass, with the new fields asserted on one line.
- [ ] **Step 2: Run them and confirm they fail.** Run `uv run pytest tests/test_live_calls.py tests/test_live_predict.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.** Keep the early path's structure. Build `chains(read_log(...), read_id_links(...))` once per season, and add each written line to its chain.
- [ ] **Step 4: Run the tests and the full suite.** Expected: all pass.
- [ ] **Step 5: Commit** with the message `feat: early, revision and final calls with an append-only chain per game; grades name the graded call and the early result`.

---

### Task 8: Cloud wiring: calls every run, an odds cache across iterations, shadow comparison of early calls

**Files:**
- Modify: `src/predictor/cloud/run.py` (`_predict`, `_rebuild`, `_odds`), `src/predictor/cloud/loop.py` (`_child_env`, cleanup), `src/predictor/config.py` (`odds_cache_dir`), `src/predictor/cloud/compare.py` (`_group`)
- Create: `src/predictor/cloud/odds_cache.py`
- Test: `tests/test_cloud_run.py`, `tests/test_cloud_loop.py`, `tests/test_odds_cache.py`, `tests/test_compare_shadow.py`

**Interfaces:**
- Consumes: Tasks 6 and 7.
- Produces:
  - `config.ODDS_CACHE_ENV = "PREDICTOR_ODDS_CACHE"`; `config.odds_cache_dir() -> Path | None`.
  - `odds_cache.save(cache_dir: Path, downloaded: odds.OddsDownload) -> None`:
    - Copies `raw_store.load(odds.RAW_SOURCE, key)` to `cache_dir/<key>`.
    - Writes `cache_dir/latest.json` (`archive_key`, `observed_at`, `requests_remaining`).
    - Removes older blobs.
  - `odds_cache.restore(con, cache_dir: Path, now: datetime) -> int`:
    - 0 when there is no cache or `now − observed_at > market.LIVE_MAX_AGE`.
    - Otherwise writes the blob into the run's raw store under `odds.RAW_SOURCE` (an archive-excluded source) and returns `odds.load(con, OddsDownload(...)).rows`.
  - `run._rebuild`, after `rebuild_db`: `restore` when a cache dir is set. A failure becomes a note, never a failed step.
  - `run._odds`, after a successful `odds.load`: `save`.
  - `run._predict`:
    - Due when `live.calls_due(con, now)`.
    - Reads `live_model.load_live_model(run.state_dir)`.
    - Loads player settings with `live_model.load_player_settings()` only when the slate's kind or any slate chain is v2.
    - Calls `predict_today(..., calls=ALL_CALLS, live_model=..., player_settings=...)`.
    - A `LiveModelError` gives `ok=False` with `"live model flag unusable: <message>"`. No lines are written, and v1 is never used as a silent fallback.
    - Detail: `f"slate {slate} (model {', '.join(sorted(versions)) or 'none'}): {predicted} early, {revisions} revision(s), {finals} final, {missed_finals} missed final, {not_predicted} not predicted, {skipped} duplicate(s) skipped"`.
  - `loop._child_env` sets `PREDICTOR_ODDS_CACHE=<base_dir>/odds-cache`. `run_loop` removes that dir at the end.
  - `compare._group` keeps only lines where `calls.call_type_of(line) == "early"`.

- [ ] **Step 1: Write the failing tests**
  - `test_predict_step_writes_finals_after_the_early_call` (`test_cloud_run.py`): early lines committed earlier and a clock of `23:22Z` give a final line. The commit subject is `"predictions: 2026-11-05 slate"`, and the predict detail contains `"1 final"`.
  - `test_bad_live_model_flag_fails_the_step_without_lines`: `state/settings/live_model.json` holds `{"model": "plyer"}`, so `predict.ok is False`, the season log is unchanged and the run exit is 0.
  - `test_flag_on_makes_v2_calls_on_its_slate`: flag `from_slate 2026-11-05` and a monkeypatched `make_caller` give lines with `model_version == "v2"`.
  - `test_odds_cache_round_trip` (`test_odds_cache.py`): `save`, then `restore` into a fresh DB, gives `market.live_lines(con, gid, now)` non-empty. A cache older than `LIVE_MAX_AGE` restores 0.
  - `test_odds_cache_outside_iteration_data_dir` (`test_cloud_loop.py`): the env's `PREDICTOR_ODDS_CACHE` is not under the iteration's `PREDICTOR_DATA_DIR`, and the raw archive bundle has no `odds` member after a restore.
  - `test_odds_cache_holds_no_key`: a fake key, built at runtime as `"0123456789abcdef" * 2`, appears in no file under the cache dir.
  - `test_compare_shadow_ignores_revisions_and_finals`: cloud lines with an early and a final (different p) against the Mac's early give a report that passes on p.
- [ ] **Step 2: Run them and confirm they fail.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests and the full suite.** Expected: all pass, and `STEP_NAMES` is unchanged.
- [ ] **Step 5: Commit** with the message `feat: the cloud makes revision and final calls each run; odds cached across loop iterations`.

---

### Task 9: The Mac's `predict-today` never diverges silently

**Files:**
- Modify: `src/predictor/cli.py` (`predict-today`), `src/predictor/model/live_model.py` (`read_published`)
- Test: `tests/test_predict_cli.py`

**Interfaces:**
- Produces:
  - `live_model.Published(live: LiveModel, cloud_lines: bool)` (frozen).
  - `live_model.read_published(repo_dir: Path, slate: date, season: str, runner=subprocess.run) -> Published`:
    - Runs `git fetch origin +refs/heads/main:refs/remotes/origin/main` (60 s timeout).
    - Then reads `git show origin/main:state/settings/live_model.json` through `parse_live_model`, and `git show origin/main:predictions/<season>.jsonl`.
    - `cloud_lines` is True iff a line has `game_date == slate.isoformat()` and a non-null `state_version`.
    - When the fetch fails, it uses the local `HEAD` files, and the CLI prints `"predict-today: could not reach origin; checked the local copy only."`.
- Behaviour, before any capture, odds fetch, prediction or grading:
  - If `kind_for_slate(published.live, slate) == "player"`, print `f"predict-today: from {from_slate} the live model is the player model, which only the cloud runs. This Mac made no calls and graded nothing for {slate}."` and exit 1.
  - Else, if `published.cloud_lines`, print `f"predict-today: the cloud already publishes main (it has calls for {slate}). This Mac made no calls and graded nothing."` and exit 1.
  - Otherwise, today's behaviour. Lines carry `call_type: "early"` and `model_version: "v1"` (the default `calls=EARLY_ONLY`).

- [ ] **Step 1: Write the failing tests**
  - `test_mac_refuses_when_flag_says_player`: a bare remote whose `main` has the flag from `from_slate` = today gives exit 1, the message above, an unchanged log, and no commit.
  - `test_mac_refuses_when_cloud_lines_on_origin_main`: the remote's log has a line for the slate with `state_version "abc1234"`, so exit 1.
  - `test_mac_predicts_as_today_when_flag_team`: lines carry `call_type "early"` and `model_version "v1"`.
  - `test_mac_fetch_failure_falls_back_to_local_flag`: the runner fails the fetch, the local flag says player, so exit 1 with both messages.
- [ ] **Step 2: Run them and confirm they fail.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests and the full suite.** Expected: all pass.
- [ ] **Step 5: Commit** with the message `feat: the Mac's predict-today stands down once v2 is live or the cloud publishes main`.

---

### Task 10: Site: latest call, revision note, call history, early-call record, model marker

**Files:**
- Create: `site/src/lib/calls.ts`, `site/tests/calls.test.ts`
- Modify:
  - `site/package.json` (`"test": "node --test tests/"`)
  - `site/src/lib/data.ts`, `site/src/lib/format.ts` (`TERM_LABELS`, `Terms`)
  - `site/src/components/GameRow.astro`, `Stamp.astro`, `Record.astro`, `Ledger.astro`
  - `site/src/pages/method.astro`, `site/src/data/site.ts`, `site/DESIGN.md` (honesty rules), `site/fixtures/make.mjs`

**Interfaces:**
- Produces in `calls.ts` (pure; only built-ins such as `Intl`; no imports):
  - `type CallType = 'early' | 'revision' | 'final'`; `callTypeOf(line) => line.call_type ?? 'early'`.
  - `groupCalls<T extends CallLine>(lines: T[]): Map<string, T[]>`, keyed `` `${game_id}|${game_date}` `` in log order.
  - `latestCall(calls)`: the last predicted line, or null.
  - `gradedCall(calls, grade)`: the last predicted line whose `predicted_at === grade.predicted_at`, or else `latestCall`.
  - `revisionNote(calls): string | null`:
    - Taken from the last predicted line that has a `revision_reason`, compared with the predicted line before it.
    - The text is `` `Revised ${HH:MM} IST: ${revision_reason}. Was ${prevTeam} ${prevPct}%, now ${curTeam} ${curPct}%.` ``, with each pick as `lean` in `format.ts` computes it (the team with p ≥ 0.5 on the home side, and its percentage rounded).
  - `earlyCorrect(grade, calls): boolean | null`: `grade.early_correct` when present. Otherwise derived from the first early predicted line (`p_home >= 0.5 === home_won`), or null.
  - `modelSwitchDate(lines): string | null`: the smallest `game_date` of a line with `model_version === 'v2'`.
  - `revisionCount(calls): number`: the number of revision lines.
- `data.ts`:
  - `Game` = the latest predicted call's fields, or the last line when nothing was predicted, plus:
    - `calls: CallEntry[]` (each line plus `seal` and `sealedBeforeTip`)
    - `graded: CallEntry | null`
    - `revisions: number`
    - `note: string | null`
  - `Game.seal` is the graded call's seal, else the latest call's.
  - `Day` gains `revisions: number` and `early: Tally`. `Day.seal` becomes the earliest seal among early calls.
  - `Totals` gains `early: Tally`, `v1: Tally`, `v2: Tally` and `modelSwitch: string | null`.
- Copy (exact):
  - `GameRow`:
    - `<p class="game__note">{note}</p>` when the note is set.
    - `<details class="game__history"><summary>Call history ({n})</summary>`, with one `<li>` per line: `{Early|Revision|Final|Missed final} · {HH:MM} IST ({ET}) · {pick}`, plus ` · graded` on the graded call and the commit short hash linked through `site.commit`.
  - `Stamp`: after the lead, `<span class="stamp__rev">{n} {n === 1 ? 'revision' : 'revisions'}</span>` when n > 0.
  - `Record`: below the bars,
    - `Early calls (18:00 IST): {right} of {of} right ({pct}).`
    - when `modelSwitch` is set: `Model v2 from {gameDate(modelSwitch)}: v1 {r} of {o}, v2 {r} of {o}.`
    - The headline numbers are the graded calls.
  - `Ledger`: before the switch day's row, `<tr class="table__marker"><th scope="row" colspan="7">Model v2 from {gameDate(date)}: the model also reads the official injury report</th></tr>`.
  - `TERM_LABELS` adds `['roster', 'Roster']` and `['injuries', 'Injuries']`. `Terms` keys become optional.
  - `method.astro`:
    - With no switch, the injuries paragraph is unchanged.
    - With a switch, it becomes: "Until {date} (model v1) the model did not know injuries or lineups. From {date} (model v2) it also rates players from box scores and reads the official NBA injury report. Each game gets an early call at 18:00 IST, a final call in the 45 minutes before tip-off, and a revision whenever a new injury report moves the home team's chance by 2 points or more. The record grades the last call sealed before tip-off; early calls have their own line."
  - `site.ts` gains `backtestV2: null | { games, model: {accuracy, brier, logLoss} }`, filled in Task 11.
  - DESIGN.md's honesty rule "The model does not know injuries or lineups." becomes "Model v1 does not know injuries or lineups; model v2 reads the injury report. The site says which model made each call."
  - Visual system: reuse existing tokens and classes. The note and history use `--ink-2` text sizes from `tokens.css`, and no new colours.

- [ ] **Step 1: Write the failing tests** in `site/tests/calls.test.ts` (`node:test`, `node:assert/strict`):
  - `groups three calls of one game in log order`.
  - `revisionNote`: LAL at DEN, early p 0.61 at `2026-11-05T12:37:00+00:00`, revision p 0.48 at `2026-11-05T18:15:00+00:00` with reason `"new injury report 23:30 IST: Murray (DEN) out"` gives `"Revised 23:45 IST: new injury report 23:30 IST: Murray (DEN) out. Was DEN 61%, now LAL 52%."`.
  - `revisionNote survives an unchanged final`: adding a final with the same p and no reason gives the same note.
  - `gradedCall picks the call the grade names`.
  - `earlyCorrect falls back for old grade lines`: a grade without `early_correct` and one v1 line gives `earlyCorrect === grade.correct`.
  - `modelSwitchDate`: v1 on 11-08 and v2 on 11-10 give `"2026-11-10"`; no v2 gives `null`.
  - `revisionCount counts only revisions`: lines early, revision, revision, final give 2.
- [ ] **Step 2: Run them and confirm they fail.** Run `cd site && npm test`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `calls.ts`, then `data.ts` and the components. Extend `fixtures/make.mjs`:
  - From 2026-10-19 the fixture lines are v2.
  - One game has a revision and a final; one has a missed final.
  - One has an early call and a final with the same p.
- [ ] **Step 4: Verify.**
  - `cd site && npm test` passes.
  - `node fixtures/make.mjs && SITE_DATA=fixtures npx astro build` exits 0.
  - `grep -l "Revised " dist/day/*/index.html` is non-empty, and `grep -c "Model v2 from" dist/index.html` is at least 1.
  - The real-data build `npx astro build` exits 0 (v1-only log: no marker, and no note).
  - Check the populated fixture build at 375 px wide with the existing Playwright/axe dev dependencies: no horizontal scroll and no new axe violations.
- [ ] **Step 5: Commit** with the message `feat(site): latest call with revision note and call history; early-call record; model v2 marker`.

---

### Task 11: Real-data run and switch readiness (controller, on the Mac)

- [ ] Full suite green (`exit=0`); final whole-branch review; fix wave; merge `calls-and-switch` to `main` with the owner's OK.
- [ ] `predictor evaluate-model` (run in background) on the real DB.
  - Record in the spec, under a new heading "Replay result (plan 3)":
    - flat against per-team home court;
    - the shrinkage chosen per season;
    - every team's home edge and the altitude coefficient;
    - the switch-over bar for the combined candidate against the current live model;
    - the early-call numbers and the market gap.
  - Note that the numbers now use the 15-minute report lag.
- [ ] `predictor fit-player-model`. If the bar is met, commit the v2 `player_settings.json` (with `correction: null`) and fill `site.ts` `backtestV2`. If it is not met, nothing is saved, the flag stays off, and the report says why.
- [ ] Time one `cloud-run` with a local flag file on a scratch clone of `main`. The predict step must take under 3 minutes; if not, stop and report.
- [ ] Leave `state/settings/live_model.json` absent. Turning v2 on is a hand-made commit, `state: switch the live model to v2 from <slate>`, with `{"model": "player", "from_slate": "<a future slate>"}`. It is made only with the owner's OK at that moment.
- [ ] Update memory (`project_cloud_injuries.md`).

---

## Self-review notes

- **Spec coverage:**
  - Early, revision and final calls, and "never at or after tip": Tasks 5 and 7.
  - Graded = last call before tip, plus early-call grades: Task 7.
  - New line fields: Task 7.
  - Market on revision and final calls with the odds budget unchanged: Tasks 7 and 8.
  - Calls after the builder check: provisional ids (2) in Task 7; revisions after the final (3) in Task 7; "Model v2 from <date>" (4) in Task 10; push rule (5) unchanged; ESPN against NBA box rows (6) in Task 4.
  - Home court per team, shrinkage walk-forward, altitude separate, report of each team's edge, and the same bar: Tasks 1–3.
  - Switch gated and the Mac guarded: Tasks 6, 8 and 9.
  - Site: Task 10.
  - Cut-over: excluded, as instructed.
- **Proportion:** signatures, test names and exact values only; no bodies except the shrinkage formula, which the tests alone would not pin.
