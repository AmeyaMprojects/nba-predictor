# Cloud operation, player ratings and injuries — design (2026-10-22)

Extends `2026-09-22-nba-predictor-design.md` (the main spec). Sections 7 (Live operation) and 8 (Market odds) there describe the system this replaces or extends.

## Intent

The owner publishes a verifiable NBA track record on LinkedIn and on the public site (https://ameyamprojects.github.io/nba-predictor/). Two weaknesses limit it today:

1. **The model ignores who is playing.** It rates teams only from results, so it misses injuries, rested stars, trades and summer roster moves. This is the main reason the closing market (67.9% winners over 7,227 walk-forward games) is ahead of the model (65.4%).
2. **Everything runs on the owner's Mac.** It is often asleep in the evening, so jobs fail or run late. It also cannot check injury news through the US evening, which is 23:00 to 08:00 IST.

This project fixes both, and adds calls sealed as close to tip-off as possible.

## Owner decisions (verbatim intent)

- **Late injury news:** when a new injury report changes a game after its call is sealed, the game is re-predicted and resealed with a new time. The site says plainly that the call changed because of a new injury report.
- **GitHub runs everything.** The daily jobs move off the Mac onto GitHub's servers, as part of this project.
- **Timing:** calls are made as close to tip-off as possible.
- **Switch-over bar:** the new model must beat the current model, and it should be a richer model (use more of the available information) than the current one.

## Goals

1. **Cloud operation.** A GitHub Actions workflow on the public repo runs every 15 minutes, all day (free for public repos). Each run does whatever is due: early calls, injury checks, revisions, final calls, odds, results, grading, committing and pushing, and rebuilding the site. No daily step depends on the Mac.
2. **Calls and their timing:**
   - Every game gets an **early call** at the first run at or after 18:00 IST (12:30 UTC) on its slate day. This is the same slate rule as today: the ET date of (now − 7h).
   - A game gets a **revision** whenever a newer official injury report changes any player status on either team enough to move that game's home win probability by **2 percentage points or more**, compared with its latest sealed call. Revisions stop at tip-off.
   - Every game gets a **final call** at the first run within **45 minutes before its tip-off**. The 45-minute margin is there because GitHub's scheduled runs can start 5 to 20 minutes late, so the margin keeps the final seal before tip-off. If nothing changed since the last sealed call, the final call is still written; it repeats the numbers and has `call_type: "final"`.
   - **Nothing is ever sealed at or after a game's tip-off.** A game that tips before any due call ran gets a `not_predicted` line for that call, as now.
   - **The graded call is the last call sealed before tip-off.** Earlier calls stay in the log and on the site. The site also reports how early calls and final calls each performed, so revising cannot hide anything.
3. **A richer model** that knows who is playing:
   - player ratings from box scores, point-in-time
   - availability from the official injury reports, point-in-time
   - team strength that reflects the players available
   - roster moves followed through player ratings
   - optionally a correction model on top
4. **Replay before switch-over.** The new model is replayed over 2020-21 to 2025-26 with the same walk-forward rules as today. At each game's final-call time, and separately at its early-call time, it uses only the injury report that was public at that moment.
5. **Honest site:**
   - Revisions show as a note on the game, e.g. "Revised 23:45 IST after the injury report: Murray ruled out. Was DEN 61%, now DEN 52%."
   - Each game shows its call history.
   - The record uses graded (final) calls, plus a secondary line for early calls.

## Non-goals

- No betting advice. The market never becomes a model input; the model stays an independent opinion.
- No paid data and no logins. Free sources only (main spec constraint).
- No full lineup or possession simulation. Player ratings feed the existing additive Stage 1 as terms; that is the planned upgrade path in the main spec.
- No change to the public site's look beyond what revisions and call history need.

## Anti-goals (would be a failure even if it works)

- **A revision sealed after tip-off,** or a log line edited or removed. The prediction log stays append-only.
- **Backtest leakage:** any player rating, availability probability or setting computed with information from after the moment of the call being replayed. Season-total stats, such as `PlayerEstimatedMetrics`, must never be used.
- **A model that is more complex but not better.** The new model may only replace the current one if it passes the switch-over bar below. A dropped component is reported, not hidden.
- **Silent failures in the cloud.** A run that cannot reach a source records that fact and moves on. Failures are visible in the status output, in an Actions run summary, and on the site when they affect a call.
- **Secrets leaking.** The Odds API key is stored only as a GitHub Actions secret, never in files, logs or the archive. The existing secret-scan test keeps running.

## Constraints and facts already established

**Reachability from GitHub's servers,** tested 2026-10-22 with a throwaway workflow:

| Source | Result |
|---|---|
| stats.nba.com (nba_api: results, player game logs, schedule) | Blocked: ReadTimeout after 30 s, twice |
| cdn.nba.com | 403 from both the cloud and the Mac. Unusable |
| Official injury PDFs | Work |
| ESPN JSON | Works. Injuries at `site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries`; the scoreboard and per-game summary/box score endpoints also work |
| The Odds API | Works |

**Official injury reports:**
- Same host as before, new filename pattern from 2025-12-22: `https://ak-static.cms.nba.com/referee/injury/Injury-Report_YYYY-MM-DD_HH_MM{AM|PM}.pdf`, where MM is 00, 15, 30 or 45 (a new report every 15 minutes).
- The old hourly pattern (`..._HHPM.pdf`) returns 403 after 2025-12-21.
- The PDF layout is unchanged. Parse the time from the report header, never from the filename.
- Requests need a browser-like User-Agent.
- The archive runs from 2019-12-01 to 2025-12-21 in `injury_status_raw` (109,819 rows, 1,361 snapshots). The gap from 2025-12-22 to now can be backfilled with the new pattern.

**Player box scores:**
- For history, `LeagueGameLog(player_or_team_abbreviation='P')` gives a whole season in one call (about 1.5 s, about 26k rows). Fields include MIN, PLUS_MINUS, PTS, REB and AST. There is no starter flag: infer starters as the top 5 by minutes, or skip starters.
- Date-filtered `LeagueDashPlayerStats` (`date_to_nullable`) is point-in-time.
- The history above comes from stats.nba.com, so it can only be fetched from the Mac.
- Live, ESPN's per-game summary gives player box scores.

**Existing rules that still hold:** point-in-time reads through AsOfView, raw-first archiving, UTC timestamps, the append-only log with `code_version`/`code_dirty`, `publish.py`'s safety rules (commit only `predictions/`, main only, refuse mid-operation), and the secret scan.

## How it works

### Cloud workflow

- One scheduled workflow (`live.yml`) runs every 15 minutes, with a concurrency group so two runs never overlap. It has `workflow_dispatch` for manual runs.
- Each run:
  1. Checks out the repo.
  2. Rebuilds the working database from the state files (below).
  3. Fetches what is due. The newest injury report runs every time. ESPN scores and box scores run when games have finished. Odds run when due.
  4. Archives raw responses.
  5. Updates the state files.
  6. Writes calls and grades.
  7. Commits and pushes.
- **Commits from the workflow:**
  - Prediction and grade commits keep the `predictions: ` subject prefix and the predictions-only rule.
  - State commits use a separate `state: ` prefix, and the safety check accepts that prefix only for paths under the state directory.
  - The two kinds are never mixed in one commit.
- **The site is rebuilt in the same run, or triggered explicitly.** Pushes made with `GITHUB_TOKEN` do not trigger other workflows, so `site.yml`'s push trigger will not fire for them.
- **Odds** are fetched twice per slate day: at the early call, and at the run before the slate's first final call. Each fetch costs 3 of the 500 free monthly requests. The key is a GitHub Actions secret, `ODDS_API_KEY`. The owner sets it themselves with `gh secret set`; it is never typed into chat.
- **ESPN games are linked to NBA game IDs** by ET date plus home and away teams, the same way odds lines are linked today. ESPN players are linked to NBA player IDs by name plus team, with an explicit alias table. Anything unmatched is stored and reported, never silently dropped.
- **Run visibility:** every run writes a short summary (what was due, what ran, what failed) to the Actions run summary. The existing `status` checks work on the cloud state.
- **Scheduled-workflow caveat:** GitHub disables scheduled workflows after 60 days with no repository activity. Daily commits prevent that, but the status output should warn if the last cloud run is more than 2 hours old.

### State storage

- The cloud keeps its state in the public repo under a state directory: compact parsed files such as results, schedule, player game logs, injury snapshots, odds snapshots, player ID links and model settings.
- Files are split by season, or by month where they are large, so a daily commit only touches small current files.
- The working DuckDB is rebuilt from them on each run; DuckDB itself is not committed.
- **Raw archive:** raw responses (PDFs, JSON) are archived before parsing, but not on `main`, to keep it lean. They go on a separate append-only branch or into GitHub release assets. Builder's choice, provided it is append-only and survives across runs.

### The Mac's remaining role

- **Occasional history refreshes from stats.nba.com,** for full player game logs and corrections. These are pushed as `state:` commits, run by hand or by one weekly launchd job.
- **Research and backtests.**
- **Cut-over:**
  1. The Mac's live jobs keep running while the cloud runs in shadow mode for at least 3 slate days. Shadow mode computes and writes everything to the run artifact, but does not push predictions.
  2. Mac and cloud outputs are compared.
  3. The owner switches publishing to the cloud.
  4. The Mac's live launchd jobs are uninstalled, with the owner's OK at that moment.

### The richer model

All new parts are additive terms in points, so every call stays explainable. The existing per-game sentence gains terms, e.g. "… injuries −3.1 (Murray out, Porter questionable) …".

- **Player ratings:**
  - A per-player impact rating in points per 100 possessions or per minute, from box scores and plus-minus.
  - It updates game by game in date order, with prior seasons carried forward and regressed toward the mean, so it is point-in-time by construction.
  - It follows players across trades and signings.
  - The builder may try a few formulas, e.g. a simple box-score impact estimate, a ridge-regularised plus-minus, or a blend. The walk-forward picks among them.
- **Expected minutes:**
  - Each player's expected minutes come from their recent minutes, point-in-time.
  - When players are unavailable, their minutes are redistributed to available teammates in proportion to those teammates' recent minutes.
- **Availability:**
  - From the latest official injury report at or before the moment of the call.
  - Status maps to a playing probability, estimated from the 2019-12..2025-26 archive by comparing reported status with who actually played. Starting guesses: Out 0, Doubtful about 0.2, Questionable about 0.5, Probable about 0.9, Available or not listed 1. Estimates may only use seasons before the one being replayed.
- **Two new terms:**
  - **Roster strength:** expected-minutes-weighted player ratings versus the team's normal rotation. It replaces or blends with the results-only team rating; the walk-forward picks the blend weight.
  - **Injury adjustment:** the expected loss from unavailable players after minutes redistribution.
- **Optional correction model (owner asked for a richer model):**
  - A gradient-boosted correction trained only on earlier seasons' errors, using recent form, pace, travel/rest interactions, lineup continuity and the injury and roster terms. Reported as one "model adjustment" line, as main spec Stage 2 describes.
  - It ships only if it improves the walk-forward on top of the additive model.
- **Settings and calibration:** chosen on earlier seasons only, as today (walk-forward). Sigma and calibration are re-estimated.

### Switch-over bar (owner: beat the current model; be richer)

The candidate replaces the current model only if, on the walk-forward replay of 2020-21..2025-26 at the final-call time:

1. pooled Brier score and log loss are both lower than the current model's;
2. pooled accuracy is not lower;
3. Brier score is lower in at least 4 of the 6 seasons;
4. calibration is still within 5 points in buckets of 50 or more games. The old always-pick-home comparison was a testing aid and is dropped (owner decision, 2026-10-22);
5. it includes at least the roster-strength and injury terms.

The correction model is included only if it improves the results further.

The report also shows:
- the gap to the closing market (no bar, shown honestly)
- results at the early-call time

If the bar fails, the current model stays live. The report says why, and the cloud operation still ships, using the current model.

### Prediction log and site changes

**New fields on prediction lines:**

| Field | Meaning |
|---|---|
| `call_type` | `early`, `revision` or `final` |
| `supersedes` | the `predicted_at` of the call it replaces |
| `revision_reason` | e.g. "new injury report 23:30 IST: Murray (DEN) out" |
| `injury_report_at` | the header time of the report used |
| `availability` | the per-player statuses that moved the line |

Grade lines refer to the graded call. A secondary early-call grade is either stored or derivable from the log.

**Site:**
- Each game shows its latest call, plus a revision note and its call history.
- The seal time shown for a game is the time of its graded or latest call.
- The record uses final calls, with a secondary "early calls" line.
- The day stamp shows the early-call seal plus "n revisions".
- Times stay IST first, ET second.

## Left to the builder

- the exact player-rating formula and its candidates
- the correction model's features and library (it must be free/open-source and pip-installable)
- file formats and partitioning under the state directory
- the raw archive location (separate branch or release assets)
- how the site rebuild is triggered from the live workflow
- run-summary formatting
- the order of work, provided the cloud cut-over plan and the switch-over bar are respected

## Calls made after the builder check (author's calls, flagged for the owner)

1. **Injury history depth for the replay.**
   - The stored archive holds one snapshot a day (the 5 PM ET report). Before replaying, the Mac backfills the report slots the replay actually needs for 2020-21..2025-26:
     - the latest report at or before each game's early-call time (08:30 ET);
     - the latest report at or before its final-call time (tip − 45 min).
   - That is roughly 4 slots per game day, about 5,000 PDFs, not the full 35,000. Spot check 2026-10-22: old-format hourly files still download (2023-01-15 08AM and 07PM returned 200), but some slots were never published or are gone (2021-03-10 09PM returned 403). So the backfill must tolerate holes.
   - If a needed slot is missing, the replay falls back to the latest earlier report and counts how often that happened.
2. **Schedule in the cloud.**
   - ESPN's scoreboard, looking 7 days ahead and refreshed every run, is the cloud's source of games and tip-off times. Each ESPN game is linked to the NBA game ID in the state schedule by ET date and teams.
   - A game with no NBA ID yet (NBA Cup knockouts, play-in, playoffs, reschedules) gets a provisional ID, `espn:<id>`. A later Mac schedule refresh reconciles it, and the log records the mapping. Provisional IDs are never rewritten in the log.
   - Tip-off time changes are taken from ESPN on every run.
3. **Revisions after the final call.**
   - Injury checks continue until tip-off. A material change after the final call produces another revision.
   - The graded call is always the **last call sealed before tip-off**, whatever its `call_type`. The site calls it "the last call before tip".
   - The "final call" is the scheduled tip − 45 min check, not a lock.
4. **2026-27 calls made before the cut-over.**
   - They were each game's last call before tip, so they stay in the record exactly as graded.
   - The site marks the date the model changed ("Model v2 from <date>"). The record can be read for v1 and v2 separately, and nothing is regraded or relabelled.
5. **Who pushes, and the push rule.**
   - **During shadow mode:** the cloud pushes only to a separate branch, `cloud-shadow`, for its state and would-be calls. It never pushes to `main`.
   - **After the cut-over:** the cloud is the only routine writer of `main` for `predictions:` and `state:` commits.
   - **The push rule changes in one narrow way:** before pushing, the publisher may run `git pull --rebase` only when every local unpushed commit is its own safe commit (`predictions:` or `state:` touching only its allowed paths). Anything else still refuses.
   - The Mac's history refreshes go to a branch and are merged by the cloud run or by hand, never pushed straight onto a busy `main`.
6. **ESPN versus NBA box scores.**
   - Both are kept, with the time each row was seen.
   - Ratings read NBA rows when present, otherwise ESPN rows, but always only rows seen before the call's time. So a later Mac correction changes future calls, never the inputs of a call already sealed.
   - Every sealed call records the `state:` commit it was computed from, so it can be reproduced exactly.

## Owner decision 2026-10-25: home court per team

Home-court advantage must not be one flat league-wide value; it differs by team.

- **Model:** each team's home edge is estimated separately, point-in-time, from its own home and away results. It is pulled toward the league-wide value in proportion to how little evidence the team has, so a lucky stretch cannot swing it.
- **Shrinkage strength:** chosen walk-forward on earlier seasons only, like every other setting. Infinite strength means the flat value, so per-team can only ship if it scores at least as well.
- **Altitude:** stays a separate term, so it is not counted twice.
- **Where it ships:** part of plan 3's model upgrade. It is evaluated with the same replay and switch-over bar, and the report shows each team's estimated home edge.
