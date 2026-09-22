# NBA Game Predictor — Design Spec

**Date:** 2026-09-22
**Status:** Approved for planning

## Purpose

A local, free-data NBA game prediction system that produces publishable analysis. The end goal is not private betting: it is building a credible public track record as a sports analyst on LinkedIn.

This shapes every design decision. The system optimizes for **explainability, calibration, and shareable artifacts** — in that order — rather than for raw predictive accuracy. A well-calibrated 62% model with readable explanations is more valuable here than an opaque 65% model.

## Success criteria

Staged, in order:

1. **Ship before opening night** — data spine archiving, backtest harness validated, prediction core calibrated on at least one replayed season.
2. **Match the market, explain better** — model probabilities close to market-implied, with explanations and stakes analysis the market does not provide.
3. **Attempt edge** — measure performance in the disagreement zone against closing lines. Stretch goal, explicitly not required for the project to succeed.

The system is considered working when it produces, unattended, a daily brief the user can write a LinkedIn post from, plus an auditable prediction log.

## Constraints

- **Free data sources only.** No paid API integrations, no subscriptions.
- **No LLM in the pipeline.** User has Claude Pro chat but no API key. The pipeline emits structured factual briefs; prose is written by the user or by pasting into a chat LLM. The pipeline stays deterministic and testable.
- **User does not write code.** All implementation by Claude. One-command entry points, plain-English errors, no step requiring the user to debug Python.
- **Local only.** macOS, no server, no cloud, no containers.
- **Python 3.14.** Verified 2026-09-22 that `nba_api` 1.11.4, `duckdb` 1.5.5, `pandas` 3.0.6 and `pdfplumber` all install and function on 3.14.6, including a live NBA stats API call. Environments managed with `uv`.
- **Time budget:** heavy (15-20+ hrs/week), roughly three weeks to opening night.

## Chosen approach

An **additive points-space model with a machine-learned residual correction**, structured so that a later upgrade to player-level simulation replaces one component rather than requiring a rewrite.

Rejected alternatives:

- *Feature-heavy GBM with SHAP.* Higher ceiling, but overfits readily on ~1300 games per season, produces explanations in log-odds that do not read to a general audience, and makes point-in-time discipline harder to enforce. Its strengths are retained as stage 2 of the chosen approach.
- *Player-level simulation.* Best narrative depth and native injury handling, but too large for the timeline and dependent on minutes projection that free data supports poorly. Deferred; the design keeps the path open.

## Decomposition

Five sub-projects. Each receives its own implementation plan. Order is dependency-forced.

| # | Sub-project | Required by opening night |
|---|---|---|
| 1 | Data spine | Yes |
| 2 | Backtest harness | Yes |
| 3 | Prediction core | Yes |
| 4 | Context engine | No |
| 5 | Artifact layer | Partial (charts + log) |

Sub-project 1 begins immediately, before the rest of the design is built out. News history cannot be recovered retroactively, so the RSS archiver should start running as early as possible. Injury reports, by contrast, are backfillable to ~2019-12 (see below), so that data is not at risk.

---

## 1. Data spine

### Storage

A single DuckDB file. Chosen over Postgres (no server or administration) and over SQLite (columnar storage and analytical query speed, which the backtest exercises heavily; native Parquet reads).

### Sources

| Domain | Source | Coverage | Notes |
|---|---|---|---|
| Games, box scores, play-by-play, rosters | `nba_api` | Historical + live | Rate-limited and intermittently unreliable; requires backoff |
| Historical seasons, advanced stats | Basketball-Reference (scrape) | Historical | Strict rate limits; throttled conservatively |
| Historical odds | Public historical odds datasets (2007→present) | Historical | Requires a cleaning and normalization pass |
| Live odds | The Odds API free tier | Live | 500 requests/month; budgeted to one pull per day |
| Injury reports | NBA official injury report PDFs (`ak-static.cms.nba.com`) | **2019-12 → present** | Published hourly; each snapshot is a true point-in-time observation |
| News | Team and league RSS feeds | **Forward only** | No historical archive |

**Injury report archive — verified 2026-09-22.** Contrary to the initial
assumption that no historical injury archive existed, the league's PDF
reports are retrievable at:

```
https://ak-static.cms.nba.com/referee/injury/Injury-Report_YYYY-MM-DD_HHPM.pdf
```

Confirmed behaviour:

- Published **hourly**; all 24 hourly slots resolve for an in-season date.
- Archive extends back to approximately **2019-12** (2019-12-10 resolves;
  2018-12-11 returns 403).
- Each PDF's header line carries its own publication timestamp
  (`Injury Report: 01/15/25 05:30 PM`), which is authoritative for
  `observed_at` and may differ from the filename hour.
- Content is a positional table: `GameDate GameTime Matchup Team
  PlayerName CurrentStatus Reason`, with group columns populated only on
  the first row of each game/team block and `Reason` wrapping across
  lines. Parsing must use word x-coordinates rather than text splitting,
  because extracted text drops intra-field spaces (`NewYorkKnicks`).

This is a material improvement: roughly six seasons of **genuine**
point-in-time injury data are available, rather than hindsight
reconstruction.

### Raw-first ingestion

Every API response and scraped page is written to disk verbatim before parsing. Parser bugs become re-parsable rather than re-fetchable, and irreplaceable forward-only data is never lost to a parsing error.

### Point-in-time correctness

**The core invariant.** Every row carries an `observed_at` timestamp recording when the fact became *knowable*, distinct from when the underlying event occurred.

Features are never read from tables directly. All feature access goes through a single accessor that takes an `as_of` timestamp and hard-filters any row with `observed_at > as_of`. One chokepoint, enforced in one place, tested aggressively.

This is the decision that determines whether the backtest means anything, and it cannot be retrofitted. It is therefore part of the foundation rather than the model.

### Reconstructed data

For seasons predating the injury-report archive (before ~2019-12), injury state is reconstructed from DNPs and lineup data. Every such row is flagged `reconstructed = true`. All backtests report metrics twice — including and excluding reconstructed rows — so hindsight contamination is visible rather than hidden.

Given that genuine point-in-time injury data covers 2019-12 onward, the default backtest window is restricted to that range and reconstruction is treated as an optional extension rather than a core dependency.

---

## 2. Backtest harness

Built **before** the prediction core, so that leakage cannot be introduced unnoticed.

### Walk-forward replay

Games are replayed in strict chronological order. For each game the harness supplies an `as_of` timestamp set to tip-off minus a buffer; the model may only observe what the accessor returns at that timestamp. Retraining occurs on a schedule inside the replay: train on all data before date X, predict the following window, advance.

Random train/test splits are prohibited. They are the standard cause of backtests that look excellent and fail live.

### Metrics

Priority order:

1. **Calibration** — reliability diagram and Brier score. Highest priority: a calibrated model is publishable and honest, while an overconfident one destroys credibility the first time a stated near-certainty loses.
2. **Accuracy against a baseline ladder** — always-pick-home (~57-58%), Elo-only, market-implied. Failing to beat always-pick-home indicates a bug, not a finding.
3. **Versus market** — frequency of beating the closing line, and calibration specifically within the disagreement zone, where any genuine edge would reside.

### Self-testing

The harness is deliberately attacked. Tests inject a future-leaking feature (final score) and assert that the as-of guard rejects it and the run fails. These remain permanent regression tests so that any future feature which accidentally leaks trips the same wire.

### Sequencing

First target is a single season replayed end to end. Multi-season replay follows only once that run is clean.

### Output

Backtest results are also the pre-season launch content: calibration curve, accuracy against baselines, and an explicit accounting of the model's limitations — published before any live prediction.

---

## 3. Prediction core

### Stage 1 — additive points model

Team strength as an Elo rating updated on margin of victory, with between-season regression toward the mean.

Adjustments, all expressed in points:

- Home court advantage — estimated from data, not assumed constant
- Rest — days off, back-to-backs, third game in four nights
- Travel — distance and time zones crossed
- Altitude — Denver, Utah
- Injuries — sum of absent players' point impact, scaled by minutes redistribution

Per-player point impact is derived from free box-score and advanced statistics. **This component is the upgrade path to player-level simulation:** it exists from day one as a scalar, and a future lineup-level simulation replaces this one component rather than the system.

Stage 1 output is a spread plus a directly publishable sentence, e.g. `Denver -4.2 rating, -1.5 Murray out, +2.4 home, +0.8 rest → -5.9`.

### Stage 2 — residual model

A gradient-boosted model trained on stage 1's errors. Features include rolling form, pace and style matchups, shooting variance regression, lineup continuity, and interactions stage 1 cannot express.

Output is a correction in points, reported as a single "model adjustment" line plus its top contributing drivers, keeping the overall explanation readable despite this stage being non-additive.

### Stage 3 — calibration

Spread converts to win probability via a normal CDF using an empirically estimated game-to-game standard deviation (~13 points), then is calibrated on held-out data so stated probabilities are honest.

### Two outputs, kept separate

- **Pure model** — the public track record. Uncontaminated by market information, so the record reflects the user's own work.
- **Market-blended** — model combined with the closing line. Usually the superior forecast.

Both are logged; only the pure model feeds the public record. Conflating them would claim the market's skill as the user's own. The divergence between them is itself recurring content.

### Explainability

Because stage 1 is additive, the explanation *is* the model rather than a post-hoc approximation of it. Explanations are stated in points, the unit in which basketball is actually discussed. This is the primary reason this approach was selected.

---

## 4. Context engine

Separate from the prediction model. Answers "why does this game matter" and "what happens if."

### Season Monte Carlo

Simulates all remaining games many times using model probabilities. Produces playoff odds, seed distributions, play-in odds, and lottery odds. Re-run nightly as ratings move.

### Game leverage

For each game, the season simulation runs twice — once conditioned on a home win, once on an away win. The difference in each team's playoff and seeding odds quantifies the stakes:

> Thunder win → 1-seed odds 61% → 68%. Lose → 61% → 53%. Fifteen points of seeding swing, highest on the slate.

This ranks every game by how much it actually matters, driving both post content and the selection of which game receives the deep dive.

**Tiebreaker limitation:** NBA tiebreakers are intricate and cascading (head-to-head, division, conference record). The first version handles common cases and explicitly flags scenarios that hinge on unmodeled tiebreak rules rather than producing an unfounded answer.

### Storyline scanners

Rule-based passes emitting structured facts: active streaks, revenge games, first meeting since a trade, coach facing a former team, milestone watch, notable head-to-head history, clinching and elimination scenarios. Inexpensive to implement, and they supply the human hooks that make numeric analysis readable.

### Key moments

**Pre-game:** what to watch. Largest matchup edges, the factors the prediction is most sensitive to, and what would have to change for the underdog to win. Derived by perturbing model inputs and observing the response.

**Post-game:** what decided it. In-game win probability computed from play-by-play, with the largest swings identified. Closes the loop on the pre-game post.

All output is structured facts with numbers attached, formatted for the user to write from.

---

## 5. Artifact layer

### Briefs

- **Daily slate brief** — one markdown file per game day. Per game: prediction and win probability, additive factor breakdown, market comparison, leverage score, storylines, and watch points. Scannable in two minutes.
- **Deep-dive brief** — the highest-leverage game, expanded. Built to be LLM-legible: labeled sections, explicit numbers, minimal prose. This is the file pasted into Claude Pro.

### Charts

Matplotlib to PNG. Fixed palette and typography so all published output is visually recognizable. LinkedIn-native dimensions.

Five recurring formats:

1. **Factor waterfall** — the additive breakdown as a cascading bar chart with the market line marked. The signature visual: it exists because of the additive model choice, is readable by non-technical audiences, and is essentially absent from mainstream sports media.
2. Slate leverage ranking
3. Calibration curve
4. Track record over time against baselines
5. Model vs market scatter

### Track record

Predictions append to a log — timestamped, never edited — and the repository commits daily. Pushing the repository publicly makes the record independently auditable: anyone can verify from git history that a call was made before tip-off.

For an analyst brand this is worth more than any individual correct prediction, costs nothing, and makes quiet revision of a bad call impossible — which is the point.

### Post drafts

Structured rather than prose: angle, supporting numbers, suggested chart. The user writes the words. The pipeline remains deterministic and the voice remains the user's.

---

## 6. Operations, failure handling, testing

### CLI

`predictor setup`, `predictor ingest`, `predictor daily`, `predictor backtest`, `predictor status`.

### Schedule

Two runs daily via launchd (chosen over cron for its handling of a sleeping laptop):

- **Morning** — refresh data, produce a preliminary slate.
- **Evening** — run after the league injury report publishes (~5:30pm ET), produce the final brief.

Predictions made before the injury report are materially worse; the evening brief is the one intended for posting.

### Failure handling

**Governing rule: never silently publish wrong data.**

Because the user does not read logs, failures must surface in the artifact the user actually opens. Every brief opens with a data-health header stating what is fresh, what is stale, and what failed. If a critical input is missing — injury report absent, odds stale — the brief refuses to emit a confident prediction for affected games and states why.

A wrong public call caused by silently stale data is the worst outcome this system can produce, worse than producing no call. Failure modes are therefore loud and difficult to ignore.

Specific degradations:

| Failure | Response |
|---|---|
| NBA API unavailable or flaky | Retry with backoff, fall back to cache, mark data stale in brief |
| Scrape structure changed | Ingest validation fails with a plain-English message naming the source |
| Odds quota exhausted | Prevented by per-day budgeting; degrades to no market comparison rather than failing |
| Injury report missing | Affected games marked; no confident prediction emitted |

`predictor status` reports system health in plain English — what is fresh, what is broken, what to do about it.

### Testing

Test-driven throughout, with effort concentrated where correctness is load-bearing:

- **As-of guard and leakage tests** receive the most attention. A leak silently invalidates everything downstream, and a public track record built on a leaky backtest is the one failure that damages the brand permanently.
- Unit tests for each adjustment term.
- Golden-file tests for brief generation.
- End-to-end integration test against a frozen data slice, so the suite never touches the network and never flakes.

### Reproducibility

Pinned dependencies, seeded simulations, and model version recorded with every prediction — so that a prediction from months earlier can be explained rather than merely displayed.

---

## Open items for planning

- Exact seasons to include in training and backtest (start with one season replay, expand once clean).
- Which historical odds dataset to adopt, pending a quality assessment during implementation of sub-project 1.
- Chart visual identity — palette and typography to be chosen during sub-project 5.
