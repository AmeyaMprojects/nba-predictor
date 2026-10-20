# Sealed before tip-off — design

## Brief (Phase 0)

- **Subject:** an independent NBA prediction model built by Ameya Mhatre from free data only. It predicts every regular-season and playoff game once a day, before tip-off, and publishes each prediction as a git commit to the public repo AmeyaMprojects/nba-predictor.
- **Audience:** LinkedIn followers (sports-analytics readers, recruiters, fellow analysts) and basketball fans who click through from a post.
- **Job:** make the track record verifiable and readable in seconds. A visitor should be able to see tonight's calls, the running record, and the proof that each call came before the game.
- **Language:** English. Times are shown in India time (IST) first, with US Eastern as the secondary label (owner decision 2026-10-21). Game dates stay the league's US date; each tip-off also shows its India date.
- **Honesty rules (from the project spec):**
  - Never claim the model beats the market. Over the 2020-21 to 2025-26 walk-forward test the market picked 67.9% and the model 65.4% (7,227 games).
  - The model does not know injuries or lineups.
  - Early-season records are small samples, and the page says so.
  - No betting advice.
- **Assumptions:**
  - Owner name "Ameya Mhatre" (from the repo account), marked placeholder in `site.ts`.
  - LinkedIn URL is a placeholder.
  - No photos exist and none will be used: this is a data site; the imagery is the data.

## Concept (Phase 1)

Three candidates:

| | Idea | Anchor | Signature | O/F/Fe/P |
|---|---|---|---|---|
| A. Material (version control) | Every call is a sealed entry: stamped with its commit hash and time, it sits on a timeline that runs seal, then tip-off, then final buzzer. The final score is what unseals it | predictions really are public git commits made before tip-off | the seal-to-buzzer timeline with a live "now" marker crawling toward tip | 4 / 5 / 4 / 2 |
| B. Formal (the court line) | Each game is drawn as the line between two baskets; the model's dot and the market's ring sit where each thinks the game lands, and after the buzzer the ball settles at the winner's end | home/away + two probabilities per game | the ball rolling to the winner's basket | 4 / 5 / 5 / 2 |
| C. Place (box score agate) | The site is a newspaper sports page in agate type | NBA box-score tradition | columns of agate filling in | 2 / 3 / 5 / 4 |

**Pick A**, because it is the only one built on what makes this project worth trusting: the calls are timestamped before the games. From B, take the per-game graphic, the court line, as the detail inside each sealed entry.

**Concept Card**
```
Idea:        Every call is sealed before tip-off and unsealed by the final score.
Metaphor:    a notary's register crossed with a version-control log.
Signature:   the seal-to-buzzer timeline: seal stamp, then tip-off, then final, with a live "now" marker
             crawling toward tip on tonight's slate; graded games show the ball settled at the winner's end
             of each court line (motion.md: scrubbed/progress timeline + stamp micro-interaction).
Emotion:     first 3 seconds: "these were called in advance, and here is the proof".
Motion verb: stamp and settle: entries land like a stamp (quick press, tiny overshoot, settle);
             markers slide along rails; nothing floats.
Screenshot:  tonight's slate: three court lines with orange model dots and ink market rings,
             a big "Sealed 18:10 IST · 6h 20m before first tip" stamp, and the now-marker on the rail.
```

## System (Phase 2)

**Palette**

| Role | Hex |
|---|---|
| ground (court paint, cool white) | `#F2F3EF` |
| ink | `#0F1113` |
| muted | `#5C6169` (AA on ground) |
| line | `#D5D7D2` |
| accent: model, ball orange | `#C8461A` (AA for large text; use ink for small text, orange only for marks and large numerals) |
| market: slate | `#33445E` |
| curtain | `#0F1113` |

**Type**
- Display: Archivo at width 62 to 75 (condensed), weight 800, for team codes, big numerals and the stamp. Tracking is negative and set per size.
- Text: Archivo at width 100, weight 400/500, 17px base, 1.5 leading.
- Data: JetBrains Mono 400/500 for timestamps, hashes, percentages in tables, and labels.

**Grid:** 12 columns, 24px gutter on desktop; 4 columns, 16px on phone. Team codes hang on the left column edge. Section rhythm varies: hero is full-bleed tall, ledger is dense, method is a narrow reading column.

**Shape:** square corners everywhere (radius 0). Only the model dot and the market ring are round.

**Motion language:**
- One ease: `expo.out` for settles, with a short `back.out(1.4)` only for the stamp press.
- Durations: stamp 0.35s, rails 0.9s, page transition 0.6s.
- The stamp scales from 1.06 to 1 with a 1px shadow collapse.
- Under reduced motion: no smooth scroll, no scrub, all markers at final positions.

**Idle state:** tonight's now-marker advances in real time (the truthful clock), and the countdown to next tip ticks on the GSAP ticker. On days with no live slate the page is still at rest, and the hover stamp carries the life.

**Composition plan**

```
HOME
[corner: wordmark top-left · nav top-right: Tonight / Record / Ledger / Method]
HERO (tonight's slate, full-bleed, asymmetric 8/4)
  STAMP: "Sealed 18:10 IST  ·  6h 20m before first tip"  commit e14df79 ->
  rail: seal ●──────── now ▸ ─────── tip 1 ── tip 2 ── tip 3 ─── finals
  game rows: BOS  ·──────────○───●───·  DET     model 55% DET · market 53% BOS
RECORD (split 5/7): left giant numerals "2–1"; right three bars model/home/market + sample-size note
LEDGER (dense table, full width): date · games · model right · market right · sealed hh:mm · commit
HOW IT IS SEALED (narrow column + three-step rail, a real sequence so numbering is allowed)
FOOTER: repo link, raw files, method link, honest one-liner

DAY PAGE /day/YYYY-MM-DD/
  hero: date set huge in condensed display + stamp
  per game: court line, sentence from the log, term breakdown as a small stacked bar
  (rating / home / rest / travel / altitude), market line + label, result + grade, commit link

METHOD /method/
  narrow reading column; one table: walk-forward 2020-21..2025-26 model vs market vs home;
  "what it does not know"; "how to verify" with git commands

404: an unsealed entry: "No call was sealed for this page."
```

**Genericness check:**
- A generic model would build a dark sportsbook dashboard with neon accents and card grids. Instead this site is cool-white court paint, a ledger and stamps, with no cards.
- A generic model would also show "accuracy: 66%" as a hero metric. Here the hero is the proof (seal time and commit), and the record comes second, with its sample size beside it.

## Data

- Build time reads `../predictions/*.jsonl` (prediction lines) and `*-grades.jsonl` (grade lines).
- Seal = the git commit that added each prediction line, found with `git blame --line-porcelain` (commit hash plus committer time). The workflow checks out full history.
- `predicted_at` is shown as the model's own clock.
- **Spread convention:**
  - Log fields: model `spread` is the home margin (positive means home favoured); `market_spread` is the betting home line (negative means home favoured).
  - Display both as "TEAM by X", computing the market's as `-market_spread` so they share one convention. Never print raw signed spreads.
- `not_predicted` lines are shown honestly as "no call: <reason>".

## Build notes

- **Idle state, as built:** the hero's "first tip in" clock ticks every second and the now-marker on the rail advances with the real clock. Both are truthful and deliberately small; `shoot.mjs` reports `idleMotion: false` because the per-second change is below its pixel threshold. No decorative idle motion was added to pass the check.
- **OG image** is a static composition (`public/og.png`), not a capture of the live hero, so a shared link never shows a stale seal time.
