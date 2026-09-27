# Schedule Source (Sub-project 2.5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ingest the NBA league schedule (`ScheduleLeagueV2`) daily as a point-in-time source, and make it the backtest harness's only runtime source of tip-off times.

**Architecture:** A new source module `src/predictor/sources/schedule.py` follows the existing raw-first shape: fetch JSON, archive it gzipped to `raw_store`, parse the archived bytes, write rows to a new `schedule_raw` point-in-time table. `backtest/tipoff.py` keeps its public interface (`tipoff_index`, `resolve_tipoff`, `parse_game_time`) but builds its index from the schedule instead of injury reports. The injury-PDF tip-off path survives only as an archive-wide cross-check test. A second launchd job runs `predictor ingest-schedule` daily.

**Tech Stack:** Python 3.14, `nba_api` (`scheduleleaguev2`), DuckDB, typer, tenacity, pytest, launchd. Run everything with `uv run`.

**Spec:** `docs/superpowers/specs/2026-09-22-nba-predictor-design.md` — section **1.1 Schedule source**.

## Global Constraints

- Free data sources only; no paid API; no LLM call anywhere in the pipeline.
- The user does not write code. Every CLI message is plain English and says what to do next. Every failure path exits non-zero.
- Physical `_raw` table names appear as string literals ONLY in `src/predictor/db.py` and `src/predictor/asof.py`. Everything else in `src/` resolves them through `db.POINT_IN_TIME_TABLES[...]` (enforced by `tests/test_leakage.py::test_no_physical_table_name_appears_outside_db_and_asof`). Tests may spell them.
- Every point-in-time timestamp written to DuckDB passes through `db.require_utc` first.
- Raw-first: every fetched payload is archived to `raw_store` BEFORE it is parsed, and the parser reads the archived bytes.
- Scores, game status, team win/loss records, and points leaders from the schedule endpoint are never parsed, stored, or exposed. A field that is not stored cannot leak.
- Tip-off establishes cutoffs; it is never a predictor feature.
- Tests must never write to the real archive `data/predictor.duckdb` (`tests/conftest.py` + the rail in `db.connect`). Real-archive tests open it with `duckdb.connect(..., read_only=True)` and skip (not fail) when it is locked or lacks the schedule table.
- `data/raw/` and `data/predictor.duckdb` are irreplaceable. Never delete them. Back up the database before the first real write in this plan.
- Baseline before this plan: `uv run pytest -q` → **347 passed**.

## Evidence gathered while writing this plan (2026-09-27)

Measured against the real archive and live `ScheduleLeagueV2` payloads for 2019-20 through 2026-27. These numbers set the expected outputs below.

- **Coverage:** all 8,289 regular-season games in `games_raw` appear in the schedule. Game date and home/away agree on **all 8,289**. There are 0 date mismatches and 0 team mismatches, so the spec's "log loudly" rescheduling rule is a guard and does not fire on today's data.
- **Cross-check with injury PDFs, all seasons.** The spec said "zero disagreements," but that was measured on 2023-24 only. Across all 7,200 games where both sources exist, the schedule tip-off equals the old PDF resolver's time (minimum across vintages) for 7,195 games. The remaining 5:
  - 2 games (`0022000001`, `0022200001`) differ by 1 minute. Opening-night ceremonial `:01` times appear in the schedule, while the PDFs round.
  - 3 games (`0021900701`, `0022000206`, `0022400624`) have a day-before PDF with an earlier time. The **game-day PDF, filed before either time**, shows the schedule's later time. The games were moved later. The schedule is right, and the stale earlier vintage is only conservative.
  - The correct invariant, which holds for **all 7,200**, is below. It is the cross-check this plan ships. For each game, the schedule tip-off matches (within 60s) the most recent PDF vintage filed before that tip-off. If no PDF was filed before tip-off, it matches the earliest filing. Otherwise it matches the minimum across all vintages; this last case covers the 2 games moved *earlier*, `0022000834` and `0022200161`.
  - Consequence: the old test `test_archive_wide_cutoff_is_strictly_before_every_recorded_tipoff_vintage` would FAIL after the switch, for those 3 moved-later games. It is replaced in Task 6, not relaxed.
- **`isNeutral` is unreliable before 2024-25.** It is `false` for every game in 2019-20 through 2023-24, including Paris, Mexico City and Las Vegas NBA Cup games. The plan therefore stores both columns:
  - `is_neutral_reported` (the league's flag)
  - `is_neutral` (derived): reported, OR the arena's (city, state) differs from the home team's usual regular-season home venue in that payload.

  Derived counts per season (all game types): 2019-20 169 (166 are Orlando bubble games), 2020-21 0, 2021-22 0, 2022-23 4, 2023-24 7, 2024-25 8, 2025-26 8, 2026-27 6. Known, accepted quirks:
  - San Antonio's Austin home games count as neutral. They have a reduced home edge, so this is arguably right.
  - Orlando's own bubble "home" games are not caught, because the city matches. That affects at most ~4 games.
- **Forward schedule (2026-27):** 1,274 games. 7 have no teams yet: 6 NBA Cup knockout slots `00226012xx` and 1 Cup final `0062600001`. Their time is `TBD` and shows as 00:00 ET. They must not become a tip-off: a 00:00 placeholder would win the minimum across vintages forever.
- **Payload size:** ~4.7 MB JSON per season and ~390 KB gzipped, so daily archiving costs ~140 MB/year.

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `src/predictor/db.py` | modify | `schedule_raw` DDL, `"schedule"` in `POINT_IN_TIME_TABLES`, `connect_with_retry` |
| `src/predictor/asof.py` | modify | default `latest()` key for `"schedule"` |
| `src/predictor/config.py` | modify | `season_label`, `previous_season_label` |
| `src/predictor/status.py` | modify | schedule freshness threshold and advice; use `config.season_label` |
| `src/predictor/sources/schedule.py` | create | fetch, archive, parse, ingest, compare-with-games |
| `src/predictor/cli.py` | modify | `ingest-schedule` command; `poll-news` uses `connect_with_retry` |
| `src/predictor/backtest/tipoff.py` | modify | index built from the schedule |
| `scripts/com.predictor.schedule.plist` | create | daily 10:30 launchd job |
| `scripts/install_schedule.sh` | modify | install both launchd jobs |
| `tests/schedule_rows.py` | create | test helper: insert one schedule row |
| `tests/real_archive.py` | create | test helper: open the real archive read-only or skip |
| `tests/test_config.py`, `tests/test_db.py`, `tests/test_status.py` | modify | new behaviour |
| `tests/test_schedule_parse.py`, `tests/test_schedule_ingest.py`, `tests/test_schedule_cli.py`, `tests/test_launchd.py`, `tests/test_tipoff_crosscheck.py` | create | new behaviour |
| `tests/test_tipoff.py` | rewrite | schedule-sourced index; archive-wide invariants |
| `tests/test_replay.py`, `tests/test_backtest_leakage.py` | modify | fixtures seed tip-offs via schedule rows |
| `docs/superpowers/specs/2026-09-22-nba-predictor-design.md` | modify | correct the cross-validation claim; neutral-site finding |

---

### Task 1: Schema, registration, season labels, status

**Files:**
- Modify: `src/predictor/db.py`, `src/predictor/asof.py`, `src/predictor/config.py`, `src/predictor/status.py`
- Test: `tests/test_config.py`, `tests/test_db.py`, `tests/test_status.py`

**Interfaces:**
- Produces: `db.POINT_IN_TIME_TABLES["schedule"] == "schedule_raw"`, with columns `game_id, season, game_date, tip_off_utc, home_team, away_team, arena_name, arena_city, arena_state, is_neutral_reported, is_neutral, observed_at`, PK `(game_id, observed_at)`.
- Produces: `config.season_label(now: datetime) -> str` and `config.previous_season_label(label: str) -> str`.
- Produces: `status.STALENESS_HOURS["schedule"] == 36`.

- [ ] **Step 0: Branch**

```bash
cd /Users/meya/projects/predictor
git checkout -b schedule-source
```

- [ ] **Step 1: Write failing tests**

Append to `tests/test_config.py`:

```python
from datetime import UTC, datetime

from predictor.config import previous_season_label, season_label


def test_season_label_from_july_is_the_upcoming_season():
    assert season_label(datetime(2026, 9, 27, tzinfo=UTC)) == "2026-27"
    assert season_label(datetime(2026, 7, 1, tzinfo=UTC)) == "2026-27"


def test_season_label_before_july_is_the_season_in_progress():
    assert season_label(datetime(2027, 3, 1, tzinfo=UTC)) == "2026-27"
    assert season_label(datetime(2026, 6, 30, tzinfo=UTC)) == "2025-26"


def test_season_label_across_a_century_boundary():
    assert season_label(datetime(2099, 10, 1, tzinfo=UTC)) == "2099-00"


def test_previous_season_label():
    assert previous_season_label("2026-27") == "2025-26"
    assert previous_season_label("2000-01") == "1999-00"
```

Append to `tests/test_db.py`:

```python
def test_schedule_table_is_registered_and_has_no_outcome_columns(con):
    assert db.POINT_IN_TIME_TABLES["schedule"] == "schedule_raw"
    cols = {r[0] for r in con.execute("DESCRIBE schedule_raw").fetchall()}
    assert cols == {
        "game_id", "season", "game_date", "tip_off_utc", "home_team",
        "away_team", "arena_name", "arena_city", "arena_state",
        "is_neutral_reported", "is_neutral", "observed_at",
    }
    # Spec 1.1: scores are never ingested -- a column that does not exist
    # cannot leak.
    for forbidden in ("home_points", "away_points", "score", "status", "wins", "losses"):
        assert not any(forbidden in c for c in cols), forbidden
```

In `tests/test_status.py`, replace both four-name sets with five names. Change:

```python
    assert names == {"games", "injury_status", "odds_snapshots", "news_items"}
```

to:

```python
    assert names == {"games", "injury_status", "odds_snapshots", "news_items", "schedule"}
```

and change:

```python
    for name in ("games", "injury_status", "odds_snapshots", "news_items"):
```

to:

```python
    for name in ("games", "injury_status", "odds_snapshots", "news_items", "schedule"):
```

Then append to `tests/test_status.py`:

```python
def test_stale_schedule_advice_names_the_command_and_the_launchd_job(con):
    health = {h.name: h for h in status.check_sources(con, NOW)}
    advice = health["schedule"].advice
    assert "predictor ingest-schedule" in advice
    assert "com.predictor.schedule" in advice


def test_schedule_fresh_within_a_day_is_not_stale(con):
    con.execute(
        "INSERT INTO schedule_raw (game_id, season, game_date, tip_off_utc,"
        " home_team, away_team, is_neutral_reported, is_neutral, observed_at)"
        " VALUES ('0022400561', '2024-25', DATE '2025-01-15', NULL, 'PHI', 'NYK',"
        " FALSE, FALSE, ?)",
        [NOW - timedelta(hours=20)],
    )
    health = {h.name: h for h in status.check_sources(con, NOW)}
    assert health["schedule"].stale is False
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_config.py tests/test_db.py tests/test_status.py -q`
Expected: FAIL. You should see `ImportError: cannot import name 'previous_season_label'`, and a missing `schedule` key / table.

- [ ] **Step 3: Implement**

`src/predictor/config.py`: add `from datetime import datetime` to the imports, then append:

```python
def season_label(now: datetime) -> str:
    """Season string (e.g. "2026-27") that `now` belongs to, for defaults.

    NBA seasons start in October and are labeled by their two years. From
    July onward the upcoming season is the relevant one; before July, the
    season already in progress (started the previous October) is.
    """
    start_year = now.year if now.month >= 7 else now.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def previous_season_label(label: str) -> str:
    """"2026-27" -> "2025-26"."""
    start_year = int(label[:4]) - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"
```

`src/predictor/db.py`: add `"schedule": "schedule_raw",` as the last entry of the `POINT_IN_TIME_TABLES` dict. Then add this block to `_SCHEMA`, immediately before `CREATE TABLE IF NOT EXISTS ingest_runs`:

```sql
-- The league schedule (nba_api ScheduleLeagueV2), one row per game per
-- fetch. observed_at is the real fetch time, so daily fetches accumulate
-- genuine schedule vintages from 2026-09-27 onward; rows for games already
-- played at fetch time are post-hoc. game_date is the Eastern calendar
-- date. tip_off_utc is NULL when the league lists the time as TBD -- a
-- placeholder must never become a tip-off, because the harness takes the
-- MINIMUM across vintages and a 00:00 placeholder would win forever.
-- is_neutral_reported is the league's own flag, which is false for every
-- game before 2024-25 (Paris, Mexico City and Las Vegas included);
-- is_neutral also marks a game whose arena differs from the home team's
-- usual regular-season venue (see sources/schedule.py). Scores, game
-- status and team records are deliberately NOT columns: the endpoint
-- carries them, and a field that is not stored cannot leak.
CREATE TABLE IF NOT EXISTS schedule_raw (
    game_id             VARCHAR NOT NULL,
    season              VARCHAR NOT NULL,
    game_date           DATE NOT NULL,
    tip_off_utc         TIMESTAMP WITH TIME ZONE,
    home_team           VARCHAR NOT NULL,
    away_team           VARCHAR NOT NULL,
    arena_name          VARCHAR,
    arena_city          VARCHAR,
    arena_state         VARCHAR,
    is_neutral_reported BOOLEAN NOT NULL,
    is_neutral          BOOLEAN NOT NULL,
    observed_at         TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (game_id, observed_at)
);

```

`src/predictor/asof.py`: add `"schedule": ("game_id",),` to `_DEFAULT_LATEST_KEY`.

`src/predictor/status.py`:
1. Delete the whole `_next_season_label` function.
2. Add `from predictor.config import season_label` beside the existing `from predictor.db import POINT_IN_TIME_TABLES`.
3. In `_advice`, change `_next_season_label(now)` to `season_label(now)`.
4. Add to `STALENESS_HOURS`:

```python
    # Fetched once a day at 10:30 local (scripts/com.predictor.schedule.plist).
    # One missed run of slack before alarming, same as injury_status.
    "schedule": 36,
```

5. Add to `_advice`, before the final `return ""`:

```python
    if name == "schedule":
        return (
            "Run: predictor ingest-schedule, and confirm the launchd agent "
            "is loaded (launchctl list | grep com.predictor.schedule). Each "
            "missed day is a day of schedule history that cannot be "
            "recaptured later."
        )
```

6. In the module docstring's bullet list, add after the `news_items` bullet:

```
- schedule: fetched once a day by its own launchd job. A missed day loses
  that day's schedule vintage (when a game moved, and when that became
  knowable) for good, so it gets the same one-missed-run threshold as
  injury_status.
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest -q`
Expected: all pass; the total is 347 + the new tests. `test_point_in_time_tables_matches_schema_exactly`, `test_default_latest_key_covers_every_point_in_time_table` and the generic leakage tests cover the new table automatically and must pass unmodified.

- [ ] **Step 5: Commit**

```bash
git add src/predictor/db.py src/predictor/asof.py src/predictor/config.py src/predictor/status.py tests/test_config.py tests/test_db.py tests/test_status.py
git commit -m "feat: add the schedule point-in-time table and its freshness check

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Schedule parser

**Files:**
- Create: `src/predictor/sources/schedule.py`
- Test: `tests/test_schedule_parse.py`

**Interfaces:**
- Produces: `schedule.ScheduleRow` (frozen dataclass: `game_id: str, season: str, game_date: date, tip_off_utc: datetime | None, home_team: str, away_team: str, arena_name: str | None, arena_city: str | None, arena_state: str | None, is_neutral_reported: bool, is_neutral: bool`).
- Produces: `schedule.ParseResult` (frozen: `rows: list[ScheduleRow], no_tipoff: list[str], undetermined: list[str]`).
- Produces: `schedule.parse_schedule(payload: bytes, season: str) -> ParseResult`, which raises `ValueError` with a plain-English message on a malformed payload or a season mismatch.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_schedule_parse.py`:

```python
import dataclasses
import json
from datetime import UTC, date, datetime

import pytest

from predictor.sources import schedule


def _game(
    game_id,
    date_est,
    utc,
    home="PHI",
    away="NYK",
    *,
    city="Philadelphia",
    state="PA",
    arena="Wells Fargo Center",
    neutral=False,
    status_text="7:00 pm ET",
):
    # Shaped like a real ScheduleLeagueV2 game, INCLUDING the outcome
    # fields the parser must ignore (score, wins, losses, pointsLeaders).
    return {
        "gameId": game_id,
        "gameStatus": 3,
        "gameStatusText": status_text,
        "gameDateEst": f"{date_est}T00:00:00Z",
        "gameDateTimeUTC": utc,
        "arenaName": arena,
        "arenaCity": city,
        "arenaState": state,
        "isNeutral": neutral,
        "homeTeam": {"teamTricode": home, "score": 119, "wins": 30, "losses": 10},
        "awayTeam": {"teamTricode": away, "score": 110, "wins": 20, "losses": 20},
        "pointsLeaders": [{"points": 40.0}],
    }


def _payload(games, season="2024-25"):
    return json.dumps(
        {
            "meta": {},
            "leagueSchedule": {
                "seasonYear": season,
                "gameDates": [{"gameDate": "01/15/2025 00:00:00", "games": games}],
            },
        }
    ).encode("utf-8")


def test_parses_a_regular_game():
    result = schedule.parse_schedule(
        _payload([_game("0022400561", "2025-01-15", "2025-01-16T00:00:00Z")]), "2024-25"
    )
    assert result.no_tipoff == [] and result.undetermined == []
    (row,) = result.rows
    assert row == schedule.ScheduleRow(
        game_id="0022400561",
        season="2024-25",
        game_date=date(2025, 1, 15),
        tip_off_utc=datetime(2025, 1, 16, 0, 0, tzinfo=UTC),
        home_team="PHI",
        away_team="NYK",
        arena_name="Wells Fargo Center",
        arena_city="Philadelphia",
        arena_state="PA",
        is_neutral_reported=False,
        is_neutral=False,
    )


def test_rows_carry_no_outcome_fields():
    names = {f.name for f in dataclasses.fields(schedule.ScheduleRow)}
    for forbidden in ("score", "points", "status", "wins", "losses", "leader"):
        assert not any(forbidden in n for n in names), forbidden


def test_tbd_time_is_no_tipoff_not_a_midnight_placeholder():
    result = schedule.parse_schedule(
        _payload([
            _game("0022400561", "2025-01-15", "2025-01-15T05:00:00Z", status_text="TBD")
        ]),
        "2024-25",
    )
    assert result.rows[0].tip_off_utc is None
    assert result.no_tipoff == ["0022400561"]


def test_tipoff_on_a_different_eastern_date_is_rejected():
    # 2025-01-17T00:00Z is 7pm ET on 01-16, not the listed 01-15.
    result = schedule.parse_schedule(
        _payload([_game("0022400561", "2025-01-15", "2025-01-17T00:00:00Z")]), "2024-25"
    )
    assert result.rows[0].tip_off_utc is None
    assert result.no_tipoff == ["0022400561"]


def test_game_with_undecided_teams_is_left_out_and_reported():
    result = schedule.parse_schedule(
        _payload([
            _game("0062400001", "2024-12-17", "2024-12-17T05:00:00Z",
                  home=None, away=None, status_text="TBD"),
            _game("0022400561", "2025-01-15", "2025-01-16T00:00:00Z"),
        ]),
        "2024-25",
    )
    assert [r.game_id for r in result.rows] == ["0022400561"]
    assert result.undetermined == ["0062400001"]


def test_reported_neutral_flag_is_kept():
    result = schedule.parse_schedule(
        _payload([
            _game("0022400621", "2025-01-23", "2025-01-23T19:00:00Z", home="IND",
                  away="SAS", city="Paris", state="", arena="Accor Arena", neutral=True)
        ]),
        "2024-25",
    )
    row = result.rows[0]
    assert row.is_neutral_reported is True and row.is_neutral is True
    assert row.arena_state is None  # empty string becomes None


def test_game_away_from_the_home_teams_usual_arena_is_derived_neutral():
    # Before 2024-25 the league's flag is false even in Paris.
    games = [
        _game("0022200001", "2022-10-20", "2022-10-20T23:00:00Z"),
        _game("0022200002", "2022-10-22", "2022-10-22T23:00:00Z"),
        _game("0022200678", "2023-01-19", "2023-01-19T19:00:00Z", city="Paris",
              state="", arena="Accor Arena"),
    ]
    result = schedule.parse_schedule(_payload(games, season="2022-23"), "2022-23")
    by_id = {r.game_id: r for r in result.rows}
    assert by_id["0022200678"].is_neutral is True
    assert by_id["0022200678"].is_neutral_reported is False
    assert by_id["0022200001"].is_neutral is False


def test_postseason_bubble_game_is_derived_neutral():
    games = [
        _game("0021900001", "2019-10-22", "2019-10-23T02:00:00Z", home="LAL",
              away="LAC", city="Los Angeles", state="CA", arena="Staples Center"),
        _game("0041900101", "2020-08-18", "2020-08-19T01:00:00Z", home="LAL",
              away="POR", city="Orlando", state="FL", arena="ESPN Wide World"),
    ]
    result = schedule.parse_schedule(_payload(games, season="2019-20"), "2019-20")
    by_id = {r.game_id: r for r in result.rows}
    assert by_id["0041900101"].is_neutral is True
    assert by_id["0021900001"].is_neutral is False


def test_wrong_season_in_payload_is_an_error():
    with pytest.raises(ValueError, match="2023-24"):
        schedule.parse_schedule(_payload([], season="2023-24"), "2024-25")


@pytest.mark.parametrize("bad", [b"not json", b"{}", b'{"leagueSchedule": {"seasonYear": "2024-25"}}'])
def test_malformed_payload_is_a_plain_error(bad):
    with pytest.raises(ValueError, match="expected shape"):
        schedule.parse_schedule(bad, "2024-25")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_schedule_parse.py -q`
Expected: FAIL with `ImportError: cannot import name 'schedule'`.

- [ ] **Step 3: Implement**

Create `src/predictor/sources/schedule.py`:

```python
"""NBA league schedule (nba_api ScheduleLeagueV2) -- spec section 1.1.

What may be read from this source:

- tip_off_utc ESTABLISHES the backtest cutoff (see backtest/tipoff.py). It
  is never a predictor feature.
- Arena and neutral-site columns are static venue facts, not
  outcome-bearing, safe to read at any time.
- Scores, game status, team records and points leaders are in the payload
  but are never read by this parser. A field that is not stored cannot
  leak.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")
_REGULAR_SEASON_PREFIX = "002"


@dataclass(frozen=True)
class ScheduleRow:
    game_id: str
    season: str
    game_date: date
    tip_off_utc: datetime | None
    home_team: str
    away_team: str
    arena_name: str | None
    arena_city: str | None
    arena_state: str | None
    is_neutral_reported: bool
    is_neutral: bool


@dataclass(frozen=True)
class ParseResult:
    rows: list[ScheduleRow]
    # Games saved without a tip-off: the league lists the time as TBD, or
    # the listed instant does not fall on the listed Eastern date.
    no_tipoff: list[str]
    # Games left out entirely because their teams are not decided yet
    # (NBA Cup knockout slots, the Cup final).
    undetermined: list[str]


def _text(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _tipoff(game: dict, game_date: date) -> datetime | None:
    """The tip-off instant, or None rather than a guess.

    A TBD game carries a 00:00 ET placeholder in gameDateTimeUTC. Accepting
    it would be conservative once, but tipoff_index takes the MINIMUM
    across vintages, so the placeholder would outlive the real time
    forever. None keeps it out of the index entirely.
    """
    if (_text(game.get("gameStatusText")) or "").upper() == "TBD":
        return None
    raw = _text(game.get("gameDateTimeUTC"))
    if raw is None:
        return None
    try:
        tip = datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None
    if tip.astimezone(EASTERN).date() != game_date:
        return None
    return tip


def parse_schedule(payload: bytes, season: str) -> ParseResult:
    try:
        doc = json.loads(payload)
        league = doc["leagueSchedule"]
        reported_season = league["seasonYear"]
        staged = []
        undetermined: list[str] = []
        for day in league["gameDates"]:
            for game in day["games"]:
                game_id = str(game["gameId"])
                home = _text((game.get("homeTeam") or {}).get("teamTricode"))
                away = _text((game.get("awayTeam") or {}).get("teamTricode"))
                if home is None or away is None:
                    undetermined.append(game_id)
                    continue
                game_date = date.fromisoformat(str(game["gameDateEst"])[:10])
                staged.append((game, game_id, game_date, home, away))
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            f"the NBA schedule response for {season} was not in the expected "
            f"shape ({exc!r})"
        ) from None
    if reported_season != season:
        raise ValueError(
            f"asked for the {season} schedule but the NBA returned the "
            f"{reported_season!r} schedule"
        )

    # Each team's usual home venue: the (city, state) it plays most of its
    # regular-season home games at, in this payload. The league's own
    # isNeutral is false for every game before 2024-25 -- Paris, Mexico
    # City and Las Vegas included -- so a game away from that venue is
    # neutral too. Measured effect: the 2019-20 Orlando bubble, the
    # Paris/Mexico City/Las Vegas games, and San Antonio's Austin games.
    venues: dict[str, Counter] = {}
    for game, game_id, _, home, _ in staged:
        if game_id.startswith(_REGULAR_SEASON_PREFIX) and not bool(game.get("isNeutral")):
            key = (_text(game.get("arenaCity")), _text(game.get("arenaState")))
            venues.setdefault(home, Counter())[key] += 1
    usual = {team: counts.most_common(1)[0][0] for team, counts in venues.items()}

    rows: list[ScheduleRow] = []
    no_tipoff: list[str] = []
    for game, game_id, game_date, home, away in staged:
        city = _text(game.get("arenaCity"))
        state = _text(game.get("arenaState"))
        reported = bool(game.get("isNeutral"))
        away_from_home = home in usual and (city, state) != usual[home]
        tip = _tipoff(game, game_date)
        if tip is None:
            no_tipoff.append(game_id)
        rows.append(
            ScheduleRow(
                game_id=game_id,
                season=season,
                game_date=game_date,
                tip_off_utc=tip,
                home_team=home,
                away_team=away,
                arena_name=_text(game.get("arenaName")),
                arena_city=city,
                arena_state=state,
                is_neutral_reported=reported,
                is_neutral=reported or away_from_home,
            )
        )
    return ParseResult(rows=rows, no_tipoff=no_tipoff, undetermined=undetermined)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_schedule_parse.py -q` and then `uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/predictor/sources/schedule.py tests/test_schedule_parse.py
git commit -m "feat: parse the league schedule, dropping every outcome field

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Fetch, archive, ingest, and compare with the games table

**Files:**
- Modify: `src/predictor/sources/schedule.py`
- Test: `tests/test_schedule_ingest.py`

**Interfaces:**
- Consumes: `parse_schedule`, `ScheduleRow` (Task 2); `db.POINT_IN_TIME_TABLES["schedule"]` (Task 1); `raw_store.store/load`; `db.require_utc`.
- Produces: `schedule.SOURCE = "schedule"`.
- Produces: `schedule.fetch_season_payload(season: str) -> bytes`.
- Produces: `schedule.archive_key(season: str, fetched_at: datetime) -> str`.
- Produces: `schedule.IngestResult` (frozen: `season: str, written: int, no_tipoff: list[str], undetermined: list[str], mismatches: list[str], blob_key: str`).
- Produces: `schedule.ingest_season(con, season: str, fetched_at: datetime | None = None, fetch: Callable[[str], bytes] = fetch_season_payload) -> IngestResult`.
- Produces: `schedule.compare_with_games(con, season: str, rows: list[ScheduleRow]) -> list[str]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_schedule_ingest.py`:

```python
import gzip
import json
from datetime import UTC, date, datetime, timedelta

import pytest

from predictor import config, db, raw_store
from predictor.config import Settings
from predictor.sources import schedule

FETCHED = datetime(2026, 9, 27, 5, 0, tzinfo=UTC)


def _game(game_id, date_est, utc, home="PHI", away="NYK"):
    return {
        "gameId": game_id, "gameStatus": 3, "gameStatusText": "Final",
        "gameDateEst": f"{date_est}T00:00:00Z", "gameDateTimeUTC": utc,
        "arenaName": "Wells Fargo Center", "arenaCity": "Philadelphia",
        "arenaState": "PA", "isNeutral": False,
        "homeTeam": {"teamTricode": home, "score": 119},
        "awayTeam": {"teamTricode": away, "score": 110},
    }


def _payload(games, season="2024-25"):
    return json.dumps(
        {"leagueSchedule": {"seasonYear": season, "gameDates": [{"games": games}]}}
    ).encode("utf-8")


@pytest.fixture
def con(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(config, "settings", s)
    monkeypatch.setattr(db, "settings", s)
    monkeypatch.setattr(raw_store, "settings", s)
    c = db.connect(tmp_path / "t.duckdb")
    db.migrate(c)
    return c


def _insert_game(con, game_id, game_date, home, away, season="2024-25"):
    g = db.POINT_IN_TIME_TABLES["games"]
    con.execute(
        f"INSERT INTO {g} (game_id, season, game_date, home_team, away_team,"
        " home_points, away_points, status, reconstructed, observed_at)"
        " VALUES (?,?,?,?,?,NULL,NULL,'SCHEDULED',TRUE,?)",
        [game_id, season, game_date, home, away, FETCHED - timedelta(days=400)],
    )


def test_ingest_archives_first_then_writes_rows_at_the_fetch_time(con):
    payload = _payload([_game("0022400561", "2025-01-15", "2025-01-16T00:00:00Z")])
    result = schedule.ingest_season(con, "2024-25", fetched_at=FETCHED, fetch=lambda s: payload)

    assert result.written == 1
    assert result.blob_key == "2024-25_20260927T050000Z.json.gz"
    archived = raw_store.load(schedule.SOURCE, result.blob_key)
    assert gzip.decompress(archived) == payload

    table = db.POINT_IN_TIME_TABLES["schedule"]
    rows = con.execute(
        f"SELECT game_id, game_date, tip_off_utc, observed_at FROM {table}"
    ).fetchall()
    assert rows == [
        ("0022400561", date(2025, 1, 15), datetime(2025, 1, 16, tzinfo=UTC), FETCHED)
    ]


def test_each_fetch_adds_a_new_vintage(con):
    first = _payload([_game("0022400561", "2025-01-15", "2025-01-16T00:00:00Z")])
    moved = _payload([_game("0022400561", "2025-01-15", "2025-01-15T22:30:00Z")])
    schedule.ingest_season(con, "2024-25", fetched_at=FETCHED, fetch=lambda s: first)
    schedule.ingest_season(
        con, "2024-25", fetched_at=FETCHED + timedelta(days=1), fetch=lambda s: moved
    )
    table = db.POINT_IN_TIME_TABLES["schedule"]
    tips = con.execute(
        f"SELECT tip_off_utc FROM {table} ORDER BY observed_at"
    ).fetchall()
    assert [t[0].hour for t in tips] == [0, 22]


def test_naive_fetch_time_is_rejected(con):
    with pytest.raises(ValueError, match="timezone-aware"):
        schedule.ingest_season(
            con, "2024-25", fetched_at=datetime(2026, 9, 27), fetch=lambda s: b""
        )


def test_unparseable_payload_is_still_archived(con):
    with pytest.raises(ValueError):
        schedule.ingest_season(con, "2024-25", fetched_at=FETCHED, fetch=lambda s: b"garbage")
    key = schedule.archive_key("2024-25", FETCHED)
    assert gzip.decompress(raw_store.load(schedule.SOURCE, key)) == b"garbage"


def test_agreeing_games_table_produces_no_mismatch(con):
    _insert_game(con, "0022400561", date(2025, 1, 15), "PHI", "NYK")
    payload = _payload([_game("0022400561", "2025-01-15", "2025-01-16T00:00:00Z")])
    result = schedule.ingest_season(con, "2024-25", fetched_at=FETCHED, fetch=lambda s: payload)
    assert result.mismatches == []


def test_date_disagreement_with_games_table_is_reported_loudly(con, capsys):
    _insert_game(con, "0022400561", date(2025, 1, 14), "PHI", "NYK")
    payload = _payload([_game("0022400561", "2025-01-15", "2025-01-16T00:00:00Z")])
    result = schedule.ingest_season(con, "2024-25", fetched_at=FETCHED, fetch=lambda s: payload)
    assert len(result.mismatches) == 1
    assert "0022400561" in result.mismatches[0]
    assert "2025-01-14" in result.mismatches[0] and "2025-01-15" in result.mismatches[0]
    assert "schedule: MISMATCH" in capsys.readouterr().out


def test_regular_season_game_missing_from_schedule_is_reported(con):
    _insert_game(con, "0022400999", date(2025, 1, 20), "BOS", "LAL")
    payload = _payload([_game("0022400561", "2025-01-15", "2025-01-16T00:00:00Z")])
    result = schedule.ingest_season(con, "2024-25", fetched_at=FETCHED, fetch=lambda s: payload)
    assert any("0022400999" in m and "not in the schedule" in m for m in result.mismatches)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_schedule_ingest.py -q`
Expected: FAIL with `AttributeError: module 'predictor.sources.schedule' has no attribute 'ingest_season'`.

- [ ] **Step 3: Implement**

In `src/predictor/sources/schedule.py`, extend the imports to:

```python
import gzip
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from nba_api.stats.endpoints import scheduleleaguev2
from tenacity import retry, stop_after_attempt, wait_exponential

from predictor import db, raw_store
```

Add `SOURCE = "schedule"` below `EASTERN`, then append:

```python
@dataclass(frozen=True)
class IngestResult:
    season: str
    written: int
    no_tipoff: list[str]
    undetermined: list[str]
    mismatches: list[str]
    blob_key: str


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
)
def fetch_season_payload(season: str) -> bytes:
    """The raw schedule JSON for one season, historical or forward."""
    endpoint = scheduleleaguev2.ScheduleLeagueV2(
        season=season, league_id="00", timeout=60
    )
    return endpoint.get_json().encode("utf-8")


def archive_key(season: str, fetched_at: datetime) -> str:
    return f"{season}_{fetched_at:%Y%m%dT%H%M%S}Z.json.gz"


def ingest_season(
    con,
    season: str,
    fetched_at: datetime | None = None,
    fetch: Callable[[str], bytes] = fetch_season_payload,
) -> IngestResult:
    """Fetch, archive, parse and load one season's schedule.

    Raw-first: the gzipped payload is archived BEFORE parsing, and the
    parser reads the archived bytes back, so a parser bug is re-parsable
    rather than lost. Each call writes a new vintage stamped with the real
    fetch time -- the table accumulates genuine point-in-time schedule
    history from the first daily run onward.
    """
    fetched_at = db.require_utc(
        fetched_at if fetched_at is not None else datetime.now(UTC), "fetched_at"
    )
    payload = fetch(season)
    key = archive_key(season, fetched_at)
    raw_store.store(
        SOURCE, key, gzip.compress(payload, mtime=0), fetched_at, meta={"season": season}
    )
    parsed = parse_schedule(gzip.decompress(raw_store.load(SOURCE, key)), season)

    table = db.POINT_IN_TIME_TABLES["schedule"]
    if parsed.rows:
        con.executemany(
            f"INSERT OR REPLACE INTO {table} (game_id, season, game_date,"
            " tip_off_utc, home_team, away_team, arena_name, arena_city,"
            " arena_state, is_neutral_reported, is_neutral, observed_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                [
                    r.game_id, r.season, r.game_date, r.tip_off_utc, r.home_team,
                    r.away_team, r.arena_name, r.arena_city, r.arena_state,
                    r.is_neutral_reported, r.is_neutral, fetched_at,
                ]
                for r in parsed.rows
            ],
        )

    mismatches = compare_with_games(con, season, parsed.rows)
    for line in mismatches:
        print(f"schedule: MISMATCH {line}")
    return IngestResult(
        season=season,
        written=len(parsed.rows),
        no_tipoff=parsed.no_tipoff,
        undetermined=parsed.undetermined,
        mismatches=mismatches,
        blob_key=key,
    )


def compare_with_games(con, season: str, rows: list[ScheduleRow]) -> list[str]:
    """Where the schedule and the games table disagree, say so -- never reconcile.

    Spec 1.1: the schedule is ground truth for when a game tipped, and a
    disagreement (chiefly a rescheduled game) is logged loudly. Measured
    2026-09-27: zero disagreements across all 8,289 regular-season games,
    so any line printed here is news.
    """
    games = db.POINT_IN_TIME_TABLES["games"]
    by_id = {r.game_id: r for r in rows}
    found = con.execute(
        f"SELECT game_id, MIN(game_date), MIN(home_team), MIN(away_team) "
        f"FROM {games} WHERE season = ? GROUP BY game_id ORDER BY game_id",
        [season],
    ).fetchall()
    out: list[str] = []
    for game_id, game_date, home, away in found:
        row = by_id.get(game_id)
        if row is None:
            if game_id.startswith(_REGULAR_SEASON_PREFIX):
                out.append(
                    f"{game_id}: in the games table ({game_date} {away}@{home}) "
                    "but not in the schedule"
                )
            continue
        if (row.game_date, row.home_team, row.away_team) != (game_date, home, away):
            out.append(
                f"{game_id}: schedule says {row.game_date} {row.away_team}@"
                f"{row.home_team}, games table says {game_date} {away}@{home}"
            )
    return out
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_schedule_ingest.py -q` and then `uv run pytest -q`
Expected: all pass. `test_no_physical_table_name_appears_outside_db_and_asof` must still pass, because `schedule.py` names no `_raw` table.

- [ ] **Step 5: Commit**

```bash
git add src/predictor/sources/schedule.py tests/test_schedule_ingest.py
git commit -m "feat: archive and ingest schedule vintages, reporting games-table disagreements

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `ingest-schedule` command and lock-tolerant connections

**Files:**
- Modify: `src/predictor/db.py`, `src/predictor/cli.py`
- Test: `tests/test_schedule_cli.py`, `tests/test_db.py`

**Interfaces:**
- Consumes: `schedule.ingest_season`, `IngestResult` (Task 3); `config.season_label`, `config.previous_season_label` (Task 1).
- Produces: `db.connect_with_retry(path: Path | None = None, *, attempts: int = 6, wait_seconds: float = 20.0, sleep=time.sleep) -> duckdb.DuckDBPyConnection`.
- Produces: the CLI command `predictor ingest-schedule [--season 2024-25]`, which exits 1 on any failure or mismatch.

Why `poll-news` changes too: the news and schedule jobs are both launchd calendar jobs. After a laptop sleeps through their slots, launchd fires both on wake, together. Without a retry, whichever job loses DuckDB's write lock fails. News is the only source that cannot be recovered retroactively.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_db.py`:

```python
def test_connect_with_retry_waits_out_a_lock_then_connects(tmp_path, monkeypatch):
    import duckdb

    real_connect = db.connect
    calls = {"n": 0}
    slept: list[float] = []

    def flaky(path=None, *, read_only=False):
        calls["n"] += 1
        if calls["n"] < 3:
            raise duckdb.IOException(
                "IO Error: Could not set lock on file: Conflicting lock is held in "
                "/x/python3 (PID 1) by user me"
            )
        return real_connect(tmp_path / "t.duckdb")

    monkeypatch.setattr(db, "connect", flaky)
    con = db.connect_with_retry(attempts=5, wait_seconds=7, sleep=slept.append)
    assert con.execute("SELECT 1").fetchone() == (1,)
    assert slept == [7, 7]


def test_connect_with_retry_does_not_retry_other_errors(monkeypatch):
    import duckdb

    calls = {"n": 0}

    def broken(path=None, *, read_only=False):
        calls["n"] += 1
        raise duckdb.IOException("IO Error: Cannot open file: No such file or directory")

    monkeypatch.setattr(db, "connect", broken)
    with pytest.raises(duckdb.IOException):
        db.connect_with_retry(attempts=5, wait_seconds=0, sleep=lambda s: None)
    assert calls["n"] == 1


def test_connect_with_retry_gives_up_after_the_last_attempt(monkeypatch):
    import duckdb

    def locked(path=None, *, read_only=False):
        raise duckdb.IOException("Conflicting lock is held in x")

    monkeypatch.setattr(db, "connect", locked)
    with pytest.raises(duckdb.IOException):
        db.connect_with_retry(attempts=3, wait_seconds=0, sleep=lambda s: None)
```

(If `pytest` is not already imported at the top of `tests/test_db.py`, add `import pytest`.)

Create `tests/test_schedule_cli.py`:

```python
from datetime import UTC, datetime

import requests
from typer.testing import CliRunner

from predictor import cli, config, db, raw_store
from predictor.config import Settings
from predictor.sources import schedule

runner = CliRunner()


def _point_settings_at_tmp(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(config, "settings", s)
    monkeypatch.setattr(db, "settings", s)
    monkeypatch.setattr(raw_store, "settings", s)
    return s


def _result(season, written=1230, no_tipoff=(), undetermined=(), mismatches=()):
    return schedule.IngestResult(
        season=season, written=written, no_tipoff=list(no_tipoff),
        undetermined=list(undetermined), mismatches=list(mismatches),
        blob_key=f"{season}_x.json.gz",
    )


def test_clean_run_reports_and_exits_zero(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)
    monkeypatch.setattr(schedule, "ingest_season", lambda con, season: _result(season))
    out = runner.invoke(cli.app, ["ingest-schedule", "--season", "2024-25"])
    assert out.exit_code == 0, out.output
    assert "2024-25: 1,230 games saved" in out.output


def test_default_season_is_the_current_one(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)
    seen = []

    def fake(con, season):
        seen.append(season)
        return _result(season)

    monkeypatch.setattr(schedule, "ingest_season", fake)
    runner.invoke(cli.app, ["ingest-schedule"])
    assert seen == [config.season_label(datetime.now(UTC))]


def test_empty_upcoming_season_falls_back_to_the_previous_one(tmp_path, monkeypatch):
    # July-August: next season's schedule is not published yet. Keep
    # refreshing the previous one so `status` does not cry wolf for weeks.
    _point_settings_at_tmp(tmp_path, monkeypatch)
    seen = []

    def fake(con, season):
        seen.append(season)
        return _result(season, written=0 if len(seen) == 1 else 1400)

    monkeypatch.setattr(schedule, "ingest_season", fake)
    out = runner.invoke(cli.app, ["ingest-schedule"])
    current = config.season_label(datetime.now(UTC))
    assert seen == [current, config.previous_season_label(current)]
    assert out.exit_code == 0, out.output
    assert "not published yet" in out.output


def test_explicit_season_with_no_games_does_not_fall_back(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)
    seen = []

    def fake(con, season):
        seen.append(season)
        return _result(season, written=0)

    monkeypatch.setattr(schedule, "ingest_season", fake)
    runner.invoke(cli.app, ["ingest-schedule", "--season", "2030-31"])
    assert seen == ["2030-31"]


def test_tbd_and_undecided_games_are_explained_not_failed(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)
    monkeypatch.setattr(
        schedule, "ingest_season",
        lambda con, season: _result(season, no_tipoff=["a", "b"], undetermined=["c"]),
    )
    out = runner.invoke(cli.app, ["ingest-schedule", "--season", "2026-27"])
    assert out.exit_code == 0, out.output
    assert "2 game(s) have no tip-off time yet" in out.output
    assert "1 game(s) left out because their teams are not decided yet" in out.output


def test_mismatch_warns_and_exits_nonzero(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)
    monkeypatch.setattr(
        schedule, "ingest_season",
        lambda con, season: _result(season, mismatches=["0022400561: schedule says ..."]),
    )
    out = runner.invoke(cli.app, ["ingest-schedule", "--season", "2024-25"])
    assert out.exit_code == 1
    assert "WARNING" in out.output and "disagree" in out.output


def test_network_failure_is_plain_english(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)

    def down(con, season):
        raise requests.ConnectionError("Name or service not known")

    monkeypatch.setattr(schedule, "ingest_season", down)
    out = runner.invoke(cli.app, ["ingest-schedule", "--season", "2024-25"])
    assert out.exit_code == 1
    assert "Could not download the NBA schedule" in out.output
    assert "Traceback" not in out.output


def test_unreadable_payload_is_plain_english(tmp_path, monkeypatch):
    _point_settings_at_tmp(tmp_path, monkeypatch)

    def bad(con, season):
        raise ValueError("the NBA schedule response for 2024-25 was not in the expected shape")

    monkeypatch.setattr(schedule, "ingest_season", bad)
    out = runner.invoke(cli.app, ["ingest-schedule", "--season", "2024-25"])
    assert out.exit_code == 1
    assert "archived" in out.output and "could not be read" in out.output
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_schedule_cli.py tests/test_db.py -q`
Expected: FAIL. You should see `No such command 'ingest-schedule'` and `AttributeError: ... connect_with_retry`.

- [ ] **Step 3: Implement**

`src/predictor/db.py`: add `import time` to the imports, and append after `connect`:

```python
def connect_with_retry(
    path: Path | None = None,
    *,
    attempts: int = 6,
    wait_seconds: float = 20.0,
    sleep=time.sleep,
) -> duckdb.DuckDBPyConnection:
    """connect(), waiting out another predictor process's write lock.

    DuckDB allows one writer per file. The news and schedule launchd jobs
    both fire on wake after a laptop sleeps through their slots, so one of
    them routinely finds the other holding the lock for a few seconds.
    Only that specific error is retried (matched on DuckDB's own message,
    as the backtest command does); anything else raises immediately.
    """
    for attempt in range(1, attempts + 1):
        try:
            return connect(path)
        except duckdb.Error as exc:
            if "conflicting lock is held" not in str(exc).lower() or attempt == attempts:
                raise
            sleep(wait_seconds)
    raise AssertionError("unreachable")
```

`src/predictor/cli.py`:
1. In `poll_news`, change `con = db.connect()` to `con = db.connect_with_retry()`.
2. Add after the `ingest-season` command:

```python
@app.command("ingest-schedule")
def ingest_schedule_cmd(
    season: str = typer.Option(
        None, help="Season to fetch, e.g. 2024-25. Defaults to the current season."
    ),
) -> None:
    """Fetch the NBA schedule for one season, archive it, and load it."""
    from datetime import UTC, datetime

    import duckdb
    import requests

    from predictor import db
    from predictor.config import previous_season_label, season_label, settings
    from predictor.sources import schedule

    settings.ensure_dirs()
    try:
        con = db.connect_with_retry()
    except duckdb.Error as exc:
        typer.echo(
            f"Could not open the database to save the schedule ({exc}). Nothing "
            "was saved; the next scheduled run will try again."
        )
        raise typer.Exit(code=1) from None
    db.migrate(con)

    targets = [season] if season else [season_label(datetime.now(UTC))]
    results = []
    for target in targets:
        try:
            result = schedule.ingest_season(con, target)
        except requests.RequestException as exc:
            typer.echo(
                f"Could not download the NBA schedule for {target} ({exc}). "
                "Nothing was saved; the next scheduled run will try again."
            )
            raise typer.Exit(code=1) from None
        except ValueError as exc:
            typer.echo(
                f"The {target} schedule was downloaded and archived, but could "
                f"not be read: {exc}. Nothing was loaded into the database."
            )
            raise typer.Exit(code=1) from None
        results.append(result)
        if result.written == 0 and season is None and len(targets) == 1:
            typer.echo(
                f"No games published yet for {target}; refreshing "
                f"{previous_season_label(target)} instead."
            )
            targets.append(previous_season_label(target))

    mismatched = False
    for result in results:
        typer.echo(f"schedule {result.season}: {result.written:,} games saved")
        if result.no_tipoff:
            typer.echo(
                f"  {len(result.no_tipoff)} game(s) have no tip-off time yet "
                "(the league lists them as TBD) -- a later run picks the time "
                "up once it is announced."
            )
        if result.undetermined:
            typer.echo(
                f"  {len(result.undetermined)} game(s) left out because their "
                "teams are not decided yet (for example NBA Cup knockout "
                "games) -- a later run picks them up."
            )
        if result.mismatches:
            mismatched = True
            typer.echo(
                f"WARNING: {len(result.mismatches)} game(s) in {result.season} "
                "disagree between the schedule and the games table (see the "
                "'schedule: MISMATCH' lines above). The schedule was saved as "
                "published; nothing was changed to make them agree. Usually a "
                "rescheduled game -- re-run 'predictor ingest-season "
                f"{result.season}' and then this command."
            )
    if mismatched:
        raise typer.Exit(code=1)
```

(Appending to `targets` while iterating is deliberate. The fallback season is processed in the same loop, at most once, and only when `--season` was not given.)

- [ ] **Step 4: Run tests**

Run: `uv run pytest -q`
Expected: all pass, including every existing `test_news_cli.py` test.

- [ ] **Step 5: Commit**

```bash
git add src/predictor/db.py src/predictor/cli.py tests/test_db.py tests/test_schedule_cli.py
git commit -m "feat: add ingest-schedule, and wait out write locks in both scheduled jobs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Backfill the real archive (operational, no code)

This task writes to the irreplaceable database. Back it up first.

- [ ] **Step 1: Back up**

```bash
cd /Users/meya/projects/predictor
mkdir -p data/backups
cp data/predictor.duckdb "data/backups/predictor-$(date +%Y%m%d-%H%M%S)-pre-schedule.duckdb"
ls -la data/backups
```

Expected: a backup file of about 15 MB. If a `predictor` command is running (for example news at 09:00, 14:00 or 19:00), `connect_with_retry` waits; that is fine.

- [ ] **Step 2: Ingest every season**

```bash
for s in 2019-20 2020-21 2021-22 2022-23 2023-24 2024-25 2025-26 2026-27; do
  uv run predictor ingest-schedule --season "$s" || echo "FAILED: $s"
done
```

Expected, matching the 2026-09-27 probe (small drift in 2026-27 is fine):
- 2019-20: 1,145 saved
- 2020-21: 1,221
- 2021-22: 1,393
- 2022-23: 1,394
- 2023-24: 1,396
- 2024-25: 1,400
- 2025-26: 1,400
- 2026-27: ~1,267 saved, with ~7 left out as undecided

There should be no `MISMATCH` lines and no `FAILED` lines. If any appear, stop and report them; do not proceed.

- [ ] **Step 3: Verify coverage**

```bash
uv run python - <<'EOF'
import duckdb
con = duckdb.connect("data/predictor.duckdb", read_only=True)
print("regular-season games without a schedule tip-off:", con.execute("""
    SELECT count(*) FROM (SELECT DISTINCT game_id FROM games_raw WHERE game_id LIKE '002%') g
    WHERE g.game_id NOT IN (SELECT game_id FROM schedule_raw WHERE tip_off_utc IS NOT NULL)
""").fetchone()[0])
print(con.execute("""
    SELECT season, count(DISTINCT game_id) AS games,
           count(DISTINCT game_id) FILTER (WHERE is_neutral) AS neutral
    FROM schedule_raw GROUP BY season ORDER BY season
""").fetchall())
EOF
```

Expected:
- `regular-season games without a schedule tip-off: 0`
- Neutral counts `169, 0, 0, 4, 7, 8, 8, 6` for 2019-20 through 2026-27.

Nothing to commit, because `data/` is git-ignored.

---

### Task 6: Tip-off index from the schedule; injury PDFs become a cross-check

**Files:**
- Modify: `src/predictor/backtest/tipoff.py`
- Create: `tests/schedule_rows.py`, `tests/real_archive.py`, `tests/test_tipoff_crosscheck.py`
- Rewrite: `tests/test_tipoff.py`
- Modify: `tests/test_replay.py`, `tests/test_backtest_leakage.py`

**Interfaces:**
- Consumes: the `schedule_raw` table (Task 1), populated in the real archive (Task 5).
- Produces: the unchanged signatures `tipoff_index(con) -> dict[tuple[date, str], datetime]`, `resolve_tipoff(index, game_date, home_team, away_team) -> datetime | None` and `parse_game_time(raw, game_date) -> datetime | None`. `replay.py` is not edited.
- Produces: test helpers `schedule_rows.insert_schedule_row(con, game_id, game_date, home_team, away_team, tip_off_utc, observed_at=None, season="2024-25")` and `real_archive.open_real_archive_or_skip() -> duckdb.DuckDBPyConnection`.

- [ ] **Step 1: Test helpers**

Create `tests/schedule_rows.py`:

```python
"""Seed one schedule row -- the harness's tip-off source since sub-project 2.5."""

from datetime import timedelta

from predictor import db


def insert_schedule_row(
    con,
    game_id,
    game_date,
    home_team,
    away_team,
    tip_off_utc,
    observed_at=None,
    season="2024-25",
):
    if observed_at is None:
        observed_at = tip_off_utc - timedelta(days=30)
    table = db.POINT_IN_TIME_TABLES["schedule"]
    con.execute(
        f"INSERT INTO {table} (game_id, season, game_date, tip_off_utc, home_team,"
        " away_team, arena_name, arena_city, arena_state, is_neutral_reported,"
        " is_neutral, observed_at) VALUES (?,?,?,?,?,?,?,?,?,FALSE,FALSE,?)",
        [game_id, season, game_date, tip_off_utc, home_team, away_team,
         "Test Arena", "Testville", "TS", observed_at],
    )
```

Create `tests/real_archive.py`:

```python
"""Read-only access to the real archive for archive-wide invariant tests.

Never db.connect(): its rail exists to stop a test from migrating or
writing the real database, not from reading it.
"""

import duckdb
import pytest

from predictor.config import PROJECT_ROOT

REAL_ARCHIVE = PROJECT_ROOT / "data" / "predictor.duckdb"


def open_real_archive_or_skip():
    if not REAL_ARCHIVE.exists():
        pytest.skip("real archive not present in this environment")
    try:
        con = duckdb.connect(str(REAL_ARCHIVE), read_only=True)
    except duckdb.Error as exc:  # pragma: no cover - timing-dependent
        # The scheduled news/schedule jobs hold the write lock briefly.
        pytest.skip(f"real archive is locked by another process: {exc}")
    con.execute("SET TimeZone='UTC'")
    has_schedule = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = 'schedule_raw'"
    ).fetchone()[0]
    if not has_schedule:
        con.close()
        pytest.skip("real archive has no schedule yet -- run predictor ingest-schedule")
    return con
```

- [ ] **Step 2: Rewrite `tests/test_tipoff.py`**

Replace the whole file with:

```python
from datetime import UTC, date, datetime, timedelta

import pytest

from predictor import db
from predictor.backtest import tipoff
from real_archive import open_real_archive_or_skip
from schedule_rows import insert_schedule_row


@pytest.fixture
def con(tmp_path):
    c = db.connect(tmp_path / "t.duckdb")
    db.migrate(c)
    return c


# --- parse_game_time: retained for the injury-PDF cross-check ------------


def test_evening_game_is_pm():
    # 7pm ET in January (EST, UTC-5) is 00:00 UTC the next day.
    got = tipoff.parse_game_time("07:00 (ET)", date(2025, 1, 15))
    assert got == datetime(2025, 1, 16, 0, 0, tzinfo=UTC)


def test_noon_game_is_not_shifted_to_midnight():
    got = tipoff.parse_game_time("12:00 (ET)", date(2025, 1, 15))
    assert got == datetime(2025, 1, 15, 17, 0, tzinfo=UTC)


def test_afternoon_game():
    got = tipoff.parse_game_time("03:30 (ET)", date(2025, 1, 15))
    assert got == datetime(2025, 1, 15, 20, 30, tzinfo=UTC)


def test_layout_without_space_before_paren_parses_identically():
    with_space = tipoff.parse_game_time("08:00 (ET)", date(2025, 1, 15))
    without = tipoff.parse_game_time("08:00(ET)", date(2025, 1, 15))
    assert with_space == without


def test_daylight_saving_is_honoured():
    summer = tipoff.parse_game_time("07:00 (ET)", date(2025, 6, 10))
    winter = tipoff.parse_game_time("07:00 (ET)", date(2025, 1, 10))
    assert summer.hour == 23
    assert winter.hour == 0


def test_unparseable_returns_none():
    for bad in ["", "TBD", "25:00 (ET)", "07:00 (PT)", "7pm"]:
        assert tipoff.parse_game_time(bad, date(2025, 1, 15)) is None


# --- tipoff_index: built from schedule vintages --------------------------


def test_index_keys_both_teams_to_the_tipoff(con):
    gd = date(2025, 1, 15)
    tip = datetime(2025, 1, 16, 0, 0, tzinfo=UTC)
    insert_schedule_row(con, "0022400561", gd, "PHI", "NYK", tip)
    index = tipoff.tipoff_index(con)
    assert index[(gd, "PHI")] == tip
    assert index[(gd, "NYK")] == tip


def test_a_later_vintage_moving_the_game_earlier_is_honoured(con):
    # FIX 14's shape: a correction to an EARLIER time arrives later. Ignoring
    # it would put the cutoff after the real tip-off.
    gd = date(2022, 11, 9)
    insert_schedule_row(con, "0022200161", gd, "ORL", "DAL",
                        datetime(2022, 11, 10, 0, 0, tzinfo=UTC),
                        observed_at=datetime(2022, 11, 1, tzinfo=UTC))
    insert_schedule_row(con, "0022200161", gd, "ORL", "DAL",
                        datetime(2022, 11, 9, 22, 30, tzinfo=UTC),
                        observed_at=datetime(2022, 11, 9, 12, tzinfo=UTC))
    index = tipoff.tipoff_index(con)
    assert index[(gd, "ORL")] == datetime(2022, 11, 9, 22, 30, tzinfo=UTC)


def test_a_later_vintage_moving_the_game_later_still_resolves_to_the_earlier_time(con):
    gd = date(2025, 1, 15)
    insert_schedule_row(con, "0022400561", gd, "PHI", "NYK",
                        datetime(2025, 1, 16, 0, 0, tzinfo=UTC),
                        observed_at=datetime(2025, 1, 1, tzinfo=UTC))
    insert_schedule_row(con, "0022400561", gd, "PHI", "NYK",
                        datetime(2025, 1, 16, 2, 30, tzinfo=UTC),
                        observed_at=datetime(2025, 1, 10, tzinfo=UTC))
    assert tipoff.tipoff_index(con)[(gd, "PHI")] == datetime(2025, 1, 16, 0, 0, tzinfo=UTC)


def test_tbd_rows_never_enter_the_index(con):
    gd = date(2026, 12, 4)
    insert_schedule_row(con, "0022601201", gd, "PHI", "NYK", None,
                        observed_at=datetime(2026, 9, 27, tzinfo=UTC))
    assert tipoff.tipoff_index(con) == {}


def test_injury_report_game_time_is_no_longer_a_tipoff_source(con):
    i = db.POINT_IN_TIME_TABLES["injury_status"]
    con.execute(
        f"INSERT INTO {i} (report_date, game_date, game_time, team, player, status,"
        " observed_at) VALUES (?,?,?,?,?,?,?)",
        [date(2025, 1, 15), date(2025, 1, 15), "07:00 (ET)", "PHI", "p", "Out",
         datetime(2025, 1, 15, 22, 30, tzinfo=UTC)],
    )
    assert tipoff.tipoff_index(con) == {}


# --- resolve_tipoff: unchanged pure function over the index --------------


def test_resolve_tipoff_takes_the_minimum_when_home_is_earlier():
    gd = date(2025, 1, 15)
    early = datetime(2025, 1, 16, 0, 0, tzinfo=UTC)
    late = datetime(2025, 1, 16, 0, 30, tzinfo=UTC)
    index = {(gd, "PHI"): early, (gd, "NYK"): late}
    assert tipoff.resolve_tipoff(index, gd, "PHI", "NYK") == early


def test_resolve_tipoff_takes_the_minimum_when_away_is_earlier():
    gd = date(2025, 1, 15)
    early = datetime(2025, 1, 16, 0, 30, tzinfo=UTC)
    late = datetime(2025, 1, 16, 1, 30, tzinfo=UTC)
    index = {(gd, "NYK"): late, (gd, "PHI"): early}
    assert tipoff.resolve_tipoff(index, gd, "NYK", "PHI") == early


def test_resolve_tipoff_falls_back_to_away_team():
    gd = date(2025, 1, 15)
    tip = datetime(2025, 1, 16, 0, 30, tzinfo=UTC)
    assert tipoff.resolve_tipoff({(gd, "NYK"): tip}, gd, "PHI", "NYK") == tip


def test_resolve_tipoff_returns_none_when_neither_team_has_an_entry():
    assert tipoff.resolve_tipoff({}, date(2025, 1, 15), "PHI", "NYK") is None


# --- archive-wide invariants (real archive, read-only) -------------------


def test_every_archived_regular_season_game_resolves_a_tipoff():
    """Sub-project 2.5's coverage promise: 8,289 of 8,289, not 7,200."""
    con = open_real_archive_or_skip()
    try:
        games = db.POINT_IN_TIME_TABLES["games"]
        index = tipoff.tipoff_index(con)
        rows = con.execute(
            f"SELECT game_id, MIN(game_date), MIN(home_team), MIN(away_team) "
            f"FROM {games} WHERE game_id LIKE '002%' GROUP BY game_id"
        ).fetchall()
        assert len(rows) >= 8289
        unresolved = [
            gid for gid, gd, h, a in rows if tipoff.resolve_tipoff(index, gd, h, a) is None
        ]
        assert unresolved == [], f"{len(unresolved)} game(s) have no tip-off: {unresolved[:10]}"
    finally:
        con.close()


@pytest.mark.parametrize("buffer_minutes", [30, 60, 120])
def test_archive_wide_cutoff_is_strictly_before_every_recorded_schedule_vintage(
    buffer_minutes,
):
    """FIX 23's archive-wide shape, now over schedule vintages.

    Compares every scored game's cutoff with EVERY tip-off recorded for
    either team on that date -- not with the resolved value, which is the
    test shape that let two tip-off leaks ship green. It becomes stricter
    every day the schedule job adds a vintage.
    """
    con = open_real_archive_or_skip()
    try:
        games = db.POINT_IN_TIME_TABLES["games"]
        sched = db.POINT_IN_TIME_TABLES["schedule"]
        index = tipoff.tipoff_index(con)
        vintages: dict[tuple[date, str], list[datetime]] = {}
        for gd, home, away, tip in con.execute(
            f"SELECT game_date, home_team, away_team, tip_off_utc FROM {sched} "
            "WHERE tip_off_utc IS NOT NULL"
        ).fetchall():
            vintages.setdefault((gd, home), []).append(tip)
            vintages.setdefault((gd, away), []).append(tip)
        rows = con.execute(
            f"SELECT DISTINCT game_id, game_date, home_team, away_team "
            f"FROM {games} WHERE game_id LIKE '002%'"
        ).fetchall()
        comparisons = 0
        unsafe = []
        for gid, gd, h, a in rows:
            tip = tipoff.resolve_tipoff(index, gd, h, a)
            if tip is None:
                continue
            cutoff = tip - timedelta(minutes=buffer_minutes)
            for v in vintages.get((gd, h), []) + vintages.get((gd, a), []):
                comparisons += 1
                if not cutoff < v:
                    unsafe.append((gid, cutoff, v))
        assert comparisons > 0
        assert unsafe == [], f"{len(unsafe)} unsafe cutoff(s): {unsafe[:5]}"
    finally:
        con.close()
```

- [ ] **Step 3: Create the cross-check test**

Create `tests/test_tipoff_crosscheck.py`:

```python
"""The injury-report PDFs as an independent check on the schedule's tip-offs.

Spec 1.1: the PDF tip-off parser is retained as a cross-check test, not a
runtime fallback -- a second source keeps catching leaks without a second
code path producing cutoffs.

The invariant, measured true for all 7,200 games where both sources exist
(2026-09-27): the schedule's tip-off matches, within 60 seconds, at least
one of
  (a) the most recent PDF vintage filed BEFORE that tip-off (or, when every
      filing came after tip-off, the earliest filing), or
  (b) the earliest time any vintage recorded.
(a) covers games moved LATER: a day-before report showed an earlier slot
and the game-day report, filed before either time, showed the schedule's
(0021900701, 0022000206, 0022400624). (b) covers games moved EARLIER,
where the correcting report came after tip-off (0022000834, 0022200161).
The 60 seconds absorb opening-night ':01' schedule times the PDFs round
(0022000001, 0022200001).

A schedule tip-off later than the real one would fail both (a) and (b):
that is exactly the leak this exists to catch.
"""

from datetime import date, datetime, timedelta

from predictor import db
from predictor.backtest import tipoff
from real_archive import open_real_archive_or_skip

_TOLERANCE = timedelta(seconds=60)


def _close(a: datetime, b: datetime) -> bool:
    return abs(a - b) <= _TOLERANCE


def test_schedule_tipoff_agrees_with_the_injury_reports_wherever_both_exist():
    con = open_real_archive_or_skip()
    try:
        games = db.POINT_IN_TIME_TABLES["games"]
        injuries = db.POINT_IN_TIME_TABLES["injury_status"]
        index = tipoff.tipoff_index(con)

        filings: dict[tuple[date, str], list[tuple[datetime, datetime]]] = {}
        for gd, team, raw, observed in con.execute(
            f"SELECT game_date, team, game_time, observed_at FROM {injuries} "
            "WHERE game_time IS NOT NULL AND game_time <> ''"
        ).fetchall():
            parsed = tipoff.parse_game_time(raw, gd)
            if parsed is not None:
                filings.setdefault((gd, team), []).append((observed, parsed))

        compared = 0
        disagreements = []
        for gid, gd, h, a in con.execute(
            f"SELECT game_id, MIN(game_date), MIN(home_team), MIN(away_team) "
            f"FROM {games} WHERE game_id LIKE '002%' GROUP BY game_id"
        ).fetchall():
            sched = tipoff.resolve_tipoff(index, gd, h, a)
            both = filings.get((gd, h), []) + filings.get((gd, a), [])
            if sched is None or not both:
                continue
            compared += 1
            before = [f for f in both if f[0] < sched]
            pool = before if before else both
            pick = max(o for o, _ in pool) if before else min(o for o, _ in pool)
            candidates = {t for o, t in pool if o == pick}
            earliest = min(t for _, t in both)
            if not (any(_close(sched, t) for t in candidates) or _close(sched, earliest)):
                disagreements.append((gid, sched, sorted(candidates), earliest))

        assert compared >= 7200, f"only {compared} games had both sources"
        assert disagreements == [], (
            f"{len(disagreements)} game(s) where the schedule's tip-off matches "
            f"no injury-report vintage: {disagreements[:5]}"
        )
    finally:
        con.close()
```

- [ ] **Step 4: Migrate the replay and leakage fixtures**

`tests/test_replay.py`:
1. Add `from schedule_rows import insert_schedule_row` after the existing imports.
2. In the `con` fixture, before `return c`, add:
   ```python
       # the tip-off source (sub-project 2.5)
       insert_schedule_row(c, "0022400561", date(2025, 1, 15), "PHI", "NYK", TIP)
   ```
3. In `test_game_without_a_resolvable_tipoff_is_skipped_and_counted` and `test_no_tipoff_skip_is_tracked_by_season`, replace:
   ```python
       i = db.POINT_IN_TIME_TABLES["injury_status"]
       con.execute(f"DELETE FROM {i}")
   ```
   with:
   ```python
       s = db.POINT_IN_TIME_TABLES["schedule"]
       con.execute(f"DELETE FROM {s}")
   ```
4. In `test_predictions_are_in_chronological_order`, inside the `for` loop after the injury insert, add:
   ```python
           insert_schedule_row(con, gid, d, "BOS", "LAL", tip)
   ```
5. In `test_final_row_with_one_null_score_does_not_abort_run`, after the injury insert, add:
   ```python
       insert_schedule_row(con, "0022400999", date(2025, 1, 19), "BOS", "LAL", later_tip)
   ```
6. In both `test_counters_reconcile_with_limit_set` and `test_counters_reconcile_with_no_limit_set`, inside the `for n in range(3)` loop after the injury insert, add:
   ```python
           insert_schedule_row(con, gid, d, "BOS", "LAL", tip)
   ```
7. In `test_season_filter_restricts_the_scored_set`, after the injury insert, add:
   ```python
       insert_schedule_row(
           con, "0022300777", date(2023, 12, 19), "BOS", "LAL", other_tip, season="2023-24"
       )
   ```

The injury inserts stay. They still feed the leakage tests' "which players did the predictor see" assertions, and they now prove that injury times no longer drive the cutoff.

`tests/test_backtest_leakage.py`:
1. Replace `from predictor.backtest import tipoff as tipoff_mod` with `from schedule_rows import insert_schedule_row`.
2. In the `con` fixture, before `return c`, add:
   ```python
       insert_schedule_row(c, "0022400561", date(2025, 1, 15), "PHI", "NYK", TIP)
   ```
3. Replace the whole of `test_every_cutoff_precedes_every_recorded_tipoff_vintage` with:

```python
def test_every_cutoff_precedes_every_recorded_tipoff_vintage(con):
    """FIX 14 (final review, part 3): compare the cutoff with EVERY recorded
    tip-off vintage, never with the resolved tip-off -- a comparison with
    the resolved value holds trivially whatever `resolve_tipoff` returns,
    which is how the FIX 14 CRITICAL regression shipped undetected.

    Since sub-project 2.5 the vintages are schedule vintages. The fixture
    adds a LATER-observed vintage that moves the game EARLIER -- FIX 14's
    exact shape. A resolver trusting the first-seen vintage would cut off
    at 6:30pm ET, after this 5:00pm ET tip-off.
    """
    insert_schedule_row(
        con, "0022400561", date(2025, 1, 15), "PHI", "NYK",
        TIP - timedelta(hours=2), observed_at=TIP - timedelta(hours=3),
    )

    preds, _ = replay.replay(con, always_home)
    assert preds, "fixture produced no predictions"

    s = db.POINT_IN_TIME_TABLES["schedule"]
    vintages = con.execute(
        f"SELECT game_id, tip_off_utc FROM {s} WHERE tip_off_utc IS NOT NULL"
    ).fetchall()

    checked = 0
    for p in preds:
        for game_id, recorded in vintages:
            if game_id != p.game_id:
                continue
            checked += 1
            assert p.cutoff < recorded, (
                f"{p.game_id}: cutoff {p.cutoff.isoformat()} is not before "
                f"recorded vintage {recorded.isoformat()}"
            )
    assert checked >= 2, "fixture did not actually exercise multiple vintages"
```

- [ ] **Step 5: Run to verify failure**

Run: `uv run pytest tests/test_tipoff.py tests/test_replay.py tests/test_backtest_leakage.py -q`
Expected: FAIL. `test_index_keys_both_teams_to_the_tipoff` and similar fail because `tipoff_index` still reads injury reports. `test_injury_report_game_time_is_no_longer_a_tipoff_source` fails too.

- [ ] **Step 6: Implement**

In `src/predictor/backtest/tipoff.py`:

1. Replace the module comment block (the lines from `# This module reads the injury-report point-in-time table directly` through the end of that comment) with:

```python
# This module reads the schedule point-in-time table directly rather than
# through AsOfView. That is a deliberate exemption, not an oversight:
# tip-off resolution ESTABLISHES the as-of cutoff for everything else, so it
# cannot itself be filtered by a cutoff without circularity. The resolved
# tip-off is used only to COMPUTE a cutoff -- it is not carried by
# ``GameToPredict`` (FIX 4) and is never a predictor feature.
#
# Sub-project 2.5: the schedule (sources/schedule.py) replaced the
# injury-report PDFs as the tip-off source. The PDFs' CDN froze after
# 2025-12-21, and they resolved 7,200 of 8,289 games; the schedule
# resolves all 8,289. `parse_game_time` below remains only for
# tests/test_tipoff_crosscheck.py, which checks the two sources against
# each other -- a second source without a second runtime code path
# producing cutoffs, which is where both earlier tip-off leaks lived.
```

2. Change the `parse_game_time` docstring's first line to `"""Resolve an injury-report game time to a UTC instant (cross-check only).`

3. Replace the entire `tipoff_index` function (signature, docstring and body) with:

```python
def tipoff_index(con) -> dict[tuple[date, str], datetime]:
    """Map (game_date, team) -> tip-off instant, from the league schedule.

    Every schedule vintage contributes, and each key keeps the MINIMUM
    instant across them. This is FIX 14's rule, carried over from the
    injury reports: a later vintage that moves a game EARLIER must be
    honoured (ignoring it puts the cutoff after the real tip-off -- a
    leak), while one that moves it LATER is safely ignored (the earlier
    time only makes the cutoff more conservative). The minimum satisfies
    both without knowing which way a change runs.

    Rows whose tip_off_utc is NULL (the league lists the time as TBD) never
    enter the index -- see sources/schedule.py for why a placeholder must
    not.

    Both teams of a game are keyed to it, so ``resolve_tipoff``'s
    minimum-across-both-teams rule (FIX 23) is preserved unchanged.
    """
    table = db.POINT_IN_TIME_TABLES["schedule"]
    rows = con.execute(
        f"SELECT game_date, home_team, away_team, tip_off_utc FROM {table} "
        "WHERE tip_off_utc IS NOT NULL"
    ).fetchall()
    index: dict[tuple[date, str], datetime] = {}
    for game_date, home_team, away_team, tip in rows:
        for team in (home_team, away_team):
            key = (game_date, team)
            current = index.get(key)
            if current is None or tip < current:
                index[key] = tip
    return index
```

4. In the `resolve_tipoff` docstring, change the first line to `"""Tip-off for a game: the MINIMUM across both teams' index entries.`. Leave the FIX 23 history and the body unchanged.

- [ ] **Step 7: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass. The three real-archive tests (`test_every_archived_regular_season_game_resolves_a_tipoff`, the parametrized schedule-vintage test, and `test_schedule_tipoff_agrees_with_the_injury_reports_wherever_both_exist`) must PASS, not SKIP. Check with:

Run: `uv run pytest tests/test_tipoff.py tests/test_tipoff_crosscheck.py -v -rs`
Expected: no `SKIPPED` lines. If a line says "locked by another process", wait a minute and re-run. If it says "no schedule yet", Task 5 was not done.

- [ ] **Step 8: Verify the backtest end to end**

Run: `uv run predictor backtest`
Expected: 8,289 games scored; always-pick-home accuracy **55.2%**; no "no tip-off time could be resolved" line.

Run: `uv run predictor backtest --season 2025-26`
Expected: 1,230 games scored (previously 426).

If either number differs, stop and report it with the full output.

- [ ] **Step 9: Commit**

```bash
git add src/predictor/backtest/tipoff.py tests/schedule_rows.py tests/real_archive.py tests/test_tipoff.py tests/test_tipoff_crosscheck.py tests/test_replay.py tests/test_backtest_leakage.py
git commit -m "feat: resolve tip-offs from the schedule; keep the injury PDFs as a cross-check

Backtest coverage rises from 7,200 to all 8,289 regular-season games, and
the always-pick-home baseline moves from 54.9% to 55.2% -- the same
underlying rate over an honest denominator.

Across all seasons the schedule matches the most recent pre-game injury
report on every one of the 7,200 games both sources cover; the three games
where a day-before report showed an earlier time were moved later, which
the game-day report confirms.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Daily launchd job

**Files:**
- Create: `scripts/com.predictor.schedule.plist`
- Modify: `scripts/install_schedule.sh`
- Test: `tests/test_launchd.py`

**Interfaces:**
- Consumes: the CLI command `predictor ingest-schedule` (Task 4).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_launchd.py`:

```python
import plistlib
import subprocess

from predictor.config import PROJECT_ROOT

SCRIPTS = PROJECT_ROOT / "scripts"


def _load(name):
    return plistlib.loads((SCRIPTS / name).read_bytes())


def _slots(plist):
    return {(d["Hour"], d.get("Minute", 0)) for d in plist["StartCalendarInterval"]}


def test_schedule_job_runs_ingest_schedule_once_a_day():
    p = _load("com.predictor.schedule.plist")
    assert p["Label"] == "com.predictor.schedule"
    assert p["ProgramArguments"] == ["PROJECT_DIR/.venv/bin/predictor", "ingest-schedule"]
    assert _slots(p) == {(10, 30)}
    assert p["StandardOutPath"] == "PROJECT_DIR/data/logs/schedule.out.log"
    assert p["StandardErrorPath"] == "PROJECT_DIR/data/logs/schedule.err.log"


def test_schedule_job_does_not_run_at_load():
    # Installing reloads both jobs at once; firing both immediately would
    # race for DuckDB's write lock on the very first run.
    assert _load("com.predictor.schedule.plist")["RunAtLoad"] is False


def test_schedule_and_news_jobs_never_share_a_start_time():
    assert _slots(_load("com.predictor.schedule.plist")).isdisjoint(
        _slots(_load("com.predictor.daily.plist"))
    )


def test_install_script_installs_both_jobs_and_is_valid_bash():
    script = SCRIPTS / "install_schedule.sh"
    text = script.read_text()
    assert "install_job com.predictor.daily" in text
    assert "install_job com.predictor.schedule" in text
    subprocess.run(["bash", "-n", str(script)], check=True)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_launchd.py -q`
Expected: FAIL with `FileNotFoundError: ... com.predictor.schedule.plist`.

- [ ] **Step 3: Implement**

Create `scripts/com.predictor.schedule.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.predictor.schedule</string>
    <key>ProgramArguments</key>
    <array>
        <string>PROJECT_DIR/.venv/bin/predictor</string>
        <string>ingest-schedule</string>
    </array>
    <key>WorkingDirectory</key>
    <string>PROJECT_DIR</string>
    <key>StartCalendarInterval</key>
    <array>
        <dict><key>Hour</key><integer>10</integer><key>Minute</key><integer>30</integer></dict>
    </array>
    <key>StandardOutPath</key>
    <string>PROJECT_DIR/data/logs/schedule.out.log</string>
    <key>StandardErrorPath</key>
    <string>PROJECT_DIR/data/logs/schedule.err.log</string>
    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
```

Replace `scripts/install_schedule.sh` with:

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$HOME/Library/LaunchAgents" "$PROJECT_DIR/data/logs"

install_job() {
    local label="$1"
    local source="$PROJECT_DIR/scripts/$label.plist"
    local target="$HOME/Library/LaunchAgents/$label.plist"

    # Use Python for safe string replacement with proper XML escaping.
    # Handles special characters (&, |, <, >, etc) that break sed or are
    # invalid in XML. Paths are passed as argv, not interpolated into the
    # Python source text: a path containing a quote, backslash, or
    # triple-quote sequence could otherwise corrupt or inject into the
    # program.
    python3 - "$PROJECT_DIR" "$source" "$target" << 'PYTHON_SCRIPT'
import sys

def escape_xml(s):
    s = s.replace("&", "&amp;")
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    return s

project_dir, source, target = sys.argv[1], sys.argv[2], sys.argv[3]

with open(source, "r") as f:
    plist_content = f.read()

with open(target, "w") as f:
    f.write(plist_content.replace("PROJECT_DIR", escape_xml(project_dir)))
PYTHON_SCRIPT

    launchctl unload "$target" 2>/dev/null || true
    launchctl load "$target"

    # Verify the job actually registered (launchctl load can exit 0 without loading)
    if launchctl list | grep -q "$label"; then
        echo "Successfully installed and loaded: $target"
    else
        echo "ERROR: the scheduled job $label failed to load."
        echo "Check system logs with: log stream --predicate 'eventMessage contains[c] predictor' --level debug"
        exit 1
    fi
}

install_job com.predictor.daily
install_job com.predictor.schedule

echo "The news archiver will run at 09:00, 14:00, and 19:00 daily."
echo "The schedule archiver will run at 10:30 daily."
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Install and verify on this machine**

```bash
bash scripts/install_schedule.sh
launchctl list | grep com.predictor
grep -o "/Users[^<]*" ~/Library/LaunchAgents/com.predictor.schedule.plist | sort -u
```

Expected:
- Both `com.predictor.daily` and `com.predictor.schedule` are listed.
- Every path begins `/Users/meya/projects/predictor/`.

Reinstalling the news job triggers its `RunAtLoad` poll. That is harmless: it is an extra news poll.

Then run the job once by hand, exactly as launchd will:

```bash
launchctl start com.predictor.schedule
```

Wait ~30 seconds, then:

```bash
cat data/logs/schedule.out.log; cat data/logs/schedule.err.log
```

Expected: `schedule 2026-27: ~1,267 games saved` in the out log, and an empty err log.

- [ ] **Step 6: Commit**

```bash
git add scripts/com.predictor.schedule.plist scripts/install_schedule.sh tests/test_launchd.py
git commit -m "feat: archive the schedule daily at 10:30 via its own launchd job

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Correct the spec, update status, merge

**Files:**
- Modify: `docs/superpowers/specs/2026-09-22-nba-predictor-design.md`

- [ ] **Step 1: Correct the spec's cross-validation paragraph**

In section **1.1 Schedule source**, replace the paragraph that begins `**Cross-validation.** Over all 1,230 games of 2023-24` with:

```markdown
**Cross-validation.** Over all 1,230 games of 2023-24, the schedule's tip-off
agrees **exactly** with the PDF-derived time on all 1,229 the PDFs resolve.
Across **all seasons** (measured 2026-09-27, 7,200 games with both sources),
it equals the old PDF resolver on 7,195. The other five are not conflicts:
two are opening-night `:01` times the PDFs round, and three are games moved
*later* whose day-before report showed the old slot while the game-day
report, filed before either time, showed the schedule's. The invariant that
holds for every one of the 7,200 — the schedule matches the most recent
pre-game report (or, for a game moved earlier, the earliest) — is what the
cross-check test asserts.

**Neutral sites.** The league's `isNeutral` flag is false for every game
before 2024-25, including Paris, Mexico City and Las Vegas. The stored
`is_neutral` therefore also marks any game whose arena differs from the home
team's usual regular-season venue that season: the 2019-20 Orlando bubble,
international games, NBA Cup Las Vegas games, and San Antonio's Austin games.
The league's own flag is kept alongside as `is_neutral_reported`.
```

In **Point-in-time status**, append this sentence to the first paragraph: `The fetch runs daily at 10:30 via its own launchd job (com.predictor.schedule), separate from the news job so that neither can block the other.`

- [ ] **Step 2: Final verification**

Run: `uv run pytest -q`
Expected: all pass, with 0 skipped among the tip-off tests. The total should be 347 plus about 45 new tests; record the exact number.

Run: `uv run predictor status`
Expected: a `schedule` line that is fresh. `injury_status` stays STALE (a known, unrelated issue).

- [ ] **Step 3: Commit and merge**

```bash
git add docs/superpowers/specs/2026-09-22-nba-predictor-design.md
git commit -m "docs: correct the schedule cross-validation claim across all seasons

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git checkout main
git merge --no-ff schedule-source -m "Merge schedule-source: league schedule as the tip-off source (sub-project 2.5)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
uv run pytest -q
```

Expected: all pass on `main`.

- [ ] **Step 4: Update project memory**

In `/Users/meya/.claude/projects/-Users-meya-projects-predictor/memory/`, create `project_schedule_source.md` recording:
- Sub-project 2.5 is merged; the test count; the new baseline (8,289 games, 55.2%).
- The daily 10:30 job.
- The `isNeutral` finding.
- That the old "cutoff before every PDF vintage" test was replaced by the cross-check, and why.
- That the gate items from `project_backtest_harness.md` still stand, except gate item 3's neutral-site note, which is now closed.

Add its line to `MEMORY.md`, and update `project_backtest_harness.md`'s baseline numbers to point at it.
