# Prediction Core (Sub-project 3: Stage 1 + Calibration) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a calibrated, fully additive points model. It uses ratings plus home court, rest, travel and altitude. It runs inside the existing backtest harness, is scored on held-out test seasons, and explains every prediction in one sentence whose terms add up.

**Architecture:**
- **Running model.** A new `src/predictor/model/` package holds a stateful `Stage1Predictor` that the harness calls once per game, in date order. On each call it reads, through the `AsOfView` it is handed, only the FINAL results that have become visible since its previous call. It updates team ratings with them, then predicts.
- **Venue facts.** Arena city, neutral site, and each team's previous games come from the schedule table. They are read directly, outside the view, under spec 1.1's static-venue-fact rule.
- **Fitting.** A separate `fit-model` command chooses every setting on the warm-up, fit and calibrate seasons only. It writes one committed JSON settings file.

**Tech Stack:** Python 3.14, DuckDB, numpy (least squares), typer, pytest. Run everything with `uv run`.

**Spec:** `docs/superpowers/specs/2026-09-22-nba-predictor-design.md`, section **3. Prediction core**, especially the subsection **"First delivery design — decided 2026-09-27"**.

## Global Constraints

- Free data sources only; no paid API; no LLM anywhere in the pipeline.
- The user does not write code. CLI messages are plain English and say what to do next. Every failure exits non-zero with no traceback for expected failures.
- Physical `_raw` table names appear as string literals ONLY in `src/predictor/db.py` and `src/predictor/asof.py`. Everything else in `src/` goes through `db.POINT_IN_TIME_TABLES[...]`. Tests may spell them.
- **The model reads results ONLY through the `AsOfView` it is handed, and only rows with `status = 'FINAL'`. It never reads SCHEDULED rows.** This closes the harness's open 2020 play-in item.
- Venue facts (arena city, neutral site, team game dates) are read directly from the schedule table. They are never outcome-bearing, and the model never uses a tip-off time as a feature.
- `fit-model` reads only the warm-up, fit and calibrate seasons, never a test season. This is asserted in code.
- **Season roles** (exact values):
  - Warm-up: `2014-15`, `2015-16`, `2016-17`, `2017-18`, `2018-19`
  - Fit: `2019-20`, `2020-21`, `2021-22`
  - Calibrate: `2022-23`
  - Test: `2023-24`, `2024-25`, `2025-26`
- Competitive game-id prefixes: `002` regular season, `004` playoffs, `005` play-in, `006` NBA Cup knockout. Preseason (`001`) and All-Star (`003`) games never touch ratings, rest or travel.
- Tests must never write `data/predictor.duckdb`. Real-archive tests open it through `tests/real_archive.py::open_real_archive_or_skip()` (read-only; waits out brief locks, then skips).
- Tests compare arithmetic against values worked out by hand in the test itself, never against the code's own output.
- `data/raw/` and `data/predictor.duckdb` are irreplaceable. Back up before any real write.
- Baseline before this plan: `uv run pytest -q` → **415 passed**.

## Terminology used throughout

- **spread**: points by which the model expects the HOME team to win. Positive means the home team is favoured.
- **rating**: a team strength in points. A rating gap of 4 means "4 points better on a neutral floor".
- **Rating update**, after a game:

  ```
  predicted = rating(home) − rating(away) + home_court
  delta = K × (clip(actual_margin, −cap, +cap) − predicted)
  rating(home) += delta
  rating(away) −= delta
  ```

  `home_court` is 0 at a neutral site. The rest, travel and altitude adjustments are NOT part of the update. They are small (about a point each), and keeping them out means ratings do not depend on the fitted coefficients, so the fit is a clean grid search plus a closed-form least squares.
- **home_court**: the mean home margin over the most recent `hca_window` visible non-neutral competitive games. It is 0 before any game is seen.
- **season regression**: when the season changes, every rating is multiplied by `1 − season_regression`.
- **win probability**: `Φ(spread / σ)`, where Φ is the standard normal CDF.

---

### Task 1: Warm-up history in the real archive (operational, no code)

**Why first:** later tasks' real-archive tests need warm-up seasons with results and schedules.

- [ ] **Step 1: Back up**

```bash
cd /Users/meya/projects/predictor
cp data/predictor.duckdb "data/backups/predictor-$(date +%Y%m%d-%H%M%S)-pre-warmup.duckdb"
ls -la data/backups
```

- [ ] **Step 2: Ingest results, then schedules, for the five warm-up seasons**

```bash
for s in 2014-15 2015-16 2016-17 2017-18 2018-19; do
  uv run predictor ingest-season "$s" || echo "FAILED games: $s"
done
for s in 2014-15 2015-16 2016-17 2017-18 2018-19; do
  uv run predictor ingest-schedule --season "$s" || echo "FAILED schedule: $s"
done
```

Expected:
- Each `ingest-season` prints `ingested <n> games for <season>`, with n ≈ 1,300–1,440 (regular season plus postseason plus preseason).
- Each `ingest-schedule` prints about 1,390–1,440 games saved.

Any `DROPPED`, `MISMATCH` or `FAILED` line: stop and report it verbatim; do not continue.

- [ ] **Step 3: Verify**

```bash
uv run python - <<'EOF'
import duckdb
con = duckdb.connect("data/predictor.duckdb", read_only=True)
print(con.execute("""
  SELECT season, count(DISTINCT game_id) FROM games_raw
  WHERE game_id LIKE '002%' GROUP BY season ORDER BY season""").fetchall())
print("regular-season games without a tip-off:", con.execute("""
  SELECT count(*) FROM (SELECT DISTINCT game_id FROM games_raw WHERE game_id LIKE '002%') g
  WHERE g.game_id NOT IN (SELECT game_id FROM schedule_raw WHERE tip_off_utc IS NOT NULL)""").fetchone()[0])
EOF
uv run pytest -q
```

Expected:
- 1,230 regular-season games in each of the 12 seasons from 2014-15 to 2025-26. The one exception is 2019-20 with 1,059 and 2020-21 with 1,080.
- `without a tip-off: 0`
- The suite shows 415 passed.

Nothing to commit (`data/` is git-ignored). Record the output in the ledger.

---

### Task 2: City table and venue index

**Files:**
- Create: `src/predictor/model/__init__.py` (empty), `src/predictor/model/cities.py`, `src/predictor/model/venues.py`
- Test: `tests/test_model_cities.py`, `tests/test_model_venues.py`, `tests/model_fixtures.py`

**Interfaces:**
- Produces from `cities.py`:
  - `City(lat: float, lon: float, tz: str)` (frozen)
  - `CITIES: dict[str, City]`
  - `ALTITUDE_CITIES: frozenset[str]`
  - `city_key(arena_city: str | None) -> str | None`
  - `distance_km(a: City, b: City) -> float`
  - `utc_offset_hours(city: City, on: date) -> float`
- Produces from `venues.py`:
  - `COMPETITIVE_PREFIXES = ("002", "004", "005", "006")`
  - `Venue(game_id: str, game_date: date, home_team: str, away_team: str, city: str | None, is_neutral: bool)` (frozen)
  - `VenueIndex(venues: Iterable[Venue])`, with `VenueIndex.from_db(con) -> VenueIndex`, `.venue(game_id: str) -> Venue | None` and `.recent_games(team: str, before: date, n: int) -> list[Venue]` (up to n most recent competitive games strictly before `before`, oldest first)
- Produces from `tests/model_fixtures.py`: `add_game(con, game_id, season, game_date, home, away, home_pts=None, away_pts=None, *, city="Boston", neutral=False, final_observed_at=None)` and `fixture_con(tmp_path)`.

- [ ] **Step 1: Write the fixture helper**

Create `tests/model_fixtures.py`:

```python
"""Build small, fully controlled archives for model tests.

Each game gets a SCHEDULED row (7 days before, reconstructed), a FINAL row
when scores are given (game_date + 1 day, 12:00 UTC -- the same stamp the
real archive uses), and a schedule row (7pm ET tip-off, observed
2026-09-27, like the real backfill).
"""

from datetime import UTC, date, datetime, time, timedelta

from predictor import db

SCHEDULE_OBSERVED = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)


def fixture_con(tmp_path):
    con = db.connect(tmp_path / "model.duckdb")
    db.migrate(con)
    return con


def add_game(
    con,
    game_id,
    season,
    game_date,
    home,
    away,
    home_pts=None,
    away_pts=None,
    *,
    city="Boston",
    neutral=False,
    final_observed_at=None,
):
    games = db.POINT_IN_TIME_TABLES["games"]
    sched = db.POINT_IN_TIME_TABLES["schedule"]
    scheduled_at = datetime.combine(game_date - timedelta(days=7), time(12), tzinfo=UTC)
    con.execute(
        f"INSERT INTO {games} (game_id, season, game_date, home_team, away_team,"
        " home_points, away_points, status, reconstructed, observed_at)"
        " VALUES (?,?,?,?,?,NULL,NULL,'SCHEDULED',TRUE,?)",
        [game_id, season, game_date, home, away, scheduled_at],
    )
    if home_pts is not None:
        if final_observed_at is None:
            final_observed_at = datetime.combine(
                game_date + timedelta(days=1), time(12), tzinfo=UTC
            )
        con.execute(
            f"INSERT INTO {games} (game_id, season, game_date, home_team, away_team,"
            " home_points, away_points, status, reconstructed, observed_at)"
            " VALUES (?,?,?,?,?,?,?,'FINAL',TRUE,?)",
            [game_id, season, game_date, home, away, home_pts, away_pts, final_observed_at],
        )
    tip = datetime.combine(game_date + timedelta(days=1), time(0), tzinfo=UTC)
    con.execute(
        f"INSERT INTO {sched} (game_id, season, game_date, tip_off_utc, home_team,"
        " away_team, arena_name, arena_city, arena_state, is_neutral_reported,"
        " is_neutral, observed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [game_id, season, game_date, tip, home, away, "Arena", city, None,
         neutral, neutral, SCHEDULE_OBSERVED],
    )
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_model_cities.py`:

```python
import math
from datetime import date

import pytest

from predictor.model import cities
from predictor.model.cities import City


def test_one_degree_of_longitude_on_the_equator():
    # 2 * pi * 6371.0 / 360 = 111.19492664...
    d = cities.distance_km(City(0.0, 0.0, "UTC"), City(0.0, 1.0, "UTC"))
    assert d == pytest.approx(111.19492664, abs=1e-6)


def test_distance_to_self_is_zero():
    boston = cities.CITIES["Boston"]
    assert cities.distance_km(boston, boston) == 0.0


def test_distance_is_symmetric():
    a, b = cities.CITIES["Boston"], cities.CITIES["Los Angeles"]
    assert cities.distance_km(a, b) == pytest.approx(cities.distance_km(b, a))


def test_new_york_offset_follows_daylight_saving():
    ny = cities.CITIES["New York"]
    assert cities.utc_offset_hours(ny, date(2025, 1, 15)) == -5.0
    assert cities.utc_offset_hours(ny, date(2025, 7, 15)) == -4.0


def test_phoenix_does_not_observe_daylight_saving():
    phx = cities.CITIES["Phoenix"]
    assert cities.utc_offset_hours(phx, date(2025, 1, 15)) == -7.0
    assert cities.utc_offset_hours(phx, date(2025, 7, 15)) == -7.0


@pytest.mark.parametrize(
    "raw, key",
    [
        ("Mexico City, Mexico", "Mexico City"),
        ("Mexico City", "Mexico City"),
        ("  Boston ", "Boston"),
        ("", None),
        (None, None),
    ],
)
def test_city_key_normalises_the_schedule_text(raw, key):
    assert cities.city_key(raw) == key


def test_altitude_cities():
    assert cities.ALTITUDE_CITIES == {"Denver", "Salt Lake City"}
    assert all(c in cities.CITIES for c in cities.ALTITUDE_CITIES)


def test_every_city_has_a_real_time_zone():
    for name, city in cities.CITIES.items():
        assert isinstance(cities.utc_offset_hours(city, date(2025, 1, 15)), float), name
        assert -90 <= city.lat <= 90 and -180 <= city.lon <= 180, name
```

Create `tests/test_model_venues.py`:

```python
from datetime import date

from model_fixtures import add_game, fixture_con
from predictor.model import cities
from predictor.model.venues import Venue, VenueIndex
from real_archive import open_real_archive_or_skip


def test_from_db_reads_city_and_neutral(tmp_path):
    con = fixture_con(tmp_path)
    add_game(con, "0022400001", "2024-25", date(2025, 1, 15), "PHI", "NYK",
             city="Mexico City, Mexico", neutral=True)
    v = VenueIndex.from_db(con).venue("0022400001")
    assert v == Venue("0022400001", date(2025, 1, 15), "PHI", "NYK", "Mexico City", True)


def test_unknown_game_has_no_venue(tmp_path):
    assert VenueIndex.from_db(fixture_con(tmp_path)).venue("0029999999") is None


def test_latest_schedule_vintage_wins(tmp_path):
    con = fixture_con(tmp_path)
    add_game(con, "0022400001", "2024-25", date(2025, 1, 15), "PHI", "NYK", city="Boston")
    from predictor import db
    from datetime import UTC, datetime
    sched = db.POINT_IN_TIME_TABLES["schedule"]
    con.execute(
        f"INSERT INTO {sched} (game_id, season, game_date, tip_off_utc, home_team,"
        " away_team, arena_city, is_neutral_reported, is_neutral, observed_at)"
        " VALUES ('0022400001','2024-25',DATE '2025-01-15',NULL,'PHI','NYK',"
        " 'Paris',TRUE,TRUE,?)",
        [datetime(2026, 9, 28, tzinfo=UTC)],
    )
    v = VenueIndex.from_db(con).venue("0022400001")
    assert v.city == "Paris" and v.is_neutral is True


def test_recent_games_are_strictly_before_and_oldest_first():
    vs = [
        Venue("0022400001", date(2025, 1, 10), "PHI", "NYK", "Philadelphia", False),
        Venue("0022400002", date(2025, 1, 12), "BOS", "PHI", "Boston", False),
        Venue("0022400003", date(2025, 1, 14), "PHI", "MIA", "Philadelphia", False),
        Venue("0022400004", date(2025, 1, 15), "PHI", "CHI", "Philadelphia", False),
    ]
    idx = VenueIndex(vs)
    got = idx.recent_games("PHI", date(2025, 1, 15), n=2)
    assert [v.game_id for v in got] == ["0022400002", "0022400003"]
    assert idx.recent_games("PHI", date(2025, 1, 10), n=2) == []
    assert idx.recent_games("LAL", date(2025, 1, 15), n=2) == []


def test_preseason_and_all_star_games_are_not_history():
    vs = [
        Venue("0012400001", date(2025, 1, 12), "PHI", "NYK", "Philadelphia", False),
        Venue("0032400001", date(2025, 1, 13), "PHI", "NYK", "Philadelphia", False),
        Venue("0042400001", date(2025, 1, 14), "PHI", "NYK", "Philadelphia", False),
    ]
    got = VenueIndex(vs).recent_games("PHI", date(2025, 1, 15), n=5)
    assert [v.game_id for v in got] == ["0042400001"]


def test_every_competitive_arena_city_in_the_archive_is_in_the_city_table():
    con = open_real_archive_or_skip()
    try:
        from predictor import db
        sched = db.POINT_IN_TIME_TABLES["schedule"]
        rows = con.execute(
            f"SELECT DISTINCT arena_city FROM {sched} "
            "WHERE substr(game_id, 1, 3) IN ('002','004','005','006')"
        ).fetchall()
        missing = sorted({cities.city_key(r[0]) for r in rows} - set(cities.CITIES))
        assert missing == [], f"add these arena cities to cities.CITIES: {missing}"
    finally:
        con.close()
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_model_cities.py tests/test_model_venues.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'predictor.model'`.

- [ ] **Step 4: Implement**

Create an empty `src/predictor/model/__init__.py`.

Create `src/predictor/model/cities.py`:

```python
"""NBA arena cities: coordinates, time zones, altitude.

Static reference data for the travel and altitude adjustments (spec 3,
"First delivery design"). Keyed by the schedule's `arena_city` with any
", Country" suffix removed ("Mexico City, Mexico" -> "Mexico City").
Coordinates are the arena's, to three decimals. Every competitive-game city
in the archive from 2014-15 on is covered (enforced by
tests/test_model_venues.py against the real archive).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

_EARTH_RADIUS_KM = 6371.0


@dataclass(frozen=True)
class City:
    lat: float
    lon: float
    tz: str


CITIES: dict[str, City] = {
    "Atlanta": City(33.757, -84.396, "America/New_York"),
    "Austin": City(30.282, -97.732, "America/Chicago"),
    "Berlin": City(52.508, 13.443, "Europe/Berlin"),
    "Boston": City(42.366, -71.062, "America/New_York"),
    "Brooklyn": City(40.683, -73.975, "America/New_York"),
    "Charlotte": City(35.225, -80.839, "America/New_York"),
    "Chicago": City(41.881, -87.674, "America/Chicago"),
    "Cleveland": City(41.496, -81.688, "America/New_York"),
    "Dallas": City(32.790, -96.810, "America/Chicago"),
    "Denver": City(39.749, -105.008, "America/Denver"),
    "Detroit": City(42.341, -83.055, "America/Detroit"),
    "Houston": City(29.751, -95.362, "America/Chicago"),
    "Indianapolis": City(39.764, -86.155, "America/Indiana/Indianapolis"),
    "Inglewood": City(33.945, -118.343, "America/Los_Angeles"),
    "Las Vegas": City(36.103, -115.178, "America/Los_Angeles"),
    "London": City(51.503, 0.003, "Europe/London"),
    "Los Angeles": City(34.043, -118.267, "America/Los_Angeles"),
    "Manchester": City(53.486, -2.199, "Europe/London"),
    "Memphis": City(35.138, -90.051, "America/Chicago"),
    "Mexico City": City(19.404, -99.096, "America/Mexico_City"),
    "Miami": City(25.781, -80.188, "America/New_York"),
    "Milwaukee": City(43.045, -87.917, "America/Chicago"),
    "Minneapolis": City(44.979, -93.276, "America/Chicago"),
    "New Orleans": City(29.949, -90.082, "America/Chicago"),
    "New York": City(40.751, -73.993, "America/New_York"),
    "Oakland": City(37.750, -122.203, "America/Los_Angeles"),
    "Oklahoma City": City(35.463, -97.515, "America/Chicago"),
    "Orlando": City(28.539, -81.384, "America/New_York"),
    "Paris": City(48.838, 2.379, "Europe/Paris"),
    "Philadelphia": City(39.901, -75.172, "America/New_York"),
    "Phoenix": City(33.446, -112.071, "America/Phoenix"),
    "Portland": City(45.532, -122.667, "America/Los_Angeles"),
    "Sacramento": City(38.580, -121.500, "America/Los_Angeles"),
    "Salt Lake City": City(40.768, -111.901, "America/Denver"),
    "San Antonio": City(29.427, -98.438, "America/Chicago"),
    "San Francisco": City(37.768, -122.388, "America/Los_Angeles"),
    "Tampa": City(27.943, -82.452, "America/New_York"),
    "Toronto": City(43.643, -79.379, "America/Toronto"),
    "Washington": City(38.898, -77.021, "America/New_York"),
}

# Arenas high enough that visitors are measurably affected.
ALTITUDE_CITIES: frozenset[str] = frozenset({"Denver", "Salt Lake City"})


def city_key(arena_city: str | None) -> str | None:
    if arena_city is None:
        return None
    key = arena_city.split(",")[0].strip()
    return key or None


def distance_km(a: City, b: City) -> float:
    """Great-circle (haversine) distance."""
    lat1, lon1, lat2, lon2 = map(math.radians, (a.lat, a.lon, b.lat, b.lon))
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def utc_offset_hours(city: City, on: date) -> float:
    """The city's UTC offset at noon local time on `on`, in hours."""
    local = datetime.combine(on, time(12, 0), tzinfo=ZoneInfo(city.tz))
    return local.utcoffset().total_seconds() / 3600
```

Create `src/predictor/model/venues.py`:

```python
"""Static venue facts from the schedule, read outside AsOfView.

Spec 1.1: arena and neutral-site columns are static venue facts, not
outcome-bearing, safe to read at any time. They cannot come through
AsOfView: every schedule row was observed on 2026-09-27 or later, so a view
cut at any historical cutoff would hide them all. A team's past game DATES
are likewise facts of games already played. This module never reads scores
or game status, and never exposes a tip-off time.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

from predictor import db
from predictor.model.cities import city_key

COMPETITIVE_PREFIXES = ("002", "004", "005", "006")


@dataclass(frozen=True)
class Venue:
    game_id: str
    game_date: date
    home_team: str
    away_team: str
    city: str | None
    is_neutral: bool


class VenueIndex:
    def __init__(self, venues: Iterable[Venue]) -> None:
        self._by_id: dict[str, Venue] = {v.game_id: v for v in venues}
        self._by_team: dict[str, list[Venue]] = {}
        ordered = sorted(self._by_id.values(), key=lambda v: (v.game_date, v.game_id))
        for v in ordered:
            if v.game_id[:3] not in COMPETITIVE_PREFIXES:
                continue
            self._by_team.setdefault(v.home_team, []).append(v)
            self._by_team.setdefault(v.away_team, []).append(v)
        self._dates = {t: [v.game_date for v in vs] for t, vs in self._by_team.items()}

    @classmethod
    def from_db(cls, con) -> VenueIndex:
        table = db.POINT_IN_TIME_TABLES["schedule"]
        rows = con.execute(
            f"SELECT game_id, game_date, home_team, away_team, arena_city, is_neutral "
            f"FROM {table} "
            "QUALIFY row_number() OVER (PARTITION BY game_id ORDER BY observed_at DESC) = 1"
        ).fetchall()
        return cls(
            Venue(gid, gd, home, away, city_key(city), bool(neutral))
            for gid, gd, home, away, city, neutral in rows
        )

    def venue(self, game_id: str) -> Venue | None:
        return self._by_id.get(game_id)

    def recent_games(self, team: str, before: date, n: int) -> list[Venue]:
        """Up to `n` most recent competitive games strictly before `before`, oldest first."""
        dates = self._dates.get(team)
        if not dates:
            return []
        end = bisect_left(dates, before)
        return self._by_team[team][max(0, end - n):end]
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_model_cities.py tests/test_model_venues.py -v -rs` and then `uv run pytest -q`
Expected: all pass. The real-archive city-coverage test must PASS, not SKIP (Task 1 ingested the warm-up schedules).

- [ ] **Step 6: Commit**

```bash
git add src/predictor/model tests/model_fixtures.py tests/test_model_cities.py tests/test_model_venues.py
git commit -m "feat: add the arena city table and a static venue index for the model

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Ratings and home court

**Files:**
- Create: `src/predictor/model/ratings.py`
- Test: `tests/test_model_ratings.py`

**Interfaces:**
- Produces:
  - `RatingParams(k: float, margin_cap: float, season_regression: float, hca_window: int)` (frozen)
  - `Result(game_id: str, season: str, game_date: date, home_team: str, away_team: str, home_points: int, away_points: int, neutral: bool)` (frozen)
  - `Ratings(params)`, with `.rating(team) -> float`, `.home_court() -> float`, `.enter_season(season: str) -> None` and `.apply(result: Result) -> None`
  - `win_probability(spread: float, sigma: float) -> float`

- [ ] **Step 1: Write failing tests**

Create `tests/test_model_ratings.py`:

```python
from datetime import date

import pytest

from predictor.model.ratings import RatingParams, Ratings, Result, win_probability

P = RatingParams(k=0.1, margin_cap=20.0, season_regression=0.5, hca_window=2)


def _r(home, away, hp, ap, season="2024-25", neutral=False, gid="0022400001"):
    return Result(gid, season, date(2025, 1, 15), home, away, hp, ap, neutral)


def test_unknown_team_is_average():
    assert Ratings(P).rating("PHI") == 0.0


def test_first_game_update_by_hand():
    # home_court is 0 before any game; predicted = 0; margin 10.
    # delta = 0.1 * (10 - 0) = 1.0
    r = Ratings(P)
    r.apply(_r("PHI", "NYK", 110, 100))
    assert r.rating("PHI") == pytest.approx(1.0)
    assert r.rating("NYK") == pytest.approx(-1.0)


def test_second_game_uses_ratings_and_home_court_by_hand():
    r = Ratings(P)
    r.apply(_r("PHI", "NYK", 110, 100))  # PHI +1, NYK -1, home margins [10]
    # predicted = 1 - (-1) + 10 = 12; margin -4; delta = 0.1 * (-4 - 12) = -1.6
    r.apply(_r("PHI", "NYK", 100, 104))
    assert r.rating("PHI") == pytest.approx(-0.6)
    assert r.rating("NYK") == pytest.approx(0.6)
    # home margins [10, -4] -> mean 3.0
    assert r.home_court() == pytest.approx(3.0)


def test_blowout_is_capped_by_hand():
    r = Ratings(P)
    r.apply(_r("PHI", "NYK", 150, 100))  # margin 50 capped to 20; delta 2.0
    assert r.rating("PHI") == pytest.approx(2.0)
    # but home court uses the real margin
    assert r.home_court() == pytest.approx(50.0)


def test_neutral_site_has_no_home_court_and_does_not_feed_it():
    r = Ratings(P)
    r.apply(_r("PHI", "NYK", 110, 100))           # home margins [10]
    r.apply(_r("BOS", "MIA", 100, 100, neutral=True))
    # neutral: predicted = 0 - 0 + 0 = 0; margin 0; delta 0
    assert r.rating("BOS") == 0.0
    assert r.home_court() == pytest.approx(10.0)


def test_home_court_window_keeps_only_the_most_recent_games():
    r = Ratings(P)  # window 2
    for margin in (10, 20, 30):
        r.apply(_r("PHI", "NYK", 100 + margin, 100))
    assert r.home_court() == pytest.approx(25.0)


def test_new_season_regresses_every_rating_by_hand():
    r = Ratings(P)
    r.apply(_r("PHI", "NYK", 110, 100))  # +1 / -1
    r.enter_season("2025-26")
    assert r.rating("PHI") == pytest.approx(0.5)
    assert r.rating("NYK") == pytest.approx(-0.5)


def test_entering_the_same_season_twice_regresses_once():
    r = Ratings(P)
    r.apply(_r("PHI", "NYK", 110, 100))
    r.enter_season("2025-26")
    r.enter_season("2025-26")
    assert r.rating("PHI") == pytest.approx(0.5)


def test_apply_enters_the_results_season():
    r = Ratings(P)
    r.apply(_r("PHI", "NYK", 110, 100))
    r.apply(_r("BOS", "MIA", 100, 100, season="2025-26", gid="0022500001"))
    assert r.rating("PHI") == pytest.approx(0.5)


def test_win_probability_by_hand():
    assert win_probability(0.0, 13.0) == pytest.approx(0.5)
    # Phi(1) = 0.841344746...
    assert win_probability(13.0, 13.0) == pytest.approx(0.8413447461, abs=1e-9)
    assert win_probability(-13.0, 13.0) == pytest.approx(0.1586552539, abs=1e-9)


def test_win_probability_rejects_nonpositive_sigma():
    with pytest.raises(ValueError):
        win_probability(1.0, 0.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_model_ratings.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `src/predictor/model/ratings.py`:

```python
"""Team ratings in points, updated on margin of victory (spec 3, Stage 1).

A rating gap of 4 means "4 points better on a neutral floor". After each
game both teams move by K x (capped actual margin - predicted margin), where
predicted = rating gap + home court. Rest, travel and altitude are NOT part
of the update: they are small, and leaving them out keeps ratings
independent of the fitted adjustment coefficients.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class RatingParams:
    k: float
    margin_cap: float
    season_regression: float
    hca_window: int


@dataclass(frozen=True)
class Result:
    game_id: str
    season: str
    game_date: date
    home_team: str
    away_team: str
    home_points: int
    away_points: int
    neutral: bool


class Ratings:
    def __init__(self, params: RatingParams) -> None:
        self.params = params
        self._ratings: dict[str, float] = {}
        self._season: str | None = None
        self._home_margins: deque[int] = deque(maxlen=params.hca_window)

    def rating(self, team: str) -> float:
        return self._ratings.get(team, 0.0)

    def home_court(self) -> float:
        """Mean home margin over the most recent `hca_window` non-neutral games."""
        if not self._home_margins:
            return 0.0
        return sum(self._home_margins) / len(self._home_margins)

    def enter_season(self, season: str) -> None:
        """Regress every rating toward 0 the first time a new season is seen."""
        if self._season is not None and season != self._season:
            keep = 1.0 - self.params.season_regression
            self._ratings = {t: r * keep for t, r in self._ratings.items()}
        self._season = season

    def apply(self, result: Result) -> None:
        self.enter_season(result.season)
        home_court = 0.0 if result.neutral else self.home_court()
        predicted = (
            self.rating(result.home_team) - self.rating(result.away_team) + home_court
        )
        margin = result.home_points - result.away_points
        cap = self.params.margin_cap
        delta = self.params.k * (max(-cap, min(cap, margin)) - predicted)
        self._ratings[result.home_team] = self.rating(result.home_team) + delta
        self._ratings[result.away_team] = self.rating(result.away_team) - delta
        if not result.neutral:
            self._home_margins.append(margin)


def win_probability(spread: float, sigma: float) -> float:
    """P(home wins) = Phi(spread / sigma)."""
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    return 0.5 * (1.0 + math.erf(spread / (sigma * math.sqrt(2.0))))
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_model_ratings.py -q` and then `uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/predictor/model/ratings.py tests/test_model_ratings.py
git commit -m "feat: add point-based team ratings with a running home-court estimate

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Rest, travel and altitude adjustments

**Files:**
- Create: `src/predictor/model/adjustments.py`
- Test: `tests/test_model_adjustments.py`

**Interfaces:**
- Consumes: `VenueIndex.recent_games`, `Venue` (Task 2); `CITIES`, `ALTITUDE_CITIES`, `distance_km`, `utc_offset_hours` (Task 2).
- Produces:
  - `TRAVEL_GAP_DAYS = 7`
  - `Situation(back_to_back: bool, third_in_four: bool, travel_km: float, tz_hours: float, has_history: bool, unknown_city: str | None)` (frozen)
  - `situation(venues: VenueIndex, team: str, game_date: date, city: str | None) -> Situation`
  - `is_altitude_game(city: str | None, neutral: bool) -> bool`
  - `FEATURE_NAMES = ("back_to_back", "third_in_four", "travel_per_1000km", "tz_per_hour", "altitude")`
  - `feature_vector(home: Situation, away: Situation, altitude_game: bool) -> tuple[float, float, float, float, float]`
  - `Coefficients(back_to_back, third_in_four, travel_per_1000km, tz_per_hour, altitude)` (frozen floats)
  - `AdjustmentTerms(rest: float, travel: float, altitude: float)` (frozen)
  - `terms(c: Coefficients, x: tuple[float, ...]) -> AdjustmentTerms`

- [ ] **Step 1: Write failing tests**

Create `tests/test_model_adjustments.py`:

```python
from datetime import date

import pytest

from predictor.model import adjustments as adj
from predictor.model.cities import CITIES, distance_km
from predictor.model.venues import Venue, VenueIndex


def _v(gid, d, home, away, city):
    return Venue(gid, d, home, away, city, False)


IDX = VenueIndex([
    _v("0022400001", date(2025, 1, 10), "PHI", "NYK", "Philadelphia"),
    _v("0022400002", date(2025, 1, 12), "BOS", "PHI", "Boston"),
    _v("0022400003", date(2025, 1, 13), "PHI", "MIA", "Philadelphia"),
    _v("0022400004", date(2025, 1, 30), "LAL", "NYK", "Los Angeles"),
    _v("0022400005", date(2025, 1, 12), "NYK", "ORL", "Atlantis"),
])


def test_no_history_is_rested_and_untravelled():
    s = adj.situation(IDX, "CHI", date(2025, 1, 14), "Chicago")
    assert s == adj.Situation(False, False, 0.0, 0.0, has_history=False, unknown_city=None)


def test_back_to_back_and_third_in_four_by_hand():
    # PHI played 01-12 (Boston) and 01-13 (Philadelphia); game on 01-14.
    s = adj.situation(IDX, "PHI", date(2025, 1, 14), "Philadelphia")
    assert s.back_to_back is True          # previous game 1 day earlier
    assert s.third_in_four is True         # 01-12 and 01-13 are within 3 days
    assert s.travel_km == 0.0              # Philadelphia -> Philadelphia
    assert s.tz_hours == 0.0


def test_two_days_off_is_neither_by_hand():
    s = adj.situation(IDX, "PHI", date(2025, 1, 16), "Philadelphia")
    assert s.back_to_back is False
    assert s.third_in_four is False        # 01-12 is 4 days before 01-16


def test_travel_distance_and_time_zones_by_hand():
    # PHI's last game was in Philadelphia (01-13); now playing in Los Angeles.
    s = adj.situation(IDX, "PHI", date(2025, 1, 15), "Los Angeles")
    expected = distance_km(CITIES["Philadelphia"], CITIES["Los Angeles"])
    assert s.travel_km == pytest.approx(expected)
    assert s.tz_hours == 3.0               # UTC-5 -> UTC-8 in January


def test_long_break_means_no_travel():
    # NYK's last game before 01-30 is 01-12, 18 days earlier.
    s = adj.situation(IDX, "NYK", date(2025, 1, 30), "Los Angeles")
    assert s.travel_km == 0.0 and s.tz_hours == 0.0 and s.has_history is True


def test_unknown_city_is_reported_not_guessed():
    # NYK's previous game (01-12) was in "Atlantis", not in the table.
    s = adj.situation(IDX, "NYK", date(2025, 1, 14), "New York")
    assert s.travel_km == 0.0 and s.tz_hours == 0.0
    assert s.unknown_city == "Atlantis"


def test_missing_game_city_is_reported():
    s = adj.situation(IDX, "PHI", date(2025, 1, 15), None)
    assert s.unknown_city == "(no city recorded)"


def test_altitude_game():
    assert adj.is_altitude_game("Denver", neutral=False) is True
    assert adj.is_altitude_game("Salt Lake City", neutral=False) is True
    assert adj.is_altitude_game("Denver", neutral=True) is False
    assert adj.is_altitude_game("Boston", neutral=False) is False
    assert adj.is_altitude_game(None, neutral=False) is False


def test_feature_vector_is_home_minus_away_by_hand():
    home = adj.Situation(True, False, 1500.0, 2.0, True, None)
    away = adj.Situation(False, True, 500.0, 3.0, True, None)
    assert adj.feature_vector(home, away, altitude_game=True) == (1.0, -1.0, 1.0, -1.0, 1.0)


def test_terms_by_hand():
    c = adj.Coefficients(back_to_back=-2.0, third_in_four=-1.0,
                         travel_per_1000km=-0.5, tz_per_hour=-0.25, altitude=1.5)
    t = adj.terms(c, (1.0, -1.0, 2.0, -1.0, 1.0))
    assert t.rest == pytest.approx(-2.0 * 1 + -1.0 * -1)       # -1.0
    assert t.travel == pytest.approx(-0.5 * 2 + -0.25 * -1)    # -0.75
    assert t.altitude == pytest.approx(1.5)


def test_feature_names_match_coefficients():
    import dataclasses
    assert adj.FEATURE_NAMES == tuple(f.name for f in dataclasses.fields(adj.Coefficients))
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_model_adjustments.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `src/predictor/model/adjustments.py`:

```python
"""Rest, travel and altitude adjustments, in points (spec 3, Stage 1).

Every input is a fact of games already played (dates and arena cities from
the schedule), so nothing here can see a result. Features are expressed
home-minus-away, so one coefficient per feature gives the home team's edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from predictor.model.cities import ALTITUDE_CITIES, CITIES, distance_km, utc_offset_hours
from predictor.model.venues import VenueIndex

# Beyond this many days since the previous game (All-Star break, season
# start), a team is treated as travelling from home: no travel penalty.
TRAVEL_GAP_DAYS = 7

FEATURE_NAMES = ("back_to_back", "third_in_four", "travel_per_1000km", "tz_per_hour", "altitude")


@dataclass(frozen=True)
class Situation:
    back_to_back: bool
    third_in_four: bool
    travel_km: float
    tz_hours: float
    has_history: bool
    unknown_city: str | None


@dataclass(frozen=True)
class Coefficients:
    back_to_back: float
    third_in_four: float
    travel_per_1000km: float
    tz_per_hour: float
    altitude: float


@dataclass(frozen=True)
class AdjustmentTerms:
    rest: float
    travel: float
    altitude: float


def situation(venues: VenueIndex, team: str, game_date: date, city: str | None) -> Situation:
    recent = venues.recent_games(team, game_date, n=2)
    if not recent:
        return Situation(False, False, 0.0, 0.0, has_history=False, unknown_city=None)
    previous = recent[-1]
    gap = (game_date - previous.game_date).days
    back_to_back = gap == 1
    # This game is the third in four nights when the two previous games both
    # fall within the three days before it.
    third_in_four = len(recent) == 2 and (game_date - recent[0].game_date).days <= 3
    travel_km = tz_hours = 0.0
    unknown: str | None = None
    if gap <= TRAVEL_GAP_DAYS:
        start = CITIES.get(previous.city) if previous.city else None
        end = CITIES.get(city) if city else None
        if start is None or end is None:
            unknown = (previous.city if start is None else city) or "(no city recorded)"
        else:
            travel_km = distance_km(start, end)
            tz_hours = abs(utc_offset_hours(start, game_date) - utc_offset_hours(end, game_date))
    return Situation(back_to_back, third_in_four, travel_km, tz_hours, True, unknown)


def is_altitude_game(city: str | None, neutral: bool) -> bool:
    return (not neutral) and city in ALTITUDE_CITIES


def feature_vector(
    home: Situation, away: Situation, altitude_game: bool
) -> tuple[float, float, float, float, float]:
    return (
        float(home.back_to_back) - float(away.back_to_back),
        float(home.third_in_four) - float(away.third_in_four),
        (home.travel_km - away.travel_km) / 1000.0,
        home.tz_hours - away.tz_hours,
        1.0 if altitude_game else 0.0,
    )


def terms(c: Coefficients, x: tuple[float, ...]) -> AdjustmentTerms:
    return AdjustmentTerms(
        rest=c.back_to_back * x[0] + c.third_in_four * x[1],
        travel=c.travel_per_1000km * x[2] + c.tz_per_hour * x[3],
        altitude=c.altitude * x[4],
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_model_adjustments.py -q` and then `uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/predictor/model/adjustments.py tests/test_model_adjustments.py
git commit -m "feat: add rest, travel and altitude adjustments from past games only

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Settings file and the Stage 1 predictor

**Files:**
- Create: `src/predictor/model/settings.py`, `src/predictor/model/stage1.py`
- Test: `tests/test_model_settings.py`, `tests/test_model_stage1.py`

**Interfaces:**
- Consumes: Tasks 2–4. `AsOfView.table("games")` returns a DuckDB relation supporting `.filter(str)`, `.project(str)`, `.order(str)` and `.fetchall()`. `view.as_of` is a UTC datetime. `GameToPredict(game_id, season, game_date, home_team, away_team)` comes from `predictor.backtest.baselines`, and `PredictionError` from the same module.
- Produces from `settings.py`:
  - `WARMUP_SEASONS`, `FIT_SEASONS`, `CALIBRATE_SEASON`, `TEST_SEASONS` (exact values from Global Constraints)
  - `season_role(season: str) -> str`, returning `"warm-up" | "fit" | "calibrate" | "test" | "unassigned"`
  - `SETTINGS_PATH: Path` = `src/predictor/model/stage1_settings.json`
  - `SettingsError(Exception)`
  - `ModelSettings(ratings: RatingParams, coefficients: Coefficients, sigma: float, fit_games: int, calibrate_games: int)` (frozen)
  - `to_json(s) -> str`, `from_json(text: str) -> ModelSettings`, `load(path: Path = SETTINGS_PATH) -> ModelSettings`, `save(s, path: Path = SETTINGS_PATH) -> None`
- Produces from `stage1.py`:
  - `Breakdown(game_id, home_team, away_team, rating, home, rest, travel, altitude, spread, p_home)` (frozen), with `.terms() -> tuple[tuple[str, float], ...]` and `.sentence() -> str`
  - `Stage1Predictor(con, settings: ModelSettings, venues: VenueIndex | None = None)`: callable `(game, view) -> float`, with `.explain(game, view) -> Breakdown`, `.breakdowns: dict[str, Breakdown]`, `.unknown_cities: collections.Counter[str]` and `.no_history: int`

- [ ] **Step 1: Write failing tests**

Create `tests/test_model_settings.py`:

```python
import pytest

from predictor.model import settings as ms
from predictor.model.adjustments import Coefficients
from predictor.model.ratings import RatingParams

S = ms.ModelSettings(
    ratings=RatingParams(k=0.08, margin_cap=20.0, season_regression=0.33, hca_window=800),
    coefficients=Coefficients(-1.1, -0.6, -0.3, -0.2, 1.4),
    sigma=13.2,
    fit_games=3369,
    calibrate_games=1230,
)


def test_round_trip(tmp_path):
    path = tmp_path / "s.json"
    ms.save(S, path)
    assert ms.load(path) == S


def test_json_is_stable_and_sorted():
    assert ms.to_json(S) == ms.to_json(ms.from_json(ms.to_json(S)))
    assert ms.to_json(S).endswith("\n")


def test_missing_file_is_a_plain_error(tmp_path):
    with pytest.raises(ms.SettingsError, match="predictor fit-model"):
        ms.load(tmp_path / "nope.json")


@pytest.mark.parametrize("bad", ["not json", "{}", '{"version": 2}'])
def test_corrupt_file_is_a_plain_error(tmp_path, bad):
    path = tmp_path / "s.json"
    path.write_text(bad)
    with pytest.raises(ms.SettingsError, match="predictor fit-model"):
        ms.load(path)


def test_nonpositive_sigma_is_rejected():
    text = ms.to_json(S).replace('"sigma": 13.2', '"sigma": 0.0')
    with pytest.raises(ms.SettingsError):
        ms.from_json(text)


@pytest.mark.parametrize(
    "season, role",
    [("2014-15", "warm-up"), ("2018-19", "warm-up"), ("2019-20", "fit"),
     ("2021-22", "fit"), ("2022-23", "calibrate"), ("2023-24", "test"),
     ("2025-26", "test"), ("2026-27", "unassigned")],
)
def test_season_roles(season, role):
    assert ms.season_role(season) == role


def test_roles_do_not_overlap():
    groups = [set(ms.WARMUP_SEASONS), set(ms.FIT_SEASONS), {ms.CALIBRATE_SEASON}, set(ms.TEST_SEASONS)]
    assert sum(len(g) for g in groups) == len(set().union(*groups)) == 12
```

Create `tests/test_model_stage1.py`:

```python
from collections import Counter
from datetime import UTC, date, datetime, time, timedelta

import pytest

from model_fixtures import add_game, fixture_con
from predictor import db
from predictor.asof import AsOfView
from predictor.backtest import replay
from predictor.backtest.baselines import GameToPredict
from predictor.model.adjustments import Coefficients
from predictor.model.ratings import RatingParams, win_probability
from predictor.model.settings import ModelSettings
from predictor.model.stage1 import Breakdown, Stage1Predictor

S = ModelSettings(
    ratings=RatingParams(k=0.1, margin_cap=20.0, season_regression=0.5, hca_window=100),
    coefficients=Coefficients(back_to_back=-2.0, third_in_four=-1.0,
                              travel_per_1000km=-0.5, tz_per_hour=-0.25, altitude=1.5),
    sigma=13.0,
    fit_games=0,
    calibrate_games=0,
)


def _cutoff(d):  # 30 minutes before a 7pm ET (00:00 UTC next day) tip-off
    return datetime.combine(d + timedelta(days=1), time(0), tzinfo=UTC) - timedelta(minutes=30)


def _game(gid, d, home, away, season="2024-25"):
    return GameToPredict(gid, season, d, home, away)


@pytest.fixture
def con(tmp_path):
    c = fixture_con(tmp_path)
    add_game(c, "0022400001", "2024-25", date(2025, 1, 10), "PHI", "NYK", 110, 100,
             city="Philadelphia")
    add_game(c, "0022400002", "2024-25", date(2025, 1, 12), "NYK", "PHI", 100, 104,
             city="New York")
    add_game(c, "0022400003", "2024-25", date(2025, 1, 15), "PHI", "NYK",
             city="Philadelphia")
    return c


def test_prediction_by_hand(con):
    # Visible at the 01-15 cutoff: both FINALs (stamped 01-11 and 01-13, 12:00 UTC).
    # After game 1: PHI +1, NYK -1, home margins [10].
    # Game 2 (NYK home): predicted = -1 - 1 + 10 = 8; margin -4;
    #   delta = 0.1 * (-4 - 8) = -1.2 -> NYK -2.2, PHI +2.2; margins [10, -4].
    # Game 3: rating = 2.2 - (-2.2) = 4.4; home = mean(10, -4) = 3.0.
    # PHI last played 01-12 (3 days) -> not B2B; NYK same. Travel: PHI
    # New York -> Philadelphia; NYK New York -> Philadelphia: equal, diff 0.
    p = Stage1Predictor(con, S)
    g = _game("0022400003", date(2025, 1, 15), "PHI", "NYK")
    b = p.explain(g, AsOfView(con, _cutoff(date(2025, 1, 15))))
    assert b.rating == pytest.approx(4.4)
    assert b.home == pytest.approx(3.0)
    assert b.rest == pytest.approx(0.0)
    assert b.travel == pytest.approx(0.0)
    assert b.altitude == 0.0
    assert b.spread == pytest.approx(7.4)
    assert b.p_home == pytest.approx(win_probability(7.4, 13.0))


def test_call_returns_the_breakdown_probability_and_records_it(con):
    p = Stage1Predictor(con, S)
    g = _game("0022400003", date(2025, 1, 15), "PHI", "NYK")
    prob = p(g, AsOfView(con, _cutoff(date(2025, 1, 15))))
    assert prob == p.breakdowns["0022400003"].p_home


def test_terms_sum_to_the_spread(con):
    p = Stage1Predictor(con, S)
    b = p.explain(_game("0022400003", date(2025, 1, 15), "PHI", "NYK"),
                  AsOfView(con, _cutoff(date(2025, 1, 15))))
    assert sum(v for _, v in b.terms()) == pytest.approx(b.spread, abs=1e-12)


def test_sentence_by_hand():
    b = Breakdown("g", "DEN", "LAL", rating=4.24, home=2.41, rest=0.84,
                  travel=-0.36, altitude=1.12, spread=8.25, p_home=0.7362)
    assert b.sentence() == (
        "LAL at DEN: rating +4.2, home +2.4, rest +0.8, travel -0.4, "
        "altitude +1.1 -> DEN by 8.1 (DEN 74% to win)"
    )


def test_sentence_for_an_away_favourite():
    b = Breakdown("g", "DEN", "LAL", rating=-6.0, home=2.0, rest=0.0,
                  travel=0.0, altitude=0.0, spread=-4.0, p_home=0.38)
    assert b.sentence().endswith("-> LAL by 4.0 (DEN 38% to win)")


def test_a_result_one_second_after_the_cutoff_moves_no_rating(tmp_path):
    con = fixture_con(tmp_path)
    cutoff = _cutoff(date(2025, 1, 15))
    add_game(con, "0022400001", "2024-25", date(2025, 1, 14), "PHI", "NYK", 130, 100,
             city="Philadelphia", final_observed_at=cutoff + timedelta(seconds=1))
    add_game(con, "0022400003", "2024-25", date(2025, 1, 15), "PHI", "NYK", city="Philadelphia")
    b = Stage1Predictor(con, S).explain(
        _game("0022400003", date(2025, 1, 15), "PHI", "NYK"), AsOfView(con, cutoff))
    assert b.rating == 0.0 and b.home == 0.0


def test_output_ignores_future_fixtures_and_invisible_results(tmp_path):
    """Closes the harness's 2020 play-in item: fixture EXISTENCE and not-yet-
    visible results must not change a single prediction."""
    def build(path, extra):
        con = fixture_con(path)
        add_game(con, "0022400001", "2024-25", date(2025, 1, 10), "PHI", "NYK", 110, 100,
                 city="Philadelphia")
        add_game(con, "0022400002", "2024-25", date(2025, 1, 12), "NYK", "PHI", 100, 104,
                 city="New York")
        add_game(con, "0022400003", "2024-25", date(2025, 1, 15), "PHI", "NYK", 99, 98,
                 city="Philadelphia")
        if extra:
            # a postseason fixture that already "exists" before the season ends
            add_game(con, "0052400001", "2024-25", date(2025, 4, 15), "PHI", "BOS",
                     city="Philadelphia")
            # a result that becomes visible only far in the future
            add_game(con, "0022400009", "2024-25", date(2025, 1, 11), "BOS", "MIA", 150, 90,
                     city="Boston",
                     final_observed_at=datetime(2030, 1, 1, tzinfo=UTC))
        return con

    plain = build(tmp_path / "a", extra=False)
    noisy = build(tmp_path / "b", extra=True)
    got = []
    for con in (plain, noisy):
        preds, _ = replay.replay(con, Stage1Predictor(con, S))
        got.append([(q.game_id, q.p_home) for q in preds if q.game_id != "0022400009"])
    assert got[0] == got[1]


def test_model_never_reads_scheduled_rows(con, monkeypatch):
    seen = []
    real_table = AsOfView.table

    def spy(self, name):
        seen.append(name)
        return real_table(self, name)

    monkeypatch.setattr(AsOfView, "table", spy)
    p = Stage1Predictor(con, S)
    p.explain(_game("0022400003", date(2025, 1, 15), "PHI", "NYK"),
              AsOfView(con, _cutoff(date(2025, 1, 15))))
    assert seen == ["games"]


def test_going_back_in_time_rebuilds_from_scratch(con):
    later = _cutoff(date(2025, 1, 15))
    earlier = _cutoff(date(2025, 1, 11))
    g = _game("0022400003", date(2025, 1, 15), "PHI", "NYK")
    reused = Stage1Predictor(con, S)
    reused.explain(g, AsOfView(con, later))
    after_rewind = reused.explain(g, AsOfView(con, earlier))
    fresh = Stage1Predictor(con, S).explain(g, AsOfView(con, earlier))
    assert after_rewind == fresh


def test_preseason_results_never_touch_ratings(tmp_path):
    con = fixture_con(tmp_path)
    add_game(con, "0012400001", "2024-25", date(2025, 1, 10), "PHI", "NYK", 150, 90,
             city="Philadelphia")
    add_game(con, "0022400003", "2024-25", date(2025, 1, 15), "PHI", "NYK", city="Philadelphia")
    b = Stage1Predictor(con, S).explain(
        _game("0022400003", date(2025, 1, 15), "PHI", "NYK"),
        AsOfView(con, _cutoff(date(2025, 1, 15))))
    assert b.rating == 0.0


def test_neutral_site_gets_no_home_court(tmp_path):
    con = fixture_con(tmp_path)
    add_game(con, "0022400001", "2024-25", date(2025, 1, 10), "PHI", "NYK", 110, 100,
             city="Philadelphia")
    add_game(con, "0022400003", "2024-25", date(2025, 1, 15), "BOS", "MIA",
             city="Paris", neutral=True)
    b = Stage1Predictor(con, S).explain(
        _game("0022400003", date(2025, 1, 15), "BOS", "MIA"),
        AsOfView(con, _cutoff(date(2025, 1, 15))))
    assert b.home == 0.0


def test_unknown_city_and_missing_history_are_counted(tmp_path):
    con = fixture_con(tmp_path)
    add_game(con, "0022400001", "2024-25", date(2025, 1, 14), "PHI", "NYK", 110, 100,
             city="Atlantis")
    add_game(con, "0022400003", "2024-25", date(2025, 1, 15), "PHI", "BOS", city="Philadelphia")
    p = Stage1Predictor(con, S)
    p.explain(_game("0022400003", date(2025, 1, 15), "PHI", "BOS"),
              AsOfView(con, _cutoff(date(2025, 1, 15))))
    assert p.unknown_cities == Counter({"Atlantis": 1})
    assert p.no_history == 1  # BOS has no earlier game


def test_game_missing_from_schedule_is_counted_not_guessed(tmp_path):
    con = fixture_con(tmp_path)
    games = db.POINT_IN_TIME_TABLES["games"]
    con.execute(
        f"INSERT INTO {games} (game_id, season, game_date, home_team, away_team,"
        " status, reconstructed, observed_at) VALUES ('0022400003','2024-25',"
        " DATE '2025-01-15','PHI','NYK','SCHEDULED',TRUE,?)",
        [datetime(2025, 1, 8, 12, tzinfo=UTC)],
    )
    p = Stage1Predictor(con, S)
    b = p.explain(_game("0022400003", date(2025, 1, 15), "PHI", "NYK"),
                  AsOfView(con, _cutoff(date(2025, 1, 15))))
    assert b.home == 0.0 or b.home == pytest.approx(0.0)
    assert p.unknown_cities["(game not in the schedule)"] == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_model_settings.py tests/test_model_stage1.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `settings.py`**

Create `src/predictor/model/settings.py`:

```python
"""Season roles and the committed Stage 1 settings file (spec 3).

Every fitted value lives in one JSON file tracked in git, so every
published number traces to exact settings. `predictor fit-model` writes it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from predictor.model.adjustments import Coefficients
from predictor.model.ratings import RatingParams

WARMUP_SEASONS = ("2014-15", "2015-16", "2016-17", "2017-18", "2018-19")
FIT_SEASONS = ("2019-20", "2020-21", "2021-22")
CALIBRATE_SEASON = "2022-23"
TEST_SEASONS = ("2023-24", "2024-25", "2025-26")

SETTINGS_PATH = Path(__file__).with_name("stage1_settings.json")
_VERSION = 1


class SettingsError(Exception):
    """The settings file is missing or unusable."""


@dataclass(frozen=True)
class ModelSettings:
    ratings: RatingParams
    coefficients: Coefficients
    sigma: float
    fit_games: int
    calibrate_games: int


def season_role(season: str) -> str:
    if season in WARMUP_SEASONS:
        return "warm-up"
    if season in FIT_SEASONS:
        return "fit"
    if season == CALIBRATE_SEASON:
        return "calibrate"
    if season in TEST_SEASONS:
        return "test"
    return "unassigned"


def to_json(s: ModelSettings) -> str:
    doc = {
        "version": _VERSION,
        "ratings": asdict(s.ratings),
        "coefficients": asdict(s.coefficients),
        "sigma": s.sigma,
        "fit_games": s.fit_games,
        "calibrate_games": s.calibrate_games,
    }
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"


def from_json(text: str) -> ModelSettings:
    hint = "Run: predictor fit-model"
    try:
        doc = json.loads(text)
        if doc.get("version") != _VERSION:
            raise SettingsError(
                f"the model settings file is from an unknown version "
                f"({doc.get('version')!r}). {hint}"
            )
        r = doc["ratings"]
        s = ModelSettings(
            ratings=RatingParams(
                k=float(r["k"]),
                margin_cap=float(r["margin_cap"]),
                season_regression=float(r["season_regression"]),
                hca_window=int(r["hca_window"]),
            ),
            coefficients=Coefficients(**{k: float(v) for k, v in doc["coefficients"].items()}),
            sigma=float(doc["sigma"]),
            fit_games=int(doc["fit_games"]),
            calibrate_games=int(doc["calibrate_games"]),
        )
    except SettingsError:
        raise
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise SettingsError(f"the model settings file is unreadable ({exc!r}). {hint}") from None
    if s.sigma <= 0 or s.ratings.hca_window < 1:
        raise SettingsError(f"the model settings file holds impossible values. {hint}")
    return s


def load(path: Path = SETTINGS_PATH) -> ModelSettings:
    try:
        text = path.read_text()
    except FileNotFoundError:
        raise SettingsError(
            f"No fitted model settings found at {path}. Run: predictor fit-model"
        ) from None
    return from_json(text)


def save(s: ModelSettings, path: Path = SETTINGS_PATH) -> None:
    path.write_text(to_json(s))
```

- [ ] **Step 4: Implement `stage1.py`**

Create `src/predictor/model/stage1.py`:

```python
"""The Stage 1 additive points model, as a harness predictor (spec 3).

The harness calls this once per game in chronological order with an
AsOfView cut before tip-off. On each call the model reads -- through that
view, and ONLY rows with status = 'FINAL' -- the results that became
visible since its previous call, updates its ratings, then predicts. It
never reads SCHEDULED rows, so fixture existence (the 2020 play-in item)
cannot reach it. If the harness ever hands it an EARLIER cutoff than the
previous call, it rebuilds from scratch rather than keep knowledge the new
cutoff does not allow.

Venue facts (arena city, neutral site, previous game dates) come from
VenueIndex, read outside the view by design -- see venues.py.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from predictor.backtest.baselines import GameToPredict
from predictor.model import adjustments as adj
from predictor.model.ratings import Ratings, Result, win_probability
from predictor.model.settings import ModelSettings
from predictor.model.venues import COMPETITIVE_PREFIXES, VenueIndex

_COMPETITIVE_SQL = ", ".join(f"'{p}'" for p in COMPETITIVE_PREFIXES)


@dataclass(frozen=True)
class Breakdown:
    game_id: str
    home_team: str
    away_team: str
    rating: float
    home: float
    rest: float
    travel: float
    altitude: float
    spread: float
    p_home: float

    def terms(self) -> tuple[tuple[str, float], ...]:
        return (
            ("rating", self.rating),
            ("home", self.home),
            ("rest", self.rest),
            ("travel", self.travel),
            ("altitude", self.altitude),
        )

    def sentence(self) -> str:
        """One publishable line. The shown total is the sum of the shown
        (rounded) terms, so the sentence always adds up on its face; the
        probability uses the exact spread."""
        shown = [(name, round(value, 1)) for name, value in self.terms()]
        total = round(sum(v for _, v in shown), 1)
        parts = ", ".join(f"{name} {v:+.1f}" for name, v in shown)
        if total > 0:
            outcome = f"{self.home_team} by {total:.1f}"
        elif total < 0:
            outcome = f"{self.away_team} by {-total:.1f}"
        else:
            outcome = "pick'em"
        return (
            f"{self.away_team} at {self.home_team}: {parts} -> {outcome} "
            f"({self.home_team} {self.p_home * 100:.0f}% to win)"
        )


class Stage1Predictor:
    def __init__(self, con, settings: ModelSettings, venues: VenueIndex | None = None) -> None:
        self.settings = settings
        self.venues = venues if venues is not None else VenueIndex.from_db(con)
        self.breakdowns: dict[str, Breakdown] = {}
        self.unknown_cities: Counter[str] = Counter()
        self.no_history = 0
        self._reset()

    def _reset(self) -> None:
        self._ratings = Ratings(self.settings.ratings)
        self._applied: set[str] = set()
        self._as_of = None

    def __call__(self, game: GameToPredict, view) -> float:
        return self.explain(game, view).p_home

    def explain(self, game: GameToPredict, view) -> Breakdown:
        self._catch_up(view)
        self._ratings.enter_season(game.season)

        venue = self.venues.venue(game.game_id)
        city = venue.city if venue else None
        neutral = venue.is_neutral if venue else False
        if venue is None:
            self.unknown_cities["(game not in the schedule)"] += 1

        home_sit = adj.situation(self.venues, game.home_team, game.game_date, city)
        away_sit = adj.situation(self.venues, game.away_team, game.game_date, city)
        for s in (home_sit, away_sit):
            if not s.has_history:
                self.no_history += 1
            if s.unknown_city is not None and venue is not None:
                self.unknown_cities[s.unknown_city] += 1

        x = adj.feature_vector(home_sit, away_sit, adj.is_altitude_game(city, neutral))
        t = adj.terms(self.settings.coefficients, x)
        rating = self._ratings.rating(game.home_team) - self._ratings.rating(game.away_team)
        home = 0.0 if neutral else self._ratings.home_court()
        spread = rating + home + t.rest + t.travel + t.altitude
        breakdown = Breakdown(
            game_id=game.game_id,
            home_team=game.home_team,
            away_team=game.away_team,
            rating=rating,
            home=home,
            rest=t.rest,
            travel=t.travel,
            altitude=t.altitude,
            spread=spread,
            p_home=win_probability(spread, self.settings.sigma),
        )
        self.breakdowns[game.game_id] = breakdown
        return breakdown

    def _catch_up(self, view) -> None:
        if self._as_of is not None and view.as_of < self._as_of:
            self._reset()
        rel = view.table("games").filter(
            "status = 'FINAL' AND home_points IS NOT NULL AND away_points IS NOT NULL "
            f"AND substr(game_id, 1, 3) IN ({_COMPETITIVE_SQL})"
        )
        if self._as_of is not None:
            rel = rel.filter(f"observed_at > TIMESTAMPTZ '{self._as_of.isoformat()}'")
        rows = (
            rel.project(
                "game_id, season, game_date, home_team, away_team, home_points, away_points"
            )
            .order("game_date, game_id")
            .fetchall()
        )
        for gid, season, gd, home, away, hp, ap in rows:
            if gid in self._applied:
                continue
            self._applied.add(gid)
            venue = self.venues.venue(gid)
            self._ratings.apply(
                Result(gid, season, gd, home, away, hp, ap, venue.is_neutral if venue else False)
            )
        self._as_of = view.as_of
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_model_settings.py tests/test_model_stage1.py -v` and then `uv run pytest -q`
Expected: all pass. Hand-check `test_prediction_by_hand` against the arithmetic in its comment. If the test disagrees with the comment, the code is wrong; the test is not.

- [ ] **Step 6: Commit**

```bash
git add src/predictor/model/settings.py src/predictor/model/stage1.py tests/test_model_settings.py tests/test_model_stage1.py
git commit -m "feat: add the stage 1 predictor that learns only through the as-of view

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Fitting, `fit-model`, and the committed settings file

**Files:**
- Create: `src/predictor/model/fit.py`, `src/predictor/model/stage1_settings.json` (generated in Step 6)
- Modify: `src/predictor/cli.py` (new `fit-model` command), `pyproject.toml` (+ `uv.lock`)
- Test: `tests/test_model_fit.py`

**Interfaces:**
- Consumes: Tasks 2–5.
- Produces:
  - `fit.GRID_K = (0.04, 0.06, 0.08, 0.10, 0.12, 0.15)`
  - `fit.GRID_CAP = (15.0, 20.0, 25.0, 30.0)`
  - `fit.GRID_REGRESSION = (0.2, 0.33, 0.5, 0.66)`
  - `fit.GRID_WINDOW = (400, 800, 1230)`
  - `fit.SIGMA_GRID`: 8.00 to 20.00 in steps of 0.05, built from integers
  - `fit.fit(con) -> ModelSettings`
  - `fit.describe(s: ModelSettings) -> str` (plain English)
  - CLI `predictor fit-model`

**How the fit works:**
1. **Load history.** Load every competitive FINAL result for warm-up, fit and calibrate seasons, using the latest FINAL row per game. The fit reads the games table directly: this is training on completed past seasons, not a prediction path. Test seasons are excluded in the SQL, and an assertion checks it.
2. **Features.** Compute each game's feature vector with `adj.situation` and `VenueIndex`, exactly as the predictor does.
3. **Simulate each grid point** (`k, cap, regression, window`). Walk the history date by date. For each date, first record every game's pre-game `rating gap` and `home court`, then apply that date's results. This matches the predictor, which sees a date's results only from the next day.
4. **Score the grid point.** On fit-season regular-season (`002`) games, set `y = margin − rating gap − home court` and solve `y ≈ X·coef` by least squares (`numpy.linalg.lstsq`, no intercept). The grid point's score is the mean squared residual. Keep the lowest score, breaking ties by grid order (strictly lower wins).
5. **Set σ.** With the chosen settings and coefficients, compute spreads for calibrate-season regular-season games. Choose the σ on `SIGMA_GRID` with the lowest log loss (strictly lower wins).
6. **Round.** Round every float to 6 decimals.

- [ ] **Step 1: Add numpy as an explicit dependency**

It is already installed as a pandas dependency; this makes it explicit.

```bash
cd /Users/meya/projects/predictor
uv add "numpy>=2.0"
```

Expected: `pyproject.toml` gains `"numpy>=2.0"` and `uv.lock` updates.

- [ ] **Step 2: Write failing tests**

Create `tests/test_model_fit.py`:

```python
from datetime import date, timedelta

import pytest
from typer.testing import CliRunner

from model_fixtures import add_game, fixture_con
from predictor import cli, config, db
from predictor.config import Settings
from predictor.model import fit as fit_mod
from predictor.model import settings as ms
from real_archive import open_real_archive_or_skip

TEAMS = ["PHI", "NYK", "BOS", "MIA"]


def _season(con, season, start, n_days, home_edge, gid_prefix="002"):
    """A tiny round-robin: every day two games, home team wins by `home_edge`
    plus a deterministic team-strength term."""
    strength = {"PHI": 3, "NYK": -3, "BOS": 1, "MIA": -1}
    n = 0
    for day in range(n_days):
        d = start + timedelta(days=2 * day)
        pairs = [(TEAMS[day % 4], TEAMS[(day + 1) % 4]), (TEAMS[(day + 2) % 4], TEAMS[(day + 3) % 4])]
        for home, away in pairs:
            n += 1
            margin = home_edge + strength[home] - strength[away]
            add_game(con, f"{gid_prefix}{season[2:4]}{n:05d}", season, d, home, away,
                     100 + max(margin, 0), 100 + max(-margin, 0), city="Boston")


def _history(con):
    seasons = ms.WARMUP_SEASONS[-1:] + ms.FIT_SEASONS + (ms.CALIBRATE_SEASON,) + ms.TEST_SEASONS
    for i, season in enumerate(seasons):
        _season(con, season, date(2015 + i, 11, 1), 20, home_edge=3)


def test_fit_is_deterministic(tmp_path):
    con = fixture_con(tmp_path)
    _history(con)
    assert fit_mod.fit(con) == fit_mod.fit(con)


def test_fit_ignores_test_seasons_entirely(tmp_path):
    con = fixture_con(tmp_path)
    _history(con)
    before = fit_mod.fit(con)
    games = db.POINT_IN_TIME_TABLES["games"]
    placeholders = ", ".join("?" for _ in ms.TEST_SEASONS)
    con.execute(
        f"UPDATE {games} SET home_points = 50, away_points = 150 "
        f"WHERE status = 'FINAL' AND season IN ({placeholders})",
        list(ms.TEST_SEASONS),
    )
    assert fit_mod.fit(con) == before


def test_fit_counts_its_games_and_picks_values_from_the_grids(tmp_path):
    con = fixture_con(tmp_path)
    _history(con)
    s = fit_mod.fit(con)
    assert s.fit_games == 3 * 40          # three fit seasons x 40 games
    assert s.calibrate_games == 40
    assert s.ratings.k in fit_mod.GRID_K
    assert s.ratings.margin_cap in fit_mod.GRID_CAP
    assert s.ratings.season_regression in fit_mod.GRID_REGRESSION
    assert s.ratings.hca_window in fit_mod.GRID_WINDOW
    assert s.sigma in fit_mod.SIGMA_GRID


def test_sigma_grid_is_exact():
    assert fit_mod.SIGMA_GRID[0] == 8.0
    assert fit_mod.SIGMA_GRID[-1] == 20.0
    assert len(fit_mod.SIGMA_GRID) == 241
    assert 13.05 in fit_mod.SIGMA_GRID


def test_describe_is_plain_english():
    s = ms.ModelSettings(
        ratings=fit_mod.RatingParams(0.08, 20.0, 0.33, 800),
        coefficients=fit_mod.Coefficients(-1.1, -0.6, -0.3, -0.2, 1.4),
        sigma=13.2, fit_games=3369, calibrate_games=1230,
    )
    text = fit_mod.describe(s)
    assert "8% of" in text
    assert "20 points" in text
    assert "33%" in text
    assert "800" in text
    assert "back-to-back" in text and "-1.1" in text
    assert "13.2" in text


def test_fit_model_command_writes_the_settings_file(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(config, "settings", s)
    monkeypatch.setattr(db, "settings", s)
    con = db.connect()
    db.migrate(con)
    _history(con)
    con.close()
    out_path = tmp_path / "stage1_settings.json"
    monkeypatch.setattr(ms, "SETTINGS_PATH", out_path)
    result = CliRunner().invoke(cli.app, ["fit-model"])
    assert result.exit_code == 0, result.output
    assert ms.load(out_path) is not None
    assert "saved" in result.output.lower()


def test_fit_model_with_no_history_is_a_plain_error(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(config, "settings", s)
    monkeypatch.setattr(db, "settings", s)
    con = db.connect()
    db.migrate(con)
    con.close()  # DuckDB refuses a read-only open while a writable one is live
    monkeypatch.setattr(ms, "SETTINGS_PATH", tmp_path / "s.json")
    result = CliRunner().invoke(cli.app, ["fit-model"])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "ingest-season" in result.output


def test_committed_settings_reproduce_from_the_real_archive():
    """A published number must trace to settings anyone can re-derive."""
    con = open_real_archive_or_skip()
    try:
        assert fit_mod.fit(con) == ms.load()
    finally:
        con.close()
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_model_fit.py -q`
Expected: FAIL with `ImportError`, because `predictor.model.fit` is missing.

- [ ] **Step 4: Implement `fit.py`**

Create `src/predictor/model/fit.py`:

```python
"""Choose every Stage 1 setting from past seasons only (spec 3).

Fit seasons choose the rating settings and adjustment sizes; the calibrate
season chooses only sigma; test seasons are never read. The fit reads the
games table directly -- it trains on completed past seasons and is not a
prediction path -- and asserts that no test season was loaded.

Simulation mirrors Stage1Predictor exactly: a date's results are applied
only after every game on that date has been given its pre-game numbers,
because in the harness a result becomes visible the day after it is played.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from itertools import groupby, product

import numpy as np

from predictor import db
from predictor.model import adjustments as adj
from predictor.model.adjustments import Coefficients
from predictor.model.ratings import RatingParams, Ratings, Result, win_probability
from predictor.model.settings import (
    CALIBRATE_SEASON,
    FIT_SEASONS,
    TEST_SEASONS,
    WARMUP_SEASONS,
    ModelSettings,
)
from predictor.model.venues import COMPETITIVE_PREFIXES, VenueIndex

GRID_K = (0.04, 0.06, 0.08, 0.10, 0.12, 0.15)
GRID_CAP = (15.0, 20.0, 25.0, 30.0)
GRID_REGRESSION = (0.2, 0.33, 0.5, 0.66)
GRID_WINDOW = (400, 800, 1230)
SIGMA_GRID = tuple(i / 100 for i in range(800, 2001, 5))


class FitError(Exception):
    """Not enough history to fit."""


@dataclass(frozen=True)
class _Game:
    result: Result
    x: tuple[float, float, float, float, float]


def _load(con, venues: VenueIndex) -> list[_Game]:
    seasons = WARMUP_SEASONS + FIT_SEASONS + (CALIBRATE_SEASON,)
    table = db.POINT_IN_TIME_TABLES["games"]
    prefixes = ", ".join(f"'{p}'" for p in COMPETITIVE_PREFIXES)
    placeholders = ", ".join("?" for _ in seasons)
    rows = con.execute(
        f"SELECT game_id, season, game_date, home_team, away_team, home_points, away_points "
        f"FROM {table} WHERE status = 'FINAL' AND home_points IS NOT NULL "
        f"AND away_points IS NOT NULL AND substr(game_id, 1, 3) IN ({prefixes}) "
        f"AND season IN ({placeholders}) "
        "QUALIFY row_number() OVER (PARTITION BY game_id ORDER BY observed_at DESC) = 1 "
        "ORDER BY game_date, game_id",
        list(seasons),
    ).fetchall()
    games: list[_Game] = []
    for gid, season, gd, home, away, hp, ap in rows:
        assert season not in TEST_SEASONS, "fit must never read a test season"
        venue = venues.venue(gid)
        city = venue.city if venue else None
        neutral = venue.is_neutral if venue else False
        x = adj.feature_vector(
            adj.situation(venues, home, gd, city),
            adj.situation(venues, away, gd, city),
            adj.is_altitude_game(city, neutral),
        )
        games.append(_Game(Result(gid, season, gd, home, away, hp, ap, neutral), x))
    return games


def _simulate(params: RatingParams, games: list[_Game]) -> list[tuple[float, float]]:
    """Pre-game (rating gap, home court) for every game, in input order."""
    ratings = Ratings(params)
    out: list[tuple[float, float]] = []
    for _, day in groupby(games, key=lambda g: g.result.game_date):
        day = list(day)
        for g in day:
            ratings.enter_season(g.result.season)
            r = g.result
            gap = ratings.rating(r.home_team) - ratings.rating(r.away_team)
            out.append((gap, 0.0 if r.neutral else ratings.home_court()))
        for g in day:
            ratings.apply(g.result)
    return out


def _residuals(games, pre, seasons):
    rows = [
        (g, gap, hc)
        for g, (gap, hc) in zip(games, pre)
        if g.result.season in seasons and g.result.game_id.startswith("002")
    ]
    X = np.array([g.x for g, _, _ in rows], dtype=float).reshape(len(rows), 5)
    y = np.array(
        [g.result.home_points - g.result.away_points - gap - hc for g, gap, hc in rows],
        dtype=float,
    )
    return rows, X, y


def fit(con) -> ModelSettings:
    venues = VenueIndex.from_db(con)
    games = _load(con, venues)
    fit_count = sum(
        1 for g in games if g.result.season in FIT_SEASONS and g.result.game_id.startswith("002")
    )
    cal_count = sum(
        1 for g in games
        if g.result.season == CALIBRATE_SEASON and g.result.game_id.startswith("002")
    )
    if fit_count == 0 or cal_count == 0:
        raise FitError(
            "not enough history to fit the model: the fit seasons "
            f"({', '.join(FIT_SEASONS)}) and the calibrate season ({CALIBRATE_SEASON}) "
            "must have results. Run 'predictor ingest-season <season>' for each."
        )

    best = None
    for k, cap, reg, window in product(GRID_K, GRID_CAP, GRID_REGRESSION, GRID_WINDOW):
        params = RatingParams(k, cap, reg, window)
        pre = _simulate(params, games)
        _, X, y = _residuals(games, pre, FIT_SEASONS)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        mse = float(np.mean((y - X @ coef) ** 2))
        if best is None or mse < best[0]:
            best = (mse, params, coef)
    _, params, coef = best
    coefficients = Coefficients(*(round(float(c), 6) for c in coef))

    pre = _simulate(params, games)
    rows, X, _ = _residuals(games, pre, (CALIBRATE_SEASON,))
    spreads = [
        gap + hc + sum(adj.astuple_terms(coefficients, g.x))
        for (g, gap, hc) in rows
    ]
    outcomes = [g.result.home_points > g.result.away_points for g, _, _ in rows]
    best_sigma = None
    for sigma in SIGMA_GRID:
        loss = 0.0
        for spread, won in zip(spreads, outcomes):
            p = min(max(win_probability(spread, sigma), 1e-12), 1 - 1e-12)
            loss -= math.log(p if won else 1 - p)
        if best_sigma is None or loss < best_sigma[0]:
            best_sigma = (loss, sigma)

    return ModelSettings(
        ratings=params,
        coefficients=coefficients,
        sigma=best_sigma[1],
        fit_games=fit_count,
        calibrate_games=cal_count,
    )


def describe(s: ModelSettings) -> str:
    r, c = s.ratings, s.coefficients
    return "\n".join([
        f"Ratings move {r.k * 100:.0f}% of each game's surprise; blowouts count as at "
        f"most {r.margin_cap:.0f} points.",
        f"Each new season, ratings fall back {r.season_regression * 100:.0f}% toward average.",
        f"Home court is the average home margin over the last {r.hca_window} games.",
        "Adjustments, in points for the home team:",
        f"  back-to-back (home minus away)       {c.back_to_back:+.1f}",
        f"  third game in four nights            {c.third_in_four:+.1f}",
        f"  per 1,000 km travelled               {c.travel_per_1000km:+.1f}",
        f"  per time zone crossed                {c.tz_per_hour:+.1f}",
        f"  playing at altitude (Denver, Utah)   {c.altitude:+.1f}",
        f"Typical game-to-game spread (sigma): {s.sigma:.2f} points.",
        f"Chosen on {s.fit_games:,} fit-season games; sigma set on "
        f"{s.calibrate_games:,} calibrate-season games.",
    ])
```

In `src/predictor/model/adjustments.py`, add this helper below `terms` (used by the fit):

```python
def astuple_terms(c: Coefficients, x: tuple[float, ...]) -> tuple[float, float, float]:
    t = terms(c, x)
    return (t.rest, t.travel, t.altitude)
```

Also re-export `RatingParams` and `Coefficients` from `fit.py`; they are already imported at the top, and the tests use `fit_mod.RatingParams` and `fit_mod.Coefficients`.

**Rounding:** `ModelSettings` must round-trip through JSON exactly.
- Grid values are exact decimal literals.
- `sigma` comes from `SIGMA_GRID`, built as `i / 100`.
- Coefficients are rounded to 6 decimals before `ModelSettings` is built.

`test_committed_settings_reproduce_from_the_real_archive` relies on this.

- [ ] **Step 5: Add the CLI command**

In `src/predictor/cli.py`, add after the `backtest` command:

```python
@app.command("fit-model")
def fit_model_cmd() -> None:
    """Choose the model's settings from past seasons and save them."""
    import duckdb

    from predictor import db
    from predictor.model import fit as fit_mod
    from predictor.model import settings as model_settings

    try:
        con = db.connect(read_only=True)
    except duckdb.Error as exc:
        typer.echo(
            f"Could not open the database ({exc}). If another 'predictor' "
            "command is running, wait a moment and try again."
        )
        raise typer.Exit(code=1) from None
    try:
        chosen = fit_mod.fit(con)
    except fit_mod.FitError as exc:
        typer.echo(f"Cannot fit the model: {exc}.")
        raise typer.Exit(code=1) from None
    except duckdb.Error as exc:
        typer.echo(
            f"The database is missing tables the model needs ({exc}). Run "
            "'predictor ingest-season <season>' and 'predictor ingest-schedule "
            "--season <season>' first."
        )
        raise typer.Exit(code=1) from None
    model_settings.save(chosen, model_settings.SETTINGS_PATH)
    typer.echo(fit_mod.describe(chosen))
    typer.echo(f"Saved to {model_settings.SETTINGS_PATH}.")
```

The CLI must read `model_settings.SETTINGS_PATH` at call time, not at import, so the test's monkeypatch takes effect.

- [ ] **Step 6: Run tests; generate and commit the real settings**

Run: `uv run pytest tests/test_model_fit.py -q -k "not committed"`
Expected: all pass.

Then fit on the real archive (read-only):

```bash
uv run predictor fit-model
git diff --stat
```

Expected: a plain-English summary, then `Saved to .../stage1_settings.json.` Sanity check:
- home court estimate positive
- back-to-back coefficient negative
- σ between 11 and 15

If any is outside those ranges, stop and report it rather than committing.

Run: `uv run pytest -q -rs`
Expected: all pass, including `test_committed_settings_reproduce_from_the_real_archive`, which must not SKIP.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/predictor/model/fit.py src/predictor/model/adjustments.py src/predictor/model/stage1_settings.json src/predictor/cli.py tests/test_model_fit.py
git commit -m "feat: fit stage 1 settings on past seasons only and commit them

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: `backtest --model stage1`, test-season headline, season table

**Files:**
- Modify: `src/predictor/backtest/report.py` (`BacktestResult.scope`, header, `format_season_table`), `src/predictor/cli.py` (`backtest` command)
- Test: `tests/test_backtest_report.py`, `tests/test_model_backtest.py` (new)

**Interfaces:**
- Consumes: `Stage1Predictor`, `settings.load`, `SettingsError`, `season_role`, `TEST_SEASONS` (Task 5).
- Produces:
  - `report.summarize(..., scope: str | None = None)` stores `BacktestResult.scope: str | None = None` (the last field, with a default). When set, the header's `Season` line shows `scope` instead of the computed season label.
  - `report.format_season_table(preds, role_of: Callable[[str], str]) -> str`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_backtest_report.py`:

```python
def test_scope_replaces_the_season_label_in_the_header():
    preds = [_pred(season="2023-24"), _pred(season="2024-25")]
    result = report.summarize(
        preds, _stats(len(preds)), model="stage1", buffer_minutes=30,
        market_available=False, market_row_count=0,
        scope="test seasons 2023-24, 2024-25, 2025-26 only",
    )
    text = report.format_report(result)
    assert "Season              : test seasons 2023-24, 2024-25, 2025-26 only" in text


def test_season_table_by_hand():
    preds = [
        _pred(season="2023-24", p_home=0.9, home_won=True),
        _pred(season="2023-24", p_home=0.2, home_won=True),
        _pred(season="2014-15", p_home=0.6, home_won=False),
    ]
    text = report.format_season_table(preds, lambda s: "test" if s == "2023-24" else "warm-up")
    lines = text.splitlines()
    assert lines[0].startswith("  By season")
    # 2014-15: 1 game, model picked home and lost -> 0.0%; home won 0% ;
    # Brier (0.6-0)^2 = 0.36
    assert "2014-15  warm-up        1 games   model   0.0%   home   0.0%   Brier 0.3600" in text
    # 2023-24: 2 games, model right once (0.9 home, won) and wrong once
    # (0.2 away, home won) -> 50.0%; home won both -> 100.0%;
    # Brier ((0.1)^2 + (0.8)^2) / 2 = 0.3250
    assert "2023-24  test           2 games   model  50.0%   home 100.0%   Brier 0.3250" in text
```

Before writing these, read the top of `tests/test_backtest_report.py`. Reuse or add module-level helpers `_pred(season=..., p_home=..., home_won=...)` and `_stats(n)` that build a `Prediction` and a `ReplayStats` in the same way the existing tests do (see `TIP` and the `Prediction(...)` / `ReplayStats(...)` constructions around lines 14–25). If helpers with these names already exist with different signatures, name the new ones `_season_pred` and `_season_stats` instead, and use those in the two new tests.

Create `tests/test_model_backtest.py`:

```python
"""`predictor backtest --model stage1` end to end, on a fixture archive.

Also closes harness open item 3: the BEATS / TOO CLOSE TO CALL verdict
branch had never run from the CLI, because both shipped baselines put every
game on the home side. This drives it with a predictor that disagrees.
"""

from datetime import date, timedelta

from typer.testing import CliRunner

from model_fixtures import add_game
from predictor import cli, config, db
from predictor.config import Settings
from predictor.model import settings as ms
from predictor.model import stage1
from predictor.model.adjustments import Coefficients
from predictor.model.ratings import RatingParams

runner = CliRunner()

SETTINGS = ms.ModelSettings(
    ratings=RatingParams(0.1, 20.0, 0.5, 100),
    coefficients=Coefficients(-1.0, -0.5, -0.3, -0.2, 1.0),
    sigma=13.0, fit_games=1, calibrate_games=1,
)


def _archive(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(config, "settings", s)
    monkeypatch.setattr(db, "settings", s)
    con = db.connect()
    db.migrate(con)
    # warm-up game, then a test-season run where the AWAY team always wins
    add_game(con, "0021800001", "2018-19", date(2019, 3, 1), "PHI", "NYK", 120, 100,
             city="Philadelphia")
    for i in range(40):
        d = date(2023, 11, 1) + timedelta(days=2 * i)
        add_game(con, f"00223{i:05d}", "2023-24", d, "NYK", "PHI", 90, 110, city="New York")
    con.close()
    return s


def test_stage1_headline_is_test_seasons_only(tmp_path, monkeypatch):
    _archive(tmp_path, monkeypatch)
    monkeypatch.setattr(ms, "load", lambda path=None: SETTINGS)
    out = runner.invoke(cli.app, ["backtest", "--model", "stage1"])
    assert out.exit_code == 0, out.output
    assert "test seasons 2023-24, 2024-25, 2025-26 only" in out.output
    assert "By season" in out.output
    assert "2018-19  warm-up" in out.output
    assert "Example explanations" in out.output
    assert " at " in out.output and "% to win)" in out.output


def test_verdict_branch_runs_from_the_cli(tmp_path, monkeypatch):
    _archive(tmp_path, monkeypatch)
    monkeypatch.setattr(ms, "load", lambda path=None: SETTINGS)

    class AwayPicker(stage1.Stage1Predictor):
        def __call__(self, game, view):
            super().__call__(game, view)
            return 0.2  # always picks the away team, which always wins here

    monkeypatch.setattr(stage1, "Stage1Predictor", AwayPicker)
    out = runner.invoke(cli.app, ["backtest", "--model", "stage1"])
    assert out.exit_code == 0, out.output
    assert "BEATS always-pick-home" in out.output


def test_missing_settings_is_a_plain_error(tmp_path, monkeypatch):
    _archive(tmp_path, monkeypatch)

    def missing(path=None):
        raise ms.SettingsError("No fitted model settings found at x. Run: predictor fit-model")

    monkeypatch.setattr(ms, "load", missing)
    out = runner.invoke(cli.app, ["backtest", "--model", "stage1"])
    assert out.exit_code == 1
    assert "predictor fit-model" in out.output
    assert "Traceback" not in out.output


def test_no_test_season_games_is_a_plain_error(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(config, "settings", s)
    monkeypatch.setattr(db, "settings", s)
    con = db.connect()
    db.migrate(con)
    add_game(con, "0021800001", "2018-19", date(2019, 3, 1), "PHI", "NYK", 120, 100,
             city="Philadelphia")
    con.close()
    monkeypatch.setattr(ms, "load", lambda path=None: SETTINGS)
    out = runner.invoke(cli.app, ["backtest", "--model", "stage1"])
    assert out.exit_code == 1
    assert "No test-season games" in out.output
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_backtest_report.py tests/test_model_backtest.py -q`
Expected: FAIL. You should see `summarize() got an unexpected keyword argument 'scope'`, a missing `format_season_table`, and the CLI rejecting model `stage1`.

- [ ] **Step 3: Implement the report changes**

In `src/predictor/backtest/report.py`:

1. Add the last field to `BacktestResult`: `scope: str | None = None`.
2. Add a keyword parameter `scope: str | None = None` to `summarize`, after `market_row_count`, and pass `scope=scope` into `BacktestResult(...)`.
3. In `_provenance_header`, after computing `season_label`, add:
   ```python
   if result.scope is not None:
       season_label = result.scope
   ```
4. Add at the end of the module:

```python
def format_season_table(preds, role_of) -> str:
    """Accuracy, home rate and Brier for every season, labeled by role, so
    one lucky season cannot carry a pooled headline unseen."""
    lines = ["  By season (model accuracy / always-pick-home / Brier):"]
    by_season: dict[str, list] = {}
    for p in preds:
        by_season.setdefault(p.season, []).append(p)
    for season in sorted(by_season):
        group = by_season[season]
        lines.append(
            f"    {season}  {role_of(season):<9}  {len(group):>5,} games   "
            f"model {metrics.accuracy(group) * 100:5.1f}%   "
            f"home {metrics.home_rate(group) * 100:5.1f}%   "
            f"Brier {metrics.brier_score(group):.4f}"
        )
    return "\n".join(lines)
```

The hand-computed expectations in the test assume this exact format string. If the test's spacing and the format string disagree, fix the format string to match the test.

- [ ] **Step 4: Implement the CLI wiring**

In `src/predictor/cli.py`'s `backtest_cmd`:

1. Update the `--model` help text to `"Which predictor to score: always-home, coin-flip, or stage1."`
2. Replace the `known = {...}` block and its validity check with:

```python
    names = ("always-home", "coin-flip", "stage1")
    if model not in names:
        typer.echo(f"Unknown model '{model}'. Available: {', '.join(names)}.")
        raise typer.Exit(code=1)
```

3. After the read-only connection has been opened successfully, build the predictor:

```python
    stage1_predictor = None
    if model == "stage1":
        from predictor.model import settings as model_settings
        from predictor.model import stage1

        try:
            loaded = model_settings.load()
        except model_settings.SettingsError as exc:
            typer.echo(f"Cannot run the stage1 model: {exc}")
            raise typer.Exit(code=1) from None
        stage1_predictor = stage1.Stage1Predictor(con, loaded)
        chosen = stage1_predictor
    elif model == "always-home":
        chosen = baselines.always_home
    else:
        chosen = baselines.fixed_probability(0.5)
```

   and pass `chosen` to `replay.replay(...)` instead of `known[model]`.

   `model_settings.load()` must be called through the module attribute, so the test's monkeypatch works. Likewise `stage1.Stage1Predictor` must be looked up at call time.

4. Just before `result = report.summarize(...)`, add:

```python
    headline = preds
    scope = None
    if stage1_predictor is not None:
        headline = [p for p in preds if model_settings.season_role(p.season) == "test"]
        if not headline:
            typer.echo(
                "No test-season games were scored "
                f"({', '.join(model_settings.TEST_SEASONS)}), so there is no honest "
                "headline to report. Ingest those seasons and try again."
            )
            raise typer.Exit(code=1)
        scope = (
            f"test seasons {', '.join(model_settings.TEST_SEASONS)} only -- "
            "the model's settings were never tuned on them"
        )
```

   Pass `headline` (not `preds`) as the first argument to `report.summarize(...)`, and pass `scope=scope`.

5. After `typer.echo(report.format_report(result))`, add:

```python
    if stage1_predictor is not None:
        typer.echo("")
        typer.echo(report.format_season_table(preds, model_settings.season_role))
        if stage1_predictor.unknown_cities or stage1_predictor.no_history:
            typer.echo("")
            typer.echo("  Games where travel could not be measured (counted as 0):")
            for city, n in sorted(stage1_predictor.unknown_cities.items()):
                typer.echo(f"    {n:,} team-game(s): {city}")
            typer.echo(
                f"    {stage1_predictor.no_history:,} team-game(s) with no earlier game "
                "(treated as fully rested, no travel)"
            )
        examples = [
            stage1_predictor.breakdowns[p.game_id]
            for p in headline[-3:]
            if p.game_id in stage1_predictor.breakdowns
        ]
        if examples:
            typer.echo("")
            typer.echo("  Example explanations (most recent test-season games):")
            for b in examples:
                typer.echo(f"    {b.sentence()}")
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_backtest_report.py tests/test_model_backtest.py tests/test_backtest_cli.py -q` and then `uv run pytest -q`
Expected: all pass.

- [ ] **Step 6: Real-archive checks**

Append to `tests/test_model_backtest.py`:

```python
import pytest

from predictor.backtest import replay
from real_archive import open_real_archive_or_skip


@pytest.mark.parametrize("season", ["2023-24", "2024-25", "2025-26"])
def test_every_test_season_game_is_predicted_and_every_explanation_adds_up(season):
    con = open_real_archive_or_skip()
    try:
        predictor = stage1.Stage1Predictor(con, ms.load())
        preds, stats = replay.replay(con, predictor, season=season)
        assert stats.considered == 1230
        assert stats.predicted == 1230, stats
        for p in preds:
            b = predictor.breakdowns[p.game_id]
            assert sum(v for _, v in b.terms()) == pytest.approx(b.spread, abs=1e-9)
            assert p.p_home == b.p_home
            shown = [round(v, 1) for _, v in b.terms()]
            assert f"by {abs(round(sum(shown), 1)):.1f}" in b.sentence() or "pick'em" in b.sentence()
    finally:
        con.close()
```

Run: `uv run pytest tests/test_model_backtest.py -v -rs`
Expected: all pass, with no SKIPPED.

- [ ] **Step 7: Run the real backtest and record it**

```bash
uv run predictor backtest --model stage1 | tee /tmp/stage1-backtest.txt
uv run predictor backtest | tail -25
```

Paste both outputs into your report. Expected for stage1:
- a header naming the test seasons
- a verdict line starting with `BEATS`, `LOSES TO` or `TOO CLOSE TO CALL`
- the season table covering 2014-15 to 2025-26
- three example sentences

Do not tune anything to change the numbers. Report them as they are.

- [ ] **Step 8: Commit**

```bash
git add src/predictor/backtest/report.py src/predictor/cli.py tests/test_backtest_report.py tests/test_model_backtest.py
git commit -m "feat: score stage 1 on held-out test seasons with a per-season table

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Record the result (controller)

- [ ] **Step 1:** In the spec's "First delivery design" subsection, append a dated paragraph **"Result — <date>"**. Give the stage1 test-season accuracy against the home baseline, the Brier score, and the per-test-season accuracy. State plainly whether the bar was met: it beats home in *every* test season, and the calibration buckets are within a few points. Commit it with the Co-Authored-By trailer.
- [ ] **Step 2:** Update memory. Create `project_prediction_core.md` with the result, the settings summary, and what comes next (odds sub-project, then live results capture). Mark the gate items in `project_backtest_harness.md` as closed:
  - item 2 (play-in leak): closed by the FINAL-only model reads and the invariance test
  - item 3 (verdict branch): closed by `test_verdict_branch_runs_from_the_cli` and the real run
  - item 1 (leak proof) still stands until live results are captured
