"""Today's slate, the append-only public prediction log, and grading.

Spec: the live-operation plan, task 2. Every public prediction is written
once, to an append-only JSONL file, before the game tips off -- never
rewritten -- and is graded only once a captured result (a FINAL row with
``reconstructed = FALSE``) exists for it. The model is read only through
``Stage1Predictor``/``AsOfView``, exactly as in the backtest harness, so a
live run cannot leak a result into its own prediction.

Local machine time is IST and launchd fires on local time, but the slate is
a US-Eastern calendar date (``EASTERN``) -- the league's own day boundary.
Every ``now`` this module takes is a timezone-aware UTC datetime; nothing
here ever reads the wall clock itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from predictor import db
from predictor.asof import AsOfView
from predictor.backtest.baselines import GameToPredict
from predictor.model.settings import ModelSettings
from predictor.model.stage1 import Stage1Predictor
from predictor.model.venues import COMPETITIVE_PREFIXES

EASTERN = ZoneInfo("America/New_York")
START_BUFFER = timedelta(minutes=30)
STALE_AFTER = timedelta(hours=36)

_NOT_PREDICTED_REASON = (
    "game had already started (or was within 30 minutes of tip-off) when "
    "the prediction run happened"
)

_COMPETITIVE_SQL = ", ".join(f"'{p}'" for p in COMPETITIVE_PREFIXES)


@dataclass(frozen=True)
class SlateGame:
    game_id: str
    season: str
    game_date: date
    home_team: str
    away_team: str
    tip_off_utc: datetime


@dataclass(frozen=True)
class RunResult:
    predicted: int
    not_predicted: int
    skipped_duplicates: int
    stale: bool
    lines_written: list[dict]


def slate_for(con, now: datetime) -> list[SlateGame]:
    """Today's (US-Eastern) competitive games, latest schedule vintage only.

    "Today" is ``now``'s Eastern calendar date -- the same date the
    schedule itself stores in ``game_date`` (spec 1.1), so no further
    timezone conversion of the schedule data is needed or correct. A game
    with no reported tip-off time (``tip_off_utc IS NULL``, a TBD listing)
    is excluded: there is nothing to compare ``now`` against.
    """
    db.require_utc(now, "now")
    table = db.POINT_IN_TIME_TABLES["schedule"]
    today = now.astimezone(EASTERN).date()
    rows = con.execute(
        f"""
        WITH latest AS (
            SELECT game_id, season, game_date, home_team, away_team, tip_off_utc,
                   row_number() OVER (
                       PARTITION BY game_id ORDER BY observed_at DESC
                   ) AS rn
            FROM {table}
        )
        SELECT game_id, season, game_date, home_team, away_team, tip_off_utc
        FROM latest
        WHERE rn = 1
          AND game_date = ?
          AND substr(game_id, 1, 3) IN ({_COMPETITIVE_SQL})
          AND tip_off_utc IS NOT NULL
        ORDER BY tip_off_utc, game_id
        """,
        [today],
    ).fetchall()
    return [
        SlateGame(game_id, season, game_date, home_team, away_team, tip_off_utc)
        for game_id, season, game_date, home_team, away_team, tip_off_utc in rows
    ]


def last_capture(con) -> datetime | None:
    """The latest ``observed_at`` of a genuinely captured FINAL result.

    ``reconstructed = FALSE`` is exactly what distinguishes a live capture
    (Task 1) from the post-hoc backfill that stamps every historical row.
    """
    table = db.POINT_IN_TIME_TABLES["games"]
    row = con.execute(
        f"SELECT max(observed_at) FROM {table} WHERE status = 'FINAL' AND reconstructed = FALSE"
    ).fetchone()
    return row[0] if row is not None else None


def in_season(con, now: datetime) -> bool:
    """True if any competitive schedule game tips off within 3 days of ``now``."""
    db.require_utc(now, "now")
    table = db.POINT_IN_TIME_TABLES["schedule"]
    row = con.execute(
        f"""
        SELECT 1 FROM {table}
        WHERE substr(game_id, 1, 3) IN ({_COMPETITIVE_SQL})
          AND tip_off_utc BETWEEN ? AND ?
        LIMIT 1
        """,
        [now - timedelta(days=3), now + timedelta(days=3)],
    ).fetchone()
    return row is not None


def log_path(repo_dir: Path, season: str) -> Path:
    return repo_dir / "predictions" / f"{season}.jsonl"


def grades_path(repo_dir: Path, season: str) -> Path:
    return repo_dir / "predictions" / f"{season}-grades.jsonl"


def read_log(path: Path) -> list[dict]:
    try:
        text = path.read_text()
    except FileNotFoundError:
        return []
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _settings_dict(settings: ModelSettings) -> dict:
    return {
        "k": settings.ratings.k,
        "margin_cap": settings.ratings.margin_cap,
        "season_regression": settings.ratings.season_regression,
        "hca_window": settings.ratings.hca_window,
        "sigma": settings.sigma,
        "half_life": settings.half_life,
    }


def _append_line(path: Path, line: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, sort_keys=True) + "\n")


def predict_today(con, settings: ModelSettings, repo_dir: Path, now: datetime) -> RunResult:
    """Predict every not-yet-logged game on today's slate, once each.

    One ``Stage1Predictor`` is built for the whole run and reused for every
    game (its catch-up of newly-visible results is incremental), all cut at
    the same ``AsOfView(con, now)`` -- so a result observed after ``now``
    cannot reach any prediction made in this run.
    """
    db.require_utc(now, "now")
    games = slate_for(con, now)
    last_cap = last_capture(con)
    stale_flag = in_season(con, now) and (
        last_cap is None or now - last_cap > STALE_AFTER
    )
    settings_dict = _settings_dict(settings)
    predictor = Stage1Predictor(con, settings)

    predicted = not_predicted = skipped_duplicates = 0
    lines_written: list[dict] = []
    existing_by_season: dict[str, set[tuple[str, str]]] = {}

    for g in games:
        if g.season not in existing_by_season:
            existing_by_season[g.season] = {
                (line["game_id"], line["game_date"]) for line in read_log(log_path(repo_dir, g.season))
            }
        existing_keys = existing_by_season[g.season]
        game_date_str = g.game_date.isoformat()
        key = (g.game_id, game_date_str)
        if key in existing_keys:
            skipped_duplicates += 1
            continue

        base = {
            "predicted_at": now.isoformat(),
            "game_id": g.game_id,
            "season": g.season,
            "game_date": game_date_str,
            "tip_off_utc": g.tip_off_utc.isoformat(),
            "home_team": g.home_team,
            "away_team": g.away_team,
            "settings": settings_dict,
            "stale_results": stale_flag,
            "last_result_capture": last_cap.isoformat() if last_cap is not None else None,
        }

        if g.tip_off_utc - START_BUFFER <= now:
            line = {
                **base,
                "status": "not_predicted",
                "reason": _NOT_PREDICTED_REASON,
                "spread": None,
                "p_home": None,
                "sentence": None,
                "terms": None,
            }
            not_predicted += 1
        else:
            breakdown = predictor.explain(
                GameToPredict(g.game_id, g.season, g.game_date, g.home_team, g.away_team),
                AsOfView(con, now),
            )
            line = {
                **base,
                "status": "predicted",
                "reason": None,
                "spread": round(breakdown.spread, 6),
                "p_home": round(breakdown.p_home, 6),
                "sentence": breakdown.sentence(),
                "terms": {name: round(value, 6) for name, value in breakdown.terms()},
            }
            predicted += 1

        _append_line(log_path(repo_dir, g.season), line)
        lines_written.append(line)
        existing_keys.add(key)

    return RunResult(predicted, not_predicted, skipped_duplicates, stale_flag, lines_written)


def grade(con, repo_dir: Path, season: str, now: datetime) -> int:
    """Grade every predicted game that now has a captured FINAL result.

    For each ``game_id``, the latest ``predicted`` line whose own
    ``predicted_at`` precedes its own ``tip_off_utc`` is the one graded --
    a ``not_predicted`` line, or a prediction logged at or after tip-off
    (which should never happen, but is never trusted either), is skipped.
    The result read is a scoring read, like replay's: the latest FINAL row
    for the game, regardless of when it was observed relative to ``now`` --
    grading only ever runs after the fact.
    """
    log = read_log(log_path(repo_dir, season))
    latest_predicted: dict[str, dict] = {}
    for line in log:
        if line["status"] != "predicted":
            continue
        tip = datetime.fromisoformat(line["tip_off_utc"])
        predicted_at = datetime.fromisoformat(line["predicted_at"])
        if predicted_at >= tip:
            continue
        current = latest_predicted.get(line["game_id"])
        if current is None or predicted_at > datetime.fromisoformat(current["predicted_at"]):
            latest_predicted[line["game_id"]] = line

    gpath = grades_path(repo_dir, season)
    already_graded = {g["game_id"] for g in read_log(gpath)}
    games_table = db.POINT_IN_TIME_TABLES["games"]

    appended = 0
    for game_id in sorted(latest_predicted):
        if game_id in already_graded:
            continue
        line = latest_predicted[game_id]
        row = con.execute(
            f"SELECT home_points, away_points FROM {games_table} "
            "WHERE game_id = ? AND status = 'FINAL' "
            "AND home_points IS NOT NULL AND away_points IS NOT NULL "
            "ORDER BY observed_at DESC LIMIT 1",
            [game_id],
        ).fetchone()
        if row is None:
            continue
        home_points, away_points = row
        home_won = home_points > away_points
        p_home = line["p_home"]
        grade_line = {
            "game_id": game_id,
            "predicted_at": line["predicted_at"],
            "p_home": p_home,
            "home_won": home_won,
            "correct": (p_home >= 0.5) == home_won,
            "graded_at": now.isoformat(),
        }
        _append_line(gpath, grade_line)
        appended += 1

    return appended
