"""Today's slate, the append-only public prediction log, and grading.

Spec: the live-operation plan, task 2. Every public prediction is written
once, to an append-only JSONL file, before the game tips off -- never
rewritten -- and is graded once a FINAL result exists for it in the games
table: ANY FINAL row counts, live-captured or backfilled -- grading does
not care how a result arrived, only that one did. (``last_result_capture``
and the ``stale``/``results_missing`` check below are the parts of this
module that care specifically about *live* captures.) The model is read
only through ``Stage1Predictor``/``AsOfView``, exactly as in the backtest
harness, so a live run cannot leak a result into its own prediction.

Local machine time is IST and launchd fires on local time, but the slate is
a US-Eastern calendar date (``EASTERN``) -- the league's own day boundary --
taken ``SLATE_DAY_OFFSET`` hours back (see ``slate_date``).
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
from predictor.model import market, publish
from predictor.model.settings import ModelSettings
from predictor.model.stage1 import Stage1Predictor
from predictor.model.venues import COMPETITIVE_PREFIXES

EASTERN = ZoneInfo("America/New_York")
START_BUFFER = timedelta(minutes=30)

# `slate_date`: a run's slate is the ET calendar date of (now - this). A
# catch-up run in the small hours after ET midnight (a laptop that slept
# through the evening run) therefore still handles the ET day that just
# ended, before last night's results are captured, rather than predicting
# the new day early. 7h -- not more -- because the scheduled run is 18:00
# IST, which is 08:30 EDT but only 07:30 EST (IST has no daylight saving):
# an 8h offset would make the scheduled run handle YESTERDAY all winter.
SLATE_DAY_OFFSET = timedelta(hours=7)

# Back-fill: competitive games dated within this many days BEFORE the slate
# date that have no line in the log at all get a not_predicted line, so a
# missed run leaves an honest gap in the public record rather than a silent
# one.
BACKFILL_DAYS = 3

# `in_season`: how far from `now`, in either direction, a game still counts
# as "nearby" (used to decide whether a quiet archive means off-season or a
# broken capture job).
IN_SEASON_WINDOW = timedelta(days=3)

# `results_missing`: a game's result only counts as overdue once it has had
# time to be reported (RESULTS_MISSING_GRACE since tip-off) and only while
# that gap is still recent (within RESULTS_MISSING_WINDOW) -- an old gap
# (a long-finished game nobody ever captured) must not pin `stale` True
# forever, and a game that only just tipped off must not trip it either.
RESULTS_MISSING_GRACE = timedelta(hours=12)
RESULTS_MISSING_WINDOW = timedelta(days=3)

_NOT_PREDICTED_REASON = (
    "game had already started (or was within 30 minutes of tip-off) when "
    "the prediction run happened"
)
_MISSED_REASON = (
    "no prediction was made before tip-off (the daily prediction run did not "
    "happen in time)"
)
# A TBD line is provisional: it never blocks a later prediction of the same
# game once its tip-off is announced (see predict_today's dedupe).
_TBD_REASON = "tip-off time not announced when the prediction run happened"

# The live odds job (com.predictor.odds) runs at 17:30 IST, half an hour
# before predict-today; this label says which line the market fields hold.
MARKET_LABEL = "market line at 17:30 IST"

_NO_MARKET = {
    "market_p_home": None,
    "market_spread": None,
    "market_books": None,
    "market_observed_at": None,
    "market_label": None,
}

_COMPETITIVE_SQL = ", ".join(f"'{p}'" for p in COMPETITIVE_PREFIXES)


class LogError(Exception):
    """A prediction/grades log file is corrupt or otherwise unusable."""


@dataclass(frozen=True)
class SlateGame:
    game_id: str
    season: str
    game_date: date
    home_team: str
    away_team: str
    # None only for a TBD listing (see `_competitive_games`); `slate_for`
    # never returns one.
    tip_off_utc: datetime | None


@dataclass(frozen=True)
class RunResult:
    predicted: int
    # Every not_predicted line written this run (too late, TBD tip-off, or
    # back-filled) -- `backfilled` is the subset for missed earlier days.
    not_predicted: int
    skipped_duplicates: int
    stale: bool
    lines_written: list[dict]
    backfilled: int = 0


def slate_date(now: datetime) -> date:
    """The US-Eastern calendar date a run at ``now`` predicts: the ET date
    of ``now - SLATE_DAY_OFFSET`` (see that constant for why)."""
    db.require_utc(now, "now")
    return (now - SLATE_DAY_OFFSET).astimezone(EASTERN).date()


def _competitive_games(con, first: date, last: date) -> list[SlateGame]:
    """Competitive games, latest schedule vintage only, whose ET
    ``game_date`` lies in ``[first, last]`` -- TBD tip-offs included."""
    table = db.POINT_IN_TIME_TABLES["schedule"]
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
          AND game_date BETWEEN ? AND ?
          AND substr(game_id, 1, 3) IN ({_COMPETITIVE_SQL})
        ORDER BY game_date, tip_off_utc NULLS LAST, game_id
        """,
        [first, last],
    ).fetchall()
    return [SlateGame(*row) for row in rows]


def slate_for(con, now: datetime) -> list[SlateGame]:
    """The slate date's competitive games, latest schedule vintage only.

    The slate date is ``slate_date(now)`` -- an ET calendar date, the same
    date the schedule itself stores in ``game_date`` (spec 1.1), so no
    further timezone conversion of the schedule data is needed or correct.
    A game with no reported tip-off time (``tip_off_utc IS NULL``, a TBD
    listing) is excluded: there is nothing to compare ``now`` against
    (``predict_today`` logs those separately).
    """
    today = slate_date(now)
    table = db.POINT_IN_TIME_TABLES["schedule"]
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
    """True if any competitive game's LATEST schedule vintage tips off
    within ``IN_SEASON_WINDOW`` of ``now``, in either direction."""
    db.require_utc(now, "now")
    table = db.POINT_IN_TIME_TABLES["schedule"]
    row = con.execute(
        f"""
        WITH latest AS (
            SELECT game_id, tip_off_utc,
                   row_number() OVER (
                       PARTITION BY game_id ORDER BY observed_at DESC
                   ) AS rn
            FROM {table}
        )
        SELECT 1 FROM latest
        WHERE rn = 1
          AND substr(game_id, 1, 3) IN ({_COMPETITIVE_SQL})
          AND tip_off_utc BETWEEN ? AND ?
        LIMIT 1
        """,
        [now - IN_SEASON_WINDOW, now + IN_SEASON_WINDOW],
    ).fetchone()
    return row is not None


def results_missing(con, now: datetime) -> list[str]:
    """Competitive games whose LATEST schedule vintage tipped off between
    ``RESULTS_MISSING_WINDOW`` and ``RESULTS_MISSING_GRACE`` ago and have no
    FINAL row in the games table at all -- ANY FINAL row counts, so this is
    blind to whether a result was live-captured or backfilled; it only
    asks whether one exists yet. Exposed as its own function because Task
    4's status command reuses it, not just this module's ``stale`` flag.
    """
    db.require_utc(now, "now")
    schedule_table = db.POINT_IN_TIME_TABLES["schedule"]
    games_table = db.POINT_IN_TIME_TABLES["games"]
    rows = con.execute(
        f"""
        WITH latest AS (
            SELECT game_id, tip_off_utc,
                   row_number() OVER (
                       PARTITION BY game_id ORDER BY observed_at DESC
                   ) AS rn
            FROM {schedule_table}
        )
        SELECT l.game_id
        FROM latest l
        WHERE l.rn = 1
          AND substr(l.game_id, 1, 3) IN ({_COMPETITIVE_SQL})
          AND l.tip_off_utc IS NOT NULL
          AND l.tip_off_utc BETWEEN ? AND ?
          AND NOT EXISTS (
              SELECT 1 FROM {games_table} g
              WHERE g.game_id = l.game_id AND g.status = 'FINAL'
          )
        ORDER BY l.game_id
        """,
        [now - RESULTS_MISSING_WINDOW, now - RESULTS_MISSING_GRACE],
    ).fetchall()
    return [r[0] for r in rows]


def _current_tip(con, game_id: str) -> datetime | None:
    """The LATEST schedule vintage's tip-off for one game, or None if the
    game is not (or no longer) in the schedule."""
    table = db.POINT_IN_TIME_TABLES["schedule"]
    row = con.execute(
        f"""
        SELECT tip_off_utc FROM {table}
        WHERE game_id = ?
        QUALIFY row_number() OVER (PARTITION BY game_id ORDER BY observed_at DESC) = 1
        """,
        [game_id],
    ).fetchone()
    return row[0] if row is not None else None


def log_path(repo_dir: Path, season: str) -> Path:
    return repo_dir / "predictions" / f"{season}.jsonl"


def grades_path(repo_dir: Path, season: str) -> Path:
    return repo_dir / "predictions" / f"{season}-grades.jsonl"


def read_log(path: Path) -> list[dict]:
    """Every line of an append-only JSONL log, or ``[]`` if it is missing.

    A non-empty file that does not end with a newline means the last
    ``write()`` was cut off mid-line (e.g. a crash during an append) --
    reading it as if the partial line were not there would silently lose
    evidence that something is wrong, and letting a caller then append a
    fresh line after that partial one would merge the two into one corrupt
    line forever. So this raises instead, and raises BEFORE any caller that
    reads-then-appends (``predict_today``, ``grade``) has a chance to write
    anything -- the file is never touched here, only read.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    if text and not text.endswith("\n"):
        raise LogError(
            f"the prediction log at {path} ends with an incomplete line; "
            "it was not modified -- inspect and repair it by hand"
        )
    lines = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            lines.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise LogError(
                f"the prediction log at {path} has a corrupt line ({exc}); "
                "it was not modified -- inspect and repair it by hand"
            ) from None
    return lines


def _settings_dict(settings: ModelSettings) -> dict:
    return {
        "k": settings.ratings.k,
        "margin_cap": settings.ratings.margin_cap,
        "season_regression": settings.ratings.season_regression,
        "hca_window": settings.ratings.hca_window,
        "sigma": settings.sigma,
        "half_life": settings.half_life,
    }


def _r6(value: float) -> float:
    """Round to 6 decimals for the public log, normalising -0.0 to 0.0
    (``-0.0 + 0.0 == 0.0``) -- a "-0.0" in a published number is noise a
    reader would rightly ask about."""
    return round(value, 6) + 0.0


def _market_fields(con, game_id: str, now: datetime, sigma: float) -> dict:
    """The live market's view of one game as seen at ``now`` -- displayed
    beside the model's numbers, never fed into them (the model is computed
    before this is called and never sees it). All fields null when no live
    line is visible."""
    view = market.market_p_home(market.live_lines(con, game_id, now), sigma)
    if view is None:
        return dict(_NO_MARKET)
    return {
        "market_p_home": _r6(view.p_home),
        "market_spread": _r6(view.spread) if view.spread is not None else None,
        "market_books": view.books,
        "market_observed_at": view.observed_at.isoformat(),
        "market_label": MARKET_LABEL,
    }


def _append_line(path: Path, line: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, sort_keys=True) + "\n")


def predict_today(con, settings: ModelSettings, repo_dir: Path, now: datetime) -> RunResult:
    """Predict every not-yet-logged game on the slate, once each, and log
    an honest not_predicted line for every game that could not be.

    The slate is ``slate_date(now)``'s games. Each gets exactly one line:
    ``predicted`` if it tips off more than ``START_BUFFER`` after ``now``,
    else ``not_predicted`` (too late). A slate game whose tip-off is still
    TBD gets a provisional ``not_predicted`` line (``_TBD_REASON``) that
    does NOT count as a prediction for dedupe -- once the tip is announced
    a later run predicts it normally. Games dated within ``BACKFILL_DAYS``
    before the slate date that already tipped off (or are TBD) and have no
    line at all are back-filled as ``not_predicted`` (``_MISSED_REASON``):
    a missed run shows up in the public record as a gap, never silently.

    One ``Stage1Predictor`` is built for the whole run and reused for every
    game (its catch-up of newly-visible results is incremental), all cut at
    the same ``AsOfView(con, now)`` -- so a result observed after ``now``
    cannot reach any prediction made in this run.
    """
    db.require_utc(now, "now")
    today = slate_date(now)
    backfill_games = [
        g
        for g in _competitive_games(con, today - timedelta(days=BACKFILL_DAYS), today - timedelta(days=1))
        if g.tip_off_utc is None or g.tip_off_utc <= now
    ]
    today_games = _competitive_games(con, today, today)
    last_cap = last_capture(con)
    stale_flag = bool(results_missing(con, now))
    settings_dict = _settings_dict(settings)
    # Which code made these lines -- see publish.code_version.
    code_sha, code_dirty = publish.code_version(repo_dir)
    predictor = Stage1Predictor(con, settings)

    predicted = not_predicted = skipped_duplicates = backfilled = 0
    lines_written: list[dict] = []
    # Per season: every (game_id, game_date) with ANY line, and the subset
    # whose line is not a provisional TBD one.
    any_line: dict[str, set[tuple[str, str]]] = {}
    blocking: dict[str, set[tuple[str, str]]] = {}

    def keys_for(season: str) -> tuple[set, set]:
        if season not in any_line:
            log = read_log(log_path(repo_dir, season))
            any_line[season] = {(line["game_id"], line["game_date"]) for line in log}
            blocking[season] = {
                (line["game_id"], line["game_date"])
                for line in log
                if line.get("reason") != _TBD_REASON
            }
        return any_line[season], blocking[season]

    def base_line(g: SlateGame) -> dict:
        return {
            "predicted_at": now.isoformat(),
            "game_id": g.game_id,
            "season": g.season,
            "game_date": g.game_date.isoformat(),
            "tip_off_utc": g.tip_off_utc.isoformat() if g.tip_off_utc is not None else None,
            "home_team": g.home_team,
            "away_team": g.away_team,
            "settings": settings_dict,
            "stale_results": stale_flag,
            "last_result_capture": last_cap.isoformat() if last_cap is not None else None,
            "code_version": code_sha,
            "code_dirty": code_dirty,
        }

    def not_predicted_line(g: SlateGame, reason: str) -> dict:
        return {
            **base_line(g),
            "status": "not_predicted",
            "reason": reason,
            "spread": None,
            "p_home": None,
            "sentence": None,
            "terms": None,
            **_NO_MARKET,
        }

    def write(g: SlateGame, line: dict) -> None:
        _append_line(log_path(repo_dir, g.season), line)
        lines_written.append(line)
        seen, block = keys_for(g.season)
        key = (g.game_id, line["game_date"])
        seen.add(key)
        if line.get("reason") != _TBD_REASON:
            block.add(key)

    for g in backfill_games:
        seen, _ = keys_for(g.season)
        if (g.game_id, g.game_date.isoformat()) in seen:
            continue
        write(g, not_predicted_line(g, _MISSED_REASON))
        not_predicted += 1
        backfilled += 1

    for g in today_games:
        seen, block = keys_for(g.season)
        key = (g.game_id, g.game_date.isoformat())
        if g.tip_off_utc is None:
            if key in seen:
                skipped_duplicates += 1
                continue
            write(g, not_predicted_line(g, _TBD_REASON))
            not_predicted += 1
            continue
        if key in block:
            skipped_duplicates += 1
            continue

        if g.tip_off_utc - START_BUFFER <= now:
            line = not_predicted_line(g, _NOT_PREDICTED_REASON)
            not_predicted += 1
        else:
            breakdown = predictor.explain(
                GameToPredict(g.game_id, g.season, g.game_date, g.home_team, g.away_team),
                AsOfView(con, now),
            )
            line = {
                **base_line(g),
                "status": "predicted",
                "reason": None,
                "spread": _r6(breakdown.spread),
                "p_home": _r6(breakdown.p_home),
                "sentence": breakdown.sentence(),
                "terms": {name: _r6(value) for name, value in breakdown.terms()},
                **_market_fields(con, g.game_id, now, settings.sigma),
            }
            predicted += 1
        write(g, line)

    return RunResult(
        predicted, not_predicted, skipped_duplicates, stale_flag, lines_written, backfilled
    )


def grade(con, repo_dir: Path, season: str, now: datetime) -> int:
    """Grade every predicted game that now has a FINAL result.

    For each ``game_id``, the latest ``predicted`` line whose own
    ``predicted_at`` precedes its own cutoff is the one graded -- a
    ``not_predicted`` line is never a candidate. The cutoff is the EARLIER
    of the line's own logged ``tip_off_utc`` and the schedule's CURRENT
    latest-vintage tip-off for that game: if the game was rescheduled
    earlier after the line was written, the old logged tip-off would wrongly
    still call a too-late prediction "before tip-off", so the live schedule
    is consulted too and whichever tip-off is earlier wins. A missing
    current schedule row (long gone from the feed) falls back to the
    line's own tip-off.

    The FINAL row read is a scoring read, like replay's: the latest FINAL
    row for the game, regardless of when it was observed relative to
    ``now`` -- grading only ever runs after the fact. It must also match
    the line's own ``game_date``: a FINAL row for the same ``game_id`` under
    a DIFFERENT date (the game itself was rescheduled, not just corrected)
    is not this prediction's result and is not used to grade it.

    ``correct`` scores the line's pick: home when ``p_home >= 0.5``, away
    otherwise -- so an exact coin-flip ``p_home == 0.5`` (e.g. two teams
    with no history and no adjustments) counts as picking the HOME team.
    That tie-break is deliberate and fixed; it is stated here because it
    is part of how the public record is scored. Each grade line also
    carries the final score and teams, so it can be checked on its own.
    """
    log = read_log(log_path(repo_dir, season))
    latest_predicted: dict[str, dict] = {}
    for line in log:
        if line["status"] != "predicted":
            continue
        own_tip = datetime.fromisoformat(line["tip_off_utc"])
        current_tip = _current_tip(con, line["game_id"])
        cutoff = own_tip if current_tip is None else min(own_tip, current_tip)
        predicted_at = datetime.fromisoformat(line["predicted_at"])
        if predicted_at >= cutoff:
            continue
        current = latest_predicted.get(line["game_id"])
        if current is None or predicted_at > datetime.fromisoformat(current["predicted_at"]):
            latest_predicted[line["game_id"]] = line

    gpath = grades_path(repo_dir, season)
    already_graded = {g["game_id"] for g in read_log(gpath)}
    games_table = db.POINT_IN_TIME_TABLES["games"]

    code_sha, code_dirty = publish.code_version(repo_dir)
    appended = 0
    for game_id in sorted(latest_predicted):
        if game_id in already_graded:
            continue
        line = latest_predicted[game_id]
        row = con.execute(
            f"SELECT home_points, away_points FROM {games_table} "
            "WHERE game_id = ? AND game_date = ? AND status = 'FINAL' "
            "AND home_points IS NOT NULL AND away_points IS NOT NULL "
            "ORDER BY observed_at DESC LIMIT 1",
            [game_id, date.fromisoformat(line["game_date"])],
        ).fetchone()
        if row is None:
            continue
        home_points, away_points = row
        home_won = home_points > away_points
        p_home = line["p_home"]
        market_p = line.get("market_p_home")
        grade_line = {
            "game_id": game_id,
            "home_team": line["home_team"],
            "away_team": line["away_team"],
            "home_points": home_points,
            "away_points": away_points,
            "predicted_at": line["predicted_at"],
            "p_home": p_home,
            "home_won": home_won,
            "correct": (p_home >= 0.5) == home_won,
            # Same tie-break as `correct`; null for a line with no market
            # (or one written before market fields existed).
            "market_correct": (
                None if market_p is None else (market_p >= 0.5) == home_won
            ),
            "graded_at": now.isoformat(),
            "code_version": code_sha,
            "code_dirty": code_dirty,
        }
        _append_line(gpath, grade_line)
        appended += 1

    return appended
