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
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import duckdb

from predictor import db
from predictor.asof import AsOfView
from predictor.backtest.baselines import GameToPredict
from predictor.model import live_model as _lm
from predictor.model import market, publish
from predictor.model.availability import InjuryHistory
from predictor.model.calls import (
    CALL_CUTOFF,
    EARLY_ONLY,
    TBD_REASON,
    GameChain,
    StatusChange,
    as_utc,
    call_type_of,
    chains,
    final_due,
    revision_due,
    revision_reason,
    status_changes,
    still_callable,
)
from predictor.model.live_model import (
    MODEL_VERSION,
    CallOutcome,
    LiveModel,
    ModelKind,
    kind_for_game,
    player_settings_dict,
)
from predictor.model.live_model import team_settings_dict as _settings_dict
from predictor.model.player_settings import PlayerModelSettings
from predictor.model.settings import ModelSettings
from predictor.model.venues import competitive_sql

EASTERN = ZoneInfo("America/New_York")
IST = ZoneInfo("Asia/Kolkata")
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
# game once its tip-off is announced (see predict_today's dedupe). The text
# lives in ``calls`` so ``calls`` never imports ``live``.
_TBD_REASON = TBD_REASON
# A game with a sealed (predicted) call but no final call once
# ``tip - CALL_CUTOFF`` has passed gets one not_predicted final line.
_MISSED_FINAL_REASON = (
    "no final call was made in the 45 minutes before tip-off (no run reached it in time)"
)

# The live odds job (com.predictor.odds) runs at 17:30 IST, half an hour
# before predict-today -- but when it missed (a sleeping Mac), predict-today
# fetches the line itself, later. So the label is derived from when the
# line was actually observed (``market_label``), never assumed.

# `market_odds_due`: a live odds snapshot observed within this long before
# `now` counts as today's market -- no fresh fetch is needed.
MARKET_FRESH = timedelta(hours=6)

_NO_MARKET = {
    "market_p_home": None,
    "market_spread": None,
    "market_books": None,
    "market_observed_at": None,
    "market_label": None,
}


# --- the 15-minute cloud runs (cloud-operation plan, Task 7) ---------------
#
# The cloud runs every 15 minutes, but there is still one call per game per
# slate: the EARLY call, made by the first run at or after 18:00 IST on the
# slate date (12:30 UTC all year -- IST has no daylight saving). GitHub cron
# starts late and skips slots, so "due" means "at or after", never "at".
EARLY_CALL_IST = time(18, 0)

# Odds budget: two windows per slate day, at most ODDS_MAX_ATTEMPTS fetches
# each, none while the last recorded quota is below ODDS_MIN_REMAINING.
# "early" opens with the early call (so the call sees a line) and stays open
# ODDS_EARLY_WINDOW; "pre_tip" opens ODDS_PRE_TIP_LEAD before the slate's
# first tip and closes at it.
ODDS_EARLY_WINDOW = timedelta(hours=2)
ODDS_PRE_TIP_LEAD = timedelta(minutes=75)
ODDS_MAX_ATTEMPTS = 2
ODDS_MIN_REMAINING = 30


@dataclass(frozen=True)
class OddsWindow:
    name: Literal["early", "pre_tip"]
    opens: datetime
    closes: datetime


def early_call_at(slate: date) -> datetime:
    """18:00 IST on the slate date, as a UTC instant (12:30 UTC)."""
    return datetime.combine(slate, EARLY_CALL_IST, tzinfo=IST).astimezone(UTC)


def early_call_due(now: datetime) -> bool:
    """Whether a run at ``now`` makes the slate's call: ``now`` is at or
    after 18:00 IST on ``slate_date(now)``. Repeat runs are harmless --
    ``predict_today`` writes each game once."""
    return now >= early_call_at(slate_date(now))


def _as_instant(value) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def odds_window_due(
    con, now: datetime, attempts: list[dict], remaining: int | None
) -> OddsWindow | None:
    """The odds window a run at ``now`` should fetch for, or None.

    ``first_tip`` is the earliest tip of ``slate_for(con, now)``; an empty
    slate is never due. A window is due when ``opens <= now < closes``, a
    slate game still tips more than ``START_BUFFER`` after ``now``, fewer
    than ``ODDS_MAX_ATTEMPTS`` attempts are recorded at or after ``opens``,
    no attempt in the window is marked ``downloaded`` (its body was
    archived, so its credits were spent) or ``ok``, and ``remaining`` (the
    last ``x-requests-remaining``) is unknown or at least
    ``ODDS_MIN_REMAINING``. ``attempts`` are the records of
    ``state/odds/attempts-*.jsonl`` (``attempted_at`` as text or datetime).

    Judged from the attempt lines alone, never from stored odds rows: odds
    are not republished in state (licensing), so a fresh runner's DB holds
    none. ``con`` is only read for the slate.
    """
    db.require_utc(now, "now")
    if remaining is not None and remaining < ODDS_MIN_REMAINING:
        return None
    slate = slate_for(con, now)
    if not slate:
        return None
    if not any(g.tip_off_utc - START_BUFFER > now for g in slate):
        return None
    first_tip = min(g.tip_off_utc for g in slate)
    early_opens = early_call_at(slate_date(now))
    pre_tip_opens = first_tip - ODDS_PRE_TIP_LEAD
    windows = (
        OddsWindow("early", early_opens, min(early_opens + ODDS_EARLY_WINDOW, pre_tip_opens)),
        OddsWindow("pre_tip", pre_tip_opens, first_tip),
    )
    window = next((w for w in windows if w.opens <= now < w.closes), None)
    if window is None:
        return None
    in_window = [a for a in attempts if _as_instant(a["attempted_at"]) >= window.opens]
    if len(in_window) >= ODDS_MAX_ATTEMPTS:
        return None
    # A response that reached the archive spent the credits, even if it then
    # failed to load: the window is done. (`ok` covers attempt lines written
    # before `downloaded` existed.)
    if any(a.get("downloaded") or a.get("ok") for a in in_window):
        return None
    return window


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
    # Predicted revision and final lines; missed finals are the
    # not_predicted final lines (not counted in ``not_predicted``).
    revisions: int = 0
    finals: int = 0
    missed_finals: int = 0
    # The ``model_version`` of every line written this run.
    model_versions: frozenset[str] = frozenset()


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
          AND {competitive_sql()}
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
          AND {competitive_sql()}
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
          AND {competitive_sql()}
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
          AND {competitive_sql("l.game_id")}
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


_SEASON_LOG = re.compile(r"^\d{4}-\d{2}\.jsonl$")


def id_links_path(repo_dir: Path, season: str) -> Path:
    """Provisional-id links for one season's log. A subdirectory, so the
    site (which reads only top-level ``predictions/*.jsonl``) ignores it."""
    return repo_dir / "predictions" / "id-links" / f"{season}.jsonl"


def read_id_links(repo_dir: Path, season: str) -> dict[str, str]:
    """provisional ``espn:`` id -> NBA game id for one season; a later line
    wins."""
    from predictor.cloud import state

    path = id_links_path(repo_dir, season)
    if not path.exists():
        return {}
    return {r["provisional_id"]: r["game_id"] for r in state.read_records(path)}


def append_id_links(repo_dir: Path, links: list[dict]) -> list[Path]:
    """Record ``{provisional_id, game_id, linked_at}`` for each link whose
    provisional id appears in a season's prediction log and whose mapping
    that season has not recorded yet. Logged lines are never touched: the
    link file is what tells dedupe and grading that a line under the
    provisional id is about the NBA game. Returns the files appended to."""
    from predictor.cloud import state

    pred_dir = repo_dir / "predictions"
    if not pred_dir.is_dir():
        return []
    touched: list[Path] = []
    for log_file in sorted(pred_dir.iterdir()):
        if not _SEASON_LOG.match(log_file.name):
            continue
        season = log_file.name.removesuffix(".jsonl")
        logged = {line["game_id"] for line in read_log(log_file)}
        known = read_id_links(repo_dir, season)
        new: list[dict] = []
        for link in links:
            prov, gid = link["provisional_id"], link["game_id"]
            if prov not in logged or known.get(prov) == gid:
                continue
            known[prov] = gid
            new.append({"provisional_id": prov, "game_id": gid, "linked_at": link["linked_at"]})
        if new:
            touched.append(state.append_records(id_links_path(repo_dir, season), new))
    return touched


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


def _r6(value: float) -> float:
    """Round to 6 decimals for the public log, normalising -0.0 to 0.0
    (``-0.0 + 0.0 == 0.0``) -- a "-0.0" in a published number is noise a
    reader would rightly ask about."""
    return round(value, 6) + 0.0


def _market_fields(con, game_id: str, now: datetime, sigma: float) -> dict:
    """The live market's view of one game as seen at ``now`` -- displayed
    beside the model's numbers, never fed into them (the model is computed
    before this is called and never sees it). All fields null when no live
    line is visible.

    Display-only, so it must never stop the prediction run: an unusable
    stored price (ValueError) or a database error reading odds leaves the
    market fields null and the prediction is still written."""
    try:
        view = market.market_p_home(market.live_lines(con, game_id, now), sigma)
    except (ValueError, duckdb.Error):
        return dict(_NO_MARKET)
    if view is None:
        return dict(_NO_MARKET)
    return {
        "market_p_home": _r6(view.p_home),
        "market_spread": _r6(view.spread) if view.spread is not None else None,
        "market_books": view.books,
        "market_observed_at": view.observed_at.isoformat(),
        "market_label": market_label(view.observed_at),
    }


def market_label(observed_at: datetime) -> str:
    """'market line at HH:MM IST' -- when the line was actually observed,
    as the owner's clock in India showed it. ``observed_at`` is the
    timezone-aware instant read back from the odds table."""
    return f"market line at {observed_at.astimezone(IST):%H:%M} IST"


def market_odds_due(con, now: datetime) -> bool:
    """Whether predict-today should fetch the live market line itself: the
    slate has at least one game it will actually predict (tip-off more
    than ``START_BUFFER`` away) and no live (``theoddsapi``) snapshot was
    observed within ``MARKET_FRESH`` before ``now``. An empty slate --
    off-season, or a day without games -- is never due, so no Odds API
    request (500 a month) is spent on it."""
    db.require_utc(now, "now")
    if not any(g.tip_off_utc - START_BUFFER > now for g in slate_for(con, now)):
        return False
    table = db.POINT_IN_TIME_TABLES["odds_snapshots"]
    (fresh,) = con.execute(
        f"""
        SELECT count(*) FROM {table}
        WHERE source = '{market.LIVE_SOURCE}'
          AND observed_at > ? AND observed_at <= ?
        """,
        [now - MARKET_FRESH, now],
    ).fetchone()
    return fresh == 0


def _append_line(path: Path, line: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, sort_keys=True) + "\n")


def latest_report_at(con, now: datetime) -> datetime | None:
    """The header time of the latest injury report seen at ``now``
    (``injury_reports``: an unchanged report adds no ``injury_status`` rows,
    but is still a newer report)."""
    db.require_utc(now, "now")
    (value,) = AsOfView(con, now).table("injury_reports").aggregate(
        "max(observed_at)").fetchone()
    return value.astimezone(UTC) if value is not None else None


def calls_due(con, now: datetime) -> bool:
    """Whether a run at ``now`` may have a call to write: the early call is
    due, or a game dated in ``[slate - BACKFILL_DAYS, slate]`` tips after
    ``now - 1 hour`` (a revision, a final or a missed final may be due)."""
    db.require_utc(now, "now")
    if early_call_due(now):
        return True
    today = slate_date(now)
    return any(
        g.tip_off_utc is not None and g.tip_off_utc > now - timedelta(hours=1)
        for g in _competitive_games(con, today - timedelta(days=BACKFILL_DAYS), today)
    )


def predict_today(
    con,
    settings: ModelSettings,
    repo_dir: Path,
    now: datetime,
    *,
    code_dir: Path | None = None,
    state_version: str | None = None,
    code_version: tuple[str | None, bool] | None = None,
    calls: frozenset[str] = EARLY_ONLY,
    live_model: LiveModel = LiveModel.team(),  # noqa: B008 (frozen value)
    player_settings: PlayerModelSettings | None = None,
    id_links: Mapping[str, str] | None = None,
) -> RunResult:
    """Make the slate's calls and log an honest not_predicted line for every
    game that could not be called.

    **Early** (always): each slate game's first call. ``predicted`` if it
    tips off more than ``START_BUFFER`` after ``now``, else ``not_predicted``
    (too late). A slate game whose tip-off is still TBD gets a provisional
    ``not_predicted`` line (``_TBD_REASON``) that does NOT block a later
    early call once the tip is announced. Games dated within
    ``BACKFILL_DAYS`` before the slate date that already tipped off (or are
    TBD) and have no line at all are back-filled as ``not_predicted``
    (``_MISSED_REASON``).

    **Final** (``"final" in calls``): a game with a predicted line and no
    final gets a ``predicted`` final call inside ``[tip - 45, tip - 15)``,
    even if nothing changed (``revision_reason`` is set only when the
    revision rule would hold). **Revision** (``"revision" in calls``,
    player-model games only): otherwise, a newer injury report with a status
    change for either team and a move of at least two points against the
    latest predicted line, while ``now < tip - CALL_CUTOFF``. **Missed
    final** (``"final" in calls``): a game dated in ``[slate -
    BACKFILL_DAYS, slate]`` with a predicted line, no final, and ``now >=
    tip - CALL_CUTOFF`` gets one ``not_predicted`` final line
    (``_MISSED_FINAL_REASON``). Nothing is ever written for a game at or
    after ``tip - CALL_CUTOFF`` except that missed-final line.

    Each game's model is ``kind_for_game`` (a game keeps the model of its
    first call); one caller is built per model per run (``make_caller``,
    with ``id_links``: the cloud's provisional-id links, for the player
    model's ESPN rows), and all calls read ``AsOfView(con, now)``, so a
    result observed after ``now`` cannot reach any call. Player settings
    are loaded here (``live_model.load_player_settings``) when a game needs
    the player model and none were passed -- e.g. a game first called by v2
    after the flag went back to team. Every unusable-flag error
    (``LiveModelError``) is raised before any line is written.

    The log is append-only: lines are only ever added. Every later call of
    a game (revision, final, missed final) is written under the ``game_id``
    of the game's first line (``GameChain.game_id``), and a logged line
    under a provisional ``espn:`` id that ``read_id_links`` maps to an NBA
    id covers that NBA id too.

    Every line carries ``state_version`` (the cloud state commit the run
    was computed from; None on the Mac) and the ``code_version`` of
    ``code_dir`` (default ``repo_dir``), or ``code_version`` (sha, dirty)
    when the caller read it once already.
    """
    db.require_utc(now, "now")
    today = slate_date(now)
    range_games = _competitive_games(con, today - timedelta(days=BACKFILL_DAYS), today)
    backfill_games = [
        g for g in range_games
        if g.game_date < today and (g.tip_off_utc is None or g.tip_off_utc <= now)
    ]
    today_games = [g for g in range_games if g.game_date == today]
    last_cap = last_capture(con)
    stale_flag = bool(results_missing(con, now))
    # Which code made these lines -- see publish.code_version.
    code_sha, code_dirty = (code_version if code_version is not None
                            else publish.code_version(code_dir or repo_dir))
    view = AsOfView(con, now)

    # Per season: the log's chains and the provisional-id links.
    season_chains: dict[str, dict[tuple[str, str], GameChain]] = {}
    season_links: dict[str, dict[str, str]] = {}

    def load_season(season: str) -> None:
        if season not in season_chains:
            links = read_id_links(repo_dir, season)
            season_links[season] = links
            season_chains[season] = chains(read_log(log_path(repo_dir, season)), links)

    def chain_key(season: str, game_id: str, game_date: str) -> tuple[str, str]:
        load_season(season)
        links = season_links[season]
        return (links.get(game_id, game_id), game_date)

    def chain_of(g: SlateGame) -> GameChain | None:
        key = chain_key(g.season, g.game_id, g.game_date.isoformat())
        return season_chains[g.season].get(key)

    # Every game's model, before anything is written: an unusable flag or an
    # unknown first-call model fails the run with no lines.
    kinds: dict[tuple[str, date], ModelKind] = {
        (g.game_id, g.game_date): kind_for_game(live_model, today, chain_of(g))
        for g in range_games
    }
    if player_settings is None and "player" in kinds.values():
        player_settings = _lm.load_player_settings()

    callers: dict[str, object] = {}

    def call(kind: ModelKind, g: SlateGame, call_type: str) -> CallOutcome:
        if kind not in callers:
            callers[kind] = _lm.make_caller(con, kind, settings, player_settings,
                                            id_links=id_links)
        game = GameToPredict(g.game_id, g.season, g.game_date, g.home_team, g.away_team)
        return callers[kind].call(game, view, call_type)  # type: ignore[attr-defined]

    def settings_for(kind: ModelKind) -> dict | None:
        if kind == "player":
            return player_settings_dict(player_settings) if player_settings is not None else None
        return _settings_dict(settings)

    report_at = latest_report_at(con, now)
    history_cache: list[InjuryHistory] = []

    def changes_since(g: SlateGame, latest: dict) -> list[StatusChange]:
        if not history_cache:
            history_cache.append(InjuryHistory.from_view(view))
        history = history_cache[0]
        previous = latest.get("injury_report_at")
        before = (history.snapshot_at(g.game_date, as_utc(previous))
                  if previous is not None else None)
        after = history.snapshot_at(g.game_date, now)
        return status_changes(before, after, (g.away_team, g.home_team))

    predicted = not_predicted = skipped_duplicates = backfilled = 0
    revisions = finals = missed_finals = 0
    lines_written: list[dict] = []

    def base_line(g: SlateGame, kind: ModelKind, chain: GameChain | None, call_type: str,
                  game_id: str) -> dict:
        latest = chain.latest_predicted if chain is not None else None
        return {
            "predicted_at": now.isoformat(),
            "game_id": game_id,
            "season": g.season,
            "game_date": g.game_date.isoformat(),
            "tip_off_utc": g.tip_off_utc.isoformat() if g.tip_off_utc is not None else None,
            "home_team": g.home_team,
            "away_team": g.away_team,
            "settings": settings_for(kind),
            "stale_results": stale_flag,
            "last_result_capture": last_cap.isoformat() if last_cap is not None else None,
            "code_version": code_sha,
            "code_dirty": code_dirty,
            "state_version": state_version,
            "call_type": call_type,
            "supersedes": latest["predicted_at"] if latest is not None else None,
            "revision_reason": None,
            "injury_report_at": None,
            "availability": None,
            "model_version": MODEL_VERSION[kind],
        }

    def not_predicted_line(g: SlateGame, kind: ModelKind, chain: GameChain | None,
                           reason: str, *, call_type: str = "early",
                           game_id: str | None = None) -> dict:
        return {
            **base_line(g, kind, chain, call_type, game_id or g.game_id),
            "status": "not_predicted",
            "reason": reason,
            "spread": None,
            "p_home": None,
            "sentence": None,
            "terms": None,
            **_NO_MARKET,
        }

    def predicted_line(g: SlateGame, kind: ModelKind, chain: GameChain | None,
                       call_type: str, out: CallOutcome, game_id: str,
                       reason: str | None = None) -> dict:
        market_fields = _market_fields(con, g.game_id, now, out.sigma)
        latest = chain.latest_predicted if chain is not None else None
        if call_type != "early" and market_fields["market_p_home"] is None and latest is not None:
            # No fresh line: carry the latest call's market (display only).
            market_fields = {k: latest.get(k) for k in _NO_MARKET}
        v2 = out.model_version == MODEL_VERSION["player"]
        return {
            **base_line(g, kind, chain, call_type, game_id),
            "status": "predicted",
            "reason": None,
            "spread": _r6(out.spread),
            "p_home": _r6(out.p_home),
            "sentence": out.sentence,
            "terms": {name: _r6(value) for name, value in out.terms.items()},
            "settings": out.settings,
            "model_version": out.model_version,
            "availability": out.availability,
            "injury_report_at": report_at.isoformat() if v2 and report_at is not None else None,
            "revision_reason": reason,
            **market_fields,
        }

    def write(g: SlateGame, line: dict) -> None:
        if line["call_type"] != "early" and line["status"] == "predicted":
            # Nothing is sealed at or after tip - CALL_CUTOFF.
            assert g.tip_off_utc is not None and now < g.tip_off_utc - CALL_CUTOFF
        _append_line(log_path(repo_dir, g.season), line)
        lines_written.append(line)
        key = chain_key(g.season, line["game_id"], line["game_date"])
        chain = season_chains[g.season].get(key)
        season_chains[g.season][key] = GameChain((chain.lines if chain else ()) + (line,))

    for g in backfill_games:
        if chain_of(g) is not None:
            continue
        write(g, not_predicted_line(g, kinds[(g.game_id, g.game_date)], None, _MISSED_REASON))
        not_predicted += 1
        backfilled += 1

    for g in today_games:
        kind = kinds[(g.game_id, g.game_date)]
        chain = chain_of(g)
        if g.tip_off_utc is None:
            if chain is not None:
                skipped_duplicates += 1
                continue
            write(g, not_predicted_line(g, kind, None, _TBD_REASON))
            not_predicted += 1
            continue
        tip = g.tip_off_utc
        wrote = False

        # Early: once per game (a TBD line does not count).
        if chain is None or chain.early is None:
            if tip - START_BUFFER <= now:
                write(g, not_predicted_line(g, kind, chain, _NOT_PREDICTED_REASON))
                not_predicted += 1
            else:
                out = call(kind, g, "early")
                write(g, predicted_line(g, kind, chain, "early", out, g.game_id))
                predicted += 1
            wrote = True
            chain = chain_of(g)

        latest = chain.latest_predicted if chain is not None else None
        if latest is not None:
            if "final" in calls and chain.final is None and final_due(tip, now):
                out = call(kind, g, "final")
                reason = None
                if kind == "player":
                    changes = changes_since(g, latest)
                    if revision_due(latest, out.p_home, report_at, changes):
                        reason = revision_reason(report_at, changes)
                write(g, predicted_line(g, kind, chain, "final", out, chain.game_id, reason))
                finals += 1
                wrote = True
            elif ("revision" in calls and kind == "player" and report_at is not None
                  and still_callable(tip, now)):
                previous = latest.get("injury_report_at")
                changes = (changes_since(g, latest)
                           if previous is None or as_utc(previous) < report_at else [])
                if changes:
                    out = call(kind, g, "revision")
                    if revision_due(latest, out.p_home, report_at, changes):
                        write(g, predicted_line(g, kind, chain, "revision", out, chain.game_id,
                                                revision_reason(report_at, changes)))
                        revisions += 1
                        wrote = True
        if not wrote:
            skipped_duplicates += 1

    if "final" in calls:
        for g in range_games:
            if g.tip_off_utc is None or now < g.tip_off_utc - CALL_CUTOFF:
                continue
            chain = chain_of(g)
            if chain is None or chain.latest_predicted is None or chain.final is not None:
                continue
            write(g, not_predicted_line(g, kinds[(g.game_id, g.game_date)], chain,
                                        _MISSED_FINAL_REASON, call_type="final",
                                        game_id=chain.game_id))
            missed_finals += 1

    return RunResult(
        predicted, not_predicted, skipped_duplicates, stale_flag, lines_written, backfilled,
        revisions=revisions, finals=finals, missed_finals=missed_finals,
        model_versions=frozenset(line["model_version"] for line in lines_written),
    )


def grade(
    con, repo_dir: Path, season: str, now: datetime, *, code_dir: Path | None = None,
    code_version: tuple[str | None, bool] | None = None,
) -> int:
    """Grade every predicted game that now has a FINAL result.

    For each ``game_id``, the latest ``predicted`` line whose own
    ``predicted_at`` precedes its own cutoff is the one graded (the last
    call before tip: early, revision or final; on equal ``predicted_at`` the
    later log line wins) -- a ``not_predicted`` line is never a candidate.
    Each grade line names the graded line's ``call_type`` and carries the
    game's early call (``early_predicted_at``, ``early_p_home``,
    ``early_correct``; null when the game has no predicted early call). The cutoff is the EARLIER
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

    A line under a provisional ``espn:`` id that ``read_id_links`` maps to
    an NBA id is graded under its own id (the log is never rewritten): the
    cutoff uses the earliest current tip found under either id, the FINAL
    row is looked up under the line's own id, then under the linked id, and
    the grade line names the link in
    ``linked_game_id`` (null when there is none). A game graded under
    either id is not graded again.

    ``code_version`` on each grade line is read from ``code_dir`` (default
    ``repo_dir``): in the cloud the state checkout and the code checkout can
    be different directories.
    """
    log = read_log(log_path(repo_dir, season))
    links = read_id_links(repo_dir, season)
    game_chains = chains(log, links)

    def ids_of(game_id: str) -> list[str]:
        return [game_id, links[game_id]] if game_id in links else [game_id]

    # Keyed by the game the line is about (the linked id when there is one).
    latest_predicted: dict[str, dict] = {}
    for line in log:
        if line["status"] != "predicted":
            continue
        own_tip = datetime.fromisoformat(line["tip_off_utc"])
        # The earliest current tip known under any of the line's ids: a
        # stale row left under the provisional id must not hide that the
        # linked game was moved earlier.
        current_tips = [
            t for t in (_current_tip(con, gid) for gid in ids_of(line["game_id"])) if t is not None
        ]
        cutoff = min([own_tip, *current_tips])
        predicted_at = datetime.fromisoformat(line["predicted_at"])
        if predicted_at >= cutoff:
            continue
        game = links.get(line["game_id"], line["game_id"])
        current = latest_predicted.get(game)
        # Ties go to the later log line (an early and a final call made in
        # one run share a predicted_at: the final is graded).
        if current is None or predicted_at >= datetime.fromisoformat(current["predicted_at"]):
            latest_predicted[game] = line

    gpath = grades_path(repo_dir, season)
    already_graded: set[str] = set()
    for g in read_log(gpath):
        already_graded.add(g["game_id"])
        if g.get("linked_game_id"):
            already_graded.add(g["linked_game_id"])
    games_table = db.POINT_IN_TIME_TABLES["games"]

    code_sha, code_dirty = (code_version if code_version is not None
                            else publish.code_version(code_dir or repo_dir))
    appended = 0
    for key in sorted(latest_predicted):
        line = latest_predicted[key]
        game_id = line["game_id"]
        ids = ids_of(game_id)
        if any(gid in already_graded for gid in ids):
            continue
        row = None
        for gid in ids:
            row = con.execute(
                f"SELECT home_points, away_points FROM {games_table} "
                "WHERE game_id = ? AND game_date = ? AND status = 'FINAL' "
                "AND home_points IS NOT NULL AND away_points IS NOT NULL "
                "ORDER BY observed_at DESC LIMIT 1",
                [gid, date.fromisoformat(line["game_date"])],
            ).fetchone()
            if row is not None:
                break
        if row is None:
            continue
        home_points, away_points = row
        home_won = home_points > away_points
        p_home = line["p_home"]
        market_p = line.get("market_p_home")
        early = game_chains[(key, line["game_date"])].early
        if early is not None and early.get("status") != "predicted":
            early = None
        grade_line = {
            "game_id": game_id,
            "linked_game_id": links.get(game_id),
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
            "call_type": call_type_of(line),
            # The chain's early call, so the early-call record can be kept
            # beside the graded one (null when it has no predicted early call).
            "early_predicted_at": early["predicted_at"] if early is not None else None,
            "early_p_home": early["p_home"] if early is not None else None,
            "early_correct": (
                None if early is None else (early["p_home"] >= 0.5) == home_won
            ),
        }
        _append_line(gpath, grade_line)
        appended += 1

    return appended


def grade_seasons(
    con, repo_dir: Path, season: str, now: datetime, *, code_dir: Path | None = None,
    code_version: tuple[str | None, bool] | None = None,
) -> tuple[int, set[str]]:
    """Grade ``season``'s prediction log and, when it exists, the previous
    season's -- the one grading rule shared by ``predict-today``,
    ``capture-results`` and ``cloud-run``. Returns (grade lines appended,
    seasons touched).

    A season boundary can leave predictions from the PREVIOUS season still
    ungraded (its games' results arrive after today's slate has already
    rolled over to a new season label) -- grade that log too, whenever it
    exists, so those predictions are not stranded ungraded forever. grade()
    is a no-op (appends nothing) for every game already graded, so this is
    always safe to run again -- whichever command ran it first."""
    from predictor.config import previous_season_label

    graded = 0
    seasons_touched = {season}
    previous_season = previous_season_label(season)
    if log_path(repo_dir, previous_season).exists():
        graded += grade(con, repo_dir, previous_season, now, code_dir=code_dir,
                        code_version=code_version)
        seasons_touched.add(previous_season)
    graded += grade(con, repo_dir, season, now, code_dir=code_dir, code_version=code_version)
    return graded, seasons_touched
