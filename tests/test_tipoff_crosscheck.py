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
