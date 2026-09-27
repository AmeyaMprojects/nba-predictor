"""The injury-report PDFs as an independent check on the schedule's tip-offs.

Spec 1.1: the PDF tip-off parser is retained as a cross-check test, not a
runtime fallback -- a second source keeps catching leaks without a second
code path producing cutoffs.

For every one of the 7,200 games where both sources exist (measured
2026-09-27), two independent conditions are checked against the schedule's
resolved tip-off (`sched`):

  1. `sched` matches, within 60 seconds, at least one of
       (a) the most recent PDF vintage filed BEFORE `sched` (or, when every
           filing came after `sched`, the earliest filing), or
       (b) the earliest time any vintage recorded.
     (a) covers games moved LATER: a day-before report showed an earlier
     slot and the game-day report, filed before either time, showed the
     schedule's (0021900701, 0022000206, 0022400624). (b) covers games
     moved EARLIER, where the correcting report came after `sched`
     (0022000834, 0022200161). The 60 seconds absorb opening-night ':01'
     schedule times the PDFs round (0022000001, 0022200001).

  2. No PDF filing observed AT OR AFTER `sched` reports a time earlier than
     `sched - 60s`. Condition 1 alone is not sufficient to catch every
     leak: it sorts filings into "before sched" and "after sched" using
     `sched` itself, so a `sched` that is WRONG in the stale-late direction
     -- the schedule claims a later tip-off than the real one -- can still
     satisfy condition 1, as long as some filing happens to land in the
     "before sched" bucket at a time that (coincidentally, or because it
     is itself stale) matches. A report filed at or after `sched` that
     nonetheless reports a materially EARLIER tip-off is direct evidence
     that `sched` itself is too late, regardless of which bucket condition
     1 sorted it into -- exactly the leak this cross-check exists to catch,
     and the one gap condition 1 alone left open.

A schedule tip-off later than the real one fails at least one of these:
condition 1 in the usual case, and condition 2 whenever a filing made at or
after the (wrong) `sched` records the real, earlier time. Measured true (0
violations of either condition) for all 7,200 games as of 2026-09-27.
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

            problems = []

            # Condition 1: sched matches (a) the most recent pre-sched
            # filing (or, if none, the earliest filing) or (b) the
            # earliest recorded time.
            before = [f for f in both if f[0] < sched]
            pool = before if before else both
            pick = max(o for o, _ in pool) if before else min(o for o, _ in pool)
            candidates = {t for o, t in pool if o == pick}
            earliest = min(t for _, t in both)
            if not (any(_close(sched, t) for t in candidates) or _close(sched, earliest)):
                problems.append(("no matching vintage", sorted(candidates), earliest))

            # Condition 2: no filing observed at or after sched may report
            # a materially earlier time -- catches a stale-late sched that
            # condition 1 alone cannot see (it never even looks at filings
            # observed at/after sched).
            stale_late = [
                (o, t) for o, t in both if o >= sched and t < sched - _TOLERANCE
            ]
            if stale_late:
                problems.append(("post-sched filing reports an earlier time", stale_late))

            if problems:
                disagreements.append((gid, sched, problems))

        assert compared >= 7200, f"only {compared} games had both sources"
        assert disagreements == [], (
            f"{len(disagreements)} game(s) where the schedule's tip-off disagrees "
            f"with the injury reports: {disagreements[:5]}"
        )
    finally:
        con.close()
