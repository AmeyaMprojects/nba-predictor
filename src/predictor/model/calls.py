"""Call rules: when a game is called, and when a call is revised.

Pure: no I/O, and every time comes in as an argument (UTC-aware).

A game's calls form one chain in the append-only prediction log: one early
call, any number of revisions, and one final call (``predicted`` or a
missed-final ``not_predicted``). Lines written before calls existed carry no
``call_type`` and read as early calls.

- The final call is due in ``[tip - FINAL_LEAD, tip - CALL_CUTOFF)``. That
  window is 30 minutes wide, so it always holds two 15-minute loop marks.
- Nothing is called at or after ``tip - CALL_CUTOFF``.
- A revision needs a newer injury report than the latest call's, a status
  change for one of the two teams, and a move of at least
  ``REVISION_MIN_MOVE`` in the home win probability.

All window arithmetic is in UTC, so a DST change does not shift it. The
reason text shows the report time in IST.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from predictor.model.availability import Snapshot

CallType = Literal["early", "revision", "final"]
ALL_CALLS: frozenset[str] = frozenset({"early", "revision", "final"})
EARLY_ONLY: frozenset[str] = frozenset({"early"})

FINAL_LEAD = timedelta(minutes=45)
CALL_CUTOFF = timedelta(minutes=15)
REVISION_MIN_MOVE = 0.02
MAX_NAMED_CHANGES = 3

IST = ZoneInfo("Asia/Kolkata")

# A TBD line is provisional: it never blocks a later prediction of the same
# game once its tip-off is announced (see live.predict_today's dedupe).
TBD_REASON = "tip-off time not announced when the prediction run happened"


def call_type_of(line: dict) -> str:
    """A line's call type; lines from before calls existed are early."""
    return line.get("call_type") or "early"


@dataclass(frozen=True)
class GameChain:
    """Every log line of one game on one date, in log order."""

    lines: tuple[dict, ...]

    @property
    def game_id(self) -> str:
        """The id every later call of this game is written under."""
        return self.lines[0]["game_id"]

    @property
    def early(self) -> dict | None:
        for line in self.lines:
            if call_type_of(line) == "early" and line.get("reason") != TBD_REASON:
                return line
        return None

    @property
    def final(self) -> dict | None:
        for line in self.lines:
            if call_type_of(line) == "final":
                return line
        return None

    @property
    def predicted(self) -> tuple[dict, ...]:
        return tuple(line for line in self.lines if line.get("status") == "predicted")

    @property
    def first_predicted(self) -> dict | None:
        predicted = self.predicted
        return predicted[0] if predicted else None

    @property
    def latest_predicted(self) -> dict | None:
        predicted = self.predicted
        return predicted[-1] if predicted else None


def chains(log: list[dict], links: dict[str, str]) -> dict[tuple[str, str], GameChain]:
    """One chain per ``(game, game_date)``, keyed by the linked NBA id when a
    provisional id has one, in order of each chain's first line."""
    grouped: dict[tuple[str, str], list[dict]] = {}
    for line in log:
        key = (links.get(line["game_id"], line["game_id"]), line["game_date"])
        grouped.setdefault(key, []).append(line)
    return {key: GameChain(tuple(lines)) for key, lines in grouped.items()}


def final_due(tip: datetime, now: datetime) -> bool:
    """True inside the final-call window ``[tip - 45 min, tip - 15 min)``."""
    return tip - FINAL_LEAD <= now < tip - CALL_CUTOFF


def still_callable(tip: datetime, now: datetime) -> bool:
    """True while a revision or final may still be written."""
    return now < tip - CALL_CUTOFF


def _norm(status: str | None) -> str | None:
    return status.strip().lower() if status is not None else None


@dataclass(frozen=True)
class StatusChange:
    """One player's injury status moving between two reports. ``player`` is
    the report's ``"Last, First"`` key; None means not on the report."""

    team: str
    player: str
    before: str | None
    after: str | None

    def text(self) -> str:
        if "," in self.player:
            surname = self.player.split(",", 1)[0].strip()
        else:
            surname = self.player.split()[-1] if self.player.split() else self.player
        status = self.after.strip().lower() if self.after else "available"
        return f"{surname} ({self.team}) {status}"


def status_changes(
    before: Snapshot | None, after: Snapshot | None, teams: tuple[str, str]
) -> list[StatusChange]:
    """Status changes for the two teams between two reports, ordered by the
    team's position in ``teams``, then player. A player missing from a
    report counts as None; statuses compare after ``strip().lower()``."""
    old = before.statuses if before is not None else {}
    new = after.statuses if after is not None else {}
    order = {team: i for i, team in enumerate(teams)}
    keys = {k for k in (*old, *new) if k[0] in order}
    out = []
    for team, player in sorted(keys, key=lambda k: (order[k[0]], k[1])):
        b, a = old.get((team, player)), new.get((team, player))
        if _norm(b) != _norm(a):
            out.append(StatusChange(team, player, b, a))
    return out


def _as_datetime(value: datetime | str) -> datetime:
    return value if isinstance(value, datetime) else datetime.fromisoformat(value)


def revision_due(
    latest: dict, new_p: float, report_at: datetime | None, changes: list[StatusChange]
) -> bool:
    """A revision is due on a newer report with a status change and a move of
    at least ``REVISION_MIN_MOVE`` against the latest sealed call."""
    if report_at is None or not changes:
        return False
    previous = latest.get("injury_report_at")
    if previous is not None and _as_datetime(previous) >= report_at:
        return False
    return round(abs(new_p - latest["p_home"]), 6) >= REVISION_MIN_MOVE


def revision_reason(report_at: datetime, changes: list[StatusChange]) -> str:
    """``"new injury report 23:30 IST: Murray (DEN) out"``, naming at most
    ``MAX_NAMED_CHANGES`` changes."""
    named = ", ".join(c.text() for c in changes[:MAX_NAMED_CHANGES])
    text = f"new injury report {report_at.astimezone(IST):%H:%M} IST: {named}"
    rest = len(changes) - MAX_NAMED_CHANGES
    return f"{text} and {rest} more" if rest > 0 else text
