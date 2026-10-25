"""`compare-shadow`: do the cloud's would-be calls on `cloud-shadow` match
the calls the Mac published on `main`?

The bar for cut-over (plan 2, Task 13/14):

- at least ``MIN_SHADOW_DAYS`` slate days on which both sides predicted;
- every game the Mac predicted has a cloud call sealed before tip-off;
- the largest win-probability difference is at most ``MAX_P_DIFF``.

Also checked: a game neither side predicted must have been logged by the
cloud for the same kind of reason (too late / missed run). Calls only the
cloud made are listed for the owner but never fail the comparison.

"Sealed" means the line's ``predicted_at``. "Tip-off" is the earliest
``tip_off_utc`` either side recorded on a predicted line for that game --
strict on purpose, so a cloud that believed a later (wrong) tip cannot pass
on its own say-so.

Pure functions only: the CLI reads both logs (``git show``) and passes the
parsed lines in.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

MIN_SHADOW_DAYS = 3
MAX_P_DIFF = 0.01

# Reason categories for not_predicted lines. Matched on a phrase of each
# reason in predictor.model.live, case-insensitively, so a reworded reason
# with the same meaning still compares equal.
_TBD = "tip-off time not announced"
_CATEGORIES = (
    ("too late", "already started"),
    ("missed run", "no prediction was made before tip-off"),
)


def reason_category(reason: str | None) -> str:
    text = (reason or "").lower()
    if _TBD in text:
        return "tip-off unknown"
    for name, phrase in _CATEGORIES:
        if phrase in text:
            return name
    return f"other: {reason}"


@dataclass(frozen=True)
class GamePair:
    game_id: str
    game_date: date
    mac_p: float | None
    cloud_p: float | None
    mac_at: datetime | None
    cloud_at: datetime | None
    cloud_before_tip: bool
    # Extras for the owner's report.
    cloud_game_id: str | None = None
    home_team: str | None = None
    away_team: str | None = None
    tip: datetime | None = None
    mac_status: str | None = None
    cloud_status: str | None = None
    mac_reason: str | None = None
    cloud_reason: str | None = None
    problem: str | None = None


@dataclass(frozen=True)
class ShadowReport:
    slate_days: list[date]
    pairs: list[GamePair]
    mac_only: list[str]
    cloud_only: list[str]
    max_abs_dp: float
    ok: bool
    reasons: list[str]
    since: date | None = None


def _at(value) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value) if isinstance(value, str) else value


def _num(x: float) -> str:
    """0.011 -> "0.011", 0.02 -> "0.02", 0.0 -> "0"."""
    text = f"{x:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def _p(x: float | None) -> str:
    return "none" if x is None else f"{x:.3f}"


def _when(x: datetime | None) -> str:
    return "never" if x is None else x.strftime("%Y-%m-%d %H:%M UTC")


def _group(lines: list[dict], since: date, links: dict[str, str]) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = {}
    for line in lines:
        day = date.fromisoformat(line["game_date"])
        if day < since:
            continue
        if line.get("status") == "not_predicted" and reason_category(line.get("reason")) == "tip-off unknown":
            continue  # provisional: never a call, never blocks one
        gid = links.get(line["game_id"], line["game_id"])
        out.setdefault((gid, day), []).append(line)
    return out


def _choose(lines: list[dict], tip: datetime | None) -> dict | None:
    """The latest predicted line sealed before tip (else the latest
    predicted line, else the latest not_predicted line)."""
    if not lines:
        return None
    predicted = sorted((x for x in lines if x.get("status") == "predicted"),
                       key=lambda x: _at(x["predicted_at"]))
    if predicted:
        sealed = [x for x in predicted if tip is None or _at(x["predicted_at"]) < tip]
        return (sealed or predicted)[-1]
    return sorted(lines, key=lambda x: _at(x["predicted_at"]))[-1]


def compare_logs(mac_lines: list[dict], cloud_lines: list[dict], since: date,
                 id_links: dict[str, str]) -> ShadowReport:
    mac = _group(mac_lines, since, id_links)
    cloud = _group(cloud_lines, since, id_links)

    pairs: list[GamePair] = []
    for key in sorted(set(mac) | set(cloud), key=lambda k: (k[1], k[0])):
        gid, day = key
        tips = [_at(x["tip_off_utc"]) for x in mac.get(key, []) + cloud.get(key, [])
                if x.get("status") == "predicted" and x.get("tip_off_utc")]
        tip = min(tips) if tips else None
        m = _choose(mac.get(key, []), tip)
        c = _choose(cloud.get(key, []), tip)
        if tip is None:
            seen = [_at(x["tip_off_utc"]) for x in (m, c) if x and x.get("tip_off_utc")]
            tip = min(seen) if seen else None
        m_pred = bool(m and m["status"] == "predicted")
        c_pred = bool(c and c["status"] == "predicted")
        cloud_at = _at(c["predicted_at"]) if c else None
        before = bool(c_pred and (tip is None or cloud_at < tip))

        problem = None
        if m_pred and not c_pred:
            problem = ("no cloud call" if c is None else
                       f"no cloud call -- the cloud logged it as not predicted: {c.get('reason')}")
        elif m_pred and not before:
            problem = f"the cloud call was sealed after tip-off ({_when(tip)})"
        elif m_pred and c_pred:
            dp = round(abs(m["p_home"] - c["p_home"]), 6)
            if dp > MAX_P_DIFF:
                problem = f"win probabilities differ by {dp:.3f} (more than {MAX_P_DIFF})"
        elif m is not None and c is None:
            problem = "the cloud has no line for a game the Mac logged as not predicted"
        elif m is not None and not c_pred:
            mc, cc = reason_category(m.get("reason")), reason_category(c.get("reason"))
            if mc != cc:
                problem = (f"neither side predicted it, but for different reasons "
                           f"(Mac: {mc}; cloud: {cc})")

        ref = m or c
        pairs.append(GamePair(
            game_id=gid, game_date=day,
            mac_p=m["p_home"] if m_pred else None,
            cloud_p=c["p_home"] if c_pred else None,
            mac_at=_at(m["predicted_at"]) if m else None,
            cloud_at=cloud_at,
            cloud_before_tip=before,
            cloud_game_id=c["game_id"] if c else None,
            home_team=ref.get("home_team"), away_team=ref.get("away_team"),
            tip=tip,
            mac_status=m["status"] if m else None,
            cloud_status=c["status"] if c else None,
            mac_reason=m.get("reason") if m else None,
            cloud_reason=c.get("reason") if c else None,
            problem=problem,
        ))

    both = [p for p in pairs if p.mac_p is not None and p.cloud_p is not None]
    slate_days = sorted({p.game_date for p in both})
    max_abs_dp = max((round(abs(p.mac_p - p.cloud_p), 6) for p in both), default=0.0)
    mac_only = [p.game_id for p in pairs if p.mac_p is not None and p.cloud_p is None]
    cloud_only = [p.cloud_game_id for p in pairs if p.cloud_p is not None and p.mac_p is None]
    late = [p.game_id for p in both if not p.cloud_before_tip]
    np_mismatch = [p.game_id for p in pairs
                   if p.mac_status == "not_predicted" and p.problem is not None]

    reasons: list[str] = []
    if len(slate_days) < MIN_SHADOW_DAYS:
        reasons.append(f"only {len(slate_days)} slate days where both sides predicted; "
                       f"need at least {MIN_SHADOW_DAYS}")
    if mac_only:
        reasons.append(f"{len(mac_only)} Mac call(s) with no cloud call: {', '.join(mac_only)}")
    if late:
        reasons.append(f"{len(late)} cloud call(s) sealed after tip-off: {', '.join(late)}")
    if max_abs_dp > MAX_P_DIFF:
        reasons.append(f"largest difference {_num(max_abs_dp)} > {_num(MAX_P_DIFF)}")
    if np_mismatch:
        reasons.append(f"{len(np_mismatch)} game(s) the Mac did not predict were logged "
                       f"differently by the cloud: {', '.join(np_mismatch)}")

    return ShadowReport(
        slate_days=slate_days, pairs=pairs, mac_only=mac_only, cloud_only=cloud_only,
        max_abs_dp=max_abs_dp, ok=not reasons, reasons=reasons, since=since,
    )


def _game(p: GamePair) -> str:
    teams = f"{p.away_team} at {p.home_team}" if p.home_team and p.away_team else ""
    ident = p.game_id
    if p.cloud_game_id and p.cloud_game_id != p.game_id:
        ident += f" (cloud id {p.cloud_game_id})"
    return f"{p.game_date.isoformat()}  {ident}  {teams}".rstrip()


def _side(status: str | None, prob: float | None, reason: str | None) -> str:
    if status is None:
        return "no line"
    if status == "predicted":
        return _p(prob)
    return f"not predicted ({reason_category(reason)})"


def format_report(report: ShadowReport) -> str:
    """Plain English for the owner: one row per game, then each mismatch
    with both probabilities, both seal times and why, then the verdict."""
    since = f" since {report.since.isoformat()}" if report.since else ""
    out = [f"Shadow comparison{since}: Mac (main) against cloud (cloud-shadow)", ""]
    if not report.pairs:
        out.append("No games to compare.")
    for p in report.pairs:
        if p.problem:
            verdict = "MISMATCH"
        elif p.mac_status is None:
            verdict = "cloud only (not a failure)"
        else:
            verdict = "match"
        out.append(f"{_game(p)} -- Mac {_side(p.mac_status, p.mac_p, p.mac_reason)}, "
                   f"cloud {_side(p.cloud_status, p.cloud_p, p.cloud_reason)} -- {verdict}")
    bad = [p for p in report.pairs if p.problem]
    if bad:
        out += ["", f"Mismatches ({len(bad)}):"]
        for p in bad:
            out.append(f"- {_game(p)} (tip-off {_when(p.tip)})")
            out.append(f"    Mac:   {_side(p.mac_status, p.mac_p, p.mac_reason)}, "
                       f"sealed {_when(p.mac_at)}")
            out.append(f"    cloud: {_side(p.cloud_status, p.cloud_p, p.cloud_reason)}, "
                       f"sealed {_when(p.cloud_at)}")
            out.append(f"    why:   {p.problem}")
    if report.cloud_only:
        out += ["", "Calls only the cloud made (reported, not a failure): "
                + ", ".join(report.cloud_only)]
    days = ", ".join(d.isoformat() for d in report.slate_days) or "none"
    out += ["", f"Slate days where both sides predicted: {len(report.slate_days)} ({days})",
            f"Largest win-probability difference: {_num(report.max_abs_dp)} "
            f"(allowed {_num(MAX_P_DIFF)})", ""]
    if report.ok:
        out.append("PASS -- the cloud matches the Mac.")
    else:
        out.append("FAIL -- the cloud does not yet meet the bar:")
        out += [f"- {r}" for r in report.reasons]
    return "\n".join(out) + "\n"
