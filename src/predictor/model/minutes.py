"""Expected minutes, their redistribution, and team strength from players.

A team plays 240 minutes. Expected minutes are scaled to that total, with
no player above ``MINUTES_CAP``; minutes cut from a capped player are spread
over the others in proportion to their own minutes. When a player may miss
the game, his minutes are scaled by his play probability and the team is
renormalised, so his minutes flow to the rest of the rotation.
"""

from __future__ import annotations

from predictor.model.player_ratings import ROOKIE_PRIOR

TEAM_MINUTES = 240.0
MINUTES_CAP = 44.0
ROTATION_GAMES = 10


def normalize_minutes(
    minutes: dict[int, float],
    total: float = TEAM_MINUTES,
    cap: float = MINUTES_CAP,
) -> dict[int, float]:
    """Scale to ``total``, then cap and re-spread until no one is over ``cap``.

    If every player ends up capped, they all stay at the cap, and if the
    excess has nowhere to go because every uncapped player has 0 minutes, it
    is dropped. In both cases the team sum is then below ``total``.
    All-zero input stays zero.
    """
    clean = {p: max(0.0, float(m)) for p, m in minutes.items()}
    current = sum(clean.values())
    if current <= 0.0:
        return {p: 0.0 for p in clean}
    out = {p: m * total / current for p, m in clean.items()}
    capped: set[int] = set()
    while True:
        over = [p for p, m in out.items() if p not in capped and m > cap]
        if not over:
            break
        for p in over:
            capped.add(p)
            out[p] = cap
        free = [p for p in out if p not in capped]
        free_sum = sum(out[p] for p in free)
        if free_sum <= 0.0:
            break
        scale = (total - cap * len(capped)) / free_sum
        for p in free:
            out[p] *= scale
    return out


def strength(
    minutes: dict[int, float],
    ratings: dict[int, float],
    default: float = ROOKIE_PRIOR,
) -> float:
    """Team strength: sum over players of (minutes / 48) x per-48 rating.

    A player missing from ``ratings`` is rated ``default`` (the rookie prior
    unless the caller says otherwise).
    """
    return sum((m / 48.0) * ratings.get(p, default) for p, m in minutes.items())


def available_minutes(
    normal: dict[int, float], play_prob: dict[int, float]
) -> dict[int, float]:
    """Expected minutes given play probabilities (missing means 1.0)."""
    return normalize_minutes({p: play_prob.get(p, 1.0) * m for p, m in normal.items()})
