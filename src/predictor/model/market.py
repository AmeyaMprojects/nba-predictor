"""Turn stored betting lines into the market's home-win probability.

This module is a *reader of* the odds table for comparison and display
only. The model itself (``Stage1Predictor``, ``fit``, ``tuning``) never
imports it and never reads odds -- the market is something the model is
measured against, not an input to it.

Conventions:

- Moneylines are American odds: -200 means "stake 200 to win 100"
  (implied 2/3), +170 means "stake 100 to win 170" (implied 100/270).
- The stored ``spread`` is quoted for the HOME team, negative when home is
  favoured (home -5.5 = home expected to win by 5.5 points). The model's
  own spread has the opposite sign (positive = home favoured), so a market
  spread ``s`` becomes the model-convention margin ``-s`` before it goes
  through ``win_probability``.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import duckdb

from predictor.asof import AsOfView
from predictor.model.ratings import win_probability

HISTORICAL_SOURCE = "kaggle_sbr"
LIVE_SOURCE = "theoddsapi"


@dataclass(frozen=True)
class OddsLine:
    """One book's line for one game, as stored."""

    book: str
    home_price: int | None
    away_price: int | None
    spread: float | None  # home spread, negative = home favoured
    observed_at: datetime


@dataclass(frozen=True)
class MarketView:
    """The market's view of one game.

    ``p_home`` is the median across the lines used; ``spread`` the median
    home spread across every line that has one (``None`` if none do);
    ``books`` how many lines went into ``p_home``; ``observed_at`` the
    latest of those lines' times; ``from_spread`` is True when no line had
    both moneylines and ``p_home`` was worked out from the spread using the
    caller's sigma.
    """

    p_home: float
    spread: float | None
    books: int
    observed_at: datetime
    from_spread: bool = False


def american_to_prob(price: int) -> float:
    """Implied probability of an American price, vig included."""
    if price == 0 or -100 < price < 100:
        raise ValueError(f"not an American price: {price}")
    if price < 0:
        return -price / (-price + 100)
    return 100 / (price + 100)


def devig(p_home_raw: float, p_away_raw: float) -> float:
    """Remove the bookmaker's margin by scaling the two sides to sum to 1."""
    return p_home_raw / (p_home_raw + p_away_raw)


def market_p_home(lines: Sequence[OddsLine], sigma: float) -> MarketView | None:
    """Median market home-win probability across ``lines``.

    Lines with both moneylines are de-vigged and used. Only when no line
    has both moneylines are spreads used, through ``win_probability(-spread,
    sigma)``. ``None`` when nothing is usable.
    """
    priced = [
        ln for ln in lines if ln.home_price is not None and ln.away_price is not None
    ]
    spreads = [ln.spread for ln in lines if ln.spread is not None]
    spread = statistics.median(spreads) if spreads else None
    if priced:
        probs = [
            devig(american_to_prob(ln.home_price), american_to_prob(ln.away_price))
            for ln in priced
        ]
        used, from_spread = priced, False
    else:
        used = [ln for ln in lines if ln.spread is not None]
        if not used:
            return None
        probs = [win_probability(-ln.spread, sigma) for ln in used]
        from_spread = True
    return MarketView(
        p_home=statistics.median(probs),
        spread=spread,
        books=len(used),
        observed_at=max(ln.observed_at for ln in used),
        from_spread=from_spread,
    )


_COLUMNS = "game_id, book, home_price, away_price, spread, observed_at"


def _line(row) -> tuple[str, OddsLine]:
    game_id, book, home_price, away_price, spread, observed_at = row
    return game_id, OddsLine(book, home_price, away_price, spread, observed_at)


def historical_lines(con, as_of: datetime | None = None) -> dict[str, list[OddsLine]]:
    """Every historical closing line (``source='kaggle_sbr'``) linked to a
    game, grouped by ``game_id``. Read through ``AsOfView`` at ``as_of``
    (default: now)."""
    view = AsOfView(con, as_of or datetime.now(UTC))
    rows = (
        view.table("odds_snapshots")
        .filter(f"source = '{HISTORICAL_SOURCE}' AND game_id IS NOT NULL")
        .project(_COLUMNS)
        .order("game_id, book, observed_at")
        .fetchall()
    )
    out: dict[str, list[OddsLine]] = {}
    for row in rows:
        game_id, line = _line(row)
        out.setdefault(game_id, []).append(line)
    return out


def live_lines(con, game_id: str, now: datetime) -> list[OddsLine]:
    """The latest live snapshot per book for ``game_id`` observed at or
    before ``now``."""
    view = AsOfView(con, now)
    # Latest per (game_key, book) -- the table's own entity key -- so a
    # historical row can never shadow a live one or vice versa.
    rows = (
        view.latest("odds_snapshots")
        .filter(f"source = '{LIVE_SOURCE}'")
        # A bound constant, not interpolated SQL text.
        .filter(duckdb.ColumnExpression("game_id") == duckdb.ConstantExpression(game_id))
        .project(_COLUMNS)
        .order("book")
        .fetchall()
    )
    return [_line(r)[1] for r in rows]

