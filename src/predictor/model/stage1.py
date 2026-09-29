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
        probability uses the exact spread.

        t7-fix1 finding 4: `round(-0.03, 1)` is `-0.0`, and `f"{-0.0:+.1f}"`
        prints "-0.0" -- a term that is, to a reader, plainly zero shown
        with a minus sign in front of it, and no legend anywhere explains
        what a sign even means. Adding 0.0 turns -0.0 into +0.0 (IEEE 754:
        adding +0.0 to -0.0 rounds to +0.0), fixing the display without
        changing which branch below fires -- `-0.0 == 0.0` in Python, so
        the total>0 / total<0 / pick'em branching below was never affected
        by the sign bit either way.
        """
        shown = [(name, round(value, 1) + 0.0) for name, value in self.terms()]
        total = round(sum(v for _, v in shown), 1) + 0.0
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
        self._last_applied_key = None
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

    def _fetch_finals(self, view, since=None):
        """Every visible FINAL competitive result, deduplicated to the
        latest-observed row per game_id (so a correction always supersedes
        the row it corrects, however the two arrive), sorted by
        (game_date, game_id).

        `since`, when given, restricts to rows newly OBSERVED since the
        previous catch-up -- but such a row can still describe an OLDER
        game (a slow first report, or a correction to a game already
        applied). That is exactly what `_needs_rebuild` below inspects.
        """
        rel = view.table("games").filter(
            "status = 'FINAL' AND home_points IS NOT NULL AND away_points IS NOT NULL "
            f"AND substr(game_id, 1, 3) IN ({_COMPETITIVE_SQL})"
        )
        if since is not None:
            rel = rel.filter(f"observed_at > TIMESTAMPTZ '{since.isoformat()}'")
        rows = rel.project(
            "game_id, season, game_date, home_team, away_team, home_points, "
            "away_points, observed_at"
        ).fetchall()
        latest: dict[str, tuple] = {}
        for row in rows:
            gid, observed_at = row[0], row[7]
            current = latest.get(gid)
            if current is None or observed_at > current[7]:
                latest[gid] = row
        return sorted(latest.values(), key=lambda r: (r[2], r[0]))

    def _needs_rebuild(self, new_rows) -> bool:
        """True when applying `new_rows` incrementally, on top of what is
        already applied, could apply results out of the order a fresh
        rebuild would use.

        A fresh rebuild always applies every visible row in strict
        (game_date, game_id) order, and that order matters WITHIN a date
        too: each apply's predicted margin uses home_court(), the rolling
        mean of margins from every earlier apply, so two games on the same
        date applied in a different relative order move ratings
        differently (task-5 fix round 2 -- the round-1 ruling that only
        compared game_date, letting same-date rows through unconditionally,
        was wrong). So this must rebuild not just when a newly-visible
        row's game_date is strictly earlier than what is already applied,
        but whenever its (game_date, game_id) key sorts AT OR BEFORE the
        LAST (highest) (game_date, game_id) key already applied --
        including equal, which is exactly a correction to a game already
        applied. A row whose key sorts strictly AFTER the last applied key
        is safe to apply incrementally: a fresh rebuild would place it
        after everything already applied too, in the same relative order.
        """
        if self._last_applied_key is None:
            return False
        for gid, _season, gd, *_rest in new_rows:
            if gid in self._applied or (gd, gid) <= self._last_applied_key:
                return True
        return False

    def _apply_rows(self, rows) -> None:
        for gid, season, gd, home, away, hp, ap, _observed_at in rows:
            self._applied.add(gid)
            venue = self.venues.venue(gid)
            self._ratings.apply(
                Result(gid, season, gd, home, away, hp, ap, venue.is_neutral if venue else False)
            )
            key = (gd, gid)
            if self._last_applied_key is None or key > self._last_applied_key:
                self._last_applied_key = key

    def _catch_up(self, view) -> None:
        if self._as_of is not None and view.as_of < self._as_of:
            self._reset()

        if self._as_of is None:
            self._apply_rows(self._fetch_finals(view))
        else:
            new_rows = self._fetch_finals(view, since=self._as_of)
            if self._needs_rebuild(new_rows):
                # Results arrived out of the order a fresh rebuild would
                # apply them in (a correction, a late first report for an
                # earlier game, or a same-date game whose id sorts before
                # one already applied): the incremental ratings state
                # built so far cannot be trusted to match what a fresh
                # rebuild in strict (game_date, game_id) order would
                # produce -- see task-5 fix rounds 1 and 2. Rebuild from
                # everything visible at this cutoff instead of trying to
                # patch the running state.
                self._reset()
                self._apply_rows(self._fetch_finals(view))
            else:
                self._apply_rows(new_rows)
        self._as_of = view.as_of
