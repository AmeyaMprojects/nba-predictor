"""The player-model candidate as an ``AsOfView`` predictor (plan 1, Task 10).

``PlayerStage1Predictor.explain(game, view, call)`` gives the candidate's
expected home margin from what ``view`` shows and nothing else:

- Team ratings and home court come from a composed ``Stage1Predictor``
  built from the settings' team params, so they are caught up exactly as
  the current model catches up (``Stage1Predictor._catch_up``).
- Player ratings, rosters and recent minutes are rebuilt from
  ``view.table("player_games")`` for games dated before the game, with the
  replay's own code (``player_sim._replay``) and the same read order: the
  books are read ``as_of`` the game date, after every earlier day and
  before any row of that day.
- The injury snapshot is the latest visible report listing the game date
  (``InjuryHistory.from_view``), and play probabilities come from the
  settings' availability table for the call.

The per-call terms are computed by ``player_sim.call_terms``, the function
the fast replay uses, so the two cannot drift apart; a parity test checks
that both give the same spread for every game and both calls.

This plan does not make the predictor live: ``live.py`` and
``predict-today`` still use the current model. Plan 3 decides the switch.

A shipped correction is retrained from the database when the predictor is
built (``correction_rows``: the shipped additive settings replayed over the
correction's training seasons), deterministically, and never pickled.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np

from predictor.backtest.baselines import GameToPredict
from predictor.model import adjustments as adj
from predictor.model import fit
from predictor.model.adjustments import FEATURE_NAMES, Coefficients
from predictor.model.availability import (
    CALLS,
    AvailabilityTable,
    InjuryHistory,
    estimate,
)
from predictor.model.candidate import (
    CallFeatures,
    CandidateEvaluation,
    CandidateResult,
    GameParts,
    best_sigma,
    design_row,
    feature_names,
    load_player_index,
    spread,
)
from predictor.model.correction import GBM_PARAMS, Correction, feature_row
from predictor.model.fit import FitError, _Game
from predictor.model.player_ids import PlayerIndex
from predictor.model.player_ratings import PlayerBook
from predictor.model.player_settings import PlayerModelSettings
from predictor.model.player_sim import (
    CONTINUITY_GAMES,
    GameFeatures,
    PlayerLines,
    Rosters,
    TeamForm,
    _diff,
    _Linker,
    _replay,
    _total,
    call_terms,
    collect_listings,
    simulate,
)
from predictor.model.ratings import Result, win_probability
from predictor.model.settings import TEST_SEASONS, TUNING_SEASONS, ModelSettings
from predictor.model.stage1 import Stage1Predictor
from predictor.model.venues import VenueIndex

_CONTINUITY_KEY = {"final": "continuity_gap", "early": "continuity_gap_early"}
_REST = ("back_to_back", "third_in_four")
_TRAVEL = ("travel_per_1000km", "tz_per_hour")


@dataclass(frozen=True)
class PlayerBreakdown:
    game_id: str
    home_team: str
    away_team: str
    rating: float  # (1 - w) x team-rating gap
    roster: float
    home: float
    rest: float
    travel: float
    altitude: float
    injuries: float
    adjustment: float  # the correction; 0.0 without one
    spread: float
    p_home: float
    absent: tuple[tuple[str, str, str], ...]  # (team, name, status)

    def terms(self) -> tuple[tuple[str, float], ...]:
        out = (
            ("rating", self.rating),
            ("roster", self.roster),
            ("home", self.home),
            ("rest", self.rest),
            ("travel", self.travel),
            ("altitude", self.altitude),
            ("injuries", self.injuries),
        )
        if self.adjustment != 0.0:
            out += (("adjustment", self.adjustment),)
        return out

    def sentence(self) -> str:
        """One publishable line, on ``stage1.Breakdown.sentence`` rules: the
        shown total is the sum of the shown (rounded) terms, no term shows
        as -0.0, and the probability uses the exact spread. Absences follow
        the injuries term."""
        shown = [(name, round(value, 1) + 0.0) for name, value in self.terms()]
        total = round(sum(v for _, v in shown), 1) + 0.0
        parts = []
        for name, v in shown:
            part = f"{name} {v:+.1f}"
            if name == "injuries" and self.absent:
                listed = ", ".join(
                    f"{_surname(n)} {status.strip().lower()}" for _, n, status in self.absent
                )
                part += f" ({listed})"
            parts.append(part)
        if total > 0:
            outcome = f"{self.home_team} by {total:.1f}"
        elif total < 0:
            outcome = f"{self.away_team} by {-total:.1f}"
        else:
            outcome = "pick'em"
        return (
            f"{self.away_team} at {self.home_team}: {', '.join(parts)} -> {outcome} "
            f"({self.home_team} {self.p_home * 100:.0f}% to win)"
        )


def _surname(name: str) -> str:
    """"Jamal Murray" -> "Murray"; everything after the first name."""
    parts = name.strip().split(" ", 1)
    return parts[-1]


def _sum_terms(settings: PlayerModelSettings, row: Sequence[float], prefix: str) -> float:
    """Coefficient x feature, summed over the features named ``prefix*``."""
    names = feature_names(settings.formula)
    coef = settings.coefficient_tuple()
    return float(sum(
        c * v for name, c, v in zip(names, coef, row, strict=True) if name.startswith(prefix)
    ))


def _stage1_settings(s: PlayerModelSettings) -> ModelSettings:
    return ModelSettings(
        ratings=s.team,
        coefficients=Coefficients(**{n: s.coefficients[n] for n in FEATURE_NAMES}),
        sigma=s.sigma,
        half_life=s.half_life,
        tuning_games=s.tuning_games,
    )


@dataclass
class _State:
    """Player state rebuilt from one view, for games on one date."""

    book: PlayerBook
    rosters: Rosters
    form: TeamForm
    lines: PlayerLines
    history: InjuryHistory
    linker: _Linker


class PlayerStage1Predictor:
    def __init__(
        self, con, settings: PlayerModelSettings, venues: VenueIndex | None = None
    ) -> None:
        self.settings = settings
        self.venues = venues if venues is not None else VenueIndex.from_db(con)
        self._team = Stage1Predictor(con, _stage1_settings(settings), self.venues)
        self._tables = {
            call: AvailabilityTable(probs=dict(settings.availability[call]), samples={})
            for call in CALLS
        }
        self._coef = settings.coefficient_tuple()
        self._state_key: tuple[datetime, date] | None = None
        self._state: _State | None = None
        self.breakdowns: dict[tuple[str, str], PlayerBreakdown] = {}
        # (game_id, call) -> the correction's inputs (CORRECTION_FEATURES order)
        self.correction_inputs: dict[tuple[str, str], tuple[float, ...]] = {}
        self.correction: Correction | None = None
        if settings.correction is not None:
            rows = correction_rows(con, settings, self.venues)
            self.correction = Correction.train(
                [r for season in sorted(rows) for r in rows[season]],
                seasons=tuple(sorted(rows)),
            )

    # -- state -------------------------------------------------------------
    def _state_for(self, view, game_date: date) -> _State:
        key = (view.as_of, game_date)
        if self._state_key == key and self._state is not None:
            return self._state
        lines = PlayerLines.from_view(view, before=game_date)
        games = [
            _Game(
                Result(gid, season, gd, home, away, hp, ap, self._neutral(gid)),
                (0.0, 0.0, 0.0, 0.0, 0.0),  # unused by the replay
            )
            for gid, season, gd, home, away, hp, ap, _obs in self._team._fetch_finals(view)
            if gd < game_date
        ]
        book = PlayerBook(self.settings.player)
        form = TeamForm()

        def visit(g: _Game, _rosters: Rosters) -> None:
            # As in player_sim.simulate: a game's season is entered before
            # that day's rows are applied.
            book.enter_season(g.result.season)

        rosters = _replay(games, lines, visit, book, form.apply_day, allow_test_seasons=True)
        identities = view.table("player_games").project(
            "player_id, player_name, team, season"
        ).distinct().fetchall()
        state = _State(
            book=book,
            rosters=rosters,
            form=form,
            lines=lines,
            history=InjuryHistory.from_view(view),
            linker=_Linker(PlayerIndex.from_rows(identities)),
        )
        self._state_key, self._state = key, state
        return state

    def _neutral(self, game_id: str) -> bool:
        venue = self.venues.venue(game_id)
        return venue.is_neutral if venue else False

    def _features(self, game: GameToPredict) -> tuple[float, float, float, float, float]:
        """The five adjustment features, computed as ``fit._load`` does."""
        venue = self.venues.venue(game.game_id)
        city = venue.city if venue else None
        neutral = venue.is_neutral if venue else False
        return adj.feature_vector(
            adj.situation(self.venues, game.home_team, game.game_date, city),
            adj.situation(self.venues, game.away_team, game.game_date, city),
            adj.is_altitude_game(city, neutral),
        )

    # -- predicting ----------------------------------------------------------
    def explain(self, game: GameToPredict, view, call: str) -> PlayerBreakdown:
        if call not in CALLS:
            raise ValueError(f"call must be one of {CALLS}, got {call!r}")
        s = self.settings
        team = self._team.explain(game, view)
        state = self._state_for(view, game.game_date)
        state.book.enter_season(game.season)
        d = game.game_date
        snap = state.history.snapshot_at(d, view.as_of)
        terms = call_terms(
            (game.home_team, game.away_team), d, game.season, snap, state.linker,
            state.rosters, state.book, self._tables[call], state.lines.names,
        )
        x = self._features(game)
        cf = CallFeatures(terms.roster_gap, terms.injury_gap, terms.absent,
                          None if snap is None else snap.observed_at, False, snap is None)
        row = design_row(x, cf, s.formula)
        additive = spread(team.rating, team.home, row, self._coef, s.w)

        names = feature_names(s.formula)
        by_name = dict(zip(names, (c * v for c, v in zip(self._coef, row, strict=True)),
                           strict=True))
        roster = _sum_terms(s, row, "roster_")
        injuries = _sum_terms(s, row, "injury_")

        form = state.form
        extras = {
            "recent_margin_gap": _diff(form.margin(game.home_team, as_of=d),
                                       form.margin(game.away_team, as_of=d)),
            "pace_sum": _total(form.pace(game.home_team, as_of=d),
                               form.pace(game.away_team, as_of=d)),
            _CONTINUITY_KEY[call]: _diff(*terms.continuity),
        }
        parts = GameParts(
            additive_spread=float(additive), team_gap=float(team.rating),
            home_court=float(team.home), x=tuple(float(v) for v in x),
            roster=roster, injury=injuries,
        )
        inputs = feature_row(parts, GameFeatures(game.game_id, cf, cf, extras), call)
        self.correction_inputs[(game.game_id, call)] = inputs
        adjustment = 0.0 if self.correction is None else self.correction.predict(inputs)
        total = float(additive) + adjustment

        breakdown = PlayerBreakdown(
            game_id=game.game_id,
            home_team=game.home_team,
            away_team=game.away_team,
            rating=(1.0 - s.w) * team.rating,
            roster=roster,
            home=team.home,
            rest=sum(by_name[n] for n in _REST),
            travel=sum(by_name[n] for n in _TRAVEL),
            altitude=by_name["altitude"],
            injuries=injuries,
            adjustment=adjustment,
            spread=total,
            p_home=win_probability(total, s.sigma),
            absent=terms.absent,
        )
        self.breakdowns[(game.game_id, call)] = breakdown
        return breakdown


# -- the shipped correction ---------------------------------------------------

def correction_rows(
    con, settings: PlayerModelSettings, venues: VenueIndex | None = None
) -> dict[str, list[tuple[tuple[float, ...], float]]]:
    """Final-call (correction inputs, residual) rows of the shipped additive
    settings, per training season, for the ``002`` games of
    ``settings.correction["training_seasons"]``.

    Read straight from the database like ``fit._load`` (a training path;
    test seasons are never loaded), replayed with ``player_sim.simulate`` and
    ``fit._simulate``. Deterministic for a given database.
    """
    if settings.correction is None:
        raise ValueError("these settings have no correction")
    seasons = set(settings.correction["training_seasons"])
    if seasons & set(TEST_SEASONS):
        raise FitError(
            "internal error: the correction lists a test season to train on; refusing"
        )
    venues = venues if venues is not None else VenueIndex.from_db(con)
    games = fit._load(con, venues)
    lines = PlayerLines.from_db(con)
    history = InjuryHistory.from_db(con)
    index = load_player_index(con)
    listings = collect_listings(con, games, history, index)
    feats, _ = simulate(games, lines, history, index, settings.player, listings)
    pre = fit._simulate(settings.team, games)
    coef = settings.coefficient_tuple()
    out: dict[str, list[tuple[tuple[float, ...], float]]] = defaultdict(list)
    for g, (gap, hc) in zip(games, pre, strict=True):
        r = g.result
        if r.season not in seasons or not r.game_id.startswith("002"):
            continue
        gf = feats[r.game_id]
        row = design_row(g.x, gf.final, settings.formula)
        s = spread(gap, hc, row, coef, settings.w)
        parts = GameParts(
            additive_spread=float(s), team_gap=float(gap), home_court=float(hc),
            x=tuple(float(v) for v in g.x),
            roster=_sum_terms(settings, row, "roster_"),
            injury=_sum_terms(settings, row, "injury_"),
        )
        out[r.season].append(
            (feature_row(parts, gf, "final"), float(r.home_points - r.away_points) - s)
        )
    return dict(out)


def correction_sigma(rows: dict[str, list[tuple[tuple[float, ...], float]]]) -> float:
    """Sigma for a shipped correction, chosen as ``apply_corrections`` does:
    each training season is corrected by a model trained on the others, and
    ``best_sigma`` (equal weights) runs on those out-of-fold spreads."""
    seasons = sorted(rows)
    spreads: list[float] = []
    won: list[bool] = []
    for e in seasons:
        others = [s for s in seasons if s != e]
        model = Correction.train([r for s in others for r in rows[s]], seasons=tuple(others))
        held = rows[e]
        delta = model.predict_many([r for r, _ in held])
        for (row, residual), dlt in zip(held, delta, strict=True):
            spreads.append(float(row[0] + dlt))
            won.append(bool(row[0] + residual > 0))
    return best_sigma(spreads, won, np.ones(len(spreads)))


def settings_from_candidate(
    con,
    cev: CandidateEvaluation,
    corrected: CandidateResult | None,
    venues: VenueIndex | None = None,
) -> PlayerModelSettings:
    """The settings file for the candidate's final, all-tuning-seasons choice.

    Availability is estimated on every tuning season's listings. When the
    correction was included, it trains on the walk-forward seasons, and
    sigma is re-chosen out of fold on them (``correction_sigma``).
    """
    final = cev.final
    tuning = [lst for lst in cev.listings if lst.season in TUNING_SEASONS]
    availability = {
        call: dict(estimate(tuning, TEST_SEASONS[0], call).probs) for call in CALLS
    }
    settings = PlayerModelSettings(
        team=final.team,
        player=final.player,
        formula=final.formula,
        w=final.w,
        coefficients=dict(zip(final.feature_names, final.coefficients, strict=True)),
        sigma=final.sigma,
        half_life=cev.winner.half_life,
        availability=availability,
        correction=None,
        tuning_games=final.games,
    )
    if corrected is None:
        return settings
    seasons = sorted({p.season for p in cev.winner.final_call.predictions})
    settings = dataclasses.replace(
        settings, correction={"training_seasons": seasons, "params": dict(GBM_PARAMS)}
    )
    rows = correction_rows(con, settings, venues)
    return dataclasses.replace(settings, sigma=correction_sigma(rows))
