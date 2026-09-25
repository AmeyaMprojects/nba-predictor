from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from predictor import db
from predictor.asof import AsOfView
from predictor.backtest import tipoff as tipoff_mod
from predictor.backtest.baselines import GameToPredict, PredictionError, Predictor

DEFAULT_BUFFER_MINUTES = 30


@dataclass(frozen=True)
class Prediction:
    game_id: str
    season: str
    game_date: date
    home_team: str
    away_team: str
    tipoff: datetime
    cutoff: datetime
    p_home: float
    home_won: bool
    reconstructed: bool


@dataclass(frozen=True)
class ReplayStats:
    considered: int
    predicted: int
    skipped_no_tipoff: int
    skipped_no_result: int
    skipped_result_visible: int
    declined: int
    failed: int


def replay(
    con,
    predictor: Predictor,
    season: str | None = None,
    buffer_minutes: int = DEFAULT_BUFFER_MINUTES,
    limit: int | None = None,
) -> tuple[list[Prediction], ReplayStats]:
    """Replay games chronologically, predicting each from a pre-tipoff view.

    The predictor is handed an AsOfView cut at tip-off minus a buffer and a
    GameToPredict that carries no score. It cannot see the result through the
    harness; the adversarial tests prove it cannot see it around the harness
    either.

    This module reads the games point-in-time table directly, via
    ``db.POINT_IN_TIME_TABLES`` rather than through ``AsOfView`` -- the same
    style of deliberate exemption ``tipoff.py`` documents for its own read.
    It is used for exactly two things, both outcome-only and neither ever
    exposed to the predictor:

    1. Enumerating which games exist (id/season/date/teams) to build the
       schedule this function walks. This carries no score.
    2. After a game's cutoff is computed, reading that game's OWN official
       result -- first to check whether it is already visible at the
       cutoff (the leak guard below), and, only once that check passes, to
       populate ``Prediction.home_won`` so the prediction can be SCORED
       after the fact.

    Neither read can leak future information into a model: the raw result
    is never placed into the ``AsOfView`` handed to the predictor and never
    carried by ``GameToPredict``. The predictor only ever sees data through
    ``AsOfView``, filtered to ``observed_at <= cutoff``.

    Leak guard -- the harness's central invariant: before a game is handed
    to the predictor, this function checks, against the real
    ``observed_at`` timestamps and with the same inclusive ``<=``
    comparison ``AsOfView`` itself uses, whether the game's FINAL result is
    already visible at the cutoff. If it is, the game is NOT predicted: it
    is counted in ``skipped_result_visible`` and logged with its game id.
    This is a live check run against the data for every single game, not
    an assertion that some timing margin holds -- a future change to how
    tip-off (and therefore the cutoff) is derived must trip this check,
    not silently slip past it.

    FIX 2 (final review, part 1): what this guard can and cannot prove on
    the archive in use today. Every row of the games point-in-time table
    currently in the database has ``reconstructed = TRUE`` -- these
    ``observed_at`` timestamps were derived at import time (see
    ``nba_stats._derive_observed_at``), not observed as results were
    published -- and every derived FINAL row sits at exactly
    ``game_date + 1 day, 12:00 UTC``, which is always later than any
    possible pre-tip-off cutoff. That makes this guard STRUCTURALLY
    INCAPABLE of ever tripping on this data: ``skipped_result_visible``
    being 0 for every game replayed so far is a consequence of how these
    timestamps were derived, not evidence that the timing was verified.
    The guard itself is correct, and it is not dead code -- it defends a
    live ingestion path (results captured as they actually arrive, each
    carrying a true publish timestamp) that does not exist yet, and it
    will start doing real work the moment that path does. Until then, do
    not read a clean run of this harness as proof of point-in-time safety
    for the *timing* of results specifically -- ``report.format_report``'s
    provenance block says this to the report's reader in plain English;
    this paragraph says it to the next engineer reading this code.
    """
    if buffer_minutes < 0:
        raise ValueError(f"buffer_minutes must be >= 0, got {buffer_minutes}")

    games_table = db.POINT_IN_TIME_TABLES["games"]
    index = tipoff_mod.tipoff_index(con)

    where = ["game_id LIKE '002%'"]
    params: list = []
    if season is not None:
        where.append("season = ?")
        params.append(season)
    clause = " AND ".join(where)

    rows = con.execute(
        f"SELECT DISTINCT game_id, season, game_date, home_team, away_team "
        f"FROM {games_table} WHERE {clause} ORDER BY game_date, game_id",
        params,
    ).fetchall()

    considered = predicted = no_tip = no_result = leaked = declined = failed = 0
    out: list[Prediction] = []

    for game_id, game_season, game_date, home_team, away_team in rows:
        if limit is not None and predicted >= limit:
            break
        considered += 1

        tip = tipoff_mod.resolve_tipoff(index, game_date, home_team, away_team)
        if tip is None:
            no_tip += 1
            continue

        cutoff = tip - timedelta(minutes=buffer_minutes)

        # Finding 1: the central invariant, verified for real against the
        # data on every game -- not inferred from how large today's margin
        # between cutoff and FINAL happens to be. A FINAL row observed at
        # or before the cutoff means the view about to be built for the
        # predictor would already contain this game's own outcome.
        already_visible = con.execute(
            f"SELECT 1 FROM {games_table} WHERE game_id = ? AND status = 'FINAL' "
            "AND observed_at <= ? LIMIT 1",
            [game_id, cutoff],
        ).fetchone()
        if already_visible is not None:
            leaked += 1
            print(
                f"backtest: SKIPPING {game_id} -- result already visible at "
                f"cutoff {cutoff.isoformat()} (leak guard tripped, not predicted)"
            )
            continue

        result = con.execute(
            f"SELECT home_points, away_points, reconstructed FROM {games_table} "
            f"WHERE game_id = ? AND status = 'FINAL' "
            "AND home_points IS NOT NULL AND away_points IS NOT NULL "
            "ORDER BY observed_at DESC LIMIT 1",
            [game_id],
        ).fetchone()
        if result is None:
            no_result += 1
            continue

        view = AsOfView(con, cutoff)
        game = GameToPredict(
            game_id=game_id,
            season=game_season,
            game_date=game_date,
            home_team=home_team,
            away_team=away_team,
            tipoff=tip,
        )

        try:
            p_home = float(predictor(game, view))
        except PredictionError as exc:
            # A predictor deliberately declining to predict is not the same
            # kind of event as a bug in it -- keep it out of `failed`. See
            # Finding 6 in task-3-report.md for why this is handled
            # separately from the bare Exception catch below.
            declined += 1
            print(f"backtest: predictor DECLINED {game_id} -- {exc}")
            continue
        except Exception as exc:  # one bad game must not abort a season
            failed += 1
            print(f"backtest: predictor FAILED on {game_id} -- {exc}")
            continue

        if not 0.0 <= p_home <= 1.0:
            failed += 1
            print(
                f"backtest: predictor returned an impossible probability "
                f"{p_home} for {game_id} -- not scored"
            )
            continue

        out.append(
            Prediction(
                game_id=game_id,
                season=game_season,
                game_date=game_date,
                home_team=home_team,
                away_team=away_team,
                tipoff=tip,
                cutoff=cutoff,
                p_home=p_home,
                home_won=result[0] > result[1],
                reconstructed=bool(result[2]),
            )
        )
        predicted += 1

    stats = ReplayStats(
        considered=considered,
        predicted=predicted,
        skipped_no_tipoff=no_tip,
        skipped_no_result=no_result,
        skipped_result_visible=leaked,
        declined=declined,
        failed=failed,
    )
    return out, stats
