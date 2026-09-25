from __future__ import annotations

from collections.abc import Mapping
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
    skipped_conflicting_metadata: int
    # FIX 7 (final review, part 2): a game whose computed cutoff falls before
    # its own earliest SCHEDULED observation -- a sanity bound against an
    # unbounded buffer. FIX 21 (final review, part 3): that SCHEDULED
    # observation is itself RECONSTRUCTED from game_date, not observed (see
    # the comment where this is checked in `replay()`), so this counts
    # games where the run is not measuring anything meaningful, not
    # literally games "asked of before they were on the schedule".
    skipped_buffer_too_early: int
    # FIX 25(b) (final review, part 4): of `skipped_buffer_too_early` above,
    # how many were skipped against a RECONSTRUCTED (not observed) schedule
    # timestamp -- read off the `reconstructed` flag on the actual row
    # selected for each game, not assumed. Lets the report state plainly
    # whether the sentence above still applies to every one of these games,
    # some of them, or none -- rather than hardcoding it as true for all of
    # them regardless of what the data says.
    skipped_buffer_too_early_reconstructed: int
    skipped_no_tipoff: int
    # FIX 11: per-season breakdown of the two counters above, keyed by
    # `Prediction.season`. `considered_by_season` is the denominator for
    # EVERY season (every row considered, regardless of outcome);
    # `skipped_no_tipoff_by_season` is the numerator for the no-tip-off
    # line specifically -- together they let the report show "<n> of
    # <season total>" per season instead of one pooled count that hides a
    # skew concentrated in one or two seasons.
    considered_by_season: Mapping[str, int]
    skipped_no_tipoff_by_season: Mapping[str, int]
    # FIX 9: split in two. `skipped_no_result` is now ONLY games with no
    # FINAL row at all (genuinely not yet played). `skipped_score_missing`
    # is games that DO have a FINAL row but whose score is NULL (played,
    # but the archive failed to record the score) -- a materially
    # different situation that needs a different message and a different
    # fix (re-ingest, not "wait for the game to be played").
    skipped_no_result: int
    skipped_score_missing: int
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

    # FIX 5 (final review, part 1): the primary key is (game_id,
    # observed_at), not game_id alone, so a re-ingest that corrects a
    # game's game_date or home/away assignment leaves the OLD row sitting
    # alongside the NEW one instead of replacing it -- a plain `SELECT
    # DISTINCT` over all five columns would then yield two rows for the
    # same game_id and silently predict (and score) it twice, one copy
    # with the wrong metadata. Grouping by game_id alone and requiring
    # every other column to be unambiguous (exactly one DISTINCT value
    # each) makes that structurally impossible: a game_id whose rows
    # disagree fails the HAVING-equivalent check below and is routed to
    # `skipped_conflicting_metadata` instead of being treated as one game.
    # MIN() is used (not ANY_VALUE/first) purely for determinism; when a
    # group is unambiguous every row agrees, so MIN() and "the" value are
    # the same thing.
    #
    # FIX 19 (final review, part 3): `clause` (which includes the
    # `--season` predicate) used to filter the rows BEFORE this GROUP BY,
    # so a game whose rows carry two different season labels could pass
    # the uniqueness check under `--season`: filtering to just the
    # matching-season row(s) first hid the OTHER, conflicting row from the
    # aggregate entirely, and the guard this whole query exists for never
    # saw the conflict. Reproduced: such a game is correctly skipped in a
    # full run but silently scored under `--season` -- the exact
    # double-scoring the guard exists to prevent.
    #
    # The subquery below decides WHICH game ids to include (every game
    # when there is no `--season`, or every game with AT LEAST ONE row
    # matching it when there is); the outer query then groups over EVERY
    # row for those game ids, conflicting rows included, so a game whose
    # rows disagree on season is caught and skipped identically whether or
    # not `--season` is passed.
    rows = con.execute(
        f"SELECT game_id, MIN(season) AS season, MIN(game_date) AS game_date, "
        f"MIN(home_team) AS home_team, MIN(away_team) AS away_team, "
        f"count(DISTINCT season) AS n_season, count(DISTINCT game_date) AS n_game_date, "
        f"count(DISTINCT home_team) AS n_home_team, count(DISTINCT away_team) AS n_away_team "
        f"FROM {games_table} WHERE game_id LIKE '002%' AND game_id IN "
        f"(SELECT game_id FROM {games_table} WHERE {clause}) "
        f"GROUP BY game_id ORDER BY game_date, game_id",
        params,
    ).fetchall()

    considered = predicted = conflicting = no_tip = no_result = leaked = declined = failed = 0
    buffer_too_early = score_missing = buffer_too_early_reconstructed = 0
    considered_by_season: dict[str, int] = {}
    no_tip_by_season: dict[str, int] = {}
    out: list[Prediction] = []

    for (
        game_id, game_season, game_date, home_team, away_team,
        n_season, n_game_date, n_home_team, n_away_team,
    ) in rows:
        if limit is not None and predicted >= limit:
            break
        considered += 1
        considered_by_season[game_season] = considered_by_season.get(game_season, 0) + 1

        if not (n_season == 1 and n_game_date == 1 and n_home_team == 1 and n_away_team == 1):
            conflicting += 1
            print(
                f"backtest: SKIPPING {game_id} -- the archive holds contradictory "
                "rows for this game (season/date/home/away disagree across "
                "ingested rows), not predicted"
            )
            continue

        tip = tipoff_mod.resolve_tipoff(index, game_date, home_team, away_team)
        if tip is None:
            no_tip += 1
            no_tip_by_season[game_season] = no_tip_by_season.get(game_season, 0) + 1
            continue

        cutoff = tip - timedelta(minutes=buffer_minutes)

        # FIX 7 (final review, part 2): an unbounded buffer (e.g.
        # --buffer-minutes 100000, ~69 days) can push the cutoff before the
        # game's own SCHEDULED observation -- a useful sanity bound on how
        # far back the buffer reaches.
        #
        # FIX 21 (final review, part 3): what this bound actually checks.
        # `earliest_scheduled` here is not necessarily an observed
        # publication time -- for every game in the archive TODAY, this
        # table's SCHEDULED row is DERIVED at ingest time from `game_date`
        # alone (7 days before for the regular season, 1 day before for the
        # postseason, both at 12:00 UTC -- see
        # `nba_stats._derive_observed_at`), because the NBA archive does not
        # record when a game was actually first announced. So "the buffer
        # reaches back before the game was even scheduled" was never a true
        # statement about this data -- it compared the cutoff to a
        # RECONSTRUCTED timestamp, not an observed one (measured: at
        # --buffer-minutes 14400, all 7,200 scored games print this -- the
        # 1,229 first recorded here were one season alone -- none of which
        # were genuinely unscheduled at that cutoff). The guard is still
        # worth keeping as a sanity bound; only the message is corrected to
        # say what it actually checks.
        #
        # FIX 25(b) (final review, part 4): that correction used to be
        # HARDCODED into the message regardless of what the row actually
        # says -- the same anti-pattern FIX 10 removed from the market
        # line. `reconstructed` is read here off the SAME row already being
        # selected (the earliest SCHEDULED observation for this game), so
        # the message stays true the moment a live ingest path writes a
        # genuinely OBSERVED SCHEDULED row instead of a derived one.
        earliest_scheduled_row = con.execute(
            f"SELECT observed_at, reconstructed FROM {games_table} "
            "WHERE game_id = ? AND status = 'SCHEDULED' "
            "ORDER BY observed_at ASC LIMIT 1",
            [game_id],
        ).fetchone()
        earliest_scheduled = earliest_scheduled_row[0] if earliest_scheduled_row else None
        if earliest_scheduled is not None and cutoff < earliest_scheduled:
            buffer_too_early += 1
            schedule_reconstructed = bool(earliest_scheduled_row[1])
            provenance = (
                "RECONSTRUCTED schedule timestamp (derived from game_date, "
                "not observed)"
                if schedule_reconstructed
                else "OBSERVED schedule timestamp"
            )
            print(
                f"backtest: SKIPPING {game_id} -- buffer reaches cutoff "
                f"{cutoff.isoformat()}, before this game's {provenance} "
                f"({earliest_scheduled.isoformat()}), so this run is not "
                "measuring anything meaningful for it, not predicted"
            )
            buffer_too_early_reconstructed += 1 if schedule_reconstructed else 0
            continue

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

        # FIX 9 (final review, part 2): "no scorable FINAL row" used to be a
        # single counter regardless of WHY, printed to the report as "not
        # yet played" -- true for a game with no FINAL row at all, but false
        # for a game that WAS played and has a FINAL row whose score is
        # NULL (an archive gap, not an unplayed game). Split into two real
        # checks so each is only ever reported under its true cause.
        final_exists = con.execute(
            f"SELECT 1 FROM {games_table} WHERE game_id = ? AND status = 'FINAL' LIMIT 1",
            [game_id],
        ).fetchone()
        if final_exists is None:
            no_result += 1
            continue

        result = con.execute(
            f"SELECT home_points, away_points, reconstructed FROM {games_table} "
            f"WHERE game_id = ? AND status = 'FINAL' "
            "AND home_points IS NOT NULL AND away_points IS NOT NULL "
            "ORDER BY observed_at DESC LIMIT 1",
            [game_id],
        ).fetchone()
        if result is None:
            score_missing += 1
            continue

        view = AsOfView(con, cutoff)
        game = GameToPredict(
            game_id=game_id,
            season=game_season,
            game_date=game_date,
            home_team=home_team,
            away_team=away_team,
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
        skipped_conflicting_metadata=conflicting,
        skipped_buffer_too_early=buffer_too_early,
        skipped_buffer_too_early_reconstructed=buffer_too_early_reconstructed,
        skipped_no_tipoff=no_tip,
        considered_by_season=considered_by_season,
        skipped_no_tipoff_by_season=no_tip_by_season,
        skipped_no_result=no_result,
        skipped_score_missing=score_missing,
        skipped_result_visible=leaked,
        declined=declined,
        failed=failed,
    )
    return out, stats


def known_seasons(con) -> list[str]:
    """Distinct regular-season labels present in the archive, sorted.

    Used only to help a user who mistyped `--season` (FIX 12(a)): when a
    season filter matches nothing at all, the report can list what IS
    actually there instead of printing a wall of zeros with no hint why.
    """
    games_table = db.POINT_IN_TIME_TABLES["games"]
    rows = con.execute(
        f"SELECT DISTINCT season FROM {games_table} WHERE game_id LIKE '002%' "
        "ORDER BY season"
    ).fetchall()
    return [r[0] for r in rows]
