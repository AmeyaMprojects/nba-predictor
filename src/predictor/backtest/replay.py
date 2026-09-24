from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from predictor import db
from predictor.asof import AsOfView
from predictor.backtest import tipoff as tipoff_mod
from predictor.backtest.baselines import GameToPredict, Predictor

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


@dataclass(frozen=True)
class ReplayStats:
    considered: int
    predicted: int
    skipped_no_tipoff: int
    skipped_no_result: int
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
    """
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

    considered = predicted = no_tip = no_result = failed = 0
    out: list[Prediction] = []

    for game_id, game_season, game_date, home_team, away_team in rows:
        considered += 1
        if limit is not None and predicted >= limit:
            break

        tip = tipoff_mod.resolve_tipoff(index, game_date, home_team, away_team)
        if tip is None:
            no_tip += 1
            continue

        result = con.execute(
            f"SELECT home_points, away_points FROM {games_table} "
            f"WHERE game_id = ? AND status = 'FINAL' "
            "AND home_points IS NOT NULL ORDER BY observed_at DESC LIMIT 1",
            [game_id],
        ).fetchone()
        if result is None:
            no_result += 1
            continue

        cutoff = tip - timedelta(minutes=buffer_minutes)
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
            )
        )
        predicted += 1

    return out, ReplayStats(considered, predicted, no_tip, no_result, failed)
