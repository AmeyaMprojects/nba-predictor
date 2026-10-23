"""Small player-log and injury-report fixtures for the player simulation.

Builds on ``model_fixtures``: games get a FINAL row stamped D+1 12:00 UTC and
a schedule row with tip-off at D+1 00:00 UTC, so a game on date D has its
final call at D 23:15 UTC and its early call at D 12:30 UTC.

Two teams of eight players each, every player 30 minutes a game:

- BOS: ids 1..8 (player 1 "Al Adams" is the star in the tests)
- NYK: ids 11..18
- MIA: ids 21..28 (only used where a third team is needed)
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta

from model_fixtures import add_game
from predictor import db
from predictor.model.player_ids import PlayerIndex

SEASON = "2024-25"

ROSTERS: dict[str, dict[int, str]] = {
    "BOS": {
        1: "Al Adams", 2: "Bo Baker", 3: "Cy Cole", 4: "Di Dunn",
        5: "Ed Eves", 6: "Fa Ford", 7: "Gu Gray", 8: "Ha Hill",
    },
    "NYK": {
        11: "Ia Ivey", 12: "Jo Jones", 13: "Ka Kent", 14: "Lu Lowe",
        15: "Mo Mann", 16: "Ny Nash", 17: "Ox Orr", 18: "Pi Pike",
    },
    "MIA": {
        21: "Qu Quinn", 22: "Ro Ross", 23: "Su Sims", 24: "Ty Tate",
        25: "Ub Upton", 26: "Vi Vance", 27: "Wu West", 28: "Xa Young",
    },
}

G1, G2, G3 = "0022400001", "0022400002", "0022400003"
D1, D2, D3 = date(2024, 11, 1), date(2024, 11, 3), date(2024, 11, 5)


def report_name(name: str) -> str:
    """"Al Adams" -> "Adams,Al" (the injury report's layout)."""
    first, last = name.split(" ", 1)
    return f"{last},{first}"


def final_at(game_date: date) -> datetime:
    return datetime.combine(game_date, time(23, 15), UTC)


def early_at(game_date: date) -> datetime:
    return datetime.combine(game_date, time(12, 30), UTC)


def add_player_row(
    con,
    game_id: str,
    game_date: date,
    player_id: int,
    team: str,
    minutes: float | None = 30.0,
    *,
    season: str = SEASON,
    season_type: str = "regular",
    name: str | None = None,
    pts: int = 15,
    plus_minus: float | None = 0.0,
    observed_at: datetime | None = None,
) -> None:
    table = db.POINT_IN_TIME_TABLES["player_games"]
    if name is None:
        name = next(r[player_id] for r in ROSTERS.values() if player_id in r)
    if observed_at is None:
        observed_at = datetime.combine(game_date + timedelta(days=1), time(12), UTC)
    con.execute(
        f"INSERT OR REPLACE INTO {table} (game_id, season, game_date, season_type,"
        " player_id, player_name, team, minutes, pts, fgm, fga, fg3m, fg3a, ftm, fta,"
        " oreb, dreb, ast, stl, blk, tov, pf, plus_minus, source, reconstructed,"
        " observed_at) VALUES (?,?,?,?,?,?,?,?,?,6,12,1,3,2,2,1,4,3,1,0,2,2,?,"
        " 'nba_stats', TRUE, ?)",
        [game_id, season, game_date, season_type, player_id, name, team, minutes,
         pts, plus_minus, observed_at],
    )


def add_team_rows(con, game_id, game_date, team, *, season=SEASON,
                  season_type="regular", minutes=30.0, skip=()):
    for pid in ROSTERS[team]:
        if pid in skip:
            continue
        add_player_row(con, game_id, game_date, pid, team, minutes,
                       season=season, season_type=season_type)


def add_report(con, observed_at, game_date, team, player, status):
    table = db.POINT_IN_TIME_TABLES["injury_status"]
    con.execute(
        f"INSERT INTO {table} (report_date, game_date, team, player, status,"
        " observed_at) VALUES (?,?,?,?,?,?)",
        [observed_at.date(), game_date, team, player, status, observed_at],
    )


def build_player_fixture(con, *, target_rows=True):
    """G1 (D1, BOS home), G2 (D2, NYK home) and the target G3 (D3, BOS home),
    all BOS vs NYK, every player 30 minutes. G3's own rows are added unless
    ``target_rows`` is False."""
    add_game(con, G1, SEASON, D1, "BOS", "NYK", 110, 100)
    add_game(con, G2, SEASON, D2, "NYK", "BOS", 100, 104)
    add_game(con, G3, SEASON, D3, "BOS", "NYK", 105, 101)
    for gid, d in ((G1, D1), (G2, D2)) + (((G3, D3),) if target_rows else ()):
        add_team_rows(con, gid, d, "BOS")
        add_team_rows(con, gid, d, "NYK")


def player_index(con) -> PlayerIndex:
    table = db.POINT_IN_TIME_TABLES["player_games"]
    rows = con.execute(
        f"SELECT DISTINCT player_id, player_name, team, season FROM {table}"
    ).fetchall()
    return PlayerIndex.from_rows(rows)
