from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
from tenacity import retry, stop_after_attempt, wait_exponential

from predictor import db


@dataclass(frozen=True)
class GameRow:
    game_id: str
    season: str
    game_date: date
    home_team: str
    away_team: str
    home_points: int | None
    away_points: int | None
    status: str


@dataclass(frozen=True)
class DroppedGame:
    """A GAME_ID group that could not be resolved into a single GameRow.

    Recorded (and always printed) rather than silently discarded -- see
    the module docstring note on neutral-site games below. Callers that
    want the list (rather than just the printed log) pass a mutable list
    via `pair_team_rows(..., dropped=...)`.
    """

    game_id: str
    matchup: str
    reason: str


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=30))
def fetch_season(season: str) -> pd.DataFrame:
    """Fetch every team-game row for a season. Two rows per game."""
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season, league_id_nullable="00", timeout=60
    )
    return finder.get_data_frames()[0]


def _as_int(value) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _parse_matchup(matchup: str) -> tuple[str, str] | None:
    """Parse a MATCHUP string into (home_abbr, away_abbr), or None.

    nba_api emits two textual forms:
      "HOME vs. AWAY"  -- the home team's own row.
      "AWAY @ HOME"    -- the away team's row.

    Critically, for a NEUTRAL-SITE game (NBA Cup semifinals/final, Paris/
    London games, some neutral-site preseason) BOTH rows carry the exact
    same "AWAY @ HOME" text -- neither row uses "vs.", because neither
    team is truly the home team. The "@" form alone still fully encodes
    the pairing (and which side the schedule nominally calls "home"), so
    it is parsed the same way regardless of which row it came from. This
    is what lets a neutral-site group resolve correctly instead of being
    dropped by a rule that requires seeing a "vs." row (the bug found in
    task-10 review: 9 real, played neutral-site games -- including an NBA
    Cup final -- were silently dropped every ingest).
    """
    if not isinstance(matchup, str):
        return None
    if " vs. " in matchup:
        home, away = matchup.split(" vs. ", 1)
        return home.strip(), away.strip()
    if " @ " in matchup:
        away, home = matchup.split(" @ ", 1)
        return home.strip(), away.strip()
    return None


def _log_dropped(
    game_id: str,
    matchup: str,
    reason: str,
    dropped: list[DroppedGame] | None,
) -> None:
    # "Never silently lose data -- surface loudly": every group that fails
    # to become a GameRow is printed here, unconditionally, whether or not
    # a caller is also collecting the structured list.
    print(f"nba_stats: DROPPED game_id={game_id} matchup=[{matchup}] -- {reason}")
    if dropped is not None:
        dropped.append(DroppedGame(game_id=game_id, matchup=matchup, reason=reason))


def pair_team_rows(
    df: pd.DataFrame,
    season: str = "",
    dropped: list[DroppedGame] | None = None,
) -> list[GameRow]:
    games: list[GameRow] = []
    for game_id, group in df.groupby("GAME_ID"):
        game_id = str(game_id)
        matchup_text = ", ".join(str(m) for m in group["MATCHUP"])

        if len(group) != 2:
            _log_dropped(
                game_id, matchup_text, f"expected 2 team-rows, got {len(group)}", dropped
            )
            continue

        parsed = [_parse_matchup(m) for m in group["MATCHUP"]]
        if any(p is None for p in parsed):
            _log_dropped(
                game_id, matchup_text, "could not parse MATCHUP into home/away teams", dropped
            )
            continue
        if parsed[0] != parsed[1]:
            _log_dropped(
                game_id,
                matchup_text,
                f"rows disagree on home/away: {parsed[0]} vs {parsed[1]}",
                dropped,
            )
            continue
        home_abbr, away_abbr = parsed[0]

        home_rows = group[group["TEAM_ABBREVIATION"] == home_abbr]
        away_rows = group[group["TEAM_ABBREVIATION"] == away_abbr]
        if len(home_rows) != 1 or len(away_rows) != 1:
            _log_dropped(
                game_id,
                matchup_text,
                f"TEAM_ABBREVIATION did not match parsed home={home_abbr!r} "
                f"away={away_abbr!r}",
                dropped,
            )
            continue
        home = home_rows.iloc[0]
        away = away_rows.iloc[0]

        home_points = _as_int(home["PTS"])
        away_points = _as_int(away["PTS"])
        played = home_points is not None and away_points is not None

        games.append(
            GameRow(
                game_id=game_id,
                season=season,
                game_date=pd.to_datetime(home["GAME_DATE"]).date(),
                home_team=str(home["TEAM_ABBREVIATION"]),
                away_team=str(away["TEAM_ABBREVIATION"]),
                home_points=home_points,
                away_points=away_points,
                status="FINAL" if played else "SCHEDULED",
            )
        )
    return games


# How long before tip-off an NBA schedule is knowable. NBA schedules are
# published months ahead of the REGULAR season, so 7 days is very
# conservative there -- see the safety rule in _derive_observed_at below.
# NOT safe for the postseason: a postseason/play-in/Cup-knockout fixture
# only exists once the teams that will play in it are actually determined
# (prior rounds/games resolve), so stamping it 7 days early back-dates its
# existence to before it was knowable. Measured against the real archive:
# 485 of 630 postseason/play-in/Cup games had a SCHEDULED row predating
# results of earlier games in the same series, and 180 of 630 were stamped
# before a participant had even finished its prior series -- fixture
# EXISTENCE itself encodes who advanced, and enumerating "upcoming games as
# of time T" is exactly how a dated prediction gets generated. A 1-day lead
# is still conservative (brackets/pairings are set immediately once the
# preceding round ends) without reaching back before that information
# existed.
_SCHEDULED_LEAD_DAYS = 7
_SCHEDULED_LEAD_DAYS_POSTSEASON = 1

# GAME_ID prefixes that mean "postseason, play-in, or in-season (Cup)
# knockout" per the NBA's own game_id numbering convention -- see
# https://github.com/swar/nba_api's documented ID scheme: 001=preseason,
# 002=regular season, 003=all-star, 004=playoffs, 005=play-in, 006=NBA Cup
# knockout rounds (the Cup's regular-season "group play" games use the
# normal 002 prefix and are NOT included here -- pairings for those are
# known from the regular-season schedule just like any other regular-season
# game; only the winner-takes-the-single-elimination-round knockout stage
# has the "only knowable once a prior round resolves" problem).
_POSTSEASON_GAME_ID_PREFIXES = ("004", "005", "006")

# How long after GAME_DATE the final score is knowable. LeagueGameFinder
# gives only a calendar date, not a tip-off time, so this cannot be derived
# from an actual final-buzzer time -- it is deliberately set past even the
# latest West-Coast game's finish. 08:00 UTC (~3-4am Eastern) leaves only
# ~2h of margin after the latest possible tip (10:30pm ET / 03:30 UTC;
# regulation alone ends ~06:00 UTC) -- a multi-overtime game plus any
# broadcast delay could reach it. 12:00 UTC costs nothing (FINAL rows are
# already "well after the fact", never on any hot path) and removes that
# whole class of near-miss.
_FINAL_LAG_DAYS = 1
_FINAL_LAG_HOUR_UTC = 12


def _is_postseason_game_id(game_id: str) -> bool:
    return game_id[:3] in _POSTSEASON_GAME_ID_PREFIXES


def _derive_observed_at(game_date: date, game_id: str = "") -> tuple[datetime, datetime]:
    """Derive (scheduled_observed_at, final_observed_at) for a historical game.

    Governing safety rule: a derived observed_at must NEVER be earlier than
    the moment the fact truly became knowable. Too late is merely
    conservative; too early is a leak, and a leak silently invalidates
    every backtest number built on top of it. Both offsets below are
    chosen deliberately on the late/conservative side:

    - SCHEDULED (fixture only -- teams, date, NULL scores): for the
      regular season, `game_date - 7 days` is a very safe lower bound
      (schedules publish months ahead). For a postseason/play-in/Cup-
      knockout game_id (see `_is_postseason_game_id`), only 1 day is used
      instead -- see the module-level note by `_SCHEDULED_LEAD_DAYS` for
      why 7 is unsafe there.
    - FINAL (the result): set to `game_date + 1 day at 12:00 UTC`, well
      after even the latest West-Coast tip-off's game has concluded.
    """
    lead_days = (
        _SCHEDULED_LEAD_DAYS_POSTSEASON
        if _is_postseason_game_id(game_id)
        else _SCHEDULED_LEAD_DAYS
    )
    scheduled_at = datetime.combine(
        game_date - timedelta(days=lead_days), time(12, 0), tzinfo=UTC
    )
    final_at = datetime.combine(
        game_date + timedelta(days=_FINAL_LAG_DAYS), time(_FINAL_LAG_HOUR_UTC, 0), tzinfo=UTC
    )
    return scheduled_at, final_at


def ingest_season(
    con,
    season: str,
    observed_at: datetime | None = None,
    dropped: list[DroppedGame] | None = None,
) -> int:
    """Fetch and ingest one season; returns the count of games ingested.

    `dropped`, mirroring `pair_team_rows`, is an optional caller-supplied
    list that gets populated with every GAME_ID group that could not be
    paired into a GameRow (and therefore was NOT written to the database).
    Passing it is how a caller -- notably `ingest_season_cmd` -- learns
    whether a season silently lost games, rather than relying solely on
    the printed log lines below. If the caller does not pass one, a local
    list is used instead so the summary print below still fires; either
    way, nothing dropped goes unreported.

    `observed_at`:

    - If supplied explicitly (the pinned-test path), it is used verbatim
      for every game, exactly as before this function grew historical-
      backfill support -- one row per game, `reconstructed=False` (the
      caller vouches for the timestamp being real, not derived).
    - If omitted (the real ingestion path, used by the CLI), observed_at is
      DERIVED per game from `game_date` via `_derive_observed_at` and TWO
      rows are written for a played game: a SCHEDULED row (NULL scores) at
      the derived pre-tipoff timestamp, and a FINAL row (real scores) at
      the derived post-game timestamp. An unplayed/future game gets only
      the SCHEDULED row. Both derived rows are stamped `reconstructed=True`
      so a backtest can report results with and without them -- these
      timestamps approximate when the fact became knowable, they were not
      observed at ingestion time the way a live injury report's publish
      timestamp is.
    """
    explicit_observed_at = observed_at is not None
    if explicit_observed_at:
        observed_at = db.require_utc(observed_at, "observed_at")
    local_dropped: list[DroppedGame] = dropped if dropped is not None else []
    games = pair_team_rows(fetch_season(season), season, dropped=local_dropped)
    # Resolved through db.POINT_IN_TIME_TABLES rather than spelled as a
    # literal here -- the physical "_raw" table names are only allowed to
    # appear as string literals in db.py/asof.py (see
    # test_no_physical_table_name_appears_outside_db_and_asof); this
    # ingestion module must not name the physical table directly either.
    # Mirrors the pattern already used by predictor.sources.injury_report.
    table = db.POINT_IN_TIME_TABLES["games"]
    insert_sql = (
        f"INSERT OR REPLACE INTO {table} (game_id, season, game_date, home_team,"
        " away_team, home_points, away_points, status, reconstructed, observed_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?)"
    )
    for game in games:
        if explicit_observed_at:
            con.execute(
                insert_sql,
                [
                    game.game_id,
                    season,
                    game.game_date,
                    game.home_team,
                    game.away_team,
                    game.home_points,
                    game.away_points,
                    game.status,
                    False,
                    observed_at,
                ],
            )
            continue

        scheduled_at, final_at = _derive_observed_at(game.game_date, game.game_id)
        # I2: this DERIVED path writes 100% of real games rows (the
        # explicit-observed_at branch above is only the pinned-test path),
        # so it is the write path `db.require_utc` actually needs to guard
        # -- unvalidated until now, correct only by accident (both derived
        # timestamps are already built with tzinfo=UTC, but nothing
        # enforced that). Mirrors the guard already applied on the
        # explicit-observed_at branch above.
        scheduled_at = db.require_utc(scheduled_at, "scheduled_at")
        final_at = db.require_utc(final_at, "final_at")
        # SCHEDULED observation: the fixture only -- scores are always
        # NULL here, even for a game that has since been played, so a
        # pre-tipoff cutoff sees a fixture with no result rather than the
        # eventual outcome leaking in through the score columns.
        con.execute(
            insert_sql,
            [
                game.game_id,
                season,
                game.game_date,
                game.home_team,
                game.away_team,
                None,
                None,
                "SCHEDULED",
                True,
                scheduled_at,
            ],
        )
        if game.status == "FINAL":
            # FINAL observation: only written for games that were actually
            # played -- a future/unplayed game gets just the SCHEDULED row.
            con.execute(
                insert_sql,
                [
                    game.game_id,
                    season,
                    game.game_date,
                    game.home_team,
                    game.away_team,
                    game.home_points,
                    game.away_points,
                    "FINAL",
                    True,
                    final_at,
                ],
            )
    if local_dropped:
        # Loud, unconditional report of anything that did NOT make it into
        # the database for this season -- individual DROPPED lines were
        # already printed by pair_team_rows/_log_dropped above; this is the
        # summary a caller (CLI or otherwise) sees at the end of the run.
        print(
            f"nba_stats: WARNING -- {len(local_dropped)} game(s) for season "
            f"{season} could not be paired into a game row and were NOT "
            f"ingested: {[d.game_id for d in local_dropped]}"
        )
    return len(games)
