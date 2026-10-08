"""Historical closing odds from Kaggle, linked to schedule games.

Dataset: Kaggle ``cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024``
(the slug is historical; the dataset has since been extended to 2025-26).
It is downloaded through Kaggle's public API with HTTP basic auth, using the
token from ``~/.kaggle/kaggle.json`` (`config.kaggle_credentials`):

    GET https://www.kaggle.com/api/v1/datasets/download/<owner>/<slug>

which returns a zip. The flow is raw-first, like sources/odds.py:

    download()  fetch -> archive the zip bytes (raw store ``odds_history``,
                key ``kaggle_<sha16>.zip``; same bytes = no-op)
    load(con)   open the ARCHIVED bytes -> parse -> link -> store -> coverage

File format, as inspected on 2026-10-08:

- The zip holds one CSV, ``nba_2008-2026.csv`` (24,440 rows, 2007-08 to
  2025-26). The member is chosen by its ``.csv`` suffix, since the name
  changes with each update.
- Columns: ``season,date,regular,playoffs,away,home,score_away,score_home,
  q1_away..ot_home,whos_favored,spread,total,moneyline_away,moneyline_home,
  h2_spread,h2_total,id_spread,id_total``. The quarter, ``h2_*`` and
  ``id_*`` columns are ignored.
- ``season`` is the END year: ``2026`` means ``2025-26``.
- ``date`` is ``YYYY-MM-DD``, the US local game date, treated as the ET date
  (our schedule's ``game_date``).
- Team codes are lowercase (``gs``, ``no``, ``ny``, ``sa``, ``utah``,
  ``wsh``, the rest are our abbreviation lower-cased); they are mapped with
  the explicit `KAGGLE_TEAMS` table. An unknown code is reported as
  unmatched, never skipped silently.
- ``spread`` is UNSIGNED and ``whos_favored`` is ``home`` or ``away``. The
  home spread (negative = home favoured) is ``-spread`` when the home team
  is favoured, else ``+spread``. A blank spread is None (3 rows).
- ``moneyline_home``/``moneyline_away`` are American odds, or blank. They
  are blank for every game from 2023-24 onward and for half of 2022-23, so
  those seasons fall back to the spread. This is expected, not an error.
- ``total`` is a float or blank. ``regular``, ``playoffs``, ``score_home``
  and ``score_away`` are kept for linking checks.

Linking and storing: rows before 2014-15 are ignored. Each remaining row is
linked by ET date + home + away against the latest schedule vintage
(competitive games only; `odds.link_game`, with its one-day fallback). If
our results have the game as FINAL, the file's scores must equal ours; a
mismatch is reported as unmatched and not stored. Each linked game is
stored once: ``book='consensus'``, ``source='kaggle_sbr'``,
``reconstructed=TRUE``, ``game_key='kaggle:<game_id>'``, observed_at = the
schedule's tip-off (a closing line is known at tip). A game with no tip-off
is skipped and counted.

Coverage: per season from 2014-15 up to the file's last season, the
regular-season (``002``) games in our schedule and how many are linked to a
line. The comparison with the
market is only trusted when every season in `GATE_SEASONS` has at least
`GATE_MIN_PCT` coverage.
"""

from __future__ import annotations

import csv
import hashlib
import io
import time
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

import requests

from predictor import db, raw_store
from predictor.sources import odds

DATASET = "cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024"
DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}"
DATASET_PAGE = f"https://www.kaggle.com/datasets/{DATASET}"
RAW_SOURCE = "odds_history"
SOURCE = "kaggle_sbr"
BOOK = "consensus"
FIRST_SEASON_END_YEAR = 2015  # 2014-15
GATE_SEASONS: tuple[str, ...] = (
    "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26",
)
GATE_MIN_PCT = 90.0
FETCH_ATTEMPTS = 3

KAGGLE_TEAMS: dict[str, str] = {
    "atl": "ATL", "bkn": "BKN", "bos": "BOS", "cha": "CHA", "chi": "CHI",
    "cle": "CLE", "dal": "DAL", "den": "DEN", "det": "DET", "gs": "GSW",
    "hou": "HOU", "ind": "IND", "lac": "LAC", "lal": "LAL", "mem": "MEM",
    "mia": "MIA", "mil": "MIL", "min": "MIN", "no": "NOP", "ny": "NYK",
    "okc": "OKC", "orl": "ORL", "phi": "PHI", "phx": "PHX", "por": "POR",
    "sa": "SAS", "sac": "SAC", "tor": "TOR", "utah": "UTA", "wsh": "WAS",
}

REQUIRED_COLUMNS = (
    "season", "date", "regular", "playoffs", "away", "home", "score_away",
    "score_home", "whos_favored", "spread", "total", "moneyline_away",
    "moneyline_home",
)


class KaggleAuthError(Exception):
    """Kaggle answered 401/403: wrong token, or the dataset terms are not accepted."""

    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class KaggleFetchError(Exception):
    """Any other failure to get the zip. Never contains the URL or the token."""


def missing_credentials_message() -> str:
    return (
        "No Kaggle API token found. Sign in at https://www.kaggle.com, open "
        "Settings, and under API click 'Create New Token'; that downloads "
        "kaggle.json. Save it as ~/.kaggle/kaggle.json, then run: "
        "chmod 600 ~/.kaggle/kaggle.json"
    )


def auth_failure_message(status_code: int) -> str:
    return (
        f"Kaggle refused the download (HTTP {status_code}): the token in "
        "~/.kaggle/kaggle.json is wrong or expired, or the dataset's terms "
        f"have not been accepted yet -- open {DATASET_PAGE} while signed in "
        "and accept them on the dataset page. Nothing was stored."
    )


def kaggle_team(code: str | None) -> str | None:
    return KAGGLE_TEAMS.get((code or "").strip().lower())


def season_label(end_year: int) -> str:
    return f"{end_year - 1}-{end_year % 100:02d}"


@dataclass(frozen=True)
class HistoryRow:
    line: int  # 1-based line in the CSV (header = 1)
    season: str
    game_date: date
    regular: bool
    playoffs: bool
    home_code: str
    away_code: str
    home: str | None  # our abbreviation, None if the code is unknown
    away: str | None
    score_home: int | None
    score_away: int | None
    home_spread: float | None  # negative = home favoured
    total: float | None
    moneyline_home: int | None
    moneyline_away: int | None

    @property
    def season_end_year(self) -> int:
        return int(self.season[:4]) + 1

    @property
    def label(self) -> str:
        home = self.home or self.home_code
        away = self.away or self.away_code
        return f"{away}@{home} {self.game_date.isoformat()}"


@dataclass(frozen=True)
class HistoryDownload:
    archive_key: str
    observed_at: datetime


@dataclass(frozen=True)
class SeasonCoverage:
    season: str
    scheduled: int  # regular-season games in our schedule
    linked: int  # of those, games with a stored line

    @property
    def pct(self) -> float:
        return 100.0 * self.linked / self.scheduled if self.scheduled else 0.0


@dataclass(frozen=True)
class HistoryLoadSummary:
    rows_read: int
    rows_in_scope: int  # 2014-15 onward
    stored: int
    no_tip: int
    unmatched: list[str] = field(default_factory=list)
    shifted: list[str] = field(default_factory=list)
    seasons: list[SeasonCoverage] = field(default_factory=list)
    # kaggle_sbr lines from an earlier load whose game this load did not
    # store (e.g. a score correction made it unmatched), deleted.
    removed_stale: int = 0

    def failing_seasons(self, gate: tuple[str, ...] | None = None) -> list[str]:
        """Gate seasons with coverage below GATE_MIN_PCT (or no schedule games)."""
        gate = GATE_SEASONS if gate is None else gate
        by_season = {c.season: c for c in self.seasons}
        failing = []
        for season in sorted(gate):
            cov = by_season.get(season)
            if cov is None or cov.scheduled == 0 or cov.pct < GATE_MIN_PCT:
                failing.append(season)
        return failing


# --- download -----------------------------------------------------------------


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def fetch_archive(username: str, key: str, session=None) -> bytes:
    """The dataset zip. Retries network errors only; never echoes URL or key."""
    if session is None:
        with requests.Session() as own:
            return _fetch_with(own, username, key)
    return _fetch_with(session, username, key)


def _fetch_with(session, username: str, key: str) -> bytes:
    for attempt in range(1, FETCH_ATTEMPTS + 1):
        try:
            response = session.get(DOWNLOAD_URL, auth=(username, key), timeout=(15, 300))
        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt == FETCH_ATTEMPTS:
                raise KaggleFetchError(
                    f"could not reach Kaggle after {FETCH_ATTEMPTS} attempts "
                    f"({type(exc).__name__})"
                ) from None
            _sleep(2 * attempt)
            continue
        except requests.RequestException as exc:
            raise KaggleFetchError(f"could not reach Kaggle ({type(exc).__name__})") from None
        break
    if response.status_code in (401, 403):
        raise KaggleAuthError(response.status_code)
    if response.status_code != 200:
        raise KaggleFetchError(f"Kaggle returned HTTP {response.status_code}")
    body = response.content
    if not zipfile.is_zipfile(io.BytesIO(body)):
        raise KaggleFetchError("Kaggle returned a response that is not a zip file")
    return body


def _utcnow() -> datetime:
    return datetime.now(UTC)


def download(
    credentials: tuple[str, str], fetch=None, now: datetime | None = None
) -> HistoryDownload:
    """Fetch the zip and archive its bytes. Touches no database.

    ``fetch(username, key) -> bytes`` defaults to `fetch_archive` (tests
    inject their own). The archive key is content-addressed, so the same
    bytes are a raw_store no-op.
    """
    fetch = fetch or fetch_archive
    username, key = credentials
    body = fetch(username, key)
    observed_at = db.require_utc(now if now is not None else _utcnow(), "observed_at")
    archive_key = f"kaggle_{hashlib.sha256(body).hexdigest()[:16]}.zip"
    raw_store.store(
        RAW_SOURCE, archive_key, body, observed_at,
        meta={"dataset": DATASET, "observed_at": observed_at.isoformat()},
    )
    return HistoryDownload(archive_key, observed_at)


# --- parse --------------------------------------------------------------------


def read_csv_member(body: bytes) -> str:
    """The text of the zip's single ``.csv`` member."""
    with zipfile.ZipFile(io.BytesIO(body)) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(
                f"expected exactly one .csv file in the Kaggle zip, found {len(members)}"
            )
        return zf.read(members[0]).decode("utf-8-sig")


def _text(raw: dict, column: str) -> str:
    """A cell's text; a cell missing from a short row (None) reads as blank."""
    value = raw.get(column)
    return value.strip() if isinstance(value, str) else ""


def _blank(value: str | None) -> bool:
    return value is None or not value.strip()


def _opt_int(value: str | None) -> int | None:
    return None if _blank(value) else int(float(value))


def _opt_float(value: str | None) -> float | None:
    return None if _blank(value) else float(value)


def _bool(value: str) -> bool:
    text = value.strip().lower()
    if text in ("true", "1"):
        return True
    if text in ("false", "0"):
        return False
    raise ValueError(f"not a boolean: {value!r}")


def parse_csv(text: str) -> list[HistoryRow]:
    reader = csv.DictReader(io.StringIO(text))
    missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
    if missing:
        raise ValueError(f"the Kaggle odds file lacks columns: {', '.join(missing)}")
    rows: list[HistoryRow] = []
    for raw in reader:
        line = reader.line_num
        try:
            spread = _opt_float(_text(raw, "spread"))
            favoured = _text(raw, "whos_favored").lower()
            if spread is not None:
                if favoured == "home":
                    spread = -spread
                elif favoured != "away":
                    raise ValueError(f"whos_favored is {favoured!r}")
            home_code = _text(raw, "home").lower()
            away_code = _text(raw, "away").lower()
            if not home_code or not away_code:
                raise ValueError("a team code is blank")
            rows.append(
                HistoryRow(
                    line=line,
                    season=season_label(int(_text(raw, "season"))),
                    game_date=date.fromisoformat(_text(raw, "date")),
                    regular=_bool(_text(raw, "regular")),
                    playoffs=_bool(_text(raw, "playoffs")),
                    home_code=home_code,
                    away_code=away_code,
                    home=kaggle_team(home_code),
                    away=kaggle_team(away_code),
                    score_home=_opt_int(_text(raw, "score_home")),
                    score_away=_opt_int(_text(raw, "score_away")),
                    home_spread=spread + 0.0 if spread is not None else None,
                    total=_opt_float(_text(raw, "total")),
                    moneyline_home=_opt_int(_text(raw, "moneyline_home")),
                    moneyline_away=_opt_int(_text(raw, "moneyline_away")),
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"the Kaggle odds file has a bad row at line {line}: {exc}") from None
    return rows


# --- link, store, coverage ----------------------------------------------------


def _final_scores(con) -> dict[str, tuple[int, int]]:
    """game_id -> (home, away) points from each game's latest FINAL result."""
    table = db.POINT_IN_TIME_TABLES["games"]
    rows = con.execute(
        f"""
        SELECT game_id, home_points, away_points FROM (
            SELECT game_id, home_points, away_points,
                   row_number() OVER (PARTITION BY game_id ORDER BY observed_at DESC) AS rn
            FROM {table}
            WHERE status = 'FINAL' AND home_points IS NOT NULL AND away_points IS NOT NULL
        ) WHERE rn = 1
        """
    ).fetchall()
    return {game_id: (home, away) for game_id, home, away in rows}


def load(con, downloaded: HistoryDownload) -> HistoryLoadSummary:
    """Parse the archived zip, link each row to a game, store, report coverage."""
    rows = parse_csv(read_csv_member(raw_store.load(RAW_SOURCE, downloaded.archive_key)))
    in_scope = [r for r in rows if r.season_end_year >= FIRST_SEASON_END_YEAR]

    games = [
        g for g in odds.latest_schedule_games(con)
        if int(g.season[:4]) + 1 >= FIRST_SEASON_END_YEAR
    ]
    by_id = {g.game_id: g for g in games}
    index = odds.build_link_index(games)
    finals = _final_scores(con)

    unmatched: list[str] = []
    shifted: list[str] = []
    no_tip = 0
    to_store: dict[str, tuple[HistoryRow, odds.ScheduleGame]] = {}
    for row in in_scope:
        where = f"{row.season} line {row.line} {row.label}"
        if row.home is None or row.away is None:
            pairs = ((row.away_code, row.away), (row.home_code, row.home))
            unknown = [code for code, team in pairs if team is None]
            unmatched.append(f"{where}: unknown team code {', '.join(unknown)}")
            continue
        game_id, matched_day = odds.link_game(index, row.game_date, row.home, row.away)
        if game_id is None:
            unmatched.append(f"{where}: no scheduled game")
            continue
        if game_id in to_store:
            unmatched.append(f"{where}: a second line for game {game_id}")
            continue
        final = finals.get(game_id)
        if final is not None and final != (row.score_home, row.score_away):
            unmatched.append(
                f"{where}: score {row.score_home}-{row.score_away} (home-away) does not "
                f"match our final {final[0]}-{final[1]} for game {game_id}"
            )
            continue
        game = by_id[game_id]
        if game.tip_off_utc is None:
            no_tip += 1
            unmatched.append(f"{where}: no tip-off in the schedule for game {game_id}")
            continue
        if matched_day != row.game_date:
            shifted.append(f"{row.label} -> {matched_day.isoformat()}")
        to_store[game_id] = (row, game)

    table = db.POINT_IN_TIME_TABLES["odds_snapshots"]
    insert_sql = (
        f"INSERT OR REPLACE INTO {table} (game_key, book, home_team, away_team,"
        " home_price, away_price, spread, total, observed_at, game_id, source,"
        " reconstructed) VALUES (?,?,?,?,?,?,?,?,?,?,?,TRUE)"
    )
    con.execute("BEGIN TRANSACTION")
    try:
        # Only this source's rows: a kaggle line whose game this load did
        # not store (now unmatched, e.g. after a score correction) must not
        # survive as a stale line. Live (theoddsapi) rows are never touched.
        keep = [f"kaggle:{game_id}" for game_id in to_store]
        removed_stale = con.execute(
            f"SELECT count(*) FROM {table} WHERE source = ?"
            " AND NOT list_contains(?::VARCHAR[], game_key)",
            [SOURCE, keep],
        ).fetchone()[0]
        con.execute(
            f"DELETE FROM {table} WHERE source = ?"
            " AND NOT list_contains(?::VARCHAR[], game_key)",
            [SOURCE, keep],
        )
        for game_id, (row, game) in to_store.items():
            game_key = f"kaggle:{game_id}"
            # A tip-off that moved since the last load would otherwise leave
            # a second row under a different observed_at.
            con.execute(
                f"DELETE FROM {table} WHERE game_key = ? AND source = ? AND observed_at <> ?",
                [game_key, SOURCE, game.tip_off_utc],
            )
            con.execute(
                insert_sql,
                [
                    game_key, BOOK, game.home_team, game.away_team,
                    row.moneyline_home, row.moneyline_away, row.home_spread,
                    row.total, game.tip_off_utc, game_id, SOURCE,
                ],
            )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise

    # Seasons after the file's last one (not played yet) are not reported.
    last_end_year = max((r.season_end_year for r in rows), default=0)
    seasons: dict[str, list[int]] = {}
    for game in games:
        if int(game.season[:4]) + 1 > last_end_year:
            continue
        counts = seasons.setdefault(game.season, [0, 0])
        if game.game_id.startswith("002"):
            counts[0] += 1
            if game.game_id in to_store:
                counts[1] += 1
    coverage = [SeasonCoverage(s, n, k) for s, (n, k) in sorted(seasons.items())]

    return HistoryLoadSummary(
        rows_read=len(rows),
        rows_in_scope=len(in_scope),
        stored=len(to_store),
        no_tip=no_tip,
        unmatched=unmatched,
        shifted=shifted,
        seasons=coverage,
        removed_stale=removed_stale,
    )
