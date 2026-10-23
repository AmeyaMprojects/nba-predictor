from datetime import date

from model_fixtures import add_game, fixture_con
from predictor.model import cities
from predictor.model.venues import Venue, VenueIndex, competitive_sql, is_competitive
from real_archive import open_real_archive_or_skip


def test_from_db_reads_city_and_neutral(tmp_path):
    con = fixture_con(tmp_path)
    add_game(con, "0022400001", "2024-25", date(2025, 1, 15), "PHI", "NYK",
             city="Mexico City, Mexico", neutral=True)
    v = VenueIndex.from_db(con).venue("0022400001")
    assert v == Venue("0022400001", date(2025, 1, 15), "PHI", "NYK", "Mexico City", True)


def test_unknown_game_has_no_venue(tmp_path):
    assert VenueIndex.from_db(fixture_con(tmp_path)).venue("0029999999") is None


def test_latest_schedule_vintage_wins(tmp_path):
    con = fixture_con(tmp_path)
    add_game(con, "0022400001", "2024-25", date(2025, 1, 15), "PHI", "NYK", city="Boston")
    from predictor import db
    from datetime import UTC, datetime
    sched = db.POINT_IN_TIME_TABLES["schedule"]
    con.execute(
        f"INSERT INTO {sched} (game_id, season, game_date, tip_off_utc, home_team,"
        " away_team, arena_city, is_neutral_reported, is_neutral, observed_at)"
        " VALUES ('0022400001','2024-25',DATE '2025-01-15',NULL,'PHI','NYK',"
        " 'Paris',TRUE,TRUE,?)",
        [datetime(2026, 9, 28, tzinfo=UTC)],
    )
    v = VenueIndex.from_db(con).venue("0022400001")
    assert v.city == "Paris" and v.is_neutral is True


def test_recent_games_are_strictly_before_and_oldest_first():
    vs = [
        Venue("0022400001", date(2025, 1, 10), "PHI", "NYK", "Philadelphia", False),
        Venue("0022400002", date(2025, 1, 12), "BOS", "PHI", "Boston", False),
        Venue("0022400003", date(2025, 1, 14), "PHI", "MIA", "Philadelphia", False),
        Venue("0022400004", date(2025, 1, 15), "PHI", "CHI", "Philadelphia", False),
    ]
    idx = VenueIndex(vs)
    got = idx.recent_games("PHI", date(2025, 1, 15), n=2)
    assert [v.game_id for v in got] == ["0022400002", "0022400003"]
    assert idx.recent_games("PHI", date(2025, 1, 10), n=2) == []
    assert idx.recent_games("LAL", date(2025, 1, 15), n=2) == []


def test_preseason_and_all_star_games_are_not_history():
    vs = [
        Venue("0012400001", date(2025, 1, 12), "PHI", "NYK", "Philadelphia", False),
        Venue("0032400001", date(2025, 1, 13), "PHI", "NYK", "Philadelphia", False),
        Venue("0042400001", date(2025, 1, 14), "PHI", "NYK", "Philadelphia", False),
    ]
    got = VenueIndex(vs).recent_games("PHI", date(2025, 1, 15), n=5)
    assert [v.game_id for v in got] == ["0042400001"]


def test_every_competitive_arena_city_in_the_archive_is_in_the_city_table():
    con = open_real_archive_or_skip()
    try:
        from predictor import db
        sched = db.POINT_IN_TIME_TABLES["schedule"]
        rows = con.execute(
            f"SELECT DISTINCT arena_city FROM {sched} "
            "WHERE substr(game_id, 1, 3) IN ('002','004','005','006')"
        ).fetchall()
        missing = sorted({cities.city_key(r[0]) for r in rows} - set(cities.CITIES))
        assert missing == [], f"add these arena cities to cities.CITIES: {missing}"
    finally:
        con.close()


def test_is_competitive():
    assert is_competitive("0022600001") and is_competitive("espn:401705555")
    assert not is_competitive("0012600001")
    for prefix in ("004", "005", "006"):
        assert is_competitive(prefix + "2600001")
    assert not is_competitive("0032600001")


def test_competitive_sql_text_and_column():
    assert competitive_sql() == (
        "(substr(game_id, 1, 3) IN ('002','004','005','006') OR game_id LIKE 'espn:%')"
    )
    assert competitive_sql("l.game_id") == (
        "(substr(l.game_id, 1, 3) IN ('002','004','005','006') OR l.game_id LIKE 'espn:%')"
    )


def test_competitive_sql_agrees_with_is_competitive():
    import duckdb
    ids = ["0022600001", "espn:401705555", "0012600001", "0042600001", "0032600001", "espn"]
    con = duckdb.connect()
    for gid in ids:
        got = con.execute(
            f"SELECT {competitive_sql()} FROM (SELECT ? AS game_id)", [gid]
        ).fetchone()[0]
        assert got is is_competitive(gid), gid


def test_provisional_espn_game_counts_toward_recent_games():
    v = Venue("espn:401705555", date(2026, 10, 22), "LAL", "BOS", None, False)
    idx = VenueIndex([v])
    assert idx.recent_games("LAL", date(2026, 10, 23), 5) == [v]
