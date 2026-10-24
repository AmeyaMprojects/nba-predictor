"""ESPN players -> NBA player ids, box-score state rows, and due summaries."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import pytest

from predictor import db
from predictor.cloud import players, state
from predictor.cloud.players import PlayerLink, normalize_name
from predictor.sources.espn import EspnPlayerLine

NOW = datetime(2026, 10, 24, 15, 0, tzinfo=UTC)  # slate 2026-10-24


def dt(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)


def line(pid="3945274", name="Luka Doncic", team="LAL", dnp=False):
    return EspnPlayerLine(
        espn_player_id=pid, name=name, team=team, starter=not dnp, did_not_play=dnp,
        minutes=None if dnp else 34.0, points=None if dnp else 27,
        rebounds=None if dnp else 8, assists=None if dnp else 9,
        plus_minus=None if dnp else 12,
    )


def index(rows):
    """rows: (player_id, name, team, season)"""
    return players.PlayerIndex.from_rows(rows)


BASIC = index([
    (1629029, "Luka Dončić", "LAL", "2026-27"),
    (1628991, "Jaren Jackson Jr.", "MEM", "2026-27"),
    (202694, "Marcus Morris", "CLE", "2026-27"),
    (1626149, "Marcus Morris", "PHI", "2026-27"),
])


def test_normalize_name():
    assert normalize_name("Luka Dončić") == "luka doncic"
    assert normalize_name("Jaren Jackson Jr.") == "jaren jackson"
    assert normalize_name("Shai Gilgeous-Alexander") == "shai gilgeous alexander"
    assert normalize_name("De'Aaron Fox") == "deaaron fox"
    assert normalize_name("  Ronald  Holland II ") == "ronald holland"
    assert normalize_name("Kelly Oubre Jr") == "kelly oubre"


def test_unique_name_match():
    link = players.link_player(line(), BASIC, {}, {}, NOW)
    assert link == PlayerLink("3945274", 1629029, "Luka Doncic", "LAL", "name", NOW)


def test_suffix_and_accent_variants_match():
    link = players.link_player(line("1", "Jaren Jackson"), BASIC, {}, {}, NOW)
    assert (link.nba_player_id, link.method) == (1628991, "name")


def test_alias_wins_over_name():
    aliases = {"by_espn_id": {"3945274": 999}}
    link = players.link_player(line(), BASIC, aliases, {}, NOW)
    assert (link.nba_player_id, link.method) == (999, "alias")
    by_name = {"by_name": {"luka doncic": 998}}
    link = players.link_player(line(), BASIC, by_name, {}, NOW)
    assert (link.nba_player_id, link.method) == (998, "alias")


def test_existing_link_is_reused_before_aliases_by_name():
    old = PlayerLink("3945274", 1629029, "Luka Doncic", "LAL", "name", dt("2026-10-01T00:00Z"))
    link = players.link_player(line(), BASIC, {"by_name": {"luka doncic": 5}},
                               {"3945274": old}, NOW)
    assert link.nba_player_id == 1629029 and link.method == "existing"
    assert link.linked_at == old.linked_at


def test_alias_by_espn_id_corrects_an_existing_link():
    old = PlayerLink("3945274", 1, "Luka Doncic", "LAL", "name", dt("2026-10-01T00:00Z"))
    link = players.link_player(line(), BASIC, {"by_espn_id": {"3945274": 1629029}},
                               {"3945274": old}, NOW)
    assert (link.nba_player_id, link.method, link.linked_at) == (1629029, "alias", NOW)


def test_ambiguous_name_resolved_by_team():
    link = players.link_player(line("77", "Marcus Morris Sr.", "PHI"), BASIC, {}, {}, NOW)
    assert (link.nba_player_id, link.method) == (1626149, "name+team")


def test_ambiguous_name_on_another_team_is_unmatched():
    link = players.link_player(line("77", "Marcus Morris", "BOS"), BASIC, {}, {}, NOW)
    assert (link.nba_player_id, link.method) == (None, "unmatched")


def test_unmatched_is_stored_and_reported(tmp_path):
    lines = [line(), line("42", "Nobody Known", "BOS")]
    result = players.link_lines(tmp_path, lines, BASIC, NOW)
    assert [u.espn_player_id for u in result.unmatched] == ["42"]
    stored = players.read_links(tmp_path)
    assert stored["42"].method == "unmatched" and stored["42"].nba_player_id is None
    assert stored["3945274"].nba_player_id == 1629029
    path = tmp_path / players.LINKS_PATH
    assert path.read_text().splitlines()[0] == '{"format":"lines","v":1}'

    # Next run: nothing new, nothing appended; still reported.
    before = path.read_text()
    again = players.link_lines(tmp_path, lines, BASIC, NOW)
    assert path.read_text() == before
    assert [u.espn_player_id for u in again.unmatched] == ["42"]
    assert again.links["3945274"].method == "existing"

    # An alias appears: the unmatched one is re-tried and appended once.
    (tmp_path / players.ALIASES_PATH).write_text(json.dumps({"by_espn_id": {"42": 7}}))
    later = players.link_lines(tmp_path, lines, BASIC, NOW)
    assert later.unmatched == []
    assert players.read_links(tmp_path)["42"].nba_player_id == 7
    assert len(path.read_text().splitlines()) == len(before.splitlines()) + 1


def test_link_records_are_sorted_compact_json(tmp_path):
    players.link_lines(tmp_path, [line()], BASIC, NOW)
    record = (tmp_path / players.LINKS_PATH).read_text().splitlines()[1]
    assert record == (
        '{"espn_player_id":"3945274","linked_at":"2026-10-24T15:00:00+00:00",'
        '"method":"name","name":"Luka Doncic","nba_player_id":1629029,"team":"LAL"}'
    )


def test_index_build_reads_static_players_and_mac_players_file(tmp_path, monkeypatch):
    monkeypatch.setattr(players, "_static_players", lambda: [
        {"id": 202694, "full_name": "Marcus Morris"},
        {"id": 1626149, "full_name": "Marcus Morris"},
        {"id": 1629029, "full_name": "Luka Dončić"},
    ])
    state.write_snapshot(tmp_path, "nba/players-2026-27.jsonl", [
        {"player_id": 1626149, "name": "Marcus Morris Sr.", "team": "PHI"},
        {"player_id": 1641705, "name": "Brand New", "team": "BOS"},
    ])
    idx = players.PlayerIndex.build(tmp_path)
    assert players.link_player(line(), idx, {}, {}, NOW).nba_player_id == 1629029
    morris = players.link_player(line("7", "Marcus Morris", "PHI"), idx, {}, {}, NOW)
    assert (morris.nba_player_id, morris.method) == (1626149, "name+team")
    rookie = players.link_player(line("8", "Brand New", "BOS"), idx, {}, {}, NOW)
    assert rookie.nba_player_id == 1641705


def test_index_build_works_offline_with_the_real_static_list(tmp_path):
    idx = players.PlayerIndex.build(tmp_path)
    link = players.link_player(line("1", "Shai Gilgeous-Alexander", "OKC"), idx, {}, {}, NOW)
    assert link.nba_player_id == 1628983


# --- box-score rows and rebuild ----------------------------------------------

GAME = {"game_id": "0022600011", "season": "2026-27", "game_date": date(2026, 10, 22)}
SEEN = dt("2026-10-23T09:00Z")


def test_box_rows_store_espn_id_and_no_nba_id():
    lines = [line(), line("42", "Nobody", "BOS", dnp=True)]
    links = {ln.espn_player_id: players.link_player(ln, BASIC, {}, {}, NOW) for ln in lines}
    rows = players.box_rows(GAME, lines, links, SEEN)
    assert rows[0] == {
        "game_id": "0022600011", "season": "2026-27", "game_date": date(2026, 10, 22),
        "team": "LAL", "espn_player_id": "3945274", "player_name": "Luka Doncic",
        "starter": True, "did_not_play": False, "minutes": 34.0, "points": 27,
        "rebounds": 8, "assists": 9, "plus_minus": 12, "source": "espn",
        "observed_at": SEEN,
    }
    assert rows[1]["did_not_play"] is True and rows[1]["minutes"] is None
    assert all("player_id" not in r and "nba_player_id" not in r for r in rows)


def test_box_rows_need_a_link_for_every_line():
    with pytest.raises(ValueError, match="42"):
        players.box_rows(GAME, [line("42", "Nobody", "BOS")], {}, SEEN)


def test_box_rows_reject_naive_time():
    with pytest.raises(ValueError):
        players.box_rows(GAME, [], {}, datetime(2026, 10, 23, 9))


def _player_ids(db_path):
    con = db.connect(db_path)
    try:
        table = db.POINT_IN_TIME_TABLES["player_boxscores"]
        return dict(con.execute(f"SELECT espn_player_id, player_id FROM {table}").fetchall())
    finally:
        con.close()


def test_rebuild_resolves_player_id_after_alias_added(tmp_path):
    st = tmp_path / "state"
    unknown = line("3945274", "Lukas Dončić", "LAL")  # misspelt: no name match
    known = line("1628991", "Jaren Jackson Jr.", "MEM")
    result = players.link_lines(st, [unknown, known], BASIC, NOW)
    state.append_rows(st, "boxscores", players.box_rows(GAME, [unknown, known],
                                                        result.links, SEEN))
    state.rebuild_db(st, tmp_path / "a.duckdb")
    assert _player_ids(tmp_path / "a.duckdb") == {"3945274": "espn:3945274",
                                                  "1628991": "1628991"}

    (st / players.ALIASES_PATH).write_text(json.dumps({"by_espn_id": {"3945274": 1630178}}))
    state.rebuild_db(st, tmp_path / "b.duckdb")  # alias alone fixes old rows
    assert _player_ids(tmp_path / "b.duckdb")["3945274"] == "1630178"

    players.link_lines(st, [unknown], BASIC, NOW)  # and the next run records it
    assert players.read_links(st)["3945274"].nba_player_id == 1630178
    (st / players.ALIASES_PATH).unlink()
    state.rebuild_db(st, tmp_path / "c.duckdb")
    assert _player_ids(tmp_path / "c.duckdb")["3945274"] == "1630178"


def test_rebuild_applies_alias_by_name_to_an_unmatched_link(tmp_path):
    st = tmp_path / "state"
    unknown = line("55", "Somebody New", "BOS")
    result = players.link_lines(st, [unknown], BASIC, NOW)
    state.append_rows(st, "boxscores", players.box_rows(GAME, [unknown], result.links, SEEN))
    (st / players.ALIASES_PATH).write_text(json.dumps({"by_name": {"somebody new": 123}}))
    state.rebuild_db(st, tmp_path / "a.duckdb")
    assert _player_ids(tmp_path / "a.duckdb") == {"55": "123"}


def test_apply_boxscores_appends_and_loads(tmp_path):
    st = tmp_path / "state"
    con = db.connect(tmp_path / "run.duckdb")
    db.migrate(con)
    try:
        lines = [line(), line("42", "Nobody", "BOS", dnp=True)]
        result = players.link_lines(st, lines, BASIC, NOW)
        rows = players.box_rows(GAME, lines, result.links, SEEN)
        touched = players.apply_boxscores(con, st, rows)
        assert [p.relative_to(st).as_posix() for p in touched] == [
            "boxscores/2026-27/2026-10.jsonl"]
        table = db.POINT_IN_TIME_TABLES["player_boxscores"]
        got = dict(con.execute(f"SELECT espn_player_id, player_id FROM {table}").fetchall())
        assert got == {"3945274": "1629029", "42": "espn:42"}
    finally:
        con.close()


# --- which summaries are due ---------------------------------------------------


@pytest.fixture
def con(tmp_path):
    c = db.connect(tmp_path / "t.duckdb")
    db.migrate(c)
    yield c
    c.close()


def add_final(con, game_id, day, status="FINAL"):
    table = db.POINT_IN_TIME_TABLES["games"]
    con.execute(
        f"INSERT INTO {table} VALUES (?,?,?,?,?,?,?,?,?,?)",
        [game_id, "2026-27", day, "BOS", "NYK", 100, 90, status, False,
         dt("2026-10-24T06:00Z")],
    )


def test_boxscores_due_caps_at_15(con):
    links = {}
    for n in range(20):
        gid = f"00226{n:05d}"
        add_final(con, gid, date(2026, 10, 21 + n % 4))
        links[f"40190{n:04d}"] = gid
    due = players.boxscores_due(con, NOW, links)
    assert len(due) == players.MAX_SUMMARIES_PER_RUN == 15
    assert len(set(due)) == 15
    # Oldest dates first, so nothing ages out of the window unfetched.
    days = [n % 4 for n in range(20)]
    assert sorted(days)[:15] == [days[int(g[-5:])] for g, _ in due]
    assert all(links[e] == g for g, e in due)


def test_boxscores_due_rules(con, tmp_path):
    add_final(con, "G_OK", date(2026, 10, 23))
    add_final(con, "G_SLATE", date(2026, 10, 24))
    add_final(con, "G_OLD", date(2026, 10, 20))          # 4 days before: out
    add_final(con, "G_SCHED", date(2026, 10, 23), status="SCHEDULED")
    add_final(con, "G_NOLINK", date(2026, 10, 23))
    add_final(con, "G_HAVE", date(2026, 10, 22))
    add_final(con, "espn:401", date(2026, 10, 22))       # provisional: id is the link
    links = {"1": "G_OK", "2": "G_SLATE", "3": "G_OLD", "4": "G_SCHED", "6": "G_HAVE"}
    game = {"game_id": "G_HAVE", "season": "2026-27", "game_date": date(2026, 10, 22)}
    rows = players.box_rows(game, [line()], {"3945274": players.link_player(
        line(), BASIC, {}, {}, NOW)}, SEEN)
    state.load_rows(con, "boxscores", rows, {})
    assert sorted(players.boxscores_due(con, NOW, links)) == [
        ("G_OK", "1"), ("G_SLATE", "2"), ("espn:401", "401")]


def test_boxscores_due_follows_id_links(con):
    # An ESPN link still naming the provisional id whose rows now live under
    # the NBA id (rebuild mapped them).
    add_final(con, "0022600099", date(2026, 10, 23))
    due = players.boxscores_due(con, NOW, {"401": "espn:401"},
                                id_links={"espn:401": "0022600099"})
    assert due == [("0022600099", "401")]


def test_game_info_reads_season_and_date(con):
    add_final(con, "G1", date(2026, 10, 23))
    assert players.game_info(con, "G1") == {
        "game_id": "G1", "season": "2026-27", "game_date": date(2026, 10, 23)}
