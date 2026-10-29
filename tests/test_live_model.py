"""The live-model flag (``state/settings/live_model.json``) and the two
callers (plan 3, Task 6)."""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, date, datetime, timedelta

import pytest

from model_fixtures import fixture_con
from player_fixtures import D3, G3 as G3_ID, SEASON, final_at
from predictor.asof import AsOfView
from predictor.backtest.baselines import GameToPredict
from predictor.cloud import state as cloud_state
from predictor.model import live, live_model
from predictor.model import player_settings as ps_mod
from predictor.model import player_stage1 as ps1
from predictor.model.calls import GameChain
from predictor.model.live_model import (
    LIVE_MODEL_FILE,
    MODEL_VERSION,
    CallOutcome,
    LiveModel,
    LiveModelError,
    PlayerCaller,
    TeamCaller,
    kind_for_game,
    kind_for_slate,
    load_live_model,
    load_player_settings,
    make_caller,
    parse_live_model,
    player_settings_dict,
    team_settings_dict,
)
from predictor.model.ratings import HomeParams
from predictor.model.stage1 import Stage1Predictor
from test_live_predict import S as TEAM_S
from test_live_predict import _SETTINGS_JSON
from test_player_stage1 import _build, _settings

FIX = "Fix state/settings/live_model.json."
G3 = GameToPredict(G3_ID, SEASON, D3, "BOS", "NYK")


def _write_flag(state_dir, doc) -> None:
    path = state_dir / LIVE_MODEL_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc if isinstance(doc, str) else json.dumps(doc))


@pytest.fixture
def con(tmp_path):
    c = fixture_con(tmp_path)
    _build(c)
    yield c
    c.close()


# -- the flag ----------------------------------------------------------------


def test_missing_flag_file_means_team(tmp_path):
    assert LIVE_MODEL_FILE == live_model.Path("settings/live_model.json")
    assert load_live_model(tmp_path) == LiveModel.team() == LiveModel("team", None)
    assert kind_for_slate(load_live_model(tmp_path), date(2030, 1, 1)) == "team"


def test_flag_switches_from_slate(tmp_path):
    _write_flag(tmp_path, {"model": "player", "from_slate": "2026-11-10"})
    flag = load_live_model(tmp_path)
    assert flag == LiveModel("player", date(2026, 11, 10))
    assert kind_for_slate(flag, date(2026, 11, 9)) == "team"
    assert kind_for_slate(flag, date(2026, 11, 10)) == "player"
    assert kind_for_slate(flag, date(2026, 11, 11)) == "player"


def test_team_flag_parses():
    assert parse_live_model('{"model": "team"}') == LiveModel.team()
    assert parse_live_model('{"model": "team", "from_slate": null}') == LiveModel.team()
    flag = parse_live_model('{"model": "team", "from_slate": "2026-11-10"}')
    assert kind_for_slate(flag, date(2026, 12, 1)) == "team"


def test_model_versions():
    assert MODEL_VERSION == {"team": "v1", "player": "v2"}


def test_game_keeps_model_of_its_first_call():
    flag = LiveModel("player", date(2026, 11, 10))
    slate = date(2026, 11, 12)
    v1_early = {"game_id": "g", "game_date": "2026-11-12", "status": "predicted"}
    chain = GameChain((v1_early,))
    assert kind_for_game(flag, slate, chain) == "team"
    # A later v2 line does not move the game: the first predicted line decides.
    chain = GameChain((v1_early, {**v1_early, "model_version": "v2"}))
    assert kind_for_game(flag, slate, chain) == "team"
    # The reverse: a game first called by v2 stays on v2 even if the flag is off.
    chain = GameChain(({**v1_early, "model_version": "v2"},))
    assert kind_for_game(LiveModel.team(), slate, chain) == "player"
    assert kind_for_game(flag, slate, GameChain(({**v1_early, "model_version": "v1"},))) == "team"


def test_game_without_predicted_line_uses_slate():
    flag = LiveModel("player", date(2026, 11, 10))
    tbd = {"game_id": "g", "game_date": "2026-11-12", "status": "not_predicted"}
    assert kind_for_game(flag, date(2026, 11, 12), GameChain((tbd,))) == "player"
    assert kind_for_game(flag, date(2026, 11, 9), GameChain((tbd,))) == "team"
    assert kind_for_game(flag, date(2026, 11, 12), None) == "player"


def test_game_with_unknown_model_version_raises():
    line = {"game_id": "g", "game_date": "2026-11-12", "status": "predicted",
            "model_version": "v9"}
    with pytest.raises(LiveModelError):
        kind_for_game(LiveModel.team(), date(2026, 11, 12), GameChain((line,)))


@pytest.mark.parametrize("text", [
    '{"model": "plyer"}',
    '{"model": "player"}',
    '{"model": "player", "from_slate": null}',
    '{"model": "player", "from_slate": "2026-11-31"}',
    '{"model": "player", "from_slate": "20261110"}',
    '{"model": "player", "from_slate": 20261110}',
    '{"model": "player", "from_slate": "2026-11-10", "extra": 1}',
    '{"from_slate": "2026-11-10"}',
    '["player"]',
    '"player"',
    "",
    "{not json",
    '{"model": "team", "from_slate": "soon"}',
])
def test_bad_flag_file_raises(tmp_path, text):
    _write_flag(tmp_path, text)
    with pytest.raises(LiveModelError) as exc:
        load_live_model(tmp_path)
    assert str(exc.value).endswith(FIX)
    with pytest.raises(LiveModelError):
        parse_live_model(text)


def test_unreadable_flag_file_raises(tmp_path):
    (tmp_path / LIVE_MODEL_FILE).mkdir(parents=True)  # a directory, not a file
    with pytest.raises(LiveModelError) as exc:
        load_live_model(tmp_path)
    assert str(exc.value).endswith(FIX)


# -- player settings -----------------------------------------------------------


def test_live_refuses_settings_with_correction(tmp_path):
    path = tmp_path / "player_settings.json"
    corrected = _settings(correction={"training_seasons": ["2023-24"],
                                      "params": dict(ps_mod.GBM_PARAMS)})
    ps_mod.save(corrected, path)
    with pytest.raises(LiveModelError) as exc:
        load_player_settings(path)
    msg = str(exc.value)
    assert msg.startswith("the player settings include a correction model, which live calls "
                          "do not support; refit without it.")
    assert msg.endswith(FIX)
    ps_mod.save(_settings(), path)
    assert load_player_settings(path) == _settings()


def test_missing_player_settings_is_a_live_model_error(tmp_path):
    with pytest.raises(LiveModelError) as exc:
        load_player_settings(tmp_path / "nope.json")
    assert str(exc.value).endswith(FIX)


def test_settings_dicts():
    assert team_settings_dict(TEAM_S) == _SETTINGS_JSON
    assert live._settings_dict is team_settings_dict
    per_team = dataclasses.replace(TEAM_S, home=HomeParams(150.0, 0.66))
    assert team_settings_dict(per_team) == {**_SETTINGS_JSON, "home_shrinkage": 150.0}
    s = _settings()
    assert player_settings_dict(s) == {
        "model": "player", "k": 0.06, "margin_cap": 20.0, "season_regression": 0.3,
        "hca_window": 400, "formula": "both", "w": 0.25,
        "player_half_life_minutes": 1500.0, "player_season_regression": 0.4,
        "home_shrinkage": None, "sigma": 12.5, "half_life": None,
    }
    assert player_settings_dict(dataclasses.replace(s, home=HomeParams(50.0)))[
        "home_shrinkage"] == 50.0
    json.dumps(player_settings_dict(s), allow_nan=False)  # no inf reaches the log


# -- callers -------------------------------------------------------------------


def test_team_caller_matches_stage1(con):
    view = AsOfView(con, final_at(G3.game_date))
    expected = Stage1Predictor(con, TEAM_S).explain(G3, view)
    caller = TeamCaller(con, TEAM_S)
    for kind in ("early", "revision", "final"):
        out = caller.call(G3, view, kind)
        assert isinstance(out, CallOutcome)
        assert out.spread == expected.spread
        assert out.p_home == expected.p_home
        assert out.sentence == expected.sentence()
        assert out.terms == dict(expected.terms())
        assert out.availability is None
        assert out.model_version == "v1"
        assert out.settings == _SETTINGS_JSON
        assert out.sigma == TEAM_S.sigma


def test_player_caller_matches_predictor(con):
    view = AsOfView(con, final_at(G3.game_date))
    expected = ps1.PlayerStage1Predictor(con, _settings()).explain(G3, view, "final")
    out = PlayerCaller(con, _settings()).call(G3, view, "final")
    assert out.spread == expected.spread
    assert out.p_home == expected.p_home
    assert out.sentence == expected.sentence()
    assert out.terms == dict(expected.terms())
    assert out.model_version == "v2"
    assert out.settings == player_settings_dict(_settings())
    assert out.sigma == 12.5
    assert out.availability == [
        {"team": t, "player": n, "status": st} for t, n, st in expected.absent
    ]
    assert out.availability  # the fixture lists absences on this date


def test_player_caller_uses_early_table_only_for_early(con, monkeypatch):
    seen: list[str] = []
    real = ps1.PlayerStage1Predictor.explain

    def spy(self, game, view, call):
        seen.append(call)
        return real(self, game, view, call)

    monkeypatch.setattr(ps1.PlayerStage1Predictor, "explain", spy)
    caller = PlayerCaller(con, _settings())
    view = AsOfView(con, final_at(G3.game_date))
    for kind in ("early", "revision", "final"):
        caller.call(G3, view, kind)
    assert seen == ["early", "final", "final"]
    with pytest.raises(ValueError):
        caller.call(G3, view, "late")


def _fake_breakdown(absent) -> ps1.PlayerBreakdown:
    return ps1.PlayerBreakdown(
        game_id="g", home_team="DEN", away_team="BOS", rating=1.0, roster=0.5, home=2.0,
        rest=0.0, travel=0.0, altitude=0.8, injuries=-1.5, adjustment=0.0, spread=2.8,
        p_home=0.6, absent=absent,
    )


def test_player_caller_lists_absences(con, monkeypatch):
    monkeypatch.setattr(
        ps1.PlayerStage1Predictor, "explain",
        lambda self, game, view, call: _fake_breakdown((("DEN", "Jamal Murray", "Out"),)),
    )
    out = PlayerCaller(con, _settings()).call(G3, AsOfView(con, final_at(G3.game_date)), "final")
    assert out.availability == [{"team": "DEN", "player": "Jamal Murray", "status": "Out"}]
    assert out.spread == 2.8 and out.p_home == 0.6


def test_player_caller_no_absences_is_empty_list(con, monkeypatch):
    monkeypatch.setattr(ps1.PlayerStage1Predictor, "explain",
                        lambda self, game, view, call: _fake_breakdown(()))
    out = PlayerCaller(con, _settings()).call(G3, AsOfView(con, final_at(G3.game_date)), "early")
    assert out.availability == []


def test_player_caller_reads_reports_at_call_time(con):
    """Live v2 has no report lag: the 15-minute lag is the replay's only."""
    caller = PlayerCaller(con, _settings())
    assert caller.predictor.report_lag == timedelta(0)


def test_player_caller_passes_cloud_id_links(con, tmp_path):
    """The cloud's id links (``state/schedule/id_links.jsonl``) reach the
    player lines, so ESPN rows under a provisional id count for the game."""
    from predictor import db

    state_dir = tmp_path / "state"
    cloud_state.append_records(
        state_dir / cloud_state.ID_LINKS_PATH,
        [{"provisional_id": "espn:77", "game_id": "0022400099",
          "linked_at": "2024-11-02T00:00:00+00:00"}],
    )
    table = db.POINT_IN_TIME_TABLES["player_boxscores"]
    con.execute(
        f"INSERT INTO {table} (game_id, season, game_date, team, player_id,"
        " espn_player_id, player_name, starter, did_not_play, minutes, points,"
        " rebounds, assists, plus_minus, source, observed_at)"
        " VALUES ('espn:77', '2024-25', DATE '2024-11-01', 'BOS', '1', '91',"
        " 'Al Adams', TRUE, FALSE, 30.0, 10, 5, 3, 4, 'espn', ?)",
        [datetime(2024, 11, 2, 5, 0, tzinfo=UTC)],
    )
    at = datetime(2024, 11, 3, 0, 0, tzinfo=UTC)
    caller = make_caller(con, "player", TEAM_S, _settings(),
                         id_links=cloud_state.read_id_links(state_dir))
    assert isinstance(caller, PlayerCaller)
    lines = caller.predictor._state_for(AsOfView(con, at), date(2024, 11, 3)).lines
    assert lines.espn.unlinked_games == 0 and lines.espn.games == 1
    plain = make_caller(con, "player", TEAM_S, _settings())
    lines = plain.predictor._state_for(AsOfView(con, at), date(2024, 11, 3)).lines
    assert lines.espn.unlinked_games == 1


def test_make_caller(con):
    team = make_caller(con, "team", TEAM_S, None)
    assert isinstance(team, TeamCaller)
    assert isinstance(make_caller(con, "player", TEAM_S, _settings()), PlayerCaller)
    with pytest.raises(LiveModelError) as exc:
        make_caller(con, "player", TEAM_S, None)
    assert str(exc.value).endswith(FIX)
    with pytest.raises(ValueError):
        make_caller(con, "both", TEAM_S, None)


def test_player_caller_refuses_correction(con):
    corrected = _settings(correction={"training_seasons": ["2023-24"],
                                      "params": dict(ps_mod.GBM_PARAMS)})
    with pytest.raises(LiveModelError):
        PlayerCaller(con, corrected)

