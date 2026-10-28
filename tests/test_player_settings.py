"""The player-model settings file and ``predictor fit-player-model`` (Task 10).

The CLI tests run on Task 7's small candidate fixture in tmp_path, with the
team grid shrunk, and point both settings files at tmp_path: no test touches
data/, the committed settings files, or the network.
"""

from __future__ import annotations

import dataclasses
import math
import os

import pytest
from typer.testing import CliRunner

from predictor import cli, config, db
from predictor.config import Settings
from predictor.model import candidate as cand
from predictor.model import player_settings as pset
from predictor.model import settings as model_settings
from predictor.model.availability import CALLS, STATUS_PRIORS
from predictor.model.correction import GBM_PARAMS
from predictor.model.player_ratings import PlayerParams
from predictor.model.ratings import FLAT_HOME, HomeParams, RatingParams

NOT_MET = "Switch-over bar not met -- nothing was saved; the current model stays live."


def _settings(**overrides) -> pset.PlayerModelSettings:
    names = cand.feature_names("box")
    values = dict(
        team=RatingParams(k=0.05, margin_cap=20.0, season_regression=0.33, hca_window=400),
        player=PlayerParams(half_life_minutes=1500.0, season_regression=0.4),
        formula="box",
        w=0.25,
        coefficients={n: 0.1 * (i + 1) for i, n in enumerate(names)},
        sigma=13.25,
        half_life=2.0,
        availability={
            "early": {**STATUS_PRIORS, "Questionable": 0.61},
            "final": {**STATUS_PRIORS, "Probable": 0.93},
        },
        correction={"training_seasons": ["2022-23", "2023-24"], "params": dict(GBM_PARAMS)},
        tuning_games=5432,
    )
    values.update(overrides)
    return pset.PlayerModelSettings(**values)


def test_settings_round_trip():
    s = _settings()
    assert pset.from_json(pset.to_json(s)) == s
    plain = _settings(correction=None, half_life=None, formula="both", coefficients={
        n: -1.5 + i for i, n in enumerate(cand.feature_names("both"))
    })
    assert pset.from_json(pset.to_json(plain)) == plain
    assert plain.coefficient_tuple() == tuple(
        -1.5 + i for i in range(len(cand.feature_names("both")))
    )


def test_settings_rejects_wrong_version():
    for version in ('"version": 3', '"version": 0', '"version": "2"'):
        text = pset.to_json(_settings()).replace('"version": 2', version)
        with pytest.raises(pset.SettingsError) as exc:
            pset.from_json(text)
        assert str(exc.value).endswith("Run: predictor fit-player-model")


def test_settings_v1_file_loads_with_flat_home():
    """The committed (plan 1, version 1) file has no "home" key and loads as
    flat home court."""
    import json

    text = pset.SETTINGS_PATH.read_text()
    assert json.loads(text)["version"] == 1
    s = pset.from_json(text)
    assert s.home.flat is True
    assert s.home == FLAT_HOME


def test_settings_v1_with_home_key_is_rejected():
    import json

    doc = json.loads(pset.to_json(_settings(home=HomeParams(150.0, 0.66))))
    doc["version"] = 1
    with pytest.raises(pset.SettingsError):
        pset.from_json(json.dumps(doc))


def test_settings_v2_round_trip_with_inf():
    import json

    s = _settings(home=HomeParams(math.inf, 0.66))
    text = pset.to_json(s)
    doc = json.loads(text)
    assert doc["version"] == 2
    assert doc["home"] == {"shrinkage": None, "altitude_allowance": 0.66, "window": 164}
    assert '"shrinkage": null' in text
    assert pset.from_json(text) == s
    team = _settings(home=HomeParams(150.0, 0.658231, window=100))
    assert json.loads(pset.to_json(team))["home"] == {
        "shrinkage": 150.0, "altitude_allowance": 0.658231, "window": 100}
    assert pset.from_json(pset.to_json(team)) == team
    # Flat is written too: "home" is always present in version 2.
    assert json.loads(pset.to_json(_settings()))["home"] == {
        "shrinkage": None, "altitude_allowance": 0.0, "window": 164}


def test_settings_from_candidate_copies_final_home(monkeypatch):
    """``settings_from_candidate`` copies the final choice's home court."""
    from predictor.model import player_stage1 as ps1

    names = cand.feature_names("box")
    final = cand.CandidateChoice(
        team=RatingParams(0.05, 20.0, 0.33, 400), player=PlayerParams(1500.0, 0.4),
        formula="box", w=0.25, coefficients=tuple(0.1 for _ in names),
        feature_names=names, sigma=12.0, games=10, home=HomeParams(150.0, 0.66),
    )
    winner = dataclasses.replace(_StubResult(), half_life=None)
    cev = _StubCev(final=final, winner=winner, listings=[])
    s = ps1.settings_from_candidate(None, cev, None)
    assert s.home == HomeParams(150.0, 0.66)


@dataclasses.dataclass(frozen=True)
class _StubResult:
    half_life: float | None = None


@dataclasses.dataclass(frozen=True)
class _StubCev:
    final: object
    winner: object
    listings: list


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(sigma=-1.0),
    lambda d: d.update(w=1.5),
    lambda d: d.update(formula="elo"),
    lambda d: d["coefficients"].pop("roster_box"),
    lambda d: d["availability"].pop("early"),
    lambda d: d["availability"]["final"].update(Out=1.7),
    lambda d: d["correction"]["params"].update(max_depth=9),
    lambda d: d.pop("team"),
    lambda d: d["team"].update(k=float("nan")),
    lambda d: d["team"].update(k=-0.1),
    lambda d: d["team"].update(margin_cap=0.0),
    lambda d: d["team"].update(margin_cap=float("inf")),
    lambda d: d["team"].update(season_regression=1.5),
    lambda d: d["team"].update(season_regression=float("nan")),
    lambda d: d.pop("home"),
    lambda d: d.update(home=None),
    lambda d: d["home"].update(shrinkage=-1.0),
    lambda d: d["home"].update(shrinkage="150"),
    lambda d: d["home"].update(altitude_allowance=None),
    lambda d: d["home"].update(window=164.5),
    lambda d: d["home"].update(window=True),
    lambda d: d["home"].update(window=0),
    lambda d: d["home"].pop("window"),
])
def test_settings_rejects_bad_files(mutate):
    import json

    doc = json.loads(pset.to_json(_settings()))
    mutate(doc)
    with pytest.raises(pset.SettingsError) as exc:
        pset.from_json(json.dumps(doc))
    assert str(exc.value).endswith("Run: predictor fit-player-model")


def test_load_missing_file(tmp_path):
    with pytest.raises(pset.SettingsError) as exc:
        pset.load(tmp_path / "nope.json")
    assert str(exc.value).endswith("Run: predictor fit-player-model")


def test_save_then_load(tmp_path):
    s = _settings()
    pset.save(s, tmp_path / "p.json")
    assert pset.load(tmp_path / "p.json") == s


# -- the CLI ---------------------------------------------------------------------

def _cli_db(tmp_path, monkeypatch):
    from test_candidate import _build, _shrink_team_grid

    _shrink_team_grid(monkeypatch)
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(config, "settings", s)
    monkeypatch.setattr(db, "settings", s)
    con = db.connect()
    db.migrate(con)
    _build(con)
    con.close()
    player_path = tmp_path / "player_settings.json"
    stage1_path = tmp_path / "stage1_settings.json"
    stage1_path.write_text("{}\n")
    os.utime(stage1_path, (1_000_000_000, 1_000_000_000))
    monkeypatch.setattr(pset, "SETTINGS_PATH", player_path)
    monkeypatch.setattr(model_settings, "SETTINGS_PATH", stage1_path)
    return player_path, stage1_path


def _bar(met: bool) -> cand.SwitchOverBar:
    return cand.SwitchOverBar(
        pooled_brier=(0.23, 0.22 if met else 0.24),
        pooled_log_loss=(0.65, 0.64 if met else 0.66),
        pooled_accuracy=(0.64, 0.65),
        seasons_better=5 if met else 2,
        seasons=6,
        worst_bucket_gap=1.0,
        calibration_met=True,
        includes_terms=True,
    )


def test_fit_player_model_writes_nothing_when_bar_fails(tmp_path, monkeypatch):
    player_path, stage1_path = _cli_db(tmp_path, monkeypatch)
    monkeypatch.setattr(cand, "switch_over_bar", lambda *a, **k: _bar(False))
    result = CliRunner().invoke(cli.app, ["fit-player-model"])
    assert result.exit_code == 0, result.output
    assert NOT_MET in result.stdout
    for reason in _bar(False).reasons():
        assert reason in result.stdout
    assert not player_path.exists()
    assert stage1_path.stat().st_mtime == 1_000_000_000
    assert stage1_path.read_text() == "{}\n"


def test_fit_player_model_saves_when_bar_met(tmp_path, monkeypatch):
    player_path, stage1_path = _cli_db(tmp_path, monkeypatch)
    monkeypatch.setattr(cand, "switch_over_bar", lambda *a, **k: _bar(True))
    result = CliRunner().invoke(cli.app, ["fit-player-model"])
    assert result.exit_code == 0, result.output
    assert str(player_path) in result.stdout
    assert NOT_MET not in result.stdout
    s = pset.load(player_path)
    assert s.formula in cand.FORMULAS
    assert set(s.availability) == set(CALLS)
    assert s.tuning_games > 0
    assert stage1_path.stat().st_mtime == 1_000_000_000


def test_fit_player_model_bar_uses_per_team_variant(tmp_path, monkeypatch):
    """The bar compares the current live model with the per-team variant's
    walk-forward, and the saved settings carry the per-team final choice."""
    from predictor.model import correction

    player_path, _ = _cli_db(tmp_path, monkeypatch)
    seen: dict = {}
    real_variants = cand.evaluate_home_variants
    real_evaluate = cli._run_evaluation

    def variants_spy(*args, **kwargs):
        seen["variants"] = real_variants(*args, **kwargs)
        return seen["variants"]

    def evaluate_spy(*args, **kwargs):
        seen["ev"] = real_evaluate(*args, **kwargs)
        return seen["ev"]

    def bar_spy(current, candidate, choice):
        seen["bar"] = (current, candidate, choice)
        return _bar(True)

    monkeypatch.setattr(cand, "evaluate_home_variants", variants_spy)
    monkeypatch.setattr(cli, "_run_evaluation", evaluate_spy)
    monkeypatch.setattr(cand, "switch_over_bar", bar_spy)
    monkeypatch.setattr(correction, "correction_helps", lambda *a, **k: False)
    result = CliRunner().invoke(cli.app, ["fit-player-model"])
    assert result.exit_code == 0, result.output
    variants = seen["variants"]
    current, candidate, choice = seen["bar"]
    assert current is seen["ev"].winner.predictions
    assert candidate is variants.per_team.winner.final_call.predictions
    assert candidate is not variants.flat.winner.final_call.predictions
    assert choice is variants.per_team.final
    saved = pset.load(player_path)
    assert saved.home == variants.per_team.final.home
    assert saved.team == variants.per_team.final.team
    assert saved.coefficient_tuple() == variants.per_team.final.coefficients


def test_fit_player_model_without_player_logs(tmp_path, monkeypatch):
    from model_fixtures import build_history

    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(config, "settings", s)
    monkeypatch.setattr(db, "settings", s)
    con = db.connect()
    db.migrate(con)
    build_history(con)
    con.close()
    player_path = tmp_path / "player_settings.json"
    monkeypatch.setattr(pset, "SETTINGS_PATH", player_path)
    result = CliRunner().invoke(cli.app, ["fit-player-model"])
    assert result.exit_code == 1
    assert "No player game logs" in result.stdout
    assert "Traceback" not in result.stdout
    assert not player_path.exists()


def test_settings_from_candidate_uses_final_choice(tmp_path, monkeypatch):
    from model_fixtures import fixture_con
    from test_candidate import _build, _shrink_team_grid

    from predictor.model import evaluate as evaluate_mod
    from predictor.model import player_stage1 as ps1

    _shrink_team_grid(monkeypatch)
    con = fixture_con(tmp_path)
    _build(con)
    ev = evaluate_mod.evaluate(con)
    cev = cand.evaluate_candidate(con, ev)
    s = ps1.settings_from_candidate(con, cev, None)
    con.close()
    f = cev.final
    assert s.team == f.team == ev.final.ratings
    assert s.player == f.player and s.formula == f.formula and s.w == f.w
    assert s.coefficient_tuple() == f.coefficients
    assert s.sigma == f.sigma and s.tuning_games == f.games
    assert s.half_life == cev.winner.half_life
    assert s.correction is None
    # Fewer than 100 listings per status in the fixture: every probability
    # keeps its prior.
    assert s.availability == {c: dict(STATUS_PRIORS) for c in CALLS}
    assert dataclasses.replace(s) == pset.from_json(pset.to_json(s))


def test_settings_from_candidate_with_correction(tmp_path, monkeypatch):
    """An included correction records the walk-forward seasons and this
    code's parameters, re-chooses sigma out of fold, and a predictor built
    from the saved file retrains it."""
    from model_fixtures import fixture_con
    from test_candidate import _build, _shrink_team_grid

    from predictor.model import evaluate as evaluate_mod
    from predictor.model import player_stage1 as ps1

    _shrink_team_grid(monkeypatch)
    con = fixture_con(tmp_path)
    _build(con)
    ev = evaluate_mod.evaluate(con)
    cev = cand.evaluate_candidate(con, ev)
    s = ps1.settings_from_candidate(con, cev, cev.winner)
    seasons = sorted({p.season for p in cev.winner.final_call.predictions})
    assert s.correction == {"training_seasons": seasons, "params": dict(GBM_PARAMS)}
    rows = ps1.correction_rows(con, s)
    assert sorted(rows) == seasons
    assert s.sigma == ps1.correction_sigma(rows)
    loaded = pset.from_json(pset.to_json(s))
    assert loaded == s
    predictor = ps1.PlayerStage1Predictor(con, loaded)
    assert predictor.correction is not None
    assert predictor.correction.seasons == tuple(seasons)
    con.close()


def test_load_unreadable_file(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_bytes(b"\xff\xfe\x00garbage")
    with pytest.raises(pset.SettingsError) as exc:
        pset.load(bad)
    assert str(exc.value).endswith("Run: predictor fit-player-model")
    folder = tmp_path / "folder.json"
    folder.mkdir()
    with pytest.raises(pset.SettingsError) as exc:
        pset.load(folder)
    assert str(exc.value).endswith("Run: predictor fit-player-model")
