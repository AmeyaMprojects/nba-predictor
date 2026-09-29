"""`predictor backtest --model stage1` end to end, on a fixture archive.

Also closes harness open item 3: the BEATS / TOO CLOSE TO CALL verdict
branch had never run from the CLI, because both shipped baselines put every
game on the home side. This drives it with a predictor that disagrees.
"""

from datetime import date, timedelta

import pytest
from typer.testing import CliRunner

from model_fixtures import add_game
from predictor import cli, config, db
from predictor.backtest import replay
from predictor.config import Settings
from predictor.model import settings as ms
from predictor.model import stage1
from predictor.model.adjustments import Coefficients
from predictor.model.ratings import RatingParams
from real_archive import open_real_archive_or_skip

runner = CliRunner()

SETTINGS = ms.ModelSettings(
    ratings=RatingParams(0.1, 20.0, 0.5, 100),
    coefficients=Coefficients(-1.0, -0.5, -0.3, -0.2, 1.0),
    sigma=13.0, fit_games=1, calibrate_games=1,
)


def _archive(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(config, "settings", s)
    monkeypatch.setattr(db, "settings", s)
    con = db.connect()
    db.migrate(con)
    # warm-up game, then a test-season run where the AWAY team always wins
    add_game(con, "0021800001", "2018-19", date(2019, 3, 1), "PHI", "NYK", 120, 100,
             city="Philadelphia")
    for i in range(40):
        d = date(2023, 11, 1) + timedelta(days=2 * i)
        add_game(con, f"00223{i:05d}", "2023-24", d, "NYK", "PHI", 90, 110, city="New York")
    con.close()
    return s


def test_stage1_headline_is_test_seasons_only(tmp_path, monkeypatch):
    _archive(tmp_path, monkeypatch)
    monkeypatch.setattr(ms, "load", lambda path=None: SETTINGS)
    out = runner.invoke(cli.app, ["backtest", "--model", "stage1"])
    assert out.exit_code == 0, out.output
    # t7-fix1 finding 2: this fixture only ingests ONE test season
    # (2023-24), so the header must name only that season -- the old
    # hard-coded "test seasons 2023-24, 2024-25, 2025-26 only" was wrong
    # for this exact archive.
    assert "test season 2023-24 only" in out.output
    # t7-fix1 finding 1: the games-scored line must state the test-season
    # count on its own, separately from the pooled total across every
    # replayed season (this fixture: 1 warm-up game + 40 test games).
    assert "40 test-season games" in out.output
    # t7-fix2 item A: the default run (no --season) replays the warm-up
    # season too, so the pooled total (41) genuinely exceeds the
    # test-season count (40) -- the "including warm-up, fit and calibrate
    # seasons" clause is TRUE here and must still be printed.
    assert (
        "40 test-season games (41 replayed in total, including warm-up, "
        "fit and calibrate seasons)" in out.output
    )
    assert "By season" in out.output
    assert "2018-19  warm-up" in out.output
    assert "Example explanations" in out.output
    assert " at " in out.output and "% to win)" in out.output
    # Final review (minor): the header names the settings this run actually
    # used -- SETTINGS above is RatingParams(k=0.1, margin_cap=20.0,
    # season_regression=0.5, hca_window=100), sigma=13.0.
    assert (
        "Settings            : k 0.1, cap 20, regression 0.5, window 100, "
        "sigma 13.00 (src/predictor/model/stage1_settings.json)" in out.output
    )
    # Final review (minor): the season-table note names all four non-test
    # roles explicitly (not just "earlier rows"), so it stays true even when
    # an 'unassigned' season appears in the table.
    assert (
        "Only 'test' rows are the published held-out test; 'warm-up', "
        "'fit' and 'calibrate' rows are seasons the model learned from or "
        "was tuned on; 'unassigned' rows are outside the published test."
    ) in out.output
    # Final review: the publishing bar block prints for a scoped (stage1)
    # run.
    assert (
        "Publishing bar (from the design spec; 'a few points' read as 5 "
        "percentage points, in buckets of 50+ games -- a reading fixed after "
        "the first test run):"
    ) in out.output


def _multi_test_season_archive(tmp_path, monkeypatch):
    """Like `_archive`, but ingests TWO test seasons (2023-24, 2024-25) --
    enough to tell a dynamically-built scope label apart from one hard-coded
    to all three TEST_SEASONS (t7-fix1 finding 2)."""
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(config, "settings", s)
    monkeypatch.setattr(db, "settings", s)
    con = db.connect()
    db.migrate(con)
    add_game(con, "0021800001", "2018-19", date(2019, 3, 1), "PHI", "NYK", 120, 100,
             city="Philadelphia")
    for i in range(10):
        d = date(2023, 11, 1) + timedelta(days=2 * i)
        add_game(con, f"00223{i:05d}", "2023-24", d, "NYK", "PHI", 90, 110, city="New York")
    for i in range(10):
        d = date(2024, 11, 1) + timedelta(days=2 * i)
        add_game(con, f"00224{i:05d}", "2024-25", d, "NYK", "PHI", 90, 110, city="New York")
    con.close()
    return s


def test_scope_header_lists_only_the_test_seasons_actually_present(tmp_path, monkeypatch):
    """t7-fix1 finding 2: the scope label is built from the seasons in the
    headline, not hard-coded -- this archive has no 2025-26 at all, so the
    header must not claim it does."""
    _multi_test_season_archive(tmp_path, monkeypatch)
    monkeypatch.setattr(ms, "load", lambda path=None: SETTINGS)
    out = runner.invoke(cli.app, ["backtest", "--model", "stage1"])
    assert out.exit_code == 0, out.output
    assert "test seasons 2023-24, 2024-25 only" in out.output
    assert "2025-26" not in out.output


def test_season_filter_narrows_the_header_to_just_that_test_season(tmp_path, monkeypatch):
    """t7-fix1 finding 2: `--season <test season>` must name only that one
    season in the header, not every test season in the archive."""
    _multi_test_season_archive(tmp_path, monkeypatch)
    monkeypatch.setattr(ms, "load", lambda path=None: SETTINGS)
    out = runner.invoke(cli.app, ["backtest", "--model", "stage1", "--season", "2023-24"])
    assert out.exit_code == 0, out.output
    lines = out.output.splitlines()
    season_line = next(line for line in lines if line.strip().startswith("Season"))
    assert season_line.strip() == (
        "Season              : test season 2023-24 only -- the model's "
        "settings were never tuned on it"
    )
    assert "2024-25" not in out.output
    # t7-fix2 item A: `--season 2023-24` makes `replay.replay` walk ONLY
    # that season, so the pooled total equals the test-season count exactly
    # -- nothing besides the headline's own games was replayed, so the
    # "including warm-up, fit and calibrate seasons" clause would be false
    # here and must not be printed.
    games_scored_line = next(
        line for line in lines if line.strip().startswith("games scored")
    )
    assert games_scored_line.strip() == "games scored        : 10 test-season games"
    assert "including" not in out.output


def test_progress_prints_to_stderr_never_stdout(tmp_path, monkeypatch):
    """t7-fix1 finding 5: the real backtest runs silently for ~7.5 minutes.
    A progress line per season must go to STDERR -- STDOUT is the
    publishable report and must stay clean."""
    _archive(tmp_path, monkeypatch)
    monkeypatch.setattr(ms, "load", lambda path=None: SETTINGS)
    out = runner.invoke(cli.app, ["backtest", "--model", "stage1"])
    assert out.exit_code == 0, out.output
    # Final review (minor): the "(n of N seasons)" total is gone -- N used
    # to come from `replay.known_seasons`, a count of seasons IN THE
    # ARCHIVE, which can differ from the seasons `replay.replay` actually
    # walks and predicts at least one game in, so the total could get stuck
    # below N forever (e.g. "12 of 13") even after the run finished.
    assert "scoring 2018-19 (1 so far)..." in out.stderr
    assert "scoring 2023-24 (2 so far)..." in out.stderr
    assert "seasons)" not in out.stderr
    assert "scoring" not in out.stdout


def test_non_test_season_filter_names_the_season_flag_as_the_cause(tmp_path, monkeypatch):
    """t7-fix1 finding 2: 2018-19 IS ingested (this fixture's warm-up game)
    -- filtering to it excludes every test-season game, and the message
    must say --season did that, not tell the user to ingest data that is
    already there."""
    _archive(tmp_path, monkeypatch)
    monkeypatch.setattr(ms, "load", lambda path=None: SETTINGS)
    out = runner.invoke(cli.app, ["backtest", "--model", "stage1", "--season", "2018-19"])
    assert out.exit_code == 1
    assert "--season 2018-19" in out.output


def test_verdict_branch_runs_from_the_cli(tmp_path, monkeypatch):
    _archive(tmp_path, monkeypatch)
    monkeypatch.setattr(ms, "load", lambda path=None: SETTINGS)

    class AwayPicker(stage1.Stage1Predictor):
        def __call__(self, game, view):
            super().__call__(game, view)
            return 0.2  # always picks the away team, which always wins here

    monkeypatch.setattr(stage1, "Stage1Predictor", AwayPicker)
    out = runner.invoke(cli.app, ["backtest", "--model", "stage1"])
    assert out.exit_code == 0, out.output
    assert "BEATS always-pick-home" in out.output


def test_missing_settings_is_a_plain_error(tmp_path, monkeypatch):
    _archive(tmp_path, monkeypatch)

    def missing(path=None):
        raise ms.SettingsError("No fitted model settings found at x. Run: predictor fit-model")

    monkeypatch.setattr(ms, "load", missing)
    out = runner.invoke(cli.app, ["backtest", "--model", "stage1"])
    assert out.exit_code == 1
    assert "predictor fit-model" in out.output
    assert "Traceback" not in out.output


def test_no_test_season_games_is_a_plain_error(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(config, "settings", s)
    monkeypatch.setattr(db, "settings", s)
    con = db.connect()
    db.migrate(con)
    add_game(con, "0021800001", "2018-19", date(2019, 3, 1), "PHI", "NYK", 120, 100,
             city="Philadelphia")
    con.close()
    monkeypatch.setattr(ms, "load", lambda path=None: SETTINGS)
    out = runner.invoke(cli.app, ["backtest", "--model", "stage1"])
    assert out.exit_code == 1
    assert "No test-season games" in out.output


@pytest.fixture(scope="module")
def real_archive_test_season_replays():
    """t7-fix1 finding 9: one Stage1Predictor, replayed once per test season
    IN CHRONOLOGICAL ORDER (2023-24, then 2024-25, then 2025-26), instead of
    three independent fresh predictors each rebuilding its ratings from the
    entire 2014-15..calibrate history from scratch.

    Reusing the same predictor across the three ordered calls produces
    IDENTICAL ratings state to three fresh predictors -- `_catch_up` never
    rewinds when `view.as_of` only ever moves forward, so this is exactly
    the same incremental-vs-rebuild equivalence the model's own tests
    (test_incremental_calls_equal_a_fresh_predictor_when_a_result_arrives_late,
    etc.) already establish -- it just does the expensive full-history
    rebuild ONCE instead of three times.
    """
    con = open_real_archive_or_skip()
    try:
        predictor = stage1.Stage1Predictor(con, ms.load())
        replays = {}
        for season in ms.TEST_SEASONS:
            replays[season] = replay.replay(con, predictor, season=season)
        yield predictor, replays
    finally:
        con.close()


@pytest.mark.parametrize("season", list(ms.TEST_SEASONS))
def test_every_test_season_game_is_predicted_and_every_explanation_adds_up(
    real_archive_test_season_replays, season
):
    predictor, replays = real_archive_test_season_replays
    preds, stats = replays[season]
    assert stats.considered == 1230
    assert stats.predicted == 1230, stats
    for p in preds:
        b = predictor.breakdowns[p.game_id]
        assert sum(v for _, v in b.terms()) == pytest.approx(b.spread, abs=1e-9)
        assert p.p_home == b.p_home
        shown = [round(v, 1) for _, v in b.terms()]
        assert f"by {abs(round(sum(shown), 1)):.1f}" in b.sentence() or "pick'em" in b.sentence()
