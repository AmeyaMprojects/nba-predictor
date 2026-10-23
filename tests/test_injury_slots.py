from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from typer.testing import CliRunner

from predictor import cli, config, db, raw_store
from predictor.config import Settings
from predictor.sources import injury_report, injury_slots

ET = ZoneInfo("America/New_York")
FIXTURE_BYTES = (
    Path(__file__).parent / "fixtures" / "Injury-Report_2025-01-15_05PM.pdf"
).read_bytes()


@pytest.fixture
def env(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(raw_store, "settings", s)
    con = db.connect(tmp_path / "t.duckdb")
    db.migrate(con)
    yield con
    con.close()


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    monkeypatch.setattr(injury_report, "_sleep", lambda seconds: None)
    monkeypatch.setattr(injury_slots, "_sleep", lambda seconds: None)


class FakeResponse:
    def __init__(self, status_code, content=b""):
        self.status_code = status_code
        self.content = content
        self.headers = {}


class MappingSession:
    def __init__(self, mapping, default_status=403):
        self.mapping = mapping
        self.default_status = default_status
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        if url in self.mapping:
            status, content = self.mapping[url]
            return FakeResponse(status, content)
        return FakeResponse(self.default_status)


def _et(*args):
    return datetime(*args, tzinfo=ET)


# --- call times -----------------------------------------------------------


def test_call_times_final_is_tip_minus_45():
    early, final = injury_slots.call_times(
        date(2024, 1, 15), datetime(2024, 1, 16, 0, 0, tzinfo=UTC)
    )
    assert final == datetime(2024, 1, 15, 23, 15, tzinfo=UTC)
    assert early == datetime(2024, 1, 15, 12, 30, tzinfo=UTC)


def test_early_call_clipped_to_final():
    early, final = injury_slots.call_times(
        date(2024, 1, 15), datetime(2024, 1, 15, 13, 0, tzinfo=UTC)
    )
    assert final == datetime(2024, 1, 15, 12, 15, tzinfo=UTC)
    assert early == final


# --- slot labels ----------------------------------------------------------


def test_slot_label_switches_pattern_on_2025_12_22():
    assert injury_slots.slot_label(_et(2025, 12, 21, 19, 0)) == "07PM"
    assert injury_slots.slot_label(_et(2025, 12, 22, 19, 15)) == "07_15PM"
    assert injury_report.report_url(date(2025, 12, 22), "07_15PM").endswith(
        "Injury-Report_2025-12-22_07_15PM.pdf"
    )


def test_slot_label_midnight_and_noon():
    assert injury_slots.slot_label(_et(2024, 1, 16, 0, 0)) == "12AM"
    assert injury_slots.slot_label(_et(2024, 1, 16, 12, 0)) == "12PM"
    assert injury_slots.slot_label(_et(2026, 1, 16, 0, 45)) == "12_45AM"


def test_candidate_slots_old_pattern_walks_back_hourly():
    target = _et(2024, 1, 15, 18, 15).astimezone(UTC)
    assert injury_slots.candidate_slots(target) == [
        (date(2024, 1, 15), "06PM"),
        (date(2024, 1, 15), "05PM"),
        (date(2024, 1, 15), "04PM"),
    ]


def test_candidate_slots_new_pattern_walks_back_quarter_hours():
    target = _et(2026, 1, 10, 18, 20).astimezone(UTC)
    slots = injury_slots.candidate_slots(target)
    assert slots[0] == (date(2026, 1, 10), "06_15PM")
    assert len(slots) == 12
    assert slots[-1] == (date(2026, 1, 10), "03_30PM")


def test_candidate_slots_cross_midnight_et():
    target = _et(2024, 1, 16, 0, 40).astimezone(UTC)
    slots = injury_slots.candidate_slots(target)
    assert slots[0] == (date(2024, 1, 16), "12AM")
    assert slots[1] == (date(2024, 1, 15), "11PM")


def test_candidate_slots_across_the_pattern_switch():
    # 00:30 ET on the first new-pattern day: the new day's quarter hours,
    # then the old day's hourly slots only.
    target = _et(2025, 12, 22, 0, 30).astimezone(UTC)
    slots = injury_slots.candidate_slots(target)
    assert slots[:3] == [
        (date(2025, 12, 22), "12_30AM"),
        (date(2025, 12, 22), "12_15AM"),
        (date(2025, 12, 22), "12_00AM"),
    ]
    assert slots[3:] == [
        (date(2025, 12, 21), "11PM"),
        (date(2025, 12, 21), "10PM"),
    ]


def test_candidate_slots_exact_slot_time_is_included():
    target = _et(2024, 1, 15, 18, 0).astimezone(UTC)
    slots = injury_slots.candidate_slots(target)
    assert slots == [
        (date(2024, 1, 15), "06PM"),
        (date(2024, 1, 15), "05PM"),
        (date(2024, 1, 15), "04PM"),
        (date(2024, 1, 15), "03PM"),
    ]


# --- backfill -------------------------------------------------------------


def _target(at, kind="final", game_id="0022400001"):
    return injury_slots.SlotTarget(game_id=game_id, kind=kind, at=at)


def test_backfill_uses_header_time_not_filename(env):
    url = injury_report.report_url(date(2025, 1, 15), "05PM")
    session = MappingSession({url: (200, FIXTURE_BYTES)})

    early = datetime(2025, 1, 15, 22, 15, tzinfo=UTC)
    stats = injury_slots.backfill_slots(env, [_target(early)], delay=0, session=session)
    assert stats["fetched"] == 1
    assert stats["satisfied"] == 0
    assert stats["unsatisfied"] == 1
    # It walked on past the 05PM file to the earlier slots.
    assert len(session.calls) == 3

    late = datetime(2025, 1, 15, 22, 45, tzinfo=UTC)
    stats = injury_slots.backfill_slots(env, [_target(late)], delay=0, session=session)
    assert stats["satisfied"] == 1
    assert stats["unsatisfied"] == 0
    assert stats["already_archived"] == 1
    assert len(session.calls) == 3, "already-archived report was re-fetched"

    table = db.POINT_IN_TIME_TABLES["injury_status"]
    assert env.execute(f"SELECT count(*) FROM {table}").fetchone()[0] > 0


def test_backfill_tolerates_holes(env):
    session = MappingSession({}, default_status=403)
    at = _et(2024, 1, 15, 18, 0).astimezone(UTC)
    stats = injury_slots.backfill_slots(env, [_target(at)], delay=0, session=session)
    assert stats["targets"] == 1
    assert stats["not_published"] == 4
    assert stats["unsatisfied"] == 1
    assert stats["satisfied"] == 0
    assert stats["transient"] == 0


def test_backfill_transient_counted_and_retried_on_rerun(env):
    url = injury_report.report_url(date(2024, 1, 15), "06PM")
    session = MappingSession({url: (503, b"")}, default_status=403)
    at = _et(2024, 1, 15, 18, 15).astimezone(UTC)

    stats = injury_slots.backfill_slots(env, [_target(at)], delay=0, session=session)
    assert stats["transient"] == 1
    first = session.calls.count(url)
    assert first == injury_report._MAX_ATTEMPTS

    stats = injury_slots.backfill_slots(env, [_target(at)], delay=0, session=session)
    assert stats["transient"] == 1
    assert session.calls.count(url) == 2 * first


def test_backfill_bad_content_counted(env):
    url = injury_report.report_url(date(2024, 1, 15), "06PM")
    session = MappingSession({url: (200, b"<html>blocked</html>")})
    at = _et(2024, 1, 15, 18, 15).astimezone(UTC)
    stats = injury_slots.backfill_slots(env, [_target(at)], delay=0, session=session)
    assert stats["bad_content"] == 1
    assert stats["not_published"] == 2


def test_backfill_shared_time_is_walked_once(env):
    session = MappingSession({}, default_status=403)
    at = _et(2024, 1, 15, 18, 15).astimezone(UTC)
    stats = injury_slots.backfill_slots(
        env, [_target(at, game_id="a"), _target(at, game_id="b")], delay=0, session=session
    )
    assert stats["targets"] == 1
    assert len(session.calls) == 3


# --- replay targets -------------------------------------------------------


def _add_game(con, game_id, game_date, home, away, tip):
    g = db.POINT_IN_TIME_TABLES["games"]
    s = db.POINT_IN_TIME_TABLES["schedule"]
    obs = datetime(2024, 1, 1, tzinfo=UTC)
    con.execute(
        f"INSERT INTO {g} (game_id, season, game_date, home_team, away_team,"
        " home_points, away_points, status, reconstructed, observed_at)"
        " VALUES (?, '2023-24', ?, ?, ?, 100, 90, 'FINAL', TRUE, ?)",
        [game_id, game_date, home, away, obs],
    )
    con.execute(
        f"INSERT INTO {s} (game_id, season, game_date, tip_off_utc, home_team,"
        " away_team, is_neutral_reported, is_neutral, observed_at)"
        " VALUES (?, '2023-24', ?, ?, ?, ?, FALSE, FALSE, ?)",
        [game_id, game_date, tip, home, away, obs],
    )


def test_replay_targets_dedupe_shared_times(env):
    tip = datetime(2024, 1, 16, 0, 0, tzinfo=UTC)
    _add_game(env, "0022300500", date(2024, 1, 15), "BOS", "NYK", tip)
    _add_game(env, "0022300501", date(2024, 1, 15), "LAL", "GSW", tip)

    targets = injury_slots.replay_targets(env, date(2024, 1, 1), date(2024, 1, 31))
    finals = [t for t in targets if t.kind == "final"]
    earlies = [t for t in targets if t.kind == "early"]
    assert len(finals) == 1
    assert finals[0].at == datetime(2024, 1, 15, 23, 15, tzinfo=UTC)
    assert len(earlies) == 1
    assert earlies[0].at == datetime(2024, 1, 15, 12, 30, tzinfo=UTC)
    assert len({t.at for t in targets}) == len(targets)


def test_replay_targets_skip_preseason_untipped_and_out_of_range(env):
    tip = datetime(2024, 1, 16, 0, 0, tzinfo=UTC)
    _add_game(env, "0012300001", date(2024, 1, 15), "BOS", "NYK", tip)  # preseason
    _add_game(env, "0022300600", date(2024, 2, 15), "LAL", "GSW",
              datetime(2024, 2, 16, 3, 0, tzinfo=UTC))  # out of range
    _add_game(env, "0022300601", date(2024, 1, 20), "MIA", "ORL", None)  # TBD tip

    assert injury_slots.replay_targets(env, date(2024, 1, 1), date(2024, 1, 31)) == []


# --- CLI ------------------------------------------------------------------

runner = CliRunner()


def _stats(**overrides):
    stats = {k: 0 for k in (
        "targets", "satisfied", "unsatisfied", "fetched", "not_published",
        "transient", "bad_content", "parse_failed", "already_archived",
    )}
    stats.update(overrides)
    return stats


def _cli_env(tmp_path, monkeypatch, stats):
    s = Settings(data_dir=tmp_path)
    s.ensure_dirs()
    monkeypatch.setattr(config, "settings", s)
    monkeypatch.setattr(db, "settings", s)
    seen = {}

    def fake_targets(con, start, end):
        seen["range"] = (start, end)
        return []

    def fake_backfill(con, targets, delay=0.4, session=None):
        seen["delay"] = delay
        return stats

    monkeypatch.setattr(injury_slots, "replay_targets", fake_targets)
    monkeypatch.setattr(injury_slots, "backfill_slots", fake_backfill)
    return seen


def test_cli_exit_codes(tmp_path, monkeypatch):
    seen = _cli_env(tmp_path, monkeypatch, _stats(targets=5, satisfied=3, unsatisfied=2))
    result = runner.invoke(
        cli.app,
        ["backfill-injury-slots", "--start", "2024-01-01", "--end", "2024-01-31",
         "--delay", "0"],
    )
    assert result.exit_code == 0, result.stdout
    assert "unsatisfied: 2" in result.stdout
    assert seen["range"] == (date(2024, 1, 1), date(2024, 1, 31))
    assert seen["delay"] == 0

    _cli_env(tmp_path, monkeypatch, _stats(targets=5, transient=1))
    result = runner.invoke(
        cli.app, ["backfill-injury-slots", "--start", "2024-01-01", "--end", "2024-01-31"]
    )
    assert result.exit_code == 1
    assert "transient: 1" in result.stdout
    assert "RUN DID NOT FULLY SUCCEED" in result.stdout


def test_cli_default_start(tmp_path, monkeypatch):
    seen = _cli_env(tmp_path, monkeypatch, _stats())
    result = runner.invoke(cli.app, ["backfill-injury-slots", "--end", "2024-01-31"])
    assert result.exit_code == 0, result.stdout
    assert seen["range"][0] == date(2020, 12, 22)
    assert seen["delay"] == 0.4
