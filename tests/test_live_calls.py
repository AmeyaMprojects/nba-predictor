"""Early, revision and final calls in ``live.predict_today``, and grading of
the last call before tip (plan 3, Task 7).

The model is faked: ``live_model.make_caller`` returns a caller whose p
comes from a table keyed by (latest report time "HH:MM" or None, call
kind). Every time is on slate 2026-11-05 unless stated."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta

import pytest

from model_fixtures import fixture_con
from predictor import db
from predictor.model import live, live_model, market
from predictor.model.availability import InjuryHistory
from predictor.model.calls import ALL_CALLS, CALL_CUTOFF, EARLY_ONLY
from predictor.model.live import (
    _MISSED_FINAL_REASON,
    append_id_links,
    calls_due,
    grade,
    grades_path,
    latest_report_at,
    log_path,
    predict_today,
    read_log,
)
from predictor.model.live_model import (
    CallOutcome,
    LiveModel,
    LiveModelError,
    player_settings_dict,
)
from schedule_rows import insert_schedule_row
from test_live_predict import S, SEASON, _insert_final, _insert_odds
from test_player_stage1 import _settings as _player_settings

D = date(2026, 11, 5)
PS = _player_settings()
V2 = LiveModel("player", D)
GID = "0022600500"
MURRAY = ("DEN", "Murray, Jamal")
FILLER = ("BOS", "Filler, Al")  # keeps every report listing the date


def at(hhmm: str, day: int = 5) -> datetime:
    h, m = map(int, hhmm.split(":"))
    return datetime(2026, 11, day, h, m, tzinfo=UTC)


class FakeCaller:
    def __init__(self, env, kind):
        self.env = env
        self.kind = kind

    def call(self, game, view, call_kind):
        rep = latest_report_at(view.con, view.as_of)
        key = rep.strftime("%H:%M") if rep is not None else None
        p = self.env.p.get((key, call_kind), self.env.p.get((key, None)))
        if p is None:
            p = self.env.p_fn(rep, call_kind) if self.env.p_fn else self.env.default
        availability = None
        if self.kind == "player":
            snap = InjuryHistory.from_view(view).snapshot_at(game.game_date, view.as_of)
            statuses = snap.statuses if snap is not None else {}
            availability = [
                {"team": t, "player": n, "status": s}
                for (t, n), s in sorted(statuses.items())
                if t in (game.home_team, game.away_team)
            ]
        return CallOutcome(
            spread=(p - 0.5) * 20, p_home=p, sentence=f"fake {call_kind}",
            terms={"rating": 1.0}, availability=availability,
            model_version="v2" if self.kind == "player" else "v1",
            settings={"fake": self.kind}, sigma=13.0,
        )


class Env:
    def __init__(self, tmp_path, monkeypatch, *, live=None, default=0.61):
        self.con = fixture_con(tmp_path)
        self.repo = tmp_path / "repo"
        self.p: dict = {}
        self.p_fn = None
        self.default = default
        self.live = live or LiveModel.team()
        self.made: list[tuple[str, object]] = []

        def fake_make_caller(con, kind, settings, player_settings, *, id_links=None):
            self.made.append((kind, id_links))
            if kind == "player":
                assert player_settings is not None
            return FakeCaller(self, kind)

        monkeypatch.setattr(live_model, "make_caller", fake_make_caller)

    def game(self, gid=GID, home="LAL", away="DEN", tip=None, game_date=D, observed_at=None):
        insert_schedule_row(self.con, gid, game_date, home, away, tip or at("00:00", 6),
                            observed_at=observed_at, season=SEASON)

    def report(self, when: datetime, rows=(), game_date=D):
        table = db.POINT_IN_TIME_TABLES["injury_status"]
        for team, player, status in [(*FILLER, "Out"), *rows]:
            self.con.execute(
                f"INSERT INTO {table} (report_date, game_date, team, player, status,"
                " observed_at) VALUES (?,?,?,?,?,?)",
                [when.date(), game_date, team, player, status, when],
            )
        reports = db.POINT_IN_TIME_TABLES["injury_reports"]
        self.con.execute(
            f"INSERT INTO {reports} (observed_at, fetched_at, source_key, row_count,"
            " content_sha) VALUES (?,?,?,?,?)",
            [when, when, f"k{when:%H%M}", 1 + len(rows), "0" * 64],
        )

    def run(self, now, **kw):
        kw.setdefault("calls", ALL_CALLS)
        kw.setdefault("live_model", self.live)
        kw.setdefault("player_settings", PS)
        return predict_today(self.con, S, self.repo, now, code_version=("abc", False), **kw)

    def log(self):
        return read_log(log_path(self.repo, SEASON))

    def grades(self):
        return read_log(grades_path(self.repo, SEASON))


@pytest.fixture
def env(tmp_path, monkeypatch):
    e = Env(tmp_path, monkeypatch)
    yield e
    e.con.close()


@pytest.fixture
def env2(tmp_path, monkeypatch):
    e = Env(tmp_path, monkeypatch, live=V2)
    yield e
    e.con.close()


NEW_FIELDS = {"call_type", "supersedes", "revision_reason", "injury_report_at",
              "availability", "model_version"}


# --- early ------------------------------------------------------------------


def test_early_line_has_new_fields(env):
    env.game()
    (line,) = env.run(at("12:37")).lines_written
    assert NEW_FIELDS <= set(line)
    assert line["status"] == "predicted"
    assert line["call_type"] == "early"
    assert line["supersedes"] is None
    assert line["revision_reason"] is None
    assert line["model_version"] == "v1"
    assert line["injury_report_at"] is None
    assert line["availability"] is None
    assert line["settings"] == {"fake": "team"}


def test_v2_early_line_records_report_and_availability(env2):
    env2.game()
    env2.report(at("12:30"), [(*MURRAY, "Questionable")])
    result = env2.run(at("12:37"))
    (line,) = result.lines_written
    assert line["model_version"] == "v2"
    assert line["injury_report_at"] == "2026-11-05T12:30:00+00:00"
    assert {"team": "DEN", "player": "Murray, Jamal", "status": "Questionable"} in line[
        "availability"]
    assert result.model_versions == frozenset({"v2"})


def test_v2_not_predicted_line_carries_player_settings_and_no_report(env2):
    env2.game(tip=at("12:45"))
    env2.report(at("12:30"))
    (line,) = env2.run(at("12:37")).lines_written
    assert line["status"] == "not_predicted"
    assert line["model_version"] == "v2"
    assert line["injury_report_at"] is None
    assert line["settings"] == player_settings_dict(PS)


def test_mac_default_calls_only_early(env):
    env.game()
    for now in (at("12:37"), at("23:22"), at("23:37"), at("23:52")):
        predict_today(env.con, S, env.repo, now, code_version=("abc", False))
    log = env.log()
    assert len(log) == 1
    assert all(line["call_type"] == "early" for line in log)


# --- final ------------------------------------------------------------------


def test_final_written_once_in_window_even_if_unchanged(env):
    env.game()  # tip 2026-11-06T00:00Z
    (early,) = env.run(at("12:37")).lines_written
    result = env.run(at("23:22"))
    (final,) = result.lines_written
    assert result.finals == 1 and result.predicted == 0 and result.skipped_duplicates == 0
    assert final["call_type"] == "final"
    assert final["status"] == "predicted"
    assert final["p_home"] == early["p_home"]
    assert final["supersedes"] == early["predicted_at"]
    assert final["revision_reason"] is None
    assert final["game_id"] == GID

    again = env.run(at("23:37"))
    assert again.lines_written == [] and again.skipped_duplicates == 1


def test_final_not_written_outside_window(env):
    env.game()
    env.run(at("12:37"))
    assert env.run(at("23:14")).lines_written == []


def test_missed_final_logged_after_cutoff(env):
    env.game()
    (early,) = env.run(at("12:37")).lines_written
    result = env.run(at("23:52"))
    (line,) = result.lines_written
    assert result.missed_finals == 1 and result.not_predicted == 0
    assert line["status"] == "not_predicted"
    assert line["call_type"] == "final"
    assert line["reason"] == _MISSED_FINAL_REASON
    assert line["reason"].startswith("no final call")
    assert line["supersedes"] == early["predicted_at"]
    assert line["p_home"] is None and line["market_p_home"] is None
    assert env.run(at("23:59")).lines_written == []
    assert env.run(at("02:07", 6)).lines_written == []


def test_no_missed_final_for_a_legacy_chain(env):
    # A chain first called before calls existed (no call_type) never had a
    # final to miss.
    env.game()
    legacy = {k: v for k, v in env.run(at("12:37"), calls=EARLY_ONLY).lines_written[0].items()
              if k not in ("call_type", "supersedes", "revision_reason", "injury_report_at",
                           "availability", "model_version")}
    path = log_path(env.repo, SEASON)
    path.write_text(json.dumps(legacy, sort_keys=True) + "\n")
    assert env.run(at("23:52")).lines_written == []
    assert env.run(at("23:59")).lines_written == []


def test_legacy_chain_still_gets_a_final_call_in_its_window(env):
    env.game()
    legacy = {k: v for k, v in env.run(at("12:37"), calls=EARLY_ONLY).lines_written[0].items()
              if k != "call_type"}
    log_path(env.repo, SEASON).write_text(json.dumps(legacy, sort_keys=True) + "\n")
    (final,) = env.run(at("23:22")).lines_written
    assert final["call_type"] == "final" and final["status"] == "predicted"


def test_missed_final_needs_a_predicted_call(env):
    env.game(tip=at("12:45"))
    (too_late,) = env.run(at("12:37")).lines_written
    assert too_late["status"] == "not_predicted"
    assert env.run(at("12:52")).lines_written == []


def test_missed_final_for_a_backfill_day_game(env):
    # A game dated the day before the slate, called but never given a final.
    env.game(gid="0022600499", tip=at("00:00", 5), game_date=date(2026, 11, 4))
    env.run(at("12:37", 4))
    (line,) = env.run(at("12:37")).lines_written
    assert (line["game_id"], line["call_type"], line["status"]) == (
        "0022600499", "final", "not_predicted")


def test_early_and_final_in_one_run_grade_the_final(env):
    # Tip 13:00Z: one 15-minute cutoff for every call, so a 12:37 run makes
    # both the early call and the final call.
    env.game(tip=at("13:00"))
    env.p[(None, "final")] = 0.40
    result = env.run(at("12:37"))
    early, final = result.lines_written
    assert (early["call_type"], final["call_type"]) == ("early", "final")
    assert early["predicted_at"] == final["predicted_at"]
    assert final["supersedes"] == early["predicted_at"]
    _insert_final(env.con, GID, SEASON, D, "LAL", "DEN", 110, 100, observed_at=at("16:00"))
    assert grade(env.con, env.repo, SEASON, at("16:00")) == 1
    (g,) = env.grades()
    assert g["call_type"] == "final"
    assert g["p_home"] == 0.40 and g["correct"] is False
    assert g["early_p_home"] == 0.61 and g["early_correct"] is True
    assert g["early_predicted_at"] == early["predicted_at"]


def test_market_carried_forward_when_no_fresh_odds(env, monkeypatch):
    env.game(home="PHI", away="NYK")
    _insert_odds(env.con, GID, "fanduel", at("12:00"))
    (early,) = env.run(at("12:37")).lines_written
    assert early["market_p_home"] is not None
    monkeypatch.setattr(market, "live_lines", lambda con, game_id, now: [])
    (final,) = env.run(at("23:22")).lines_written
    for field in ("market_p_home", "market_spread", "market_books", "market_observed_at",
                  "market_label"):
        assert final[field] == early[field], field


def test_final_uses_fresh_odds_when_visible(env):
    env.game(home="PHI", away="NYK")
    _insert_odds(env.con, GID, "fanduel", at("12:00"), spread=-5.5)
    (early,) = env.run(at("12:37")).lines_written
    _insert_odds(env.con, GID, "fanduel", at("23:00"), home=-400, away=300, spread=-9.5)
    (final,) = env.run(at("23:22")).lines_written
    assert final["market_spread"] == -9.5
    assert final["market_observed_at"] == at("23:00").isoformat()


# --- revisions --------------------------------------------------------------


def _murray_out(env2):
    """v2, tip 02:00Z: early p 0.61 on the 22:30 report, then Murray Out at
    23:30 and p 0.55, so the 23:37 run revises."""
    env2.game(tip=at("02:00", 6))
    env2.report(at("22:30"), [(*MURRAY, "Questionable")])
    (early,) = env2.run(at("22:37")).lines_written
    env2.report(at("23:30"), [(*MURRAY, "Out")])
    env2.p[("23:30", "revision")] = 0.55
    result = env2.run(at("23:37"))
    return early, result


def test_revision_written_on_new_report_with_two_point_move(env2):
    early, result = _murray_out(env2)
    assert early["injury_report_at"] == "2026-11-05T22:30:00+00:00"
    (rev,) = result.lines_written
    assert result.revisions == 1 and result.skipped_duplicates == 0
    assert rev["call_type"] == "revision"
    assert rev["status"] == "predicted"
    assert rev["p_home"] == 0.55
    assert rev["supersedes"] == early["predicted_at"]
    assert rev["revision_reason"] == "new injury report 05:00 IST: Murray (DEN) out"
    assert rev["injury_report_at"] == "2026-11-05T23:30:00+00:00"
    assert {"team": "DEN", "player": "Murray, Jamal", "status": "Out"} in rev["availability"]
    assert rev["model_version"] == "v2"


def test_no_duplicate_revision_for_same_report(env2):
    _murray_out(env2)
    before = log_path(env2.repo, SEASON).read_bytes()
    env2.p[("23:30", "revision")] = 0.40  # even a bigger move: same report
    assert env2.run(at("23:52")).lines_written == []
    assert log_path(env2.repo, SEASON).read_bytes() == before


def test_unchanged_report_with_new_time_does_not_revise(env2):
    _murray_out(env2)
    env2.report(at("23:45"), [(*MURRAY, "Out")])
    env2.p[("23:45", "revision")] = 0.40
    assert env2.run(at("23:52")).lines_written == []


def test_status_flipping_back_revises_again(env2):
    _, result = _murray_out(env2)
    (first,) = result.lines_written
    env2.report(at("23:45"))  # Murray dropped from the report
    env2.p[("23:45", "revision")] = 0.61
    (second,) = env2.run(at("23:52")).lines_written
    assert second["call_type"] == "revision"
    assert second["supersedes"] == first["predicted_at"]
    assert second["revision_reason"].endswith("Murray (DEN) available")
    assert second["p_home"] == 0.61


def test_flip_back_without_a_two_point_move_against_latest_is_no_revision(env2):
    _murray_out(env2)
    env2.report(at("23:45"))
    # 0.565 is 4 points from the early call but only 1.5 from the latest.
    env2.p[("23:45", "revision")] = 0.565
    assert env2.run(at("23:52")).lines_written == []


def test_small_move_no_revision(env2):
    env2.game(tip=at("02:00", 6))
    env2.report(at("22:30"), [(*MURRAY, "Questionable")])
    env2.run(at("22:37"))
    env2.report(at("23:30"), [(*MURRAY, "Out")])
    env2.p[("23:30", "revision")] = 0.591
    assert env2.run(at("23:37")).lines_written == []


def test_team_model_never_revises(env):
    env.game(tip=at("02:00", 6))
    env.report(at("22:30"), [(*MURRAY, "Questionable")])
    env.run(at("22:37"))
    env.report(at("23:30"), [(*MURRAY, "Out")])
    env.p[("23:30", None)] = 0.40
    assert env.run(at("23:37")).lines_written == []
    # Its final carries no revision reason either: v1 reads no reports.
    (final,) = env.run(at("01:22", 6)).lines_written
    assert final["call_type"] == "final" and final["revision_reason"] is None
    assert final["injury_report_at"] is None and final["availability"] is None


def test_revision_after_final_becomes_graded_call(env2):
    env2.game(tip=at("02:00", 6))
    env2.report(at("22:30"), [(*MURRAY, "Questionable")])
    (early,) = env2.run(at("22:37")).lines_written
    (final,) = env2.run(at("01:22", 6)).lines_written
    assert final["call_type"] == "final" and final["p_home"] == 0.61
    env2.report(at("01:30", 6), [(*MURRAY, "Out")])
    env2.p[("01:30", "revision")] = 0.45
    (rev,) = env2.run(at("01:37", 6)).lines_written
    assert rev["call_type"] == "revision"
    assert rev["supersedes"] == final["predicted_at"]
    assert rev["revision_reason"] == "new injury report 07:00 IST: Murray (DEN) out"

    _insert_final(env2.con, GID, SEASON, D, "LAL", "DEN", 110, 100, observed_at=at("06:00", 6))
    assert grade(env2.con, env2.repo, SEASON, at("06:00", 6)) == 1
    (g,) = env2.grades()
    assert g["call_type"] == "revision"
    assert g["p_home"] == 0.45 and g["correct"] is False
    assert g["early_correct"] is True and g["early_p_home"] == early["p_home"]


def test_late_scratch_after_cutoff_is_not_called(env2):
    env2.game(tip=at("02:00", 6))
    env2.report(at("22:30"), [(*MURRAY, "Questionable")])
    env2.run(at("22:37"))
    env2.run(at("01:22", 6))
    env2.report(at("01:50", 6), [(*MURRAY, "Out")])
    env2.p[("01:50", "revision")] = 0.45
    env2.p[("01:50", "final")] = 0.45
    assert env2.run(at("01:52", 6)).lines_written == []
    assert env2.run(at("02:07", 6)).lines_written == []

    _insert_final(env2.con, GID, SEASON, D, "LAL", "DEN", 110, 100, observed_at=at("06:00", 6))
    grade(env2.con, env2.repo, SEASON, at("06:00", 6))
    (g,) = env2.grades()
    assert g["call_type"] == "final" and g["correct"] is True


def test_final_carries_revision_reason_when_rule_fires(env2):
    env2.game()  # tip 00:00Z
    env2.report(at("12:30"))
    (early,) = env2.run(at("12:37")).lines_written
    env2.report(at("23:00"), [(*MURRAY, "Out")])
    env2.p[("23:00", "final")] = 0.50
    result = env2.run(at("23:22"))
    (final,) = result.lines_written
    assert result.finals == 1 and result.revisions == 0
    assert final["call_type"] == "final"
    assert final["p_home"] == 0.50
    assert final["revision_reason"] == "new injury report 04:30 IST: Murray (DEN) out"
    assert final["injury_report_at"] == "2026-11-05T23:00:00+00:00"
    assert final["supersedes"] == early["predicted_at"]


def test_final_without_a_status_change_has_no_reason_even_if_p_moves(env2):
    env2.game()
    env2.report(at("12:30"))
    env2.run(at("12:37"))
    env2.report(at("23:00"))  # same statuses, newer report
    env2.p[("23:00", "final")] = 0.40
    (final,) = env2.run(at("23:22")).lines_written
    assert final["call_type"] == "final" and final["revision_reason"] is None


# --- postponed games and provisional ids --------------------------------------


def test_postponed_game_gets_no_final_under_old_date(env):
    env.game()
    (early,) = env.run(at("12:37")).lines_written
    # The schedule moves the game to 2026-11-20.
    env.game(tip=datetime(2026, 11, 21, 0, 0, tzinfo=UTC), game_date=date(2026, 11, 20),
             observed_at=at("13:00"))
    for now in (at("23:22"), at("23:37"), at("23:52"), at("02:07", 6)):
        assert env.run(now).lines_written == []
    (fresh,) = env.run(datetime(2026, 11, 20, 12, 37, tzinfo=UTC)).lines_written
    assert fresh["game_date"] == "2026-11-20"
    assert fresh["call_type"] == "early" and fresh["status"] == "predicted"
    assert fresh["supersedes"] is None


def test_provisional_chain_keeps_its_id(env):
    prov, nba = "espn:401705555", "0042600101"
    env.game(gid=prov)
    (early,) = env.run(at("12:37")).lines_written
    assert early["game_id"] == prov
    append_id_links(env.repo, [{"provisional_id": prov, "game_id": nba,
                                "linked_at": at("14:00").isoformat()}])
    env.game(gid=nba, observed_at=at("14:00"))
    result = env.run(at("23:22"))
    (final,) = result.lines_written
    assert final["call_type"] == "final"
    assert final["game_id"] == prov
    assert final["supersedes"] == early["predicted_at"]
    assert env.run(at("23:52")).lines_written == []


# --- the model per game ---------------------------------------------------------


def test_game_keeps_v2_after_flag_back_to_team(env2, monkeypatch):
    env2.game()
    env2.report(at("12:30"))
    env2.run(at("12:37"))
    loaded = []

    def load():
        loaded.append(True)
        return PS

    monkeypatch.setattr(live_model, "load_player_settings", load)
    result = env2.run(at("23:22"), live_model=LiveModel.team(), player_settings=None)
    (final,) = result.lines_written
    assert final["model_version"] == "v2" and final["call_type"] == "final"
    assert loaded == [True]
    assert env2.made[-1][0] == "player"


def test_player_settings_not_loaded_for_a_team_only_run(env, monkeypatch):
    def boom():
        raise AssertionError("must not load player settings")

    monkeypatch.setattr(live_model, "load_player_settings", boom)
    env.game()
    env.run(at("12:37"), player_settings=None)


def test_one_caller_per_kind_per_run_and_id_links_passed(env2):
    env2.game(gid="0022600501", tip=at("00:00", 6))
    env2.game(gid="0022600502", home="BOS", away="MIA", tip=at("00:30", 6))
    links = {"espn:1": "0022600777"}
    env2.run(at("12:37"), id_links=links)
    assert env2.made == [("player", links)]


def test_unknown_first_model_fails_before_any_line(env):
    env.game(gid="0022600501")
    env.game(gid="0022600502", home="BOS", away="MIA")
    env.run(at("12:37"))
    # Hand-edit a copy so the first call names an unknown model.
    path = log_path(env.repo, SEASON)
    lines = env.log()
    lines[1]["model_version"] = "v9"
    path.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in lines))
    before = path.read_bytes()
    with pytest.raises(LiveModelError):
        env.run(at("23:22"))
    assert path.read_bytes() == before


# --- the cutoff, end to end -------------------------------------------------------


def test_never_sealed_inside_the_cutoff(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch, live=V2)
    games = {
        "0022600601": ("LAL", "DEN", at("23:00")),
        "0022600602": ("BOS", "MIA", at("23:30")),
        "0022600603": ("PHI", "NYK", at("00:00", 6)),
        "0022600604": ("GSW", "PHX", at("02:30", 6)),
    }
    for gid, (home, away, tip) in games.items():
        env.game(gid=gid, home=home, away=away, tip=tip)
    # A report every hour; each flips one player per team, and p flips by
    # four points with it.
    report_at = at("12:00")
    n = 0
    while report_at <= at("04:00", 6):
        status = "Out" if n % 2 else "Questionable"
        env.report(report_at, [(t, f"Player{t}, X", status)
                               for home, away, _ in games.values() for t in (home, away)])
        report_at += timedelta(hours=1)
        n += 1
    env.p_fn = lambda rep, kind: 0.50 + (0.04 if rep is not None and rep.hour % 2 else 0.0)

    now = at("12:07")
    previous = b""
    while now <= at("03:52", 6):
        env.run(now)
        text = log_path(env.repo, SEASON).read_bytes()
        assert text.startswith(previous)  # append-only
        previous = text
        now += timedelta(minutes=15)

    log = env.log()
    tips = {gid: tip for gid, (_, _, tip) in games.items()}
    later = [x for x in log if x["call_type"] in ("revision", "final")
             and x["status"] == "predicted"]
    assert any(x["call_type"] == "revision" for x in later)
    for x in later:
        assert datetime.fromisoformat(x["predicted_at"]) < tips[x["game_id"]] - CALL_CUTOFF
    for gid in games:
        finals = [x for x in log if x["game_id"] == gid and x["call_type"] == "final"]
        assert len(finals) == 1 and finals[0]["status"] == "predicted", gid
        earlies = [x for x in log if x["game_id"] == gid and x["call_type"] == "early"]
        assert len(earlies) == 1
    # Each revision is due against the call it supersedes.
    by_at = {}
    for x in log:
        if x["status"] == "predicted":
            if x["call_type"] == "revision":
                prev = by_at[x["game_id"]]
                assert x["supersedes"] == prev["predicted_at"]
                assert abs(x["p_home"] - prev["p_home"]) >= 0.02 - 1e-9
            by_at[x["game_id"]] = x


# --- helpers --------------------------------------------------------------------


def test_latest_report_at_is_point_in_time(env):
    assert latest_report_at(env.con, at("12:00")) is None
    env.report(at("12:30"))
    env.report(at("13:30"))
    assert latest_report_at(env.con, at("13:00")) == at("12:30")
    assert latest_report_at(env.con, at("13:30")) == at("13:30")


def test_calls_due(env):
    # 12:00-12:30Z (EST) is the only stretch where the new slate's early call
    # is not yet due; then a game in the slate's range tipping after
    # now - 1h makes it due.
    assert not live.early_call_due(at("12:10"))
    env.game(gid="0022600499", tip=at("00:00"), game_date=date(2026, 11, 4))
    assert not calls_due(env.con, at("12:10"))
    assert calls_due(env.con, at("12:30"))
    env.game(gid="0022600498", tip=at("11:30"), game_date=date(2026, 11, 4))
    assert calls_due(env.con, at("12:10"))
    assert not calls_due(env.con, datetime(2026, 11, 6, 12, 10, tzinfo=UTC))
    env.game(tip=at("02:00", 6))
    assert calls_due(env.con, at("12:10"))


def test_early_only_constant_is_the_default():
    import inspect

    sig = inspect.signature(predict_today)
    assert sig.parameters["calls"].default == EARLY_ONLY
    assert sig.parameters["live_model"].default == LiveModel.team()
    assert sig.parameters["id_links"].default is None
