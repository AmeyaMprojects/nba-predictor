"""Call rules: windows, chains, injury status changes and the revision rule."""

from datetime import UTC, date, datetime, timedelta

from predictor.cloud.loop import next_mark
from predictor.model import calls, live
from predictor.model.availability import Snapshot
from predictor.model.calls import (
    ALL_CALLS,
    CALL_CUTOFF,
    EARLY_ONLY,
    FINAL_LEAD,
    MAX_NAMED_CHANGES,
    REVISION_MIN_MOVE,
    TBD_REASON,
    GameChain,
    StatusChange,
    call_type_of,
    chains,
    final_due,
    revision_due,
    revision_reason,
    status_changes,
    still_callable,
)


def dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(UTC)


def line(game_id="0022600100", game_date="2026-11-05", *, status="predicted",
         call_type=None, at="2026-11-05T12:37:00+00:00", reason=None, p=0.6, **extra):
    out = {"game_id": game_id, "game_date": game_date, "status": status,
           "predicted_at": at, "p_home": p if status == "predicted" else None}
    if call_type is not None:
        out["call_type"] = call_type
    if reason is not None:
        out["reason"] = reason
    out.update(extra)
    return out


def snap(at: str, statuses: dict) -> Snapshot:
    return Snapshot(dt(at), dict(statuses))


MURRAY_OUT = StatusChange("DEN", "Murray, Jamal", "Questionable", "Out")


# ---------------------------------------------------------------- constants


def test_constants():
    assert ALL_CALLS == frozenset({"early", "revision", "final"})
    assert EARLY_ONLY == frozenset({"early"})
    assert FINAL_LEAD == timedelta(minutes=45)
    assert CALL_CUTOFF == timedelta(minutes=15)
    assert REVISION_MIN_MOVE == 0.02
    assert MAX_NAMED_CHANGES == 3


def test_tbd_reason_moved_here_and_live_uses_it():
    assert TBD_REASON == "tip-off time not announced when the prediction run happened"
    assert live._TBD_REASON is calls.TBD_REASON


def test_call_type_of_defaults_to_early():
    assert call_type_of({}) == "early"
    assert call_type_of({"call_type": None}) == "early"
    assert call_type_of({"call_type": "final"}) == "final"
    assert call_type_of({"call_type": "revision"}) == "revision"


# ---------------------------------------------------------------- windows


def test_final_due_window():
    tip = dt("2026-11-06T00:00Z")
    assert not final_due(tip, dt("2026-11-05T23:14:59Z"))
    assert final_due(tip, dt("2026-11-05T23:15Z"))
    assert final_due(tip, dt("2026-11-05T23:44:59Z"))
    assert not final_due(tip, dt("2026-11-05T23:45Z"))
    assert not final_due(tip, dt("2026-11-06T00:30Z"))
    assert still_callable(tip, dt("2026-11-05T23:44:59Z"))
    assert not still_callable(tip, dt("2026-11-05T23:45Z"))
    assert still_callable(tip, dt("2026-11-05T12:00Z"))


def test_final_due_across_dst_change():
    # 2026-11-01: EDT -> EST at 06:00Z. Tip 00:00Z on Nov 2 is 19:00 EST on Nov 1.
    tip = dt("2026-11-02T00:00Z")
    assert final_due(tip, dt("2026-11-01T23:15Z"))
    assert not final_due(tip, dt("2026-11-01T23:14:59Z"))
    assert not final_due(tip, dt("2026-11-01T23:45Z"))
    assert live.slate_date(dt("2026-11-01T23:15Z")) == date(2026, 11, 1)
    # The window is plain UTC arithmetic: always exactly 30 minutes wide,
    # including a tip on the morning of the switch itself.
    early_tip = dt("2026-11-01T06:30Z")
    assert final_due(early_tip, dt("2026-11-01T05:45Z"))
    assert final_due(early_tip, dt("2026-11-01T06:14:59Z"))
    assert not final_due(early_tip, dt("2026-11-01T06:15Z"))


def test_final_window_holds_two_loop_marks():
    for minute in range(60):
        tip = dt("2026-11-06T00:00Z") + timedelta(minutes=minute)
        start, end = tip - FINAL_LEAD, tip - CALL_CUTOFF
        marks, mark = [], next_mark(tip - timedelta(hours=2), 15)
        while mark < tip:
            marks.append(mark)
            mark = next_mark(mark, 15)
        inside = [m for m in marks if start <= m < end]
        assert len(inside) == 2, (minute, inside)
        assert all(final_due(tip, m) for m in inside)
        # A late iteration (one mark skipped) still leaves one in the window.
        assert any(final_due(tip, m) for m in inside[1:])


def test_every_tip_minute_has_two_callable_final_marks():
    # One cutoff for every call: both final marks are also callable (early,
    # revision and final alike), for every tip minute of a day.
    from predictor.model.live import START_BUFFER

    assert START_BUFFER == CALL_CUTOFF
    for minute in range(24 * 60):
        tip = dt("2026-11-05T12:00Z") + timedelta(minutes=minute)
        marks, mark = [], next_mark(tip - timedelta(hours=2), 15)
        while mark < tip:
            marks.append(mark)
            mark = next_mark(mark, 15)
        inside = [m for m in marks if final_due(tip, m)]
        assert len(inside) == 2, (minute, inside)
        assert all(still_callable(tip, m) for m in inside)
        assert all(tip - START_BUFFER > m for m in inside)


# ---------------------------------------------------------------- chains


def test_chain_reads_old_lines_as_early():
    old = line()
    c = chains([old], {})[("0022600100", "2026-11-05")]
    assert c.early is old
    assert c.final is None
    assert c.first_predicted is old and c.latest_predicted is old
    assert c.predicted == (old,)
    assert c.game_id == "0022600100"


def test_chain_skips_tbd_for_early_and_orders_calls():
    tbd = line(status="not_predicted", reason=TBD_REASON, at="2026-11-05T10:07:00+00:00")
    early = line(call_type="early", p=0.61)
    rev = line(call_type="revision", at="2026-11-05T22:37:00+00:00", p=0.55)
    fin = line(call_type="final", at="2026-11-05T23:22:00+00:00", p=0.55)
    rev2 = line(call_type="revision", at="2026-11-05T23:37:00+00:00", p=0.45)
    other = line("0022600101", p=0.4)
    out = chains([tbd, early, other, rev, fin, rev2], {})
    assert list(out) == [("0022600100", "2026-11-05"), ("0022600101", "2026-11-05")]
    c = out[("0022600100", "2026-11-05")]
    assert c.lines == (tbd, early, rev, fin, rev2)
    assert c.early is early
    assert c.final is fin
    assert c.first_predicted is early
    assert c.latest_predicted is rev2
    assert c.predicted == (early, rev, fin, rev2)


def test_chain_tbd_only_has_no_early_and_no_predicted():
    tbd = line(status="not_predicted", reason=TBD_REASON)
    c = chains([tbd], {})[("0022600100", "2026-11-05")]
    assert c.early is None
    assert c.first_predicted is None and c.latest_predicted is None
    assert c.predicted == ()


def test_missed_final_line_is_final_but_not_predicted():
    early = line(call_type="early")
    missed = line(status="not_predicted", call_type="final",
                  reason="no final call was made", at="2026-11-05T23:52:00+00:00")
    c = chains([early, missed], {})[("0022600100", "2026-11-05")]
    assert c.final is missed
    assert c.latest_predicted is early


def test_chain_follows_provisional_link():
    a = line("espn:401705555", "2026-12-16", call_type="early")
    b = line("espn:401705555", "2026-12-16", call_type="final", at="2026-12-16T23:22:00+00:00")
    out = chains([a, b], {"espn:401705555": "0042600101"})
    assert list(out) == [("0042600101", "2026-12-16")]
    c = out[("0042600101", "2026-12-16")]
    assert c.game_id == "espn:401705555"
    assert c.lines == (a, b)


def test_provisional_and_nba_lines_join_one_chain_under_the_first_id():
    a = line("espn:401705555", "2026-12-16", call_type="early")
    b = line("0042600101", "2026-12-16", call_type="final", at="2026-12-16T23:22:00+00:00")
    out = chains([a, b], {"espn:401705555": "0042600101"})
    c = out[("0042600101", "2026-12-16")]
    assert c.game_id == "espn:401705555"
    assert c.early is a and c.final is b


def test_postponed_game_has_a_separate_chain_per_date():
    old = line(game_date="2026-11-05", call_type="early")
    new = line(game_date="2026-11-20", call_type="early", at="2026-11-20T12:37:00+00:00")
    out = chains([old, new], {})
    assert out[("0022600100", "2026-11-05")].lines == (old,)
    assert out[("0022600100", "2026-11-20")].lines == (new,)
    assert out[("0022600100", "2026-11-20")].final is None


def test_game_chain_is_frozen():
    c = GameChain((line(),))
    try:
        c.lines = ()  # type: ignore[misc]
    except AttributeError:
        pass
    else:
        raise AssertionError("GameChain must be frozen")


# ---------------------------------------------------------------- status changes


def test_status_changes_detects_new_flip_and_drop():
    before = snap("2026-11-05T22:30Z", {("DEN", "Murray, Jamal"): "Questionable",
                                        ("DEN", "Porter Jr., Michael"): "Out"})
    after = snap("2026-11-05T23:30Z", {("DEN", "Murray, Jamal"): "Out"})
    out = status_changes(before, after, ("DEN", "BOS"))
    assert out == [StatusChange("DEN", "Murray, Jamal", "Questionable", "Out"),
                   StatusChange("DEN", "Porter Jr., Michael", "Out", None)]
    assert [c.text() for c in out] == ["Murray (DEN) out", "Porter Jr. (DEN) available"]


def test_status_changes_newly_listed_player():
    before = snap("2026-11-05T22:30Z", {})
    after = snap("2026-11-05T23:30Z", {("BOS", "Tatum, Jayson"): "Doubtful"})
    out = status_changes(before, after, ("DEN", "BOS"))
    assert out == [StatusChange("BOS", "Tatum, Jayson", None, "Doubtful")]
    assert out[0].text() == "Tatum (BOS) doubtful"


def test_status_changes_ignore_other_teams_and_case():
    before = snap("2026-11-05T22:30Z", {("DEN", "Murray, Jamal"): "OUT",
                                        ("LAL", "James, LeBron"): "Questionable"})
    after = snap("2026-11-05T23:30Z", {("DEN", "Murray, Jamal"): " Out ",
                                       ("LAL", "James, LeBron"): "Out"})
    assert status_changes(before, after, ("DEN", "BOS")) == []


def test_status_changes_order_by_team_position_then_player():
    after = snap("2026-11-05T23:30Z", {("DEN", "Murray, Jamal"): "Out",
                                       ("BOS", "Tatum, Jayson"): "Out",
                                       ("BOS", "Brown, Jaylen"): "Out",
                                       ("DEN", "Jokic, Nikola"): "Out"})
    out = status_changes(None, after, ("BOS", "DEN"))
    assert [(c.team, c.player) for c in out] == [
        ("BOS", "Brown, Jaylen"), ("BOS", "Tatum, Jayson"),
        ("DEN", "Jokic, Nikola"), ("DEN", "Murray, Jamal")]


def test_unknown_status_counts_as_a_change():
    before = snap("2026-11-05T22:30Z", {("DEN", "Murray, Jamal"): "Available"})
    after = snap("2026-11-05T23:30Z", {("DEN", "Murray, Jamal"): "Not With Team"})
    out = status_changes(before, after, ("DEN", "BOS"))
    assert out == [StatusChange("DEN", "Murray, Jamal", "Available", "Not With Team")]
    assert out[0].text() == "Murray (DEN) not with team"


def test_status_flipping_back_is_a_change_each_way():
    q = snap("2026-11-05T22:30Z", {("DEN", "Murray, Jamal"): "Questionable"})
    o = snap("2026-11-05T23:30Z", {("DEN", "Murray, Jamal"): "Out"})
    back = snap("2026-11-05T23:45Z", {("DEN", "Murray, Jamal"): "Questionable"})
    assert [c.text() for c in status_changes(q, o, ("DEN", "BOS"))] == ["Murray (DEN) out"]
    assert [c.text() for c in status_changes(o, back, ("DEN", "BOS"))] == [
        "Murray (DEN) questionable"]
    assert status_changes(q, back, ("DEN", "BOS")) == []


def test_before_snapshot_none_lists_every_listed_player():
    after = snap("2026-11-05T23:30Z", {("DEN", "Murray, Jamal"): "Out",
                                       ("BOS", "Tatum, Jayson"): "Questionable",
                                       ("LAL", "James, LeBron"): "Out"})
    out = status_changes(None, after, ("DEN", "BOS"))
    assert out == [StatusChange("DEN", "Murray, Jamal", None, "Out"),
                   StatusChange("BOS", "Tatum, Jayson", None, "Questionable")]


def test_after_snapshot_none_lists_every_player_as_available():
    before = snap("2026-11-05T22:30Z", {("DEN", "Murray, Jamal"): "Out"})
    out = status_changes(before, None, ("DEN", "BOS"))
    assert [c.text() for c in out] == ["Murray (DEN) available"]
    assert status_changes(None, None, ("DEN", "BOS")) == []


def test_surname_without_comma_is_last_word():
    assert StatusChange("DEN", "Nikola Jokic", None, "Out").text() == "Jokic (DEN) out"
    assert StatusChange("DEN", "Nene", None, "Out").text() == "Nene (DEN) out"


# ---------------------------------------------------------------- revision rule


LATEST = {"p_home": 0.61, "injury_report_at": "2026-11-05T22:30:00+00:00"}


def test_revision_due_needs_newer_report_change_and_two_points():
    report = dt("2026-11-05T23:30Z")
    assert revision_due(LATEST, 0.59, report, [MURRAY_OUT])
    assert revision_due(LATEST, 0.63, report, [MURRAY_OUT])
    assert not revision_due(LATEST, 0.5901, report, [MURRAY_OUT])
    assert not revision_due(LATEST, 0.59, dt("2026-11-05T22:30Z"), [MURRAY_OUT])
    assert not revision_due(LATEST, 0.59, dt("2026-11-05T22:00Z"), [MURRAY_OUT])
    assert not revision_due(LATEST, 0.59, report, [])
    assert not revision_due(LATEST, 0.59, None, [MURRAY_OUT])


def test_revision_due_with_no_report_on_latest_line():
    report = dt("2026-11-05T23:30Z")
    assert revision_due({"p_home": 0.61, "injury_report_at": None}, 0.55, report, [MURRAY_OUT])
    # Old lines carry no injury_report_at key at all.
    assert revision_due({"p_home": 0.61}, 0.55, report, [MURRAY_OUT])


def test_revision_due_exact_two_points_despite_float_noise():
    report = dt("2026-11-05T23:30Z")
    # 0.61 - 0.59 is 0.020000000000000018 and 0.3 - 0.28 is 0.01999999999999999.
    assert revision_due({"p_home": 0.3, "injury_report_at": None}, 0.28, report, [MURRAY_OUT])
    assert revision_due({"p_home": 0.28, "injury_report_at": None}, 0.3, report, [MURRAY_OUT])
    assert not revision_due({"p_home": 0.3, "injury_report_at": None}, 0.281, report,
                            [MURRAY_OUT])


def test_revision_due_same_report_twice_is_not_due():
    # Duplicate guard: once a revision records report_at, the same report never revises again.
    report = dt("2026-11-05T23:30Z")
    revised = {"p_home": 0.55, "injury_report_at": report.isoformat()}
    assert not revision_due(revised, 0.40, report, [MURRAY_OUT])


def test_revision_reason_text():
    assert (revision_reason(dt("2026-11-05T18:00Z"), [MURRAY_OUT])
            == "new injury report 23:30 IST: Murray (DEN) out")
    five = [StatusChange("DEN", f"P{i}, X", None, "Out") for i in range(5)]
    text = revision_reason(dt("2026-11-05T23:30Z"), five)
    assert text == ("new injury report 05:00 IST: P0 (DEN) out, P1 (DEN) out, "
                    "P2 (DEN) out and 2 more")
    assert text.endswith(" and 2 more")
    three = five[:3]
    assert not revision_reason(dt("2026-11-05T23:30Z"), three).endswith("more")


# ---------------------------------------------------------------- carried from the Task 5 review


def test_available_to_missing_is_not_a_change():
    # A player listed "Available" who then drops off the report reads
    # "available" both times: no change, so no revision.
    before = snap("2026-11-05T22:30Z", {("DEN", "Murray, Jamal"): "Available"})
    after = snap("2026-11-05T23:30Z", {})
    assert status_changes(before, after, ("DEN", "BOS")) == []
    # ...and a newly listed "Available" player is no change either.
    assert status_changes(after, before, ("DEN", "BOS")) == []
    # A real status dropping off still reads "available".
    out_before = snap("2026-11-05T22:30Z", {("DEN", "Murray, Jamal"): "Out"})
    assert [c.text() for c in status_changes(out_before, after, ("DEN", "BOS"))] == [
        "Murray (DEN) available"]


def test_revision_due_tolerates_a_naive_report_time_as_utc():
    report = dt("2026-11-05T23:30Z")
    naive_same = {"p_home": 0.61, "injury_report_at": "2026-11-05T23:30:00"}
    naive_older = {"p_home": 0.61, "injury_report_at": "2026-11-05T22:30:00"}
    assert not revision_due(naive_same, 0.50, report, [MURRAY_OUT])
    assert revision_due(naive_older, 0.50, report, [MURRAY_OUT])
    assert calls.as_utc("2026-11-05T22:30:00") == dt("2026-11-05T22:30Z")
    assert calls.as_utc(datetime(2026, 11, 5, 22, 30)) == dt("2026-11-05T22:30Z")
