"""Tests for the live-timing fold and live sources (pure, offline).

The frames here are built by hand rather than downloaded, because the bugs that
matter are all *shape* bugs: a NaN position on a retiring car, a stale time on a
stopped car, a lapped car, a status string that concatenates several flags. Naming
those shapes in a fixture is what makes the regression assertion obvious.
"""

from __future__ import annotations

import pandas as pd
import pytest

from f1_sim.loaders import load_snapshot
from f1_sim.tuning.live import (
    fold_lap_frame,
    is_green_lap,
    snapshot_prefix,
    track_flag,
    write_snapshot,
)
from f1_sim.tuning.recording import PostSessionSource, RecordingLiveSource


def _td(seconds: float) -> pd.Timedelta:
    return pd.to_timedelta(seconds, unit="s")


def _row(
    code: str,
    num: int,
    lap: int,
    seconds: float,
    *,
    position: float,
    compound: str = "medium",
    stint: float = 1,
    tyre: float = 1.0,
    status: str = "1",
) -> dict:
    return {
        "Driver": code,
        "DriverNumber": num,
        "Team": f"{code} Team",
        "LapNumber": float(lap),
        "Time": _td(seconds),
        "Position": position,
        "Compound": compound,
        "Stint": stint,
        "TyreLife": tyre,
        "TrackStatus": status,
    }


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _by_code(payload: dict) -> dict[str, dict]:
    return {d["driver_code"]: d for d in payload["drivers"]}


# --------------------------------------------------------------------------
# track status
# --------------------------------------------------------------------------


def test_track_flag_uses_last_status_code():
    # A lap that ran yellow -> safety car -> red -> clear ended under a green flag.
    assert track_flag("2451") == "GREEN"
    assert track_flag("1245") == "SAFETY_CAR"
    assert track_flag("26") == "VSC"
    assert track_flag("671") == "GREEN"
    assert track_flag("2") == "YELLOW"
    assert track_flag("6") == "VSC"


def test_track_flag_named_forms():
    assert track_flag("VSC") == "VSC"
    assert track_flag("SC") == "SAFETY_CAR"


def test_track_flag_uses_the_last_element_not_a_name_match():
    # "VSC" contains "SC", so a naive substring check reports a safety car for a
    # virtual one. And when a named form is followed by a code, the *last* element
    # is what describes the field at the capture instant: a VSC stood down before
    # the lap ended is green again.
    assert track_flag("1VSC") == "VSC"
    assert track_flag("1245VSC") == "VSC"
    assert track_flag("VSC1") == "GREEN"
    assert track_flag("SC1") == "GREEN"


def test_is_green_lap_includes_laps_that_ended_green():
    assert is_green_lap("2451") is True
    assert is_green_lap("1") is True
    # Ends under red, so green was never in effect for the whole lap.
    assert is_green_lap("245") is False


def test_unknown_track_status_counts_as_green():
    # FastF1 leaves TrackStatus NaN/'' until the topic delivers. Reading "no
    # status" as "not green" dropped every lap of the session, so a live capture
    # wrote nothing at all and reported no error. Absence is not evidence of a red
    # flag, so it must not filter.
    assert is_green_lap(float("nan")) is True
    assert is_green_lap(None) is True
    assert is_green_lap("") is True
    assert is_green_lap("   ") is True


def test_fold_survives_an_absent_or_unfilled_track_status():
    # The same reasoning as above, one level up: all three ways a status column can
    # be missing have to fold, not silently return None.
    rows = [_row("VER", 1, 5, 450.0, position=1.0), _row("LEC", 16, 5, 451.0, position=2.0)]

    dropped = [dict(row) for row in rows]
    for row in dropped:
        del row["TrackStatus"]
    for label, frame in (
        ("column absent", _frame(dropped)),
        ("all NaN", _frame([{**r, "TrackStatus": float("nan")} for r in rows])),
        ("all empty", _frame([{**r, "TrackStatus": ""} for r in rows])),
    ):
        payload = fold_lap_frame(frame, circuit_id="monza")
        assert payload is not None, f"{label} should still fold"
        assert payload["current_lap"] == 5
        assert len(payload["drivers"]) == 2


# --------------------------------------------------------------------------
# retirement
# --------------------------------------------------------------------------


def test_retirement_nan_position_does_not_raise():
    # The retiring driver's final lap has no Position, which is what a real
    # retirement looks like; a straight int() on it would raise. A retiring car
    # stops while the leader keeps going, so it falls more than a lap behind.
    rows = [
        _row("VER", 1, 5, 450.0, position=1.0),
        _row("LEC", 16, 4, 361.5, position=2.0),
        _row("LEC", 16, 5, 451.5, position=float("nan")),
        _row("VER", 1, 6, 540.0, position=1.0),
        _row("VER", 1, 7, 630.0, position=1.0),
    ]
    payload = fold_lap_frame(_frame(rows), circuit_id="monza")
    lec = _by_code(payload)["LEC"]
    assert lec["running"] is False
    assert lec["dnf_reason"] == "retired"


def test_retired_cars_parked_after_the_running_field():
    # SAI last held position 2; by the time it retires PIA holds it. The stale
    # position must not put two cars in the same slot.
    rows = [
        _row("SAI", 5, 4, 360.0, position=2.0),
        _row("SAI", 5, 5, 451.0, position=float("nan")),
        _row("VER", 1, 5, 450.0, position=1.0),
        _row("PIA", 81, 5, 452.0, position=3.0),
        _row("VER", 1, 6, 540.0, position=1.0),
        _row("PIA", 81, 6, 542.0, position=2.0),
        _row("VER", 1, 7, 630.0, position=1.0),
        _row("PIA", 81, 7, 632.0, position=2.0),
    ]
    payload = fold_lap_frame(_frame(rows), circuit_id="monza")
    positions = [d["position"] for d in payload["drivers"]]
    assert sorted(positions) == list(range(1, len(positions) + 1))
    assert len(set(positions)) == len(positions)

    by_code = _by_code(payload)
    assert by_code["PIA"]["position"] == 2
    assert by_code["SAI"]["position"] == 3
    assert by_code["SAI"]["running"] is False


def test_one_lap_down_is_still_running():
    # A car that pits on the leader's final lap is a lap down while still running;
    # calling it a DNF would be worse than letting it recover next poll.
    rows = [
        _row("VER", 1, 5, 450.0, position=1.0),
        _row("LEC", 16, 5, 451.0, position=2.0),
        _row("LEC", 16, 4, 360.0, position=2.0),
        _row("VER", 1, 6, 540.0, position=1.0),
    ]
    payload = fold_lap_frame(_frame(rows), circuit_id="monza")
    assert _by_code(payload)["LEC"]["running"] is True


def test_two_laps_down_after_a_safety_car_pit_is_read_as_retired():
    # Pinned limitation, not desired behaviour. Retirement is inferred from lap
    # count because the fold only sees FastF1's parsed laps, which carry no
    # per-driver status. During a *long* safety car a car that pits can end up more
    # than a lap down while perfectly healthy, and the inference calls it a DNF.
    #
    # One lap of tolerance absorbs a normal pit (see the test above); two laps is
    # where it stops being safe. Making this exact needs the raw TimingData Status
    # field, which FastF1's parser discards.
    rows = [
        _row("VER", 1, 5, 450.0, position=1.0),
        _row("LEC", 16, 5, 451.0, position=2.0),
        _row("VER", 1, 6, 540.0, position=1.0),
        _row("VER", 1, 7, 630.0, position=1.0),
    ]
    payload = fold_lap_frame(_frame(rows), circuit_id="monza")
    lec = _by_code(payload)["LEC"]
    assert lec["laps_completed"] == 5
    assert lec["running"] is False
    assert lec["dnf_reason"] == "retired"


# --------------------------------------------------------------------------
# gaps and intervals
# --------------------------------------------------------------------------


def test_leader_gap_is_zero():
    rows = [
        _row("VER", 1, 5, 450.0, position=1.0),
        _row("LEC", 16, 5, 451.5, position=2.0),
    ]
    payload = fold_lap_frame(_frame(rows), circuit_id="monza")
    assert payload["drivers"][0]["driver_code"] == "VER"
    assert payload["drivers"][0]["gap_to_leader_seconds"] == 0.0


def test_gap_for_lapped_car_uses_the_leader_at_the_same_lap():
    # A car a lap down must be compared against the leader's time for the lap *it*
    # ran, not the leader's current time, which is ~90s further on and would make
    # the gap collapse.
    rows = [
        _row("VER", 1, 4, 360.0, position=1.0),
        _row("VER", 1, 5, 450.0, position=1.0),
        _row("NOR", 4, 5, 451.0, position=2.0),
        _row("LEC", 16, 4, 366.0, position=3.0),
    ]
    payload = fold_lap_frame(_frame(rows), circuit_id="monza")
    by_code = _by_code(payload)
    # Leader's lap-4 time is 360.0, so LEC is 6s behind where the leader was.
    assert by_code["LEC"]["gap_to_leader_seconds"] == pytest.approx(6.0)
    assert by_code["LEC"]["laps_completed"] == 4


def test_gap_never_negative_and_intervals_never_negative():
    rows = [
        _row("VER", 1, 5, 450.0, position=1.0),
        _row("LEC", 16, 5, 451.5, position=2.0),
        # A stopped car keeps an old time, older than the leader's same-lap time
        # would suggest once it is compared naively.
        _row("SAI", 5, 3, 300.0, position=2.0),
        _row("SAI", 5, 4, 330.0, position=float("nan")),
        _row("VER", 1, 3, 270.0, position=1.0),
    ]
    payload = fold_lap_frame(_frame(rows), circuit_id="monza")
    for driver in payload["drivers"]:
        assert driver["gap_to_leader_seconds"] >= 0.0
        assert driver["interval_to_ahead_seconds"] >= 0.0


def test_intervals_chain_past_a_retired_car():
    # Regression: the retired car is parked between the leader and the third car,
    # holding a stale time. Chaining intervals through it handed the car behind a
    # negative gap that clamped to 0, so its true interval was lost.
    rows = [
        _row("VER", 1, 3, 270.0, position=1.0),
        _row("VER", 1, 5, 450.0, position=1.0),
        _row("SAI", 5, 3, 200.0, position=2.0),
        _row("NOR", 4, 5, 454.0, position=3.0),
    ]
    payload = fold_lap_frame(_frame(rows), circuit_id="monza")
    by_code = _by_code(payload)
    # Positions: runners VER(1) and NOR(2) in known order, SAI parked last.
    assert by_code["NOR"]["position"] == 2
    assert by_code["SAI"]["position"] == 3
    # The interval is measured against the leader, not against SAI's stale gap.
    assert by_code["NOR"]["gap_to_leader_seconds"] == pytest.approx(4.0)
    assert by_code["NOR"]["interval_to_ahead_seconds"] == pytest.approx(4.0)
    assert by_code["SAI"]["interval_to_ahead_seconds"] == 0.0


def test_intervals_chain_consecutively_among_runners():
    rows = [
        _row("VER", 1, 5, 450.0, position=1.0),
        _row("LEC", 16, 5, 451.5, position=2.0),
        _row("NOR", 4, 5, 455.0, position=3.0),
    ]
    payload = fold_lap_frame(_frame(rows), circuit_id="monza")
    by_code = _by_code(payload)
    assert by_code["VER"]["interval_to_ahead_seconds"] == 0.0
    assert by_code["LEC"]["interval_to_ahead_seconds"] == pytest.approx(1.5)
    assert by_code["NOR"]["interval_to_ahead_seconds"] == pytest.approx(3.5)


# --------------------------------------------------------------------------
# compounds, laps, flags
# --------------------------------------------------------------------------


def test_compounds_used_so_far_ordered_by_first_use():
    rows = [
        _row("VER", 1, 1, 90.0, position=1.0, compound="soft", stint=1),
        _row("VER", 1, 2, 180.0, position=1.0, compound="soft", stint=1),
        _row("VER", 1, 3, 270.0, position=1.0, compound="hard", stint=2),
        _row("VER", 1, 4, 360.0, position=1.0, compound="hard", stint=2),
    ]
    payload = fold_lap_frame(_frame(rows), circuit_id="monza")
    ver = _by_code(payload)["VER"]
    assert ver["compounds_used_so_far"] == ["soft", "hard"]
    assert ver["current_tire_compound"] == "hard"


def test_at_lap_truncates_the_frame():
    rows = [
        _row("VER", 1, 1, 90.0, position=1.0),
        _row("VER", 1, 2, 180.0, position=1.0),
        _row("VER", 1, 3, 270.0, position=1.0),
    ]
    payload = fold_lap_frame(_frame(rows), circuit_id="monza", at_lap=2)
    assert payload["current_lap"] == 2
    assert _by_code(payload)["VER"]["laps_completed"] == 2


def test_non_green_laps_are_excluded():
    rows = [
        _row("VER", 1, 1, 90.0, position=1.0, status="1"),
        _row("VER", 1, 2, 180.0, position=1.0, status="245"),
        _row("VER", 1, 3, 270.0, position=1.0, status="2451"),
    ]
    payload = fold_lap_frame(_frame(rows), circuit_id="monza")
    # Lap 2 is dropped; lap 3 ran partly under red but ended clear, so it counts.
    assert payload["current_lap"] == 3


def test_race_flag_comes_from_the_end_of_the_frame():
    # A lap run entirely under a VSC is not green, so it is filtered out of the lap
    # data -- the flag still has to report VSC, which means it cannot be read from
    # the filtered rows.
    rows = [
        _row("VER", 1, 1, 90.0, position=1.0, status="1"),
        _row("VER", 1, 2, 180.0, position=1.0, status="6"),
    ]
    payload = fold_lap_frame(_frame(rows), circuit_id="monza")
    assert payload["current_lap"] == 1
    assert payload["race_flag"] == "VSC"


def test_empty_and_unusable_frames_return_none():
    assert fold_lap_frame(_frame([]), circuit_id="monza") is None
    assert (
        fold_lap_frame(
            _frame([_row("VER", 1, 1, 90.0, position=1.0, status="2")]),
            circuit_id="monza",
        )
        is None
    )


def test_missing_column_raises_actionable_error():
    frame = _frame([_row("VER", 1, 1, 90.0, position=1.0)]).drop(columns=["Time"])
    with pytest.raises(ValueError, match="Time"):
        fold_lap_frame(frame, circuit_id="monza")


def test_payload_loads_through_the_real_loader():
    rows = [
        _row("VER", 1, 3, 270.0, position=1.0),
        _row("VER", 1, 5, 450.0, position=1.0),
        _row("LEC", 16, 5, 451.5, position=2.0, compound="soft"),
        _row("SAI", 5, 3, 200.0, position=2.0),
    ]
    payload = fold_lap_frame(_frame(rows), circuit_id="monza")
    snapshot = load_snapshot(payload)
    assert snapshot.circuit_id == "monza"
    assert snapshot.current_lap == 5
    assert len(snapshot.drivers) == 3


# --------------------------------------------------------------------------
# recording prefix
# --------------------------------------------------------------------------


def test_snapshot_prefix_trims_a_partial_final_line(tmp_path):
    recording = tmp_path / "live.txt"
    recording.write_text("one\ntwo\nthree\n", encoding="utf-8")
    # Cut inside "three".
    dest = tmp_path / "prefix.txt"
    result = snapshot_prefix(recording, upto_bytes=12, dest=dest)
    assert result is not None
    assert result.read_text(encoding="utf-8") == "one\ntwo\n"


def test_snapshot_prefix_whole_file(tmp_path):
    recording = tmp_path / "live.txt"
    recording.write_text("one\ntwo\n", encoding="utf-8")
    result = snapshot_prefix(recording, dest=tmp_path / "prefix.txt")
    assert result.read_text(encoding="utf-8") == "one\ntwo\n"


def test_snapshot_prefix_returns_none_when_nothing_usable(tmp_path):
    missing = tmp_path / "nope.txt"
    assert snapshot_prefix(missing) is None

    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    assert snapshot_prefix(empty) is None

    # A single unterminated message is not a complete line yet.
    partial = tmp_path / "partial.txt"
    partial.write_text("no newline", encoding="utf-8")
    assert snapshot_prefix(partial, dest=tmp_path / "p.txt") is None


def test_snapshot_prefix_trims_across_buffer_windows(tmp_path):
    # The newline search scans backwards in fixed windows. A partial line that
    # begins many windows from the end -- and a file whose first newline is far
    # from the tail -- both have to land on the same answer as a small file, and
    # without reading the whole thing to find it.
    line = b"x" * 200 + b"\n"
    whole_lines = 1200
    recording = tmp_path / "big.txt"
    recording.write_bytes(line * whole_lines + b"PARTIAL-CUT-ME")

    result = snapshot_prefix(recording, dest=tmp_path / "prefix.txt")
    assert result is not None
    assert result.stat().st_size == len(line) * whole_lines
    assert result.read_bytes().endswith(b"\n")

    # A partial line far longer than one window, so the backwards scan has to walk
    # through several windows before it finds a newline. (A 200KB partial line does
    # not happen with real SignalR messages, but it is the case that would break a
    # single-window scan.)
    long_partial = tmp_path / "long_partial.txt"
    long_partial.write_bytes(line * whole_lines + b"z" * 200_000)
    trimmed = snapshot_prefix(long_partial, dest=tmp_path / "long_prefix.txt")
    assert trimmed.read_bytes() == line * whole_lines


# --------------------------------------------------------------------------
# live sources
# --------------------------------------------------------------------------


def _three_car_laps() -> pd.DataFrame:
    return _frame(
        [
            _row("VER", 1, 1, 90.0, position=1.0),
            _row("LEC", 16, 1, 91.0, position=2.0),
            _row("VER", 1, 2, 180.0, position=1.0),
            _row("LEC", 16, 2, 181.0, position=2.0),
        ]
    )


def test_recording_source_returns_none_before_any_data(tmp_path):
    source = RecordingLiveSource(
        tmp_path / "live.txt",
        circuit_id="monza",
        year=2026,
        gp="italian",
        loader=lambda p: _three_car_laps(),
    )
    assert source.poll() is None
    assert source.exhausted is False


def test_recording_source_folds_new_bytes_then_waits(tmp_path):
    recording = tmp_path / "live.txt"
    recording.write_text('["TimingData", {}, "2026-01-01T00:00:00"]\n', encoding="utf-8")

    seen: list[int] = []

    def loader(prefix: object) -> pd.DataFrame:
        seen.append(prefix.stat().st_size)
        return _three_car_laps()

    source = RecordingLiveSource(
        recording, circuit_id="monza", year=2026, gp="italian", loader=loader
    )
    payload = source.poll()
    assert payload is not None
    assert payload["current_lap"] == 2

    # Nothing appended: the same bytes must not produce a duplicate frame.
    assert source.poll() is None

    recording.write_text(
        '["TimingData", {}, "2026-01-01T00:00:00"]\n["TimingData", {}, "2026-01-01T00:00:01"]\n',
        encoding="utf-8",
    )
    assert source.poll() is not None
    assert len(seen) == 2
    assert seen[1] > seen[0]


def test_recording_source_survives_an_unreadable_prefix(tmp_path):
    recording = tmp_path / "live.txt"
    recording.write_text("partial feed\n", encoding="utf-8")

    def loader(prefix: object) -> pd.DataFrame:
        raise ValueError("feed is not interpretable yet")

    source = RecordingLiveSource(
        recording, circuit_id="monza", year=2026, gp="italian", loader=loader
    )
    # A half-received feed is expected mid-session, not a reason to stop capture.
    assert source.poll() is None


def test_recording_without_timing_topics_never_falls_back_to_the_archive(tmp_path):
    # Regression: FastF1's data loaders fall back to the network archive when the
    # recording is missing a topic. Mid-session the archive 403s, and for a
    # finished session it would hand back the entire race -- making a dead recorder
    # look like a working one. The source must refuse and say why.
    recording = tmp_path / "live.txt"
    recording.write_text(
        '["Heartbeat",{"Remaining":10},"2026-01-01T00:00:00.000"]\n', encoding="utf-8"
    )
    source = RecordingLiveSource(
        recording, circuit_id="monza", year=2026, gp="italian", session="R"
    )
    assert source.poll() is None
    assert source.last_error is not None
    assert "TimingData" in source.last_error
    assert "DriverList" in source.last_error


def test_recording_source_reprocesses_a_trimmed_tail(tmp_path):
    # A prefix cut mid-line is trimmed, so the next poll must pick the remainder
    # up rather than skipping it.
    recording = tmp_path / "live.txt"
    recording.write_text("one\ntwo\n", encoding="utf-8")
    sizes: list[int] = []

    source = RecordingLiveSource(
        recording,
        circuit_id="monza",
        year=2026,
        gp="italian",
        loader=lambda p: sizes.append(p.stat().st_size) or _three_car_laps(),
    )
    source.poll()
    recording.write_text("one\ntwo\nthree\n", encoding="utf-8")
    source.poll()
    # Second prefix covers everything complete, which is more than the first.
    assert sizes[1] > sizes[0]
    assert sizes[1] == len("one\ntwo\nthree\n")


def test_post_session_source_walks_one_lap_per_poll():
    source = PostSessionSource(
        year=2026,
        gp="italian",
        session_identifier="R",
        circuit_id="monza",
        laps=_three_car_laps(),
    )
    assert [source.poll()["current_lap"] for _ in range(2)] == [1, 2]
    assert source.poll() is None
    assert source.exhausted is True


def test_post_session_source_respects_at_lap():
    source = PostSessionSource(
        year=2026,
        gp="italian",
        session_identifier="R",
        circuit_id="monza",
        at_lap=1,
        laps=_three_car_laps(),
    )
    assert source.poll()["current_lap"] == 1
    assert source.poll() is None
    assert source.exhausted is True


def test_write_snapshot_is_lap_addressable_and_reloadable(tmp_path):
    payload = fold_lap_frame(_three_car_laps(), circuit_id="monza")
    path = write_snapshot(payload, tmp_path, "monza", payload["current_lap"])
    assert path.name == "monza_lap02.json"
    assert load_snapshot(str(path)).circuit_id == "monza"
