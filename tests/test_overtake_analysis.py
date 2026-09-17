"""Tests for empirical overtake analysis (pure, offline)."""

from __future__ import annotations

import pandas as pd
import pytest

from f1_sim.tuning.overtake_analysis import (
    aggregate_pace_delta,
    detect_on_track_overtakes,
)


def _td(seconds: float) -> pd.Timedelta:
    return pd.to_timedelta(seconds, unit="s")


def _race(positions_by_lap: dict[int, dict[str, int]], lap_times: dict[int, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for lap, order in positions_by_lap.items():
        for driver, position in order.items():
            rows.append(
                {
                    "Driver": driver,
                    "LapNumber": lap,
                    "LapTime": _td(lap_times[lap][driver]),
                    "Position": position,
                    "PitInTime": pd.NaT,
                    "PitOutTime": pd.NaT,
                    "TrackStatus": "1",
                }
            )
    df = pd.DataFrame(rows)
    for col in ("PitInTime", "PitOutTime"):
        df[col] = pd.Series([pd.NaT] * len(df), dtype="timedelta64[ns]")
    return df


def _two_car_pass(
    pass_lap: int, deltas: dict[int, float], *, n_laps: int = 10, default_delta: float = 2.0
) -> pd.DataFrame:
    """Build a two-car race where HAM defends until ``pass_lap`` then VER leads.

    ``deltas[L]`` is ``HAM_lap_time - VER_lap_time`` on lap ``L`` (positive = VER faster).
    """
    laps = range(1, n_laps + 1)
    positions = {
        L: ({"HAM": 1, "VER": 2} if L < pass_lap else {"VER": 1, "HAM": 2}) for L in laps
    }
    times = {}
    for L in laps:
        gap = deltas.get(L, default_delta)
        base = 90.0
        times[L] = {"HAM": base + gap, "VER": base}
    return _race(positions, times)


# ---------------------------------------------------------------- aggregate helper


def test_aggregate_pace_delta_methods():
    deltas = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert aggregate_pace_delta(deltas, "median") == pytest.approx(3.5)
    assert aggregate_pace_delta(deltas, "mean") == pytest.approx(3.5)
    assert aggregate_pace_delta(deltas, "quantile", 0.10) == pytest.approx(1.5)
    assert aggregate_pace_delta(deltas, "quantile", 0.25) == pytest.approx(2.25)


def test_aggregate_pace_delta_validation():
    with pytest.raises(ValueError):
        aggregate_pace_delta([], "median")
    with pytest.raises(ValueError):
        aggregate_pace_delta([1.0], "mode")
    with pytest.raises(ValueError):
        aggregate_pace_delta([1.0], "quantile", 0.0)


# ---------------------------------------------------------------- smoothing window


def test_detect_excludes_pass_lap():
    """A single DRS-spike lap cannot dominate the reported gap."""
    deltas = {L: 2.0 for L in range(1, 11)}
    deltas[5] = 10.0  # pass-lap spike
    sample = detect_on_track_overtakes(
        _two_car_pass(5, deltas, default_delta=2.0),
        circuit_id="monza",
        year=2024,
    )[0]
    assert sample.pace_delta == pytest.approx(2.0)
    assert sample.lap == 5
    assert sample.window_n == 6  # laps 2,3,4,6,7,8


def test_detect_include_pass_lap_changes_aggregate():
    deltas = {L: 2.0 for L in range(1, 11)}
    deltas[5] = 10.0
    laps = _two_car_pass(5, deltas, default_delta=2.0)

    excluded = detect_on_track_overtakes(
        laps, circuit_id="monza", year=2024, aggregator="mean"
    )[0]
    included = detect_on_track_overtakes(
        laps, circuit_id="monza", year=2024, aggregator="mean", include_pass_lap=True
    )[0]

    assert excluded.window_n == 6
    assert included.window_n == 7
    assert included.pace_delta > excluded.pace_delta
    assert included.pace_delta == pytest.approx((6 * 2.0 + 10.0) / 7)


def test_detect_quantile_aggregator():
    deltas = {L: float(L) for L in range(1, 11)}
    sample = detect_on_track_overtakes(
        _two_car_pass(5, deltas, default_delta=1.0),
        circuit_id="monza",
        year=2024,
        aggregator="quantile",
        quantile=0.10,
    )[0]
    # Window laps 2..8 excluding 5 -> deltas [2,3,4,6,7,8]; 10th pct ~= 2.5.
    assert sample.pace_delta == pytest.approx(2.5)


def test_detect_drops_passes_with_too_few_window_laps():
    laps = _two_car_pass(2, {}, default_delta=3.0)
    samples = detect_on_track_overtakes(
        laps,
        circuit_id="monza",
        year=2024,
        window_before=3,
        window_after=0,
        min_window_laps=4,
    )
    assert samples == []


# ---------------------------------------------------------------- gating / filtering


def test_detect_gates_out_slower_attacker():
    """A position swap where the attacker was not faster over the window is dropped."""
    laps = _two_car_pass(5, {}, default_delta=-2.0)
    assert detect_on_track_overtakes(laps, circuit_id="monza", year=2024) == []


def test_detect_skips_pit_and_non_green_window_laps():
    laps = _two_car_pass(5, {}, default_delta=2.0, n_laps=10)
    # Push the defender through the pits on lap 3 and flag lap 6 non-green.
    laps.loc[(laps["Driver"] == "HAM") & (laps["LapNumber"] == 3), "PitInTime"] = _td(90.0)
    laps.loc[laps["LapNumber"] == 6, "TrackStatus"] = "4"

    sample = detect_on_track_overtakes(laps, circuit_id="monza", year=2024)[0]
    assert sample.window_n == 4  # 2,4,7,8 (3 pit, 5 pass, 6 non-green)


def test_detect_missing_columns_raises():
    laps = _two_car_pass(5, {}).drop(columns=["TrackStatus"])
    with pytest.raises(ValueError, match="missing required columns"):
        detect_on_track_overtakes(laps, circuit_id="monza", year=2024)
