"""Tests for the FastF1 ingestion mapping (pure, offline)."""

from __future__ import annotations

import pandas as pd
import pytest

from f1_sim.models.calibration import GridPenalty
from f1_sim.tuning.dataset import _Lookups
from f1_sim.tuning.fastf1 import (
    _fill_qualifying_positions,
    _laps_to_dataset,
    _resolve_circuit_id,
    _session_kind,
    apply_grid_penalties,
    grid_positions_by_code,
    lap_time_seconds,
    unknown_driver_codes,
    unknown_team_names,
)


def _td(seconds: float) -> pd.Timedelta:
    return pd.to_timedelta(seconds, unit="s")


def _laps_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Driver": ["VER", "LEC", "VER", "NOR"],
            "Team": [
                "Red Bull Racing",
                "Scuderia Ferrari",
                "Red Bull Racing",
                "McLaren Formula 1 Team",
            ],
            "LapNumber": [1, 1, 2, 1],
            "LapTime": [_td(81.100), _td(81.500), _td(80.900), _td(81.300)],
            "Compound": ["SOFT", "MEDIUM", "SOFT", "HARD"],
            "TyreLife": [1.0, 4.0, 2.0, 3.0],
            "Stint": [1.0, 1.0, 1.0, 2.0],
            "Position": [1.0, 2.0, 1.0, 3.0],
        }
    )


def test_laps_to_dataset_mapping():
    dataset = _laps_to_dataset(
        _laps_frame(),
        circuit_id="monza",
        session_kind="practice1",
        lookups=_Lookups(),
    )
    assert dataset.circuit_id == "monza"
    assert len(dataset) == 4
    assert all(o.flag == "GREEN" for o in dataset.observations)
    assert all(o.session == "practice1" for o in dataset.observations)

    first = dataset.observations[0]
    assert first.driver_id == "verstappen"
    assert first.team_id == "red_bull"
    assert first.compound == "Soft"
    assert first.lap == 1
    assert first.lap_time == pytest.approx(81.100)
    assert first.tire_age_at_lap_start == 1
    assert first.stint == 1
    assert first.position == 1


def test_laps_to_dataset_skips_missing_lap_times():
    frame = pd.DataFrame(
        {
            "Driver": ["VER", "LEC", "ZHO"],
            "Team": ["Red Bull Racing", "Scuderia Ferrari", "Kick Sauber"],
            "LapNumber": [1, 1, 99],
            "LapTime": [_td(81.100), _td(81.500), pd.NaT],
            "Compound": ["SOFT", "SOFT", "SOFT"],
            "TyreLife": [1.0, 1.0, 1.0],
            "Stint": [1.0, 1.0, 1.0],
            "Position": [1.0, 2.0, None],
        }
    )
    dataset = _laps_to_dataset(frame, circuit_id="monza", session_kind="race", lookups=_Lookups())
    assert len(dataset) == 2
    assert [o.driver_id for o in dataset.observations] == ["verstappen", "leclerc"]


def test_laps_to_dataset_unknown_driver_raises():
    frame = _laps_frame()
    frame.loc[0, "Driver"] = "ZZZ"
    lookups = _Lookups()
    with pytest.raises(ValueError, match="Unknown driver"):
        _laps_to_dataset(frame, circuit_id="monza", session_kind="race", lookups=lookups)


def test_laps_to_dataset_skips_unknown_driver_when_requested():
    frame = _laps_frame()
    frame.loc[0, "Driver"] = "COL"
    dataset = _laps_to_dataset(
        frame,
        circuit_id="monza",
        session_kind="race",
        lookups=_Lookups(),
        skip_unknown_drivers=True,
    )
    assert [o.driver_id for o in dataset.observations] == ["leclerc", "verstappen", "norris"]


def test_unknown_driver_codes_reports_unknowns():
    frame = _laps_frame()
    frame.loc[0, "Driver"] = "COL"
    assert unknown_driver_codes(frame) == ["COL"]


def test_unknown_team_names_reports_unknowns():
    frame = _laps_frame()
    frame.loc[0, "Team"] = "Cadillac"
    assert unknown_team_names(frame) == ["Cadillac"]


def test_laps_to_dataset_unknown_team_raises():
    frame = _laps_frame()
    frame.loc[0, "Team"] = "Cadillac"
    with pytest.raises(ValueError, match="Unknown team"):
        _laps_to_dataset(frame, circuit_id="monza", session_kind="race", lookups=_Lookups())


def test_laps_to_dataset_skips_unknown_team_when_requested():
    frame = _laps_frame()
    frame.loc[0, "Team"] = "Cadillac"
    dataset = _laps_to_dataset(
        frame,
        circuit_id="monza",
        session_kind="race",
        lookups=_Lookups(),
        skip_unknown_teams=True,
    )
    assert [o.driver_id for o in dataset.observations] == ["leclerc", "verstappen", "norris"]


def test_session_kind_mapping():
    assert _session_kind("R") == ("R", "race")
    assert _session_kind("race") == ("R", "race")
    assert _session_kind("FP1") == ("FP1", "practice1")
    assert _session_kind("fp3") == ("FP3", "practice3")
    assert _session_kind("Q") == ("Q", "qualifying")


def test_session_kind_rejects_sprint():
    with pytest.raises(ValueError, match="FP1"):
        _session_kind("SS")
    with pytest.raises(ValueError, match="FP1"):
        _session_kind("Sprint")


def test_circuit_resolution_maps_bundled_tracks():
    assert _resolve_circuit_id({"Location": "Monza", "Country": "Italy"}, None) == "monza"
    assert _resolve_circuit_id({"Location": "Silverstone", "Country": "Great Britain"}, None) == "silverstone"
    assert _resolve_circuit_id({"Location": "Spa", "Country": "Belgium"}, None) == "spa"
    assert _resolve_circuit_id({"Location": "Monte Carlo", "Country": "Monaco"}, None) == "monaco"


def test_circuit_resolution_override_takes_precedence():
    assert _resolve_circuit_id({"Location": "Spa", "Country": "Belgium"}, "silverstone") == "silverstone"


def test_circuit_resolution_unknown_raises():
    with pytest.raises(ValueError, match="pass --circuit"):
        _resolve_circuit_id({"Location": "Suzuka", "Country": "Japan"}, None)


def test_fill_qualifying_positions_sets_grid_order():
    laps = pd.DataFrame(
        {
            "Driver": ["GAS", "RUS", "PIA", "GAS", "LEC"],
            "Team": ["Alpine", "Mercedes", "McLaren", "Alpine", "Ferrari"],
            "LapNumber": [1, 1, 1, 2, 1],
            "LapTime": [_td(80.0), _td(80.4), _td(80.6), _td(80.1), _td(80.9)],
        }
    )
    positions = {"GAS": 1, "RUS": 2, "PIA": 3, "LEC": 4}
    filled = _fill_qualifying_positions(laps, positions)
    times = {r["Driver"]: int(r["Position"]) for _, r in filled.iterrows()}
    assert times == {"GAS": 1, "RUS": 2, "PIA": 3, "LEC": 4}


def test_grid_positions_with_penalties():
    # Antonelli qualified P7 but starts P19 (grid penalty): GridPosition wins.
    # GAS has no GridPosition -> skipped (not a race-session grid entry).
    results = pd.DataFrame(
        {
            "Abbreviation": ["ANT", "RUS", "VER", "GAS"],
            "Position": [7.0, 2.0, 3.0, 1.0],
            "GridPosition": [19.0, 2.0, 5.0, float("nan")],
        }
    )
    assert grid_positions_by_code(results) == {"ANT": 19, "VER": 5, "RUS": 2}


def test_grid_positions_empty_when_unpopulated():
    # Qualifying results carry an all-NaN GridPosition -> empty, signalling the
    # caller to fall back to the classification grid.
    results = pd.DataFrame(
        {
            "Abbreviation": ["GAS", "RUS", "PIA"],
            "Position": [1.0, 2.0, 3.0],
            "GridPosition": [float("nan"), float("nan"), float("nan")],
        }
    )
    assert grid_positions_by_code(results) == {}


def test_grid_positions_classification_fallback():
    # The pre-penalty grid is read from the classification Position column.
    results = pd.DataFrame(
        {
            "Abbreviation": ["GAS", "RUS", "PIA"],
            "Position": [1.0, 2.0, 3.0],
            "GridPosition": [float("nan"), float("nan"), float("nan")],
        }
    )
    assert grid_positions_by_code(results, grid_position=False) == {"GAS": 1, "RUS": 2, "PIA": 3}


def test_laps_to_dataset_carries_qualifying_position():
    laps = pd.DataFrame(
        {
            "Driver": ["VER", "LEC", "VER"],
            "Team": ["Red Bull Racing", "Scuderia Ferrari", "Red Bull Racing"],
            "LapNumber": [1, 1, 2],
            "LapTime": [_td(81.0), _td(81.4), _td(81.1)],
            "Position": [1.0, 2.0, 1.0],
            "Compound": ["SOFT", "SOFT", "SOFT"],
        }
    )
    dataset = _laps_to_dataset(laps, circuit_id="monza", session_kind="qualifying", lookups=_Lookups())
    positions = {o.driver_id: o.position for o in dataset.observations}
    assert positions == {"verstappen": 1, "leclerc": 2}


def test_lap_time_seconds_coerces_timedelta_and_float():
    assert lap_time_seconds(_td(81.100)) == pytest.approx(81.100)
    assert lap_time_seconds(81.100) == pytest.approx(81.100)


# --- grid penalty application -------------------------------------------------


def test_apply_grid_penalties_no_penalties_unchanged():
    classification = {"GAS": 1, "RUS": 2, "PIA": 3, "LEC": 4, "HAM": 5}
    assert apply_grid_penalties(classification, []) == dict(classification)


def test_apply_grid_penalties_simple_drop_moves_field_up():
    # Pole-sitter drops 10 places; the rest move up to fill the gap.
    classification = {f"D{i}": i for i in range(1, 21)}  # noqa: C416
    cleared = apply_grid_penalties(classification, [GridPenalty(driver_id="D1", places=10)])
    assert cleared["D1"] == 11
    assert cleared["D2"] == 1
    assert cleared["D3"] == 2
    assert cleared["D11"] == 10
    assert len(set(cleared.values())) == len(cleared)


def test_apply_grid_penalties_cumulative_stack():
    classification = {**{f"D{i}": i for i in range(4, 13)}, "VER": 1, "LEC": 2, "NOR": 3}
    cleared = apply_grid_penalties(
        classification,
        [GridPenalty(driver_id="VER", places=5), GridPenalty(driver_id="VER", places=3)],
    )
    assert cleared["VER"] == 9  # 1 + 5 + 3
    assert cleared["LEC"] == 1
    assert cleared["NOR"] == 2


def test_apply_grid_penalties_back_of_grid_places_zero():
    classification = {"VER": 1, "LEC": 2, "NOR": 3, "PIA": 4, "HAM": 5}
    cleared = apply_grid_penalties(classification, [GridPenalty(driver_id="VER", places=0)])
    assert cleared["VER"] == 5  # back of 5-car grid
    assert cleared["LEC"] == 1


def test_apply_grid_penalties_drop_past_grid_clamped_to_back():
    classification = {f"D{i}": i for i in range(1, 11)}
    cleared = apply_grid_penalties(classification, [GridPenalty(driver_id="D1", places=999)])
    assert cleared["D1"] == 10  # clamped to back


def test_apply_grid_penalties_collision_resolved_by_quali_order():
    # Two drivers both land on slot 5; faster qualifier (D2) keeps it, D7 moves back.
    classification = {"D1": 1, "D2": 2, "D3": 3, "D4": 4, "D5": 5, "D6": 6, "D7": 7}
    cleared = apply_grid_penalties(
        classification,
        [GridPenalty(driver_id="D2", places=3), GridPenalty(driver_id="D7", places=0)],
    )
    assert cleared["D2"] == 5
    assert cleared["D7"] == 7  # bumped to the next free slot behind
    assert len(set(cleared.values())) == len(cleared)


def test_apply_grid_penalties_ignores_absent_driver():
    classification = {"VER": 1, "LEC": 2}
    cleared = apply_grid_penalties(classification, [GridPenalty(driver_id="ZHO", places=5)])
    assert cleared == classification


def test_apply_grid_penalties_unique_slots():
    classification = {f"D{i}": i for i in range(1, 23)}
    cleared = apply_grid_penalties(classification, [GridPenalty(driver_id="D1", places=0)])
    assert len(set(cleared.values())) == 22
    assert min(cleared.values()) == 1
    assert max(cleared.values()) == 22