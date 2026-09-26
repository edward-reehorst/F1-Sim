"""Tests for Milestone 7: Automatic Tuning & Calibration (offline core)."""

import csv

import pytest

from f1_sim.engine import RaceEngine, compute_clean_air_lap_time
from f1_sim.loaders import (
    build_race_config,
    load_circuit,
    load_compound,
    load_driver,
    load_team,
)
from f1_sim.models import CalibrationConfig, GridPenalty, GridSlot, LapObservation, TelemetryDataset
from f1_sim.tuning import calibrate_from_dataset, generate_synthetic_dataset, load_dataset

# ---------------------------------------------------------------- fixtures

@pytest.fixture
def overlay() -> CalibrationConfig:
    return CalibrationConfig(
        driver_pace_offsets={"leclerc": 0.40, "sainz": 0.20, "hamilton": 0.15, "russell": 0.10},
        compound_base_deltas={"medium": -0.10, "hard": 0.05},
        compound_wear_multipliers={"soft": 1.20, "medium": 0.95},
        compound_cliff_adjustments={"soft": -3, "hard": 2},
        circuit_base_adjust=-0.35,
        fuel_penalty_per_kg=0.030,
    )


@pytest.fixture
def clean_dataset(overlay) -> TelemetryDataset:
    return generate_synthetic_dataset("monza", overlay=overlay, noise_seconds=0.0)


@pytest.fixture
def long_hard_dataset(overlay) -> TelemetryDataset:
    """Dataset with a 50-lap Hard stint so ages pass the Hard cliff (42 + 2 adjustment)."""
    return generate_synthetic_dataset(
        "monza",
        overlay=overlay,
        noise_seconds=0.0,
        stints=(18, 18, 50),
        compounds=("Soft", "Medium", "Hard"),
    )


# ---------------------------------------------------------------- dataset loading

def _write_csv(path, header, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _aligned_csv(tmp_path, dataset):
    """Write a dataset in a real-world-style aliased schema and return its path."""
    from f1_sim.loaders import load_all_drivers, load_all_teams

    drivers = load_all_drivers()
    teams = load_all_teams()
    path = tmp_path / "monza_export.csv"
    rows = []
    for o in dataset.observations:
        rows.append(
            [
                drivers[o.driver_id].code,
                teams[o.team_id].name,
                o.lap,
                f"{o.lap_time:.3f}",
                o.compound,
                o.tire_age_at_lap_start,
                o.fuel_remaining_kg,
                "GREEN",
                o.position,
                o.session,
            ]
        )
    _write_csv(
        path,
        ["DriverCode", "Team", "LapNumber", "LapTime", "Compound", "TyreLife", "FuelRemaining", "TrackStatus", "Position", "Session"],
        rows,
    )
    return path


def test_load_dataset_aliased_csv(tmp_path, clean_dataset):
    path = _aligned_csv(tmp_path, clean_dataset)
    loaded = load_dataset(path, circuit_id="monza")

    assert len(loaded) == len(clean_dataset)
    assert loaded.circuit_id == "monza"
    first = loaded.observations[0]
    assert first.driver_id == clean_dataset.observations[0].driver_id
    assert first.team_id == clean_dataset.observations[0].team_id
    assert first.compound == clean_dataset.observations[0].compound
    assert first.tire_age_at_lap_start == clean_dataset.observations[0].tire_age_at_lap_start
    assert first.flag == "GREEN"
    assert first.source is not None


def test_load_dataset_self_described_json(tmp_path, clean_dataset):
    path = tmp_path / "monza.json"
    path.write_text(clean_dataset.model_dump_json(), encoding="utf-8")
    loaded = load_dataset(path)
    assert len(loaded) == len(clean_dataset)
    assert loaded.circuit_id == "monza"


def test_load_dataset_missing_columns_raises(tmp_path, clean_dataset):
    path = tmp_path / "missing.csv"
    _write_csv(path, ["DriverCode", "Compound", "LapTime"], [["VER", "Soft", "81.5"]])
    with pytest.raises(ValueError, match="missing required fields"):
        load_dataset(path, circuit_id="monza")


def test_load_dataset_unknown_driver_raises(tmp_path):
    from f1_sim.loaders import load_all_drivers

    codes = sorted(d.code for d in load_all_drivers().values())
    path = tmp_path / "bad_driver.csv"
    _write_csv(
        path,
        ["DriverCode", "Team", "LapNumber", "LapTime", "Compound"],
        [["ZZZ", "Red Bull", 1, "81.5", "Soft"]],
    )
    with pytest.raises(ValueError, match="Unknown driver 'ZZZ'"):
        load_dataset(path, circuit_id="monza")


def test_load_dataset_unknown_compound_raises(tmp_path):
    path = tmp_path / "bad_compound.csv"
    _write_csv(
        path,
        ["DriverCode", "Team", "LapNumber", "LapTime", "Compound"],
        [["VER", "Red Bull", 1, "81.5", "Banana"]],
    )
    with pytest.raises(ValueError, match="Unknown compound 'Banana'"):
        load_dataset(path, circuit_id="monza")


def test_load_dataset_requires_circuit(tmp_path, clean_dataset):
    path = _aligned_csv(tmp_path, clean_dataset)
    with pytest.raises(ValueError, match="circuit"):
        load_dataset(path)


def test_load_dataset_invalid_session_raises():
    from f1_sim.models.telemetry import LapObservation

    with pytest.raises(ValueError, match="Unknown session"):
        LapObservation(driver_id="verstappen", team_id="red_bull", lap=1, compound="Soft", lap_time=81.0, session="bogus")


# ---------------------------------------------------------------- filtering

def _pace_obs(driver_id, lap, compound, lap_time) -> LapObservation:
    return LapObservation(
        driver_id=driver_id,
        team_id="red_bull",
        lap=lap,
        compound=compound,
        lap_time=lap_time,
        session="practice2",
    )


def test_filter_for_pace_isolates_clean_laps_from_traffic():
    # Practice-like noise: a few very slow traffic/aero laps drag the mean, not the spread.
    obs = [
        _pace_obs("a", 1, "Soft", 84.9),
        _pace_obs("a", 2, "Soft", 85.0),
        _pace_obs("a", 3, "Soft", 85.1),
        _pace_obs("b", 1, "Soft", 85.2),
        _pace_obs("b", 2, "Soft", 85.3),
        _pace_obs("b", 3, "Soft", 85.4),
        _pace_obs("c", 1, "Soft", 86.0),
        _pace_obs("c", 2, "Soft", 86.1),
        _pace_obs("c", 3, "Soft", 86.2),
    ]
    traffic = [
        _pace_obs("a", 4, "Soft", 84.9 + 25.0),
        _pace_obs("b", 4, "Soft", 85.2 + 18.0),
        _pace_obs("c", 4, "Soft", 86.0 + 30.0),
    ]
    dataset = TelemetryDataset(circuit_id="monza", observations=obs + traffic)

    paced = dataset.filter_for_pace(buffer_seconds=1.5)
    assert len(paced) == len(obs)
    assert all(t not in paced for t in traffic)


def test_filter_for_pace_uses_per_compound_best():
    # A driver's hard laps (slower than their soft best) still survive: the group
    # best is per (driver, compound), not the driver's overall best.
    obs = [
        _pace_obs("a", 1, "Soft", 84.9),
        _pace_obs("a", 2, "Soft", 85.0),
        _pace_obs("a", 3, "Hard", 86.0),
        _pace_obs("a", 4, "Hard", 86.5),
    ]
    dataset = TelemetryDataset(circuit_id="monza", observations=obs)

    assert len(dataset.filter_for_pace(buffer_seconds=1.0)) == 4
    aged = dataset.filter_for_pace(buffer_seconds=0.4)
    assert all(o.lap != 4 for o in aged)  # Hard lap 86.5 is 0.5s off the Hard best 86.0


def test_filter_for_calibration_removes_sc_and_outliers(clean_dataset):
    # Flag a lap as neutralized and inject a slow outlier lap.
    obs = clean_dataset.observations
    obs[-1] = obs[-1].model_copy(update={"flag": "SAFETY_CAR"})
    slow = obs[0].model_copy(update={"lap_time": obs[0].lap_time + 25.0})
    obs[0] = slow
    dataset = TelemetryDataset(circuit_id="monza", observations=obs)

    filtered = dataset.filter_for_calibration()
    assert len(filtered) == len(clean_dataset) - 2  # slow outlier + SC lap dropped
    assert all(o.flag == "GREEN" for o in filtered)
    assert all(o.lap_time < clean_dataset.observations[0].lap_time + 10.0 for o in filtered)


# ---------------------------------------------------------------- calibration

def test_synthetic_recovery_exact(long_hard_dataset, overlay):
    # albon has no overlay offset and is the reference anchor here.
    report = calibrate_from_dataset(long_hard_dataset, "monza", ridge=0.0, anchor_driver_id="albon")
    cfg = report.config

    assert report.in_sample_rmse < 1e-6
    for key, expected in overlay.driver_pace_offsets.items():
        assert cfg.driver_pace_offsets.get(key, 0.0) == pytest.approx(expected, abs=0.01)
    for key, expected in overlay.compound_base_deltas.items():
        assert cfg.compound_base_deltas.get(key, 0.0) == pytest.approx(expected, abs=0.01)
    for key, expected in overlay.compound_wear_multipliers.items():
        assert cfg.compound_wear_multipliers.get(key, 1.0) == pytest.approx(expected, abs=0.03)
    for key, expected in overlay.compound_cliff_adjustments.items():
        assert cfg.compound_cliff_adjustments.get(key, 0) == expected
    assert cfg.circuit_base_adjust == pytest.approx(overlay.circuit_base_adjust, abs=0.01)
    assert cfg.fuel_penalty_per_kg == pytest.approx(overlay.fuel_penalty_per_kg, abs=0.001)


def test_calibration_noisy_recovery_with_ridge(overlay):
    dataset = generate_synthetic_dataset("monza", overlay=overlay, noise_seconds=0.25)
    report = calibrate_from_dataset(dataset, "monza", ridge=0.5, anchor_driver_id="albon")
    cfg = report.config

    assert report.in_sample_rmse < 0.31
    for key, expected in overlay.driver_pace_offsets.items():
        assert cfg.driver_pace_offsets.get(key, 0.0) == pytest.approx(expected, abs=0.15)
    assert cfg.circuit_base_adjust == pytest.approx(overlay.circuit_base_adjust, abs=0.30)


def test_calibration_determinism(clean_dataset, overlay):
    report_a = calibrate_from_dataset(clean_dataset, "monza", ridge=0.5, seed=99, holdout=0.2)
    report_b = calibrate_from_dataset(clean_dataset, "monza", ridge=0.5, seed=99, holdout=0.2)
    assert report_a.config.model_dump() == report_b.config.model_dump()
    assert report_a.holdout_rmse == report_b.holdout_rmse


def test_calibration_holdout_reporting(clean_dataset, overlay):
    report = calibrate_from_dataset(clean_dataset, "monza", ridge=0.5, holdout=0.2, seed=7)
    assert report.n_holdout_observations > 0
    assert report.holdout_rmse is not None
    assert report.n_fit_observations + report.n_holdout_observations == report.n_observations


def test_calibration_mismatched_circuit_raises(clean_dataset):
    with pytest.raises(ValueError, match="does not match"):
        calibrate_from_dataset(clean_dataset, "silverstone")


def test_calibration_constraints(overlay):
    dataset = generate_synthetic_dataset("monza", overlay=overlay, noise_seconds=0.2)
    report = calibrate_from_dataset(dataset, "monza", ridge=0.5)
    cfg = report.config
    assert all(v >= 0.0 for v in cfg.compound_wear_multipliers.values())
    assert cfg.fuel_penalty_per_kg is None or 0.0 <= cfg.fuel_penalty_per_kg <= 0.10


def test_calibration_anchor_driver_zero(clean_dataset):
    report = calibrate_from_dataset(clean_dataset, "monza", ridge=0.5, anchor_driver_id="albon")
    assert report.anchor_driver_id == "albon"
    assert report.config.driver_pace_offsets["albon"] == 0.0


def test_calibration_only_drivers_not_teams(clean_dataset):
    report = calibrate_from_dataset(clean_dataset, "monza", ridge=0.5)
    assert report.config.team_pace_offsets == {}
    assert "not identifiable" in report.note


# ---------------------------------------------------------------- overlay physics

def test_overlay_applied_to_lap_time(overlay):
    circuit = load_circuit("monza")
    team = load_team("red_bull")
    driver = load_driver("verstappen")
    tire = load_compound("Soft")
    fuel_kg = 60.0
    age = 10

    base_t = compute_clean_air_lap_time(circuit, team, driver, tire, tire_age=age, fuel_mass_kg=fuel_kg)
    cal_t = compute_clean_air_lap_time(
        circuit, team, driver, tire, tire_age=age, fuel_mass_kg=fuel_kg, calibration=overlay
    )
    # Expected overlay effects for soft/verstappen: base -0.35, fuel 0.030 vs 0.033,
    # soft wear multiplier 1.2 (soft base/cliff unchanged at age 10).
    fuel_delta = (overlay.fuel_penalty_per_kg - 0.033) * fuel_kg
    wear_delta = tire.wear_rate * circuit.tire_wear_factor * driver.tire_wear_multiplier * age * 0.20
    expected_delta = overlay.circuit_base_adjust + fuel_delta + wear_delta
    assert cal_t == pytest.approx(base_t + expected_delta, abs=1e-9)


def test_effective_tire_clamps_non_negativity():
    overlay = CalibrationConfig(
        compound_base_deltas={"soft": -5.0},
        compound_wear_multipliers={"soft": 1.0},
        compound_cliff_adjustments={"soft": -100},
    )
    tire = load_compound("Soft")
    effective = overlay.effective_tire(tire)
    assert effective.base_delta >= 0.0
    assert effective.wear_rate >= 0.0
    assert effective.cliff_lap >= 1


def test_calibration_config_json_roundtrip(tmp_path, overlay):
    path = tmp_path / "monza_calibration.json"
    overlay.write_json(path)
    loaded = CalibrationConfig.read_json(path)
    assert loaded.model_dump() == overlay.strip_defaults().model_dump()
    assert loaded.driver_pace_offsets == overlay.driver_pace_offsets


def test_build_race_config_with_calibration(overlay):
    config = build_race_config("monza", seed=42, calibration=overlay)
    assert config.calibration is overlay
    assert config.calibration.circuit_base_adjust == overlay.circuit_base_adjust


# ---------------------------------------------------------------- teammate fallback

def _grid_roster() -> list[tuple[str, str]]:
    return [
        ("verstappen", "red_bull"),
        ("perez", "red_bull"),
        ("norris", "mclaren"),
        ("piastri", "mclaren"),
        ("tsunoda", "rb"),
        ("ricciardo", "rb"),
    ]


def test_teammate_fallback_fills_missing_drivers():
    cal = CalibrationConfig(driver_pace_offsets={"verstappen": 0.80, "norris": 0.30, "piastri": 0.50})
    filled = cal.with_teammate_fallbacks(_grid_roster())
    assert filled.driver_offset("perez") == pytest.approx(0.80)
    assert filled.driver_offset("norris") == pytest.approx(0.30)
    assert filled.driver_offset("piastri") == pytest.approx(0.50)
    assert filled.driver_offset("tsunoda") == 0.0  # RB has no calibrated driver -> no fallback
    assert filled.driver_offset("ricciardo") == 0.0
    assert "perez" not in cal.driver_pace_offsets  # receiver untouched


def test_teammate_fallback_uses_mean_of_available_teammates():
    cal = CalibrationConfig(driver_pace_offsets={"a": 1.0, "b": 3.0})
    filled = cal.with_teammate_fallbacks([("a", "t"), ("b", "t"), ("c", "t"), ("d", "t")])
    assert filled.driver_offset("c") == pytest.approx(2.0)
    assert filled.driver_offset("d") == pytest.approx(2.0)
    assert cal.driver_offset("c") == 0.0


def test_teammate_fallback_returns_self_when_complete(overlay):
    fresh = CalibrationConfig(driver_pace_offsets={"a": 1.0, "b": 2.0})
    assert fresh.with_teammate_fallbacks([("a", "t"), ("b", "t")]) is fresh


def test_build_race_config_applies_teammate_fallback():
    cal = CalibrationConfig(driver_pace_offsets={"verstappen": -0.50})
    config = build_race_config("monza", calibration=cal)
    assert config.calibration.driver_offset("perez") == pytest.approx(-0.50)


def test_write_json_preserves_zero_anchor_teammate_fallback(tmp_path):
    # The fitted anchor (0.0s) must survive serialization, otherwise teammate
    # fallback would replace it with a teammate's offset after a round-trip.
    cal = CalibrationConfig(driver_pace_offsets={"leclerc": 0.0, "sainz": 0.50})
    path = tmp_path / "monza_cal.json"
    cal.write_json(path)
    loaded = CalibrationConfig.read_json(path)
    assert loaded.driver_pace_offsets == {"leclerc": 0.0, "sainz": 0.50}
    config = build_race_config("monza", calibration=loaded)
    assert config.calibration.driver_offset("leclerc") == 0.0
    assert config.calibration.driver_offset("sainz") == pytest.approx(0.50)


# ---------------------------------------------------------------- qualifying grid

def _qualifying_dataset() -> TelemetryDataset:
    spec = [
        ("gasly", "alpine", "GAS", 1, 80.0),
        ("russell", "mercedes", "RUS", 2, 80.4),
        ("piastri", "mclaren", "PIA", 3, 80.6),
        ("leclerc", "ferrari", "LEC", 4, 80.9),
    ]
    obs = []
    for driver_id, team_id, code, pos, lap_time in spec:
        obs.append(
            LapObservation(
                session="qualifying",
                driver_id=driver_id,
                team_id=team_id,
                lap=1,
                compound="Soft",
                lap_time=lap_time,
                flag="GREEN",
                position=pos,
                source={"driver_code": code},
            )
        )
    return TelemetryDataset(circuit_id="monza", observations=obs)


def test_calibration_captures_qualifying_starting_grid():
    report = calibrate_from_dataset(_qualifying_dataset(), circuit_id="monza")
    assert report.config.starting_grid == [
        GridSlot(position=1, driver_id="gasly", team_id="alpine"),
        GridSlot(position=2, driver_id="russell", team_id="mercedes"),
        GridSlot(position=3, driver_id="piastri", team_id="mclaren"),
        GridSlot(position=4, driver_id="leclerc", team_id="ferrari"),
    ]


def test_calibration_applies_grid_penalties_by_driver_code():
    # gasly is the pole-sitter; a 3-place penalty drops him to slot 4 and the
    # rest of the field moves up one.
    report = calibrate_from_dataset(
        _qualifying_dataset(),
        circuit_id="monza",
        grid_penalties=[GridPenalty(driver_id="GAS", places=3)],
    )
    assert report.config.starting_grid == [
        GridSlot(position=1, driver_id="russell", team_id="mercedes"),
        GridSlot(position=2, driver_id="piastri", team_id="mclaren"),
        GridSlot(position=3, driver_id="leclerc", team_id="ferrari"),
        GridSlot(position=4, driver_id="gasly", team_id="alpine"),
    ]


def test_calibration_applies_grid_penalties_by_slug():
    report = calibrate_from_dataset(
        _qualifying_dataset(),
        circuit_id="monza",
        grid_penalties=[GridPenalty(driver_id="russell", places=0)],
    )
    positions = {slot.driver_id: slot.position for slot in report.config.starting_grid}
    assert positions["russell"] == 4  # back of the 4-car field
    assert positions["gasly"] == 1
    assert positions["leclerc"] == 3


def test_calibration_keeps_preset_grid_for_non_qualifying_sessions():
    obs = [
        LapObservation(session="practice2", driver_id="verstappen", team_id="red_bull", lap=1, compound="Soft", lap_time=81.0, flag="GREEN", position=None),
        LapObservation(session="practice2", driver_id="verstappen", team_id="red_bull", lap=2, compound="Soft", lap_time=81.1, flag="GREEN", position=None),
        LapObservation(session="practice2", driver_id="leclerc", team_id="ferrari", lap=1, compound="Soft", lap_time=81.4, flag="GREEN", position=None),
        LapObservation(session="practice2", driver_id="leclerc", team_id="ferrari", lap=2, compound="Soft", lap_time=81.5, flag="GREEN", position=None),
    ]
    dataset = TelemetryDataset(circuit_id="monza", observations=obs)
    report = calibrate_from_dataset(dataset, circuit_id="monza")
    assert report.config.starting_grid == []


def test_build_race_config_uses_qualifying_grid_and_appends_rest():
    cal = CalibrationConfig(
        starting_grid=[
            GridSlot(position=1, driver_id="gasly", team_id="alpine"),
            GridSlot(position=2, driver_id="russell", team_id="mercedes"),
        ]
    )
    config = build_race_config("monza", calibration=cal)
    grid = config.grid
    assert len(grid) == 20
    front = grid[:3]
    assert [(e.driver_id, e.team_id, e.starting_position) for e in front[:2]] == [
        ("gasly", "alpine", 1),
        ("russell", "mercedes", 2),
    ]
    covered = {e.driver_id for e in grid[:2]}
    tail = [e for e in grid[2:] if e.driver_id not in covered]
    assert tail[0].starting_position == 3
    assert all(e.driver_id in covered or e.starting_position >= 3 for e in grid)


def test_build_race_config_teammate_fallback_uses_qualifying_pairs():
    # Hamilton is at Ferrari in the grid data, so his missing offset must inherit
    # his Ferrari teammate (leclerc), not the bundled Mercedes pairing.
    cal = CalibrationConfig(
        driver_pace_offsets={"leclerc": 0.0, "russell": -0.2},
        starting_grid=[
            GridSlot(position=1, driver_id="leclerc", team_id="ferrari"),
            GridSlot(position=2, driver_id="hamilton", team_id="ferrari"),
            GridSlot(position=3, driver_id="russell", team_id="mercedes"),
        ],
    )
    config = build_race_config("monza", calibration=cal)
    assert config.calibration.driver_offset("hamilton") == pytest.approx(0.0)
    config_grid = {e.driver_id: e.team_id for e in config.grid}
    assert config_grid["hamilton"] == "ferrari"


def test_calibration_grid_survives_json_roundtrip(tmp_path):
    cal = CalibrationConfig(
        starting_grid=[GridSlot(position=1, driver_id="gasly", team_id="alpine")],
        driver_pace_offsets={"gasly": 0.3},
    )
    path = tmp_path / "quali_cal.json"
    cal.write_json(path)
    loaded = CalibrationConfig.read_json(path)
    assert loaded.starting_grid == cal.starting_grid


def test_race_with_calibration_deterministic(overlay):
    results = []
    for _ in range(2):
        config = build_race_config("monza", seed=11, calibration=overlay)
        engine = RaceEngine(config, enable_incidents=False)
        results.append(engine.simulate())
    assert results[0].winner_time == results[1].winner_time
    assert results[0].winner_id == results[1].winner_id