"""Unit tests for f1-sim data models."""

import pytest
from pydantic import ValidationError

from f1_sim.models import (
    Circuit,
    Driver,
    GridEntry,
    RaceConfig,
    Team,
    TireCompound,
)


def test_driver_model_valid() -> None:
    driver = Driver(
        id="verstappen",
        name="Max Verstappen",
        code="VER",
        number=1,
        pace_rating=98.0,
        tire_management=94.0,
    )
    assert driver.id == "verstappen"
    assert driver.code == "VER"
    assert driver.pace_delta_seconds == pytest.approx((100.0 - 98.0) * 0.020)
    assert driver.tire_wear_multiplier < 1.0


def test_driver_model_code_validation() -> None:
    with pytest.raises(ValidationError):
        Driver(
            id="test",
            name="Test Driver",
            code="INVALID_CODE",
            number=99,
            pace_rating=90.0,
        )


def test_team_model() -> None:
    team = Team(
        id="ferrari",
        name="Scuderia Ferrari",
        engine_power=96.0,
        aero_efficiency=95.0,
        mechanical_grip=94.0,
        reliability=0.98,
        pit_crew_speed=2.3,
    )
    assert team.composite_car_rating > 90.0
    assert team.car_delta_seconds >= 0.0


def test_circuit_model() -> None:
    circuit = Circuit(
        id="monza",
        name="Monza",
        country="Italy",
        total_laps=53,
        base_lap_time=81.0,
        overtake_threshold_seconds=0.7031,
        tire_wear_factor=0.90,
        pit_transit_loss=21.0,
        safety_car_pit_loss=11.5,
        fuel_burn_per_lap=1.85,
    )
    assert circuit.total_laps == 53
    assert circuit.safety_car_pit_loss < circuit.pit_transit_loss


def test_tire_compound_degradation() -> None:
    soft = TireCompound(
        compound_name="Soft",
        color_code="red",
        base_delta=0.0,
        wear_rate=0.08,
        cliff_lap=15,
        cliff_penalty_factor=0.15,
    )
    # Age 0 has base delta
    assert soft.degradation_at_age(0) == 0.0

    # Before cliff: linear wear
    deg_lap_10 = soft.degradation_at_age(10)
    assert deg_lap_10 == pytest.approx(0.08 * 10)

    # After cliff: quadratic spike
    deg_lap_17 = soft.degradation_at_age(17)
    expected_cliff = 0.15 * (2**2)  # 2 laps past 15
    assert deg_lap_17 == pytest.approx((0.08 * 17) + expected_cliff)


def test_race_config_defaults() -> None:
    circuit = Circuit(
        id="monaco",
        name="Monaco",
        country="Monaco",
        total_laps=78,
        base_lap_time=74.0,
        overtake_threshold_seconds=3.5,
        tire_wear_factor=0.65,
        pit_transit_loss=20.0,
        safety_car_pit_loss=10.5,
        fuel_burn_per_lap=1.5,
    )
    grid = [
        GridEntry(driver_id="verstappen", team_id="red_bull", starting_position=1),
        GridEntry(driver_id="leclerc", team_id="ferrari", starting_position=2),
    ]
    config = RaceConfig(circuit=circuit, grid=grid)
    # Should resolve defaults
    assert config.laps == 78
    assert config.initial_fuel_kg == pytest.approx((78 * 1.5) + 5.0)
    assert config.mandatory_two_compounds is True
