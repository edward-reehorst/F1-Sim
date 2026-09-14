"""Unit tests for Milestone 2: Core Lap Physics Engine."""

import pytest

from f1_sim.engine import (
    CarState,
    compute_clean_air_lap_time,
    compute_clean_air_lap_time_breakdown,
    compute_fuel_delta,
    compute_tire_delta,
    run_time_trial,
)
from f1_sim.loaders import (
    load_circuit,
    load_compound,
    load_driver,
    load_team,
)


@pytest.fixture
def monza_circuit():
    return load_circuit("monza")


@pytest.fixture
def red_bull():
    return load_team("red_bull")


@pytest.fixture
def sauber():
    return load_team("sauber")


@pytest.fixture
def verstappen():
    return load_driver("verstappen")


@pytest.fixture
def sargeant():
    return load_driver("sargeant")


@pytest.fixture
def soft_tire():
    return load_compound("Soft")


@pytest.fixture
def medium_tire():
    return load_compound("Medium")


@pytest.fixture
def hard_tire():
    return load_compound("Hard")


def test_clean_air_determinism(monza_circuit, red_bull, verstappen, soft_tire):
    """Running identical inputs in clean air must yield bitwise identical lap times."""
    t1 = compute_clean_air_lap_time(
        circuit=monza_circuit,
        team=red_bull,
        driver=verstappen,
        tire=soft_tire,
        tire_age=5,
        fuel_mass_kg=50.0,
    )
    t2 = compute_clean_air_lap_time(
        circuit=monza_circuit,
        team=red_bull,
        driver=verstappen,
        tire=soft_tire,
        tire_age=5,
        fuel_mass_kg=50.0,
    )
    assert t1 == t2


def test_fuel_monotonicity():
    """Higher fuel mass must strictly and monotonically increase lap time penalty."""
    f0 = compute_fuel_delta(0.0)
    f50 = compute_fuel_delta(50.0)
    f100 = compute_fuel_delta(100.0)

    assert f0 == 0.0
    assert f50 > f0
    assert f100 > f50
    assert f100 == pytest.approx(100.0 * 0.033)


def test_tire_degradation_monotonicity(soft_tire):
    """Tire degradation must increase monotonically with age."""
    d0 = compute_tire_delta(soft_tire, tire_age=0)
    d5 = compute_tire_delta(soft_tire, tire_age=5)
    d15 = compute_tire_delta(soft_tire, tire_age=15)
    d20 = compute_tire_delta(soft_tire, tire_age=20)  # past cliff (18)

    assert d0 == 0.0
    assert d5 > d0
    assert d15 > d5
    assert d20 > d15
    # Past cliff should be significantly higher due to quadratic term
    linear_rate = (d15 - d5) / 10.0
    cliff_rate = (d20 - d15) / 5.0
    assert cliff_rate > linear_rate


def test_driver_tire_management_effect(soft_tire, verstappen, sargeant):
    """A driver with better tire management must experience lower tire degradation."""
    # Verstappen has tire_management 94 vs Sargeant 82
    assert verstappen.tire_management > sargeant.tire_management
    assert verstappen.tire_wear_multiplier < sargeant.tire_wear_multiplier

    deg_ver = compute_tire_delta(
        soft_tire,
        tire_age=10,
        driver_wear_multiplier=verstappen.tire_wear_multiplier,
    )
    deg_sar = compute_tire_delta(
        soft_tire,
        tire_age=10,
        driver_wear_multiplier=sargeant.tire_wear_multiplier,
    )

    assert deg_ver < deg_sar


def test_lap_time_breakdown_summation(monza_circuit, red_bull, verstappen, medium_tire):
    """All components in LapTimeBreakdown must sum exactly to total_time."""
    bd = compute_clean_air_lap_time_breakdown(
        circuit=monza_circuit,
        team=red_bull,
        driver=verstappen,
        tire=medium_tire,
        tire_age=8,
        fuel_mass_kg=40.0,
    )

    expected_sum = (
        bd.base_time + bd.car_delta + bd.driver_delta + bd.fuel_delta + bd.tire_delta
    )
    assert bd.total_time == pytest.approx(expected_sum)


def test_team_performance_hierarchy(red_bull, sauber):
    """Top team must have lower car_delta_seconds than backmarker."""
    assert red_bull.car_delta_seconds < sauber.car_delta_seconds


def test_driver_pace_hierarchy(verstappen, sargeant):
    """Top driver must have lower pace_delta_seconds than backmarker."""
    assert verstappen.pace_delta_seconds < sargeant.pace_delta_seconds


def test_compound_grip_hierarchy(soft_tire, medium_tire, hard_tire):
    """Fresh Soft must be faster than fresh Medium, which is faster than fresh Hard."""
    assert soft_tire.degradation_at_age(0) < medium_tire.degradation_at_age(0)
    assert medium_tire.degradation_at_age(0) < hard_tire.degradation_at_age(0)


def test_car_state_simulation_advancement(monza_circuit, red_bull, verstappen, soft_tire):
    """CarState must properly track fuel burn, tire wear, and cumulative time."""
    initial_fuel = 30.0
    car = CarState(
        driver=verstappen,
        team=red_bull,
        current_tire=soft_tire,
        fuel_remaining_kg=initial_fuel,
    )

    assert car.laps_completed == 0
    assert car.tire_age == 0

    rec, bd = car.simulate_clean_air_lap(monza_circuit)

    assert car.laps_completed == 1
    assert car.tire_age == 1
    assert car.fuel_remaining_kg == pytest.approx(initial_fuel - monza_circuit.fuel_burn_per_lap)
    assert car.cumulative_time == rec.lap_time
    assert len(car.lap_records) == 1


def test_car_state_pit_stop(verstappen, red_bull, soft_tire, hard_tire):
    """CarState fit_new_tires must reset tire age, update compound, and record pit stop."""
    car = CarState(
        driver=verstappen,
        team=red_bull,
        current_tire=soft_tire,
        tire_age=15,
        fuel_remaining_kg=20.0,
    )

    pit_record = car.fit_new_tires(
        new_tire=hard_tire,
        stationary_time=2.4,
        transit_loss=21.0,
        lap_number=16,
    )

    assert car.current_tire.compound_name == "Hard"
    assert car.tire_age == 0
    assert len(car.pit_stops) == 1
    assert pit_record.compound_in == "Soft"
    assert pit_record.compound_out == "Hard"
    assert pit_record.total_pit_loss == pytest.approx(23.4)
    assert "Hard" in car.compounds_used


def test_time_trial_execution(monza_circuit, red_bull, verstappen, soft_tire):
    """run_time_trial must run specified laps and compute correct summary totals."""
    laps_to_run = 12
    result = run_time_trial(
        circuit=monza_circuit,
        team=red_bull,
        driver=verstappen,
        tire=soft_tire,
        laps=laps_to_run,
    )

    assert result.total_laps == laps_to_run
    assert len(result.lap_records) == laps_to_run
    assert len(result.breakdowns) == laps_to_run

    sum_lap_times = sum(rec.lap_time for rec in result.lap_records)
    assert result.total_time == pytest.approx(sum_lap_times)
    assert result.fastest_lap_time <= min(rec.lap_time for rec in result.lap_records)
