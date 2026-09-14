"""Unit tests for Milestone 4: Race Control, Safety Cars, DNFs, and Regulations."""

import random
import pytest

from f1_sim.engine.car_state import CarState
from f1_sim.engine.race import RaceEngine
from f1_sim.engine.race_control import RaceControlManager
from f1_sim.loaders import build_race_config, load_circuit, load_compound, load_driver, load_team
from f1_sim.models import GridEntry, RaceConfig
from f1_sim.models.flags import RaceFlag


@pytest.fixture
def monza():
    return load_circuit("monza")


@pytest.fixture
def verstappen():
    return load_driver("verstappen")


@pytest.fixture
def red_bull():
    return load_team("red_bull")


@pytest.fixture
def norris():
    return load_driver("norris")


@pytest.fixture
def mclaren():
    return load_team("mclaren")


@pytest.fixture
def medium():
    return load_compound("Medium")


def test_race_control_flag_transition(monza):
    """RaceControlManager must properly count down neutralization laps and restore GREEN."""
    rng = random.Random(42)
    rc = RaceControlManager(circuit=monza, total_laps=50, rng=rng)

    assert rc.current_flag == RaceFlag.GREEN

    rc.current_flag = RaceFlag.SAFETY_CAR
    rc.neutralization_laps_remaining = 2

    # Step 1: Still SC, 1 lap remaining
    flag = rc.update_flag_state()
    assert flag == RaceFlag.SAFETY_CAR
    assert rc.neutralization_laps_remaining == 1

    # Step 2: neutralization expired, returns GREEN
    flag = rc.update_flag_state()
    assert flag == RaceFlag.GREEN
    assert rc.neutralization_laps_remaining == 0


def test_safety_car_pit_loss_reduction(monza):
    """Pitting under Safety Car must yield lower transit loss than green flag transit loss."""
    rng = random.Random(42)
    rc = RaceControlManager(circuit=monza, total_laps=50, rng=rng)

    # Green flag transit loss
    green_loss = rc.get_pit_transit_loss()
    assert green_loss == monza.pit_transit_loss

    # SC transit loss
    rc.current_flag = RaceFlag.SAFETY_CAR
    sc_loss = rc.get_pit_transit_loss()
    assert sc_loss == monza.safety_car_pit_loss
    assert sc_loss < green_loss


def test_safety_car_field_compression(monza, verstappen, red_bull, norris, mclaren, medium):
    """Safety Car must compress large gaps between cars down towards target train spacing."""
    rng = random.Random(42)
    rc = RaceControlManager(circuit=monza, total_laps=50, rng=rng)

    # Car 1 at 100.0s, Car 2 at 115.0s (15s gap!)
    car1 = CarState(driver=verstappen, team=red_bull, current_tire=medium, fuel_remaining_kg=40.0, cumulative_time=100.0)
    car2 = CarState(driver=norris, team=mclaren, current_tire=medium, fuel_remaining_kg=40.0, cumulative_time=115.0)

    cars = [car1, car2]
    initial_gap = car2.cumulative_time - car1.cumulative_time
    assert initial_gap == 15.0

    rc.compress_field_under_safety_car(cars, target_interval_seconds=0.60)

    compressed_gap = car2.cumulative_time - car1.cumulative_time
    # Gap must be substantially reduced, but car 2 still behind car 1
    assert compressed_gap < initial_gap
    assert compressed_gap >= 0.60
    assert car1.cumulative_time == 100.0  # Leader time preserved


def test_mandatory_two_compound_penalty(monza):
    """A finisher who fails to use at least two compounds in a dry race must receive a 30s penalty."""
    grid = [
        GridEntry(driver_id="verstappen", team_id="red_bull", starting_position=1, starting_tire="Medium"),
        GridEntry(driver_id="norris", team_id="mclaren", starting_position=2, starting_tire="Medium"),
    ]
    # 5 laps race where cars do not pit (only use Medium)
    config = RaceConfig(
        circuit=monza,
        grid=grid,
        laps=5,
        mandatory_two_compounds=True,
        seed=42,
    )
    engine = RaceEngine(config, enable_incidents=False)
    # Force cars not to pit by setting empty strategies or running short race
    result = engine.simulate()

    # Both drivers used only Medium, so both received 30s penalties
    assert len(result.driver_summaries[0].compounds_used) == 1
    # 5 laps of Monza is ~410s + 30s penalty = ~440s
    assert result.winner_time > 430.0
