"""Unit tests for Milestone 4: Strategy & Pit Stops."""

import pytest

from f1_sim.engine import CarState
from f1_sim.loaders import load_all_compounds, load_compound, load_driver, load_team
from f1_sim.models.flags import RaceFlag
from f1_sim.strategy import StandardStrategy


@pytest.fixture
def compounds():
    return load_all_compounds()


@pytest.fixture
def verstappen():
    return load_driver("verstappen")


@pytest.fixture
def red_bull():
    return load_team("red_bull")


def test_strategy_tire_cliff_trigger(compounds, verstappen, red_bull):
    """When a car hits its tire cliff, the strategy must immediately trigger a pit stop."""
    soft = compounds["soft"]
    strategy = StandardStrategy()

    car = CarState(
        driver=verstappen,
        team=red_bull,
        current_tire=soft,
        tire_age=soft.cliff_lap,  # Hit cliff!
        fuel_remaining_kg=40.0,
    )

    should_pit, target_tire = strategy.should_pit(
        car=car,
        current_lap=18,
        total_laps=53,
        race_flag=RaceFlag.GREEN,
        available_compounds=compounds,
        mandatory_two_compounds=True,
    )

    assert should_pit is True
    assert target_tire is not None
    assert target_tire.compound_name != "Soft"  # Must switch compound


def test_strategy_safety_car_opportunism(compounds, verstappen, red_bull):
    """Under Safety Car, a car on non-fresh tires should opportunistically pit for cheap delta."""
    medium = compounds["medium"]
    strategy = StandardStrategy()

    car = CarState(
        driver=verstappen,
        team=red_bull,
        current_tire=medium,
        tire_age=12,  # Not fresh, not at cliff yet
        fuel_remaining_kg=40.0,
    )

    # Under green flag: would not pit yet
    green_pit, _ = strategy.should_pit(
        car=car,
        current_lap=15,
        total_laps=53,
        race_flag=RaceFlag.GREEN,
        available_compounds=compounds,
    )
    assert green_pit is False

    # Under Safety Car: takes the cheap pit stop!
    sc_pit, target_tire = strategy.should_pit(
        car=car,
        current_lap=15,
        total_laps=53,
        race_flag=RaceFlag.SAFETY_CAR,
        available_compounds=compounds,
    )
    assert sc_pit is True
    assert target_tire is not None
    assert target_tire.compound_name == "Hard"


def test_strategy_no_pit_on_final_lap(compounds, verstappen, red_bull):
    """Cars should never pit with 1 or fewer laps remaining."""
    soft = compounds["soft"]
    strategy = StandardStrategy()

    car = CarState(
        driver=verstappen,
        team=red_bull,
        current_tire=soft,
        tire_age=soft.cliff_lap + 5,  # Terribly degraded tires
        fuel_remaining_kg=5.0,
    )

    should_pit, _ = strategy.should_pit(
        car=car,
        current_lap=52,
        total_laps=53,  # 1 lap remaining
        race_flag=RaceFlag.GREEN,
        available_compounds=compounds,
    )
    assert should_pit is False


def test_mandatory_two_compound_compliance(compounds, verstappen, red_bull):
    """Strategy must pick an unused compound to comply with the 2-compound rule."""
    medium = compounds["medium"]
    strategy = StandardStrategy()

    car = CarState(
        driver=verstappen,
        team=red_bull,
        current_tire=medium,
        tire_age=25,
        fuel_remaining_kg=30.0,
    )

    target_tire = strategy.select_target_compound(
        car=car,
        remaining_laps=28,
        available_compounds=compounds,
        mandatory_two_compounds=True,
    )

    assert target_tire.compound_name != "Medium"
    assert target_tire.compound_name in ("Hard", "Soft")
