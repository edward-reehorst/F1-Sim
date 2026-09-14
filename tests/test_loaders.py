"""Unit tests for dataset loaders and config builders."""

import pytest

from f1_sim.loaders import (
    build_race_config,
    load_all_circuits,
    load_all_compounds,
    load_all_drivers,
    load_all_teams,
    load_circuit,
    load_compound,
    load_driver,
    load_preset,
    load_team,
)


def test_load_all_circuits() -> None:
    circuits = load_all_circuits()
    assert "monza" in circuits
    assert "silverstone" in circuits
    assert "spa" in circuits
    assert "monaco" in circuits
    assert circuits["monza"].safety_car_pit_loss == 11.5


def test_load_circuit_by_name() -> None:
    monza = load_circuit("monza")
    assert monza.id == "monza"
    assert monza.total_laps == 53


def test_load_circuit_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_circuit("non_existent_track")


def test_load_all_teams() -> None:
    teams = load_all_teams()
    assert len(teams) == 10
    assert "red_bull" in teams
    assert "ferrari" in teams
    assert "mclaren" in teams


def test_load_team_by_id() -> None:
    mclaren = load_team("mclaren")
    assert mclaren.name == "McLaren Formula 1 Team"


def test_load_all_drivers() -> None:
    drivers = load_all_drivers()
    assert len(drivers) == 20
    assert "verstappen" in drivers
    assert "hamilton" in drivers
    assert "norris" in drivers


def test_load_driver_by_id() -> None:
    ver = load_driver("verstappen")
    assert ver.code == "VER"
    assert ver.number == 1


def test_load_all_compounds() -> None:
    compounds = load_all_compounds()
    assert "soft" in compounds
    assert "medium" in compounds
    assert "hard" in compounds
    assert "intermediate" in compounds
    assert "wet" in compounds


def test_load_compound() -> None:
    soft = load_compound("Soft")
    assert soft.compound_name == "Soft"
    assert soft.base_delta == 0.0


def test_load_preset() -> None:
    preset = load_preset("2024_default")
    assert "grid" in preset
    assert len(preset["grid"]) == 20


def test_build_race_config() -> None:
    config = build_race_config(circuit_name_or_circuit="monza", preset_name="2024_default")
    assert config.circuit.id == "monza"
    assert len(config.grid) == 20
    assert config.grid[0].driver_id == "verstappen"
    assert config.laps == 53
    assert config.mandatory_two_compounds is True


def test_build_race_config_wet_preset() -> None:
    config = build_race_config(circuit_name_or_circuit="spa_wet", preset_name="spa_wet")
    assert config.circuit.id == "spa_wet"
    assert config.mandatory_two_compounds is False
    assert config.grid[0].starting_tire == "Intermediate"
