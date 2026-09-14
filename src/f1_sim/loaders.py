"""Data loader utilities for circuits, teams, drivers, compounds, and race presets."""

import json
from pathlib import Path

from f1_sim.models.circuit import Circuit
from f1_sim.models.config import GridEntry, RaceConfig
from f1_sim.models.driver import Driver
from f1_sim.models.team import Team
from f1_sim.models.tire import TireCompound


def get_data_dir() -> Path:
    """Return the absolute path to the bundled package data directory."""
    return Path(__file__).resolve().parent / "data"


def load_all_circuits() -> dict[str, Circuit]:
    """Load all bundled circuits keyed by circuit id."""
    circuits_dir = get_data_dir() / "circuits"
    circuits: dict[str, Circuit] = {}
    for filepath in sorted(circuits_dir.glob("*.json")):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            circuit = Circuit.model_validate(data)
            circuits[circuit.id] = circuit
    return circuits


def load_circuit(name_or_path: str | Path) -> Circuit:
    """Load a circuit by identifier (e.g. 'monza') or from an explicit file path."""
    path = Path(name_or_path)
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            return Circuit.model_validate(json.load(f))

    # Look up in bundled circuits
    bundled_path = get_data_dir() / "circuits" / f"{name_or_path}.json"
    if bundled_path.is_file():
        with open(bundled_path, "r", encoding="utf-8") as f:
            return Circuit.model_validate(json.load(f))

    available = [p.stem for p in (get_data_dir() / "circuits").glob("*.json")]
    raise FileNotFoundError(
        f"Circuit '{name_or_path}' not found. Available bundled circuits: {available}"
    )


def load_all_teams() -> dict[str, Team]:
    """Load all bundled teams keyed by team id."""
    teams_file = get_data_dir() / "teams.json"
    with open(teams_file, "r", encoding="utf-8") as f:
        teams_data = json.load(f)
    return {item["id"]: Team.model_validate(item) for item in teams_data}


def load_team(team_id: str) -> Team:
    """Load a specific team by id."""
    teams = load_all_teams()
    if team_id not in teams:
        raise KeyError(f"Team '{team_id}' not found. Available teams: {list(teams.keys())}")
    return teams[team_id]


def load_all_drivers() -> dict[str, Driver]:
    """Load all bundled drivers keyed by driver id."""
    drivers_file = get_data_dir() / "drivers.json"
    with open(drivers_file, "r", encoding="utf-8") as f:
        drivers_data = json.load(f)
    return {item["id"]: Driver.model_validate(item) for item in drivers_data}


def load_driver(driver_id: str) -> Driver:
    """Load a specific driver by id."""
    drivers = load_all_drivers()
    if driver_id not in drivers:
        raise KeyError(f"Driver '{driver_id}' not found. Available drivers: {list(drivers.keys())}")
    return drivers[driver_id]


def load_all_compounds() -> dict[str, TireCompound]:
    """Load all bundled tire compounds keyed by lowercase compound name."""
    compounds_file = get_data_dir() / "compounds.json"
    with open(compounds_file, "r", encoding="utf-8") as f:
        compounds_data = json.load(f)
    return {item["compound_name"].lower(): TireCompound.model_validate(item) for item in compounds_data}


def load_compound(compound_name: str) -> TireCompound:
    """Load a specific tire compound by name (case-insensitive)."""
    compounds = load_all_compounds()
    key = compound_name.strip().lower()
    if key not in compounds:
        raise KeyError(f"Compound '{compound_name}' not found. Available: {list(compounds.keys())}")
    return compounds[key]


def load_preset(preset_name_or_path: str | Path) -> dict:
    """Load a preset dictionary by name (e.g. '2024_default') or from a file path."""
    path = Path(preset_name_or_path)
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    bundled_path = get_data_dir() / "presets" / f"{preset_name_or_path}.json"
    if bundled_path.is_file():
        with open(bundled_path, "r", encoding="utf-8") as f:
            return json.load(f)

    available = [p.stem for p in (get_data_dir() / "presets").glob("*.json")]
    raise FileNotFoundError(
        f"Preset '{preset_name_or_path}' not found. Available presets: {available}"
    )


def build_race_config(
    circuit_name_or_circuit: str | Circuit,
    preset_name: str = "2024_default",
    laps: int | None = None,
    seed: int = 42,
) -> RaceConfig:
    """Construct a full RaceConfig from a circuit and a preset."""
    preset_data = load_preset(preset_name)

    if isinstance(circuit_name_or_circuit, Circuit):
        circuit = circuit_name_or_circuit
    else:
        # Check if preset provides a circuit override
        if "circuit_override" in preset_data and preset_data["circuit_override"].get("id") == circuit_name_or_circuit:
            circuit = Circuit.model_validate(preset_data["circuit_override"])
        else:
            circuit = load_circuit(circuit_name_or_circuit)

    grid = [GridEntry.model_validate(entry) for entry in preset_data["grid"]]
    mandatory_two = preset_data.get("mandatory_two_compounds", True)

    return RaceConfig(
        circuit=circuit,
        grid=grid,
        laps=laps if laps is not None else circuit.total_laps,
        seed=seed,
        mandatory_two_compounds=mandatory_two,
    )
