"""Data loader utilities for circuits, teams, drivers, compounds, and race presets."""

import json
from pathlib import Path

from f1_sim.models.calibration import CalibrationConfig, GridSlot
from f1_sim.models.circuit import Circuit
from f1_sim.models.config import GridEntry, RaceConfig
from f1_sim.models.driver import Driver
from f1_sim.models.snapshot import LiveRaceSnapshot
from f1_sim.models.team import Team
from f1_sim.models.tire import TireCompound


def _apply_calibration_grid(preset_grid: list[GridEntry], starting_grid: list[GridSlot]) -> list[GridEntry]:
    """Replace the preset grid with a qualifying grid.

    Qualifying slots keep their real positions/teams; bundled drivers absent from
    the qualifying data are appended behind them in preset order (same team and
    starting tire) so the grid stays full.
    """
    preset_by_driver = {e.driver_id: e for e in preset_grid}
    grid: list[GridEntry] = [
        GridEntry(
            driver_id=slot.driver_id,
            team_id=slot.team_id,
            starting_position=slot.position,
            starting_tire=preset_by_driver[slot.driver_id].starting_tire
            if slot.driver_id in preset_by_driver
            else "Medium",
        )
        for slot in sorted(starting_grid, key=lambda s: s.position)
    ]
    covered = {e.driver_id for e in grid}
    tail = len(grid)
    for entry in sorted(preset_grid, key=lambda e: e.starting_position):
        if entry.driver_id in covered:
            continue
        tail += 1
        grid.append(entry.model_copy(update={"starting_position": tail}))
    return grid


def get_data_dir() -> Path:
    """Return the absolute path to the bundled package data directory."""
    return Path(__file__).resolve().parent / "data"


def load_all_circuits() -> dict[str, Circuit]:
    """Load all bundled circuits keyed by circuit id."""
    circuits_dir = get_data_dir() / "circuits"
    circuits: dict[str, Circuit] = {}
    for filepath in sorted(circuits_dir.glob("*.json")):
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            circuit = Circuit.model_validate(data)
            circuits[circuit.id] = circuit
    return circuits


def load_circuit(name_or_path: str | Path) -> Circuit:
    """Load a circuit by identifier (e.g. 'monza') or from an explicit file path."""
    path = Path(name_or_path)
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            return Circuit.model_validate(json.load(f))

    # Look up in bundled circuits
    bundled_path = get_data_dir() / "circuits" / f"{name_or_path}.json"
    if bundled_path.is_file():
        with open(bundled_path, encoding="utf-8") as f:
            return Circuit.model_validate(json.load(f))

    available = [p.stem for p in (get_data_dir() / "circuits").glob("*.json")]
    raise FileNotFoundError(
        f"Circuit '{name_or_path}' not found. Available bundled circuits: {available}"
    )


def load_all_teams() -> dict[str, Team]:
    """Load all bundled teams keyed by team id."""
    teams_file = get_data_dir() / "teams.json"
    with open(teams_file, encoding="utf-8") as f:
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
    with open(drivers_file, encoding="utf-8") as f:
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
    with open(compounds_file, encoding="utf-8") as f:
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
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    bundled_path = get_data_dir() / "presets" / f"{preset_name_or_path}.json"
    if bundled_path.is_file():
        with open(bundled_path, encoding="utf-8") as f:
            return json.load(f)

    available = [p.stem for p in (get_data_dir() / "presets").glob("*.json")]
    raise FileNotFoundError(
        f"Preset '{preset_name_or_path}' not found. Available presets: {available}"
    )


def load_snapshot(snapshot_path_or_data: str | Path | dict) -> LiveRaceSnapshot:
    """Load a live in-race snapshot from a local file — zero network access.

    Accepts a path to a FastF1 ``LiveTimingData`` timing dump (a free local file
    that already exposes running positions, gaps, compounds, and track status) or
    a plain JSON / dict with the same shape. Reads are local-file-only: the
    simulated engine needs no additional FastF1 API calls (reading a file is
    free).
    """
    if isinstance(snapshot_path_or_data, dict):
        return LiveRaceSnapshot.model_validate(snapshot_path_or_data)

    path = Path(snapshot_path_or_data)
    if not path.is_file():
        raise FileNotFoundError(f"Live race snapshot not found: {path}")

    if path.suffix.lower() == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return LiveRaceSnapshot.model_validate(data)

    # Non-JSON timing dump (e.g. FastF1 LiveTimingData pickle) — read a local
    # file with zero extra API calls and coerce the shape we recognise.
    import pickle

    with open(path, "rb") as f:
        data = pickle.load(f)
    payload = getattr(data, "timing_data", data)
    return LiveRaceSnapshot.model_validate(payload)


def _derive_grid_from_snapshot(preset_grid: list[GridEntry], snapshot: LiveRaceSnapshot) -> list[GridEntry]:
    """Rebuild the starting grid from the real in-race running order in a snapshot.

    Each snapshot driver keeps the team pairing from the bundled preset grid; the
    captured running order becomes the grid order, DNF'd cars are flagged rather
    than removed, and every car inherits the tire compound it was actually on at
    capture time.
    """
    # Snapshot driver codes are 3-char FIA codes (e.g. 'VER'); bundled grid entries
    # are keyed by lowercase slug id (e.g. 'verstappen'). Translate via the real
    # Driver records so a captured code resolves to its bundled starting-slot entry.
    code_to_id = {d.code: driver_id for driver_id, d in load_all_drivers().items()}
    preset_by_driver = {e.driver_id: e for e in preset_grid}
    grid: list[GridEntry] = []
    for state in sorted(snapshot.drivers, key=lambda d: d.position):
        driver_id = code_to_id.get(state.driver_code, state.driver_code)
        base = preset_by_driver.get(driver_id)
        if base is None:
            continue
        grid.append(
            GridEntry(
                driver_id=base.driver_id,
                team_id=base.team_id,
                starting_position=state.position,
                starting_tire=state.current_tire_compound,
            )
        )
    # Bundled drivers not present in the snapshot trail the field (e.g. reserve/spare).
    known = {e.driver_id for e in grid}
    tail = max((e.starting_position for e in grid), default=0) + 1
    for entry in sorted(preset_grid, key=lambda e: e.starting_position):
        if entry.driver_id in known:
            continue
        grid.append(entry.model_copy(update={"starting_position": tail}))
        tail += 1
    return grid


def build_race_config(
    circuit_name_or_circuit: str | Circuit,
    preset_name: str = "2024_default",
    laps: int | None = None,
    seed: int = 42,
    calibration: CalibrationConfig | None = None,
    live_snapshot: LiveRaceSnapshot | None = None,
) -> RaceConfig:
    """Construct a full RaceConfig from a circuit and a preset.

    An optional ``calibration`` overlay is layered onto the bundled preset without
    mutating it (see f1_sim.tuning).
    """
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

    if calibration is not None:
        # A qualifying grid replaces the preset grid (real pairs + order); bundled
        # drivers without a qualifying slot are appended behind the field.
        if calibration.starting_grid:
            grid = _apply_calibration_grid(grid, calibration.starting_grid)
        # Drivers missing per-driver offsets inherit their teammate's mean offset.
        calibration = calibration.with_teammate_fallbacks((e.driver_id, e.team_id) for e in grid)

    if live_snapshot is not None:
        # Resuming a live Grand Prix: layer the real in-race state (running order,
        # compounds, gaps, DNFs) onto the bundled preset instead of the preset grid.
        if live_snapshot.circuit_id and circuit.id != live_snapshot.circuit_id:
            raise ValueError(
                f"Snapshot circuit {live_snapshot.circuit_id!r} does not match race circuit {circuit.id!r}."
            )
        grid = _derive_grid_from_snapshot(grid, live_snapshot)
        laps = (laps if laps is not None else circuit.total_laps) - live_snapshot.current_lap

    return RaceConfig(
        circuit=circuit,
        grid=grid,
        laps=laps if laps is not None else circuit.total_laps,
        seed=seed,
        mandatory_two_compounds=mandatory_two,
        calibration=calibration,
        live_snapshot=live_snapshot,
    )
