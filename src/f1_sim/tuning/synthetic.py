"""Synthetic ground-truth telemetry generation for calibrator tests and demos."""

from __future__ import annotations

import random

from f1_sim.engine.physics import compute_clean_air_lap_time
from f1_sim.loaders import load_all_drivers, load_compound, load_circuit, load_team, load_preset
from f1_sim.models.calibration import CalibrationConfig
from f1_sim.models.telemetry import LapObservation, TelemetryDataset


def _overlayable(overlay: CalibrationConfig | None) -> CalibrationConfig:
    return overlay if overlay is not None else CalibrationConfig()


def generate_synthetic_dataset(
    circuit_id: str = "monza",
    *,
    driver_ids: list[str] | None = None,
    stints: tuple[int, int, int] = (18, 18, 17),
    compounds: tuple[str, str, str] = ("Soft", "Medium", "Hard"),
    sessions: tuple[str, str, str] = ("practice3", "race", "race"),
    overlay: CalibrationConfig | None = None,
    noise_seconds: float = 0.0,
    include_fuel: bool = True,
    include_outliers: bool = False,
    seed: int = 12345,
) -> TelemetryDataset:
    """Generate laps from the physics engine under an optional ground-truth overlay.

    Each driver runs a fixed stint structure (lengths, compounds, sessions) with
    deterministic fuel burn. With ``noise_seconds == 0`` the output recovers the
    overlay parameters exactly, which makes it the reference oracle for
    synthetic-data recovery tests.
    """
    circuit = load_circuit(circuit_id)
    drivers = load_all_drivers()
    preset = load_preset("2024_default")
    team_by_driver = {entry["driver_id"]: entry["team_id"] for entry in preset["grid"]}

    if driver_ids is None:
        driver_ids = sorted(team_by_driver)

    if len(stints) != len(compounds) != len(sessions):
        raise ValueError("stints, compounds, and sessions must have equal length")

    rng = random.Random(seed)
    overlay = _overlayable(overlay)
    observations: list[LapObservation] = []

    total_laps = sum(stints)
    for driver_id in driver_ids:
        if driver_id not in team_by_driver:
            raise ValueError(f"Driver '{driver_id}' not found in the 2024_default grid")
        team = load_team(team_by_driver[driver_id])
        driver = drivers[driver_id]

        fuel = (total_laps * circuit.fuel_burn_per_lap) + 5.0
        lap_num = 0

        for compound_name, stint_laps, session in zip(compounds, stints, sessions, strict=True):
            tire = load_compound(compound_name)
            tire_age = 0
            for _ in range(stint_laps):
                lap_num += 1
                tire_age += 1
                fuel = max(0.0, fuel - circuit.fuel_burn_per_lap)

                lap_time = compute_clean_air_lap_time(
                    circuit=circuit,
                    team=team,
                    driver=driver,
                    tire=tire,
                    tire_age=tire_age,
                    fuel_mass_kg=fuel if include_fuel else 0.0,
                    calibration=overlay,
                )

                is_outlier = include_outliers and lap_num == 1
                if is_outlier:
                    lap_time += 18.0

                observations.append(
                    LapObservation(
                        session=session,
                        driver_id=driver_id,
                        team_id=team.id,
                        lap=lap_num,
                        compound=tire.compound_name,
                        lap_time=lap_time + rng.gauss(0.0, noise_seconds),
                        fuel_remaining_kg=fuel if include_fuel else None,
                        flag="GREEN",
                        position=1,
                        stint=1 + sum(1 for l in stints if lap_num > l),
                        tire_age_at_lap_start=tire_age,
                    )
                )

    return TelemetryDataset(circuit_id=circuit.id, observations=observations)