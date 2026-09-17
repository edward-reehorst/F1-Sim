"""Simulation engine components for f1-sim."""

from f1_sim.engine.batch import BatchResult, BatchSimulator, SimulationSnapshot
from f1_sim.engine.car_state import CarState
from f1_sim.engine.physics import (
    LapTimeBreakdown,
    compute_clean_air_lap_time,
    compute_clean_air_lap_time_breakdown,
    compute_fuel_delta,
    compute_tire_delta,
)
from f1_sim.engine.race import RaceEngine
from f1_sim.engine.time_trial import TimeTrialResult, run_time_trial
from f1_sim.engine.traffic import (
    OvertakeEvent,
    compute_overtake_probability,
    evaluate_overtake,
)

__all__ = [
    "BatchResult",
    "BatchSimulator",
    "CarState",
    "LapTimeBreakdown",
    "OvertakeEvent",
    "RaceEngine",
    "SimulationSnapshot",
    "TimeTrialResult",
    "compute_clean_air_lap_time",
    "compute_clean_air_lap_time_breakdown",
    "compute_fuel_delta",
    "compute_overtake_probability",
    "compute_tire_delta",
    "evaluate_overtake",
    "run_time_trial",
]
