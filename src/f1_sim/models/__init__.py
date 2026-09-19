"""Live race snapshot models for resuming a Grand Prix from a real in-race state.

A :class:`LiveRaceSnapshot` captures the running race state at some lap *N* —
grid order, per-driver gaps to the leader, tires/compounds, fuel, and DNF
status. The simulation engine can seed itself from this snapshot instead of
re-simulating laps ``1..N``, then forecast only the remaining distance.

The snapshot data source is deliberately pluggable. The engine itself needs no
network access: a snapshot is normally produced by reading the F1 official Live
Timing app's ``timing.json`` (a free local file that exposes current positions,
gaps, compounds, and track status) or an equivalent FastF1-cached timing dump.
"""

from f1_sim.models.calibration import CalibrationConfig, GridPenalty, GridSlot
from f1_sim.models.circuit import Circuit
from f1_sim.models.config import GridEntry, RaceConfig
from f1_sim.models.driver import Driver
from f1_sim.models.results import (
    DriverLapRecord,
    DriverRaceSummary,
    PitStopRecord,
    RaceResult,
)
from f1_sim.models.snapshot import LiveDriverState, LiveRaceSnapshot
from f1_sim.models.team import Team
from f1_sim.models.telemetry import LapObservation, TelemetryDataset
from f1_sim.models.tire import TireCompound

__all__ = [
    "CalibrationConfig",
    "Circuit",
    "Driver",
    "DriverLapRecord",
    "DriverRaceSummary",
    "GridEntry",
    "GridPenalty",
    "GridSlot",
    "LapObservation",
    "LiveDriverState",
    "LiveRaceSnapshot",
    "PitStopRecord",
    "RaceConfig",
    "RaceResult",
    "Team",
    "TelemetryDataset",
    "TireCompound",
]
