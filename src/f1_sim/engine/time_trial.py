"""Clean-air time trial simulation runner for a single car and driver."""

from pydantic import BaseModel, Field

from f1_sim.engine.car_state import CarState
from f1_sim.engine.physics import LapTimeBreakdown
from f1_sim.models.calibration import CalibrationConfig
from f1_sim.models.circuit import Circuit
from f1_sim.models.driver import Driver
from f1_sim.models.results import DriverLapRecord
from f1_sim.models.team import Team
from f1_sim.models.tire import TireCompound


class TimeTrialResult(BaseModel):
    """Results from a clean-air single-car time trial simulation."""

    driver_id: str
    circuit_id: str
    total_laps: int
    total_time: float
    fastest_lap_time: float
    fastest_lap_number: int
    lap_records: list[DriverLapRecord]
    breakdowns: list[LapTimeBreakdown] = Field(default_factory=list)


def run_time_trial(
    circuit: Circuit,
    team: Team,
    driver: Driver,
    tire: TireCompound,
    laps: int | None = None,
    initial_fuel_kg: float | None = None,
    fuel_penalty_per_kg: float = 0.033,
    calibration: CalibrationConfig | None = None,
) -> TimeTrialResult:
    """Run a single car in clean air over a specified number of laps.

    An optional ``calibration`` overlay layers tuned corrections onto the bundled
    physics defaults (see f1_sim.tuning).
    """
    total_laps = laps if laps is not None else circuit.total_laps

    if initial_fuel_kg is None:
        initial_fuel_kg = (total_laps * circuit.fuel_burn_per_lap) + 5.0

    car_state = CarState(
        driver=driver,
        team=team,
        current_tire=tire,
        fuel_remaining_kg=initial_fuel_kg,
    )

    records: list[DriverLapRecord] = []
    breakdowns: list[LapTimeBreakdown] = []

    fastest_lap_time = float("inf")
    fastest_lap_num = 1

    for _ in range(total_laps):
        rec, breakdown = car_state.simulate_clean_air_lap(
            circuit=circuit,
            fuel_penalty_per_kg=fuel_penalty_per_kg,
            calibration=calibration,
        )
        records.append(rec)
        breakdowns.append(breakdown)

        if rec.lap_time < fastest_lap_time:
            fastest_lap_time = rec.lap_time
            fastest_lap_num = rec.lap

    return TimeTrialResult(
        driver_id=driver.id,
        circuit_id=circuit.id,
        total_laps=total_laps,
        total_time=car_state.cumulative_time,
        fastest_lap_time=fastest_lap_time,
        fastest_lap_number=fastest_lap_num,
        lap_records=records,
        breakdowns=breakdowns,
    )
