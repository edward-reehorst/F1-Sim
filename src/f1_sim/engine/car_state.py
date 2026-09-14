"""Dynamic state tracker for a single car and driver during a simulation."""

from pydantic import BaseModel, Field

from f1_sim.engine.physics import (
    LapTimeBreakdown,
    compute_clean_air_lap_time_breakdown,
)
from f1_sim.models.circuit import Circuit
from f1_sim.models.driver import Driver
from f1_sim.models.results import DriverLapRecord, PitStopRecord
from f1_sim.models.team import Team
from f1_sim.models.tire import TireCompound


class CarState(BaseModel):
    """Tracks dynamic state of a car/driver as a race progresses."""

    driver: Driver
    team: Team
    current_tire: TireCompound
    tire_age: int = Field(default=0, ge=0, description="Completed laps on current tire set")
    fuel_remaining_kg: float = Field(ge=0.0, description="Onboard fuel mass in kilograms")
    laps_completed: int = Field(default=0, ge=0, description="Number of laps completed so far")
    cumulative_time: float = Field(default=0.0, ge=0.0, description="Total elapsed race time in seconds")
    pit_stops: list[PitStopRecord] = Field(default_factory=list, description="Log of all pit stops taken")
    lap_records: list[DriverLapRecord] = Field(default_factory=list, description="Historical lap records")
    compounds_used: list[str] = Field(default_factory=list, description="Unique compounds used in race")
    is_dnf: bool = Field(default=False)
    dnf_reason: str | None = None

    def model_post_init(self, __context: object) -> None:
        if not self.compounds_used:
            self.compounds_used.append(self.current_tire.compound_name)

    def fit_new_tires(
        self,
        new_tire: TireCompound,
        stationary_time: float,
        transit_loss: float,
        lap_number: int,
        under_safety_car: bool = False,
    ) -> PitStopRecord:
        """Perform a pit stop, fitting a fresh tire set and logging the record."""
        total_loss = stationary_time + transit_loss
        record = PitStopRecord(
            lap=lap_number,
            driver_id=self.driver.id,
            compound_in=self.current_tire.compound_name,
            compound_out=new_tire.compound_name,
            stationary_time=stationary_time,
            total_pit_loss=total_loss,
            under_safety_car=under_safety_car,
        )
        self.pit_stops.append(record)
        self.current_tire = new_tire
        self.tire_age = 0
        if new_tire.compound_name not in self.compounds_used:
            self.compounds_used.append(new_tire.compound_name)
        return record

    def simulate_clean_air_lap(
        self,
        circuit: Circuit,
        fuel_penalty_per_kg: float = 0.033,
        pit_time_loss: float = 0.0,
        in_pit: bool = False,
    ) -> tuple[DriverLapRecord, LapTimeBreakdown]:
        """Simulate a single lap in clean air, updating fuel, tire age, and cumulative time."""
        breakdown = compute_clean_air_lap_time_breakdown(
            circuit=circuit,
            team=self.team,
            driver=self.driver,
            tire=self.current_tire,
            tire_age=self.tire_age,
            fuel_mass_kg=self.fuel_remaining_kg,
            fuel_penalty_per_kg=fuel_penalty_per_kg,
        )

        lap_time = breakdown.total_time + pit_time_loss
        self.cumulative_time += lap_time
        current_lap = self.laps_completed + 1

        record = DriverLapRecord(
            lap=current_lap,
            driver_id=self.driver.id,
            position=1,  # Single-car / clean-air default position
            lap_time=lap_time,
            cumulative_time=self.cumulative_time,
            tire_compound=self.current_tire.compound_name,
            tire_age=self.tire_age,
            gap_to_leader=0.0,
            interval_ahead=0.0,
            fuel_remaining_kg=self.fuel_remaining_kg,
            in_pit=in_pit,
            in_dirty_air=False,
        )

        self.lap_records.append(record)
        self.laps_completed += 1
        self.tire_age += 1
        self.fuel_remaining_kg = max(0.0, self.fuel_remaining_kg - circuit.fuel_burn_per_lap)

        return record, breakdown
