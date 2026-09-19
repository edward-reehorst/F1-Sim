"""Race configuration and grid entry models for f1-sim."""

from pydantic import BaseModel, Field, model_validator

from f1_sim.models.calibration import CalibrationConfig
from f1_sim.models.circuit import Circuit
from f1_sim.models.snapshot import LiveRaceSnapshot


class GridEntry(BaseModel):
    """Represents a driver and car placed on the starting grid."""

    driver_id: str = Field(description="Unique driver ID matching a Driver record")
    team_id: str = Field(description="Unique team ID matching a Team record")
    starting_position: int = Field(ge=1, le=30, description="1-indexed grid position (1 = Pole Position)")
    starting_tire: str = Field(default="Medium", description="Compound name fitted at the race start")


class RaceConfig(BaseModel):
    """Complete parameters defining a race simulation instance."""

    circuit: Circuit = Field(description="Circuit specification for this event")
    grid: list[GridEntry] = Field(description="List of cars / drivers on the starting grid")
    calibration: CalibrationConfig | None = Field(
        default=None,
        description="Tuning/calibration overlay layered onto bundled defaults without mutating them.",
    )
    laps: int | None = Field(
        default=None,
        ge=1,
        le=150,
        description="Number of race laps. If None, circuit.total_laps is used.",
    )
    seed: int = Field(default=42, description="RNG seed for deterministic incidents and pit stops")
    mandatory_two_compounds: bool = Field(
        default=True,
        description="Enforces the rule requiring finishers to run at least two different dry compounds.",
    )
    fuel_penalty_per_kg: float = Field(
        default=0.033,
        ge=0.0,
        le=0.10,
        description="Lap time penalty in seconds per kilogram of onboard fuel mass.",
    )
    initial_fuel_kg: float | None = Field(
        default=None,
        ge=0.0,
        description="Starting fuel load in kg. If None, automatically sized for race distance + safety margin.",
    )
    dirty_air_penalty_seconds: float = Field(
        default=0.40,
        ge=0.0,
        le=2.0,
        description="Time loss added to lap time when following within 1.5s of another car.",
    )
    dirty_air_wear_multiplier: float = Field(
        default=1.25,
        ge=1.0,
        le=2.5,
        description="Tire degradation rate multiplier when following in dirty air.",
    )
    overtake_threshold_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description="Optional override of the circuit's pace delta required to trigger overtake probability (None = use the circuit value).",
    )
    dirty_air_distance_seconds: float = Field(
        default=1.5,
        ge=0.1,
        le=5.0,
        description="Interval threshold to preceding car below which dirty air takes effect.",
    )
    striking_distance_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=3.0,
        description="Interval threshold within which an overtake attempt can be launched.",
    )
    min_following_interval_seconds: float = Field(
        default=0.20,
        ge=0.05,
        le=1.0,
        description="Minimum gap held when a trailing car cannot complete an overtake.",
    )

    @model_validator(mode="after")
    def resolve_defaults(self) -> "RaceConfig":
        """Resolve laps and fuel load defaults based on circuit if not explicitly provided."""
        if self.laps is None:
            self.laps = self.circuit.total_laps

        if self.initial_fuel_kg is None:
            # Race distance fuel + 5 kg safety buffer
            self.initial_fuel_kg = (self.laps * self.circuit.fuel_burn_per_lap) + 5.0

        return self
