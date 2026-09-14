"""Circuit model definition for f1-sim."""

from pydantic import BaseModel, Field


class Circuit(BaseModel):
    """Represents a Grand Prix circuit and its physical / strategic profile."""

    id: str = Field(description="Unique circuit identifier, lowercase slug (e.g. 'monza')")
    name: str = Field(description="Official name of the circuit (e.g. 'Autodromo Nazionale Monza')")
    country: str = Field(description="Host country of the Grand Prix")
    total_laps: int = Field(ge=1, le=120, description="Standard race distance in completed laps")
    base_lap_time: float = Field(
        ge=30.0,
        le=180.0,
        description="Clean-air baseline benchmark lap time in seconds under ideal conditions.",
    )
    overtaking_difficulty: float = Field(
        ge=0.0,
        le=1.0,
        description="Overtaking difficulty index between 0.0 (very easy, e.g. Monza) and 1.0 (nearly impossible, e.g. Monaco).",
    )
    tire_wear_factor: float = Field(
        default=1.0,
        ge=0.2,
        le=3.0,
        description="Tire degradation multiplier for the track surface. 1.0 is average, >1.0 high wear (Silverstone), <1.0 low wear (Monaco).",
    )
    pit_transit_loss: float = Field(
        ge=10.0,
        le=45.0,
        description="Total transit time lost in pit lane under green flag conditions in seconds (excluding stationary work).",
    )
    safety_car_pit_loss: float = Field(
        ge=5.0,
        le=35.0,
        description="Effective pit lane transit time lost when pitting under Safety Car conditions in seconds.",
    )
    fuel_burn_per_lap: float = Field(
        default=1.80,
        ge=0.5,
        le=3.5,
        description="Average fuel consumption per lap in kilograms.",
    )
