"""Tire compound model definition for f1-sim."""

from pydantic import BaseModel, Field


class TireCompound(BaseModel):
    """Represents a tire compound and its degradation characteristics."""

    compound_name: str = Field(description="Name of the compound (e.g. 'Soft', 'Medium', 'Hard', 'Intermediate', 'Wet')")
    color_code: str = Field(description="Display color / badge code (e.g. 'red', 'yellow', 'white', 'green', 'blue')")
    base_delta: float = Field(
        ge=0.0,
        description="Pace delta in seconds relative to the fastest dry slick (Soft = 0.0s, Medium ~0.6s, Hard ~1.2s).",
    )
    wear_rate: float = Field(
        ge=0.0,
        description="Linear pace degradation in seconds per lap in clean air on a neutral circuit.",
    )
    cliff_lap: int = Field(
        ge=1,
        description="Tire age in laps at which the compound hits its severe degradation cliff.",
    )
    cliff_penalty_factor: float = Field(
        default=0.15,
        ge=0.0,
        description="Quadratic time penalty coefficient applied to laps run beyond cliff_lap.",
    )

    def degradation_at_age(
        self,
        age: int,
        track_wear_factor: float = 1.0,
        driver_wear_factor: float = 1.0,
    ) -> float:
        """Computes total lap-time penalty for a tire of a given age.

        Penalty = base_delta + linear_wear + cliff_drop
        """
        if age <= 0:
            return self.base_delta

        effective_wear_rate = self.wear_rate * track_wear_factor * driver_wear_factor
        linear_wear = effective_wear_rate * age

        laps_past_cliff = max(0, age - self.cliff_lap)
        cliff_wear = self.cliff_penalty_factor * (laps_past_cliff**2)

        return self.base_delta + linear_wear + cliff_wear
