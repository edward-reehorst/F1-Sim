"""Team and car performance model definition for f1-sim."""

from pydantic import BaseModel, Field


class Team(BaseModel):
    """Represents a Formula 1 constructor / team with car performance metrics."""

    id: str = Field(description="Unique team identifier, lowercase slug (e.g. 'red_bull')")
    name: str = Field(description="Full commercial team name (e.g. 'Oracle Red Bull Racing')")
    engine_power: float = Field(
        ge=0.0,
        le=100.0,
        description="Power unit output rating out of 100. Contributes to straight-line pace.",
    )
    aero_efficiency: float = Field(
        ge=0.0,
        le=100.0,
        description="Aerodynamic downforce vs drag efficiency rating out of 100.",
    )
    mechanical_grip: float = Field(
        ge=0.0,
        le=100.0,
        description="Suspension and low-speed mechanical grip rating out of 100.",
    )
    reliability: float = Field(
        default=0.98,
        ge=0.0,
        le=1.0,
        description="Mechanical reliability probability per race distance (e.g. 0.98 = 2% DNF risk).",
    )
    pit_crew_speed: float = Field(
        default=2.4,
        ge=1.5,
        le=6.0,
        description="Base average stationary pit stop time in seconds.",
    )
    pit_crew_consistency: float = Field(
        default=85.0,
        ge=0.0,
        le=100.0,
        description="Pit stop consistency rating out of 100. Higher reduces slow-stop probability.",
    )

    @property
    def composite_car_rating(self) -> float:
        """Weighted car performance index (0-100)."""
        return (
            0.35 * self.engine_power
            + 0.40 * self.aero_efficiency
            + 0.25 * self.mechanical_grip
        )

    @property
    def car_delta_seconds(self) -> float:
        """Translates composite car rating into a lap time delta in seconds.

        A rating of 100 corresponds to 0.0s delta (fastest car).
        Each rating point below 100 adds 0.025 seconds per lap.
        For example:
          100.0 -> 0.00s
           95.0 -> +0.125s
           90.0 -> +0.250s
           80.0 -> +0.500s
        """
        return (100.0 - self.composite_car_rating) * 0.025
