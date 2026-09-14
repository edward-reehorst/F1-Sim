"""Driver model definition for f1-sim."""

from pydantic import BaseModel, Field, field_validator


class Driver(BaseModel):
    """Represents a Formula 1 driver with performance characteristics."""

    id: str = Field(description="Unique driver identifier, lowercase slug (e.g. 'verstappen')")
    name: str = Field(description="Full name of the driver (e.g. 'Max Verstappen')")
    code: str = Field(description="Three-letter FIA driver abbreviation (e.g. 'VER')")
    number: int = Field(ge=1, le=99, description="Permanent car number")
    pace_rating: float = Field(
        ge=0.0,
        le=100.0,
        description="Raw pace rating out of 100. Higher is faster.",
    )
    tire_management: float = Field(
        default=80.0,
        ge=0.0,
        le=100.0,
        description="Tire conservation rating out of 100. Higher reduces wear rate.",
    )

    @field_validator("code")
    @classmethod
    def validate_code(cls, v: str) -> str:
        code = v.strip().upper()
        if len(code) != 3:
            raise ValueError(f"Driver code must be exactly 3 characters, got '{v}'")
        return code

    @property
    def pace_delta_seconds(self) -> float:
        """Translates pace_rating (0-100) into a lap time delta in seconds.

        A rating of 100.0 corresponds to 0.0s delta (cleanest, fastest pace).
        Each point below 100 adds 0.020 seconds to the lap time.
        For example:
          100.0 -> 0.00s
           95.0 -> +0.10s
           90.0 -> +0.20s
           85.0 -> +0.30s
        """
        return (100.0 - self.pace_rating) * 0.020

    @property
    def tire_wear_multiplier(self) -> float:
        """Translates tire_management (0-100) into a degradation multiplier.

        Rating of 80.0 is neutral (1.0x wear).
        Higher reduces wear (e.g. 95 -> ~0.925x).
        Lower accelerates wear (e.g. 60 -> ~1.10x).
        """
        return max(0.70, 1.0 - (self.tire_management - 80.0) * 0.005)
