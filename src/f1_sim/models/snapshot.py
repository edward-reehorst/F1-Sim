"""Live timing snapshot models for resuming a race from its real current state.

A :class:`LiveRaceSnapshot` captures the state of an *ongoing* Grand Prix at a
given lap so the simulation engine can seed its cars and forecast only the
remaining laps. It is deliberately data-source agnostic: the loader accepts a
FastF1 ``LiveTimingData`` timing dump or a plain JSON snapshot with the same
shape, and it performs **zero** additional FastF1 API calls (reading a local
file is free).
"""

from pydantic import BaseModel, Field


class LiveDriverState(BaseModel):
    """Per-driver telemetry captured from an in-progress race."""

    driver_code: str = Field(min_length=3, max_length=3, description="Three-letter FIA driver code (e.g. VER)")
    position: int = Field(ge=1, description="1-indexed running position at snapshot time")
    gap_to_leader_seconds: float = Field(default=0.0, ge=0.0, description="Time behind race leader in seconds")
    interval_to_ahead_seconds: float = Field(default=0.0, ge=0.0, description="Interval to the car directly ahead")
    laps_completed: int = Field(default=0, ge=0, description="Laps completed by this driver in the live race")
    current_tire_compound: str = Field(default="Medium", description="Compound fitted at snapshot time")
    tire_age: int = Field(default=0, ge=0, description="Completed laps on the current tire set")
    compounds_used_so_far: list[str] | None = Field(
        default=None,
        description=(
            "Distinct dry-slick compounds this driver has already run before the "
            "snapshot (ordered by first use). When set, the engine primes each "
            "car's ``compounds_used`` with this history so the two-compound "
            "sporting rule can be judged from the *real* in-race stint history "
            "even on a live resume. None (the default) keeps the conservative "
            "assumption that only the snapshot's current compound is known. "
            "``tire_age`` is always the current stint's age, independent of this."
        ),
    )
    fuel_remaining_kg: float | None = Field(default=None, description="Onboard fuel mass in kg (None = accept simulation default)")
    running: bool = Field(default=True, description="Whether the car is still active (False = DNF)")
    dnf_reason: str | None = Field(default=None, description="Reason for retirement if not running")


class LiveRaceSnapshot(BaseModel):
    """Whole-field state of a Grand Prix at the moment of capture."""

    circuit_id: str = Field(description="Circuit ID the snapshot was captured at")
    current_lap: int = Field(ge=0, description="Lap number the race is currently at (0 = before race start)")
    total_laps: int | None = Field(default=None, ge=1, description="Total scheduled laps (defaults to circuit full distance)")
    race_flag: str = Field(default="GREEN", description="Track flag at snapshot time (GREEN, YELLOW, VSC, SAFETY_CAR)")
    drivers: list[LiveDriverState] = Field(description="Running-state captures for every car on track")
