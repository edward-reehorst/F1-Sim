"""Race results and telemetry models for f1-sim."""

from pydantic import BaseModel, Field


class PitStopRecord(BaseModel):
    """Details of a single pit stop event."""

    lap: int = Field(ge=1, description="Race lap number on which the pit stop occurred")
    driver_id: str = Field(description="ID of the driver pitting")
    compound_in: str = Field(description="Tire compound being removed")
    compound_out: str = Field(description="Tire compound being fitted")
    stationary_time: float = Field(description="Stationary tire change time in seconds")
    total_pit_loss: float = Field(description="Total pit delta time added to the lap time")
    under_safety_car: bool = Field(default=False, description="Whether stop took place under Safety Car")


class DriverLapRecord(BaseModel):
    """Snapshot of a driver's state and performance at the completion of a single lap."""

    lap: int = Field(ge=1, description="Completed lap number")
    driver_id: str = Field(description="Unique driver ID")
    position: int = Field(ge=1, description="Track position at the end of the lap")
    lap_time: float = Field(description="Lap time in seconds")
    cumulative_time: float = Field(description="Total elapsed race time in seconds")
    tire_compound: str = Field(description="Compound used during this lap")
    tire_age: int = Field(ge=0, description="Age of the current set of tires in completed laps")
    gap_to_leader: float = Field(ge=0.0, description="Time behind race leader in seconds")
    interval_ahead: float = Field(ge=0.0, description="Time behind the car directly ahead in seconds")
    fuel_remaining_kg: float = Field(ge=0.0, description="Mass of fuel remaining in car at end of lap")
    in_pit: bool = Field(default=False, description="Whether the driver pitted on this lap")
    in_dirty_air: bool = Field(default=False, description="Whether the driver was running in dirty air (<1.5s)")
    race_flag: str = Field(default="GREEN", description="Track status on this lap (GREEN, VSC, SAFETY_CAR)")


class DriverRaceSummary(BaseModel):
    """Final post-race classification summary for an individual driver."""

    driver_id: str = Field(description="Unique driver ID")
    driver_code: str = Field(description="Three-letter FIA driver abbreviation")
    team_id: str = Field(description="Team ID")
    starting_position: int = Field(ge=1, description="Grid position at race start")
    finish_position: int = Field(ge=1, description="Official finishing position")
    total_time: float = Field(description="Total race time in seconds (or time until DNF)")
    gap_to_winner: float = Field(ge=0.0, description="Total gap to the race winner in seconds")
    points: int = Field(default=0, ge=0, description="F1 championship points awarded")
    fastest_lap_time: float | None = Field(default=None, description="Fastest valid lap time set by driver")
    fastest_lap_number: int | None = Field(default=None, description="Lap number of fastest lap")
    pit_stops: list[PitStopRecord] = Field(default_factory=list, description="All pit stops taken by driver")
    compounds_used: list[str] = Field(default_factory=list, description="Unique tire compounds used in race")
    dnf: bool = Field(default=False, description="Whether driver failed to finish")
    dnf_lap: int | None = Field(default=None, description="Lap number on which DNF occurred")
    dnf_reason: str | None = Field(default=None, description="Reason for retirement")


class RaceResult(BaseModel):
    """Complete results, standings, and time-series telemetry of a Grand Prix."""

    circuit_id: str = Field(description="Circuit identifier")
    circuit_name: str = Field(description="Official circuit name")
    total_laps: int = Field(ge=1, description="Total laps simulated")
    winner_id: str = Field(description="Driver ID of race winner")
    winner_time: float = Field(description="Total winning time in seconds")
    fastest_lap_driver_id: str | None = Field(default=None, description="Driver with fastest lap of the race")
    fastest_lap_time: float | None = Field(default=None, description="Fastest lap time of the race in seconds")
    driver_summaries: list[DriverRaceSummary] = Field(
        description="List of driver summaries sorted in official finishing order",
    )
    lap_records: list[DriverLapRecord] = Field(
        default_factory=list,
        description="Comprehensive list of lap records for all drivers across all laps",
    )

    def get_driver_summary(self, driver_id: str) -> DriverRaceSummary | None:
        """Fetch post-race summary for a specific driver."""
        for summary in self.driver_summaries:
            if summary.driver_id == driver_id:
                return summary
        return None

    def get_driver_lap_records(self, driver_id: str) -> list[DriverLapRecord]:
        """Fetch all chronological lap records for a specific driver."""
        return [rec for rec in self.lap_records if rec.driver_id == driver_id]

    def get_driver_lap_times(self, driver_id: str) -> list[float]:
        """Extract chronological lap times for a specific driver."""
        return [rec.lap_time for rec in self.get_driver_lap_records(driver_id)]

    def get_driver_positions(self, driver_id: str) -> list[int]:
        """Extract chronological positions lap-by-lap for a specific driver."""
        return [rec.position for rec in self.get_driver_lap_records(driver_id)]

    def plot_lap_chart(self, save_path: str | None = None, show: bool = False):
        """Generate and optionally save/show a position progression lap chart."""
        from f1_sim.viz.plots import plot_lap_chart
        return plot_lap_chart(self, save_path=save_path, show=show)

    def plot_gap_to_leader(self, save_path: str | None = None, show: bool = False):
        """Generate and optionally save/show a gap-to-leader progression chart."""
        from f1_sim.viz.plots import plot_gap_to_leader
        return plot_gap_to_leader(self, save_path=save_path, show=show)

    def plot_stints(self, save_path: str | None = None, show: bool = False):
        """Generate and optionally save/show a pit strategy and stint breakdown chart."""
        from f1_sim.viz.plots import plot_stints
        return plot_stints(self, save_path=save_path, show=show)

    def plot_pace_decay(self, save_path: str | None = None, show: bool = False):
        """Generate and optionally save/show a tire wear and pace decay chart."""
        from f1_sim.viz.plots import plot_pace_decay
        return plot_pace_decay(self, save_path=save_path, show=show)
