"""Telemetry dataset schema for lap-level observations from real-world sessions."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field, model_validator

SESSION_TYPES = frozenset({"practice1", "practice2", "practice3", "qualifying", "race", "live"})
GREEN_FLAG = "GREEN"


class LapObservation(BaseModel):
    """A single observed lap from any session.

    Canonical fields reference bundled domain IDs (driver_id, team_id slugs) and
    bundled compound names so a dataset can be validated and immediately consumed
    by the calibrator.
    """

    session: str | None = Field(default=None, description="Session type (practice1|practice2|practice3|qualifying|race|live)")
    driver_id: str = Field(description="Canonical driver ID (lowercase slug, e.g. 'verstappen')")
    team_id: str = Field(description="Canonical team ID (lowercase slug, e.g. 'red_bull')")
    lap: int = Field(ge=1, description="Lap number within the session")
    compound: str = Field(description="Tire compound name (case preserved, e.g. 'Soft', 'Medium')")
    lap_time: float = Field(gt=0.0, description="Lap time in seconds")
    fuel_remaining_kg: float | None = Field(default=None, description="Onboard fuel mass at lap start, if known")
    flag: str = Field(default="GREEN", description="Track flag status on this lap (GREEN, YELLOW, VSC, SAFETY_CAR)")
    position: int | None = Field(default=None, ge=1, description="Track position at the end of the lap, if known")
    stint: int | None = Field(default=None, ge=1, description="Stint number, if known")
    tire_age_at_lap_start: int = Field(default=0, ge=0, description="Completed laps on the current tire set at lap start")
    source: dict[str, Any] | None = Field(default=None, description="Raw source record for provenance")

    @model_validator(mode="after")
    def normalize_fields(self) -> "LapObservation":
        """Normalize session and flag strings to canonical forms."""
        self.flag = self.flag.strip().upper()
        if self.session is not None:
            session = self.session.strip().lower()
            if session not in SESSION_TYPES:
                raise ValueError(
                    f"Unknown session '{self.session}'. Expected one of {sorted(SESSION_TYPES)}"
                )
            self.session = session
        return self


class TelemetryDataset(BaseModel):
    """An ordered collection of LapObservations for a single circuit."""

    circuit_id: str = Field(description="Canonical circuit ID the observations were recorded at")
    observations: list[LapObservation] = Field(default_factory=list, description="Observations in recorded order")

    def __len__(self) -> int:
        return len(self.observations)

    def green_flag_observations(self) -> list[LapObservation]:
        """Return only normal green-flag laps."""
        return [o for o in self.observations if o.flag == GREEN_FLAG]

    def filter_for_pace(self, buffer_seconds: float = 1.5) -> list[LapObservation]:
        """Keep green-flag laps within ``buffer_seconds`` of each (driver, compound) best.

        Practice sessions are dominated by non-pace laps (traffic, aero runs,
        exploratory runs), so their mean is far slower than the genuine pace signal.
        This filter isolates the clean pace laps per driver and compound, which is the
        standard way to calibrate from free-practice data. Laps slower than the
        per-(driver, compound) best by more than ``buffer_seconds`` are dropped.
        """
        green = self.green_flag_observations()
        best: dict[tuple[str, str], float] = {}
        for obs in green:
            key = (obs.driver_id, obs.compound)
            best[key] = min(best.get(key, float("inf")), obs.lap_time)

        return [o for o in green if o.lap_time - best[(o.driver_id, o.compound)] <= buffer_seconds]

    def filter_for_calibration(
        self,
        outlier_mad_multiplier: float = 6.0,
        min_outlier_gap_seconds: float = 3.0,
    ) -> list[LapObservation]:
        """Return green-flag laps with robust per-(driver, compound) outliers removed.

        A lap is excluded when it deviates from the group median by more than
        ``max(min_outlier_gap_seconds, outlier_mad_multiplier * noise)``, where the
        noise is the median absolute deviation of lap times. The absolute floor keeps
        the screen focused on genuine spikes (pit in/out laps, sensor glitches) rather
        than the smooth degradation/cliff curvature that legitimately stretches a
        stint's spread. Groups with fewer than 10 observations are kept intact to avoid
        discarding legitimate small samples.
        """
        green = self.green_flag_observations()
        groups: dict[tuple[str, str], list[LapObservation]] = {}
        for obs in green:
            groups.setdefault((obs.driver_id, obs.compound), []).append(obs)

        kept: list[LapObservation] = []
        for group in groups.values():
            if len(group) < 10:
                kept.extend(group)
                continue
            times = np.array([o.lap_time for o in group], dtype=float)
            median = float(np.median(times))
            mad = float(np.median(np.abs(times - median)))
            noise = max(mad, min_outlier_gap_seconds * 0.05)
            threshold = max(min_outlier_gap_seconds, outlier_mad_multiplier * noise)
            for obs in group:
                if abs(obs.lap_time - median) <= threshold:
                    kept.append(obs)
        return kept