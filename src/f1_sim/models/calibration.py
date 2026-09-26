"""Calibration overlay model: a non-mutating correction layer over bundled presets."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from f1_sim.models.tire import TireCompound


class GridSlot(BaseModel):
    """A single starting-grid slot recovered from a qualifying session."""

    position: int = Field(ge=1, description="1-indexed grid position (1 = pole)")
    driver_id: str = Field(description="Driver ID matching a Driver record")
    team_id: str = Field(description="Team ID matching a Team record")


class GridPenalty(BaseModel):
    """A grid drop penalty applied to a qualifying classification.

    Penalties are cumulative: multiple penalties for the same driver stack.
    A driver whose total drops exceed the grid size starts at the back.
    """

    driver_id: str = Field(description="Driver code (e.g. 'VER', 'LEC')")
    places: int = Field(ge=0, description="Grid drop in places (0 = back of grid)")


class CalibrationConfig(BaseModel):
    """Additive/multiplicative overlay layered into a RaceConfig at build time.

    The overlay never mutates bundled datasets; every field is an optional
    correction on top of the bundled default value. All dict lookups use lowercase
    slugs: driver/team IDs and lowercase compound names.

    Additive offsets relative to the bundled preset are allowed to be negative
    (representing real-world pace *faster* than the preset default). Physically
    constrained values (wear multipliers, fuel penalty, effective tire deltas) are
    clamped at application time.
    """

    driver_pace_offsets: dict[str, float] = Field(
        default_factory=dict,
        description="Per-driver additive lap-time offsets in seconds, relative to a calibration anchor driver at 0.0s.",
    )
    team_pace_offsets: dict[str, float] = Field(
        default_factory=dict,
        description="Per-team additive lap-time offsets in seconds. Not auto-fit from single-circuit data (see report).",
    )
    compound_base_deltas: dict[str, float] = Field(
        default_factory=dict,
        description="Per-compound additive correction to TireCompound.base_delta in seconds, keyed by lowercase compound name.",
    )
    compound_wear_multipliers: dict[str, float] = Field(
        default_factory=dict,
        description="Per-compound multiplicative correction to TireCompound.wear_rate, keyed by lowercase compound name.",
    )
    compound_cliff_adjustments: dict[str, int] = Field(
        default_factory=dict,
        description="Per-compound additive correction to TireCompound.cliff_lap in laps, keyed by lowercase compound name.",
    )
    circuit_base_adjust: float = Field(
        default=0.0,
        description="Additive correction to Circuit.base_lap_time in seconds (negative = track faster than preset).",
    )
    tire_wear_factor_adjust: float = Field(
        default=1.0,
        ge=0.0,
        description="Multiplicative correction to Circuit.tire_wear_factor.",
    )
    fuel_penalty_per_kg: float | None = Field(
        default=None,
        ge=0.0,
        le=0.10,
        description="Overrides the global fuel lap-time penalty (s/kg) when set.",
    )
    starting_grid: list[GridSlot] = Field(
        default_factory=list,
        description="Starting grid recovered from a qualifying session (position, driver, team). "
        "When set, replaces the preset's grid in the same order.",
    )

    @field_validator("compound_wear_multipliers")
    @classmethod
    def _validate_wear_multipliers(cls, v: dict[str, float]) -> dict[str, float]:
        for key, value in v.items():
            if value < 0.0:
                raise ValueError(f"compound_wear_multipliers[{key!r}] must be >= 0, got {value}")
        return v

    def is_identity(self) -> bool:
        """True when the overlay applies no corrections anywhere."""
        return self.strip_defaults().model_dump() == {}

    def strip_defaults(self) -> CalibrationConfig:
        """Return a copy containing only entries that differ from default values.

        ``driver_pace_offsets`` is never pruned: an explicit 0.0s entry is a
        meaningful calibration (the fitted anchor driver), not an absent default,
        and must survive a JSON round-trip for teammate fallbacks to work.
        """
        update: dict[str, object] = {
            "driver_pace_offsets": dict(self.driver_pace_offsets),
            "team_pace_offsets": {k: v for k, v in self.team_pace_offsets.items() if v != 0.0},
            "compound_base_deltas": {k: v for k, v in self.compound_base_deltas.items() if v != 0.0},
            "compound_wear_multipliers": {k: v for k, v in self.compound_wear_multipliers.items() if v != 1.0},
            "compound_cliff_adjustments": {k: v for k, v in self.compound_cliff_adjustments.items() if v != 0},
            "circuit_base_adjust": self.circuit_base_adjust if self.circuit_base_adjust != 0.0 else 0.0,
            "tire_wear_factor_adjust": self.tire_wear_factor_adjust if self.tire_wear_factor_adjust != 1.0 else 1.0,
            "fuel_penalty_per_kg": self.fuel_penalty_per_kg,
        }
        if self.starting_grid:
            update["starting_grid"] = list(self.starting_grid)
        return self.model_copy(update=update)

    def driver_offset(self, driver_id: str) -> float:
        """Additive pace offset in seconds for a driver (default 0.0)."""
        return self.driver_pace_offsets.get(driver_id, 0.0)

    def with_teammate_fallbacks(self, roster: Iterable[tuple[str, str]]) -> CalibrationConfig:
        """Return a copy whose missing driver offsets inherit their teammates' mean.

        ``roster`` is an iterable of ``(driver_id, team_id)`` pairs describing the
        grid. Any driver in the roster that has no per-driver offset receives the
        mean offset of the other drivers from the same team that do have offsets.
        Drivers in teams with no calibrated driver keep no offset (treated as the
        anchor 0.0s). The receiver is never mutated.
        """
        team_members: dict[str, list[str]] = {}
        for driver_id, team_id in roster:
            team_members.setdefault(team_id, []).append(driver_id)

        completed = dict(self.driver_pace_offsets)
        for members in team_members.values():
            known = [completed[m] for m in members if m in completed]
            if not known:
                continue
            inherited = sum(known) / len(known)
            for member in members:
                completed.setdefault(member, inherited)

        if completed == self.driver_pace_offsets:
            return self
        return self.model_copy(update={"driver_pace_offsets": completed})

    def team_offset(self, team_id: str) -> float:
        """Additive pace offset in seconds for a team (default 0.0)."""
        return self.team_pace_offsets.get(team_id, 0.0)

    def adjusted_base_time(self, default_base_time: float) -> float:
        """Circuit base lap time with the overlay correction applied."""
        return default_base_time + self.circuit_base_adjust

    def adjusted_circuit_wear_factor(self, default_wear_factor: float) -> float:
        """Circuit tire-wear factor with the overlay multiplier applied."""
        return default_wear_factor * self.tire_wear_factor_adjust

    def adjusted_fuel_penalty(self, default_penalty_per_kg: float) -> float:
        """Effective fuel penalty per kg, overriding the default when set."""
        return self.fuel_penalty_per_kg if self.fuel_penalty_per_kg is not None else default_penalty_per_kg

    def effective_tire(self, tire: TireCompound) -> TireCompound:
        """Return a tire copy with base delta / wear rate / cliff corrections applied.

        Physically required bounds are enforced: base delta and wear rate stay
        non-negative, and the effective cliff lap stays at least 1.
        """
        key = tire.compound_name.lower()
        base_delta = max(0.0, tire.base_delta + self.compound_base_deltas.get(key, 0.0))
        wear_rate = max(0.0, tire.wear_rate * self.compound_wear_multipliers.get(key, 1.0))
        cliff_lap = max(1, tire.cliff_lap + self.compound_cliff_adjustments.get(key, 0))
        return tire.model_copy(update={"base_delta": base_delta, "wear_rate": wear_rate, "cliff_lap": cliff_lap})

    def write_json(self, path: str | Path) -> None:
        """Persist the overlay (defaults stripped) as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.strip_defaults().model_dump_json(indent=2), encoding="utf-8")

    @staticmethod
    def read_json(path: str | Path) -> CalibrationConfig:
        """Load an overlay from a JSON file."""
        return CalibrationConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))