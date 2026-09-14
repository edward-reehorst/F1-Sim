"""Core deterministic physics calculations for lap-by-lap simulation."""

from pydantic import BaseModel, Field

from f1_sim.models.circuit import Circuit
from f1_sim.models.driver import Driver
from f1_sim.models.team import Team
from f1_sim.models.tire import TireCompound


class LapTimeBreakdown(BaseModel):
    """Component-by-component breakdown of a simulated lap time."""

    base_time: float = Field(description="Circuit baseline benchmark lap time (seconds)")
    car_delta: float = Field(description="Constructor performance delta (seconds)")
    driver_delta: float = Field(description="Driver skill pace delta (seconds)")
    fuel_delta: float = Field(description="Time penalty due to onboard fuel mass (seconds)")
    tire_delta: float = Field(description="Tire compound base delta + age degradation (seconds)")
    total_time: float = Field(description="Final computed lap time in seconds")


def compute_fuel_delta(fuel_mass_kg: float, penalty_per_kg: float = 0.033) -> float:
    """Compute lap time penalty in seconds from onboard fuel mass.

    Penalty scales linearly with fuel weight (~0.033s per kg).
    """
    return max(0.0, fuel_mass_kg * penalty_per_kg)


def compute_tire_delta(
    compound: TireCompound,
    tire_age: int,
    circuit_wear_factor: float = 1.0,
    driver_wear_multiplier: float = 1.0,
) -> float:
    """Compute tire grip delta and age degradation penalty in seconds."""
    return compound.degradation_at_age(
        age=tire_age,
        track_wear_factor=circuit_wear_factor,
        driver_wear_factor=driver_wear_multiplier,
    )


def compute_clean_air_lap_time_breakdown(
    circuit: Circuit,
    team: Team,
    driver: Driver,
    tire: TireCompound,
    tire_age: int,
    fuel_mass_kg: float,
    fuel_penalty_per_kg: float = 0.033,
) -> LapTimeBreakdown:
    """Compute the full deterministic clean-air lap time and its breakdown components."""
    base_time = circuit.base_lap_time
    car_delta = team.car_delta_seconds
    driver_delta = driver.pace_delta_seconds
    fuel_delta = compute_fuel_delta(fuel_mass_kg, fuel_penalty_per_kg)
    tire_delta = compute_tire_delta(
        compound=tire,
        tire_age=tire_age,
        circuit_wear_factor=circuit.tire_wear_factor,
        driver_wear_multiplier=driver.tire_wear_multiplier,
    )

    total_time = base_time + car_delta + driver_delta + fuel_delta + tire_delta

    return LapTimeBreakdown(
        base_time=base_time,
        car_delta=car_delta,
        driver_delta=driver_delta,
        fuel_delta=fuel_delta,
        tire_delta=tire_delta,
        total_time=total_time,
    )


def compute_clean_air_lap_time(
    circuit: Circuit,
    team: Team,
    driver: Driver,
    tire: TireCompound,
    tire_age: int,
    fuel_mass_kg: float,
    fuel_penalty_per_kg: float = 0.033,
) -> float:
    """Compute clean air lap time directly in seconds."""
    return compute_clean_air_lap_time_breakdown(
        circuit=circuit,
        team=team,
        driver=driver,
        tire=tire,
        tire_age=tire_age,
        fuel_mass_kg=fuel_mass_kg,
        fuel_penalty_per_kg=fuel_penalty_per_kg,
    ).total_time
