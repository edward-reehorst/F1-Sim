"""Race control engine: flags, incident generation, Safety Car field compression, and DNFs."""

import random

from pydantic import BaseModel

from f1_sim.engine.car_state import CarState
from f1_sim.models.circuit import Circuit
from f1_sim.models.flags import RaceFlag


class IncidentEvent(BaseModel):
    """Details of a race incident resulting in a DNF or flag change."""

    lap: int
    driver_id: str
    incident_type: str  # 'mechanical' or 'collision'
    description: str
    flag_deployed: RaceFlag
    neutralization_laps: int = 0


class RaceControlManager:
    """Manages race status flags, stochastic incidents, and Safety Car field compression."""

    def __init__(self, circuit: Circuit, total_laps: int, rng: random.Random) -> None:
        self.circuit = circuit
        self.total_laps = total_laps
        self.rng = rng

        self.current_flag: RaceFlag = RaceFlag.GREEN
        self.neutralization_laps_remaining: int = 0
        self.incident_log: list[IncidentEvent] = []

    def update_flag_state(self) -> RaceFlag:
        """Advance neutralization counter and return current flag."""
        if self.neutralization_laps_remaining > 0:
            self.neutralization_laps_remaining -= 1
            if self.neutralization_laps_remaining == 0:
                self.current_flag = RaceFlag.GREEN
        return self.current_flag

    def evaluate_incidents_for_lap(
        self,
        lap: int,
        active_cars: list[CarState],
        enable_incidents: bool = True,
    ) -> list[IncidentEvent]:
        """Check active cars for mechanical failures or collision DNFs."""
        new_incidents: list[IncidentEvent] = []
        if not enable_incidents or not active_cars:
            return new_incidents

        # Evaluate mechanical reliability per car
        for car in active_cars:
            if car.is_dnf:
                continue

            # Per-lap failure probability
            fail_prob = (1.0 - car.team.reliability) / max(1, self.total_laps)
            if self.rng.random() < fail_prob:
                car.is_dnf = True
                car.dnf_reason = "Mechanical failure"

                # Severity roll for track neutralization
                roll = self.rng.random()
                if roll < 0.35:
                    flag = RaceFlag.SAFETY_CAR
                    duration = self.rng.randint(2, 4)
                elif roll < 0.70:
                    flag = RaceFlag.VSC
                    duration = self.rng.randint(1, 2)
                else:
                    flag = RaceFlag.YELLOW
                    duration = 1

                self.current_flag = flag
                self.neutralization_laps_remaining = duration

                event = IncidentEvent(
                    lap=lap,
                    driver_id=car.driver.id,
                    incident_type="mechanical",
                    description=f"Lap {lap}: {car.driver.code} retired with mechanical failure ({flag.value} deployed)",
                    flag_deployed=flag,
                    neutralization_laps=duration,
                )
                self.incident_log.append(event)
                new_incidents.append(event)
                # Only trigger one major incident per lap
                break

        return new_incidents

    def get_pit_transit_loss(self) -> float:
        """Return pit transit loss in seconds according to current track flag."""
        if self.current_flag == RaceFlag.SAFETY_CAR:
            return self.circuit.safety_car_pit_loss
        elif self.current_flag == RaceFlag.VSC:
            # VSC saves roughly half the transit loss delta
            return (self.circuit.pit_transit_loss + self.circuit.safety_car_pit_loss) / 2.0
        return self.circuit.pit_transit_loss

    def compress_field_under_safety_car(
        self,
        cars: list[CarState],
        target_interval_seconds: float = 0.60,
    ) -> None:
        """Compress intervals between cars in the pack behind the Safety Car.

        The leader's cumulative time is preserved, and each trailing car is bunched up
        closer to the car ahead, reducing spread without allowing passes.
        """
        active_cars = [c for c in cars if not c.is_dnf]
        if len(active_cars) <= 1:
            return

        for i in range(1, len(active_cars)):
            ahead = active_cars[i - 1]
            behind = active_cars[i]

            current_gap = behind.cumulative_time - ahead.cumulative_time
            # Compress excessive gaps down towards target interval (e.g. 0.60s)
            if current_gap > target_interval_seconds:
                # Close 65% of the gap per SC lap
                compression = (current_gap - target_interval_seconds) * 0.65
                behind.cumulative_time -= compression
