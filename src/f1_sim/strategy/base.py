"""Abstract base strategy interface for race strategy controllers."""

from abc import ABC, abstractmethod

from f1_sim.engine.car_state import CarState
from f1_sim.models.flags import RaceFlag
from f1_sim.models.tire import TireCompound


class BaseStrategy(ABC):
    """Protocol for pit stop strategy planners and reactive race-engineer agents."""

    @abstractmethod
    def should_pit(
        self,
        car: CarState,
        current_lap: int,
        total_laps: int,
        race_flag: RaceFlag,
        available_compounds: dict[str, TireCompound],
        mandatory_two_compounds: bool = True,
    ) -> tuple[bool, TireCompound | None]:
        """Determine whether the car should pit at the end of the current lap.

        Returns:
            A tuple of (pit_decision: bool, target_compound: TireCompound | None).
        """
        raise NotImplementedError
