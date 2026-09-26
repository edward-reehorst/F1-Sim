"""Standard rule-based F1 pit strategy controller with reactive incident opportunism."""

from f1_sim.engine.car_state import CarState
from f1_sim.models.flags import RaceFlag
from f1_sim.models.tire import TireCompound
from f1_sim.strategy.base import BaseStrategy


class StandardStrategy(BaseStrategy):
    """Realistic heuristic race engineer strategy planner.

    Handles target pit windows, tire cliff detection, Safety Car opportunism,
    and mandatory two-compound sporting compliance.
    """

    def __init__(self, target_stint_length: int | None = None) -> None:
        self.target_stint_length = target_stint_length

    def select_target_compound(
        self,
        car: CarState,
        remaining_laps: int,
        available_compounds: dict[str, TireCompound],
        mandatory_two_compounds: bool = True,
    ) -> TireCompound:
        """Select the optimal compound to fit during a pit stop."""
        current_name = car.current_tire.compound_name.lower()

        # If currently wet, stay on wet compounds
        if current_name in ("intermediate", "wet"):
            return available_compounds.get(current_name, car.current_tire)

        # Check which dry slick compounds have been used
        used_compounds = {c.lower() for c in car.compounds_used}
        unused_slick_candidates = [
            c for name, c in available_compounds.items()
            if name in ("soft", "medium", "hard") and name not in used_compounds
        ]

        # Prioritize fulfilling mandatory 2-compound rule if required
        if mandatory_two_compounds and len(used_compounds) < 2 and unused_slick_candidates:
            # Pick the best unused candidate based on remaining laps
            if remaining_laps > 22 and "hard" in [c.compound_name.lower() for c in unused_slick_candidates]:
                return available_compounds["hard"]
            elif "medium" in [c.compound_name.lower() for c in unused_slick_candidates]:
                return available_compounds["medium"]
            return unused_slick_candidates[0]

        # Standard compound selection based on remaining laps
        if remaining_laps <= 15 and "soft" in available_compounds:
            return available_compounds["soft"]
        elif remaining_laps <= 25 and "medium" in available_compounds:
            return available_compounds["medium"]
        elif "hard" in available_compounds:
            return available_compounds["hard"]

        # Default fallback to Medium or whatever is available
        return available_compounds.get("medium", car.current_tire)

    def should_pit(
        self,
        car: CarState,
        current_lap: int,
        total_laps: int,
        race_flag: RaceFlag,
        available_compounds: dict[str, TireCompound],
        mandatory_two_compounds: bool = True,
    ) -> tuple[bool, TireCompound | None]:
        """Evaluate whether to pit at the end of current_lap."""
        remaining_laps = total_laps - current_lap

        # Never pit on the final lap
        if remaining_laps <= 1:
            return False, None

        current_tire = car.current_tire
        tire_age = car.tire_age

        # 1. Immediate Pit Trigger: Severe tire cliff reached
        if tire_age >= current_tire.cliff_lap:
            target = self.select_target_compound(
                car, remaining_laps, available_compounds, mandatory_two_compounds
            )
            return True, target

        # 2. Opportunistic "Cheap Pit Stop" under Safety Car or VSC
        # If tires have run at least 8 laps and there are enough remaining laps to benefit
        if (
            race_flag in (RaceFlag.SAFETY_CAR, RaceFlag.VSC)
            and tire_age >= 8
            and remaining_laps >= 6
        ):
            target = self.select_target_compound(
                car, remaining_laps, available_compounds, mandatory_two_compounds
            )
            return True, target

        # 3. Mandatory compound compliance urgency trigger
        # If approaching the end of the race without having used a 2nd compound
        if (
            mandatory_two_compounds
            and len(car.compounds_used) < 2
            and remaining_laps <= 8
            and tire_age >= 10
        ):
            target = self.select_target_compound(
                car, remaining_laps, available_compounds, mandatory_two_compounds
            )
            return True, target

        # 4. Planned pit window based on compound lifecycle
        planned_window = self.target_stint_length or (current_tire.cliff_lap - 4)
        if tire_age >= planned_window and remaining_laps >= 8:
            target = self.select_target_compound(
                car, remaining_laps, available_compounds, mandatory_two_compounds
            )
            return True, target

        return False, None
