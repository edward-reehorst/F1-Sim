"""Multi-car race simulation engine with strategy, pit stops, Safety Cars, and DNFs."""

import random

from f1_sim.engine.car_state import CarState
from f1_sim.engine.physics import (
    compute_clean_air_lap_time_breakdown,
)
from f1_sim.engine.race_control import IncidentEvent, RaceControlManager
from f1_sim.engine.traffic import OvertakeEvent, evaluate_overtake
from f1_sim.loaders import (
    load_all_compounds,
    load_all_drivers,
    load_all_teams,
)
from f1_sim.models.circuit import Circuit
from f1_sim.models.config import RaceConfig
from f1_sim.models.driver import Driver
from f1_sim.models.flags import RaceFlag
from f1_sim.models.results import (
    DriverLapRecord,
    DriverRaceSummary,
    RaceResult,
)
from f1_sim.models.team import Team
from f1_sim.models.tire import TireCompound
from f1_sim.strategy.base import BaseStrategy
from f1_sim.strategy.standard import StandardStrategy

F1_POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]


class RaceEngine:
    """Manages the step-by-step execution of a full Grand Prix."""

    def __init__(
        self,
        config: RaceConfig,
        drivers: dict[str, Driver] | None = None,
        teams: dict[str, Team] | None = None,
        compounds: dict[str, TireCompound] | None = None,
        strategy_map: dict[str, BaseStrategy] | None = None,
        enable_incidents: bool = True,
    ) -> None:
        self.config = config
        self.circuit: Circuit = config.circuit
        self.total_laps: int = config.laps or self.circuit.total_laps
        self.rng = random.Random(config.seed)
        self.enable_incidents = enable_incidents

        # Load domain entities
        self.drivers = drivers or load_all_drivers()
        self.teams = teams or load_all_teams()
        self.compounds = compounds or load_all_compounds()

        # Build initial car states sorted by starting grid position
        self.cars: list[CarState] = []
        sorted_grid = sorted(config.grid, key=lambda entry: entry.starting_position)

        for entry in sorted_grid:
            driver = self.drivers[entry.driver_id]
            team = self.teams[entry.team_id]
            tire = self.compounds[entry.starting_tire.lower()]

            # Initial grid stagger: each position starts ~0.25s behind preceding car
            grid_delay = (entry.starting_position - 1) * 0.25

            car_state = CarState(
                driver=driver,
                team=team,
                current_tire=tire,
                fuel_remaining_kg=config.initial_fuel_kg or 100.0,
                cumulative_time=grid_delay,
            )
            self.cars.append(car_state)

        # Strategy assignment: default to StandardStrategy for each car
        self.strategies: dict[str, BaseStrategy] = {}
        for car in self.cars:
            if strategy_map and car.driver.id in strategy_map:
                self.strategies[car.driver.id] = strategy_map[car.driver.id]
            else:
                self.strategies[car.driver.id] = StandardStrategy()

        self.race_control = RaceControlManager(
            circuit=self.circuit,
            total_laps=self.total_laps,
            rng=self.rng,
        )

        self.current_lap: int = 0
        self.lap_records: list[DriverLapRecord] = []
        self.overtake_events: list[OvertakeEvent] = []
        self.incident_events: list[IncidentEvent] = []

    def _execute_pit_stop(
        self,
        car: CarState,
        new_compound: TireCompound,
        is_sc: bool,
    ) -> float:
        """Perform stationary tire change and return total pit delta time."""
        # Compute stationary time based on team pit crew speed and consistency
        base_stationary = car.team.pit_crew_speed
        # Small random variance (-0.15s to +0.20s)
        variance = self.rng.uniform(-0.15, 0.20)
        stationary = max(1.8, base_stationary + variance)

        # Pit crew blunder check
        blunder_prob = max(0.005, (100.0 - car.team.pit_crew_consistency) * 0.001)
        if self.rng.random() < blunder_prob:
            blunder_delay = self.rng.uniform(2.5, 4.5)
            stationary += blunder_delay

        # Transit loss based on current track status
        transit_loss = self.race_control.get_pit_transit_loss()

        car.fit_new_tires(
            new_tire=new_compound,
            stationary_time=round(stationary, 3),
            transit_loss=round(transit_loss, 3),
            lap_number=self.current_lap,
            under_safety_car=is_sc,
        )

        return stationary + transit_loss

    def step_lap(self) -> list[DriverLapRecord]:
        """Advance the simulation by exactly one lap."""
        self.current_lap += 1

        # 1. Update track flag state from previous neutralizations
        flag = self.race_control.update_flag_state()

        # 2. Check for new stochastic incidents / mechanical failures
        active_cars = [c for c in self.cars if not c.is_dnf]
        new_incidents = self.race_control.evaluate_incidents_for_lap(
            lap=self.current_lap,
            active_cars=active_cars,
            enable_incidents=self.enable_incidents,
        )
        for inc in new_incidents:
            self.incident_events.append(inc)

        # Flag after evaluating potential new incidents
        flag = self.race_control.current_flag
        is_sc = flag == RaceFlag.SAFETY_CAR
        is_vsc = flag == RaceFlag.VSC
        is_neutralized = is_sc or is_vsc

        # Active runners remaining
        active_cars = [c for c in self.cars if not c.is_dnf]
        num_active = len(active_cars)

        if num_active == 0:
            return []

        # 3. Strategy evaluation: check which active cars want to pit this lap
        pitting_cars: dict[str, TireCompound] = {}
        for car in active_cars:
            strat = self.strategies[car.driver.id]
            wants_pit, target_comp = strat.should_pit(
                car=car,
                current_lap=self.current_lap,
                total_laps=self.total_laps,
                race_flag=flag,
                available_compounds=self.compounds,
                mandatory_two_compounds=self.config.mandatory_two_compounds,
            )
            if wants_pit and target_comp is not None:
                pitting_cars[car.driver.id] = target_comp

        # 4. Determine initial intervals and dirty air status
        in_dirty_air: dict[str, bool] = {}
        striking_distance: dict[str, bool] = {}

        for i in range(num_active):
            d_id = active_cars[i].driver.id
            if i == 0 or is_neutralized:
                in_dirty_air[d_id] = False
                striking_distance[d_id] = False
            else:
                interval_start = max(
                    0.0,
                    active_cars[i].cumulative_time - active_cars[i - 1].cumulative_time,
                )
                in_dirty_air[d_id] = interval_start <= self.config.dirty_air_distance_seconds
                striking_distance[d_id] = interval_start <= self.config.striking_distance_seconds

        # 5. Compute base projected lap times for each car
        raw_lap_times: dict[str, float] = {}

        for car in active_cars:
            d_id = car.driver.id
            is_dirty = in_dirty_air[d_id]

            # Adjust driver wear if following closely under green flag
            driver_wear = car.driver.tire_wear_multiplier
            if is_dirty and not is_neutralized:
                driver_wear *= self.config.dirty_air_wear_multiplier

            # Neutralized laps reduce tire wear and fuel burn
            wear_factor = 0.50 if is_sc else (0.60 if is_vsc else 1.0)
            circuit_wear = self.circuit.tire_wear_factor * wear_factor

            bd = compute_clean_air_lap_time_breakdown(
                circuit=self.circuit,
                team=car.team,
                driver=car.driver,
                tire=car.current_tire,
                tire_age=car.tire_age,
                fuel_mass_kg=car.fuel_remaining_kg,
                fuel_penalty_per_kg=self.config.fuel_penalty_per_kg,
                calibration=self.config.calibration,
            )

            if is_sc:
                # Under SC, pace is restricted by the Safety Car target delta (~38% slower)
                pace = self.circuit.base_lap_time * 1.38
            elif is_vsc:
                # Under VSC, delta time is ~33% slower across all sectors
                pace = self.circuit.base_lap_time * 1.33
            else:
                pace = bd.total_time
                if is_dirty:
                    pace += self.config.dirty_air_penalty_seconds

            # If pitting this lap, add pit delta
            if d_id in pitting_cars:
                target_comp = pitting_cars[d_id]
                pit_delta = self._execute_pit_stop(car, target_comp, is_sc)
                pace += pit_delta

            raw_lap_times[d_id] = pace

        # 6. Overtake resolution (only under green / local yellow flag conditions)
        actual_lap_times: dict[str, float] = dict(raw_lap_times)

        if not is_neutralized:
            for i in range(1, num_active):
                ahead_car = active_cars[i - 1]
                behind_car = active_cars[i]

                ahead_id = ahead_car.driver.id
                behind_id = behind_car.driver.id

                pace_ahead = actual_lap_times[ahead_id]
                pace_behind = actual_lap_times[behind_id]
                pace_delta = pace_ahead - pace_behind

                can_challenge = striking_distance[behind_id]
                overtook = False

                # Cars pitting cannot overtake or defend normally
                neither_in_pit = (ahead_id not in pitting_cars) and (behind_id not in pitting_cars)

                if can_challenge and pace_delta > 0 and neither_in_pit:
                    roll = self.rng.random()
                    threshold = (
                        self.config.overtake_threshold_seconds
                        if self.config.overtake_threshold_seconds is not None
                        else self.circuit.overtake_threshold_seconds
                    )
                    success, event = evaluate_overtake(
                        lap=self.current_lap,
                        attacker_id=behind_id,
                        defender_id=ahead_id,
                        position=i,
                        pace_delta=pace_delta,
                        threshold=threshold,
                        rng_value=roll,
                    )
                    self.overtake_events.append(event)

                    if success:
                        overtook = True
                        active_cars[i - 1], active_cars[i] = active_cars[i], active_cars[i - 1]

                if not overtook and neither_in_pit:
                    min_gap = self.config.min_following_interval_seconds
                    projected_time_ahead = ahead_car.cumulative_time + actual_lap_times[ahead_id]
                    projected_time_behind = behind_car.cumulative_time + actual_lap_times[behind_id]

                    if projected_time_behind < (projected_time_ahead + min_gap):
                        actual_lap_times[behind_id] = (projected_time_ahead + min_gap) - behind_car.cumulative_time

        # 7. Advance car metrics (fuel, tire age, cumulative time)
        fuel_burn_mult = 0.50 if is_sc else (0.60 if is_vsc else 1.0)
        burn_kg = self.circuit.fuel_burn_per_lap * fuel_burn_mult

        for car in active_cars:
            d_id = car.driver.id
            car.cumulative_time += actual_lap_times[d_id]
            car.laps_completed += 1
            car.tire_age += 1
            car.fuel_remaining_kg = max(0.0, car.fuel_remaining_kg - burn_kg)

        # 8. If under Safety Car, compress the field
        if is_sc:
            self.race_control.compress_field_under_safety_car(active_cars)

        # 9. Sort active cars by cumulative time to set running order
        active_cars.sort(key=lambda c: c.cumulative_time)
        leader_time = active_cars[0].cumulative_time

        current_lap_records: list[DriverLapRecord] = []

        for pos_idx, car in enumerate(active_cars, start=1):
            d_id = car.driver.id
            gap_leader = car.cumulative_time - leader_time
            interval_ahead = 0.0 if pos_idx == 1 else (car.cumulative_time - active_cars[pos_idx - 2].cumulative_time)

            record = DriverLapRecord(
                lap=self.current_lap,
                driver_id=d_id,
                position=pos_idx,
                lap_time=actual_lap_times[d_id],
                cumulative_time=car.cumulative_time,
                tire_compound=car.current_tire.compound_name,
                tire_age=car.tire_age,
                gap_to_leader=round(gap_leader, 3),
                interval_ahead=round(interval_ahead, 3),
                fuel_remaining_kg=round(car.fuel_remaining_kg, 2),
                in_pit=(d_id in pitting_cars),
                in_dirty_air=in_dirty_air.get(d_id, False),
                race_flag=flag.value,
            )
            car.lap_records.append(record)
            self.lap_records.append(record)
            current_lap_records.append(record)

        return current_lap_records

    def simulate(self) -> RaceResult:
        """Execute the entire race simulation from current lap to total_laps."""
        while self.current_lap < self.total_laps:
            self.step_lap()
        return self.build_race_result()

    def build_race_result(self) -> RaceResult:
        """Compile the official post-race classification, points, and results."""
        active_cars = [c for c in self.cars if not c.is_dnf]
        active_cars.sort(key=lambda c: c.cumulative_time)

        # Mandatory compound sporting rule check (apply 30s penalty if violated in dry race)
        if self.config.mandatory_two_compounds:
            for car in active_cars:
                if len(car.compounds_used) < 2:
                    # Applied post-race penalty for failure to use at least two dry slick compounds
                    car.cumulative_time += 30.0

            # Re-sort after penalties
            active_cars.sort(key=lambda c: c.cumulative_time)

        dnf_cars = [c for c in self.cars if c.is_dnf]
        # Sort DNF cars by laps completed (descending)
        dnf_cars.sort(key=lambda c: c.laps_completed, reverse=True)

        final_order = active_cars + dnf_cars
        leader = final_order[0]
        winner_time = leader.cumulative_time

        # Find fastest lap across entire race
        fastest_record: DriverLapRecord | None = None
        for rec in self.lap_records:
            if rec.race_flag == RaceFlag.GREEN.value:  # Only valid under green
                if fastest_record is None or rec.lap_time < fastest_record.lap_time:
                    fastest_record = rec

        fastest_driver_id = fastest_record.driver_id if fastest_record else None
        fastest_time = fastest_record.lap_time if fastest_record else None

        start_positions = {entry.driver_id: entry.starting_position for entry in self.config.grid}
        summaries: list[DriverRaceSummary] = []

        for finish_pos, car in enumerate(final_order, start=1):
            d_id = car.driver.id
            is_finisher = not car.is_dnf
            gap = (car.cumulative_time - winner_time) if is_finisher else 0.0

            pts = 0
            if is_finisher and finish_pos <= 10:
                pts = F1_POINTS[finish_pos - 1]
                if fastest_driver_id == d_id:
                    pts += 1

            driver_laps = car.lap_records
            best_lap = min((r.lap_time for r in driver_laps if r.race_flag == RaceFlag.GREEN.value), default=None)
            best_lap_num = None
            for r in driver_laps:
                if r.lap_time == best_lap:
                    best_lap_num = r.lap
                    break

            summary = DriverRaceSummary(
                driver_id=d_id,
                driver_code=car.driver.code,
                team_id=car.team.id,
                starting_position=start_positions.get(d_id, finish_pos),
                finish_position=finish_pos,
                total_time=car.cumulative_time,
                gap_to_winner=round(gap, 3),
                points=pts,
                fastest_lap_time=best_lap,
                fastest_lap_number=best_lap_num,
                pit_stops=car.pit_stops,
                compounds_used=car.compounds_used,
                dnf=car.is_dnf,
                dnf_lap=car.laps_completed if car.is_dnf else None,
                dnf_reason=car.dnf_reason,
            )
            summaries.append(summary)

        return RaceResult(
            circuit_id=self.circuit.id,
            circuit_name=self.circuit.name,
            total_laps=self.total_laps,
            winner_id=leader.driver.id,
            winner_time=winner_time,
            fastest_lap_driver_id=fastest_driver_id,
            fastest_lap_time=fastest_time,
            driver_summaries=summaries,
            lap_records=self.lap_records,
        )
