"""Monte Carlo batch simulation engine with multiprocessing and result aggregation."""

import os
import random
import time as time_module
from collections.abc import Callable
from statistics import mean

from pydantic import BaseModel, Field

from f1_sim.engine.race import RaceEngine
from f1_sim.loaders import load_all_compounds, load_all_drivers, load_all_teams
from f1_sim.models.config import RaceConfig


class DriverBatchRow(BaseModel):
    """Lightweight per-driver outcome from a single simulated race."""

    driver_id: str = Field(description="Unique driver ID")
    finish_position: int = Field(ge=1, description="Position in the official classification")
    points: int = Field(ge=0, description="Championship points earned (incl. fastest-lap bonus)")
    dnf: bool = Field(default=False, description="Whether the driver failed to finish")
    fastest_lap: bool = Field(default=False, description="Whether the driver set the race's fastest lap")
    compound_sequence: list[str] = Field(
        default_factory=list,
        description="Order of tire compounds used, starting compound first",
    )


class SimulationSnapshot(BaseModel):
    """Observable aggregate state of a single race used for batch statistics."""

    finishing_order: list[str] = Field(description="Driver IDs in classification order")
    rows: list[DriverBatchRow] = Field(description="Per-driver outcomes")
    dnfs: list[str] = Field(default_factory=list, description="Driver IDs that retired")

    @property
    def winner_id(self) -> str | None:
        """Return the race winner's driver ID, or None if no car finished."""
        return self.finishing_order[0] if self.finishing_order else None


def _simulate_chunk(
    config: RaceConfig,
    seeds: list[int],
    enable_incidents: bool,
) -> list[SimulationSnapshot]:
    """Simulate a chunk of races within a single worker process.

    Top-level function so it can be pickled for multiprocessing dispatch.
    """
    drivers = load_all_drivers()
    teams = load_all_teams()
    compounds = load_all_compounds()
    start_compound_by_driver = {entry.driver_id: entry.starting_tire for entry in config.grid}

    snapshots: list[SimulationSnapshot] = []
    for seed in seeds:
        race_config = config.model_copy(update={"seed": seed})
        engine = RaceEngine(
            race_config,
            drivers=drivers,
            teams=teams,
            compounds=compounds,
            enable_incidents=enable_incidents,
        )
        result = engine.simulate()

        rows: list[DriverBatchRow] = []
        for summary in result.driver_summaries:
            compound_sequence = [start_compound_by_driver.get(summary.driver_id, "Medium")]
            compound_sequence += [stop.compound_out for stop in summary.pit_stops]
            rows.append(
                DriverBatchRow(
                    driver_id=summary.driver_id,
                    finish_position=summary.finish_position,
                    points=summary.points,
                    dnf=summary.dnf,
                    fastest_lap=summary.driver_id == result.fastest_lap_driver_id,
                    compound_sequence=compound_sequence,
                )
            )

        snapshots.append(
            SimulationSnapshot(
                finishing_order=[summary.driver_id for summary in result.driver_summaries],
                rows=rows,
                dnfs=[summary.driver_id for summary in result.driver_summaries if summary.dnf],
            )
        )

    return snapshots


class DriverBatchStat(BaseModel):
    """Aggregated Monte Carlo statistics for a single driver."""

    driver_id: str = Field(description="Unique driver ID")
    driver_code: str = Field(description="Three-letter FIA driver abbreviation")
    wins: int = Field(ge=0, description="Number of race victories")
    win_rate: float = Field(ge=0.0, le=1.0, description="Fraction of races won")
    podiums: int = Field(ge=0, description="Number of podium finishes (P1-P3)")
    podium_rate: float = Field(ge=0.0, le=1.0, description="Fraction of races finishing on the podium")
    total_points: float = Field(ge=0.0, description="Total championship points accumulated")
    avg_points: float = Field(ge=0.0, description="Average championship points per race")
    avg_finish_position: float = Field(ge=1.0, description="Average classified finishing position")
    fastest_laps: int = Field(ge=0, description="Number of fastest race laps set")
    dnfs: int = Field(ge=0, description="Number of retirements")
    dnf_rate: float = Field(ge=0.0, le=1.0, description="Fraction of races retired")


class PitStrategyStat(BaseModel):
    """Aggregated statistics for a single tire compound sequence."""

    compound_sequence: list[str] = Field(description="Order of compounds used across the race")
    strategy_label: str = Field(description="Human-readable strategy label, e.g. 'M → H'")
    usage_count: int = Field(ge=0, description="Number of car-races using this sequence")
    share: float = Field(ge=0.0, le=1.0, description="Share of all car-races using this sequence")
    wins: int = Field(ge=0, description="Number of wins achieved with this sequence")
    avg_position: float = Field(ge=1.0, description="Average finishing position with this sequence")


class BatchResult(BaseModel):
    """Complete aggregate statistics for a Monte Carlo batch run."""

    circuit_id: str = Field(description="Circuit identifier")
    circuit_name: str = Field(description="Official circuit name")
    total_laps: int = Field(ge=1, description="Laps simulated per race")
    num_sims: int = Field(ge=1, description="Number of simulations run")
    master_seed: int = Field(description="Seed used to derive per-simulation seeds")
    enable_incidents: bool = Field(default=True, description="Whether incidents were simulated")
    total_runtime_ms: float = Field(ge=0.0, description="Wall-clock duration of the batch in milliseconds")
    avg_runtime_ms: float = Field(ge=0.0, description="Average duration per simulation in milliseconds")
    num_dnfs: int = Field(ge=0, description="Total retirements across all simulations")
    dnf_rate: float = Field(ge=0.0, le=1.0, description="Fraction of car-races that retired")
    driver_stats: list[DriverBatchStat] = Field(description="Per-driver aggregated statistics")
    pit_strategies: list[PitStrategyStat] = Field(description="Most frequently used pit strategies")

    @property
    def winner(self) -> DriverBatchStat | None:
        """Return the aggregate statistics of the most successful driver."""
        return self.driver_stats[0] if self.driver_stats else None

    def format_report(self) -> str:
        """Render a Markdown report summarizing the batch run."""
        lines: list[str] = [
            "# Monte Carlo Batch Report",
            "",
            f"- **Circuit**: {self.circuit_name} ({self.total_laps} laps)",
            f"- **Simulations**: {self.num_sims}",
            f"- **Master seed**: {self.master_seed}",
            f"- **Incidents**: {'enabled' if self.enable_incidents else 'disabled'}",
            f"- **Runtime**: {self.total_runtime_ms / 1000:.2f}s total, {self.avg_runtime_ms:.2f} ms/race",
            f"- **DNF rate**: {self.dnf_rate * 100:.1f}% ({self.num_dnfs} retirements)",
            "",
            "## Driver Statistics",
            "",
            "| Driver | Wins | Win % | Podium % | Avg Pts | Avg Pos | Fastest Laps | DNF % |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for stat in self.driver_stats:
            lines.append(
                f"| {stat.driver_code} | {stat.wins} | {stat.win_rate * 100:.1f}% "
                f"| {stat.podium_rate * 100:.1f}% | {stat.avg_points:.2f} "
                f"| {stat.avg_finish_position:.2f} | {stat.fastest_laps} | {stat.dnf_rate * 100:.1f}% |"
            )

        lines += ["", "## Optimal Pit Strategies", "", "| Strategy | Usage | Share | Wins | Avg Pos |", "|---|---:|---:|---:|---:|"]
        for strat in self.pit_strategies:
            lines.append(
                f"| {strat.strategy_label} | {strat.usage_count} | {strat.share * 100:.1f}% "
                f"| {strat.wins} | {strat.avg_position:.2f} |"
            )
        lines.append("")
        return "\n".join(lines)


def _build_batch_result(
    config: RaceConfig,
    snapshots: list[SimulationSnapshot],
    master_seed: int,
    enable_incidents: bool,
    runtime_seconds: float,
) -> BatchResult:
    """Aggregate individual race snapshots into overall Monte Carlo statistics."""
    driver_codes = {driver.id: driver.code for driver in load_all_drivers().values()}
    all_rows = [row for snapshot in snapshots for row in snapshot.rows]
    total_rows = len(all_rows)

    per_driver: dict[str, list[DriverBatchRow]] = {}
    for row in all_rows:
        per_driver.setdefault(row.driver_id, []).append(row)

    driver_stats: list[DriverBatchStat] = []
    for driver_id, rows in per_driver.items():
        wins = sum(1 for r in rows if r.finish_position == 1)
        podiums = sum(1 for r in rows if r.finish_position <= 3)
        dnfs = sum(1 for r in rows if r.dnf)
        num_rows = len(rows)
        driver_stats.append(
            DriverBatchStat(
                driver_id=driver_id,
                driver_code=driver_codes.get(driver_id, driver_id.upper()),
                wins=wins,
                win_rate=wins / num_rows,
                podiums=podiums,
                podium_rate=podiums / num_rows,
                total_points=sum(r.points for r in rows),
                avg_points=sum(r.points for r in rows) / num_rows,
                avg_finish_position=mean(r.finish_position for r in rows),
                fastest_laps=sum(1 for r in rows if r.fastest_lap),
                dnfs=dnfs,
                dnf_rate=dnfs / num_rows,
            )
        )

    driver_stats.sort(key=lambda s: (-s.wins, -s.total_points))

    sequence_rows: dict[tuple[str, ...], list[DriverBatchRow]] = {}
    for row in all_rows:
        sequence_rows.setdefault(tuple(row.compound_sequence), []).append(row)

    pit_strategies: list[PitStrategyStat] = []
    for sequence, rows in sequence_rows.items():
        pit_strategies.append(
            PitStrategyStat(
                compound_sequence=list(sequence),
                strategy_label=" → ".join(sequence),
                usage_count=len(rows),
                share=len(rows) / total_rows,
                wins=sum(1 for r in rows if r.finish_position == 1),
                avg_position=mean(r.finish_position for r in rows),
            )
        )
    pit_strategies.sort(key=lambda s: (-s.usage_count, -s.wins))

    num_dnfs = sum(1 for r in all_rows if r.dnf)

    return BatchResult(
        circuit_id=config.circuit.id,
        circuit_name=config.circuit.name,
        total_laps=config.laps or config.circuit.total_laps,
        num_sims=len(snapshots),
        master_seed=master_seed,
        enable_incidents=enable_incidents,
        total_runtime_ms=runtime_seconds * 1000.0,
        avg_runtime_ms=(runtime_seconds / max(1, len(snapshots))) * 1000.0,
        num_dnfs=num_dnfs,
        dnf_rate=num_dnfs / total_rows if total_rows else 0.0,
        driver_stats=driver_stats,
        pit_strategies=pit_strategies[:12],
    )


class BatchSimulator:
    """Run many seeded race simulations and aggregate Monte Carlo statistics."""

    def __init__(
        self,
        config: RaceConfig,
        num_sims: int = 1000,
        master_seed: int = 42,
        enable_incidents: bool = True,
        n_workers: int | None = None,
    ) -> None:
        self.config = config
        self.num_sims = num_sims
        self.master_seed = master_seed
        self.enable_incidents = enable_incidents
        self.n_workers = n_workers or os.cpu_count() or 1

    def _generate_seeds(self) -> list[int]:
        """Derive a deterministic, reproducible seed for every simulation."""
        rng = random.Random(self.master_seed)
        return [rng.randrange(0, 2**31) for _ in range(self.num_sims)]

    def run(
        self,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> BatchResult:
        """Execute the batch and return aggregated statistics.

        Args:
            on_progress: Optional callback invoked as ``(completed, total)`` as
                simulations finish. Useful for driving a Rich progress bar.
        """
        seeds = self._generate_seeds()
        snapshots: list[SimulationSnapshot] = []
        start = time_module.perf_counter()

        # Round-robin partition so each worker simulates a balanced subset
        num_workers = min(self.n_workers, self.num_sims)
        chunks: list[list[int]] = [[] for _ in range(num_workers)]
        for index, seed in enumerate(seeds):
            chunks[index % num_workers].append(seed)

        def consume(chunk_results: list[SimulationSnapshot]) -> None:
            snapshots.extend(chunk_results)
            if on_progress is not None:
                on_progress(len(snapshots), self.num_sims)

        if num_workers <= 1:
            consume(_simulate_chunk(self.config, seeds, self.enable_incidents))
        else:
            import multiprocessing as mp

            args = [(self.config, chunk, self.enable_incidents) for chunk in chunks if chunk]
            context = mp.get_context("spawn")
            with context.Pool(processes=num_workers) as pool:
                for chunk_results in pool.starmap(_simulate_chunk, args):
                    consume(chunk_results)

        elapsed = time_module.perf_counter() - start
        return _build_batch_result(
            config=self.config,
            snapshots=snapshots,
            master_seed=self.master_seed,
            enable_incidents=self.enable_incidents,
            runtime_seconds=elapsed,
        )