"""F1 Race Simulation Engine."""

from __future__ import annotations

from f1_sim.engine import (
    BatchResult,
    BatchSimulator,
    CarState,
    LapTimeBreakdown,
    OvertakeEvent,
    RaceEngine,
    SimulationSnapshot,
    TimeTrialResult,
    compute_clean_air_lap_time,
    compute_clean_air_lap_time_breakdown,
    compute_fuel_delta,
    compute_overtake_probability,
    compute_overtake_threshold,
    compute_tire_delta,
    evaluate_overtake,
    run_time_trial,
)
from f1_sim.engine.race_control import IncidentEvent, RaceControlManager
from f1_sim.loaders import (
    build_race_config,
    load_all_circuits,
    load_all_compounds,
    load_all_drivers,
    load_all_teams,
    load_circuit,
    load_compound,
    load_driver,
    load_preset,
    load_team,
)
from f1_sim.models import (
    Circuit,
    Driver,
    DriverLapRecord,
    DriverRaceSummary,
    GridEntry,
    PitStopRecord,
    RaceConfig,
    RaceResult,
    Team,
    TireCompound,
)
from f1_sim.models.flags import RaceFlag
from f1_sim.strategy import BaseStrategy, StandardStrategy

__version__ = "0.1.0"

_VISUALIZATION_EXPORTS: dict[str, tuple[str, str]] = {
    "create_dashboard_layout": ("f1_sim.viz.live", "create_dashboard_layout"),
    "run_live_race": ("f1_sim.viz.live", "run_live_race"),
    "plot_gap_to_leader": ("f1_sim.viz.plots", "plot_gap_to_leader"),
    "plot_lap_chart": ("f1_sim.viz.plots", "plot_lap_chart"),
    "plot_pace_decay": ("f1_sim.viz.plots", "plot_pace_decay"),
    "plot_stints": ("f1_sim.viz.plots", "plot_stints"),
}


def __getattr__(name: str):
    """Lazily expose visualization helpers to keep package imports fast."""
    if name in _VISUALIZATION_EXPORTS:
        module_name, attr = _VISUALIZATION_EXPORTS[name]
        import importlib

        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BaseStrategy",
    "BatchResult",
    "BatchSimulator",
    "CarState",
    "Circuit",
    "Driver",
    "DriverLapRecord",
    "DriverRaceSummary",
    "GridEntry",
    "IncidentEvent",
    "LapTimeBreakdown",
    "OvertakeEvent",
    "PitStopRecord",
    "RaceConfig",
    "RaceControlManager",
    "RaceEngine",
    "RaceFlag",
    "RaceResult",
    "SimulationSnapshot",
    "StandardStrategy",
    "Team",
    "TimeTrialResult",
    "TireCompound",
    "build_race_config",
    "compute_clean_air_lap_time",
    "compute_clean_air_lap_time_breakdown",
    "compute_fuel_delta",
    "compute_overtake_probability",
    "compute_overtake_threshold",
    "compute_tire_delta",
    "create_dashboard_layout",
    "evaluate_overtake",
    "load_all_circuits",
    "load_all_compounds",
    "load_all_drivers",
    "load_all_teams",
    "load_circuit",
    "load_compound",
    "load_driver",
    "load_preset",
    "load_team",
    "plot_gap_to_leader",
    "plot_lap_chart",
    "plot_pace_decay",
    "plot_stints",
    "run_live_race",
    "run_time_trial",
]


def _build_app() -> typer.Typer:
    """Build the typer CLI application for f1-sim."""
    import typer
    from pathlib import Path
    from rich.console import Console
    from rich.table import Table

    app = typer.Typer(help="F1 Race Simulation CLI", no_args_is_help=True)
    console = Console()

    @app.command()
    def info() -> None:
        """Display f1-sim package and dataset status."""
        circuits = load_all_circuits()
        teams = load_all_teams()
        drivers = load_all_drivers()
        compounds = load_all_compounds()

        console.print(f"[bold red]f1-sim[/bold red] [green]v{__version__}[/green] initialized.")
        console.print(
            f"Loaded [cyan]{len(circuits)}[/cyan] circuits, "
            f"[cyan]{len(teams)}[/cyan] teams, "
            f"[cyan]{len(drivers)}[/cyan] drivers, and "
            f"[cyan]{len(compounds)}[/cyan] tire compounds."
        )

    @app.command()
    def circuits() -> None:
        """List all available circuits and their core properties."""
        all_circuits = load_all_circuits()
        table = Table(title="Available F1 Circuits")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="bold")
        table.add_column("Country")
        table.add_column("Laps", justify="right")
        table.add_column("Base Lap Time", justify="right")
        table.add_column("SC Pit Loss", justify="right")

        for c in all_circuits.values():
            table.add_row(
                c.id,
                c.name,
                c.country,
                str(c.total_laps),
                f"{c.base_lap_time:.2f}s",
                f"{c.safety_car_pit_loss:.1f}s",
            )

        console.print(table)

    @app.command()
    def time_trial(
        circuit_id: str = typer.Option("monza", "--circuit", "-c", help="Circuit ID (e.g. monza, spa)"),
        driver_id: str = typer.Option("verstappen", "--driver", "-d", help="Driver ID (e.g. verstappen)"),
        team_id: str = typer.Option("red_bull", "--team", "-t", help="Team ID (e.g. red_bull)"),
        compound_name: str = typer.Option("Soft", "--tire", help="Tire compound (e.g. Soft, Medium, Hard)"),
        laps: int = typer.Option(15, "--laps", "-l", help="Number of laps to run"),
    ) -> None:
        """Run a single-car clean-air time trial and display telemetry."""
        circuit = load_circuit(circuit_id)
        driver = load_driver(driver_id)
        team = load_team(team_id)
        compound = load_compound(compound_name)

        result = run_time_trial(
            circuit=circuit,
            team=team,
            driver=driver,
            tire=compound,
            laps=laps,
        )

        table = Table(title=f"Time Trial: {driver.name} ({team.name}) @ {circuit.name}")
        table.add_column("Lap", justify="right", style="cyan")
        table.add_column("Lap Time", justify="right", style="bold")
        table.add_column("Fuel (kg)", justify="right")
        table.add_column("Fuel Delta", justify="right")
        table.add_column("Tire Age", justify="right")
        table.add_column("Tire Delta", justify="right")
        table.add_column("Total Time", justify="right")

        for rec, bd in zip(result.lap_records, result.breakdowns, strict=True):
            is_fastest = rec.lap == result.fastest_lap_number
            lap_style = "bold magenta" if is_fastest else "bold"
            table.add_row(
                str(rec.lap),
                f"[{lap_style}]{rec.lap_time:.3f}s[/{lap_style}]",
                f"{rec.fuel_remaining_kg:.1f}",
                f"+{bd.fuel_delta:.3f}s",
                str(rec.tire_age),
                f"+{bd.tire_delta:.3f}s",
                f"{rec.cumulative_time:.2f}s",
            )

        console.print(table)
        console.print(
            f"Fastest Lap: [bold magenta]{result.fastest_lap_time:.3f}s[/bold magenta] (Lap {result.fastest_lap_number}) | "
            f"Total Stint Time: [bold green]{result.total_time:.2f}s[/bold green]"
        )

    @app.command()
    def race(
        circuit_id: str = typer.Option("monza", "--circuit", "-c", help="Circuit ID (e.g. monza, silverstone)"),
        preset: str = typer.Option("2024_default", "--preset", "-p", help="Starting grid preset name"),
        laps: int | None = typer.Option(None, "--laps", "-l", help="Number of laps (defaults to circuit full distance)"),
        seed: int = typer.Option(42, "--seed", "-s", help="RNG seed for overtaking and incidents"),
        incidents: bool = typer.Option(True, "--incidents/--no-incidents", help="Enable or disable Safety Cars and DNFs"),
        live: bool = typer.Option(False, "--live/--no-live", help="Render a live Rich terminal dashboard while simulating"),
        speed: float = typer.Option(0.20, "--speed", help="Live dashboard playback speed in seconds per lap"),
        plot: bool = typer.Option(False, "--plot", help="Save Matplotlib visualizations after the race"),
        plot_dir: str = typer.Option("output", "--plot-dir", help="Directory where race plots are saved"),
    ) -> None:
        """Simulate a full Grand Prix with pit stops, strategy, and race control."""
        from f1_sim.viz.live import run_live_race
        from f1_sim.viz.plots import (
            plot_gap_to_leader,
            plot_lap_chart,
            plot_pace_decay,
            plot_stints,
        )

        config = build_race_config(circuit_name_or_circuit=circuit_id, preset_name=preset, laps=laps, seed=seed)
        engine = RaceEngine(config, enable_incidents=incidents)

        if live:
            console.print(f"[bold green]Live simulation of {config.circuit.name} ({engine.total_laps} laps)...[/bold green]")
            result = run_live_race(engine, speed_seconds_per_lap=speed)
        else:
            console.print(f"[bold green]Simulating {config.circuit.name} ({engine.total_laps} laps)...[/bold green]")
            result = engine.simulate()

        table = Table(title=f"Race Results: {result.circuit_name} ({result.total_laps} Laps)")
        table.add_column("Pos", justify="right", style="cyan")
        table.add_column("Driver", style="bold")
        table.add_column("Team")
        table.add_column("Time / Gap", justify="right")
        table.add_column("Stops", justify="center")
        table.add_column("Points", justify="right", style="green")
        table.add_column("Best Lap", justify="right")

        all_drivers = load_all_drivers()
        all_teams = load_all_teams()

        for summary in result.driver_summaries:
            driver_name = all_drivers[summary.driver_id].name
            team_name = all_teams[summary.team_id].name

            if summary.dnf:
                gap_str = f"[red]DNF (Lap {summary.dnf_lap})[/red]"
            elif summary.finish_position == 1:
                gap_str = f"{summary.total_time:.2f}s"
            else:
                gap_str = f"+{summary.gap_to_winner:.3f}s"

            stops_str = str(len(summary.pit_stops))
            if summary.pit_stops:
                sequence = " → ".join(c[0] for c in summary.compounds_used)
                stops_str = f"{len(summary.pit_stops)} ({sequence})"

            best_lap_str = (
                f"{summary.fastest_lap_time:.3f}s"
                if summary.fastest_lap_time is not None
                else "N/A"
            )

            is_fastest_lap = summary.driver_id == result.fastest_lap_driver_id
            if is_fastest_lap and not summary.dnf:
                best_lap_str = f"[bold magenta]{best_lap_str} (FL)[/bold magenta]"

            table.add_row(
                f"P{summary.finish_position}",
                f"{summary.driver_code} - {driver_name}",
                team_name,
                gap_str,
                stops_str,
                str(summary.points),
                best_lap_str,
            )

        console.print(table)

        # Print incident and overtake statistics
        sc_events = [i for i in engine.incident_events if i.flag_deployed == RaceFlag.SAFETY_CAR]
        vsc_events = [i for i in engine.incident_events if i.flag_deployed == RaceFlag.VSC]
        console.print(
            f"Race Control: [bold yellow]{len(sc_events)} Safety Car(s)[/bold yellow], "
            f"[bold cyan]{len(vsc_events)} VSC(s)[/bold cyan], "
            f"[bold red]{len(engine.incident_events)} Total Incident(s)[/bold red]"
        )

        successful_overtakes = [e for e in engine.overtake_events if e.success]
        console.print(
            f"Overtakes: [bold green]{len(successful_overtakes)} successful[/bold green] "
            f"out of [cyan]{len(engine.overtake_events)} attempts[/cyan]"
        )

        if plot:
            out_dir = Path(plot_dir)
            plot_lap_chart(result, save_path=out_dir / f"{result.circuit_id}_lap_chart.png")
            plot_gap_to_leader(result, save_path=out_dir / f"{result.circuit_id}_gap_to_leader.png")
            plot_stints(result, save_path=out_dir / f"{result.circuit_id}_stints.png")
            plot_pace_decay(result, save_path=out_dir / f"{result.circuit_id}_pace_decay.png")
            console.print(
                f"[bold green]Plots saved to[/bold green] [cyan]{out_dir.resolve()}[/cyan]"
            )

    @app.command()
    def batch(
        circuit_id: str = typer.Option("monza", "--circuit", "-c", help="Circuit ID (e.g. monza, silverstone)"),
        preset: str = typer.Option("2024_default", "--preset", "-p", help="Starting grid preset name"),
        laps: int | None = typer.Option(None, "--laps", "-l", help="Number of laps (defaults to circuit full distance)"),
        seed: int = typer.Option(42, "--seed", "-s", help="Master seed for reproducible batch runs"),
        sims: int = typer.Option(1000, "--sims", help="Number of Monte Carlo simulations to run"),
        workers: int | None = typer.Option(None, "--workers", "-w", help="Worker processes (defaults to CPU count)"),
        incidents: bool = typer.Option(True, "--incidents/--no-incidents", help="Enable or disable Safety Cars and DNFs"),
        out_dir: str = typer.Option("output", "--out-dir", help="Directory where batch reports are saved"),
    ) -> None:
        """Run Monte Carlo batch simulations and summarize win rates and strategy."""
        from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

        from f1_sim.engine.batch import BatchSimulator

        config = build_race_config(circuit_name_or_circuit=circuit_id, preset_name=preset, laps=laps, seed=seed)
        simulator = BatchSimulator(
            config=config,
            num_sims=sims,
            master_seed=seed,
            enable_incidents=incidents,
            n_workers=workers,
        )

        progress_columns = [
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}"),
            BarColumn(),
            TextColumn("[cyan]{task.completed}/{task.total}[/cyan]"),
            TimeElapsedColumn(),
        ]

        console.print(
            f"[bold green]Monte Carlo batch:[/bold green] [cyan]{sims}[/cyan] simulations "
            f"of {config.circuit.name} ([cyan]{config.laps}[/cyan] laps) across "
            f"[cyan]{simulator.n_workers}[/cyan] workers..."
        )

        with Progress(console=console, transient=False, *progress_columns) as progress:
            task = progress.add_task(f"Simulating {circuit_id}...", total=sims)
            result = simulator.run(on_progress=lambda done, total: progress.update(task, completed=done))

        summary_table = Table(title=f"Monte Carlo: {result.circuit_name} ({result.num_sims} Sims)")
        summary_table.add_column("Driver", style="bold")
        summary_table.add_column("Wins", justify="right", style="green")
        summary_table.add_column("Win %", justify="right")
        summary_table.add_column("Podium %", justify="right")
        summary_table.add_column("Avg Pts", justify="right")
        summary_table.add_column("Avg Pos", justify="right")
        summary_table.add_column("FL", justify="right", style="magenta")
        summary_table.add_column("DNF %", justify="right")

        for stat in result.driver_stats:
            summary_table.add_row(
                f"{stat.driver_code} - {stat.driver_id.replace('_', ' ').title()}",
                str(stat.wins),
                f"{stat.win_rate * 100:.1f}%",
                f"{stat.podium_rate * 100:.1f}%",
                f"{stat.avg_points:.2f}",
                f"{stat.avg_finish_position:.2f}",
                str(stat.fastest_laps),
                f"{stat.dnf_rate * 100:.1f}%",
            )
        console.print(summary_table)

        strategy_table = Table(title="Optimal Pit Strategies (by usage)")
        strategy_table.add_column("Strategy", style="bold cyan")
        strategy_table.add_column("Usage", justify="right")
        strategy_table.add_column("Share", justify="right")
        strategy_table.add_column("Wins", justify="right")
        strategy_table.add_column("Avg Pos", justify="right")

        for strat in result.pit_strategies[:10]:
            strategy_table.add_row(
                strat.strategy_label,
                str(strat.usage_count),
                f"{strat.share * 100:.1f}%",
                str(strat.wins),
                f"{strat.avg_position:.2f}",
            )
        console.print(strategy_table)

        console.print(
            f"Race Control: [bold red]{result.num_dnfs} DNF(s)[/bold red] "
            f"({result.dnf_rate * 100:.1f}% of car-races)  |  "
            f"Runtime: [bold green]{result.total_runtime_ms / 1000:.2f}s[/bold green] "
            f"({result.avg_runtime_ms:.2f} ms/race)"
        )

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        out.joinpath("summary.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
        out.joinpath("report.md").write_text(result.format_report(), encoding="utf-8")
        console.print(
            f"[bold green]Batch report saved to[/bold green] [cyan]{out.resolve()}[/cyan] "
            f"(summary.json, report.md)"
        )

    return app


def main() -> None:
    """CLI entry point for f1-sim."""
    _build_app()()
