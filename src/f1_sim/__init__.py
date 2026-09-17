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
    CalibrationConfig,
    Circuit,
    Driver,
    DriverLapRecord,
    DriverRaceSummary,
    GridEntry,
    LapObservation,
    PitStopRecord,
    RaceConfig,
    RaceResult,
    Team,
    TelemetryDataset,
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
    "calibrate_from_dataset": ("f1_sim.tuning.calibration", "calibrate_from_dataset"),
    "generate_synthetic_dataset": ("f1_sim.tuning.synthetic", "generate_synthetic_dataset"),
    "load_dataset": ("f1_sim.tuning.dataset", "load_dataset"),
    "write_observations_csv": ("f1_sim.tuning.export", "write_observations_csv"),
    "write_observations_json": ("f1_sim.tuning.export", "write_observations_json"),
    "fetch_session": ("f1_sim.tuning.fastf1", "fetch_session"),
    "apply_grid_penalties": ("f1_sim.tuning.fastf1", "apply_grid_penalties"),
    "collect_overtake_samples": ("f1_sim.tuning.overtake_analysis", "collect_overtake_samples"),
    "detect_on_track_overtakes": ("f1_sim.tuning.overtake_analysis", "detect_on_track_overtakes"),
    "pace_delta_stats": ("f1_sim.tuning.overtake_analysis", "pace_delta_stats"),
    "match_overtake_threshold": ("f1_sim.tuning.overtake_calibration", "match_overtake_threshold"),
    "overtake_count_for_threshold": ("f1_sim.tuning.overtake_calibration", "overtake_count_for_threshold"),
    "unknown_driver_codes": ("f1_sim.tuning.fastf1", "unknown_driver_codes"),
    "unknown_team_names": ("f1_sim.tuning.fastf1", "unknown_team_names"),
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
    "CalibrationConfig",
    "CarState",
    "Circuit",
    "Driver",
    "DriverLapRecord",
    "DriverRaceSummary",
    "GridEntry",
    "GridPenalty",
    "IncidentEvent",
    "LapObservation",
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
    "TelemetryDataset",
    "TimeTrialResult",
    "TireCompound",
    "build_race_config",
    "apply_grid_penalties",
    "calibrate_from_dataset",
    "compute_clean_air_lap_time",
    "compute_clean_air_lap_time_breakdown",
    "compute_fuel_delta",
    "compute_overtake_probability",
    "compute_tire_delta",
    "create_dashboard_layout",
    "evaluate_overtake",
    "fetch_session",
    "generate_synthetic_dataset",
    "load_all_circuits",
    "load_all_compounds",
    "load_all_drivers",
    "load_all_teams",
    "load_circuit",
    "load_compound",
    "load_dataset",
    "load_driver",
    "load_preset",
    "load_team",
    "plot_gap_to_leader",
    "plot_lap_chart",
    "plot_pace_decay",
    "plot_stints",
    "collect_overtake_samples",
    "detect_on_track_overtakes",
    "pace_delta_stats",
    "match_overtake_threshold",
    "overtake_count_for_threshold",
    "run_live_race",
    "run_time_trial",
    "write_observations_csv",
    "write_observations_json",
    "unknown_driver_codes",
    "unknown_team_names",
]


def _build_app() -> typer.Typer:
    """Build the typer CLI application for f1-sim."""
    import typer
    import json
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
        calibration: str = typer.Option(None, "--calibration", help="Path to a calibration overlay JSON from `f1-sim tune`"),
    ) -> None:
        """Run a single-car clean-air time trial and display telemetry."""
        circuit = load_circuit(circuit_id)
        driver = load_driver(driver_id)
        team = load_team(team_id)
        compound = load_compound(compound_name)

        overlay = CalibrationConfig.read_json(calibration) if calibration else None
        if overlay is not None:
            console.print(f"Using calibration [cyan]{Path(calibration).resolve()}[/cyan]")

        result = run_time_trial(
            circuit=circuit,
            team=team,
            driver=driver,
            tire=compound,
            laps=laps,
            calibration=overlay,
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
    def tune(
        circuit_id: str = typer.Option("monza", "--circuit", "-c", help="Circuit ID the dataset was recorded at (e.g. monza)"),
        dataset: str = typer.Option(..., "--dataset", help="Path to telemetry CSV or JSON (bundled-schema or aliased columns)"),
        out: str = typer.Option(None, "--out", help="Output path for the calibration JSON (defaults to calibration/<circuit>.json)"),
        holdout: float = typer.Option(0.0, "--holdout", help="Fraction of observations held out for validation (0.0 to 0.5)"),
        ridge: float = typer.Option(0.5, "--ridge", help="Ridge regularization lambda (0.0 disables shrinkage)"),
        seed: int = typer.Option(42, "--seed", "-s", help="RNG seed for the holdout split"),
        anchor_driver: str = typer.Option(None, "--anchor-driver", help="Reference driver pinned at 0.0s pace offset (default: most-observed driver)"),
        max_cliff_adjust: int = typer.Option(8, "--max-cliff-adjust", help="Max |adjustment| explored per compound cliff threshold, in laps"),
        fit_fuel: bool = typer.Option(True, "--fit-fuel/--no-fit-fuel", help="Fit the fuel penalty when all laps carry fuel_remaining_kg"),
        penalties: list[str] = typer.Option([], "--penalty", help="Grid drop applied to the starting grid; repeatable as CODE:PLACES (e.g. VER:10, or CODE:0 for back of grid)"),
    ) -> None:
        """Fit a CalibrationConfig overlay from lap telemetry and save it as JSON."""
        from f1_sim.models.calibration import GridPenalty
        from f1_sim.tuning.calibration import calibrate_from_dataset
        from f1_sim.tuning.dataset import load_dataset

        parsed_penalties: list[GridPenalty] = []
        for spec in penalties:
            code, _, places = spec.partition(":")
            if not code or not places:
                raise typer.BadParameter(f"--penalty expects CODE:PLACES, got {spec!r}")
            try:
                places_n = int(places)
            except ValueError:
                raise typer.BadParameter(f"--penalty places must be an integer, got {places!r} in {spec!r}")
            if places_n < 0:
                raise typer.BadParameter(f"--penalty places must be >= 0, got {places_n} in {spec!r}")
            parsed_penalties.append(GridPenalty(driver_id=code.strip().upper(), places=places_n))
        if parsed_penalties:
            console.print(
                f"Applying {len(parsed_penalties)} grid penalty(ies): "
                + ", ".join(f"{p.driver_id}:{p.places}" for p in parsed_penalties)
                + "."
            )

        try:
            ds = load_dataset(dataset, circuit_id=circuit_id)
        except FileNotFoundError as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")
            raise typer.Exit(1)
        report = calibrate_from_dataset(
            ds,
            circuit_id,
            holdout=holdout,
            seed=seed,
            ridge=ridge,
            anchor_driver_id=anchor_driver,
            max_cliff_adjust=max_cliff_adjust,
            fit_fuel_penalty=fit_fuel,
            grid_penalties=parsed_penalties,
        )

        summary = Table(title=f"Calibration Report: {report.circuit_id}")
        summary.add_column("Metric")
        summary.add_column("Value", justify="right", style="bold")
        summary.add_row("Observations", str(report.n_observations))
        summary.add_row("Fit observations", str(report.n_fit_observations))
        summary.add_row("Holdout observations", str(report.n_holdout_observations))
        summary.add_row("In-sample RMSE", f"{report.in_sample_rmse:.4f}s")
        summary.add_row(
            "Holdout RMSE",
            f"{report.holdout_rmse:.4f}s" if report.holdout_rmse is not None else "N/A",
        )
        summary.add_row("Anchor driver", report.anchor_driver_id or "N/A")
        summary.add_row("Circuit base adjust", f"{report.circuit_base_adjust:+.4f}s")
        summary.add_row(
            "Fuel penalty",
            f"{report.fuel_penalty_per_kg:.4f}s/kg" if report.fuel_penalty_per_kg is not None else "N/A",
        )
        summary.add_row("Ridge lambda", f"{report.ridge_lambda:.2f}")
        console.print(summary)

        offsets = report.config.driver_pace_offsets
        if offsets:
            table = Table(title="Driver Pace Offsets (s)")
            table.add_column("Driver ID", style="cyan")
            table.add_column("Offset", justify="right")
            for driver_id, offset in sorted(offsets.items(), key=lambda kv: kv[1], reverse=True):
                style = "bold green" if offset < 0 else ("bold red" if offset > 0 else "dim")
                table.add_row(driver_id, f"[{style}]{offset:+.4f}s[/{style}]")
            console.print(table)

        wear = report.config.compound_wear_multipliers
        cliffs = report.config.compound_cliff_adjustments
        compound_table = Table(title="Compound Adjustments")
        compound_table.add_column("Compound")
        compound_table.add_column("Wear x", justify="right")
        compound_table.add_column("Cliff shift", justify="right")
        observed_compounds = sorted({o.compound.lower() for o in ds.observations})
        for compound in observed_compounds:
            compound_table.add_row(
                compound,
                f"{wear.get(compound, 1.0):.3f}" if compound in wear else "—",
                f"{cliffs.get(compound, 0):+d} laps" if compound in cliffs else "—",
            )
        console.print(compound_table)
        if report.config.starting_grid:
            grid_n = len(report.config.starting_grid)
            console.print(f"[dim]Starting grid from {grid_n} qualifying positions will replace the preset grid.[/dim]")
        console.print(f"[dim]{report.note}[/dim]")

        out_path = Path(out) if out is not None else Path("calibration") / f"{circuit_id}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report.config.write_json(out_path)
        console.print(f"[bold green]Calibration saved to[/bold green] [cyan]{out_path.resolve()}[/cyan]")

    @app.command()
    def ingest(
        year: int = typer.Option(..., "--year", help="Season to fetch (e.g. 2024)"),
        gp: str = typer.Option(..., "--gp", help="Grand Prix event name or location (e.g. monza, belgian)"),
        session: str = typer.Option("R", "--session", "-s", help="Session identifier (FP1, FP2, FP3, Q, R)"),
        circuit_id: str = typer.Option(None, "--circuit", "-c", help="Bundled circuit id (defaults to auto-detection)"),
        out: str = typer.Option(None, "--out", help="Output file path (defaults to observations/<gp>_<session>_<year>.<format>)"),
        as_format: str = typer.Option("csv", "--format", help="Output format: csv or json"),
        pace_filter: float = typer.Option(
            0.0,
            "--pace-filter",
            help="Isolate practice pace: keep laps within N seconds of each driver's per-compound best (0 disables)",
        ),
    ) -> None:
        """Fetch green-flag session laps via FastF1 and export them as observations."""
        from f1_sim.tuning.export import write_observations_csv, write_observations_json
        from f1_sim.tuning.fastf1 import fetch_session

        try:
            dataset = fetch_session(year=year, gp=gp, session_identifier=session, circuit_id=circuit_id)
            if pace_filter > 0:
                dataset.observations[:] = dataset.filter_for_pace(buffer_seconds=pace_filter)
        except ValueError as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")
            raise typer.Exit(1)

        slug = Path(gp).stem.strip().lower().replace(" ", "_")
        out_path = Path(out) if out is not None else (
            Path("observations") / f"{slug}_{session.lower()}_{year}.{as_format}"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if as_format == "json":
            write_observations_json(dataset, out_path)
        else:
            write_observations_csv(dataset, out_path)

        drivers = sorted({o.driver_id for o in dataset.observations})
        compounds = sorted({o.compound for o in dataset.observations})
        console.print(
            f"Ingested [cyan]{len(dataset)}[/cyan] green-flag laps at [bold]{dataset.circuit_id}[/bold] "
            f"({len(drivers)} drivers, {len(compounds)} compounds)."
        )
        console.print(f"[bold green]Observations saved to[/bold green] [cyan]{out_path.resolve()}[/cyan]")

    @app.command()
    def overtake_stats(
        start: int = typer.Option(2018, "--start", help="First season to analyze"),
        end: int = typer.Option(2026, "--end", help="Last season to analyze (inclusive)"),
        circuits: str = typer.Option(
            "monza,silverstone,spa,monaco", "--circuits", "-c", help="Comma-separated circuit ids (default: all bundled)"
        ),
        cache: str = typer.Option(
            None, "--cache", help="FastF1 disk-cache directory (defaults to ~/.cache/f1_sim_fastf1)",
        ),
        aggregator: str = typer.Option(
            "median", "--aggregator", help="Per-pass gap aggregator: median, mean, or quantile"
        ),
        quantile: float = typer.Option(
            0.10, "--quantile", help="Quantile to use when --aggregator quantile (0 < q < 1)"
        ),
        window_before: int = typer.Option(
            3, "--window-before", help="Laps before the pass included in the smoothing window"
        ),
        window_after: int = typer.Option(
            3, "--window-after", help="Laps after the pass included in the smoothing window"
        ),
        min_laps: int = typer.Option(
            3, "--min-laps", help="Minimum valid window laps required to keep a pass"
        ),
        include_pass_lap: bool = typer.Option(
            False, "--include-pass-lap", help="Include the pass lap itself in the smoothing window"
        ),
        out: str = typer.Option(
            None, "--out", help="Write per-circuit stats JSON to this path"
        ),
    ) -> None:
        """Reconstruct on-track overtakes from real races and report per-circuit pace-delta statistics."""
        from f1_sim.tuning.overtake_analysis import (
            CIRCUIT_GP_NAMES,
            collect_overtake_samples,
            pace_delta_stats,
        )

        if end < start:
            raise typer.BadParameter("--end must be >= --start")
        if aggregator not in ("median", "mean", "quantile"):
            raise typer.BadParameter(
                f"Unknown aggregator {aggregator!r}; expected median, mean, or quantile"
            )
        if aggregator == "quantile" and not 0.0 < quantile < 1.0:
            raise typer.BadParameter("--quantile must be in (0, 1)")
        if window_before < 0 or window_after < 0:
            raise typer.BadParameter("--window-before/--window-after must be >= 0")
        if min_laps < 1:
            raise typer.BadParameter("--min-laps must be >= 1")
        circuit_ids = [c.strip().lower() for c in circuits.split(",") if c.strip()]
        unknown = [c for c in circuit_ids if c not in CIRCUIT_GP_NAMES]
        if unknown:
            raise typer.BadParameter(
                f"Unknown circuit id(s): {', '.join(unknown)}. Bundled: {', '.join(sorted(CIRCUIT_GP_NAMES))}"
            )

        cache_dir = cache or str(Path.home() / ".cache" / "f1_sim_fastf1")
        console.print(
            f"Reconstructing overtakes from real races, circuits=[{', '.join(circuit_ids)}], "
            f"seasons {start}-{end} (cache: {cache_dir})..."
        )
        console.print(
            f"[dim]aggregator={aggregator}"
            + (f" (q={quantile})" if aggregator == "quantile" else "")
            + f", window={window_before}+{window_after} laps, min_laps={min_laps}, "
            + f"include_pass_lap={include_pass_lap}[/dim]"
        )
        samples = collect_overtake_samples(
            range(start, end + 1),
            circuit_ids,
            cache_dir=cache_dir,
            aggregator=aggregator,
            quantile=quantile,
            window_before=window_before,
            window_after=window_after,
            min_window_laps=min_laps,
            include_pass_lap=include_pass_lap,
        )
        stats = pace_delta_stats(samples)

        table = Table(title=f"Overtake Pace-Delta Statistics ({start}-{end})")
        table.add_column("Circuit")
        table.add_column("Races", justify="right")
        table.add_column("Overtakes", justify="right")
        table.add_column("Q1 (s)", justify="right")
        table.add_column("Median (s)", justify="right")
        table.add_column("Q3 (s)", justify="right")
        for circuit_id in circuit_ids:
            s = stats.get(circuit_id)
            if s is None or s.median is None:
                table.add_row(circuit_id, "-", "0", "-", "-", "-", style="dim")
                continue
            table.add_row(
                circuit_id,
                str(s.n_races),
                str(s.n_samples),
                f"{s.q1:.3f}" if s.q1 is not None else "-",
                f"{s.median:.3f}",
                f"{s.q3:.3f}" if s.q3 is not None else "-",
            )
        console.print(table)
        suggested = ", ".join(
            f"{cid}={stats[cid].median:.4f}"
            for cid in circuit_ids
            if stats.get(cid) and stats[cid].median is not None
        )
        if suggested:
            console.print(f"[dim]Suggested overtake_threshold_seconds:[/dim] {suggested}")

        if out:
            out_path = Path(out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "start_season": start,
                "end_season": end,
                "method": {
                    "aggregator": aggregator,
                    "quantile": quantile if aggregator == "quantile" else None,
                    "window_before": window_before,
                    "window_after": window_after,
                    "min_window_laps": min_laps,
                    "include_pass_lap": include_pass_lap,
                },
                "circuits": {
                    cid: (
                        {
                            "n_races": s.n_races,
                            "n_overtakes": s.n_samples,
                            "q1_seconds": s.q1,
                            "median_seconds": s.median,
                            "q3_seconds": s.q3,
                        }
                        if s.median is not None
                        else {"n_races": s.n_races, "n_overtakes": 0}
                    )
                    for cid, s in stats.items()
                },
            }
            out_path.write_text(json.dumps(payload, indent=2) + "\n")
            console.print(f"[bold green]Overtake stats saved to[/bold green] [cyan]{out_path.resolve()}[/cyan]")

    @app.command()
    def overtake_calibrate(
        stats_path: str = typer.Option(
            "src/f1_sim/data/overtake_stats.json",
            "--stats-path",
            help="Path to overtake_stats JSON carrying real race/pass counts",
        ),
        preset: str = typer.Option("2024_default", "--preset", "-p", help="Grid preset used for the search races"),
        seeds: int = typer.Option(5, "--seeds", help="RNG seeds averaged per threshold evaluation"),
        tolerance: float = typer.Option(1.0, "--tolerance", help="Pass/race tolerance for stopping the search"),
        max_threshold: float = typer.Option(5.0, "--max-threshold", help="Upper bound of the threshold search"),
        max_clean_delta: float = typer.Option(
            8.0, "--max-clean-delta", help="Passes with a pace gap above this are treated as pit/give-way laps and excluded"
        ),
        incidents: bool = typer.Option(True, "--incidents/--no-incidents", help="Enable Safety Cars/DNFs in search races"),
        laps: int | None = typer.Option(None, "--laps", "-l", help="Race laps for search races (defaults to full distance)"),
        skip_circuits: str = typer.Option(
            "monaco", "--skip-circuits", help="Comma-separated circuits excluded from the match (kept at their manually calibrated thresholds)"
        ),
        out: str = typer.Option(None, "--out", help="Write matched thresholds JSON to this path"),
        apply: bool = typer.Option(
            False, "--apply", help="Write matched thresholds into the bundled circuit JSONs (and scale spa_wet)"
        ),
    ) -> None:
        """Frequency-match overtake thresholds to real per-race overtake rates."""
        from f1_sim.tuning.overtake_calibration import (
            load_stat_targets,
            match_overtake_threshold,
        )

        if seeds < 1:
            raise typer.BadParameter("--seeds must be >= 1")
        if tolerance <= 0.0:
            raise typer.BadParameter("--tolerance must be > 0")

        stats_path_obj = Path(stats_path)
        if not stats_path_obj.exists():
            raise typer.BadParameter(f"Stats JSON not found: {stats_path_obj.resolve()}")
        stats_payload = json.loads(stats_path_obj.read_text())
        targets = load_stat_targets(stats_payload)
        if not targets:
            raise typer.BadParameter("No per-circuit race/pass counts found in stats JSON.")

        console.print(
            f"Frequency-matching overtake thresholds, target = real passes per race, "
            f"preset={preset}, seeds=1..{seeds}, tolerance={tolerance:g}..."
        )

        table = Table(title="Frequency-Matched Overtake Thresholds")
        table.add_column("Circuit")
        table.add_column("Races", justify="right")
        table.add_column("Real passes", justify="right")
        table.add_column("Target/race", justify="right")
        table.add_column("Threshold (s)", justify="right")
        table.add_column("Sim/race", justify="right")
        table.add_column("Converged")
        results: dict[str, dict] = {}
        skip_ids = {c.strip().lower() for c in skip_circuits.split(",") if c.strip()}
        for circuit_id in sorted(targets):
            if circuit_id in skip_ids:
                console.print(f"[dim]Skipping {circuit_id} (manual threshold; excluded from the match)[/dim]")
                continue
            stat_data = stats_payload["circuits"][circuit_id]
            n_races = int(stat_data["n_races"])
            n_overtakes = int(stat_data["n_overtakes"])
            result = match_overtake_threshold(
                circuit_id,
                targets[circuit_id],
                n_races=n_races,
                n_overtakes=n_overtakes,
                preset=preset,
                seeds=range(1, seeds + 1),
                incidents=incidents,
                laps=laps,
                hi=max_threshold,
                tolerance=tolerance,
                max_clean_delta=max_clean_delta,
            )
            results[circuit_id] = {
                "n_races": result.n_races,
                "n_overtakes": result.n_overtakes,
                "target_per_race": result.target_per_race,
                "threshold_seconds": round(result.threshold, 4),
                "achieved_per_race": result.achieved_per_race,
                "converged": result.converged,
            }
            table.add_row(
                circuit_id,
                str(result.n_races),
                str(result.n_overtakes),
                f"{result.target_per_race:.2f}",
                f"{result.threshold:.4f}",
                f"{result.achieved_per_race:.2f}",
                "[green]yes[/green]" if result.converged else "[red]no[/red]",
            )
        console.print(table)
        console.print(
            "[dim]Suggested overtake_threshold_seconds:[/dim] "
            + ", ".join(f"{cid}={r['threshold_seconds']}" for cid, r in sorted(results.items()))
        )

        if out:
            out_path = Path(out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "preset": preset,
                "seeds": seeds,
                "tolerance": tolerance,
                "incidents": incidents,
                "laps": laps,
                "circuits": results,
            }
            out_path.write_text(json.dumps(payload, indent=2) + "\n")
            console.print(f"[bold green]Thresholds saved to[/bold green] [cyan]{out_path.resolve()}[/cyan]")

        if apply:
            data_dir = Path(__file__).resolve().parent / "data"
            circuits_dir = data_dir / "circuits"
            presets_dir = data_dir / "presets"

            spa_old = None
            spa_wet_path = presets_dir / "spa_wet.json"
            spa_path = circuits_dir / "spa.json"
            if spa_path.exists() and spa_wet_path.exists():
                spa_old = float(json.loads(spa_path.read_text())["overtake_threshold_seconds"])
                spa_wet_old = float(json.loads(spa_wet_path.read_text())["circuit_override"]["overtake_threshold_seconds"])
                spa_wet_ratio = spa_wet_old / spa_old if spa_old else 1.2222
            else:
                spa_wet_ratio = None

            for circuit_id, record in sorted(results.items()):
                circuit_path = circuits_dir / f"{circuit_id}.json"
                if not circuit_path.exists():
                    console.print(f"[red]  Skipping {circuit_id}: {circuit_path} not found[/red]")
                    continue
                payload = json.loads(circuit_path.read_text())
                payload["overtake_threshold_seconds"] = record["threshold_seconds"]
                circuit_path.write_text(json.dumps(payload, indent=2) + "\n")
                console.print(f"[bold green]Updated[/bold green] [cyan]{circuit_path}[/cyan] -> {record['threshold_seconds']}")

            if spa_path.exists() and spa_wet_path.exists() and spa_wet_ratio is not None and "spa" in results:
                wet_payload = json.loads(spa_wet_path.read_text())
                wet_payload["circuit_override"]["overtake_threshold_seconds"] = round(
                    results["spa"]["threshold_seconds"] * spa_wet_ratio, 4
                )
                spa_wet_path.write_text(json.dumps(wet_payload, indent=2) + "\n")
                console.print(
                    f"[bold green]Updated[/bold green] [cyan]{spa_wet_path}[/cyan] -> "
                    f"{wet_payload['circuit_override']['overtake_threshold_seconds']}"
                )

    @app.command()
    def race(
        circuit_id: str = typer.Option("monza", "--circuit", "-c", help="Circuit ID (e.g. monza, silverstone)"),
        preset: str = typer.Option("2024_default", "--preset", "-p", help="Starting grid preset name"),
        laps: int | None = typer.Option(None, "--laps", "-l", help="Number of laps (defaults to circuit full distance)"),
        seed: int = typer.Option(42, "--seed", "-s", help="RNG seed for overtaking and incidents"),
        incidents: bool = typer.Option(True, "--incidents/--no-incidents", help="Enable or disable Safety Cars and DNFs"),
        calibration: str = typer.Option(None, "--calibration", help="Path to a calibration overlay JSON from `f1-sim tune`"),
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

        overlay = CalibrationConfig.read_json(calibration) if calibration else None
        if overlay is not None:
            console.print(f"Using calibration [cyan]{Path(calibration).resolve()}[/cyan]")

        config = build_race_config(circuit_name_or_circuit=circuit_id, preset_name=preset, laps=laps, seed=seed, calibration=overlay)
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
        calibration: str = typer.Option(None, "--calibration", help="Path to a calibration overlay JSON from `f1-sim tune`"),
        out_dir: str = typer.Option("output", "--out-dir", help="Directory where batch reports are saved"),
    ) -> None:
        """Run Monte Carlo batch simulations and summarize win rates and strategy."""
        from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

        from f1_sim.engine.batch import BatchSimulator

        overlay = CalibrationConfig.read_json(calibration) if calibration else None
        if overlay is not None:
            console.print(f"Using calibration [cyan]{Path(calibration).resolve()}[/cyan]")

        config = build_race_config(circuit_name_or_circuit=circuit_id, preset_name=preset, laps=laps, seed=seed, calibration=overlay)
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
