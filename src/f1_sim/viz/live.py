"""Rich terminal live spectator dashboard with real-time leaderboard and commentary."""

import time
from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from f1_sim.engine.race import RaceEngine
from f1_sim.models.flags import RaceFlag
from f1_sim.models.results import DriverLapRecord, RaceResult

TIRE_BADGES: dict[str, str] = {
    "soft": "[bold white on #E10600] S [/bold white on #E10600]",
    "medium": "[bold black on #FFD100] M [/bold black on #FFD100]",
    "hard": "[bold black on #FFFFFF] H [/bold black on #FFFFFF]",
    "intermediate": "[bold white on #39B54A] I [/bold white on #39B54A]",
    "wet": "[bold white on #00A3E0] W [/bold white on #00A3E0]",
}

FLAG_STYLES: dict[str, str] = {
    "GREEN": "[bold green]● GREEN FLAG[/bold green]",
    "YELLOW": "[bold yellow]● YELLOW FLAG[/bold yellow]",
    "VSC": "[bold black on cyan] VSC DEPLOYED [/bold black on cyan]",
    "SAFETY_CAR": "[bold black on yellow] SAFETY CAR [/bold black on yellow]",
}


def create_dashboard_layout(
    circuit_name: str,
    lap: int,
    total_laps: int,
    flag: RaceFlag,
    records: list[DriverLapRecord],
    recent_events: list[str],
    drivers_dict: dict,
    teams_dict: dict,
    fastest_lap_record: DriverLapRecord | None = None,
) -> Panel:
    """Build a Rich panel containing header, live leaderboard, and commentary feed.

    Args:
        teams_dict: Mapping of ``driver_id`` to the driver's ``Team``.
    """

    teams_by_driver = teams_dict
    flag_text = FLAG_STYLES.get(flag.value, f"[bold]{flag.value}[/bold]")
    pct = int((lap / total_laps) * 100) if total_laps else 0

    header_text = Text.from_markup(
        f"[bold white]{circuit_name}[/bold white]  |  "
        f"Lap [bold cyan]{lap}[/bold cyan] / [bold]{total_laps}[/bold] ([yellow]{pct}%[/yellow])  |  "
        f"{flag_text}"
    )

    if fastest_lap_record:
        fl_driver = drivers_dict.get(fastest_lap_record.driver_id)
        fl_code = fl_driver.code if fl_driver else fastest_lap_record.driver_id.upper()
        header_text.append_text(
            Text.from_markup(f"  |  FL: [bold magenta]{fl_code} ({fastest_lap_record.lap_time:.3f}s)[/bold magenta]")
        )

    # Leaderboard Table
    table = Table(expand=True, box=None)
    table.add_column("Pos", justify="right", style="cyan", width=4)
    table.add_column("Driver", style="bold", width=8)
    table.add_column("Team", width=22)
    table.add_column("Gap", justify="right", width=10)
    table.add_column("Interval", justify="right", width=10)
    table.add_column("Tire", justify="center", width=6)
    table.add_column("Age", justify="right", width=5)
    table.add_column("Stops", justify="center", width=7)
    table.add_column("Last Lap", justify="right", width=10)

    for rec in records:
        driver = drivers_dict.get(rec.driver_id)
        driver_code = driver.code if driver else rec.driver_id[:3].upper()
        team = teams_by_driver.get(rec.driver_id)
        team_name = team.name if team else "F1 Team"

        gap_str = "LEADER" if rec.position == 1 else f"+{rec.gap_to_leader:.3f}s"
        int_str = "—" if rec.position == 1 else f"+{rec.interval_ahead:.3f}s"
        tire_badge = TIRE_BADGES.get(rec.tire_compound.lower(), f"[{rec.tire_compound[0]}]")

        stop_str = "[bold black on cyan] PIT [/bold black on cyan]" if rec.in_pit else "—"

        is_fl = fastest_lap_record and rec.driver_id == fastest_lap_record.driver_id and rec.lap_time == fastest_lap_record.lap_time
        lap_style = "bold magenta" if is_fl else ""
        lap_str = f"[{lap_style}]{rec.lap_time:.3f}s[/{lap_style}]" if lap_style else f"{rec.lap_time:.3f}s"

        table.add_row(
            f"P{rec.position}",
            driver_code,
            team_name[:20],
            gap_str,
            int_str,
            tire_badge,
            str(rec.tire_age),
            stop_str,
            lap_str,
        )

    # Recent Event Log / Commentary
    commentary_text = Text()
    if recent_events:
        for ev in recent_events[-4:]:
            commentary_text.append(f"• {ev}\n", style="italic yellow")
    else:
        commentary_text.append("• Race underway. All cars circulating cleanly.\n", style="dim")

    commentary_panel = Panel(
        commentary_text,
        title="[bold yellow]Race Radio & Live Events[/bold yellow]",
        border_style="dim yellow",
        height=6,
    )

    content = Group(
        Panel(header_text, border_style="cyan"),
        table,
        commentary_panel,
    )

    return Panel(content, title="[bold red]F1 SIMULATION DASHBOARD[/bold red]", border_style="red")


def run_live_race(
    engine: RaceEngine,
    speed_seconds_per_lap: float = 0.20,
) -> RaceResult:
    """Simulate a race while rendering a live terminal dashboard."""
    drivers_dict = engine.drivers
    # Map driver_id to team
    driver_to_team = {}
    for entry in engine.config.grid:
        driver_to_team[entry.driver_id] = engine.teams[entry.team_id]

    recent_events: list[str] = []
    fastest_record: DriverLapRecord | None = None

    with Live(auto_refresh=False, screen=True) as live:
        for lap_idx in range(engine.total_laps):
            lap_records = engine.step_lap()

            # Track fastest lap
            for r in lap_records:
                if r.race_flag == RaceFlag.GREEN.value:
                    if fastest_record is None or r.lap_time < fastest_record.lap_time:
                        fastest_record = r
                        driver_code = drivers_dict[r.driver_id].code
                        recent_events.append(f"Lap {r.lap}: FASTEST LAP set by {driver_code} ({r.lap_time:.3f}s)")

            # Check new overtakes from this lap
            for ov in engine.overtake_events:
                if ov.lap == engine.current_lap and ov.success:
                    recent_events.append(f"{ov.description}")

            # Check pit entries
            for r in lap_records:
                if r.in_pit:
                    code = drivers_dict[r.driver_id].code
                    recent_events.append(f"Lap {r.lap}: {code} entered PIT LANE for fresh tires")

            # Check incidents
            for inc in engine.incident_events:
                if inc.lap == engine.current_lap:
                    recent_events.append(f"{inc.description}")

            panel = create_dashboard_layout(
                circuit_name=engine.circuit.name,
                lap=engine.current_lap,
                total_laps=engine.total_laps,
                flag=engine.race_control.current_flag,
                records=lap_records,
                recent_events=recent_events,
                drivers_dict=drivers_dict,
                teams_dict=driver_to_team,
                fastest_lap_record=fastest_record,
            )

            live.update(panel, refresh=True)

            if speed_seconds_per_lap > 0:
                time.sleep(speed_seconds_per_lap)

    return engine.build_race_result()
