"""Post-race visualization tools using Matplotlib."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for CLI and headless runs
import matplotlib.pyplot as plt

from f1_sim.models.results import DriverLapRecord, RaceResult

# Standard F1 2024 Team Color Palette
TEAM_COLORS: dict[str, str] = {
    "red_bull": "#1E41FF",
    "mclaren": "#FF8000",
    "ferrari": "#E8002D",
    "mercedes": "#27F4D2",
    "aston_martin": "#229971",
    "rb": "#6692FF",
    "haas": "#B6BABD",
    "alpine": "#0093CC",
    "williams": "#64C4FF",
    "sauber": "#52E252",
}

COMPOUND_COLORS: dict[str, str] = {
    "Soft": "#E10600",
    "Medium": "#FFD100",
    "Hard": "#FFFFFF",
    "Intermediate": "#39B54A",
    "Wet": "#00A3E0",
}


def plot_lap_chart(
    result: RaceResult,
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Generate a classic position progression lap chart across all race laps.

    X-axis: Lap number (1 to total_laps)
    Y-axis: Track Position (P1 at top, P20 at bottom)
    Lines: Position progression per driver colored by team.
    """
    fig, ax = plt.subplots(figsize=(14, 8), facecolor="#15151e")
    ax.set_facecolor("#1e1e2c")

    # Map drivers to their team and summaries
    summary_map = {s.driver_id: s for s in result.driver_summaries}

    # Group lap records by driver
    driver_laps: dict[str, list[tuple[int, int]]] = {}
    for rec in result.lap_records:
        if rec.driver_id not in driver_laps:
            driver_laps[rec.driver_id] = []
        driver_laps[rec.driver_id].append((rec.lap, rec.position))

    # Plot each driver's position trace
    for driver_id, laps in driver_laps.items():
        summary = summary_map.get(driver_id)
        if not summary:
            continue

        team_color = TEAM_COLORS.get(summary.team_id, "#CCCCCC")
        x_vals = [pt[0] for pt in laps]
        y_vals = [pt[1] for pt in laps]

        ax.plot(
            x_vals,
            y_vals,
            label=f"{summary.driver_code} ({summary.team_id.replace('_', ' ').title()})",
            color=team_color,
            linewidth=2.2,
            alpha=0.85,
        )

        # Label start position on left
        if x_vals and y_vals:
            ax.text(
                0.6,
                summary.starting_position,
                summary.driver_code,
                color=team_color,
                fontsize=8,
                va="center",
                ha="right",
                fontweight="bold",
            )
            # Label finish position on right
            last_x = x_vals[-1]
            last_y = y_vals[-1]
            finish_label = f"{summary.driver_code} (P{summary.finish_position})"
            if summary.dnf:
                finish_label += " DNF"
            ax.text(
                last_x + 0.4,
                last_y,
                finish_label,
                color=team_color,
                fontsize=8,
                va="center",
                ha="left",
                fontweight="bold",
            )

    ax.set_title(
        f"{result.circuit_name} — Lap Chart (Position Progression)",
        color="white",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Race Lap", color="#AAAAAA", fontsize=12, labelpad=10)
    ax.set_ylabel("Track Position", color="#AAAAAA", fontsize=12, labelpad=10)

    # Invert Y-axis so P1 is at the top
    ax.set_ylim(len(result.driver_summaries) + 0.8, 0.2)
    ax.set_yticks(range(1, len(result.driver_summaries) + 1))
    ax.set_yticklabels([f"P{i}" for i in range(1, len(result.driver_summaries) + 1)], color="#CCCCCC")

    ax.set_xlim(0, result.total_laps + 5)
    ax.tick_params(colors="#CCCCCC")
    ax.grid(True, linestyle="--", alpha=0.2, color="#FFFFFF")

    plt.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")

    if show:
        plt.show()

    return fig


def plot_gap_to_leader(
    result: RaceResult,
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Generate a Gap to Leader progression chart over time.

    Shows time delta to the race leader for top contenders, visualizing
    pit stops, undercuts, and Safety Car field compressions.
    """
    fig, ax = plt.subplots(figsize=(14, 7), facecolor="#15151e")
    ax.set_facecolor("#1e1e2c")

    summary_map = {s.driver_id: s for s in result.driver_summaries}

    # Plot top 10 finishers
    top_finishers = [s.driver_id for s in result.driver_summaries[:10]]

    driver_gaps: dict[str, list[tuple[int, float]]] = {d: [] for d in top_finishers}
    for rec in result.lap_records:
        if rec.driver_id in driver_gaps:
            driver_gaps[rec.driver_id].append((rec.lap, rec.gap_to_leader))

    for driver_id, gaps in driver_gaps.items():
        summary = summary_map[driver_id]
        team_color = TEAM_COLORS.get(summary.team_id, "#CCCCCC")
        x_vals = [pt[0] for pt in gaps]
        y_vals = [pt[1] for pt in gaps]

        ax.plot(
            x_vals,
            y_vals,
            label=f"P{summary.finish_position} {summary.driver_code}",
            color=team_color,
            linewidth=2.0,
            alpha=0.9,
        )

    ax.set_title(
        f"{result.circuit_name} — Gap to Leader Progression (Top 10)",
        color="white",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Lap", color="#AAAAAA", fontsize=12)
    ax.set_ylabel("Gap to Leader (seconds)", color="#AAAAAA", fontsize=12)

    ax.tick_params(colors="#CCCCCC")
    ax.grid(True, linestyle="--", alpha=0.2, color="#FFFFFF")
    ax.legend(facecolor="#222233", edgecolor="#444455", labelcolor="white", loc="upper left")

    plt.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")

    if show:
        plt.show()

    return fig


def plot_pace_decay(
    result: RaceResult,
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Generate a lap-time pace decay chart showing tire aging per stint.

    Each driver's lap time evolution is drawn as per-stint line segments
    colored by compound, visualizing tire wear curves and cliff degradation.
    """
    fig, ax = plt.subplots(figsize=(14, 8), facecolor="#15151e")
    ax.set_facecolor("#1e1e2c")

    summary_map = {s.driver_id: s for s in result.driver_summaries}

    # Group lap records by driver
    driver_records: dict[str, list[DriverLapRecord]] = {}
    for rec in result.lap_records:
        driver_records.setdefault(rec.driver_id, []).append(rec)

    legend_used: set[str] = set()
    legend_handles: list[plt.Line2D] = []

    for driver_id, records in driver_records.items():
        summary = summary_map.get(driver_id)
        if not summary or not records:
            continue

        color = TEAM_COLORS.get(summary.team_id, "#CCCCCC")

        # Split into stints whenever the compound changes
        stint_start = 0
        for idx in range(1, len(records) + 1):
            new_compound = idx < len(records) and records[idx].tire_compound != records[idx - 1].tire_compound
            end_of_records = idx == len(records)
            if not new_compound and not end_of_records:
                continue

            stint = records[stint_start:idx]
            if len(stint) >= 2:
                compound_color = COMPOUND_COLORS.get(stint[0].tire_compound, color)
                (line,) = ax.plot(
                    [r.lap for r in stint],
                    [r.lap_time for r in stint],
                    color=compound_color,
                    linewidth=1.6,
                    alpha=0.75,
                )
                compound = stint[0].tire_compound
                if compound not in legend_used:
                    line.set_label(f"{summary.driver_code} ({compound})")
                    legend_handles.append(line)
                    legend_used.add(compound)
            stint_start = idx

    ax.set_title(
        f"{result.circuit_name} — Tire Wear & Pace Decay",
        color="white",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Lap", color="#AAAAAA", fontsize=12)
    ax.set_ylabel("Lap Time (seconds)", color="#AAAAAA", fontsize=12)

    ax.tick_params(colors="#CCCCCC")
    ax.grid(True, linestyle="--", alpha=0.2, color="#FFFFFF")
    if legend_handles:
        ax.legend(handles=legend_handles, facecolor="#222233", edgecolor="#444455", labelcolor="white", loc="upper left")

    plt.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")

    if show:
        plt.show()

    return fig


def plot_stints(
    result: RaceResult,
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Generate a Gantt-style pit strategy timeline for all drivers.

    Displays every stint with compound coloring (Soft=Red, Medium=Yellow, Hard=White).
    """
    fig, ax = plt.subplots(figsize=(14, 9), facecolor="#15151e")
    ax.set_facecolor("#1e1e2c")

    drivers = result.driver_summaries
    y_ticks = []
    y_labels = []

    for idx, summary in enumerate(drivers):
        y_pos = len(drivers) - idx
        y_ticks.append(y_pos)
        y_labels.append(f"P{summary.finish_position} {summary.driver_code}")

        records = [r for r in result.lap_records if r.driver_id == summary.driver_id]
        if not records:
            continue

        stint_start = 1
        current_compound = records[0].tire_compound

        for r in records:
            if r.in_pit or r.lap == records[-1].lap:
                stint_end = r.lap
                stint_len = stint_end - stint_start + 1
                color = COMPOUND_COLORS.get(current_compound, "#888888")

                ax.barh(
                    y_pos,
                    stint_len,
                    left=stint_start - 0.5,
                    height=0.6,
                    color=color,
                    edgecolor="#222222",
                    alpha=0.9,
                )
                # Label compound letter
                midpoint = stint_start + (stint_len / 2.0) - 0.5
                text_color = "black" if current_compound in ("Medium", "Hard") else "white"
                ax.text(
                    midpoint,
                    y_pos,
                    current_compound[0],
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                    fontweight="bold",
                )

                stint_start = r.lap + 1
                current_compound = r.tire_compound

    ax.set_title(
        f"{result.circuit_name} — Tire Strategy & Stint Breakdown",
        color="white",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Lap Number", color="#AAAAAA", fontsize=12)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, color="#CCCCCC", fontsize=9)
    ax.set_xlim(0, result.total_laps + 1)
    ax.tick_params(colors="#CCCCCC")
    ax.grid(True, axis="x", linestyle="--", alpha=0.2, color="#FFFFFF")

    # Custom compound legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=comp)
        for comp, color in COMPOUND_COLORS.items()
    ]
    ax.legend(
        handles=legend_handles,
        facecolor="#222233",
        edgecolor="#444455",
        labelcolor="white",
        loc="lower right",
    )

    plt.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")

    if show:
        plt.show()

    return fig
