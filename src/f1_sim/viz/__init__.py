"""Visualization package for f1-sim."""

from f1_sim.viz.live import create_dashboard_layout, run_live_race
from f1_sim.viz.plots import (
    COMPOUND_COLORS,
    TEAM_COLORS,
    plot_gap_to_leader,
    plot_lap_chart,
    plot_pace_decay,
    plot_stints,
)

__all__ = [
    "COMPOUND_COLORS",
    "TEAM_COLORS",
    "create_dashboard_layout",
    "plot_gap_to_leader",
    "plot_lap_chart",
    "plot_pace_decay",
    "plot_stints",
    "run_live_race",
]