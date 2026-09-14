"""Unit tests for Milestone 5: Rich Live Dashboard & Matplotlib Visualizations."""

import pytest
from rich.console import Console
from rich.panel import Panel

from f1_sim.engine import RaceEngine
from f1_sim.loaders import build_race_config
from f1_sim.models.flags import RaceFlag
from f1_sim.viz.live import create_dashboard_layout, run_live_race
from f1_sim.viz.plots import (
    plot_gap_to_leader,
    plot_lap_chart,
    plot_pace_decay,
    plot_stints,
)


@pytest.fixture
def race_result():
    config = build_race_config("monza", preset_name="2024_default", laps=5, seed=42)
    return RaceEngine(config).simulate()


@pytest.fixture
def engine():
    config = build_race_config("monza", preset_name="2024_default", laps=3, seed=42)
    return RaceEngine(config)


def _assert_png_saved(fig, save_path):
    import matplotlib

    assert isinstance(fig, matplotlib.figure.Figure)
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_lap_chart_saves_png(race_result, tmp_path):
    target = tmp_path / "lap_chart.png"
    fig = plot_lap_chart(race_result, save_path=target)
    _assert_png_saved(fig, target)


def test_plot_gap_to_leader_saves_png(race_result, tmp_path):
    target = tmp_path / "gap.png"
    fig = plot_gap_to_leader(race_result, save_path=target)
    _assert_png_saved(fig, target)


def test_plot_stints_saves_png(race_result, tmp_path):
    target = tmp_path / "stints.png"
    fig = plot_stints(race_result, save_path=target)
    _assert_png_saved(fig, target)


def test_plot_pace_decay_saves_png(race_result, tmp_path):
    target = tmp_path / "pace.png"
    fig = plot_pace_decay(race_result, save_path=target)
    _assert_png_saved(fig, target)


def test_race_result_plot_methods_save_png(race_result, tmp_path):
    targets = {
        "lap": tmp_path / "m_lap.png",
        "gap": tmp_path / "m_gap.png",
        "stints": tmp_path / "m_stints.png",
        "pace": tmp_path / "m_pace.png",
    }
    race_result.plot_lap_chart(save_path=targets["lap"])
    race_result.plot_gap_to_leader(save_path=targets["gap"])
    race_result.plot_stints(save_path=targets["stints"])
    race_result.plot_pace_decay(save_path=targets["pace"])

    for target in targets.values():
        assert target.exists()
        assert target.stat().st_size > 0


def test_plot_pace_decay_handles_single_lap(engine, tmp_path):
    engine.step_lap()
    result = engine.build_race_result()
    target = tmp_path / "pace_single.png"
    plot_pace_decay(result, save_path=target)
    assert target.exists()


def test_create_dashboard_layout_renders_header_and_rows(engine):
    records = engine.step_lap()
    driver_to_team = {entry.driver_id: engine.teams[entry.team_id] for entry in engine.config.grid}

    panel = create_dashboard_layout(
        circuit_name=engine.circuit.name,
        lap=engine.current_lap,
        total_laps=engine.total_laps,
        flag=engine.race_control.current_flag,
        records=records,
        recent_events=["Lap 1: VER set the fastest lap"],
        drivers_dict=engine.drivers,
        teams_dict=driver_to_team,
    )
    assert isinstance(panel, Panel)
    assert engine.circuit.name in panel.renderable.renderables[0].renderable
    assert panel.title == "[bold red]F1 SIMULATION DASHBOARD[/bold red]"


def test_create_dashboard_layout_shows_flag_badges(engine):
    records = engine.step_lap()
    driver_to_team = {entry.driver_id: engine.teams[entry.team_id] for entry in engine.config.grid}

    console = Console(record=True, width=200)
    panel = create_dashboard_layout(
        circuit_name=engine.circuit.name,
        lap=1,
        total_laps=engine.total_laps,
        flag=RaceFlag.GREEN,
        records=records,
        recent_events=[],
        drivers_dict=engine.drivers,
        teams_dict=driver_to_team,
    )
    console.print(panel)
    text = console.export_text()
    assert "Lap 1 /" in text
    assert "LEADER" in text


def test_run_live_race_simulates_full_distance(engine):
    from f1_sim.models.results import RaceResult

    result = run_live_race(engine, speed_seconds_per_lap=0.0)
    assert isinstance(result, RaceResult)
    assert engine.current_lap == engine.total_laps == 3
    assert len(result.driver_summaries) == 20
    assert sorted(s.finish_position for s in result.driver_summaries) == list(range(1, 21))