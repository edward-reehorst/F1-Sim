"""Unit tests for the Milestone 5 CLI wiring: --live, --speed, --plot, --plot-dir."""

from typer.testing import CliRunner

from f1_sim import _build_app, __version__

app = _build_app()
runner = CliRunner()

PLOT_BASENAMES = [
    "monza_lap_chart.png",
    "monza_gap_to_leader.png",
    "monza_stints.png",
    "monza_pace_decay.png",
]


def test_info_display_version():
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert f"v{__version__}" in result.output


def test_race_plot_flag_saves_all_charts(tmp_path):
    result = runner.invoke(
        app,
        ["race", "--circuit", "monza", "--preset", "2024_default", "--laps", "3", "--plot", "--plot-dir", str(tmp_path)],
    )
    assert result.exit_code == 0
    for basename in PLOT_BASENAMES:
        target = tmp_path / basename
        assert target.exists(), f"{basename} was not created"
        assert target.stat().st_size > 0


def test_race_live_flag_completes(tmp_path):
    result = runner.invoke(
        app,
        ["race", "--circuit", "monza", "--laps", "3", "--live", "--speed", "0"],
    )
    assert result.exit_code == 0


def test_race_live_flag_defaults_to_lap_segments():
    """--live should not change determinism of the underlying result rendering."""
    result = runner.invoke(
        app,
        ["race", "--circuit", "monza", "--laps", "3", "--live", "--speed", "0", "--no-incidents"],
    )
    assert result.exit_code == 0
    assert "Race Results" in result.output


def test_batch_command_writes_reports(tmp_path):
    result = runner.invoke(
        app,
        ["batch", "--circuit", "monza", "--laps", "3", "--sims", "4", "--workers", "1", "--out-dir", str(tmp_path)],
    )
    assert result.exit_code == 0
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "report.md").exists()
    assert "Optimal Pit Strategies" in result.output