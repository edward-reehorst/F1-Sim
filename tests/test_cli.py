"""Tests for the f1-sim CLI calibration surface (`tune`, `--calibration` flags)."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from typer.testing import CliRunner

from f1_sim import generate_synthetic_dataset
from f1_sim.models import CalibrationConfig
from f1_sim import _build_app

runner = CliRunner()
app = _build_app()

_ALIASED_HEADERS = [
    "driver_id",
    "constructor_id",
    "lap_number",
    "compound",
    "lap_time",
    "fuel_remaining",
    "track_status",
    "position",
    "stint",
    "tire_life",
    "session_type",
]


def _write_synthetic_csv(path: Path, overlay: CalibrationConfig | None = None) -> Path:
    dataset = generate_synthetic_dataset("monza", overlay=overlay, noise_seconds=0.2)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_ALIASED_HEADERS)
        writer.writeheader()
        for obs in dataset.observations:
            writer.writerow(
                {
                    "driver_id": obs.driver_id,
                    "constructor_id": obs.team_id,
                    "lap_number": obs.lap,
                    "compound": obs.compound,
                    "lap_time": f"{obs.lap_time:.6f}",
                    "fuel_remaining": f"{obs.fuel_remaining_kg:.6f}",
                    "track_status": obs.flag,
                    "position": obs.position,
                    "stint": obs.stint,
                    "tire_life": obs.tire_age_at_lap_start,
                    "session_type": obs.session,
                }
            )
    return path


def test_tune_command_writes_calibration_json(tmp_path):
    csv_path = _write_synthetic_csv(tmp_path / "practice.csv")
    out_path = tmp_path / "monza.json"

    result = runner.invoke(
        app,
        ["tune", "--circuit", "monza", "--dataset", str(csv_path), "--out", str(out_path)],
    )

    assert result.exit_code == 0, result.output
    assert "Calibration Report" in result.output
    assert "Calibration saved to" in result.output
    assert out_path.exists()

    payload = json.loads(out_path.read_text())
    assert payload["driver_pace_offsets"] != {}
    assert "circuit_base_adjust" in payload
    assert "team_pace_offsets" in payload


def test_tune_command_missing_dataset_fails(tmp_path):
    result = runner.invoke(
        app,
        ["tune", "--circuit", "monza", "--dataset", str(tmp_path / "nope.csv")],
    )
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_tune_command_accepts_grid_penalties(tmp_path):
    csv_path = _write_synthetic_csv(tmp_path / "practice.csv")
    out_path = tmp_path / "monza.json"

    result = runner.invoke(
        app,
        [
            "tune",
            "--circuit",
            "monza",
            "--dataset",
            str(csv_path),
            "--out",
            str(out_path),
            "--penalty",
            "VER:10",
            "--penalty",
            "LEC:3",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Applying 2 grid penalty(ies): VER:10, LEC:3." in result.output


def test_tune_command_rejects_malformed_penalty(tmp_path):
    csv_path = _write_synthetic_csv(tmp_path / "practice.csv")
    for spec in ("VER", "VER:x", "VER:-3"):
        result = runner.invoke(
            app,
            [
                "tune",
                "--circuit",
                "monza",
                "--dataset",
                str(csv_path),
                "--out",
                str(tmp_path / "monza.json"),
                "--penalty",
                spec,
            ],
        )
        assert result.exit_code != 0, f"expected failure for {spec!r}"
        assert "--penalty" in result.output.lower()


def test_tune_command_accepts_holdout_and_anchor(tmp_path):
    csv_path = _write_synthetic_csv(tmp_path / "practice.csv")
    out_path = tmp_path / "monza.json"

    result = runner.invoke(
        app,
        [
            "tune",
            "--circuit",
            "monza",
            "--dataset",
            str(csv_path),
            "--out",
            str(out_path),
            "--holdout",
            "0.2",
            "--anchor-driver",
            "albon",
            "--ridge",
            "0.0",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Anchor driver" in result.output and "albon" in result.output
    payload = json.loads(out_path.read_text())
    assert payload["driver_pace_offsets"].get("albon", 0.0) == 0.0


def test_tune_output_roundtrips_into_race(tmp_path):
    csv_path = _write_synthetic_csv(
        tmp_path / "practice.csv",
        overlay=CalibrationConfig(circuit_base_adjust=-0.35, fuel_penalty_per_kg=0.030),
    )
    cal_path = tmp_path / "monza.json"
    tune_result = runner.invoke(
        app, ["tune", "--circuit", "monza", "--dataset", str(csv_path), "--out", str(cal_path)]
    )
    assert tune_result.exit_code == 0, tune_result.output

    race_result = runner.invoke(
        app,
        ["race", "--circuit", "monza", "--laps", "5", "--no-incidents", "--seed", "7", "--calibration", str(cal_path)],
    )
    assert race_result.exit_code == 0, race_result.output
    assert "Using calibration" in race_result.output
    assert "Race Results" in race_result.output


def test_time_trial_with_and_without_calibration_differs(tmp_path):
    cal_path = tmp_path / "monza.json"
    CalibrationConfig(circuit_base_adjust=-0.35).write_json(cal_path)

    def fastest_lap(use_cal: bool) -> float:
        args = ["time-trial", "--circuit", "monza", "--laps", "10"]
        if use_cal:
            args += ["--calibration", str(cal_path)]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        match = re.search(r"Fastest Lap: ([\d.]+)s", result.output)
        assert match is not None
        return float(match.group(1))

    base = fastest_lap(False)
    calibrated = fastest_lap(True)
    assert calibrated < base - 0.1


def test_batch_command_with_calibration(tmp_path):
    cal_path = tmp_path / "monza.json"
    CalibrationConfig(circuit_base_adjust=-0.35).write_json(cal_path)
    out_dir = tmp_path / "batch_out"

    result = runner.invoke(
        app,
        [
            "batch",
            "--circuit",
            "monza",
            "--sims",
            "2",
            "--no-incidents",
            "--seed",
            "1",
            "--calibration",
            str(cal_path),
            "--out-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Using calibration" in result.output
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "report.md").exists()


def test_ingest_command_listed_in_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ingest" in result.output


def test_ingest_command_rejects_unsupported_session():
    # _session_kind validation runs before any network access, so this is offline-safe.
    result = runner.invoke(
        app,
        ["ingest", "--year", "2024", "--gp", "monza", "--session", "SS"],
    )
    assert result.exit_code != 0
    assert "FP1" in result.output