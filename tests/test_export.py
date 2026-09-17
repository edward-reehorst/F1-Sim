"""Tests for the observation exporter (CSV/JSON writers)."""

from __future__ import annotations

import json
import csv

from f1_sim.tuning import generate_synthetic_dataset, load_dataset, write_observations_csv, write_observations_json


def _without_source(dataset) -> list[dict]:
    return [{k: v for k, v in o.model_dump().items() if k != "source"} for o in dataset.observations]


def _assert_equal_minus_source(loaded, dataset):
    assert _without_source(loaded) == _without_source(dataset)


def test_csv_roundtrip_through_load_dataset(tmp_path):
    dataset = generate_synthetic_dataset("monza", noise_seconds=0.2, seed=3)
    path = tmp_path / "practice.csv"
    write_observations_csv(dataset, path)

    assert path.exists()
    with path.open() as fh:
        headers = list(csv.DictReader(fh).fieldnames)
    assert "driver" in headers and "lap_number" in headers and "tire_life" in headers

    loaded = load_dataset(path, circuit_id="monza")
    assert loaded.circuit_id == dataset.circuit_id
    _assert_equal_minus_source(loaded, dataset)


def test_json_roundtrip_through_load_dataset(tmp_path):
    dataset = generate_synthetic_dataset("monza", noise_seconds=0.2, seed=3)
    path = tmp_path / "practice.json"
    write_observations_json(dataset, path)

    payload = json.loads(path.read_text())
    assert payload["circuit_id"] == "monza"
    assert len(payload["observations"]) == len(dataset)

    loaded = load_dataset(path)
    assert loaded.circuit_id == dataset.circuit_id
    _assert_equal_minus_source(loaded, dataset)


def test_csv_canonical_headers_roundtrip(tmp_path):
    dataset = generate_synthetic_dataset("silverstone", noise_seconds=0.0, seed=1)
    path = tmp_path / "canonical.csv"
    write_observations_csv(dataset, path, aliased=False)

    loaded = load_dataset(path, circuit_id="silverstone")
    _assert_equal_minus_source(loaded, dataset)


def test_exporter_ignores_fuel_when_absent(tmp_path):
    from f1_sim.tuning import generate_synthetic_dataset

    dataset = generate_synthetic_dataset("monza", noise_seconds=0.0, include_fuel=False, seed=1)
    path = tmp_path / "no_fuel.csv"
    write_observations_csv(dataset, path)
    loaded = load_dataset(path, circuit_id="monza")
    assert all(o.fuel_remaining_kg is None for o in loaded.observations)
    assert all(o.fuel_remaining_kg is None for o in dataset.observations)