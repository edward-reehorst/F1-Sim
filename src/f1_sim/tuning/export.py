"""Export a ``TelemetryDataset`` to CSV or JSON in the format ``load_dataset`` reads."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from f1_sim.models.telemetry import LapObservation, TelemetryDataset

# Canonical column order; aliased headers are what real-world tooling tends to use.
_CANONICAL_FIELDS = [
    "session",
    "driver_id",
    "driver_code",
    "team_id",
    "lap",
    "compound",
    "lap_time",
    "fuel_remaining_kg",
    "flag",
    "position",
    "stint",
    "tire_age_at_lap_start",
]

_ALIASED_HEADERS = [
    "session_type",
    "driver",
    "driver_code",
    "constructor",
    "lap_number",
    "compound",
    "lap_time",
    "fuel_remaining",
    "track_status",
    "position",
    "stint",
    "tire_life",
]


def _field_map(aliased: bool) -> list[tuple[str, str]]:
    """Return (output_key, canonical_field) pairs for CSV writing."""
    if not aliased:
        return [(f, f) for f in _CANONICAL_FIELDS]
    return list(zip(_ALIASED_HEADERS, _CANONICAL_FIELDS, strict=True))


def _row(obs: LapObservation, pairs: list[tuple[str, str]]) -> dict[str, object]:
    data = {
        "session": obs.session,
        "driver_id": obs.driver_id,
        "driver_code": (obs.source or {}).get("driver_code", ""),
        "team_id": obs.team_id,
        "lap": obs.lap,
        "compound": obs.compound,
        "lap_time": obs.lap_time,
        "fuel_remaining_kg": obs.fuel_remaining_kg,
        "flag": obs.flag,
        "position": obs.position,
        "stint": obs.stint,
        "tire_age_at_lap_start": obs.tire_age_at_lap_start,
    }
    return {out_key: ("" if data[field] is None else data[field]) for out_key, field in pairs}


def write_observations_csv(dataset: TelemetryDataset, path: str | Path, *, aliased: bool = True) -> Path:
    """Write a dataset to CSV using headers compatible with ``load_dataset``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pairs = _field_map(aliased)
    headers = [key for key, _ in pairs]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for obs in dataset.observations:
            writer.writerow(_row(obs, pairs))
    return path


def write_observations_json(dataset: TelemetryDataset, path: str | Path) -> Path:
    """Write a dataset as self-described JSON (top-level ``circuit_id`` + ``observations``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"circuit_id": dataset.circuit_id, "observations": [_row(o, _field_map(False)) for o in dataset.observations]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path