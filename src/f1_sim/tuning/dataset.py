"""Telemetry dataset loaders supporting canonical and real-world (aliased) CSV/JSON exports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from f1_sim.loaders import load_all_compounds, load_all_drivers, load_all_teams
from f1_sim.models.telemetry import LapObservation, TelemetryDataset

_REQUIRED = frozenset({"driver_id", "team_id", "lap", "compound", "lap_time"})


def _normalize(key: str) -> str:
    """Normalize a raw column name to a lowercase, underscore-joined key."""
    return key.strip().replace("-", "_").replace(" ", "_").replace("/", "_").lower()


_FIELD_ALIASES: dict[str, list[str]] = {
    "driver_id": [
        "driver_id",
        "driver",
        "driver_name",
        "driver_number",
        "drivernumber",
        "number",
        "rapid_driver",
    ],
    "driver_code": [
        "driver_code",
        "drivercode",
        "abbreviation",
        "tla",
        "code",
    ],
    "team_id": [
        "team_id",
        "team",
        "team_name",
        "teamname",
        "constructor",
        "constructor_id",
        "constructorid",
    ],
    "lap": ["lap", "lapnumber", "lap_number", "lap_num", "lapnum", "laps"],
    "compound": ["compound", "compound_name", "compoundname", "tirecompound", "tyrecompound", "tire", "tyre"],
    "lap_time": ["lap_time", "laptime", "lap_time_in_seconds", "laptimeinseconds", "time", "total_time"],
    "fuel_remaining_kg": ["fuel_remaining_kg", "fuel", "fuel_remaining", "fuelremaining", "fuel_mass", "fuelmass", "fuel_kg"],
    "flag": ["flag", "flag_status", "flagstatus", "track_status", "trackstatus", "trkstatus"],
    "position": ["position", "pos", "race_position", "raceposition"],
    "stint": ["stint", "stint_number", "stintnumber", "stintnum"],
    "tire_age_at_lap_start": [
        "tire_age_at_lap_start",
        "tire_age",
        "tireage",
        "tyre_age",
        "tyreage",
        "tire_life",
        "tirelife",
        "tyre_life",
        "tyrelife",
        "tyre_life_in_laps",
    ],
    "session": ["session", "session_type", "sessiontype", "gp_session"],
    "circuit_id": ["circuit_id", "circuit", "circuit_name", "circuitname", "track", "track_id", "trackid"],
}

_ALIAS_LOOKUP: dict[str, str] = {
    norm: field
    for field, aliases in _FIELD_ALIASES.items()
    for norm in {_normalize(a) for a in aliases}
}


def _canonicalize_column(raw: str) -> str | None:
    """Map a raw column header to a canonical field name, or None if unrecognized."""
    return _ALIAS_LOOKUP.get(_normalize(raw))


class _Lookups:
    """Indexes over bundled drivers/teams/compounds for ID resolution."""

    def __init__(self) -> None:
        drivers = load_all_drivers()
        teams = load_all_teams()
        compounds = load_all_compounds()

        self.drivers = drivers
        self.driver_by: dict[str, Any] = {"id": {}, "code": {}, "number": {}, "name": {}}
        for d in drivers.values():
            self.driver_by["id"][d.id] = d
            self.driver_by["code"][d.code.upper()] = d
            self.driver_by["number"][str(d.number)] = d
            self.driver_by["name"][d.name.strip().lower()] = d

        self.teams = teams
        self.team_by: dict[str, Any] = {"id": {}, "name": {}}
        for t in teams.values():
            self.team_by["id"][t.id] = t
            self.team_by["name"][t.name.strip().lower()] = t

        self.compounds = compounds

    def resolve_driver(self, value: Any, raw_value: Any) -> str:
        if value is None:
            raise ValueError("Missing driver reference in a data row")
        text = str(value).strip()
        driver = (
            self.driver_by["id"].get(text)
            or self.driver_by["code"].get(text.upper())
            or self.driver_by["name"].get(text.lower())
            or self.driver_by["number"].get(text.split(".")[0])
        )
        if driver is None:
            raise ValueError(
                f"Unknown driver {text!r} (source value: {raw_value!r}). "
                f"Known codes: {sorted(self.driver_by['code'])}"
            )
        return driver.id

    def resolve_team(self, value: Any, raw_value: Any) -> str:
        if value is None:
            raise ValueError("Missing team reference in a data row")
        text = str(value).strip()
        team = self.team_by["id"].get(text) or self.team_by["name"].get(text.lower())
        if team is None:
            candidates = [
                t
                for t in self.teams.values()
                if text.lower() in t.name.lower() or t.name.lower() in text.lower()
            ]
            if len(candidates) == 1:
                team = candidates[0]
            elif len(candidates) > 1:
                raise ValueError(
                    f"Ambiguous team {text!r} (source value: {raw_value!r}). "
                    f"Matches: {[c.id for c in candidates]}"
                )
        if team is None:
            raise ValueError(
                f"Unknown team {text!r} (source value: {raw_value!r}). "
                f"Known teams: {sorted(self.team_by['id'])}"
            )
        return team.id

    def resolve_compound(self, value: Any) -> str:
        text = str(value).strip()
        if text.lower() not in self.compounds:
            raise ValueError(
                f"Unknown compound {text!r}. "
                f"Known compounds: {sorted(c.compound_name for c in self.compounds.values())}"
            )
        return self.compounds[text.lower()].compound_name


def _read_rows(path: Path) -> list[dict[str, Any]]:
    """Read a dataset file into a list of raw key/value row dicts."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", newline="", encoding="utf-8") as f:
            return [dict(row) for row in csv.DictReader(f)]
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            records = data.get("observations", data.get("laps", data.get("records", [])))
            if not isinstance(records, list):
                raise ValueError("JSON dataset must be a list of lap records or contain an 'observations' list")
            rows: list[dict[str, Any]] = []
            top_level_circuit = data.get("circuit_id")
            for record in records:
                if not isinstance(record, dict):
                    raise ValueError("JSON lap records must be objects")
                row = dict(record)
                if top_level_circuit and not row.get("circuit_id"):
                    row["circuit_id"] = top_level_circuit
                rows.append(row)
            return rows
        raise ValueError("JSON dataset must be a list or an object with an 'observations' list")
    raise ValueError(f"Unsupported dataset format '{suffix}'. Use .csv or .json")


def _map_row(raw: dict[str, Any]) -> dict[str, Any]:
    """Map a raw row to canonical keys (aliased columns tolerated).

    Unknown columns are preserved in the canonical dict so they survive into
    ``source`` for provenance or penalty-matching (e.g. the original driver code
    column that FastF1 ingests produce).
    """
    canonical: dict[str, Any] = {}
    for raw_key, value in raw.items():
        field = _canonicalize_column(raw_key)
        if field is None:
            canonical[raw_key] = value
        else:
            canonical[field] = value
    return canonical


def _parse_scalar(value: Any, cast, field: str) -> Any:
    try:
        return cast(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-numeric value for '{field}': {value!r}") from exc


def load_dataset(path: str | Path, *, circuit_id: str | None = None) -> TelemetryDataset:
    """Load a telemetry dataset from a self-described JSON/CSV or a real-world export.

    Aliased column mapping tolerates real tooling headers (e.g. ``LapNumber`` ->
    ``lap``, ``LapTime`` -> ``lap_time``, ``TyreLife``/``TyreAge`` ->
    ``tire_age_at_lap_start``, ``DriverNumber``/``DriverCode`` -> canonical driver id).
    Unknown drivers, teams, or compounds raise clear, actionable errors.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    raw_rows = _read_rows(path)
    lookups = _Lookups()

    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        row = _map_row(raw)
        # Backward compatibility: when only a driver_code/drivercode column is
        # present (no driver_id or driver column), use it as the driver_id.
        if not row.get("driver_id") and row.get("driver_code"):
            row["driver_id"] = row["driver_code"]
        missing = sorted(
            field for field in _REQUIRED
            if row.get(field) in (None, "", "None")
        )
        if missing:
            raise ValueError(f"Row is missing required fields {missing}: {raw}")
        rows.append(row)

    circuit: str | None = None
    circuit_values = [row.get("circuit_id") for row in rows]
    non_null = [v for v in circuit_values if v not in (None, "")]
    if non_null:
        circuit = str(non_null[0]).strip()
    if circuit is None:
        circuit = circuit_id
    if circuit is None:
        raise ValueError("Dataset has no circuit column; pass circuit_id explicitly")

    observations: list[LapObservation] = []
    for row in rows:
        driver_id = lookups.resolve_driver(row.get("driver_id"), row.get("driver_id"))
        lap = _parse_scalar(row.get("lap"), int, "lap")
        lap_time = _parse_scalar(row.get("lap_time"), float, "lap_time")
        if lap_time <= 0.0:
            raise ValueError(f"Non-positive lap time {lap_time!r}")
        compound = lookups.resolve_compound(row.get("compound"))

        fuel: float | None = None
        if row.get("fuel_remaining_kg") not in (None, ""):
            fuel = _parse_scalar(row.get("fuel_remaining_kg"), float, "fuel_remaining_kg")

        tire_age = 0
        if row.get("tire_age_at_lap_start") not in (None, ""):
            tire_age = _parse_scalar(row.get("tire_age_at_lap_start"), int, "tire_age_at_lap_start")

        position: int | None = None
        if row.get("position") not in (None, ""):
            position = _parse_scalar(row.get("position"), int, "position")

        stint: int | None = None
        if row.get("stint") not in (None, ""):
            stint = _parse_scalar(row.get("stint"), int, "stint")

        flag = str(row.get("flag") or "GREEN").strip().upper()
        session = row.get("session")
        session = str(session).strip() if session not in (None, "") else None

        observations.append(
            LapObservation(
                session=session,
                driver_id=driver_id,
                team_id=lookups.resolve_team(row.get("team_id"), row.get("team_id")),
                lap=lap,
                compound=compound,
                lap_time=float(lap_time),
                fuel_remaining_kg=fuel,
                flag=flag,
                position=position,
                stint=stint,
                tire_age_at_lap_start=tire_age,
                source=dict(row),
            )
        )

    return TelemetryDataset(circuit_id=circuit, observations=observations)