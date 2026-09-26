"""Fetch racing-telemetry observations via FastF1 and map them into the local schema."""

from __future__ import annotations

from typing import Any

from f1_sim.models.calibration import GridPenalty
from f1_sim.models.telemetry import LapObservation, TelemetryDataset
from f1_sim.tuning.dataset import _Lookups

# FastF1 session identifier -> (canonical identifier form, local session type).
_SESSION_IDENTIFIERS = {
    "FP1": ("FP1", "practice1"),
    "FP2": ("FP2", "practice2"),
    "FP3": ("FP3", "practice3"),
    "Q": ("Q", "qualifying"),
    "QUALIFYING": ("Q", "qualifying"),
    "R": ("R", "race"),
    "RACE": ("R", "race"),
}

# Bundled circuit ids and the event/location/country strings that map to them.
_CIRCUIT_KEYS: dict[str, tuple[str, ...]] = {
    "monza": ("monza", "italian", "italy"),
    "silverstone": ("silverstone", "british", "great britain"),
    "spa": ("spa", "spa-francorchamps", "belgian", "belgium"),
    "monaco": ("monaco",),
}


def _session_kind(identifier: str) -> tuple[str, str]:
    """Return (canonical FastF1 identifier, local session type)."""
    key = identifier.strip().upper()
    if key not in _SESSION_IDENTIFIERS:
        raise ValueError(
            f"Unsupported session identifier {identifier!r}; use one of "
            f"FP1, FP2, FP3, Q, R."
        )
    return _SESSION_IDENTIFIERS[key]


def _resolve_circuit_id(session: Any, circuit_id: str | None) -> str:
    """Map a FastF1 session to a bundled circuit id, or use the caller's override."""
    if circuit_id is not None:
        return circuit_id.strip().lower()

    try:
        event = session.event
        haystack = " ".join(str(v).lower() for v in event.values())
    except Exception:
        haystack = str(session).lower()

    for circuit, keys in _CIRCUIT_KEYS.items():
        if any(key in haystack for key in keys):
            return circuit

    raise ValueError(
        f"Could not identify circuit from session {session!r}; pass --circuit explicitly. "
        f"Bundled circuits: {', '.join(sorted(_CIRCUIT_KEYS))}"
    )


def lap_time_seconds(value: Any) -> float:
    """Coerce a FastF1 lap time (Timedelta or seconds float) to seconds."""
    if hasattr(value, "total_seconds"):
        return float(value.total_seconds())
    return float(value)


def unknown_driver_codes(laps: Any, lookups: _Lookups | None = None) -> list[str]:
    """Driver codes present in ``laps`` that the bundled grid cannot resolve."""
    lookups = lookups or _Lookups()
    codes = sorted({str(d) for d in laps["Driver"]})
    known = set(lookups.driver_by["code"])
    return [c for c in codes if c not in known]


def unknown_team_names(laps: Any, lookups: _Lookups | None = None) -> list[str]:
    """Team names present in ``laps`` that the bundled grid cannot resolve."""
    lookups = lookups or _Lookups()
    teams = sorted({str(t).strip() for t in laps["Team"]})
    unknown: list[str] = []
    for team in teams:
        try:
            lookups.resolve_team(team, team)
        except ValueError:
            unknown.append(team)
    return unknown


def grid_positions_by_code(results: Any, *, grid_position: bool = True) -> dict[str, int]:
    """Positions keyed by driver code from a results frame.

    With ``grid_position=True`` (default) the official starting grid
    ``GridPosition`` (penalties applied) is read, returning empty when the column
    is not populated. With ``grid_position=False`` the classification ``Position``
    (raw qualifying order) is read instead. Rows with no usable value are skipped.
    """
    column = "GridPosition" if grid_position else "Position"
    out: dict[str, int] = {}
    if column not in results.columns:
        return out
    for _, row in results.iterrows():
        value = row.get(column)
        if value is not None and value == value:  # not NaN
            out[str(row["Abbreviation"]).strip().upper()] = int(value)
    return out


def _fill_qualifying_positions(laps: Any, position_by_code: dict[str, int]) -> Any:
    """Set each lap's ``Position`` from a driver-code -> grid position map."""
    out_laps = laps.copy(deep=False)
    out_laps["Position"] = (
        out_laps["Driver"].map(lambda code: position_by_code.get(str(code).strip().upper())).astype("float64")
    )
    return out_laps


def apply_grid_penalties(
    classification: dict[str, int],
    penalties: list[GridPenalty],
    *,
    grid_size: int = 22,
) -> dict[str, int]:
    """Return a starting grid after applying cumulative grid drops.

    ``classification`` maps driver codes to qualifying positions. Working
    fastest-qualifier-first (the F1 rule that breaks ties in favour of the driver
    who qualified ahead):

    - Penalized drivers first: each drops ``places`` (stacked when penalized more
      than once). ``places=0`` drops to the back of the grid; any total drop past
      the grid back row is clamped to the back. If a penalty lands on an already
      taken slot, the driver takes the next free slot behind it.
    - Unpenalized drivers then fill the lowest free slots in qualifying order, so
      they move up to claim slots vacated by penalized drivers (e.g. pole-sitter
      dropped to the back moves the whole field up one).

    Penalties for drivers absent from the classification are ignored. Returns the
    same driver-code -> grid-position mapping with contiguous, unique slots. The
    back row is ``min(grid_size, number of classified drivers)``, so partial
    grids (e.g. only drivers in the bundled dataset) stay compact.
    """
    if grid_size < 1:
        raise ValueError(f"grid_size must be >= 1, got {grid_size}")

    quals = sorted(classification.items(), key=lambda kv: (kv[1], kv[0]))
    total: dict[str, int] = {}
    for p in penalties:
        if p.places < 0:
            raise ValueError(f"grid drop must be >= 0, got {p.places}")
        total[p.driver_id] = total.get(p.driver_id, 0) + p.places

    back = min(grid_size, len(classification))
    grid: dict[str, int] = {}
    occupied: set[int] = set()

    def _place(code: str, target: int) -> None:
        while target in occupied and target < back:
            target += 1
        grid[code] = target
        occupied.add(target)

    # Phase 1: penalized drivers, fastest qualifying position first.
    for code, quali in quals:
        drops = total.get(code, 0)
        if code not in total:
            continue
        target = back if drops == 0 else min(quali + drops, back)
        _place(code, target)

    # Phase 2: unpenalized drivers claim the lowest free slot in quali order.
    for code, _quali in quals:
        if code in grid:
            continue
        _place(code, min(s for s in range(1, back + 1) if s not in occupied))

    return dict(sorted(grid.items(), key=lambda kv: (kv[1], kv[0])))


def _load_session(session: Any, *, year: int, gp: str, identifier: str, laps: bool = True) -> Any:
    """Load a FastF1 session for lap-level use and surface swallowed load failures.

    ``Session.load`` wraps its lap fetch in ``@soft_exceptions``, so an
    unavailable timing feed is only logged as ``Failed to load timing data!`` and
    leaves ``session.laps`` unset, which later surfaces as a misleading
    ``DataNotLoadedError``. Telemetry, weather and messages are disabled because
    the exported observations are lap-level only (and each adds a large
    download). With ``laps=False`` only session results are loaded.
    """
    label = f"{year} {gp} {identifier}"
    if not session.f1_api_support:
        raise ValueError(
            f"{label}: the official timing API does not support this session, "
            "so no lap data can be fetched."
        )
    session.load(laps=laps, telemetry=False, weather=False, messages=False)
    if laps and getattr(session, "_laps", None) is None:
        raise ValueError(
            f"{label}: FastF1 loaded no lap data (it logged \"Failed to load timing "
            "data!\" and hid the underlying error). The session most likely has not "
            "finished yet, or its timing feed is not published; retry once the "
            "session is over."
        )
    return session


def _laps_to_dataset(
    laps: Any,
    *,
    circuit_id: str,
    session_kind: str,
    lookups: _Lookups,
    pos_col: str | None = "Position",
    skip_unknown_drivers: bool = False,
    skip_unknown_teams: bool = False,
) -> TelemetryDataset:
    """Map a FastF1-style lap DataFrame into a local dataset (no network).

    ``laps`` should already be green-flag laps (see :func:`fetch_session`).
    """
    observations: list[LapObservation] = []
    unknown = unknown_driver_codes(laps, lookups)
    unknown_teams = unknown_team_names(laps, lookups)
    for _, row in laps.iterrows():
        lap_time = row.get("LapTime")
        if lap_time is None or lap_time != lap_time:  # NaN
            continue

        driver_code = str(row["Driver"]).strip()
        if driver_code in unknown:
            if skip_unknown_drivers:
                continue
            raise ValueError(
                f"Unknown driver {driver_code!r} in fetched session. Use skip_unknown_drivers=True "
                f"(default in fetch_session) to drop laps from drivers outside the bundled grid."
            )

        team_name = str(row.get("Team")).strip()
        if team_name in unknown_teams:
            if skip_unknown_teams:
                continue
            raise ValueError(
                f"Unknown team {team_name!r} in fetched session. Use skip_unknown_teams=True "
                f"(default in fetch_session) to drop laps from teams outside the bundled grid."
            )

        driver_id = lookups.resolve_driver(row["Driver"], row["Driver"])
        team_id = lookups.resolve_team(row.get("Team"), row.get("Team"))

        tire_age_raw = row.get("TyreLife", 0)
        try:
            tire_age = int(tire_age_raw) if tire_age_raw == tire_age_raw else 0
        except (TypeError, ValueError):
            tire_age = 0

        stint_raw = row.get("Stint")
        try:
            stint = int(stint_raw) if stint_raw == stint_raw else None
        except (TypeError, ValueError):
            stint = None

        position_raw = row.get(pos_col) if pos_col else None
        position = None
        if position_raw is not None and position_raw == position_raw:
            try:
                position = int(position_raw)
            except (TypeError, ValueError):
                position = None

        observations.append(
            LapObservation(
                session=session_kind,
                driver_id=driver_id,
                team_id=team_id,
                lap=int(row["LapNumber"]),
                compound=lookups.resolve_compound(row["Compound"]),
                lap_time=lap_time_seconds(lap_time),
                flag="GREEN",
                position=position,
                stint=stint,
                tire_age_at_lap_start=tire_age,
                source={"driver_code": str(row["Driver"]), "team_name": str(row.get("Team"))},
            )
        )

    return TelemetryDataset(circuit_id=circuit_id, observations=observations)


def fetch_session(
    year: int,
    gp: str,
    session_identifier: str,
    circuit_id: str | None = None,
    *,
    drop_unknown_drivers: bool = True,
    drop_unknown_teams: bool = True,
) -> TelemetryDataset:
    """Download a FastF1 session's green-flag laps and map them into a local dataset.

    Requires network access to the FastF1 timing API. Only laps run under a green
    flag (track status ``1``) are exported; incomplete laps (missing lap times) are
    dropped. Per-lap fuel is intentionally not fetched: the exporter calls
    ``session.load()`` with telemetry disabled, so observations carry no fuel column
    and the calibrator skips the fuel term.

    Laps from drivers or teams outside the bundled grid (e.g. weekend substitutes
    or new teams) are dropped by default and never raise; set
    ``drop_unknown_drivers=False`` / ``drop_unknown_teams=False`` to fail loudly
    instead.
    """
    import fastf1

    canonical_id, session_kind = _session_kind(session_identifier)
    session = _load_session(
        fastf1.get_session(year, gp, canonical_id), year=year, gp=gp, identifier=canonical_id
    )
    green_laps = session.laps.pick_track_status("1")
    if session_kind == "qualifying":
        # Qualifying laps have no position; the official starting grid (penalties
        # applied) lives in the race session's GridPosition. Fall back to the
        # qualifying classification only when the race data is not populated yet.
        race_session = _load_session(
            fastf1.get_session(year, gp, "R"), year=year, gp=gp, identifier="R", laps=False
        )
        grid_positions = grid_positions_by_code(race_session.results)
        if not grid_positions:
            grid_positions = grid_positions_by_code(session.results, grid_position=False)
        green_laps = _fill_qualifying_positions(green_laps, grid_positions)
    resolved_circuit = _resolve_circuit_id(session, circuit_id)
    return _laps_to_dataset(
        green_laps,
        circuit_id=resolved_circuit,
        session_kind=session_kind,
        lookups=_Lookups(),
        skip_unknown_drivers=drop_unknown_drivers,
        skip_unknown_teams=drop_unknown_teams,
    )