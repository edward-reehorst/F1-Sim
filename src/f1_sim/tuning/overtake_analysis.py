"""Empirical overtaking-difficulty estimation from real F1 race sessions.

The sim's overtake check compares a car's per-lap pace gap against a circuit
offset (``overtake_threshold_seconds``). These numbers used to be derived from a
hand-chosen ``overtaking_difficulty`` index. This module instead measures the
real-world distribution of the same quantity: on-track passes reconstructed from
official race lap data.

A single lap time is a noisy readout (DRS, tow, traffic, a defending line), so
the reported ``pace_delta`` is a robust aggregate of the defender-minus-attacker
per-lap deltas over a window of laps *around* the pass, excluding the pass lap
itself by default. The aggregator is tunable (median, mean, or a quantile).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from f1_sim.tuning.fastf1 import lap_time_seconds

# FastF1-style GP event names per bundled circuit, tried in order. The first name
# that resolves in a given season wins (e.g. "Italian Grand Prix" full name,
# "Monza" location, or a short alias).
CIRCUIT_GP_NAMES: dict[str, tuple[str, ...]] = {
    "monza": ("Italian Grand Prix", "Monza", "Italian"),
    "silverstone": ("British Grand Prix", "Silverstone", "British"),
    "spa": ("Belgian Grand Prix", "Spa-Francorchamps", "Belgian"),
    "monaco": ("Monaco Grand Prix", "Monaco",),
}

# Canonical event names a circuit's race must resolve to. FastF1 silently
# "corrects" a non-existent event to the closest match (e.g. the cancelled 2020
# Monaco GP maps to the Italian Grand Prix at Monza), so the resolved event is
# validated against this set to avoid attributing another race's data.
CIRCUIT_EVENT_NAMES: dict[str, frozenset[str]] = {
    "monza": frozenset({"italian grand prix"}),
    "silverstone": frozenset({"british grand prix"}),
    "spa": frozenset({"belgian grand prix"}),
    "monaco": frozenset({"monaco grand prix"}),
}

_NEEDED_COLUMNS = {"Driver", "LapNumber", "LapTime", "Position", "PitInTime", "PitOutTime", "TrackStatus"}


@dataclass
class OvertakeSample:
    """A single detected on-track pass in a real race session."""

    circuit_id: str
    year: int
    lap: int
    attacker: str
    defender: str
    pace_delta: float
    window_n: int = 0


@dataclass
class CircuitOvertakeStats:
    """Pooled overtake pace-delta statistics for one circuit across seasons."""

    circuit_id: str
    n_races: int
    n_samples: int
    q1: float | None
    median: float | None
    q3: float | None

    @property
    def sample_size(self) -> int:
        return self.n_samples

    @property
    def threshold_seconds(self) -> float | None:
        return self.median


_AGGREGATORS = ("median", "mean", "quantile")


def aggregate_pace_delta(deltas: list[float], method: str = "median", quantile: float = 0.10) -> float:
    """Reduce a window of per-lap pace deltas to a single representative gap.

    ``method`` is one of ``"median"``, ``"mean"`` or ``"quantile"``. For
    ``"quantile"`` the ``quantile`` argument (0 < q < 1) selects the percentile,
    e.g. ``0.10`` gives the 10th-percentile gap (a conservative threshold).
    """
    if not deltas:
        raise ValueError("Cannot aggregate an empty delta window.")
    if method == "median":
        return float(np.median(deltas))
    if method == "mean":
        return float(np.mean(deltas))
    if method == "quantile":
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"quantile must be in (0, 1), got {quantile!r}.")
        return float(np.percentile(deltas, quantile * 100.0))
    raise ValueError(f"Unknown aggregator {method!r}; expected one of {_AGGREGATORS}.")


def detect_on_track_overtakes(
    laps: Any,
    *,
    circuit_id: str,
    year: int,
    aggregator: str = "median",
    quantile: float = 0.10,
    window_before: int = 3,
    window_after: int = 3,
    min_window_laps: int = 3,
    include_pass_lap: bool = False,
) -> list[OvertakeSample]:
    """Reconstruct on-track overtakes from a race session's lap data.

    A pass is a 1:1 swap of adjacent positions between consecutive lap lists,
    where both cars ran the pass lap fully under a green flag (track status 1)
    without a pit stop.

    The reported ``pace_delta`` is a robust aggregate of the per-lap
    ``defender_lap_time - attacker_lap_time`` deltas over a window of
    ``window_before`` .. ``window_after`` laps around the pass (the pass lap
    itself is excluded unless ``include_pass_lap`` is true), so a single
    DRS/tow/defensive-line lap cannot dominate. Laps where either driver lacks a
    green, non-pit lap time (SC/VSC, pit stops, retirement, lapping) are skipped.
    Passes with fewer than ``min_window_laps`` valid window laps are dropped.
    Samples are only kept when the aggregate gap is positive (the engine gates
    overtake attempts on ``pace_delta > 0``; see
    :func:`f1_sim.engine.race.RaceEngine._advance_race_lap`).
    """
    if aggregator not in _AGGREGATORS:
        raise ValueError(f"Unknown aggregator {aggregator!r}; expected one of {_AGGREGATORS}.")
    if aggregator == "quantile" and not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile!r}.")

    missing = {c for c in _NEEDED_COLUMNS if c not in laps.columns}
    if missing:
        raise ValueError(f"Laps data is missing required columns: {sorted(missing)}")

    df = laps[list(_NEEDED_COLUMNS)].copy()
    df = df[df["LapTime"].notna()]
    # Fully-green lap: the track status witnessed across the whole lap is "1".
    df = df[df["TrackStatus"] == "1"]
    # Exclude rows where either car is in, entering, or leaving the pit lane.
    df = df[pd.isna(df["PitInTime"]) & pd.isna(df["PitOutTime"])]
    df = df[df["Position"].notna()]

    if df.empty:
        return []

    def _code(v: object) -> str:
        return str(v).strip().upper()

    order_by_lap: dict[int, dict[int, str]] = {}
    for lap_n, sub in df.groupby("LapNumber"):
        order_by_lap[int(lap_n)] = {
            int(row["Position"]): _code(row["Driver"]) for _, row in sub.iterrows()
        }

    times: dict[tuple[int, str], float] = {
        (int(row["LapNumber"]), _code(row["Driver"])): lap_time_seconds(row["LapTime"])
        for _, row in df.iterrows()
    }

    samples: list[OvertakeSample] = []
    lap_numbers = sorted(order_by_lap)
    for lap_prev, lap_cur in zip(lap_numbers, lap_numbers[1:]):
        prev_order = order_by_lap[lap_prev]
        cur_order = order_by_lap[lap_cur]
        for pos in range(1, min(len(prev_order), len(cur_order))):
            defender = prev_order.get(pos)
            attacker = prev_order.get(pos + 1)
            if not defender or not attacker:
                continue
            if cur_order.get(pos) != attacker or cur_order.get(pos + 1) != defender:
                continue

            deltas: list[float] = []
            for lap in range(lap_cur - window_before, lap_cur + window_after + 1):
                if lap == lap_cur and not include_pass_lap:
                    continue
                def_time = times.get((lap, defender))
                att_time = times.get((lap, attacker))
                if def_time is None or att_time is None:
                    continue
                deltas.append(def_time - att_time)
            if len(deltas) < min_window_laps:
                continue

            pace_delta = aggregate_pace_delta(deltas, aggregator, quantile)
            if pace_delta > 0:
                samples.append(
                    OvertakeSample(
                        circuit_id=circuit_id,
                        year=year,
                        lap=lap_cur,
                        attacker=attacker,
                        defender=defender,
                        pace_delta=pace_delta,
                        window_n=len(deltas),
                    )
                )
    return samples


def pace_delta_stats(samples: list[OvertakeSample]) -> dict[str, CircuitOvertakeStats]:
    """Pool overtake samples into per-circuit quarter/median statistics."""
    grouped: dict[str, list[OvertakeSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.circuit_id, []).append(sample)

    stats: dict[str, CircuitOvertakeStats] = {}
    for circuit_id, circuit_samples in grouped.items():
        deltas = sorted(s.pace_delta for s in circuit_samples)
        n_races = len({s.year for s in circuit_samples})
        if deltas:
            q1, med, q3 = np.percentile(deltas, [25, 50, 75])
            stats[circuit_id] = CircuitOvertakeStats(
                circuit_id=circuit_id,
                n_races=n_races,
                n_samples=len(deltas),
                q1=float(q1),
                median=float(med),
                q3=float(q3),
            )
        else:
            stats[circuit_id] = CircuitOvertakeStats(
                circuit_id=circuit_id, n_races=n_races, n_samples=0, q1=None, median=None, q3=None
            )
    return stats


def collect_overtake_samples(
    years: range,
    circuits: list[str] | None = None,
    *,
    cache_dir: str | None = None,
    aggregator: str = "median",
    quantile: float = 0.10,
    window_before: int = 3,
    window_after: int = 3,
    min_window_laps: int = 3,
    include_pass_lap: bool = False,
) -> list[OvertakeSample]:
    """Fetch race sessions and reconstruct on-track overtakes for all seasons.

    Sessions that FastF1 cannot resolve (e.g. a circuit not on that season's
    calendar, or canceled events) are skipped silently. The windowing/aggregation
    arguments are forwarded to :func:`detect_on_track_overtakes`.
    """
    import fastf1

    if cache_dir:
        cache_path = Path(cache_dir).expanduser()
        cache_path.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(cache_path)

    circuit_ids = circuits or sorted(CIRCUIT_GP_NAMES)
    samples: list[OvertakeSample] = []
    for year in years:
        for circuit_id in circuit_ids:
            session = _try_race_session(fastf1, year, circuit_id)
            if session is None:
                continue
            try:
                session.load(laps=True, telemetry=False, weather=False, messages=False)
                laps = session.laps
            except Exception:
                continue
            if laps is None or laps.empty:
                continue
            samples.extend(
                detect_on_track_overtakes(
                    laps,
                    circuit_id=circuit_id,
                    year=year,
                    aggregator=aggregator,
                    quantile=quantile,
                    window_before=window_before,
                    window_after=window_after,
                    min_window_laps=min_window_laps,
                    include_pass_lap=include_pass_lap,
                )
            )
    return samples


def _try_race_session(fastf1: Any, year: int, circuit_id: str) -> Any | None:
    """Return a race session object for the first matching GP name, else None.

    The resolved event name is checked against :data:`CIRCUIT_EVENT_NAMES` so a
    FastF1 fuzzy-correction to a different race is never accepted.
    """
    allowed = CIRCUIT_EVENT_NAMES[circuit_id]
    for gp_name in CIRCUIT_GP_NAMES[circuit_id]:
        try:
            session = fastf1.get_session(year, gp_name, "R")
        except Exception:
            continue
        event_name = str(session.event.get("EventName", "")).strip().lower()
        if event_name in allowed:
            return session
    return None