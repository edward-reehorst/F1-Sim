"""Frequency-matching calibration of per-circuit overtake thresholds.

An empirically measured pace-delta median sets the 50%-pass threshold, but how
often the sim actually overtakes also depends on the incidence and scale of the
per-lap pace deltas the engine generates. This module tunes
``overtake_threshold_seconds`` per circuit so that the *number of successful
overtakes per race in the simulation* matches the real-world rate reconstructed
by :mod:`f1_sim.tuning.overtake_analysis`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from f1_sim.engine import RaceEngine
from f1_sim.loaders import build_race_config


@dataclass
class FrequencyMatchResult:
    """Outcome of a per-circuit threshold search."""

    circuit_id: str
    n_races: int
    n_overtakes: int
    target_per_race: float
    threshold: float
    achieved_per_race: float
    evaluations: int
    converged: bool


def overtake_count_for_threshold(
    circuit_id: str,
    threshold: float,
    *,
    preset: str = "2024_default",
    seeds: range = range(1, 6),
    incidents: bool = True,
    laps: int | None = None,
    max_clean_delta: float = 8.0,
) -> float:
    """Mean number of successful overtakes per race at a given threshold.

    Mirrors :func:`f1_sim.tuning.overtake_analysis.detect_on_track_overtakes`,
    which only counts passes where both cars ran a clean lap:
    * passes completed while either car is on its pit-stop lap are excluded, and
    * passes whose pace gap exceeds ``max_clean_delta`` (a pit in/out lap or a
      lapped car's give-way lap costs 15-25s) are excluded.
    """
    total = 0
    count = 0
    for seed in seeds:
        config = build_race_config(
            circuit_name_or_circuit=circuit_id,
            preset_name=preset,
            laps=laps,
            seed=seed,
        )
        config.overtake_threshold_seconds = threshold
        engine = RaceEngine(config, enable_incidents=incidents)
        result = engine.simulate()

        pit_laps: set[tuple[str, int]] = {
            (stop.driver_id, stop.lap) for summary in result.driver_summaries for stop in summary.pit_stops
        }
        total += sum(
            1
            for e in engine.overtake_events
            if e.success
            and e.pace_delta <= max_clean_delta
            and (e.attacker_id, e.lap) not in pit_laps
            and (e.defender_id, e.lap) not in pit_laps
        )
        count += 1
    return total / count


def match_overtake_threshold(
    circuit_id: str,
    target_per_race: float,
    *,
    n_races: int = 0,
    n_overtakes: int = 0,
    preset: str = "2024_default",
    seeds: range = range(1, 6),
    incidents: bool = True,
    laps: int | None = None,
    lo: float = 0.0,
    hi: float = 5.0,
    tolerance: float = 1.0,
    max_iterations: int = 12,
    max_clean_delta: float = 8.0,
) -> FrequencyMatchResult:
    """Binary-search the threshold whose simulated pass rate hits ``target_per_race``.

    Simulated pass rate is non-increasing in the threshold, so the search
    brackets the target. When a bound cannot reach the target (either no
    threshold reduces the rate enough, or even ``hi`` still overtakes too much),
    the nearest bound is returned with ``converged=False``.
    """
    if target_per_race <= 0.0:
        return FrequencyMatchResult(
            circuit_id=circuit_id,
            n_races=n_races,
            n_overtakes=n_overtakes,
            target_per_race=target_per_race,
            threshold=hi,
            achieved_per_race=0.0,
            evaluations=0,
            converged=True,
        )

    def evaluate(threshold):
        return overtake_count_for_threshold(
            circuit_id,
            threshold,
            preset=preset,
            seeds=seeds,
            incidents=incidents,
            laps=laps,
            max_clean_delta=max_clean_delta,
        )

    evaluations = 0
    f_lo = evaluate(lo)
    evaluations += 1
    if f_lo < target_per_race:
        return FrequencyMatchResult(
            circuit_id=circuit_id,
            n_races=n_races,
            n_overtakes=n_overtakes,
            target_per_race=target_per_race,
            threshold=lo,
            achieved_per_race=f_lo,
            evaluations=evaluations,
            converged=False,
        )

    f_hi = evaluate(hi)
    evaluations += 1
    if f_hi > target_per_race:
        return FrequencyMatchResult(
            circuit_id=circuit_id,
            n_races=n_races,
            n_overtakes=n_overtakes,
            target_per_race=target_per_race,
            threshold=hi,
            achieved_per_race=f_hi,
            evaluations=evaluations,
            converged=False,
        )

    best_threshold, best_rate = lo, f_lo
    for _ in range(max_iterations):
        mid = (lo + hi) / 2.0
        f_mid = evaluate(mid)
        evaluations += 1
        if abs(f_mid - target_per_race) <= abs(best_rate - target_per_race):
            best_threshold, best_rate = mid, f_mid

        if f_mid > target_per_race:
            lo = mid
        else:
            hi = mid

        if abs(f_mid - target_per_race) <= tolerance:
            break

    return FrequencyMatchResult(
        circuit_id=circuit_id,
        n_races=n_races,
        n_overtakes=n_overtakes,
        target_per_race=target_per_race,
        threshold=best_threshold,
        achieved_per_race=best_rate,
        evaluations=evaluations,
        converged=abs(best_rate - target_per_race) <= tolerance,
    )


def load_stat_targets(stats: dict[str, Any]) -> dict[str, float]:
    """Derive per-race overtake targets from an overtake_stats JSON payload.

    Returns ``{circuit_id: n_overtakes / n_races}`` for circuits with data.
    """
    targets: dict[str, float] = {}
    for circuit_id, data in stats.get("circuits", {}).items():
        n_races = data.get("n_races", 0)
        n_overtakes = data.get("n_overtakes", 0)
        if n_races and n_overtakes:
            targets[circuit_id] = n_overtakes / n_races
    return targets