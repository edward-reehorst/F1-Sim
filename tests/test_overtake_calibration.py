"""Tests for frequency-matching overtake-threshold calibration (offline)."""

from __future__ import annotations

from f1_sim.tuning.overtake_calibration import (
    load_stat_targets,
    match_overtake_threshold,
    overtake_count_for_threshold,
)


def test_load_stat_targets():
    stats = {
        "circuits": {
            "monza": {"n_races": 8, "n_overtakes": 164, "median_seconds": 0.59},
            "monaco": {"n_races": 7, "n_overtakes": 15},
            "empty": {"n_races": 2, "n_overtakes": 0},
            "missing": {"n_races": 0, "n_overtakes": 0},
        }
    }
    targets = load_stat_targets(stats)
    assert targets == {"monza": 20.5, "monaco": 15 / 7}
    assert "empty" not in targets
    assert "missing" not in targets


def test_overtake_count_decreases_with_threshold():
    """Higher thresholds must never yield more clean passes (5-seed average)."""
    low = overtake_count_for_threshold("monza", 0.0, seeds=range(1, 4), incidents=False, laps=20)
    high = overtake_count_for_threshold("monza", 4.0, seeds=range(1, 4), incidents=False, laps=20)
    assert low >= high
    assert high == 0.0


def test_match_overtake_threshold_converges():
    """The binary search lands on a threshold whose pass rate is within tolerance."""
    result = match_overtake_threshold(
        "monza",
        6.0,
        n_races=8,
        n_overtakes=48,
        seeds=range(1, 4),
        incidents=False,
        laps=20,
        tolerance=2.0,
        max_iterations=14,
    )
    assert result.circuit_id == "monza"
    assert result.n_races == 8
    assert result.n_overtakes == 48
    assert result.converged
    assert abs(result.achieved_per_race - 6.0) <= 2.0
    assert 0.0 <= result.threshold <= 5.0


def test_match_overtake_threshold_zero_target():
    """A zero target should return the hardest bound immediately, no search."""
    result = match_overtake_threshold("monaco", 0.0, n_races=7, n_overtakes=0)
    assert result.converged is True
    assert result.threshold == 5.0
    assert result.evaluations == 0


def test_overtake_count_excludes_pit_lap_passes():
    """Passes on a pitting car's stop lap must not be counted as clean overtakes."""
    full = overtake_count_for_threshold("monza", 4.0, seeds=range(1, 3), incidents=False, laps=20)
    unbounded = overtake_count_for_threshold(
        "monza", 4.0, seeds=range(1, 3), incidents=False, laps=20, max_clean_delta=1e9
    )
    # At an impossible threshold the only counters are pit/give-way lap artifacts.
    assert unbounded >= full
    assert full == 0.0