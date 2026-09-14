"""Unit tests for Milestone 3: Multi-Car Traffic & Overtaking Model."""

import pytest

from f1_sim.engine import (
    RaceEngine,
    compute_overtake_probability,
    compute_overtake_threshold,
    evaluate_overtake,
)
from f1_sim.loaders import build_race_config, load_circuit, load_compound, load_driver, load_team
from f1_sim.models import GridEntry, RaceConfig


def test_overtake_threshold_scaling():
    """Circuits with higher overtaking difficulty must require higher pace delta to pass."""
    t_monza = compute_overtake_threshold(circuit_overtaking_difficulty=0.25)
    t_silverstone = compute_overtake_threshold(circuit_overtaking_difficulty=0.45)
    t_monaco = compute_overtake_threshold(circuit_overtaking_difficulty=0.95)

    assert t_monza < t_silverstone < t_monaco
    assert t_monza == pytest.approx(0.35 * (1.0 + 0.25))
    assert t_monaco == pytest.approx(0.35 * (1.0 + 0.95))


def test_overtake_probability_monotonicity():
    """Probability of overtake must increase monotonically with pace delta and be 0 when <= 0."""
    threshold = 0.50

    p_neg = compute_overtake_probability(pace_delta=-0.10, threshold=threshold)
    p_zero = compute_overtake_probability(pace_delta=0.0, threshold=threshold)
    p_low = compute_overtake_probability(pace_delta=0.20, threshold=threshold)
    p_mid = compute_overtake_probability(pace_delta=threshold, threshold=threshold)
    p_high = compute_overtake_probability(pace_delta=0.80, threshold=threshold)
    p_huge = compute_overtake_probability(pace_delta=2.00, threshold=threshold)

    assert p_neg == 0.0
    assert p_zero == 0.0
    assert 0.0 < p_low < p_mid
    assert p_mid == pytest.approx(0.50)  # Sigmoid at midpoint is 50%
    assert p_mid < p_high < p_huge
    assert p_huge > 0.99


def test_evaluate_overtake_success_and_failure():
    """evaluate_overtake must determine success based on RNG roll and return valid event."""
    lap = 5
    pace_delta = 0.60
    base_thresh = 0.35
    circuit_diff = 0.25

    # Low roll (0.10) should succeed
    success_pass, event_pass = evaluate_overtake(
        lap=lap,
        attacker_id="norris",
        defender_id="verstappen",
        position=1,
        pace_delta=pace_delta,
        circuit_difficulty=circuit_diff,
        base_threshold=base_thresh,
        rng_value=0.10,
    )
    assert success_pass is True
    assert event_pass.success is True
    assert "overtook" in event_pass.description

    # High roll (0.99) should fail
    success_fail, event_fail = evaluate_overtake(
        lap=lap,
        attacker_id="norris",
        defender_id="verstappen",
        position=1,
        pace_delta=pace_delta,
        circuit_difficulty=circuit_diff,
        base_threshold=base_thresh,
        rng_value=0.99,
    )
    assert success_fail is False
    assert event_fail.success is False
    assert "could not pass" in event_fail.description


def test_dirty_air_detection_and_penalty():
    """A car trailing closely behind must experience dirty air and a lap time penalty."""
    monza = load_circuit("monza")
    grid = [
        GridEntry(driver_id="verstappen", team_id="red_bull", starting_position=1, starting_tire="Medium"),
        GridEntry(driver_id="norris", team_id="mclaren", starting_position=2, starting_tire="Medium"),
    ]
    config = RaceConfig(circuit=monza, grid=grid, laps=3, seed=42)
    engine = RaceEngine(config)

    # Initial step
    records_lap1 = engine.step_lap()
    assert len(records_lap1) == 2

    # Leader is never in dirty air
    leader_rec = [r for r in records_lap1 if r.position == 1][0]
    assert leader_rec.in_dirty_air is False

    # Second car started 0.25s behind, which is <= 1.5s dirty air distance
    p2_rec = [r for r in records_lap1 if r.position == 2][0]
    assert p2_rec.in_dirty_air is True


def test_traffic_hold_up_prevents_ghost_pass():
    """A trailing car that is faster but fails to overtake must be held up behind the lead car."""
    monza = load_circuit("monza")
    # Car 1 is very slow, Car 2 is very fast
    grid = [
        GridEntry(driver_id="sargeant", team_id="williams", starting_position=1, starting_tire="Hard"),
        GridEntry(driver_id="verstappen", team_id="red_bull", starting_position=2, starting_tire="Soft"),
    ]
    # High base threshold to guarantee overtake fails
    config = RaceConfig(
        circuit=monza,
        grid=grid,
        laps=1,
        overtake_threshold_seconds=10.0,  # Impossible threshold
        min_following_interval_seconds=0.20,
        seed=1,
    )
    engine = RaceEngine(config)
    records = engine.step_lap()

    # Sargeant must remain P1 because Verstappen could not pass
    assert engine.cars[0].driver.id == "sargeant"
    assert engine.cars[1].driver.id == "verstappen"

    # Verstappen's cumulative time must not be less than Sargeant's
    p1_time = engine.cars[0].cumulative_time
    p2_time = engine.cars[1].cumulative_time
    assert p2_time >= p1_time + 0.199


def test_race_engine_simulation_and_points():
    """RaceEngine.simulate must run full race and calculate points correctly."""
    config = build_race_config("monza", preset_name="2024_default", laps=5, seed=123)
    engine = RaceEngine(config)
    result = engine.simulate()

    assert result.total_laps == 5
    assert len(result.driver_summaries) == 20
    assert result.winner_id is not None
    assert result.fastest_lap_time is not None

    # Winner gets at least 25 points
    winner_summary = result.driver_summaries[0]
    assert winner_summary.finish_position == 1
    assert winner_summary.points in (25, 26)  # 25 + 1 if fastest lap

    # Positions 11-20 should have 0 points
    for summary in result.driver_summaries[10:]:
        assert summary.points == 0


def test_race_engine_seed_determinism():
    """Two simulations with the exact same seed must produce identical race results."""
    config1 = build_race_config("monza", preset_name="2024_default", laps=5, seed=42)
    config2 = build_race_config("monza", preset_name="2024_default", laps=5, seed=42)

    res1 = RaceEngine(config1).simulate()
    res2 = RaceEngine(config2).simulate()

    assert res1.winner_id == res2.winner_id
    assert res1.winner_time == pytest.approx(res2.winner_time)
    assert res1.fastest_lap_time == pytest.approx(res2.fastest_lap_time)

    for s1, s2 in zip(res1.driver_summaries, res2.driver_summaries, strict=True):
        assert s1.driver_id == s2.driver_id
        assert s1.finish_position == s2.finish_position
        assert s1.total_time == pytest.approx(s2.total_time)
