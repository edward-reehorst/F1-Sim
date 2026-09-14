"""Unit tests for Milestone 6: Monte Carlo Batch Engine & Aggregation."""

import pytest

from f1_sim.engine.batch import BatchSimulator
from f1_sim.loaders import build_race_config


def _make_simulator(num_sims=20, master_seed=42, n_workers=1, laps=6, incidents=True):
    config = build_race_config("monza", preset_name="2024_default", laps=laps, seed=master_seed)
    return BatchSimulator(
        config=config,
        num_sims=num_sims,
        master_seed=master_seed,
        enable_incidents=incidents,
        n_workers=n_workers,
    )


def test_batch_returns_expected_shape():
    result = _make_simulator(num_sims=20).run()
    assert result.num_sims == 20
    assert result.circuit_id == "monza"
    assert len(result.driver_stats) == 20


def test_batch_win_rates_sum_to_one():
    result = _make_simulator(num_sims=20).run()
    assert sum(stat.win_rate for stat in result.driver_stats) == pytest.approx(1.0)


def test_batch_pit_strategy_shares_sum_to_one():
    result = _make_simulator(num_sims=20).run()
    assert sum(strat.share for strat in result.pit_strategies) == pytest.approx(1.0)
    shares = [strat.share for strat in result.pit_strategies]
    assert shares == sorted(shares, reverse=True)


def test_batch_is_deterministic_for_same_seed():
    result_a = _make_simulator(master_seed=42).run()
    result_b = _make_simulator(master_seed=42).run()
    assert [stat.wins for stat in result_a.driver_stats] == [stat.wins for stat in result_b.driver_stats]
    assert [stat.avg_points for stat in result_a.driver_stats] == pytest.approx(
        [stat.avg_points for stat in result_b.driver_stats]
    )


def test_batch_seeds_depend_on_master_seed():
    simulator_a = _make_simulator(master_seed=42)
    simulator_b = _make_simulator(master_seed=99)
    assert simulator_a._generate_seeds() != simulator_b._generate_seeds()
    assert simulator_a._generate_seeds() == simulator_a._generate_seeds()


def test_batch_serial_and_multiprocessing_match():
    result_serial = _make_simulator(num_sims=8, n_workers=1).run()
    result_parallel = _make_simulator(num_sims=8, n_workers=2).run()
    assert [stat.wins for stat in result_serial.driver_stats] == [stat.wins for stat in result_parallel.driver_stats]
    assert [strat.strategy_label for strat in result_serial.pit_strategies] == [
        strat.strategy_label for strat in result_parallel.pit_strategies
    ]


def test_batch_report_contains_key_sections():
    result = _make_simulator(num_sims=6).run()
    report = result.format_report()
    assert "# Monte Carlo Batch Report" in report
    assert "## Driver Statistics" in report
    assert "## Optimal Pit Strategies" in report


def test_batch_stats_reference_winner():
    result = _make_simulator(num_sims=20).run()
    assert result.winner is not None
    assert result.winner.wins == max(stat.wins for stat in result.driver_stats)