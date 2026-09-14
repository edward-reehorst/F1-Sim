# f1-sim

A modular, high-performance Formula 1 race simulation engine in Python.

## Features
- **Deterministic Lap-by-Lap Physics**: Predictable and consistent clean-air lap times driven by car performance, driver ratings, fuel burn, and non-linear tire wear curves.
- **Traffic & Overtaking Dynamics**: Dynamic interval calculation, dirty air drag & tire wear penalties (<1.5s), and pure pace-offset overtaking probability models.
- **Realistic Pit Strategy**: Rule-based race engineer strategies, target pit windows, tire cliff detection, Safety Car opportunism ("cheap pit stops"), and mandatory two-compound compliance.
- **Race Control & Incidents**: Stochastic mechanical failures, collision DNFs, Virtual Safety Cars (VSC), full Safety Cars (SC) with field compression, and flag neutralizations.
- **Modern F1 Presets**: Bundled 2024 grid data for 10 teams, 20 drivers, and iconic circuits (Monza, Silverstone, Spa, Monaco).
- **Blazing Fast Performance**: Full 50+ lap Grand Prix with 20 cars simulated in **< 7 ms**.
- **Rich Terminal CLI**: Interactive spectatorship, single-car time trials, and full race simulations.

## Quickstart

### CLI Commands
```bash
# View package info and loaded circuits
uv run f1-sim info
uv run f1-sim circuits

# Single-car clean-air time trial with lap-by-lap telemetry breakdown
uv run f1-sim time-trial --circuit monza --driver verstappen --team red_bull --tire Soft --laps 10

# Full multi-car Grand Prix simulation with pit stops, strategy, and race control
uv run f1-sim race --circuit monza --preset 2024_default

# Disable incidents for a pure strategic race
uv run f1-sim race --circuit silverstone --no-incidents

# Live Rich terminal dashboard with playback speed control
uv run f1-sim race --circuit monza --live --speed 0.2

# Save post-race Matplotlib charts (lap chart, gap, stints, pace decay)
uv run f1-sim race --circuit monza --plot --plot-dir output

# Monte Carlo batch: win rates, podiums, and optimal pit strategies (multiprocessed)
uv run f1-sim batch --circuit silverstone --sims 1000 --out-dir output
```

### Python API
```python
from f1_sim import build_race_config, RaceEngine

# Configure and run a full Grand Prix
config = build_race_config("monza", preset_name="2024_default", laps=53, seed=42)
engine = RaceEngine(config)
result = engine.simulate()

print(f"Winner: {result.winner_id} in {result.winner_time:.2f}s")
print(f"Fastest Lap: {result.fastest_lap_driver_id} ({result.fastest_lap_time:.3f}s)")

for summary in result.driver_summaries[:10]:
    stops = len(summary.pit_stops)
    print(f"P{summary.finish_position}: {summary.driver_code} (+{summary.gap_to_winner:.3f}s, {stops} stops, {summary.points} pts)")
```

## Running Tests
```bash
uv run pytest
```
