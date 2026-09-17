# f1-sim Usage Guide

This guide covers the CLI and Python API. The engine ships with bundled 2024
data (10 teams, 20 drivers, circuits Monza / Silverstone / Spa / Monaco) and can
be calibrated against real session data fetched from the F1 timing API.

## Install

```bash
uv sync
```

All commands below use `uv run f1-sim <command>`. `f1-sim --help` lists every
command; `<command> --help` lists its options.

## Core commands

```bash
# Package info and bundled dataset status
uv run f1-sim info
uv run f1-sim circuits

# Single-car, clean-air time trial with lap-by-lap telemetry
uv run f1-sim time-trial --circuit monza --driver verstappen --team red_bull --tire Soft --laps 10

# Full Grand Prix with pit stops, strategy, and race control
uv run f1-sim race --circuit monza --preset 2024_default

# Pure strategic race (no Safety Cars / DNFs)
uv run f1-sim race --circuit silverstone --no-incidents

# Live Rich terminal dashboard (playback speed in seconds per lap)
uv run f1-sim race --circuit monza --live --speed 0.2

# Save post-race charts (lap chart, gaps, stints, pace decay)
uv run f1-sim race --circuit monza --plot --plot-dir output

# Deterministic seed for reproducible races
uv run f1-sim race --circuit monza --seed 7

# Monte Carlo batch: win rates, podiums, optimal pit strategies
uv run f1-sim batch --circuit silverstone --sims 1000 --out-dir output
```

## Calibration workflow

Calibrate the bundled pace model against **real session data** from anywhere in
1996–present seasons, then race or batch with the fitted overlay.

The workflow is three steps:

```bash
# 1. Fetch and export a session's green-flag laps
uv run f1-sim ingest --year 2026 --gp Italy --session FP2 --circuit monza \
  --out data/monza_fp2.csv

# 2. Fit driver pace offsets, compound corrections, and grid against that data
uv run f1-sim tune --circuit monza --dataset data/monza_fp2.csv \
  --out calibration/monza.json --anchor-driver leclerc

# 3. Race or batch using the fitted overlay
uv run f1-sim race --circuit monza --calibration calibration/monza.json
uv run f1-sim batch --circuit monza --sims 2000 --calibration calibration/monza.json
```

### `ingest` — fetching real data

```bash
uv run f1-sim ingest \
  --year 2024 \
  --gp monza \
  --session Q \            # FP1, FP2, FP3, Q, or R
  --circuit monza \        # optional; auto-detected when the GP maps to a bundled circuit
  --pace-filter 1.5 \      # keep laps within N s of each driver's per-compound best (practice)
  --format csv \           # csv or json
  --out data/monza_q.csv
```

Notes:

- Only green-flag laps (track status 1) are exported; incomplete laps are dropped.
- **Teams outside the bundled grid are dropped** (e.g. 2026 Cadillac, substitute
  drivers). Race/practice ingests print how many were dropped.
- `--pace-filter` isolates representative pace laps for practice sessions where
  most laps are traffic/fuel-heavy; without it the calibrator tends to fit the
  slow laps.
- The export includes per-lap qualifying **positions** (used by `tune` to build
  a starting grid) and a `driver_code` column so grid penalties can reference F1
  codes.

### `tune` — fitting the calibration overlay

```bash
uv run f1-sim tune \
  --circuit monza \
  --dataset data/monza_q.csv \
  --anchor-driver leclerc \   # driver pinned to 0.0s offset (default: most-observed)
  --penalty GAS:0 \           # repeatable; CODE:PLACES grid drops (0 = back of grid)
  --penalty VER:3 \
  --holdout 0.2 \
  --seed 7 \
  --out calibration/monza.json
```

What the fitted overlay contains:

| Field | Meaning |
| --- | --- |
| `driver_pace_offsets` | Per-driver lap-time offset (seconds) vs. the anchor |
| `circuit_base_adjust` | Track faster/slower than the bundled baseline |
| `compound_base_deltas` | Per-compound tire grip corrections |
| `compound_wear_multipliers` | Per-compound tire wear rate corrections |
| `compound_cliff_adjustments` | Per-compound cliff-lap shifts |
| `starting_grid` | Real qualifying grid that replaces the preset grid |
| `fuel_penalty_per_kg` | Fitted fuel penalty, when telemetry carried fuel |

Behavior:

- **Teammate fallback** — a driver missing from the calibration data inherits
  the mean offset of their teammates **on the real grid** (so a driver who moved
  teams, e.g. Hamilton at Ferrari, inherits Ferrari pace, not the bundled team).
- The fit is deterministic for a given dataset/seed. `--holdout` reserves a
  fraction of laps for out-of-sample RMSE reporting.
- Most-observed driver is the default anchor; set `--anchor-driver` explicitly
  (that driver gets 0.0s).

### Starting grid: replay vs. race day

`tune` builds `starting_grid` from the ingests dataset **only when every
observation is from a qualifying session**.

- **After the race (replay):** `ingest --session Q` prefers the *official grid*
  (`GridPosition` from the race results, penalties applied), e.g. Piastri
  starts P6 despite qualifying P3.
- **Before the race (preview):** the official grid is not published yet, so
  `ingest` falls back to the raw qualifying classification. Add any known
  penalties manually with `--penalty`:

  ```bash
  # Verstappen +3, Gasly sent to the back of the grid
  uv run f1-sim tune ... --penalty VER:3 --penalty GAS:0
  ```

Penalty rules followed (fastest qualifier first):

- Multiple `--penalty` for one driver **stack** (5 + 3 = 8 places).
- `CODE:0` drops the driver to the back of the grid.
- Any drop past the grid size clamps to the back row.
- Drivers shedding penalties move up to fill the gaps (a dropped pole-sitter
  moves the whole field up one).

### `overtake-stats` — empirical overtake thresholds

`overtake_threshold_seconds` is measured from real races rather than chosen by
hand: the sim compares a defender's and attacker's lap time on a pass lap, so
the threshold is the observed pace gap at which real on-track overtakes happen.

```bash
# Reconstruct real overtakes (2018-2026) and report per-circuit pace-gap medians
uv run f1-sim overtake-stats --start 2018 --end 2026 \
  --out src/f1_sim/data/overtake_stats.json
```

A pass only counts when adjacent cars swap positions on a fully-green lap with no
pit involvement, and the attacker was faster than the defender on the pass lap.
The JSON artifact (n races, n overtakes, Q1/median/Q3 in seconds) is the source
of the baked `overtake_threshold_seconds` values in the bundled circuits
(`src/f1_sim/data/circuits/*.json`).

### `overtake-calibrate` — frequency-match the thresholds

The rounded empirical median is a good starting point, but overtake count also
depends on how often the engine generates a given pace gap. `overtake-calibrate`
runs the simulation across several seeds, binary-searches each circuit's
`overtake_threshold_seconds` so the simulated **number of passes per race**
matches the real rate from `overtake_stats.json`, then bakes the result.

```bash
# Defaults: preset 2024_default, 5 seeds, tolerance 1.0 pass/race, max 5.0s
uv run f1-sim overtake-calibrate \
  --stats-path src/f1_sim/data/overtake_stats.json \
  --seeds 5 --tolerance 1.0 \
  --out src/f1_sim/data/overtake_calibration.json --apply
```

Non-converged rows mean the target rate is outside what any threshold between
`--lo` and `--max-threshold` can produce — typically the hardest track
(Monaco) where a handful of lapping/give-way artifacts persist. Passes with a
pace gap above `--max-clean-delta` (default 8 s) and passes on either car's
pit-stop lap are excluded from the count the same way as the real analysis.

## Python API

```python
from f1_sim import build_race_config, RaceEngine, run_time_trial

# Full Grand Prix
config = build_race_config("monza", preset_name="2024_default", laps=53, seed=42)
engine = RaceEngine(config)
result = engine.simulate()

for summary in result.driver_summaries[:10]:
    print(f"P{summary.finish_position}: {summary.driver_code} "
          f"(+{summary.gap_to_winner:.3f}s, {len(summary.pit_stops)} stops, {summary.points} pts)")
```

```python
from f1_sim import calibrate_from_dataset, load_dataset, write_observations_csv, fetch_session

# Fetch and export (equivalent of the ingest CLI)
dataset = fetch_session(year=2026, gp="Italy", session_identifier="Q", circuit_id="monza")
dataset.observations[:] = dataset.filter_for_pace(buffer_seconds=1.5)
write_observations_csv(dataset, "data/monza_q.csv")

# Fit (equivalent of tune, with penalties by F1 driver code)
loaded = load_dataset("data/monza_q.csv", circuit_id="monza")
report = calibrate_from_dataset(
    loaded,
    "monza",
    anchor_driver_id="leclerc",
    grid_penalties=[GridPenalty(driver_id="VER", places=3)],
)
print(report.in_sample_rmse, report.config.driver_pace_offsets)
report.config.write_json("calibration/monza.json")

# Race with the fitted overlay
config = build_race_config("monza", calibration=report.config)
```

## Tests

```bash
uv run pytest
```