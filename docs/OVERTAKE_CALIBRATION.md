# Overtake threshold calibration

`overtake_threshold_seconds` controls how big a per-lap pace gap must be before
the engine treats a pass as likely: on a clean lap the probability of a
successful overtake is the logistic

```
P(pass) = 1 / (1 + exp(-8 * (pace_gap - threshold)))
```

so a smaller threshold means passes happen more easily. The bundled per
circuit value comes from a two-stage pipeline measured from **real races**:

1. **`overtake-stats`** — reconstruct real on-track overtakes from FastF1 and
   measure the observed pace-gap distribution per circuit (seconds).
2. **`overtake-calibrate`** — frequency-match: binary-search each circuit's
   threshold so the simulated **passes per race** equals the real rate.

This doc describes both stages and how thresholds get baked into
`src/f1_sim/data/circuits/*.json`.

## Stage 1 — `overtake-stats`: measure the real pace gap

```bash
uv run f1-sim overtake-stats --start 2018 --end 2026 \
  --out src/f1_sim/data/overtake_stats.json
```

For each circuit the tool reconstructs real passing from FastF1 lap-by-lap
timing and reports, per circuit: number of races, number of on-track
overtakes, and the Q1 / median / Q3 of the **pace gap** (defender lap time minus
attacker lap time on the pass lap, seconds).

### What counts as an overtake

A real overtake is only counted when:

- the two cars are adjacent in the race order;
- the positions swap on a fully-green lap (no safety car / VSC);
- **neither** car is on a pit-stop lap; and
- the attacker was faster than the defender on the pass lap.

The pace gap used is the smoothed (median-of-three) lap delta, so a single
fast/stuck lap does not dominate. The empirical median per circuit (in seconds)
is the initial `overtake_threshold_seconds` before frequency-matching.

## Stage 2 — `overtake-calibrate`: match frequency

A threshold chosen purely from the empirical gap distribution does not
guarantee the sim produces the same **number** of passes: how often the engine
generates a given pace gap matters too. `overtake-calibrate` closes the loop:

```bash
uv run f1-sim overtake-calibrate \
  --stats-path src/f1_sim/data/overtake_stats.json \
  --seeds 5 --tolerance 1.0 \
  --out src/f1_sim/data/overtake_calibration.json
```

For each circuit it computes the real target rate

```
target passes / race = n_overtakes / n_races
```

then binary-searches `overtake_threshold_seconds` in
`[--lo, --max-threshold]` (defaults 0.0 s .. 5.0 s). For each candidate
threshold it runs the full race over `--seeds` seeds (default 5), applying
`--incidents` by default, and averages the successful overtake count. The
search continues until the simulated passes/race is within `--tolerance`
(default 1.0 pass) of the target, or the bound is exhausted.

### What the sim count filters out

The simulated overtake count must match what the real analysis counts, so the
same exclusions apply:

- passes on a **pit-stop lap** for either the attacker or defender are ignored;
- passes whose pace gap exceeds `--max-clean-delta` (default 8 s) are ignored —
  such big gaps are the signature of a lapping/hold-up or give-way lap, not a
  clean fighting pass.

Without these filters lapped-traffic artifacts inflate the count at the hardest
tracks (Monaco in particular), where real passes are few. With them the
frequency match converges for all four bundled circuits.

### `--apply`: bake into the bundled circuits

```bash
uv run f1-sim overtake-calibrate \
  --stats-path src/f1_sim/data/overtake_stats.json \
  --seeds 5 --tolerance 1.0 --apply
```

`--apply` writes the matched threshold into each circuit's JSON
(`overtake_threshold_seconds`) and additionally scales the wet preset:
`spa_wet.json` keeps its historical wet/dry ratio relative to the freshly
matched dry `spa` threshold, so a weather preset does not silently calibrate to
a stale ratio. The ratio and matched thresholds are saved to
`src/f1_sim/data/overtake_calibration.json`.

## Current baked values (2018-2026 stats)

Real targets from `overtake_stats.json` and the matched sim result:

| Circuit | races | real passes | target / race | threshold (s) | sim / race (5 seeds) | converged |
|---------|-------|-------------|---------------|---------------|----------------------|-----------|
| monaco  | 7     | 15          | 2.14          | **3.5 (manual)** | ~0       | n/a      |
| baku    | 7     | 97          | 13.86         | **0.7422**      | 13.40                | yes      |
| cota    | 7     | 140         | 20.00         | **0.7190**      | 20.00                | yes       |
| singapore | 6   | 79          | 13.17         | **0.8594**      | 13.20                | yes       |
| interlagos | 9  | 147         | 16.33         | **0.7129**      | 16.40                | yes       |
| vegas   | 4     | 64          | 16.00         | **0.6445**      | 16.00                | yes       |
| monza   | 8     | 164         | 20.50         | 0.7031        | 20.00                | yes       |
| silverstone | 9  | 116         | 12.89        | 0.8008        | 12.20                | yes       |
| spa     | 8     | 145         | 18.12         | 0.7812        | 18.40                | yes       |
| spa_wet | —     | —           | (wet scale)   | 0.9548        | —                    | yes       |

Re-run the match and `--apply` whenever the real dataset changes (new seasons,
filter tweaks). The empirical standards for the excluded artifacts are stable;
Monaco's low pass count is a real feature, not a bug — the reference data
(7 races, 15 passes) is simply small there.

## Eyeballing the search

Each circuit's search logs the evaluated thresholds and per-seed pass counts.
A row with `converged=no` means the target passes/race sits outside what any
threshold in `[--lo, --max-threshold]` can produce for that circuit/preset; the
closest threshold is still returned. With `--apply` on it leaves the artifact
with the equivalent safest value.
