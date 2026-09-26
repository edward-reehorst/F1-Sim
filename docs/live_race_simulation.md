# Live Race Simulation — Resuming a Real Grand Prix Offline

> **What it is.** `f1-sim` can take a **snapshot of a Grand Prix that is
> actually in progress** — real running order, gaps, tire compounds, and DNF
> state — and resume the simulation from that exact point, forecasting
> **only the remaining laps**. Today's Grand Prix data comes from F1TV's
> live-timing feed over the network; this feature deliberately keeps that network
> access behind **one small, local-file-only seam**, so the whole feature still
> runs in a firewall'd or fully-offline sandbox.
>
> In short: **zero network calls at resume time.** The capture side *may* touch
> the network; the resume side never does.

---

## 1. Two sides, one contract

The feature has two halves that share a single JSON contract:

| Side | Network? | Responsibility |
|------|----------|----------------|
| **Capture** (`scripts/network_capture.py`) | Capture *may* read the F1TV live stream; nothing else does | Record the live timing feed and write a lap-addressable snapshot of the **current** state of a real race |
| **Resume** (`f1-sim race --live <file>`) | **Never** | Load the snapshot from a **local file**, seed every car from its real state, simulate only the remaining laps |

The contract is the `LiveRaceSnapshot` shape (see §4). Both recorders emit
exactly that shape; the loader validates it; the engine consumes it. Nothing
else crosses between capture and resume.

---

## 2. Quickstart

### Capture a live race (networked seam)

F1TV live timing is authenticated, so set that up first — see
**[f1tv_activation.md](f1tv_activation.md)**. Then start the recorder **before
the session begins**:

```bash
uv run python scripts/network_capture.py record \
    --year 2026 --gp italian \
    --circuit monza --out out/snapshots --poll 5
```

One JSON is written per newly-seen lap, e.g. `out/snapshots/monza_lap40.json`.

### Capture a race that has already finished (no token needed)

```bash
uv run python scripts/network_capture.py record \
    --source post-session --year 2026 --gp italian --session R \
    --circuit monza --out out/snapshots
```

This walks the FastF1 static archive one lap at a time, producing the same files.
It is the offline path, and the one used to test the capture code against real
races.

### Resume from the frame (zero network)

```bash
uv run f1-sim race --circuit monza --live out/snapshots/monza_lap40.json
```

That's it. The engine reads `monza_lap40.json` (a local file — no API call),
restores the field exactly as captured, and simulates `total_laps - current_lap`
remaining laps.

---

## 3. The resume chain, end to end

A captured snapshot flows through four real seams, each verified against the
bundled loader/engine in this project:

```
F1TV live timing (network) ─▶ LiveRaceSnapshot ─▶ RaceConfig ─▶ RaceEngine ─▶ RaceResult
   (only in                    (JSON file,        (build_race_config  (seeds every    (only the
    capture)                     local read)        live_snapshot=...)   car)            remaining
                                                                                        laps)
```

Concretely the engine seeding contract (`src/f1_sim/engine/race.py`) does, per
grid entry that appears in the snapshot:

- `current_tire_compound`, `tire_age` ← snapshot
- `fuel_remaining_kg` ← snapshot (falls back to bundled default when the frame
  carries no fuel column)
- `cumulative_time` ← `gap_to_leader_seconds` (the real time behind, not the
  artificial grid stagger)
- `laps_completed` ← snapshot
- `is_dnf` / `dnf_reason` ← snapshot (`running=False` + reason)

Then `engine.simulate()` runs the `total_laps - current_lap` remaining laps.

### Two-compound rule: judged from *real* stint history, not the current fit

The snapshot can also carry each driver's **distinct compounds used so far**
(the ordered list of every slick they have actually run in the live race,
e.g. `["soft", "hard"]`). When present, the engine primes
`CarState.compounds_used` from it before resuming instead of assuming the
current fit is the only stint. The two-compound sporting rule is then
**auto-detected** exactly the way a live race is judged — and the verdict
is corrected on resume, not re-derived from a single current compound:

* `len(car.compounds_used) >= 2` → rule already met in the real stint history
  → **no** penalty, no forced additional stop. A driver who used soft *and*
  hard before lap 40 gets neither the 30s backstop nor a spare pit.
* `len(car.compounds_used) < 2` → rule still owed for the remaining dry laps
  → strategy continues to force the unused slick, and the 30s post-race rule
  check still applies should it stay unmet.

Ground-truth verdict against the live resume chain (Monza resume at lap
40/53, real stint list):

```
VER: compounds_used=['soft','hard']  -> two-compound rule MET  (no penalty, no forced stop)
NOR: compounds_used=['soft']         -> two-compound rule UNMET -> forced second stop owed
```

This closes the resume blind spot: a resumed race no longer judges the rule
from the *current* compound alone, so a driver who already satisfied the
mandatory two-compound rule before the snapshot is not wrongly handed a 30s
penalty or an extra unwanted pit across the remaining laps.

Then `engine.simulate()` runs the `total_laps - current_lap` remaining laps —
and the mandatory-compound verdict is judged from that real stint history for
every car, exactly as in a fresh race.

> **Driver-identity note.** Snapshot rows carry 3-letter FIA codes (`VER`); the
> bundled grid keys cars by slug (`verstappen`). The loader's
> `_derive_grid_from_snapshot` and the engine both translate codes→slugs with
> the same `code_to_driver_id` map, so a snapshot always resolves onto the
> bundled grid. Race control, pit strategy, incidents, and calibration apply
> exactly as in a fresh race.

---

## 4. Snapshot schema (`LiveRaceSnapshot`)

A snapshot is a JSON document with a top-level `drivers` list. The full shape
lives in `src/f1_sim/models/snapshot.py`; here is the per-driver frame:

```json
{
  "driver_code": "VER",
  "position": 1,
  "gap_to_leader_seconds": 0.0,
  "interval_to_ahead_seconds": 0.18,
  "laps_completed": 40,
  "current_tire_compound": "soft",
  "tire_age": 9,
  "fuel_remaining_kg": null,
  "compounds_used_so_far": ["soft", "hard"],
  "running": true,
  "dnf_reason": null
}
```

Top-level:

```json
{
  "circuit_id": "monza",
  "current_lap": 40,
  "total_laps": null,
  "race_flag": "GREEN",
  "origin_note": "captured from real timing data; resume is simulated only over the remaining laps",
  "drivers": [ ... ]
}
```

- Compound names are lowercase, matching the bundled compound keys. The engine
  lowercases again before lookup, so either case loads.
- `race_flag` is one of `GREEN`, `YELLOW`, `VSC`, `SAFETY_CAR` — uppercase, as in
  the `RaceFlag` enum.
- `fuel_remaining_kg` is `null` (unknown) for live captures → engine uses the
  bundled default. The live timing feed does not carry a usable fuel figure.
- `total_laps` is `null` when unknown → engine uses the circuit's default race
  distance.
- `running: false` + `dnf_reason: "retired"` marks a retired car that stays in the
  finishing classification instead of being dropped.

---

## 5. How capture actually works

### The problem

FastF1 cannot read a session while it is running. Its static archive returns
`403` until the session ends, and re-downloading it is pointless when the answer
would be the same data 90 minutes later. Meanwhile the live timing stream returns
`401` without an F1TV token.

### The approach

**Record the live stream, then fold a growing prefix of the recording.** A live
recording is an append-only text file that only ever grows, and FastF1 can
interpret a *partial* recording just as well as a complete one:

```
F1TV live stream ─▶ <circuit>_live.txt (grows during the session)
                          │
                          │  each poll: copy the bytes received so far,
                          │  trimmed back to the last complete message
                          ▼
                   LiveTimingData(prefix) ─▶ session.laps ─▶ fold ─▶ snapshot JSON
```

Re-processing the whole prefix on every poll is O(n²) over a session, but costs
well under a second for a race-length recording — and it means there is no
incremental parser that can drift out of sync with the recording it is replaying.

Only the recorder touches the network. Everything after it reads from disk.

### The two sources

`--source recording` (default) is the live path above and needs an F1TV token;
see [f1tv_activation.md](f1tv_activation.md).

`--source post-session` reads the static archive of a **finished** session, one
lap per poll, and needs no token. It produces byte-identical snapshot files and
is how the capture path is exercised against real races.

### Where the arithmetic lives

`src/f1_sim/tuning/live.py` holds the fold, which is pure and offline: real timing
rows in, a snapshot payload out. `src/f1_sim/tuning/recording.py` holds the
sources behind one `LiveSource` interface (`poll() -> snapshot | None`). Keeping
the arithmetic apart from the fetching is what lets a live recording and a
finished session share a single code path.

The fold carries a few rules that are not obvious, each of which was a real bug
against real timing data:

- A retiring driver's final lap carries `Position = NaN`, so positions are
  recovered from the last position each driver actually held.
- A stopped car keeps its last on-track time, which is *older* than the reference
  used for it. Gaps compare a car against the leader's time **for the same lap
  number**, and are floored at zero.
- A retiree's last on-track position collides with whoever has since taken that
  slot, so retirees are ranked behind the running field. Positions are always
  unique and contiguous.
- Intervals chain only through cars still running — chaining through a stopped
  car's stale time hands the car behind it a negative interval.
- `TrackStatus` concatenates *every* status seen during a lap (`"2451"`), so
  green means "contains `1`" and the flag in effect at the capture instant is the
  **last element** (a named form like `VSC1` stands down, so it reads as green).
  A red flag maps to `SAFETY_CAR`, the nearest thing the `RaceFlag` enum can
  express, because that is what it means to the engine.
- A status that is **missing** is treated as green. FastF1 leaves `TrackStatus`
  `NaN` or empty until the topic delivers, and reading that as "not green"
  discards every lap of the session — a capture that produces nothing and reports
  no error.
- **Retirement is inferred from lap count**, not read from the feed: a car more
  than one lap behind the leader is recorded as `running: false`. The one lap of
  tolerance absorbs a normal pit stop, but during a *long* safety car a car that
  pits can be two laps down while perfectly healthy and will be recorded as a DNF.
  FastF1's parsed laps carry no per-driver status, so closing this gap means
  reading the raw `TimingData` `Status` field that its parser discards.

---

## 6. CLI surface

### Capture

```
scripts/network_capture.py record
    --circuit CIRCUIT        # e.g. monza
    --out OUT                # snapshot output dir
    --source recording       # live F1TV stream (default; needs auth)
                             # post-session = a finished race, no token needed
    --year YEAR              # required for both sources; resolves the event
    --gp GP                  # required for both sources, e.g. italian
    --session {Q,R}          # session identifier (default R)
    --poll POLL              # seconds between polls (default 5)
    --max-laps N             # stop once this lap is captured
    --at-lap N               # capture only this lap
    --recording PATH         # live recording file (default <out>/<circuit>_live.txt)
    --recorder-timeout SECS  # give up if no message arrives for this long (default 60)
    --startup-wait SECS      # wait this long for the first message (default 30)
```

### Resume

```
f1-sim race --circuit CIRCUIT --live <snapshot.json>
```

The `race` command also supports the dashboard/replay surface introduced
alongside live resume:

```
--replay/--no-replay      # Rich terminal live dashboard while simulating
--live/-L <snapshot>      # resume a race from a local snapshot file
--watch/--no-watch        # auto-refresh the snapshot file each lap while resuming
--calibration <json>      # lay a calibration overlay onto the bundled preset
--incidents/--no-incidents
--seed, --laps, --speed, --plot, --plot-dir
```

---

## 7. Offline / firewall note

- **Resume is always local-file-only.** The loader `load_snapshot` performs
  zero network calls no matter what; it only reads the file you point it at.
- **Capture is opt-in network.** `network_capture.py --source recording` is the
  one path that opens a socket, and only while recording. Everything after the
  recorder reads from disk.
- In a fully offline sandbox, use `--source post-session` against a
  pre-downloaded archive, or feed a recording you already have on disk; the same
  resume command works unchanged.

---

## 8. Example: Monza, captured at lap 40

```python
from f1_sim.loaders import load_snapshot, build_race_config
from f1_sim.engine import RaceEngine

snapshot = load_snapshot("out/snapshots/monza_lap40.json")   # zero network
config = build_race_config("monza", "2024_default",
                           laps=None, seed=42, live_snapshot=snapshot)
engine = RaceEngine(config, enable_incidents=True)

result = engine.simulate()
assert result.total_laps == 13   # only the remaining laps
print(result.winner_id, result.winner_time)
```

> **Naming note.** With a live snapshot, `config.laps` — and therefore
> `engine.total_laps` and `result.total_laps` — is the number of laps **left to
> simulate**, not the race distance. `build_race_config` computes it as
> `circuit.total_laps - snapshot.current_lap` (53 − 40 = 13 here). `engine.current_lap`
> counts the laps being simulated and starts at 0, so it is not the real lap
> number of the snapshot. To get the real race distance, use
> `config.circuit.total_laps`.

Verified against the 2026 Italian Grand Prix at Monza, captured at lap 40 from
the FastF1 archive with the fold in this repo: `total_laps == 13`, with the
running order, gaps, tire compounds, stint history, and DNFs seeded from the
snapshot rather than from the bundled grid.
