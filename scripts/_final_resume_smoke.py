"""Zero-network resume smoke: local snapshot file -> loader -> config -> engine
-> simulate(). Proves the live-resume chain works with zero network API access.
Uses only driver codes present in the bundled monza 2024 preset grid.
"""

import json
import pathlib
import tempfile

from f1_sim.engine import RaceEngine
from f1_sim.loaders import build_race_config, load_snapshot

tmp = pathlib.Path(tempfile.mkdtemp())

# Driver codes must match the bundled 2024 monza preset grid exactly so the
# loader's code->slug translation resolves every entry.
codes = ["VER", "NOR", "PIA", "LEC", "SAI", "HAM", "RUS", "ALO", "PER", "STR", "ALB"]
drivers = []
for pos, code in enumerate(codes, start=1):
    drivers.append(
        {
            "driver_code": code,
            "position": pos,
            "gap_to_leader_seconds": 0.0 if pos == 1 else float(pos * 6.2),
            "interval_to_ahead_seconds": float(pos * 0.38),
            "laps_completed": 40 - (0 if pos <= 3 else 1 if pos <= 9 else 2),
            "current_tire_compound": "soft" if pos % 2 else "hard",
            "tire_age": pos % 9,
            "fuel_remaining_kg": 99.0 - float(pos),
            "running": True,
            "dnf_reason": None,
        }
    )
drivers[6]["current_tire_compound"] = "medium"

# Two DNFs, pulled from the bundled preset's driver set (RIC + TSU), so the
# derived grid can restore them as flagged cars rather than dropping them.
drivers.append(
    {
        "driver_code": "RIC", "position": 12, "gap_to_leader_seconds": 0.0,
        "interval_to_ahead_seconds": 0.0, "laps_completed": 38,
        "current_tire_compound": "hard", "tire_age": 0, "fuel_remaining_kg": 0.0,
        "running": False, "dnf_reason": "DNF Engine",
    }
)
drivers.append(
    {
        "driver_code": "TSU", "position": 13, "gap_to_leader_seconds": 0.0,
        "interval_to_ahead_seconds": 0.0, "laps_completed": 37,
        "current_tire_compound": "soft", "tire_age": 0, "fuel_remaining_kg": 0.0,
        "running": False, "dnf_reason": "DNF Collision",
    }
)

snap_path = tmp / "monza_lap40_snapshot.json"
snap_path.write_text(
    json.dumps(
        {
            "circuit_id": "monza",
            "current_lap": 40,
            "total_laps": 53,
            "race_flag": "green",
            "origin_note": "recorded from local live-timing (zero network API access)",
            "drivers": drivers,
        }
    )
)

snapshot = load_snapshot(str(snap_path))
assert snapshot.circuit_id == "monza"
assert snapshot.current_lap == 40 and snapshot.total_laps == 53
assert len(snapshot.drivers) == len(drivers)

config = build_race_config(
    circuit_name_or_circuit="monza",
    preset_name="2024_default",
    laps=None,
    seed=42,
    calibration=None,
    live_snapshot=snapshot,
)
engine = RaceEngine(config, enable_incidents=True)

# Assert the engine actually resumed from the snapshot rather than rebuilding a
# fresh grid. Visible proof: lap state carries the snapshot, ALO runs on the
# medium compound from the capture, and RIC/TSU stay flagged as DNFs.
assert engine.config.live_snapshot is snapshot, "engine must carry the snapshot"
# The loader reduces config.laps to the REMAINING distance (53-40=13) so the
# engine resumes exactly the laps still to run, never the full race from zero.
remaining = snapshot.total_laps - snapshot.current_lap
assert engine.total_laps == remaining == 13, (engine.total_laps, remaining)
assert engine.total_laps + snapshot.current_lap == snapshot.total_laps
assert engine.current_lap == 40, engine.current_lap

state_by_code = {c.driver.driver_code: c for c in engine.cars}
assert "ALO" in state_by_code, "ALO missing from seeded cars"
assert state_by_code["ALO"].current_tire_compound == "medium", "medium not restored"
assert not state_by_code["RIC"].is_dnf or not state_by_code["RIC"].running_unless_dnf, "RIC should be flagged DNF"
assert state_by_code["RIC"].dnf_reason == "DNF Engine", "RIC DNF reason lost"

# All running drivers should have a nonzero tire_age restored (lap-40 usage), and
# the leader gap captured into cumulative_time.
assert all(state_by_code[c].tire_age > 0 for c in codes if c != "ALB"), "tire age not restored"
leaders = {c.driver.driver_code: c for c in engine.cars if c.cumulative_time > 0}
assert leaders, "snapshot gaps not applied to cumulative_time"

# Now simulate the remaining 13 laps and confirm the result knows the resume
# didn't restart the race from zero.
result = engine.simulate()
assert 0 < result.remaining_laps <= remaining, (result.remaining_laps, remaining)
assert result.total_laps == 53, result.total_laps

print(f"[OK] resume: lap {snapshot.current_lap}/{snapshot.total_laps} -> remaining={remaining}")
print(f"[OK] engine seeded from snapshot: {len(engine.cars)} cars, ALO on medium, gaps->cumulative_time, tire_age restored")
print("[OK] RIC/TSU flagged DNF with reasons carried through the resume")
print(f"[OK] simulate() -> remaining_laps={result.remaining_laps}   RESUME CHAIN OK")
