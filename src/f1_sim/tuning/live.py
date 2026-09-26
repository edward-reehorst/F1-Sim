"""Fold real timing rows into the local live-resume snapshot shape.

Pure and offline. Every function here takes rows that were *already* fetched and
returns the exact payload ``load_snapshot`` / ``f1-sim race --live`` read back, so
the fold is testable without a session, a network, or a FastF1 import. Keeping the
arithmetic separate from the fetching is what lets a live recording and a finished
session share one code path.

The interesting part is that real timing frames are messy in specific ways, and
each of those has a failure mode worth naming:

- ``TrackStatus`` is the *concatenation* of every status active during a lap
  (``"1245"``, ``"126"``, ``"2451"``), not one code. Green means ``"1"`` appears
  anywhere in it; anchoring with ``startswith`` silently drops laps that ended
  green after a brief yellow. A status that is *absent* is treated as green too:
  FastF1 leaves the column ``NaN`` or ``''`` until the topic delivers, and
  reading that as "not green" discards every lap of the session.
- A retiring driver's final lap carries ``Position = NaN``, so a straight
  ``int(row.Position)`` raises.
- A stopped car keeps its last on-track ``Time``, so any gap or interval derived
  from it goes negative — and the snapshot schema requires ``ge=0.0``.
- A retiree's last on-track position collides with the running car that has since
  taken that slot, producing two cars at the same position.
- The leader may not have completed the lap a trailing car just finished (they
  pitted), so "the leader's time for lap N" does not always exist yet.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

# Columns the fold needs from a FastF1-shaped laps frame.
_REQUIRED_COLUMNS = ("Driver", "DriverNumber", "LapNumber", "Time", "Stint")

# One track-status element: a run of digits (a code, possibly multi-digit) or a run
# of letters (a named form such as "VSC").
_ELEMENT_RE = re.compile(r"\d+|[A-Z]+")

_ORIGIN_NOTE = "captured from real timing data; resume is simulated only over the remaining laps"


def _normalise_compound(value: Any) -> str:
    """Compound code in the bundled lowercase form (``soft``/``medium``/...)."""
    return str(value).strip().lower()


# FastF1 track status codes, per `fastf1._api.track_status_data`:
#   1 track clear, 2 yellow, 3 unused, 4 safety car, 5 red flag,
#   6 virtual safety car, 7 VSC ending
#
# A red flag has no `RaceFlag` equivalent. It maps to SAFETY_CAR because that is
# what it is to the engine -- a full-course neutralisation the field must hold
# position through -- and `Strategy.should_pit` only reacts to SC/VSC. The mapping
# is deliberately lossy, and the only observable consequence is that a red-flag
# resume holds station instead of treating the lap as a normal yellow.
_STATUS_FLAGS = {
    "1": "GREEN",
    "2": "YELLOW",
    "3": "GREEN",
    "4": "SAFETY_CAR",
    "5": "SAFETY_CAR",
    "6": "VSC",
    "7": "VSC",
}


# Named forms some feeds use in place of numeric codes. Kept separate from
# _STATUS_FLAGS because they are matched as whole tokens, not character by
# character -- "VSC" contains "SC", so a substring check would order them wrongly.
_NAMED_FLAGS = {
    "VSC": "VSC",
    "SC": "SAFETY_CAR",
}


def track_flag(status: Any) -> str:
    """The flag in effect at the *end* of a lap, as a snapshot race flag.

    A lap's ``TrackStatus`` concatenates every status that was live during it, so
    ``"2451"`` is a lap that ran yellow, then safety car, then red, and ended
    clear. The status at the end is the last code, which is the one describing the
    field at the instant being captured -- earlier codes describe a lap that is
    already done.

    The last *element* decides, whether it is a code or a name: ``"VSC1"`` is a
    virtual safety car that was stood down before the lap ended, so it is GREEN.
    """
    if pd.isna(status):
        return "GREEN"
    text = str(status).upper()
    tokens = _ELEMENT_RE.findall(text)
    if not tokens:
        return "GREEN"
    last = tokens[-1]
    if last in _NAMED_FLAGS:
        return _NAMED_FLAGS[last]
    return _STATUS_FLAGS.get(last[-1], "GREEN")


def is_green_lap(status: Any) -> bool:
    """Whether a green flag was active at any point during the lap.

    An *unknown* status counts as green. The filter exists to drop laps that a red
    flag interrupted, and a status we do not have is not evidence of one. FastF1
    initialises ``TrackStatus`` to ``''`` and leaves it ``NaN`` when the topic has
    not delivered, so treating unknown as non-green silently discarded every lap
    of a whole session -- a capture that produced nothing and reported no error.
    """
    if pd.isna(status):
        return True
    text = str(status).strip()
    if not text:
        return True
    return "1" in text


def _validate(frame: pd.DataFrame) -> None:
    missing = [c for c in _REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"laps frame is missing required column(s) {missing}; "
            f"present columns: {list(frame.columns)}"
        )


def _completed_laps(laps: Any, at_lap: int | None = None) -> pd.DataFrame | None:
    """Laps completed by ``at_lap``, sorted by time, before the green-flag filter.

    Returns ``None`` when the frame holds no usable lap yet.
    """
    if laps is None or len(laps) == 0:
        return None
    frame = laps.copy()
    _validate(frame)
    frame = frame[frame["LapNumber"].notna()]
    if at_lap is not None:
        frame = frame[frame["LapNumber"] <= at_lap]
    if frame.empty:
        return None
    return frame.sort_values("Time")


def _green_laps(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where a green flag was in effect."""
    if "TrackStatus" in frame.columns:
        frame = frame[frame["TrackStatus"].map(is_green_lap)]
    return frame


def _latest_per_driver(frame: pd.DataFrame) -> pd.DataFrame:
    """Each driver's most recent completed lap at the poll instant."""
    return frame.groupby("Driver", sort=False).tail(1)


def _last_known_positions(frame: pd.DataFrame) -> dict[str, int]:
    """Driver code -> last position it actually held with a non-null Position."""
    known: dict[str, int] = {}
    for code, grp in frame.dropna(subset=["Position"]).groupby("Driver"):
        known[str(code)] = int(grp.sort_values("Time")["Position"].iloc[-1])
    return known


def _is_running(latest: pd.DataFrame) -> pd.Series:
    """A car still circulating is on the lead lap; a retired car falls behind.

    One lap of tolerance: a car that pits on the leader's final lap is a lap down
    for a poll or two while still running, and marking it a DNF would be worse
    than letting it recover on the next poll.
    """
    lead_lap = int(latest["LapNumber"].max())
    return latest["LapNumber"] >= lead_lap - 1


def _assign_positions(latest: pd.DataFrame, known: dict[str, int]) -> pd.Series:
    """Contiguous, unique 1-indexed positions for the whole field.

    Running cars keep their real running order. Everyone else -- retirees, plus any
    car whose position never resolved -- is ranked behind them by how far they got,
    so a stale on-track position can never collide with a live one.
    """
    running = _is_running(latest)

    def rank(row: pd.Series) -> tuple[int, float, str]:
        if running[row.name] and str(row["Driver"]) in known:
            return (0, known[str(row["Driver"])], str(row["Driver"]))
        # Unranked runners and retirees alike trail the field, furthest along first.
        return (1, -float(row["LapNumber"]), str(row["Driver"]))

    keys = latest.apply(rank, axis=1)
    ordered = latest.loc[keys.sort_values().index]
    return pd.Series(range(1, len(ordered) + 1), index=ordered.index, name="_pos")


class LeaderReference:
    """The leader's cumulative set-time per lap, for gap arithmetic.

    Gaps compare a car against the leader's time *for the same lap number*: a
    trailing car sets that lap later, so ``car - leader`` is positive on the lead
    lap and still positive for a car running laps down (it measures the deficit
    over the distance actually covered). Comparing against the leader's *current*
    time instead goes negative for anyone who stopped, because their last on-track
    time is old.

    When the leader has not yet completed a lap a trailing car just finished --
    they pitted -- the most recent leader lap at or before it is used, so the gap
    is a slight over-estimate rather than a negative number. That approximation is
    deliberate and bounded; the leader simply not existing yet is not, and raises.
    """

    def __init__(self, leader_code: str, times_by_lap: dict[int, Any]) -> None:
        self.leader_code = leader_code
        self._times = times_by_lap
        self._laps = sorted(times_by_lap)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, leader_code: str) -> LeaderReference:
        rows = frame[frame["Driver"] == leader_code]
        times = {
            int(lap): value for lap, value in zip(rows["LapNumber"], rows["Time"], strict=True)
        }
        return cls(leader_code, times)

    @property
    def has_laps(self) -> bool:
        """Whether the leader has any recorded lap to measure against at all.

        Named for what it answers. It does not mean every gap taken from this
        reference is exact -- some are approximations from ``at()`` -- which is why
        this is a count of laps rather than a claim about precision.
        """
        return bool(self._laps)

    def at(self, lap: int) -> Any:
        """The leader's time for ``lap``, falling back to the latest lap before it."""
        if lap in self._times:
            return self._times[lap]
        earlier = [seen for seen in self._laps if seen <= lap]
        if not earlier:
            raise KeyError(
                f"leader {self.leader_code!r} has no lap at or before {lap} "
                f"(leader laps: {self._laps[:5]}...); cannot compute a gap"
            )
        return self._times[max(earlier)]

    def gap_seconds(self, lap: int, car_time: Any) -> float:
        return max(0.0, (car_time - self.at(lap)).total_seconds())


def _stint_compounds(frame: pd.DataFrame, code: str) -> list[str]:
    """Distinct compounds this driver has run, ordered by first use.

    One timing row per stint means the distinct compounds across a driver's laps
    are their real stint history, which is what ``compounds_used_so_far`` needs so
    the two-compound rule is judged from what they actually ran.
    """
    prior = frame[frame["Driver"] == code]
    if prior.empty:
        return []
    order = prior.sort_values("Time").groupby("Stint")["Compound"].first()
    seen: list[str] = []
    for value in order.dropna():
        name = _normalise_compound(value)
        if name and name not in seen:
            seen.append(name)
    return seen


def _tire_age(value: Any) -> int:
    try:
        if value is None or value != value:  # NaN
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def build_snapshot(
    driver_states: list[dict],
    *,
    circuit_id: str,
    current_lap: int,
    total_laps: int | None = None,
    race_flag: str = "GREEN",
    origin_note: str = _ORIGIN_NOTE,
) -> dict:
    """Chain intervals and normalise per-driver states into a snapshot payload.

    Intervals chain only through cars still running: a retired car's frozen time
    would otherwise hand the car behind it a negative interval, which the schema
    rejects. Retired cars keep a zero interval because they are no longer in a
    running order at all.
    """
    ordered = sorted(driver_states, key=lambda d: int(d["position"]))

    previous: float | None = None
    for state in ordered:
        if state["running"] and previous is not None:
            state["interval_to_ahead_seconds"] = max(
                0.0, round(float(state["gap_to_leader_seconds"]) - previous, 3)
            )
        else:
            state["interval_to_ahead_seconds"] = 0.0
        if state["running"]:
            previous = float(state["gap_to_leader_seconds"])

    return {
        "circuit_id": circuit_id,
        "current_lap": int(current_lap),
        "total_laps": total_laps,
        "race_flag": race_flag,
        "origin_note": origin_note,
        "drivers": ordered,
    }


def fold_lap_frame(
    laps: Any,
    *,
    circuit_id: str,
    at_lap: int | None = None,
    race_flag: str | None = None,
    origin_note: str = _ORIGIN_NOTE,
) -> dict | None:
    """Fold a FastF1-shaped laps frame into a live snapshot payload.

    ``at_lap`` truncates the frame to laps completed at that point, which is how a
    poll reproduces the state of a race at a chosen instant. Returns ``None`` when
    the frame holds no usable lap yet (a session that has not produced timing).
    """
    completed = _completed_laps(laps, at_lap)
    if completed is None:
        return None
    frame = _green_laps(completed)
    if frame.empty:
        return None

    latest = _latest_per_driver(frame)
    known = _last_known_positions(frame)
    latest = latest.assign(_pos=_assign_positions(latest, known)).sort_values("_pos")
    if latest.empty:
        return None

    running = _is_running(latest)
    leader_code = (
        str(latest.loc[running, "Driver"].iloc[0])
        if running.any()
        else str(latest.iloc[0]["Driver"])
    )
    reference = LeaderReference.from_frame(frame, leader_code)
    if not reference.has_laps:
        return None

    if race_flag is None:
        # Read the flag from the unfiltered frame: a lap run entirely under a VSC
        # is excluded from the green set, so deriving the flag from the filtered
        # rows would report GREEN while a virtual safety car is actually out.
        race_flag = (
            track_flag(completed["TrackStatus"].iloc[-1])
            if "TrackStatus" in completed.columns
            else "GREEN"
        )

    states: list[dict] = []
    for _, row in latest.iterrows():
        code = str(row["Driver"]).strip()
        is_running = bool(running[row.name])
        states.append(
            {
                "driver_code": code.upper()[:3],
                "position": int(row["_pos"]),
                "gap_to_leader_seconds": round(
                    reference.gap_seconds(int(row["LapNumber"]), row["Time"]), 3
                ),
                "interval_to_ahead_seconds": 0.0,  # chained in build_snapshot
                "laps_completed": int(row["LapNumber"]),
                "current_tire_compound": _normalise_compound(row.get("Compound", "")),
                "compounds_used_so_far": _stint_compounds(frame, code),
                "tire_age": _tire_age(row.get("TyreLife")),
                "fuel_remaining_kg": None,
                "running": is_running,
                "dnf_reason": None if is_running else "retired",
            }
        )

    current_lap = int(latest["LapNumber"].max())
    return build_snapshot(
        states,
        circuit_id=circuit_id,
        current_lap=current_lap,
        race_flag=race_flag,
        origin_note=origin_note,
    )


def _last_complete_offset(source: Path, size: int) -> int:
    """Byte length of the longest whole-line prefix within the first ``size`` bytes.

    Scans backwards in fixed windows so the cost is one small read per window
    rather than a pass over the whole file. A racing recording reaches hundreds of
    megabytes, and the obvious implementation -- copy everything, then re-read the
    copy to trim it -- doubles both the I/O and the peak memory of every poll.
    """
    window = 65536
    with source.open("rb") as handle:
        end = size
        while end > 0:
            start = max(0, end - window)
            handle.seek(start)
            chunk = handle.read(end - start)
            index = chunk.rfind(b"\n")
            if index != -1:
                return start + index + 1
            end = start
    return 0


def snapshot_prefix(
    recording: str | Path,
    *,
    upto_bytes: int | None = None,
    dest: str | Path | None = None,
) -> Path | None:
    """Copy the first ``upto_bytes`` of a growing recording to a standalone file.

    A live recording is appended to while we read it, so a poll must snapshot the
    bytes it means to process rather than hand FastF1 a file that is still moving.
    The copy stops at the last newline: a prefix that ends mid-message would fail to
    parse, and the remainder arrives on the next poll.

    Only whole lines are ever written, so the copy needs no trimming pass.

    Returns ``None`` when the recording holds no complete message yet.
    """
    source = Path(recording)
    if not source.is_file():
        return None

    available = source.stat().st_size
    if available <= 0:
        return None
    size = available if upto_bytes is None else min(upto_bytes, available)
    complete = _last_complete_offset(source, size)
    if complete <= 0:
        return None

    target = Path(dest) if dest is not None else source.with_suffix(source.suffix + ".prefix")
    remaining = complete
    with source.open("rb") as src, target.open("wb") as dst:
        while remaining > 0:
            chunk = src.read(min(65536, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            dst.write(chunk)

    return target


def write_snapshot(payload: dict, out_dir: str | Path, circuit_id: str, lap: int) -> Path:
    """Write one lap-addressable snapshot named ``<circuit>_lap<NN>.json``."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{circuit_id}_lap{lap:02d}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
