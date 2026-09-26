"""Capture live F1 race state into lap-addressable snapshots for ``f1-sim --live``.

This is the *only* part of the project that touches the network. It writes
``<out>/<circuit>_lap<NN>.json`` files in the exact shape ``load_snapshot`` /
``f1-sim race --live`` read back, then gets out of the way: resuming a race from
one of those files is a local read and touches nothing online.

Two sources, same output:

``recording`` (live)
    Starts an F1TV live-timing recorder, then polls the file it is writing,
    folding a growing prefix into a snapshot each time. Requires an F1TV token --
    the live stream answers ``401`` without one. **Start it before the session:**
    the recording is the only source of the driver and session metadata the lap
    data needs.

``post-session`` (offline)
    Reads the FastF1 static archive of a finished session, one lap per poll. No
    token needed; this is how the capture path is tested against real races.

Usage::

    # live: record F1TV timing and write a snapshot per lap
    uv run python scripts/network_capture.py record \\
        --year 2026 --gp italian \\
        --circuit monza --out out/snapshots --poll 5

    # offline: replay a finished race, writing a snapshot per lap
    uv run python scripts/network_capture.py record \\
        --circuit monza --out out/snapshots \\
        --source post-session --year 2026 --gp italian --session R
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

from f1_sim.loaders import load_snapshot
from f1_sim.tuning.live import write_snapshot
from f1_sim.tuning.recording import (
    AUTH_HELP,
    AuthenticationRequired,
    LiveSource,
    PostSessionSource,
    RecordingLiveSource,
    SignalRRecorder,
)

_STOP = False


def _handle_stop(signum: int, frame: object) -> None:
    global _STOP
    _STOP = True
    print(f"\nstopping (signal {signum}); snapshots already written are complete")


def build_source(args: argparse.Namespace) -> tuple[LiveSource, SignalRRecorder | None]:
    """Create the source, plus the recorder when one is needed."""
    if args.source == "post-session":
        source: LiveSource = PostSessionSource(
            year=args.year,
            gp=args.gp,
            session_identifier=args.session,
            circuit_id=args.circuit,
            at_lap=args.at_lap,
        )
        return source, None

    if not args.year or not args.gp:  # defensive: main() validates this up front
        raise SystemExit(
            "error: --year and --gp are required so FastF1 can resolve the event\n"
            "schedule for the session being recorded (e.g. --year 2026 --gp monza).\n"
            "The timing data itself comes from the recording, not the archive.\n\n" + AUTH_HELP
        )

    recording = Path(args.recording or (Path(args.out) / f"{args.circuit}_live.txt"))
    recorder = SignalRRecorder(recording, timeout=args.recorder_timeout)
    recorder.start()
    print(f"recording live timing -> {recording}")
    print("waiting for the first messages (start this before the session begins)...")
    if not recorder.wait_for_data(timeout=args.startup_wait):
        recorder.raise_if_failed()
        print(
            f"error: no live data arrived within {args.startup_wait:g}s.\n"
            "Either the F1TV token is missing, no session is live, or FastF1 is\n"
            "waiting for an interactive F1TV login in the browser (it prints a\n"
            "f1login.fastf1.dev URL when that happens -- complete that login, or\n"
            "authenticate up front).\n\n" + AUTH_HELP,
            file=sys.stderr,
        )
        raise SystemExit(4)

    source = RecordingLiveSource(
        recording,
        circuit_id=args.circuit,
        year=args.year,
        gp=args.gp,
        session=args.session,
        at_lap=args.at_lap,
    )
    return source, recorder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    record = sub.add_parser("record", help="record and write snapshots")
    record.add_argument(
        "--source",
        choices=("recording", "post-session"),
        default="recording",
        help="recording = live F1TV stream (needs auth); post-session = finished race.",
    )
    record.add_argument("--circuit", required=True, help="Bundled circuit id, e.g. monza.")
    record.add_argument("--out", type=Path, required=True, help="Output directory.")
    record.add_argument("--poll", type=float, default=5.0, help="Seconds between polls.")
    record.add_argument(
        "--max-laps",
        type=int,
        default=None,
        help="Stop once this lap has been captured (default: run until stopped).",
    )
    record.add_argument(
        "--at-lap",
        type=int,
        default=None,
        help="Only capture this lap (post-session source; useful for testing).",
    )
    record.add_argument(
        "--recording",
        type=Path,
        default=None,
        help="Live recording file (default: <out>/<circuit>_live.txt).",
    )
    record.add_argument(
        "--recorder-timeout",
        type=int,
        default=60,
        help="Seconds without a message before the recorder gives up.",
    )
    record.add_argument(
        "--startup-wait",
        type=float,
        default=30.0,
        help="Seconds to wait for the first live message.",
    )
    record.add_argument("--year", type=int, default=None)
    record.add_argument("--gp", default=None, help="FastF1 gp slug, e.g. italian.")
    record.add_argument("--session", default="R", help="Session identifier (R, Q, ...).")

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    if args.command != "record":  # pragma: no cover - argparse enforces this
        parser.error(f"unknown command: {args.command}")

    # Both sources resolve the event through FastF1's schedule: post-session to
    # fetch the laps, live to name the session the recording belongs to. Without
    # them, post-session would otherwise fail deep inside FastF1 with a confusing
    # error instead of a clear one here.
    if not args.year or not args.gp:
        parser.error(
            "--year and --gp are required for both --source recording and "
            "--source post-session, so FastF1 can resolve the event schedule "
            "(e.g. --year 2026 --gp italian)"
        )

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    try:
        source, recorder = build_source(args)
    except AuthenticationRequired as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4

    captured: set[int] = set()
    reported_reason: str | None = None
    poll = 0
    try:
        while not _STOP:
            poll += 1
            try:
                payload = source.poll()
            except AuthenticationRequired as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 4
            except Exception as exc:  # noqa: BLE001 - a bad poll must not kill capture
                print(f"[poll {poll}] {exc}", file=sys.stderr)
                payload = None

            if payload is not None:
                # Re-validate through the real loader, so a written snapshot is
                # always one the engine can actually resume from.
                snapshot = load_snapshot(payload)
                lap = snapshot.current_lap
                if lap in captured:
                    print(f"[poll {poll}] lap {lap} already captured; waiting.")
                elif args.max_laps is not None and lap > args.max_laps:
                    print(f"[poll {poll}] lap {lap} > max-laps {args.max_laps}; finishing.")
                    break
                else:
                    path = write_snapshot(payload, out, snapshot.circuit_id, lap)
                    captured.add(lap)
                    running = sum(1 for d in snapshot.drivers if d.running)
                    print(
                        f"[poll {poll}] lap {lap} {snapshot.race_flag}: "
                        f"{len(snapshot.drivers)} cars, {running} running -> {path.name}"
                    )
            else:
                if getattr(source, "exhausted", False):
                    break
                reason = getattr(source, "last_error", None)
                if reason and reason != reported_reason:
                    # Only print on change: a warming-up feed reports this every poll.
                    reported_reason = reason
                    print(f"[poll {poll}] waiting on timing data: {reason}")
                else:
                    print(f"[poll {poll}] no usable timing data yet; waiting.")

            if captured and (
                getattr(source, "exhausted", False)
                or (args.max_laps is not None and max(captured) >= args.max_laps)
                or (args.at_lap is not None and max(captured) >= args.at_lap)
            ):
                break

            # The archive is already in memory, so consecutive polls are just the
            # next lap; only a live feed needs a wait between polls.
            if args.source != "post-session":
                time.sleep(max(0.05, args.poll))
    finally:
        if recorder is not None:
            recorder.stop()

    print(f"wrote {len(captured)} snapshot(s) to {out}")
    for lap in sorted(captured):
        print(f"  {out}/{args.circuit}_lap{lap:02d}.json")
    return 0 if captured else 3


if __name__ == "__main__":
    raise SystemExit(main())
