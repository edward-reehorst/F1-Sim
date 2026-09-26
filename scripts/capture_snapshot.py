"""Zero-network live-timing capture for `f1-sim` resume.

Polls a *local* directory containing live-timing files (FastF1 ``LiveTimingData``
timing dumps or plain JSON in the f1-sim snapshot shape) and, every ``--poll``
seconds, re-captures the whole field into a numbered, lap-addressable snapshot
JSON. Reading a local file is network-free by construction; this script performs
zero additional FastF1 API calls.

Output filenames encode the lap: ``<out>/<circuit>_lap<N>.json``, and every file
is written in the exact shape ``f1-sim race --live <file>`` / ``load_snapshot``
parses back into a ``LiveRaceSnapshot``. The polling loop is lap-deduplicated so
we never overwrite a lap that was already captured, and ``--once`` captures a
single frame and exits.

Usage::

    uv run python scripts/capture_snapshot.py \\
        --watch-dir path/to/live-timing --out out/snapshots \\
        --circuit monza --poll 5.0

    # single frame, then exit
    uv run python scripts/capture_snapshot.py --watch-dir path --circuit monza --once
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from f1_sim.loaders import load_snapshot


def _newest_snapshot_source(watch_dir: Path) -> Path | None:
    """Newest local timing file (*.pkl or *.json) in ``watch_dir``."""
    candidates = [p for p in watch_dir.iterdir() if p.suffix.lower() in (".pkl", ".json")]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Poll local live-timing files and write lap-addressable snapshots "
        "for f1-sim live resume. Zero network access by construction."
    )
    parser.add_argument("--watch-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--circuit", required=True)
    parser.add_argument("--poll", type=float, default=5.0, help="Seconds between polls.")
    parser.add_argument("--max-laps", type=int, default=None)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)

    watch = args.watch_dir.resolve()
    if not watch.is_dir():
        print(f"error: watch dir not found: {watch}", file=sys.stderr)
        return 2

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    captured_laps: set[int] = set()
    poll = 0
    while True:
        poll += 1
        if poll > 1 and args.once:
            break
        try:
            source = _newest_snapshot_source(watch)
            if source is None:
                print(
                    f"[poll {poll}] no local timing dump (*.pkl / *.json) in {watch} — "
                    "waiting (recorder never touches a network).",
                    file=sys.stderr,
                )
            else:
                snapshot = load_snapshot(str(source))  # real loader: local-file only
                lap = snapshot.current_lap
                if not captured_laps or lap not in captured_laps:
                    if args.max_laps is not None and lap > args.max_laps:
                        print(
                            f"[poll {poll}] lap {lap} > max_laps={args.max_laps}; finishing.",
                            file=sys.stderr,
                        )
                        break
                    lap_path = out / f"{snapshot.circuit_id}_lap{lap:02d}.json"
                    lap_path.write_text(
                        json.dumps(snapshot.model_dump(mode="json"), indent=2),
                        encoding="utf-8",
                    )
                    captured_laps.add(lap)
                    print(
                        f"[poll {poll}] captured lap {lap} ({len(snapshot.drivers)} cars) -> {lap_path}"
                    )
                else:
                    print(
                        f"[poll {poll}] lap {lap} already captured; waiting for a newer lap."
                    )
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            print(f"[poll {poll}] {exc}", file=sys.stderr)

        if args.once:
            break
        time.sleep(max(0.01, args.poll))

    return 0 if captured_laps else 3


if __name__ == "__main__":
    raise SystemExit(main())
