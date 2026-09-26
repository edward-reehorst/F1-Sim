"""Live sources: where race state comes from, behind one interface.

The capture script polls a :class:`LiveSource` and writes whatever snapshot comes
back. Sources differ only in how they obtain the timing rows; the fold that turns
rows into a snapshot lives in :mod:`f1_sim.tuning.live` and is shared by all of
them.

**F1TV is required.** The live timing stream returns ``401`` without an
authenticated F1TV token, and there is no anonymous alternative. FastF1 also
cannot read a session in progress from its static archive (it answers ``403``
until the session ends), which is why a session is read by recording the live
stream *and then* folding a growing prefix of that recording.

Start the recorder **before** the session. It is the recording alone that carries
``DriverList``/``SessionInfo``, which the lap data needs to be interpretable.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, Protocol

from f1_sim.tuning.live import fold_lap_frame, snapshot_prefix

_LOG = logging.getLogger(__name__)

AUTH_HELP = """\
F1TV authentication is required for live timing (the stream returns 401 without
it). To set it up:

  1. Subscribe to F1 TV with live timing (in the US, F1 TV Premium is included
     with an Apple TV subscription and must be activated separately).
  2. Authenticate the token cache -- this prints a browser URL:
       python -m fastf1 auth f1tv --authenticate
  3. Open that URL, sign in with your F1 account, and approve the request.
  4. Verify it:
       python -m fastf1 auth f1tv --status

See docs/f1tv_activation.md for the full walkthrough."""


class LiveSource(Protocol):
    """A pollable source of live race state."""

    def poll(self) -> dict | None:
        """Return a snapshot payload, or ``None`` if there is nothing new yet."""


class AuthenticationRequired(RuntimeError):
    """Raised when the live stream refuses the connection for lack of a token."""


class RecordingNotReady(RuntimeError):
    """Raised when a recording prefix does not yet carry usable timing data."""


# Topics a prefix must carry before it can be folded. Without DriverList the lap
# data has no drivers attached to it, so FastF1 would produce an empty frame.
REQUIRED_TOPICS = ("TimingData", "DriverList")


def load_laps_from_recording(prefix: Path, *, year: int, gp: str, session: str) -> Any:
    """Load laps from a live-timing recording prefix via FastF1.

    ``Session.load(livedata=...)`` asks FastF1 to interpret a recording rather than
    download one, so the lap and timing data come entirely from the file. Building
    the ``Session`` still resolves the event schedule, which is a small cached
    lookup -- the schedule for a race is published well before it starts.

    The required topics are checked up front on purpose. FastF1's data loaders fall
    back to the network archive when the recording is missing a topic
    (``_extended_timing_data`` hits the API unless ``livedata.has('TimingData')``),
    which for this project would be actively harmful: during a live session the
    archive answers ``403``, and for a finished session it would quietly hand back
    the whole race, making a dead recorder look like a working one.
    """
    import fastf1
    from fastf1.livetiming.data import LiveTimingData

    livedata = LiveTimingData(str(prefix))
    missing = [topic for topic in REQUIRED_TOPICS if not livedata.has(topic)]
    if missing:
        raise RecordingNotReady(
            f"recording has no {', '.join(missing)} yet ({prefix.stat().st_size} bytes captured)"
        )

    f1_session = fastf1.get_session(year, gp, session)
    f1_session.load(
        laps=True,
        telemetry=False,
        weather=False,
        messages=False,
        livedata=livedata,
    )
    return f1_session.laps


class SignalRRecorder:
    """Background F1TV live-timing recorder.

    Wraps ``SignalRClient``, whose ``start()`` blocks for the whole session. It
    runs in a daemon thread so the capture loop can poll the file it is writing.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        timeout: int = 60,
        logger: Any | None = None,
    ) -> None:
        self.path = Path(path)
        self.timeout = timeout
        self._logger = logger
        self._client: Any | None = None
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _run(self) -> None:
        try:
            from fastf1.livetiming.client import SignalRClient

            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._client = SignalRClient(
                str(self.path),
                filemode="a",
                timeout=self.timeout,
                logger=self._logger,
            )
            self._client.start()
        except BaseException as exc:  # surfaced by the capture loop
            self._error = exc
            self._client = None

    def start(self) -> SignalRRecorder:
        """Start recording in the background.

        Returns immediately; connection problems surface from :meth:`raise_if_failed`.
        """
        self._thread = threading.Thread(target=self._run, name="f1tv-recorder", daemon=True)
        self._thread.start()
        return self

    def raise_if_failed(self) -> None:
        """Re-raise a connection/auth failure with actionable instructions."""
        if self._error is None:
            return
        error, self._error = self._error, None
        text = str(error).lower()
        if "401" in text or "unauthor" in text or "auth" in text:
            raise AuthenticationRequired(f"{error}\n\n{AUTH_HELP}") from error
        raise error

    def wait_for_data(self, timeout: float = 30.0, poll: float = 0.25) -> bool:
        """Block until the recording has content, the thread dies, or we time out."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.raise_if_failed()
            if self.path.is_file() and self.path.stat().st_size > 0:
                return True
            if self._thread and not self._thread.is_alive():
                self.raise_if_failed()
                return False
            time.sleep(poll)
        return self.path.is_file() and self.path.stat().st_size > 0

    def stop(self, timeout: float = 5.0) -> None:
        """Stop recording.

        ``SignalRClient`` has no public stop, and its supervise loop exits when no
        message arrives within ``timeout`` seconds. Shrinking that threshold is the
        supported way to make it shut down and close its file cleanly.
        """
        client = self._client
        if client is not None:
            client.timeout = 1
            if client.logger is not None:
                client.logger.setLevel(logging.ERROR)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout)

    def __enter__(self) -> SignalRRecorder:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


class RecordingLiveSource:
    """Poll a growing live-timing recording, folding each new prefix into a snapshot.

    Each poll snapshots the bytes received so far, hands that prefix to FastF1, and
    folds the resulting laps into a snapshot. Re-processing the whole prefix each
    time is O(n²) over a session but costs well under a second for a race-length
    recording, and it means no incremental parser can drift out of sync with the
    recording it is replaying.

    The timing data comes off disk. ``year``/``gp``/``session`` are only used to
    resolve the event schedule, not to fetch results.
    """

    def __init__(
        self,
        recording: str | Path,
        *,
        circuit_id: str,
        year: int,
        gp: str,
        session: str = "R",
        loader: Callable[[Path], Any] | None = None,
        tmp_dir: str | Path | None = None,
        at_lap: int | None = None,
    ) -> None:
        self.recording = Path(recording)
        self.circuit_id = circuit_id
        self.year = year
        self.gp = gp
        self.session = session
        self.at_lap = at_lap
        if loader is None:
            loader = partial(load_laps_from_recording, year=year, gp=gp, session=session)
        self._loader = loader
        self._prefix = Path(tmp_dir or self.recording.parent) / f"{self.recording.stem}.prefix"
        self._consumed = 0
        self.polls = 0
        # A live feed never runs out: a None poll means "not yet", not "finished".
        self.exhausted = False
        self.last_error: str | None = None

    def poll(self) -> dict | None:
        """Fold everything recorded so far, or ``None`` if nothing is new."""
        if not self.recording.is_file():
            return None
        size = self.recording.stat().st_size
        if size <= self._consumed:
            return None

        prefix = snapshot_prefix(self.recording, upto_bytes=size, dest=self._prefix)
        if prefix is None:
            return None
        # Count what was actually read, not what existed: a prefix trimmed back to
        # the last newline leaves a partial message for the next poll.
        self._consumed = prefix.stat().st_size

        self.polls += 1
        try:
            laps = self._loader(prefix)
        except RecordingNotReady as exc:
            # Expected while the feed is still warming up; worth surfacing, since a
            # recorder that never delivers these topics will never capture anything.
            self.last_error = str(exc)
            return None
        except Exception as exc:  # noqa: BLE001 - a partial feed is expected mid-session
            _LOG.debug("poll %d: recording not yet interpretable: %s", self.polls, exc)
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None
        self.last_error = None
        if laps is None or len(laps) == 0:
            return None
        return fold_lap_frame(laps, circuit_id=self.circuit_id, at_lap=self.at_lap)


class PostSessionSource:
    """Source from the FastF1 static archive, walked one lap per poll.

    Only usable once a session has ended, but it needs no F1TV token, so it is how
    the capture path gets exercised offline and against finished races. One poll per
    lap also exercises the capture loop's lap-deduplication for real.
    """

    def __init__(
        self,
        *,
        year: int,
        gp: str,
        session_identifier: str,
        circuit_id: str,
        at_lap: int | None = None,
        laps: Any | None = None,
    ) -> None:
        self.year = year
        self.gp = gp
        self.session_identifier = session_identifier
        self.circuit_id = circuit_id
        self.at_lap = at_lap
        self._laps = laps
        self._last_lap: int | None = None
        self.exhausted = False

    def _ensure_laps(self) -> Any:
        if self._laps is None:
            from f1_sim.tuning.fastf1 import _load_session

            self._laps = _load_session(
                _session(self.year, self.gp, self.session_identifier),
                year=self.year,
                gp=self.gp,
                identifier=self.session_identifier,
            ).laps
        return self._laps

    def total_laps(self) -> int | None:
        laps = self._ensure_laps()
        if laps is None or len(laps) == 0:
            return None
        return int(laps["LapNumber"].max())

    def poll(self) -> dict | None:
        """Next un-emitted lap, or ``None`` once every lap has been written."""
        laps = self._ensure_laps()
        if laps is None or len(laps) == 0:
            self.exhausted = True
            return None
        limit = self.at_lap if self.at_lap is not None else self.total_laps()
        if limit is None:
            self.exhausted = True
            return None
        start = 1 if self._last_lap is None else self._last_lap + 1
        if start > limit:
            self.exhausted = True
            return None
        self._last_lap = start
        return fold_lap_frame(laps, circuit_id=self.circuit_id, at_lap=start)


def _session(year: int, gp: str, identifier: str) -> Any:
    import fastf1

    return fastf1.get_session(year, gp, identifier)
