# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""How long a cloud engine is allowed to spend sending a recording.

There is a trap in `requests` worth writing down, because it cost a real
failure and the symptom points the wrong way.

A timeout pair is `(connect, read)`, which reads as "how long to reach the
server" and "how long to wait for its answer". It is not what happens.
urllib3 sets the socket timeout to the *connect* value before it writes the
request body, and only swaps in the read value once the body is fully sent
(`connectionpool.py`, "Reset the timeout for the recv() on the socket").

So the connect timeout governs the whole upload. With the usual 30 seconds, a
70 MB recording had to leave the machine at 19 Mbit/s sustained or the socket
gave up — and it surfaced as ``Connection aborted: The write operation timed
out``, which reads like the server is unreachable rather than like a file that
is simply large.

The budget therefore has to be derived from the size of the thing being sent.
"""

from __future__ import annotations

import time
from pathlib import Path

#: The slowest upload still worth waiting for, in bytes per second. About
#: 1.2 Mbit/s — a poor connection, not a broken one. Set it too high and a slow
#: link fails a job it would have finished; too low and a dead transfer hangs
#: for longer than anyone wants to watch.
MIN_UPLOAD_BYTES_PER_SEC = 150_000

#: Floor for a small file, covering DNS, the TLS handshake and a slow start.
MIN_UPLOAD_SECONDS = 120

#: Ceiling, so a hung socket cannot hold a job open indefinitely. At the floor
#: rate this covers roughly 300 MB, which is the largest upload any engine here
#: accepts.
MAX_UPLOAD_SECONDS = 2400


def upload_seconds(size_bytes: int) -> int:
    """How long sending this many bytes may take before the socket gives up."""
    needed = size_bytes / MIN_UPLOAD_BYTES_PER_SEC if size_bytes > 0 else 0
    return int(min(MAX_UPLOAD_SECONDS, max(MIN_UPLOAD_SECONDS, needed)))


def timeout_for(path: str | Path, read_seconds: int) -> tuple[int, int]:
    """The `(connect, read)` pair to hand `requests` for uploading this file.

    The first value is not really a connect timeout — see the module note — it
    is the budget for getting the file out of the door. A host that is genuinely
    unreachable still fails quickly, because a refused or unroutable connection
    errors on its own rather than waiting out the clock.
    """
    try:
        size = Path(path).stat().st_size
    except OSError:
        size = 0
    return upload_seconds(size), read_seconds


def describe(size_bytes: int) -> str:
    """The budget in words, for a log line that has to explain a long wait."""
    seconds = upload_seconds(size_bytes)
    return f"{size_bytes / 1e6:.1f} MB, up to {seconds // 60} min to send"


def _mb(size_bytes: float) -> str:
    """Megabytes, with a decimal only where a whole number would read as zero.

    A small file otherwise reported "0 of 0 MB (4%)", which says less than
    nothing.
    """
    value = size_bytes / 1e6
    return f"{value:.1f}" if value < 10 else f"{value:.0f}"


# ---- watching an upload leave -----------------------------------------
#: How often a long upload says where it has got to. Often enough to prove the
#: transfer is alive, rare enough not to bury the run log.
PROGRESS_LOG_SECONDS = 15


class UploadProgress:
    """Turns bytes-sent into a progress fraction and an occasional log line.

    A cloud transcription is two long waits with nothing between them: sending
    the audio, then waiting for the model. Without this they look identical
    from the outside — a bar stuck at the same place — and there is no way to
    tell a slow upload from a hung one.
    """

    def __init__(self, total_bytes: int, *, log=None, progress_cb=None,
                 base: float = 0.0, span: float = 1.0):
        self.total = max(1, int(total_bytes))
        self.log = log
        self.progress_cb = progress_cb
        self.base = base
        self.span = span
        self.sent = 0
        self._last_log = 0.0
        self._started = time.monotonic()

    def __call__(self, monitor) -> None:
        """Called by the encoder as each chunk goes out."""
        self.update(getattr(monitor, "bytes_read", 0))

    def update(self, sent_bytes: int) -> None:
        self.sent = min(self.total, int(sent_bytes))
        fraction = self.sent / self.total
        if self.progress_cb:
            self.progress_cb(self.base + self.span * fraction)
        now = time.monotonic()
        if self.log and (now - self._last_log) >= PROGRESS_LOG_SECONDS:
            self._last_log = now
            self.log(self.describe_now())

    def elapsed(self) -> float:
        return time.monotonic() - self._started

    def describe_now(self) -> str:
        elapsed = max(0.001, time.monotonic() - self._started)
        rate = self.sent / elapsed
        remaining = (self.total - self.sent) / rate if rate > 0 else 0
        return (
            f"Uploading: {_mb(self.sent)} of {_mb(self.total)} MB "
            f"({self.sent / self.total:.0%}) at {rate / 1e6:.1f} MB/s"
            + (f" — about {remaining / 60:.0f} min left" if remaining > 90 else "")
        )
