# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cooperative cancellation for long-running job actions."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeAlias

ShouldCancel: TypeAlias = Callable[[], bool] | None


class JobCancelled(Exception):
    """Raised when the user cancels a running job action."""


def check_cancel(
    should_cancel: ShouldCancel,
    log_cb: Callable[[str], None] | None = None,
    *,
    message: str = "Cancelled.",
) -> None:
    if should_cancel and should_cancel():
        if log_cb:
            log_cb(message)
        raise JobCancelled(message)


def sleep_cancellable(
    seconds: float,
    should_cancel: ShouldCancel,
    log_cb: Callable[[str], None] | None = None,
) -> None:
    """Sleep in short slices so cancel is picked up quickly."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        check_cancel(should_cancel, log_cb, message="Cancelled.")
        time.sleep(min(0.25, end - time.monotonic()))
