# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test-wide setup that has to happen before Qt is imported."""

from __future__ import annotations

import os

# No display on CI, and none wanted locally either: the tests build real
# widgets but must never put a window on someone's screen.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
