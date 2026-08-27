# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Persistent local cache for Plaud recording audio downloads."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from .config import APP_DIR
from .models import Recording, Source

CACHE_DIR = APP_DIR / "audio_cache"


def cache_path(plaud_id: str) -> Path:
    return CACHE_DIR / f"{plaud_id}.mp3"


def _legacy_temp_path(plaud_id: str) -> Path:
    return Path(tempfile.gettempdir()) / f"plaud_{plaud_id}.mp3"


def is_cached(plaud_id: str) -> bool:
    path = cache_path(plaud_id)
    if path.is_file() and path.stat().st_size > 0:
        return True
    legacy = _legacy_temp_path(plaud_id)
    return legacy.is_file() and legacy.stat().st_size > 0


def _ensure_cached_file(plaud_id: str) -> Path | None:
    path = cache_path(plaud_id)
    if path.is_file() and path.stat().st_size > 0:
        return path
    legacy = _legacy_temp_path(plaud_id)
    if legacy.is_file() and legacy.stat().st_size > 0:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(legacy), str(path))
            return path
        except OSError:
            shutil.copy2(str(legacy), str(path))
            return path
    return None


def attach_if_cached(recording: Recording) -> Recording:
    """Set local_path when a cached copy already exists on disk."""
    if recording.source != Source.PLAUD:
        return recording
    path = _ensure_cached_file(recording.id)
    if path is not None:
        recording.local_path = str(path)
    return recording


def audio_status_label(recording: Recording) -> str:
    if not recording.audio_available:
        return "—"
    if is_cached(recording.id):
        return "cached ✓"
    return "cloud"
