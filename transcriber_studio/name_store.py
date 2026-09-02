# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Names the user has given Plaud recordings, and whether Plaud knows yet.

A rename is committed here first and pushed to Plaud second. That order is the
whole design: pushing is the part that can fail — an expired token, a moved
endpoint, no network — and a rename that only half happened should leave the
name where the user can see it, not lose it.

The first name Plaud gave a recording is kept alongside, so a rename can always
be undone even after Plaud has forgotten what the recording used to be called.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass

from .config import APP_DIR

STORE_PATH = APP_DIR / "plaud_names.json"

_lock = threading.Lock()
_cache: dict[str, LocalName] | None = None


@dataclass
class LocalName:
    """One renamed recording."""

    name: str
    #: What the recording was called before this app first renamed it.
    original: str = ""
    #: False when the name is only local — the push failed, or there was no
    #: token to push with. The UI says so, and a later push can clear it.
    pushed: bool = False


def _read() -> dict[str, LocalName]:
    if not STORE_PATH.exists():
        return {}
    try:
        raw = json.loads(STORE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}      # unreadable or half-written: the names are a convenience
    if not isinstance(raw, dict):
        return {}
    out: dict[str, LocalName] = {}
    for file_id, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "")
        if not name:
            continue
        out[str(file_id)] = LocalName(
            name=name,
            original=str(entry.get("original") or ""),
            pushed=bool(entry.get("pushed")),
        )
    return out


def load(force: bool = False) -> dict[str, LocalName]:
    global _cache
    with _lock:
        if _cache is None or force:
            _cache = _read()
        return dict(_cache)


def _write(entries: dict[str, LocalName]) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STORE_PATH.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps({k: asdict(v) for k, v in entries.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(STORE_PATH)      # never leave a half-written file to be read


def get(file_id: str) -> LocalName | None:
    return load().get(file_id)


def name_for(file_id: str, fallback: str) -> str:
    """The name to show for a recording: the local one if there is one."""
    entry = get(file_id)
    return entry.name if entry else fallback


def record(file_id: str, name: str, *, original: str, pushed: bool) -> LocalName:
    """Save a rename. Keeps the earliest known original name.

    ``original`` is only taken the first time: renaming twice must still be able
    to get back to what Plaud called the recording to begin with, not to the
    intermediate name this app invented.
    """
    global _cache
    with _lock:
        entries = _read()
        existing = entries.get(file_id)
        entry = LocalName(
            name=name,
            original=(existing.original if existing and existing.original else original),
            pushed=pushed,
        )
        entries[file_id] = entry
        _write(entries)
        _cache = entries
        return entry


def mark_pushed(file_id: str) -> None:
    """Note that Plaud has accepted the name we already stored."""
    global _cache
    with _lock:
        entries = _read()
        entry = entries.get(file_id)
        if entry is None or entry.pushed:
            return
        entry.pushed = True
        entries[file_id] = entry
        _write(entries)
        _cache = entries


def unsynced() -> dict[str, LocalName]:
    """Renames Plaud has not accepted yet."""
    return {k: v for k, v in load().items() if not v.pushed}


def forget(file_id: str) -> None:
    """Drop a local rename, so the recording shows whatever Plaud calls it."""
    global _cache
    with _lock:
        entries = _read()
        if entries.pop(file_id, None) is None:
            return
        _write(entries)
        _cache = entries


def clear() -> None:
    global _cache
    with _lock:
        _write({})
        _cache = {}
