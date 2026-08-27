# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Durable record of what has been done with each recording.

The jobs list is a work queue: rows come and go as the user clears them. This
file is the memory that outlives it — for every recording ever processed, what
happened, when, and where the files landed. The Plaud recordings browser reads
it to fill its Status column, so a recording transcribed last month still says
so long after its job row was removed.

Entries are keyed by recording id (the Plaud file id, or the absolute path for
a local file) and survive queue clears, app restarts, and re-logins.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import APP_DIR
from .models import Recording

HISTORY_PATH = APP_DIR / "history.json"

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"

#: States that mean "nothing was produced" — safe to forget when a job row goes away.
UNFINISHED = (QUEUED, RUNNING)

_LABELS = {
    QUEUED: "Queued",
    RUNNING: "In progress…",
    DONE: "✓ Transcribed",
    FAILED: "✗ Failed",
    CANCELLED: "Cancelled",
}


@dataclass
class Entry:
    """What is known about one recording's trip through the app."""

    id: str
    name: str = ""
    source: str = "plaud"
    state: str = DONE
    ai_cleanup: bool = False
    speakers: int = 0
    outputs: list[str] = field(default_factory=list)
    error: str = ""
    updated_at: float = 0.0

    @property
    def label(self) -> str:
        """Short text for the Status column."""
        base = _LABELS.get(self.state, self.state.title())
        if self.state == DONE and self.ai_cleanup:
            base += " + AI cleanup"
        return base

    @property
    def when(self) -> str:
        if not self.updated_at:
            return ""
        return datetime.fromtimestamp(self.updated_at).strftime("%Y-%m-%d %H:%M")

    def tooltip(self) -> str:
        lines = [self.label]
        if self.when:
            lines.append(f"Last activity: {self.when}")
        if self.state == DONE and self.speakers:
            lines.append(f"{self.speakers} speaker(s) detected")
        if self.error:
            lines.append(f"Error: {self.error}")
        if self.outputs:
            lines.append("")
            lines.append("Output files:")
            for path in self.outputs:
                mark = "" if Path(path).is_file() else "  (missing)"
                lines.append(f"  {Path(path).name}{mark}")
        return "\n".join(lines)


# -- storage ------------------------------------------------------------
# The recordings browser asks for a status on every row it draws, so the file
# is parsed once and re-read only when it changes underneath us.
_cache: dict[str, Entry] | None = None
_cache_stamp: float | None = None


def _entry_from_dict(d: dict[str, Any]) -> Entry | None:
    try:
        known = {f for f in asdict(Entry(id=""))}
        return Entry(**{k: v for k, v in d.items() if k in known})
    except (TypeError, ValueError):
        return None


def _stamp() -> float | None:
    try:
        return HISTORY_PATH.stat().st_mtime
    except OSError:
        return None


def load(force: bool = False) -> dict[str, Entry]:
    """All entries, keyed by recording id. Cached against the file's mtime."""
    global _cache, _cache_stamp
    stamp = _stamp()
    if not force and _cache is not None and stamp == _cache_stamp:
        return _cache
    entries: dict[str, Entry] = {}
    if stamp is not None:
        try:
            data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
            for raw in data.get("items", []):
                entry = _entry_from_dict(raw)
                if entry and entry.id:
                    entries[entry.id] = entry
        except (OSError, json.JSONDecodeError, AttributeError):
            entries = {}       # a truncated file is not worth losing the app over
    _cache, _cache_stamp = entries, stamp
    return entries


def _save(entries: dict[str, Entry]) -> None:
    global _cache, _cache_stamp
    items = sorted(entries.values(), key=lambda e: e.updated_at, reverse=True)
    payload = json.dumps(
        {"items": [asdict(e) for e in items]}, indent=2, ensure_ascii=False
    )
    try:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        tmp = HISTORY_PATH.with_suffix(".json.tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(HISTORY_PATH)
    except OSError:
        return                 # history is a convenience; never break a job over it
    _cache, _cache_stamp = entries, _stamp()


def possible_labels() -> list[str]:
    """Every label the Status column can show, so a view can size itself to fit."""
    labels = list(_LABELS.values())
    labels.append(f"{_LABELS[DONE]} + AI cleanup")
    return labels


def get(recording_id: str) -> Entry | None:
    return load().get(recording_id)


def get_for(recording: Recording) -> Entry | None:
    return get(recording.id)


def record(recording: Recording, state: str, **changes: Any) -> Entry:
    """Upsert what just happened to a recording.

    Fields not named in `changes` keep whatever the previous run recorded, so a
    status update mid-job does not erase the output paths of the last one.
    """
    entries = dict(load())
    entry = entries.get(recording.id)
    if entry is None:
        entry = Entry(id=recording.id)
    entry.name = recording.display_name or entry.name
    entry.source = recording.source.value
    entry.state = state
    if state in UNFINISHED:
        entry.error = ""
    for key, value in changes.items():
        if hasattr(entry, key):
            setattr(entry, key, value)
    entry.outputs = [str(p) for p in entry.outputs]
    entry.updated_at = time.time()
    entries[recording.id] = entry
    _save(entries)
    return entry


def record_many(recordings: Iterable[Recording], state: str) -> None:
    """Same as `record` for a whole batch, in one write instead of one each."""
    entries = dict(load())
    now = time.time()
    for rec in recordings:
        entry = entries.get(rec.id) or Entry(id=rec.id)
        entry.name = rec.display_name or entry.name
        entry.source = rec.source.value
        entry.state = state
        if state in UNFINISHED:
            entry.error = ""
        entry.updated_at = now
        entries[rec.id] = entry
    _save(entries)


def forget(recording_id: str) -> None:
    entries = dict(load())
    if entries.pop(recording_id, None) is not None:
        _save(entries)


def drop_unfinished(recordings: Iterable[Recording]) -> None:
    """Forget rows that never produced anything (queued/running when removed).

    A job the user deletes before it finishes leaves nothing behind worth
    remembering; one that completed keeps its entry forever.
    """
    entries = dict(load())
    changed = False
    for rec in recordings:
        entry = entries.get(rec.id)
        if entry is not None and entry.state in UNFINISHED:
            del entries[rec.id]
            changed = True
    if changed:
        _save(entries)


def clear() -> None:
    """Wipe the whole history (used by tests and a deliberate user reset)."""
    global _cache, _cache_stamp
    try:
        HISTORY_PATH.unlink(missing_ok=True)
    except OSError:
        pass
    _cache, _cache_stamp = None, None
