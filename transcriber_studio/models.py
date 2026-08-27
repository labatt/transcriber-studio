# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared data models used across the app."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Source(str, Enum):
    PLAUD = "plaud"
    LOCAL = "local"


@dataclass
class Recording:
    """A transcribable item — either a Plaud cloud recording or a local file."""

    source: Source
    # Plaud recordings use the Plaud file id; local files use the absolute path.
    id: str
    name: str
    date: str = ""              # YYYY-MM-DD (best-effort)
    datetime: str = ""          # ISO start datetime when known
    duration: str = ""          # human string e.g. "16m59s"
    duration_seconds: float = 0.0
    audio_available: bool = True
    local_path: str | None = None      # populated for local files / after download
    serial_number: str = ""

    @property
    def display_name(self) -> str:
        return self.name or self.id


@dataclass
class Segment:
    """One transcribed segment."""

    start: float
    end: float
    text: str
    speaker: str | None = None  # resolved (renamed) speaker label
    channel: str | None = None  # channel label when per-channel mode used


@dataclass
class TranscriptResult:
    recording: Recording
    segments: list[Segment] = field(default_factory=list)
    language: str = ""
    model: str = ""
    speakers: list[str] = field(default_factory=list)  # ordered unique speaker labels

    @property
    def speaker_count(self) -> int:
        return len(self.speakers)
