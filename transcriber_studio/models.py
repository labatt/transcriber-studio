# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared data models used across the app."""

from __future__ import annotations

import math
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
    # --- how sure the decoder was; None when the engine did not say ---
    #: Mean log probability per token. Whisper's own measure of how confident
    #: the decode was; see ``confidence`` for the readable version.
    avg_logprob: float | None = None
    #: Probability the decoder assigned to "this is not speech at all". High
    #: values on a segment that still produced text are the signature of words
    #: invented over music, noise or silence.
    no_speech_prob: float | None = None
    #: gzip ratio of the segment text. A decode loop — the same phrase repeated
    #: until the window ends — compresses far better than speech does, which is
    #: what makes this worth keeping next to the text it describes.
    compression_ratio: float | None = None

    @property
    def confidence(self) -> float | None:
        """0..1, or None when the engine reported nothing.

        The geometric mean of the per-token probabilities, which is what
        ``avg_logprob`` is the log of. Reported this way because "0.42" is a
        number a person can act on and "-0.87" is not.
        """
        if self.avg_logprob is None:
            return None
        return float(math.exp(self.avg_logprob))


@dataclass
class TranscriptResult:
    recording: Recording
    segments: list[Segment] = field(default_factory=list)
    language: str = ""
    model: str = ""
    speakers: list[str] = field(default_factory=list)  # ordered unique speaker labels
    #: The voice vector diarization pooled for each speaker, keyed by the label
    #: shown in the transcript. Kept so that naming someone in the rename dialog
    #: can also teach the app their voice for next time. Empty for the cloud
    #: engines, which diarize without exposing anything to compare.
    speaker_embeddings: dict[str, list[float]] = field(default_factory=dict)
    #: How long each speaker spoke. Enrolling a voice from a few seconds would
    #: poison every later recording, so the dialog needs to know.
    speaker_seconds: dict[str, float] = field(default_factory=dict)

    @property
    def speaker_count(self) -> int:
        return len(self.speakers)
