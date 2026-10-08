# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""What one run should do, decided at the moment Go is pressed.

The Options panel holds defaults, and defaults persist: "at most 2 speakers"
set for last week's call is still set for today's six-person meeting. A plan
is the set of choices for *this* run only. It is applied to a copy of the
settings, so nothing here is ever saved back.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .config import Settings


@dataclass
class RunPlan:
    #: Detect the speakers first, play samples of each, and let the user name
    #: them before a word is transcribed. The names go on the transcript and,
    #: where asked, into the voiceprint store.
    identify_speakers: bool = False
    #: Stop after the naming dialog: no transcription, nothing written. For
    #: teaching the app a room full of voices without spending an engine run
    #: on it. Implies identify_speakers.
    identify_only: bool = False
    #: How many people are in the recording, 0 for "let the model decide".
    #: Passed as both the lower and upper bound, because a person who has just
    #: been asked knows the answer exactly or not at all.
    speakers: int = 0
    denoise: bool = False
    detect_speakers: bool = True
    ai_cleanup: bool = False
    #: Cut every recording to its first N minutes before anything sees it.
    #: 0 is the whole recording. For a cheap look at how an engine or a
    #: glossary does on a long meeting.
    limit_minutes: int = 0
    #: What to call the transcript files, in place of the usual naming: a
    #: template ("{date}_{name}") or a plain name used exactly as typed. Empty
    #: is the usual naming. Several recordings under one plain name get the
    #: " (2)", " (3)" suffixes rather than overwriting each other.
    transcript_name: str = ""
    #: Keep a copy of each PLAUD recording's audio, in audio_dir.
    save_audio: bool = False
    audio_dir: str = ""

    @classmethod
    def from_settings(cls, settings: Settings) -> RunPlan:
        """The plan the Options panel implies, as the dialog's starting point."""
        bounds = (settings.min_speakers, settings.max_speakers)
        return cls(
            identify_speakers=False,
            speakers=bounds[0] if bounds[0] and bounds[0] == bounds[1] else 0,
            denoise=settings.denoise_enabled,
            detect_speakers=settings.diarization_enabled,
            ai_cleanup=settings.ai_cleanup_enabled,
        )

    def apply(self, settings: Settings) -> Settings:
        """A copy of the settings with this run's choices on it.

        The speaker count is only written when one was given; "auto" leaves
        the panel's own bounds alone rather than erasing them.
        """
        run = replace(
            settings,
            denoise_enabled=self.denoise,
            diarization_enabled=self.detect_speakers or self.identify_speakers or self.identify_only,
            ai_cleanup_enabled=self.ai_cleanup and not self.identify_only,
            limit_minutes=max(0, int(self.limit_minutes)),
            filename_override=(self.transcript_name or "").strip(),
            save_audio_dir=(self.audio_dir or "").strip() if self.save_audio else "",
        )
        if self.speakers:
            run.min_speakers = self.speakers
            run.max_speakers = self.speakers
        return run

    @property
    def bounds(self) -> tuple[int, int]:
        return (self.speakers, self.speakers) if self.speakers else (0, 0)

    def describe(self) -> str:
        """One line for the log, so a run's choices are on the record."""
        parts = []
        if self.limit_minutes:
            parts.append(f"first {self.limit_minutes} minutes only")
        who = f"{self.speakers} people" if self.speakers else "speaker count left to the model"
        if self.save_audio and (self.audio_dir or "").strip():
            parts.append(f"save the audio to {self.audio_dir.strip()}")
        if self.identify_only:
            parts.append(f"identify speakers only, no transcription ({who})")
            parts.append("denoise" if self.denoise else "no denoise")
            return ", ".join(parts)
        if self.identify_speakers:
            parts.append(f"identify speakers first ({who})")
        parts.append("denoise" if self.denoise else "no denoise")
        parts.append("detect speakers" if self.detect_speakers else "no speaker detection")
        parts.append("AI cleanup" if self.ai_cleanup else "no AI cleanup")
        if (self.transcript_name or "").strip():
            parts.append(f"transcript named {self.transcript_name.strip()!r}")
        return ", ".join(parts)
