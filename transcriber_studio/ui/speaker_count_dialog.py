# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Ask how many speakers to look for before detecting them on a finished job.

The Options panel holds the defaults, and they persist. That is exactly how a
meeting of six people gets transcribed as two: "at most 2" was right for the
last recording and nobody changed it. So the limits for a re-run are asked for
here, for this run only, prefilled from the saved settings.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

from ..config import Settings
from ..models import TranscriptResult

MAX_SPEAKERS = 20


class SpeakerCountDialog(QDialog):
    def __init__(
        self, result: TranscriptResult, settings: Settings, parent=None,
        *, cleanup_applied: bool = False,
    ):
        super().__init__(parent)
        self.setWindowTitle("Detect speakers")
        layout = QVBoxLayout(self)

        intro = QLabel(self._intro(result, cleanup_applied))
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        self.min_speakers = QSpinBox()
        self.min_speakers.setRange(0, MAX_SPEAKERS)
        self.min_speakers.setSpecialValueText("auto")
        self.min_speakers.setValue(int(getattr(settings, "min_speakers", 0) or 0))
        self.max_speakers = QSpinBox()
        self.max_speakers.setRange(0, MAX_SPEAKERS)
        self.max_speakers.setSpecialValueText("auto")
        self.max_speakers.setValue(int(getattr(settings, "max_speakers", 0) or 0))
        row = QHBoxLayout()
        row.addWidget(QLabel("at least"))
        row.addWidget(self.min_speakers)
        row.addWidget(QLabel("at most"))
        row.addWidget(self.max_speakers)
        row.addStretch(1)
        form.addRow("How many:", row)
        layout.addLayout(form)

        note = QLabel(
            "These limits apply to this run only. The defaults live in the Options "
            "panel under Speakers. \"auto\" lets the model decide."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: palette(mid);")
        layout.addWidget(note)

        ok, cancel = QDialogButtonBox.StandardButton.Ok, QDialogButtonBox.StandardButton.Cancel
        buttons = QDialogButtonBox(ok | cancel)
        buttons.button(ok).setText("Detect")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _intro(result: TranscriptResult, cleanup_applied: bool) -> str:
        lines = []
        if result.speakers:
            n = len(result.speakers)
            lines.append(
                f"This transcript already has {n} speaker{'s' if n != 1 else ''}. "
                "Detecting again replaces those labels and any names you gave them."
            )
            if cleanup_applied:
                lines.append(
                    "The AI cleanup pass will be undone, because the turns it rewrote "
                    "are redrawn; run it again afterwards."
                )
        else:
            lines.append("Detect who spoke when, without transcribing again.")
        if result.words:
            lines.append("Turns are regrouped word by word from the original timings.")
        else:
            lines.append(
                "This transcript has no word timings, so they are recovered first by "
                "aligning its text to the audio (English), then the turns are regrouped "
                "word by word. Where a word cannot be aligned its time is estimated."
            )
        return " ".join(lines)

    def bounds(self) -> tuple[int, int]:
        """(min, max), 0 meaning auto. Never a maximum below the minimum."""
        lo, hi = self.min_speakers.value(), self.max_speakers.value()
        if lo and hi and hi < lo:
            hi = lo
        return lo, hi

    def accept(self) -> None:
        lo, hi = self.bounds()
        self.max_speakers.setValue(hi)
        super().accept()
