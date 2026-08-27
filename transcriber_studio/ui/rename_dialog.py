# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Rename detected speakers for a finished transcript, then re-export."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from ..models import TranscriptResult
from .theme import SheetDialog


class SpeakerRenameDialog(SheetDialog):
    def __init__(self, result: TranscriptResult, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Rename speakers — {result.recording.display_name}")
        self.setMinimumWidth(420)
        self.result = result
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Give each detected speaker a name. The transcript will be re-exported "
            "with the new names."
        ))
        form = QFormLayout()
        self.edits: dict[str, QLineEdit] = {}
        if not result.speakers:
            layout.addWidget(QLabel("No speakers were detected for this recording."))
        for spk in result.speakers:
            e = QLineEdit(spk)
            # Show a short sample line so the user can tell speakers apart.
            sample = next((s.text for s in result.segments if s.speaker == spk), "")
            e.setPlaceholderText(sample[:60])
            self.edits[spk] = e
            form.addRow(spk, e)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def renames(self) -> dict[str, str]:
        out = {}
        for original, edit in self.edits.items():
            new = edit.text().strip()
            if new and new != original:
                out[original] = new
        return out
