# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Rename detected speakers for a finished transcript, then re-export.

Naming a speaker here is also the natural moment to teach the app that voice:
the user has just listened enough to know who it is, and diarization has
already pooled a vector describing them. Ticking "remember" stores it, and the
next recording says "Alice" instead of "Speaker 2".
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from .. import voiceprints
from ..models import TranscriptResult
from .theme import SheetDialog


def _too_short_note(seconds: float) -> str:
    """Why a speaker cannot be remembered, in the reader's terms."""
    return (
        f"only {seconds:.0f}s of speech — "
        f"{voiceprints.MIN_ENROLL_SECONDS:.0f}s needed to remember a voice"
    )


class SpeakerRenameDialog(SheetDialog):
    def __init__(self, result: TranscriptResult, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Rename speakers — {result.recording.display_name}")
        self.setMinimumWidth(520)
        self.transcript = result
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Give each detected speaker a name. The transcript will be re-exported "
            "with the new names."
        ))
        form = QFormLayout()
        self.edits: dict[str, QLineEdit] = {}
        self.remember: dict[str, QCheckBox] = {}
        if not result.speakers:
            layout.addWidget(QLabel("No speakers were detected for this recording."))
        for spk in result.speakers:
            e = QLineEdit(spk)
            # Show a short sample line so the user can tell speakers apart.
            sample = next((s.text for s in result.segments if s.speaker == spk), "")
            e.setPlaceholderText(sample[:60])
            self.edits[spk] = e
            form.addRow(spk, self._row(spk, e))
        layout.addLayout(form)

        if any(cb.isEnabled() for cb in self.remember.values()):
            note = QLabel(
                "Remembering a voice lets this speaker be named automatically in "
                "later recordings. It is only ever a suggestion — a voice that is "
                "not a clear match is left as Speaker N."
            )
            note.setWordWrap(True)
            note.setStyleSheet("color: gray;")
            layout.addWidget(note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    def _row(self, speaker: str, edit: QLineEdit) -> QWidget:
        """The name field, plus the offer to remember this voice."""
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit, stretch=1)

        box = QCheckBox("Remember this voice")
        vector = self.transcript.speaker_embeddings.get(speaker)
        seconds = float(self.transcript.speaker_seconds.get(speaker, 0.0))

        if not vector:
            box.setEnabled(False)
            box.setToolTip(
                "This recording has no voice data for the speaker — cloud "
                "engines do not provide any, and neither do older results."
            )
        elif seconds < voiceprints.MIN_ENROLL_SECONDS:
            box.setEnabled(False)
            box.setToolTip(_too_short_note(seconds))
        else:
            # Whether this is a new person or another sample of a known one
            # depends on the name being typed, so the hint follows the field.
            self._refresh_hint(box, seconds, edit.text())
            edit.textChanged.connect(
                lambda text, b=box, s=seconds: self._refresh_hint(b, s, text)
            )
        self.remember[speaker] = box
        row.addWidget(box)
        return holder

    @staticmethod
    def _refresh_hint(box: QCheckBox, seconds: float, name: str) -> None:
        """What ticking this box will do, for the name currently typed."""
        known = voiceprints.get_profile(name.strip()) is not None if name.strip() else False
        box.setToolTip(
            f"{seconds:.0f}s of this speaker will be stored as a voiceprint."
            + (
                ""
                if seconds >= voiceprints.COMFORTABLE_ENROLL_SECONDS
                else "  A longer sample recognises more reliably."
            )
            + ("  Already remembered — this adds another sample." if known else "")
        )

    # ------------------------------------------------------------------
    def renames(self) -> dict[str, str]:
        out = {}
        for original, edit in self.edits.items():
            new = edit.text().strip()
            if new and new != original:
                out[original] = new
        return out

    def enrollments(self) -> list[tuple[str, str]]:
        """(name to store under, speaker label in this transcript) to remember.

        The name is whatever the user typed, so ticking the box without typing
        a name would enrol somebody as "Speaker 2" — useless, and confusing the
        first time it matched. Those are dropped.
        """
        out: list[tuple[str, str]] = []
        for speaker, box in self.remember.items():
            if not (box.isEnabled() and box.isChecked()):
                continue
            name = self.edits[speaker].text().strip()
            if not name or name == speaker:
                continue
            out.append((name, speaker))
        return out

    def apply_enrollments(self, log=None) -> list[str]:
        """Store the ticked voices. Returns what was remembered.

        Never raises: failing to remember a voice must not cost the rename the
        user actually came here to do.
        """
        remembered: list[str] = []
        for name, speaker in self.enrollments():
            vector = self.transcript.speaker_embeddings.get(speaker)
            if not vector:
                continue
            try:
                voiceprints.enroll(
                    name,
                    vector,
                    seconds=float(self.transcript.speaker_seconds.get(speaker, 0.0)),
                    source=self.transcript.recording.display_name,
                )
            except Exception as e:
                if log:
                    log(f"Could not remember {name}: {e}")
                continue
            remembered.append(name)
            if log:
                log(f"Remembered {name}'s voice — later recordings can name them.")
        return remembered
