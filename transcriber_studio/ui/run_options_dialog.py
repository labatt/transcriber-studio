# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Choose what this run does before it starts.

The Options panel is where defaults live. This is where one run departs from
them: identify the speakers by ear first, say how many people were in the
room, skip the denoiser, skip the AI pass. Nothing chosen here is saved.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

from .. import config, diarization
from ..run_plan import RunPlan
from ..transcriber import ENGINE_GEMINI, ENGINE_LABELS
from .speaker_count_dialog import MAX_SPEAKERS
from .theme import SheetDialog, muted_small

#: What the "only the first…" box proposes. Ten minutes is enough to judge an
#: engine or a glossary on and cheap on every cloud engine.
DEFAULT_LIMIT_MINUTES = 10


class RunOptionsDialog(SheetDialog):
    def __init__(self, settings, recordings, parent=None):
        super().__init__(parent)
        self.s = settings
        self.setWindowTitle("Go with options")
        self.setMinimumWidth(520)
        plan = RunPlan.from_settings(settings)

        layout = QVBoxLayout(self)
        count = len(recordings)
        engine = ENGINE_LABELS.get(settings.stt_engine, settings.stt_engine)
        intro = QLabel(
            f"{count} recording{'s' if count != 1 else ''} with {engine}. "
            "These choices apply to this run only; the Options panel keeps its defaults."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()

        self.identify = QCheckBox("Identify speakers first")
        self.identify.setToolTip(
            "Detect the speakers before transcribing, play a few samples of each, "
            "and let you name them. The names go on the transcript, and ticked "
            "voices are remembered for later recordings."
        )
        can_identify, why_not = self._can_identify()
        self.identify.setEnabled(can_identify)
        self.identify.setChecked(False)
        form.addRow(self.identify)
        if not can_identify:
            note = QLabel(why_not)
            note.setWordWrap(True)
            note.setStyleSheet(muted_small())
            form.addRow(note)

        self.identify_only = QCheckBox("…and stop there — do not transcribe")
        self.identify_only.setToolTip(
            "Detect the speakers, play the samples, remember the names, and finish. "
            "Nothing is transcribed and no files are written. Names you type are "
            "kept for this session, so a later Go on the same recording uses them."
        )
        self.identify_only.setEnabled(False)
        self.identify_only.setChecked(False)
        only_row = QHBoxLayout()
        only_row.addSpacing(24)
        only_row.addWidget(self.identify_only)
        only_row.addStretch()
        form.addRow(only_row)

        self.speakers = QSpinBox()
        self.speakers.setRange(0, MAX_SPEAKERS)
        self.speakers.setSpecialValueText("let the model decide")
        self.speakers.setValue(plan.speakers)
        self.speakers.setToolTip(
            "How many people are in the recording. Telling the model the exact "
            "count is the single biggest help you can give it."
        )
        count_row = QHBoxLayout()
        count_row.addWidget(self.speakers)
        count_row.addStretch()
        form.addRow("People in the recording:", count_row)

        self.limit = QCheckBox("Only the first")
        self.limit.setToolTip(
            "Cut every recording to its first few minutes before anything sees it: "
            "a cheap way to try an engine, a glossary or the denoiser on a long "
            "meeting. The cut is a stream copy, so the audio is otherwise untouched."
        )
        self.limit_minutes = QSpinBox()
        self.limit_minutes.setRange(1, 600)
        self.limit_minutes.setValue(DEFAULT_LIMIT_MINUTES)
        self.limit_minutes.setSuffix(" min")
        self.limit_minutes.setEnabled(False)
        self.limit.toggled.connect(self.limit_minutes.setEnabled)
        limit_row = QHBoxLayout()
        limit_row.addWidget(self.limit)
        limit_row.addWidget(self.limit_minutes)
        limit_row.addWidget(QLabel("of each recording"))
        limit_row.addStretch()
        form.addRow(limit_row)

        self.denoise = QCheckBox("Denoise before transcribing")
        self.denoise.setChecked(plan.denoise)
        form.addRow(self.denoise)

        self.detect = QCheckBox("Detect speakers")
        self.detect.setChecked(plan.detect_speakers)
        form.addRow(self.detect)

        self.cleanup = QCheckBox("AI cleanup afterwards")
        self.cleanup.setChecked(plan.ai_cleanup)
        has_model = bool(config.cleanup_provider(settings) and config.cleanup_model(settings))
        if not has_model:
            self.cleanup.setEnabled(False)
            self.cleanup.setChecked(False)
            self.cleanup.setToolTip("Pick a cleanup provider and model in the Options panel first.")
        form.addRow(self.cleanup)
        layout.addLayout(form)

        self._cleanup_allowed = has_model
        self.identify.toggled.connect(self._on_identify_toggled)
        self.identify_only.toggled.connect(self._on_identify_toggled)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("▶  Go")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._buttons = buttons
        self._on_identify_toggled()

    def _can_identify(self) -> tuple[bool, str]:
        """Identification needs pyannote here, whatever engine does the words."""
        if self.s.stt_engine == ENGINE_GEMINI:
            return False, (
                "Identifying speakers is not available with Gemini: it separates "
                "speakers itself and the local speaker stage never runs."
            )
        if not (self.s.hf_token or "").strip():
            return False, "Identifying speakers needs a HuggingFace token (Settings → Speakers)."
        if not diarization.is_available():
            return False, "Identifying speakers needs pyannote.audio installed (Setup)."
        return True, ""

    def _on_identify_toggled(self, *_args) -> None:
        # Identifying speakers is speaker detection with a listen in the
        # middle, so the plain detection box is implied and locked on. Stopping
        # there makes the transcription-side choices moot, so they are greyed.
        on = self.identify.isChecked()
        if on:
            self.detect.setChecked(True)
        self.detect.setEnabled(not on)
        self.identify_only.setEnabled(on)
        if not on:
            self.identify_only.setChecked(False)
        only = on and self.identify_only.isChecked()
        self.cleanup.setEnabled(self._cleanup_allowed and not only)
        ok = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok.setText("▶  Identify" if only else "▶  Go")

    def plan(self) -> RunPlan:
        identify = self.identify.isEnabled() and self.identify.isChecked()
        return RunPlan(
            identify_speakers=identify,
            identify_only=identify and self.identify_only.isChecked(),
            speakers=int(self.speakers.value()),
            denoise=self.denoise.isChecked(),
            detect_speakers=self.detect.isChecked(),
            ai_cleanup=self.cleanup.isEnabled() and self.cleanup.isChecked(),
            limit_minutes=int(self.limit_minutes.value()) if self.limit.isChecked() else 0,
        )
