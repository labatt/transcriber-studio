# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""The output-options panel: formats, speakers, channels, line splitting,
output folder, and the filename-template builder launcher.

Applies to the current batch of selected recordings.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .. import ai_providers, config, denoise, filename_builder, vad, vocab_bias
from ..config import Settings
from ..transcriber import ENGINE_ELEVENLABS, ENGINE_LABELS, ENGINE_LOCAL, faster_whisper_available
from .glossary_dialog import GlossaryLibraryDialog, populate_glossary_combo
from .template_dialog import TemplateDialog
from .theme import hint, muted_small

FORMAT_OPTIONS = [
    ("txt", "Text (.txt)"),
    ("srt", "Subtitles (.srt)"),
    ("vtt", "WebVTT (.vtt)"),
    ("json", "JSON (.json)"),
    ("md", "Markdown (.md)"),
]


def _narrow_combo(combo: QComboBox, min_chars: int = 12) -> QComboBox:
    combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    combo.setMinimumContentsLength(min_chars)
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    return combo


class OptionsPanel(QWidget):
    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.s = settings
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 12, 8)
        root.setSpacing(8)

        # ---- Engine ----
        # First thing in the panel because it decides where the audio goes:
        # local Whisper keeps it on this machine, ElevenLabs uploads it.
        eng_box = QGroupBox("Transcription engine")
        ef = QFormLayout(eng_box)
        ef.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        ef.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.engine = _narrow_combo(QComboBox())
        for engine_id in (ENGINE_LOCAL, ENGINE_ELEVENLABS):
            self.engine.addItem(ENGINE_LABELS[engine_id], engine_id)
        self.engine.setItemData(
            0, "Runs on this PC. Speakers need pyannote + a HuggingFace token.",
            Qt.ItemDataRole.ToolTipRole,
        )
        self.engine.setItemData(
            1, "Uploads the audio to ElevenLabs. Transcribes and detects speakers in one pass.",
            Qt.ItemDataRole.ToolTipRole,
        )
        idx = self.engine.findData(settings.stt_engine)
        self.engine.setCurrentIndex(idx if idx >= 0 else 0)
        self.engine.currentIndexChanged.connect(self._update_engine_status)
        ef.addRow("Engine:", self.engine)

        self.engine_status = QLabel("")
        self.engine_status.setWordWrap(True)
        self.engine_status.setStyleSheet(muted_small())
        ef.addRow(self.engine_status)
        root.addWidget(eng_box)

        # ---- Audio pipeline ----
        # Three layers in front of the decoder. On hard audio they matter more
        # than which Whisper model is picked, which is why they sit next to the
        # engine rather than buried in Settings.
        pipe_box = QGroupBox("Audio pipeline")
        pf = QFormLayout(pipe_box)
        pf.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        pf.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self.denoise_on = QCheckBox("1. Denoise before transcribing")
        self.denoise_on.setChecked(settings.denoise_enabled)
        self.denoise_on.setToolTip(
            "Noise suppression is the single biggest win on bad audio: the "
            "decoder stops guessing from its language model."
        )
        self.denoise_on.toggled.connect(self._update_pipeline_controls)
        pf.addRow(self.denoise_on)

        self.denoise_backend = _narrow_combo(QComboBox())
        for backend_id in denoise.ORDER:
            info = denoise.BACKENDS[backend_id]
            self.denoise_backend.addItem(info.label, backend_id)
            self.denoise_backend.setItemData(
                self.denoise_backend.count() - 1, info.detail, Qt.ItemDataRole.ToolTipRole
            )
        idx = self.denoise_backend.findData(settings.denoise_backend)
        self.denoise_backend.setCurrentIndex(idx if idx >= 0 else 0)
        self.denoise_backend.currentIndexChanged.connect(self._update_pipeline_status)
        pf.addRow("Denoiser:", self.denoise_backend)

        self.denoise_status = QLabel("")
        self.denoise_status.setWordWrap(True)
        self.denoise_status.setStyleSheet(muted_small())
        pf.addRow(self.denoise_status)

        self.vad_on = QCheckBox("2. Skip non-speech (voice activity detection)")
        self.vad_on.setChecked(settings.vad_enabled)
        self.vad_on.setToolTip(
            "Never showing the decoder silence or noise removes Whisper's worst "
            "failure mode: fluent text invented over nothing."
        )
        self.vad_on.toggled.connect(self._update_pipeline_controls)
        pf.addRow(self.vad_on)

        vad_row = QHBoxLayout()
        self.vad_threshold = QDoubleSpinBox()
        self.vad_threshold.setRange(0.05, 0.95)
        self.vad_threshold.setSingleStep(0.05)
        self.vad_threshold.setDecimals(2)
        self.vad_threshold.setValue(settings.vad_threshold)
        self.vad_threshold.setToolTip(
            "How sure the detector must be that a frame is speech. Raise it on "
            "a noisy room, lower it if quiet talking is being dropped."
        )
        self.vad_min_silence = QSpinBox()
        self.vad_min_silence.setRange(100, 10_000)
        self.vad_min_silence.setSingleStep(100)
        self.vad_min_silence.setValue(settings.vad_min_silence_ms)
        self.vad_min_silence.setToolTip("Silence this long (ms) splits the audio.")
        self.vad_pad = QSpinBox()
        self.vad_pad.setRange(0, 2_000)
        self.vad_pad.setSingleStep(50)
        self.vad_pad.setValue(settings.vad_speech_pad_ms)
        self.vad_pad.setToolTip(
            "Audio kept either side of each speech run (ms), so the first and "
            "last syllable are not clipped."
        )
        self.vad_min_speech = QSpinBox()
        self.vad_min_speech.setRange(0, 5_000)
        self.vad_min_speech.setSingleStep(50)
        self.vad_min_speech.setValue(settings.vad_min_speech_ms)
        self.vad_min_speech.setToolTip(
            "Speech runs shorter than this (ms) are discarded. Raise it to drop "
            "coughs and door clicks the detector counts as speech; 0 keeps "
            "everything it finds."
        )
        self.vad_max_speech = QDoubleSpinBox()
        self.vad_max_speech.setRange(0.0, 3_600.0)
        self.vad_max_speech.setSingleStep(30.0)
        self.vad_max_speech.setDecimals(0)
        self.vad_max_speech.setValue(settings.vad_max_speech_s)
        self.vad_max_speech.setToolTip(
            "Force a split after this many seconds of unbroken speech. 0 means "
            "no cap, which is right unless a monologue is running the decoder "
            "out of context."
        )
        for label, widget in (
            ("Threshold", self.vad_threshold),
            ("Silence ms", self.vad_min_silence),
            ("Pad ms", self.vad_pad),
        ):
            vad_row.addWidget(QLabel(label))
            vad_row.addWidget(widget)
        vad_row.addStretch()
        pf.addRow(self._wrap(vad_row))

        vad_row2 = QHBoxLayout()
        for label, widget in (
            ("Min speech ms", self.vad_min_speech),
            ("Max speech s", self.vad_max_speech),
        ):
            vad_row2.addWidget(QLabel(label))
            vad_row2.addWidget(widget)
        vad_row2.addStretch()
        pf.addRow(self._wrap(vad_row2))

        self.vad_status = QLabel("")
        self.vad_status.setWordWrap(True)
        self.vad_status.setStyleSheet(muted_small())
        pf.addRow(self.vad_status)

        self.bias_on = QCheckBox("3. Bias the decoder toward known vocabulary")
        self.bias_on.setChecked(settings.bias_enabled)
        self.bias_on.setToolTip(
            "Names, products and jargon from the shared glossary this job uses, "
            "fed to the decoder before it starts guessing at them."
        )
        self.bias_on.toggled.connect(self._update_pipeline_controls)
        pf.addRow(self.bias_on)

        self.bias_terms = QPlainTextEdit(settings.bias_extra_terms)
        self.bias_terms.setPlaceholderText(
            "Extra words to expect: names, companies, product names — one per "
            "line or comma separated"
        )
        self.bias_terms.setMaximumHeight(64)
        self.bias_terms.textChanged.connect(self._update_pipeline_status)
        pf.addRow("Extra vocabulary:", self.bias_terms)

        self.bias_budget = QSpinBox()
        self.bias_budget.setRange(0, 850)
        self.bias_budget.setSingleStep(50)
        self.bias_budget.setValue(settings.bias_max_chars)
        self.bias_budget.setToolTip(
            "How many characters of vocabulary to hand the decoder. Whisper "
            "truncates a longer prompt silently, so terms past the budget are "
            "dropped from the end of the list — which is why the most valuable "
            "ones go first."
        )
        self.bias_budget.valueChanged.connect(self._update_pipeline_status)
        pf.addRow("Vocabulary budget:", self.bias_budget)

        self.bias_status = QLabel("")
        self.bias_status.setWordWrap(True)
        self.bias_status.setStyleSheet(muted_small())
        pf.addRow(self.bias_status)

        self.hallucination_guard = QCheckBox("Guard against hallucinated passages")
        self.hallucination_guard.setChecked(settings.hallucination_guard)
        self.hallucination_guard.setToolTip(
            "Stops one invented passage seeding the next window, and drops "
            "segments the decoder produced over long silences."
        )
        pf.addRow(self.hallucination_guard)
        root.addWidget(pipe_box)

        # ---- Formats ----
        fmt_box = QGroupBox("Output formats")
        fl = QVBoxLayout(fmt_box)
        self.format_checks: dict[str, QCheckBox] = {}
        for key, label in FORMAT_OPTIONS:
            cb = QCheckBox(label)
            cb.setChecked(key in settings.formats)
            self.format_checks[key] = cb
            fl.addWidget(cb)
        root.addWidget(fmt_box)

        # ---- Speakers ----
        spk_box = QGroupBox("Speakers")
        sf = QFormLayout(spk_box)
        sf.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        sf.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.include_speakers = QCheckBox("Include speaker names in transcript")
        self.include_speakers.setChecked(settings.include_speakers)
        sf.addRow(self.include_speakers)

        self.channel_mode = _narrow_combo(QComboBox())
        self.channel_mode.addItem("Downmix to mono (use diarization)", "downmix")
        self.channel_mode.setItemData(0, "Downmix to mono (use diarization for speakers)", Qt.ItemDataRole.ToolTipRole)
        self.channel_mode.addItem("Per channel (channel = speaker)", "per_channel")
        self.channel_mode.setItemData(1, "Transcribe each channel separately (channel = speaker)", Qt.ItemDataRole.ToolTipRole)
        self.channel_mode.setCurrentIndex(0 if settings.channel_mode == "downmix" else 1)
        sf.addRow("Channels:", self.channel_mode)

        self.channel_names = QLineEdit(settings.channel_names)
        self.channel_names.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.channel_names.setPlaceholderText("Per-channel names, e.g. Agent,Customer")
        sf.addRow("Channel names:", self.channel_names)
        root.addWidget(spk_box)

        # ---- Lines / timestamps ----
        line_box = QGroupBox("Line formatting")
        lf = QFormLayout(line_box)
        lf.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        lf.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.line_mode = _narrow_combo(QComboBox())
        self.line_mode.addItem("One line per speaker turn", "segment")
        self.line_mode.addItem("One line per sentence", "sentence")
        self.line_mode.addItem("Wrap at N characters", "wrap")
        self.line_mode.setCurrentIndex(
            {"segment": 0, "sentence": 1, "wrap": 2}.get(settings.line_mode, 0)
        )
        lf.addRow("Split into lines:", self.line_mode)

        self.wrap_chars = QSpinBox(); self.wrap_chars.setRange(20, 300)
        self.wrap_chars.setValue(settings.wrap_chars)
        lf.addRow("Wrap width:", self.wrap_chars)

        self.include_ts = QCheckBox("Prefix each line with a timestamp")
        self.include_ts.setChecked(settings.include_timestamps)
        lf.addRow(self.include_ts)

        self.newline = _narrow_combo(QComboBox())
        self.newline.addItem("Windows (CRLF)", "crlf")
        self.newline.addItem("Unix (LF)", "lf")
        self.newline.setCurrentIndex(0 if settings.newline == "crlf" else 1)
        lf.addRow("Line endings:", self.newline)
        root.addWidget(line_box)

        # ---- Output destination + filename builder ----
        out_box = QGroupBox("Output")
        of = QFormLayout(out_box)
        of.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        of.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        dir_row = QHBoxLayout()
        self.out_dir = QLineEdit(settings.output_dir)
        self.out_dir.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        browse = QPushButton("Browse…"); browse.clicked.connect(self._browse)
        dir_row.addWidget(self.out_dir, stretch=1)
        dir_row.addWidget(browse)
        of.addRow("Folder:", self._wrap(dir_row))

        tmpl_row = QHBoxLayout()
        self.template = QLineEdit(settings.filename_template)
        self.template.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.template.textChanged.connect(self._update_preview)
        edit = QPushButton("Builder…"); edit.clicked.connect(self._open_builder)
        tmpl_row.addWidget(self.template, stretch=1)
        tmpl_row.addWidget(edit)
        of.addRow("Filename template:", self._wrap(tmpl_row))

        self.preview = QLabel()
        self.preview.setWordWrap(True)
        self.preview.setStyleSheet(hint())
        of.addRow("Example:", self.preview)

        self.overwrite = QCheckBox("Overwrite existing files (otherwise add a numbered suffix)")
        self.overwrite.setChecked(settings.overwrite)
        of.addRow(self.overwrite)

        self.sanitize = QCheckBox("Sanitize names for the filesystem")
        self.sanitize.setChecked(settings.sanitize_names)
        of.addRow(self.sanitize)

        self.owner_names = QLineEdit(settings.owner_names)
        self.owner_names.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.owner_names.setPlaceholderText("e.g. Alex, Alex R, Alex Rivera")
        self.owner_names.setToolTip(
            "Your own name as it appears as a speaker label, in every spelling "
            "you get labelled with. Once the other person on a recording has a "
            "real name, the file is named after them — this is how the app knows "
            "which speaker is you. Leave empty and the first named speaker is used."
        )
        of.addRow("Your name(s):", self.owner_names)
        root.addWidget(out_box)

        # ---- AI Cleanup ----
        ai_box = QGroupBox("AI Cleanup")
        af = QFormLayout(ai_box)
        af.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        af.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.ai_cleanup_on = QCheckBox("Run AI Cleanup after transcription")
        self.ai_cleanup_on.setChecked(settings.ai_cleanup_enabled)
        self.ai_cleanup_on.toggled.connect(self._update_ai_controls)
        af.addRow(self.ai_cleanup_on)

        provider_row = QHBoxLayout()
        self.ai_provider = _narrow_combo(QComboBox())
        self.ai_provider.currentIndexChanged.connect(self._on_ai_provider_changed)
        self.ai_refresh_models = QPushButton("Refresh models")
        self.ai_refresh_models.clicked.connect(self._refresh_ai_models)
        provider_row.addWidget(self.ai_provider, stretch=1)
        provider_row.addWidget(self.ai_refresh_models)
        af.addRow("Provider:", self._wrap(provider_row))

        self.ai_model = _narrow_combo(QComboBox())
        self.ai_model.setEditable(False)
        af.addRow("Model:", self.ai_model)

        self.glossary_on = QCheckBox("Extract glossary before cleanup (speaker roster + terms)")
        self.glossary_on.setChecked(settings.glossary_enabled)
        self.glossary_on.toggled.connect(self._update_ai_controls)
        af.addRow(self.glossary_on)

        glossary_row = QHBoxLayout()
        self.shared_glossary = _narrow_combo(QComboBox())
        self.shared_glossary.currentIndexChanged.connect(self._update_pipeline_status)
        self.shared_glossary.setToolTip(
            "A glossary shared by several jobs: this run reads its terms before "
            "cleanup and writes back the ones its own transcript turned up."
        )
        populate_glossary_combo(self.shared_glossary, settings.glossary_shared_id)
        self.manage_glossaries = QPushButton("Manage…")
        self.manage_glossaries.setAutoDefault(False)
        self.manage_glossaries.clicked.connect(self._manage_glossaries)
        glossary_row.addWidget(self.shared_glossary, stretch=1)
        glossary_row.addWidget(self.manage_glossaries)
        af.addRow("Shared glossary:", self._wrap(glossary_row))

        self.glossary_model = QLineEdit(settings.glossary_model)
        self.glossary_model.setPlaceholderText("Leave empty to use cleanup model")
        self.glossary_model.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        af.addRow("Glossary model:", self.glossary_model)

        self.glossary_temperature = QDoubleSpinBox()
        self.glossary_temperature.setRange(0.0, 1.0)
        self.glossary_temperature.setSingleStep(0.1)
        self.glossary_temperature.setDecimals(1)
        self.glossary_temperature.setValue(settings.glossary_temperature)
        af.addRow("Glossary temperature:", self.glossary_temperature)

        self.glossary_chunk_threshold = QSpinBox()
        self.glossary_chunk_threshold.setRange(1_000, 500_000)
        self.glossary_chunk_threshold.setSingleStep(1_000)
        self.glossary_chunk_threshold.setValue(settings.glossary_chunk_token_threshold)
        self.glossary_chunk_threshold.setToolTip(
            "Transcripts larger than this (estimated tokens) are split for glossary extraction."
        )
        af.addRow("Glossary chunk threshold:", self.glossary_chunk_threshold)

        self.force_reextract = QCheckBox("Force re-extract glossary (ignore saved .glossary.json)")
        self.force_reextract.setChecked(settings.force_reextract)
        af.addRow(self.force_reextract)

        self.prompt_cache_on = QCheckBox("Use prompt caching when supported (Anthropic, OpenAI, etc.)")
        self.prompt_cache_on.setChecked(settings.prompt_cache_enabled)
        af.addRow(self.prompt_cache_on)

        self.ai_status = QLabel("")
        self.ai_status.setWordWrap(True)
        self.ai_status.setStyleSheet(muted_small())
        af.addRow(self.ai_status)
        root.addWidget(ai_box)

        self._glossary_controls = (
            self.glossary_model,
            self.glossary_temperature,
            self.glossary_chunk_threshold,
            self.force_reextract,
        )

        root.addStretch()

        self._refresh_ai_providers()
        self._update_ai_controls()
        self._update_pipeline_controls()
        self._update_engine_status()
        self._update_preview()

    # ------------------------------------------------------------------
    @staticmethod
    def _wrap(layout):
        w = QWidget()
        w.setLayout(layout)
        w.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        return w

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Choose output folder", self.out_dir.text())
        if d:
            self.out_dir.setText(d)

    def _open_builder(self):
        dlg = TemplateDialog(self.template.text(), self.sanitize.isChecked(), self)
        if dlg.exec():
            self.template.setText(dlg.template())

    def _manage_glossaries(self):
        current = self.shared_glossary.currentData() or ""
        dlg = GlossaryLibraryDialog(self, selected=current)
        dlg.exec()
        # Whatever was left selected in the library is the obvious pick here.
        populate_glossary_combo(self.shared_glossary, dlg.selected_id() or current)

    def _update_preview(self):
        stem = filename_builder.render(
            self.template.text(), filename_builder.sample_values(), self.sanitize.isChecked()
        )
        self.preview.setText(f"{stem}.txt")

    # ---- engine -------------------------------------------------------
    def _update_engine_status(self, _index: int = 0):
        """Say what the current engine needs, before Go finds out the hard way."""
        if self.engine.currentData() == ENGINE_ELEVENLABS:
            model = self.s.elevenlabs_model or "scribe_v1"
            if self.s.elevenlabs_api_key.strip():
                self.engine_status.setText(
                    f"Audio is uploaded to ElevenLabs ({model}). Speaker detection "
                    "comes from Scribe — no HuggingFace token or GPU needed."
                )
            else:
                self.engine_status.setText(
                    "No ElevenLabs API key yet — add one in Settings before running."
                )
        elif faster_whisper_available():
            self.engine_status.setText(
                "Runs on this PC; nothing leaves the machine. Speakers come from "
                "pyannote (HuggingFace token in Settings)."
            )
        else:
            self.engine_status.setText(
                "faster-whisper is not installed — run: pip install faster-whisper"
            )

    def refresh_engine_status(self):
        """Call after Settings saves an ElevenLabs key or model."""
        self._update_engine_status()

    def ensure_engine_ready(self) -> str | None:
        """Validate the engine before a run, the way AI Cleanup is validated."""
        if self.engine.currentData() != ENGINE_ELEVENLABS:
            return None
        if not self.s.elevenlabs_api_key.strip():
            return (
                "ElevenLabs is selected as the transcription engine but no API key "
                "is saved. Add one in Settings, or switch back to local Whisper."
            )
        return None

    def _update_pipeline_controls(self, *_args):
        denoise_on = self.denoise_on.isChecked()
        self.denoise_backend.setEnabled(denoise_on)
        vad_on = self.vad_on.isChecked()
        for widget in (
            self.vad_threshold, self.vad_min_silence, self.vad_pad,
            self.vad_min_speech, self.vad_max_speech,
        ):
            widget.setEnabled(vad_on)
        bias_on = self.bias_on.isChecked()
        self.bias_terms.setEnabled(bias_on)
        self.bias_budget.setEnabled(bias_on)
        self._update_pipeline_status()

    def _update_pipeline_status(self, *_args):
        """Say what each layer will actually do, from the machine's point of view."""
        draft = self._pipeline_draft()
        self.denoise_status.setText(denoise.describe(draft))
        self.vad_status.setText(vad.describe(draft))
        terms = vocab_bias.collect_terms(draft)
        prompt = vocab_bias.build(terms, draft.bias_max_chars)
        if not draft.bias_enabled:
            self.bias_status.setText("Biasing off — the decoder gets no vocabulary hints.")
        elif not terms:
            self.bias_status.setText(
                "No vocabulary yet: pick a shared glossary above, or type terms here. "
                "A glossary fills itself in as jobs run."
            )
        else:
            self.bias_status.setText(vocab_bias.summarize(terms, prompt))

    def _pipeline_draft(self) -> Settings:
        """A copy of settings carrying what the panel currently shows.

        The status lines have to reflect the boxes as they are now, not as they
        were when the panel was built.
        """
        draft = Settings(**self.s.to_dict())
        draft.denoise_enabled = self.denoise_on.isChecked()
        draft.denoise_backend = self.denoise_backend.currentData() or denoise.AUTO
        draft.vad_enabled = self.vad_on.isChecked()
        draft.vad_threshold = self.vad_threshold.value()
        draft.vad_min_silence_ms = self.vad_min_silence.value()
        draft.vad_speech_pad_ms = self.vad_pad.value()
        draft.vad_min_speech_ms = self.vad_min_speech.value()
        draft.vad_max_speech_s = self.vad_max_speech.value()
        draft.bias_max_chars = self.bias_budget.value()
        draft.bias_enabled = self.bias_on.isChecked()
        draft.bias_extra_terms = self.bias_terms.toPlainText()
        draft.glossary_shared_id = self.shared_glossary.currentData() or ""
        return draft

    def _update_ai_controls(self):
        enabled = self.ai_cleanup_on.isChecked()
        self.ai_provider.setEnabled(enabled)
        self.ai_model.setEnabled(enabled)
        self.ai_refresh_models.setEnabled(enabled)
        self.glossary_on.setEnabled(enabled)
        self.prompt_cache_on.setEnabled(enabled)
        # A shared glossary stays usable with extraction off: the curated terms
        # are still worth handing the cleanup model.
        self.shared_glossary.setEnabled(enabled)
        self.manage_glossaries.setEnabled(enabled)
        glossary_on = enabled and self.glossary_on.isChecked()
        for control in self._glossary_controls:
            control.setEnabled(glossary_on)

    def refresh_ai_providers(self):
        """Call after Settings saves new API keys or a new default model."""
        self._refresh_ai_providers()

    def refresh_glossaries(self):
        """Call after the shared-glossary library has been edited elsewhere."""
        populate_glossary_combo(
            self.shared_glossary, self.shared_glossary.currentData() or ""
        )
        self._update_pipeline_status()   # the glossary IS the biasing vocabulary

    def _refresh_ai_providers(self):
        current = self.ai_provider.currentData()
        self.ai_provider.blockSignals(True)
        self.ai_provider.clear()
        providers = ai_providers.configured_providers(self.s)
        if not providers:
            self.ai_provider.addItem("(add API keys in Settings)", "")
        else:
            for pid in providers:
                self.ai_provider.addItem(ai_providers.PROVIDER_LABELS.get(pid, pid), pid)
        wanted = current or config.cleanup_provider(self.s)
        if wanted:
            idx = self.ai_provider.findData(wanted)
            if idx < 0 and current:
                idx = self.ai_provider.findData(config.cleanup_provider(self.s))
            if idx >= 0:
                self.ai_provider.setCurrentIndex(idx)
        self.ai_provider.blockSignals(False)
        self._refresh_ai_models()

    def _on_ai_provider_changed(self, _index: int = 0):
        self._refresh_ai_models()

    def _refresh_ai_models(self):
        provider = self.ai_provider.currentData()
        if not provider:
            self.ai_model.clear()
            self.ai_status.setText("Add provider API keys in Settings to enable AI Cleanup.")
            return
        saved_model = config.cleanup_model(self.s)
        self.ai_model.clear()
        self.ai_model.addItem("Loading…", "")
        self.ai_status.setText("Fetching models from provider…")
        self.ai_model.setEnabled(False)
        self.ai_refresh_models.setEnabled(False)
        try:
            models = ai_providers.list_models(self.s, provider)
        except Exception as e:
            self.ai_model.clear()
            self.ai_model.addItem("(could not load models)", "")
            self.ai_status.setText(str(e))
            self.ai_model.setEnabled(self.ai_cleanup_on.isChecked())
            self.ai_refresh_models.setEnabled(self.ai_cleanup_on.isChecked())
            return
        self.ai_model.clear()
        if not models:
            self.ai_model.addItem("(no models returned)", "")
            self.ai_status.setText("Provider returned no models.")
        else:
            for m in models:
                self.ai_model.addItem(m, m)
            for candidate in (saved_model, self.s.ai_default_model):
                if not candidate:
                    continue
                idx = self.ai_model.findData(candidate)
                if idx >= 0:
                    self.ai_model.setCurrentIndex(idx)
                    break
            self.ai_status.setText(f"{len(models)} model(s) available.")
        self.ai_model.setEnabled(self.ai_cleanup_on.isChecked())
        self.ai_refresh_models.setEnabled(self.ai_cleanup_on.isChecked())

    def ensure_ai_cleanup_ready(self) -> str | None:
        """Validate provider/model before a new transcription run."""
        if not self.ai_cleanup_on.isChecked():
            return None
        return self.ensure_ai_cleanup_config()

    def ensure_ai_cleanup_config(self) -> str | None:
        """Validate provider/model (for manual or automatic cleanup)."""
        self._refresh_ai_models()
        provider = self.ai_provider.currentData()
        model = self.ai_model.currentData()
        if not provider:
            return "No AI provider is configured. Add API keys in Settings."
        if not model or str(model).startswith("("):
            return "No AI model is selected. Choose a model or click Refresh models."
        return None

    # ------------------------------------------------------------------
    def apply_to(self, s: Settings) -> Settings:
        s.stt_engine = self.engine.currentData() or ENGINE_LOCAL
        s.formats = [k for k, cb in self.format_checks.items() if cb.isChecked()] or ["txt"]
        s.include_speakers = self.include_speakers.isChecked()
        s.channel_mode = self.channel_mode.currentData()
        s.channel_names = self.channel_names.text().strip()
        s.line_mode = self.line_mode.currentData()
        s.wrap_chars = self.wrap_chars.value()
        s.include_timestamps = self.include_ts.isChecked()
        s.newline = self.newline.currentData()
        s.output_dir = self.out_dir.text().strip() or s.output_dir
        s.filename_template = self.template.text().strip() or "{date}_{name}"
        s.overwrite = self.overwrite.isChecked()
        s.sanitize_names = self.sanitize.isChecked()
        s.owner_names = self.owner_names.text().strip()
        s.ai_cleanup_enabled = self.ai_cleanup_on.isChecked()
        s.ai_cleanup_provider = self.ai_provider.currentData() or ""
        model = self.ai_model.currentData()
        s.ai_cleanup_model = model if model else ""
        s.glossary_enabled = self.glossary_on.isChecked()
        s.glossary_model = self.glossary_model.text().strip()
        s.glossary_temperature = self.glossary_temperature.value()
        s.glossary_chunk_token_threshold = self.glossary_chunk_threshold.value()
        s.force_reextract = self.force_reextract.isChecked()
        s.glossary_shared_id = self.shared_glossary.currentData() or ""
        s.prompt_cache_enabled = self.prompt_cache_on.isChecked()
        s.denoise_enabled = self.denoise_on.isChecked()
        s.denoise_backend = self.denoise_backend.currentData() or denoise.AUTO
        s.vad_enabled = self.vad_on.isChecked()
        s.vad_threshold = self.vad_threshold.value()
        s.vad_min_silence_ms = self.vad_min_silence.value()
        s.vad_speech_pad_ms = self.vad_pad.value()
        s.vad_min_speech_ms = self.vad_min_speech.value()
        s.vad_max_speech_s = self.vad_max_speech.value()
        s.bias_max_chars = self.bias_budget.value()
        s.bias_enabled = self.bias_on.isChecked()
        s.bias_extra_terms = self.bias_terms.toPlainText().strip()
        s.hallucination_guard = self.hallucination_guard.isChecked()
        return s
