# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""The output-options panel: formats, speakers, channels, line splitting,
output folder, and the filename-template builder launcher.

Applies to the current batch of selected recordings.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .. import ai_providers, config, denoise, vad, vocab_bias
from ..config import Settings
from ..transcriber import (
    ENGINE_ELEVENLABS,
    ENGINE_GEMINI,
    ENGINE_LABELS,
    ENGINE_LOCAL,
    faster_whisper_available,
)
from .glossary_dialog import GlossaryLibraryDialog, populate_glossary_combo
from .theme import muted_small

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
    #: Asks the main window to open Settings on a named tab ("" for wherever it
    #: was). The panel does not own that dialog, so it asks rather than opens.
    open_settings = Signal(str)

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
        tips = {
            ENGINE_LOCAL: "Runs on this PC. Speakers need pyannote + a HuggingFace token.",
            ENGINE_ELEVENLABS:
                "Uploads the audio to ElevenLabs. Transcribes and detects speakers in one pass.",
            ENGINE_GEMINI:
                "Uploads the audio to Google. Transcribes and separates speakers in one pass, "
                "using the same Google AI key as AI Cleanup.",
        }
        for row, engine_id in enumerate((ENGINE_LOCAL, ENGINE_ELEVENLABS, ENGINE_GEMINI)):
            self.engine.addItem(ENGINE_LABELS[engine_id], engine_id)
            self.engine.setItemData(row, tips[engine_id], Qt.ItemDataRole.ToolTipRole)
        idx = self.engine.findData(settings.stt_engine)
        self.engine.setCurrentIndex(idx if idx >= 0 else 0)
        self.engine.currentIndexChanged.connect(self._update_engine_status)
        self.engine.currentIndexChanged.connect(self._update_pipeline_status)
        ef.addRow("Engine:", self.engine)

        self.engine_status = QLabel("")
        self.engine_status.setWordWrap(True)
        self.engine_status.setStyleSheet(muted_small())
        ef.addRow(self.engine_status)
        ef.addRow(self._settings_link("Models, keys and devices…", "Engines"))
        root.addWidget(eng_box)

        # ---- Audio pipeline ----
        # Three layers in front of the decoder. On hard audio they matter more
        # than which Whisper model is picked, which is why they sit next to the
        # engine rather than buried in Settings.
        pipe_box = QGroupBox("Audio pipeline")
        pf = QFormLayout(pipe_box)
        pf.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.denoise_on = QCheckBox("Denoise before transcribing")
        self.denoise_on.setChecked(settings.denoise_enabled)
        self.denoise_on.setToolTip(
            "Cleans hiss, hum and room noise out of the audio first. The single "
            "biggest win on hard recordings."
        )
        self.denoise_on.toggled.connect(self._update_pipeline_controls)
        pf.addRow(self.denoise_on)

        self.vad_on = QCheckBox("Voice activity detection")
        self.vad_on.setChecked(settings.vad_enabled)
        self.vad_on.setToolTip(
            "Cuts silence and non-speech before the decoder sees it — faster, and "
            "far less prone to inventing words over noise."
        )
        self.vad_on.toggled.connect(self._update_pipeline_controls)
        pf.addRow(self.vad_on)

        self.bias_on = QCheckBox("Vocabulary biasing")
        self.bias_on.setChecked(settings.bias_enabled)
        self.bias_on.setToolTip(
            "Feeds names and jargon to the decoder so it spells them right. "
            "The terms themselves are in Settings → Audio."
        )
        self.bias_on.toggled.connect(self._update_pipeline_controls)
        pf.addRow(self.bias_on)

        self.pipeline_status = QLabel("")
        self.pipeline_status.setWordWrap(True)
        self.pipeline_status.setStyleSheet("color: gray;")
        pf.addRow(self.pipeline_status)
        pf.addRow(self._settings_link(
            "Denoising, VAD tuning and vocabulary…", "Audio"))
        root.addWidget(pipe_box)

        # ---- Speakers ----
        spk_box = QGroupBox("Speakers")
        sf = QFormLayout(spk_box)
        sf.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.include_speakers = QCheckBox("Label who is speaking")
        self.include_speakers.setChecked(settings.include_speakers)
        sf.addRow(self.include_speakers)

        # How many people are in this recording is a fact about this recording,
        # not a preference — and telling the diarizer is the single biggest
        # lever on how well it does. Zero leaves it to guess.
        count_row = QHBoxLayout()
        count_row.setContentsMargins(0, 0, 0, 0)
        self.min_speakers = QSpinBox()
        self.min_speakers.setRange(0, 20)
        self.min_speakers.setSpecialValueText("auto")
        self.min_speakers.setValue(settings.min_speakers)
        self.max_speakers = QSpinBox()
        self.max_speakers.setRange(0, 20)
        self.max_speakers.setSpecialValueText("auto")
        self.max_speakers.setValue(settings.max_speakers)
        for spin in (self.min_speakers, self.max_speakers):
            spin.setToolTip(
                "How many people are talking. Set both to the same number when "
                "you know it — guessing the count wrong is where most bad "
                "speaker labelling starts. Leave on auto to let the model decide."
            )
        count_row.addWidget(QLabel("at least"))
        count_row.addWidget(self.min_speakers)
        count_row.addWidget(QLabel("at most"))
        count_row.addWidget(self.max_speakers)
        count_row.addStretch()
        sf.addRow("How many:", self._wrap(count_row))

        sf.addRow(self._settings_link(
            "Speaker detection and channels…", "Speakers"))
        root.addWidget(spk_box)

        out_box = QGroupBox("Output")
        of = QFormLayout(out_box)
        of.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        formats = QHBoxLayout()
        formats.setContentsMargins(0, 0, 0, 0)
        self.format_checks: dict[str, QCheckBox] = {}
        for key, label in FORMAT_OPTIONS:
            cb = QCheckBox(label)
            cb.setChecked(key in settings.formats)
            self.format_checks[key] = cb
            formats.addWidget(cb)
        formats.addStretch()
        of.addRow("Formats:", self._wrap(formats))

        self.out_dir = QLineEdit(settings.output_dir)
        dir_row = QHBoxLayout()
        dir_row.setContentsMargins(0, 0, 0, 0)
        dir_row.addWidget(self.out_dir, stretch=1)
        browse = QPushButton("Browse…")
        browse.setAutoDefault(False)
        browse.clicked.connect(self._browse)
        dir_row.addWidget(browse)
        of.addRow("Folder:", self._wrap(dir_row))
        of.addRow(self._settings_link(
            "Line formatting and file names…", "Output"))
        root.addWidget(out_box)

        ai_box = QGroupBox("AI Cleanup")
        af = QFormLayout(ai_box)
        af.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.ai_cleanup_on = QCheckBox("Tidy the transcript with an AI model")
        self.ai_cleanup_on.setChecked(settings.ai_cleanup_enabled)
        self.ai_cleanup_on.toggled.connect(self._update_ai_controls)
        af.addRow(self.ai_cleanup_on)

        self.ai_provider = QComboBox()
        provider_row = QHBoxLayout()
        provider_row.setContentsMargins(0, 0, 0, 0)
        provider_row.addWidget(self.ai_provider, stretch=1)
        af.addRow("Provider:", self._wrap(provider_row))
        self.ai_provider.currentIndexChanged.connect(self._on_ai_provider_changed)

        self.ai_model = QComboBox()
        self.ai_model.setEditable(True)
        af.addRow("Model:", self.ai_model)

        self.glossary_on = QCheckBox("Build a glossary of names and terms")
        self.glossary_on.setChecked(settings.glossary_enabled)
        self.glossary_on.toggled.connect(self._update_ai_controls)
        af.addRow(self.glossary_on)

        self.shared_glossary = QComboBox()
        glossary_row = QHBoxLayout()
        glossary_row.setContentsMargins(0, 0, 0, 0)
        glossary_row.addWidget(self.shared_glossary, stretch=1)
        manage = QPushButton("Manage…")
        manage.setAutoDefault(False)
        manage.clicked.connect(self._manage_glossaries)
        glossary_row.addWidget(manage)
        af.addRow("Shared glossary:", self._wrap(glossary_row))
        populate_glossary_combo(self.shared_glossary, settings.glossary_shared_id)

        self.ai_status = QLabel("")
        self.ai_status.setWordWrap(True)
        self.ai_status.setStyleSheet("color: gray;")
        af.addRow(self.ai_status)
        af.addRow(self._settings_link(
            "Provider keys and glossary tuning…", "AI Cleanup"))
        root.addWidget(ai_box)

        # Everything else about cleanup — glossary model, temperature, chunking,
        # prompt caching — is tuning you set once, and lives in Settings.
        advanced = QPushButton("Advanced settings…")
        advanced.setAutoDefault(False)
        advanced.setToolTip(
            "VAD tuning, vocabulary terms, line formatting, file names and "
            "glossary tuning — the things you set once rather than per job."
        )
        advanced.clicked.connect(lambda: self.open_settings.emit(""))
        root.addWidget(advanced)

        root.addStretch()

        self._refresh_ai_providers()
        self._update_ai_controls()
        self._update_pipeline_controls()
        self._update_engine_status()

    # ------------------------------------------------------------------
    def _settings_link(self, text: str, tab: str) -> QLabel:
        """A quiet link to the Settings tab that holds the rest of this section.

        Sending people to a general "Advanced settings…" button means they
        arrive somewhere and still have to go looking.
        """
        label = QLabel(f'<a href="{tab}">{text}</a>')
        label.setStyleSheet(muted_small())
        label.setOpenExternalLinks(False)
        label.setAlignment(Qt.AlignmentFlag.AlignRight)
        label.linkActivated.connect(self.open_settings.emit)
        return label

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

    def _manage_glossaries(self):
        current = self.shared_glossary.currentData() or ""
        dlg = GlossaryLibraryDialog(self, selected=current)
        dlg.exec()
        # Whatever was left selected in the library is the obvious pick here.
        populate_glossary_combo(self.shared_glossary, dlg.selected_id() or current)

    # ---- engine -------------------------------------------------------
    def _update_engine_status(self, _index: int = 0):
        """Say what the current engine needs, before Go finds out the hard way."""
        if self.engine.currentData() == ENGINE_GEMINI:
            if self.s.ai_key_google.strip():
                self.engine_status.setText(
                    f"Audio is uploaded to Google ({self.s.gemini_model}). It transcribes "
                    "and separates speakers in one pass — no HuggingFace token or GPU "
                    "needed. Vocabulary biasing is unavailable on this engine: the API "
                    "refuses it alongside speakers and timestamps."
                )
            else:
                self.engine_status.setText(
                    "No Google AI key yet — add one in Settings (the same key AI Cleanup uses)."
                )
        elif self.engine.currentData() == ENGINE_ELEVENLABS:
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
        engine = self.engine.currentData()
        if engine == ENGINE_ELEVENLABS and not self.s.elevenlabs_api_key.strip():
            return (
                "ElevenLabs is selected as the transcription engine but no API key "
                "is saved. Add one in Settings, or switch back to local Whisper."
            )
        if engine == ENGINE_GEMINI and not self.s.ai_key_google.strip():
            return (
                "Gemini is selected as the transcription engine but no Google AI key "
                "is saved. Add one in Settings — it is the same key AI Cleanup uses — "
                "or switch back to local Whisper."
            )
        return None

    def _update_pipeline_controls(self, *_args):
        self._update_pipeline_status()

    def _update_pipeline_status(self, *_args):
        """One line per active layer, so the column says what will happen.

        The three separate status labels this replaces were most of the reason
        the panel needed scrolling.
        """
        draft = self._pipeline_draft()
        lines = [denoise.describe(draft), vad.describe(draft)]
        if self.engine.currentData() != ENGINE_LOCAL:
            lines.append(
                "Vocabulary biasing applies to the local engine only — the cloud "
                "engines do not take one."
            )
        elif not draft.bias_enabled:
            lines.append("Biasing off — the decoder gets no vocabulary hints.")
        else:
            terms = vocab_bias.collect_terms(draft)
            if terms:
                lines.append(vocab_bias.summarize(terms, vocab_bias.build(terms, draft.bias_max_chars)))
            else:
                # The link directly beneath already says where the terms live.
                lines.append("No vocabulary yet — pick a shared glossary below, or add terms.")
        self.pipeline_status.setText("\n".join(x for x in lines if x))

    def _pipeline_draft(self) -> Settings:
        """A copy of settings carrying what the panel currently shows.

        The status lines have to reflect the boxes as they are now, not as they
        were when the panel was built.
        """
        draft = Settings(**self.s.to_dict())
        # Only the switches live here now; the tuning behind them comes from
        # saved settings, where the Settings dialog put it.
        draft.denoise_enabled = self.denoise_on.isChecked()
        draft.vad_enabled = self.vad_on.isChecked()
        draft.bias_enabled = self.bias_on.isChecked()
        draft.glossary_shared_id = self.shared_glossary.currentData() or ""
        return draft

    def _update_ai_controls(self):
        enabled = self.ai_cleanup_on.isChecked()
        self.ai_provider.setEnabled(enabled)
        self.ai_model.setEnabled(enabled)
        self.glossary_on.setEnabled(enabled)
        # A shared glossary stays usable with extraction off: the curated terms
        # are still worth handing the cleanup model.
        self.shared_glossary.setEnabled(enabled)

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
        try:
            models = ai_providers.list_models(self.s, provider)
        except Exception as e:
            self.ai_model.clear()
            self.ai_model.addItem("(could not load models)", "")
            self.ai_status.setText(str(e))
            self.ai_model.setEnabled(self.ai_cleanup_on.isChecked())
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
        """Write back only what this panel owns.

        Everything else on Settings was set in the Settings dialog and must
        survive untouched — writing a stale widget value here would quietly
        undo it.
        """
        s.stt_engine = self.engine.currentData() or ENGINE_LOCAL
        s.formats = [k for k, cb in self.format_checks.items() if cb.isChecked()] or ["txt"]
        s.include_speakers = self.include_speakers.isChecked()
        s.min_speakers = self.min_speakers.value()
        s.max_speakers = self.max_speakers.value()
        s.output_dir = self.out_dir.text().strip() or s.output_dir
        s.ai_cleanup_enabled = self.ai_cleanup_on.isChecked()
        s.ai_cleanup_provider = self.ai_provider.currentData() or ""
        model = self.ai_model.currentData()
        s.ai_cleanup_model = model if model else ""
        s.glossary_enabled = self.glossary_on.isChecked()
        s.glossary_shared_id = self.shared_glossary.currentData() or ""
        s.denoise_enabled = self.denoise_on.isChecked()
        s.vad_enabled = self.vad_on.isChecked()
        s.bias_enabled = self.bias_on.isChecked()
        return s
