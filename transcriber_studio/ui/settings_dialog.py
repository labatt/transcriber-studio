# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Engine / diarization / account settings — all stored locally."""

from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .. import ai_providers, denoise, diarization, stt_elevenlabs, stt_gemini, whisper_models
from ..config import Settings
from ..hardware import (
    CUDA_TORCH_INSTALL_CMD,
    cuda_available,
    cuda_device_name,
    diarization_device_label,
    torch_cuda_available,
)
from .theme import SheetDialog, WrappedNote, muted

LANGS = ["auto", "en", "es", "fr", "de", "it", "pt", "nl", "ja", "zh", "ko", "ru", "ar", "hi"]


class _ProviderTestWorker(QThread):
    ok = Signal(str)
    failed = Signal(str)

    def __init__(self, settings: Settings, provider: str, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.provider = provider

    def run(self):
        try:
            self.ok.emit(ai_providers.test_provider(self.settings, self.provider))
        except Exception as e:
            self.failed.emit(str(e))


class _ElevenLabsTestWorker(QThread):
    ok = Signal(str)
    failed = Signal(str)

    def __init__(self, api_key: str, parent=None):
        super().__init__(parent)
        self.api_key = api_key

    def run(self):
        try:
            self.ok.emit(stt_elevenlabs.test_key(self.api_key))
        except Exception as e:
            self.failed.emit(str(e))


class _ModelListWorker(QThread):
    ok = Signal(list)
    failed = Signal(str)

    def __init__(self, settings: Settings, provider: str, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.provider = provider

    def run(self):
        try:
            self.ok.emit(ai_providers.list_models(self.settings, self.provider))
        except Exception as e:
            self.failed.emit(str(e))


class SettingsDialog(SheetDialog):
    def __init__(self, settings: Settings, parent=None, tab: str = ""):
        super().__init__(parent)
        self.s = settings
        self._open_on_tab = tab
        self._test_worker: _ProviderTestWorker | None = None
        self._el_worker: _ElevenLabsTestWorker | None = None
        self._model_worker: _ModelListWorker | None = None
        self._test_buttons: list[QPushButton] = []
        self.setWindowTitle("Settings")
        self.setMinimumWidth(640)
        self.setMinimumHeight(520)

        outer = QVBoxLayout(self)
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs)

        # One scrolling page per tab. Everything used to be a single column of
        # nine group boxes, which meant hunting for a checkbox by scrolling.
        engines = self._tab("Engines")
        audio = self._tab("Audio")
        speakers = self._tab("Speakers")
        output = self._tab("Output")
        ai_tab = self._tab("AI Cleanup")
        plaud_tab = self._tab("Plaud")

        # --- ElevenLabs Scribe ---
        # The engine itself is picked per run in the Options panel; what lives
        # here is the account it needs.
        el = QGroupBox("ElevenLabs Scribe (cloud engine — key stored locally only)")
        elf = QFormLayout(el)
        self.el_key = self._key_field(settings.elevenlabs_api_key)
        self.el_key.setPlaceholderText("sk_… (stored locally only)")
        el_row = QHBoxLayout()
        el_row.setContentsMargins(0, 0, 0, 0)
        el_row.addWidget(self.el_key, stretch=1)
        self.el_test = QPushButton("Test")
        self.el_test.setFixedWidth(56)
        self.el_test.clicked.connect(self._test_elevenlabs)
        el_row.addWidget(self.el_test)
        elf.addRow("API key:", self._wrap(el_row))

        self.el_model = QComboBox()
        self.el_model.addItems(stt_elevenlabs.MODELS)
        self.el_model.setCurrentText(settings.elevenlabs_model or stt_elevenlabs.DEFAULT_MODEL)
        elf.addRow("Scribe model:", self.el_model)

        self.el_audio_events = QCheckBox("Tag non-speech audio events (laughter, applause…)")
        self.el_audio_events.setChecked(settings.elevenlabs_tag_audio_events)
        elf.addRow(self.el_audio_events)

        el_note = QLabel(
            "Choose this engine in the Options panel (\"Transcription engine\"). "
            "Scribe transcribes and detects speakers in one pass, so the pyannote "
            "settings below do not apply to it — the Speakers options and the "
            "speaker-count maximum still do. Audio is uploaded to ElevenLabs and "
            "billed to your ElevenLabs account."
        )
        el_note.setWordWrap(True)
        el_note.setStyleSheet("color: gray;")
        elf.addRow(el_note)
        engines.addWidget(el)

        # --- Gemini transcription ---
        # The key lives in the AI providers group below: it is the same Google
        # account, and asking for it twice would be a trap.
        gem = QGroupBox("Gemini 3.5 Transcribe (cloud engine)")
        gemf = QFormLayout(gem)
        self.gemini_model = QComboBox()
        self.gemini_model.addItems(stt_gemini.MODELS)
        self.gemini_model.setCurrentText(settings.gemini_model or stt_gemini.DEFAULT_MODEL)
        gemf.addRow("Model:", self.gemini_model)

        self.gemini_mode = QComboBox()
        for mode in stt_gemini.MODES:
            self.gemini_mode.addItem(stt_gemini.MODE_LABELS[mode], mode)
        idx = self.gemini_mode.findData(settings.gemini_mode or stt_gemini.DEFAULT_MODE)
        self.gemini_mode.setCurrentIndex(idx if idx >= 0 else 0)
        self.gemini_mode.setToolTip(
            "Verbatim is the one that gives you speakers and timestamps — Google's "
            "API refuses both in smart mode, which returns one block of punctuated "
            "prose. Verbatim keeps the fillers; AI Cleanup is where tidying belongs."
        )
        gemf.addRow("Transcription mode:", self.gemini_mode)

        gem_row = QHBoxLayout()
        gem_row.setContentsMargins(0, 0, 0, 0)
        self.gemini_test = QPushButton("Test key and model access")
        self.gemini_test.clicked.connect(self._test_gemini)
        gem_row.addWidget(self.gemini_test)
        gem_row.addStretch()
        gemf.addRow(self._wrap(gem_row))

        gem_note = WrappedNote(
            "Choose this engine in the Options panel. It uses the Google AI key from "
            "the AI providers section below — the same one AI Cleanup uses. It "
            "transcribes and separates speakers in one pass, so the pyannote settings "
            "do not apply to it. Google documents a 60 minute limit per request, or 30 "
            "with speaker separation. Vocabulary biasing is not available on this "
            "engine: the API refuses a custom vocabulary alongside speakers or "
            "timestamps."
        )
        gem_note.setStyleSheet("color: gray;")
        gemf.addRow(gem_note)
        engines.addWidget(gem)

        # --- Whisper engine ---
        eng = QGroupBox("Whisper engine (local — faster-whisper)")
        f = QFormLayout(eng)
        self.model = QComboBox()
        pick = whisper_models.recommended(cuda_available())
        for model_id in whisper_models.ORDER:
            name = whisper_models.label(model_id)
            self.model.addItem(
                f"{name}  (recommended)" if model_id == pick else name, model_id
            )
        idx = self.model.findData(settings.model)
        self.model.setCurrentIndex(idx if idx >= 0 else self.model.findData(pick))
        self.model.currentIndexChanged.connect(self._update_model_note)
        f.addRow("Model:", self.model)

        # Same guidance as the setup wizard: what each size costs in VRAM,
        # speed and accuracy on this particular machine. It spans the whole row
        # rather than sitting in the field column: the fine-tunes carry several
        # lines of caveats, and a narrow column turns those into a clipped
        # paragraph and a horizontal scrollbar.
        self.model_note = WrappedNote()
        self.model_note.setStyleSheet(muted())
        f.addRow(self.model_note)

        self.device = QComboBox(); self.device.addItems(["auto", "cuda", "cpu"])
        self.device.setCurrentText(settings.device)
        f.addRow("Device:", self.device)

        self.compute = QComboBox()
        self.compute.addItems(["auto", "float16", "int8_float16", "int8", "float32"])
        self.compute.setCurrentText(settings.compute_type)
        f.addRow("Compute type:", self.compute)

        self.lang = QComboBox(); self.lang.addItems(LANGS)
        self.lang.setCurrentText(settings.language)
        f.addRow("Language:", self.lang)

        name = cuda_device_name()
        if cuda_available():
            gpu = f"GPU detected ✓ ({name})"
            diar_where = diarization_device_label()
            gpu += f"\nWhisper: GPU · Diarization: {diar_where}"
            if not torch_cuda_available():
                gpu += (
                    "\n\nTo speed up speaker detection on your NVIDIA GPU, reinstall "
                    "PyTorch with CUDA (all three packages together):\n"
                    f"  {CUDA_TORCH_INSTALL_CMD}\n"
                    "Then restart the app. Whisper will keep using the GPU."
                )
        else:
            gpu = "No CUDA GPU detected — will use CPU"
        # Wrapped, and spanning the row: the CUDA install command in this text
        # is one long line, and a plain label would set the dialog's width by it.
        f.addRow(WrappedNote(gpu))
        engines.addWidget(eng)

        # --- Diarization ---
        diar = QGroupBox("Speaker diarization (pyannote.audio)")
        df = QFormLayout(diar)
        self.diar_on = QCheckBox("Enable speaker detection / names")
        self.diar_on.setChecked(settings.diarization_enabled)
        df.addRow(self.diar_on)

        self.hf = QLineEdit(settings.hf_token)
        self.hf.setEchoMode(QLineEdit.EchoMode.Password)
        self.hf.setPlaceholderText("hf_… (stored locally only)")
        df.addRow("HuggingFace token:", self.hf)

        status = "installed ✓" if diarization.is_available() else "NOT installed — run: pip install pyannote.audio"
        lbl = WrappedNote(
            f"Status: pyannote {status}.\n"
            "Before speaker detection works, sign in at huggingface.co and click "
            "\"Agree and access\" on ALL three model pages:\n"
            "  • huggingface.co/pyannote/speaker-diarization-community-1\n"
            "  • huggingface.co/pyannote/segmentation-3.0\n"
            "  • huggingface.co/pyannote/speaker-diarization-3.1\n"
            "Then create a Read token (Settings → Access Tokens) and paste it above."
        )
        lbl.setStyleSheet("color: gray;")
        df.addRow(lbl)
        speakers.addWidget(diar)

        # --- Audio front-end ---
        # Where the denoiser lives on this machine. Whether it runs at all is a
        # per-run choice in the Options panel.
        dn = QGroupBox("Audio front-end (denoising)")
        dnf = QFormLayout(dn)
        df_row = QHBoxLayout()
        df_row.setContentsMargins(0, 0, 0, 0)
        self.deep_filter_path = QLineEdit(settings.deep_filter_path)
        self.deep_filter_path.setPlaceholderText(
            "Leave empty to use deep-filter from PATH"
        )
        self.deep_filter_path.textChanged.connect(self._update_denoise_status)
        browse_df = QPushButton("Browse…")
        browse_df.setFixedWidth(80)
        browse_df.clicked.connect(self._browse_deep_filter)
        df_row.addWidget(self.deep_filter_path, stretch=1)
        df_row.addWidget(browse_df)
        dnf.addRow("deep-filter binary:", self._wrap(df_row))

        self.denoise_model_path = QLineEdit(settings.denoise_model_path)
        self.denoise_model_path.setPlaceholderText(
            "Optional: a DeepFilterNet model .tar.gz to use instead of the built-in one"
        )
        dnf.addRow("Model override:", self.denoise_model_path)

        self.denoise_postfilter = QCheckBox(
            "Post-filter (attenuates more noise, at some risk to quiet speech)"
        )
        self.denoise_postfilter.setChecked(settings.denoise_postfilter)
        dnf.addRow(self.denoise_postfilter)

        self.denoise_atten = QSpinBox()
        self.denoise_atten.setRange(0, 100)
        self.denoise_atten.setSuffix(" dB")
        self.denoise_atten.setValue(settings.denoise_atten_lim_db)
        self.denoise_atten.setToolTip(
            "How much noise DeepFilterNet may remove. 100 dB is full suppression "
            "(its own default). Lower it if heavy suppression starts eating "
            "consonants — a smeared consonant costs more accuracy than the hiss "
            "it removed."
        )
        dnf.addRow("Noise reduction limit:", self.denoise_atten)

        cache_row = QHBoxLayout()
        cache_row.setContentsMargins(0, 0, 0, 0)
        self.denoise_cache_label = QLabel("")
        self.denoise_cache_label.setStyleSheet("color: gray;")
        clear_cache = QPushButton("Clear")
        clear_cache.setFixedWidth(80)
        clear_cache.clicked.connect(self._clear_denoise_cache)
        cache_row.addWidget(self.denoise_cache_label, stretch=1)
        cache_row.addWidget(clear_cache)
        dnf.addRow("Enhanced audio cache:", self._wrap(cache_row))

        self.denoise_status = QLabel("")
        self.denoise_status.setWordWrap(True)
        self.denoise_status.setStyleSheet("color: gray;")
        dnf.addRow(self.denoise_status)
        audio.addWidget(dn)

        # --- Plaud cloud ---
        plaud = QGroupBox("Plaud cloud")
        pf = QFormLayout(plaud)
        self.plaud_page_size = QSpinBox()
        self.plaud_page_size.setRange(10, 200)
        self.plaud_page_size.setValue(settings.plaud_page_size)
        self.plaud_page_size.setToolTip(
            "How many recordings to fetch per page when browsing your Plaud account."
        )
        pf.addRow("Recordings per page:", self.plaud_page_size)
        plaud_note = QLabel(
            "Used on the Plaud Recordings tab when you refresh or page through your library."
        )
        plaud_note.setWordWrap(True)
        plaud_note.setStyleSheet("color: gray;")
        pf.addRow(plaud_note)
        plaud_tab.addWidget(plaud)

        # --- AI provider keys ---
        ai = QGroupBox("AI providers (for AI Cleanup — keys stored locally only)")
        af = QFormLayout(ai)
        self.ai_openrouter = self._key_field(settings.ai_key_openrouter)
        af.addRow("OpenRouter:", self._provider_row(self.ai_openrouter, "openrouter"))
        self.ai_openai = self._key_field(settings.ai_key_openai)
        af.addRow("OpenAI:", self._provider_row(self.ai_openai, "openai"))
        self.ai_anthropic = self._key_field(settings.ai_key_anthropic)
        af.addRow("Anthropic:", self._provider_row(self.ai_anthropic, "anthropic"))
        self.ai_google = self._key_field(settings.ai_key_google)
        af.addRow("Google (Gemini):", self._provider_row(self.ai_google, "google"))
        self.ai_grok = self._key_field(settings.ai_key_grok)
        af.addRow("Grok (xAI):", self._provider_row(self.ai_grok, "grok"))
        self.ai_ollama_cloud = self._key_field(settings.ai_key_ollama_cloud)
        af.addRow("Ollama Cloud:", self._provider_row(self.ai_ollama_cloud, "ollama_cloud"))
        self.ollama_local_url = QLineEdit(settings.ollama_local_url)
        self.ollama_local_url.setPlaceholderText("http://localhost:11434")
        af.addRow("Ollama local URL:", self._provider_row(self.ollama_local_url, "ollama_local", is_url=True))
        ai_note = QLabel(
            "Add API keys for any providers you want to use. Click Test to verify access "
            "before saving. Ollama local needs no key — just a running server at the URL above."
        )
        ai_note.setWordWrap(True)
        ai_note.setStyleSheet("color: gray;")
        af.addRow(ai_note)
        ai_tab.addWidget(ai)

        # --- AI cleanup defaults ---
        # The provider keys above say what the app *can* use; this says what it
        # reaches for by default, so a model chosen once is not re-picked on
        # every job. A per-run pick in the Options panel or the AI Cleanup
        # dialog still wins for that run.
        dflt = QGroupBox("AI Cleanup defaults")
        dfl = QFormLayout(dflt)
        self.default_provider = QComboBox()
        self.default_provider.addItem("(none — pick per job)", "")
        for pid in ai_providers.PROVIDERS:
            self.default_provider.addItem(ai_providers.PROVIDER_LABELS.get(pid, pid), pid)
        idx = self.default_provider.findData(settings.ai_default_provider)
        self.default_provider.setCurrentIndex(idx if idx >= 0 else 0)
        self.default_provider.currentIndexChanged.connect(self._on_default_provider_changed)
        dfl.addRow("Default provider:", self.default_provider)

        model_row = QHBoxLayout()
        model_row.setContentsMargins(0, 0, 0, 0)
        self.default_model = QComboBox()
        self.default_model.setEditable(True)   # a model the list does not return still works
        self.default_model.setMinimumWidth(280)
        self._reset_default_model_items()
        self.load_models_btn = QPushButton("Load models")
        self.load_models_btn.setAutoDefault(False)
        self.load_models_btn.clicked.connect(self._load_default_models)
        model_row.addWidget(self.default_model, stretch=1)
        model_row.addWidget(self.load_models_btn)
        dfl.addRow("Default model:", self._wrap(model_row))

        self.default_model_status = QLabel(
            "Used whenever a run has no model of its own. Load models to pick "
            "from the provider's list, or type an id."
        )
        self.default_model_status.setWordWrap(True)
        self.default_model_status.setStyleSheet("color: gray;")
        dfl.addRow(self.default_model_status)
        ai_tab.addWidget(dflt)

        # --- VAD, biasing, guard: the tuning that used to crowd the Options
        # column. Per-run on/off switches stay there; the numbers live here.
        vad = QGroupBox("Voice activity detection (tuning)")
        vf = QFormLayout(vad)
        self.vad_threshold = QDoubleSpinBox()
        self.vad_threshold.setRange(0.05, 0.95)
        self.vad_threshold.setSingleStep(0.05)
        self.vad_threshold.setValue(settings.vad_threshold)
        self.vad_threshold.setToolTip(
            "How loud a sound must be to count as speech. Raise it if breathing "
            "and keyboard noise are being transcribed; lower it if quiet talking "
            "goes missing."
        )
        vf.addRow("Speech threshold:", self.vad_threshold)
        self.vad_min_silence = QSpinBox()
        self.vad_min_silence.setRange(0, 10000)
        self.vad_min_silence.setSingleStep(50)
        self.vad_min_silence.setValue(settings.vad_min_silence_ms)
        vf.addRow("Minimum silence (ms):", self.vad_min_silence)
        self.vad_pad = QSpinBox()
        self.vad_pad.setRange(0, 2000)
        self.vad_pad.setSingleStep(50)
        self.vad_pad.setValue(settings.vad_speech_pad_ms)
        self.vad_pad.setToolTip("Kept either side of speech, so words are not clipped.")
        vf.addRow("Speech padding (ms):", self.vad_pad)
        self.vad_min_speech = QSpinBox()
        self.vad_min_speech.setRange(0, 5000)
        self.vad_min_speech.setSingleStep(50)
        self.vad_min_speech.setValue(settings.vad_min_speech_ms)
        vf.addRow("Minimum speech (ms):", self.vad_min_speech)
        self.vad_max_speech = QDoubleSpinBox()
        # Zero means no cap, so it has to be reachable — a minimum of 1.0 would
        # quietly turn "never split" into "split every second".
        self.vad_max_speech.setRange(0.0, 3600.0)
        self.vad_max_speech.setSingleStep(5.0)
        self.vad_max_speech.setSpecialValueText("no cap")
        self.vad_max_speech.setValue(settings.vad_max_speech_s)
        vf.addRow("Maximum speech (s):", self.vad_max_speech)
        audio.addWidget(vad)

        bias = QGroupBox("Vocabulary biasing")
        bf = QFormLayout(bias)
        self.bias_terms = QPlainTextEdit(settings.bias_extra_terms)
        self.bias_terms.setPlaceholderText("Names, products, jargon — one per line or comma separated")
        self.bias_terms.setMaximumHeight(90)
        self.bias_terms.setToolTip(
            "Fed to the decoder as hotwords, on top of anything the glossary "
            "already knows. Not available on the Gemini engine."
        )
        bf.addRow("Extra vocabulary:", self.bias_terms)
        self.bias_budget = QSpinBox()
        self.bias_budget.setRange(0, 4000)
        self.bias_budget.setSingleStep(50)
        self.bias_budget.setValue(settings.bias_max_chars)
        self.bias_budget.setToolTip(
            "Characters of vocabulary sent to the decoder. Too many and the "
            "model starts hearing them everywhere."
        )
        bf.addRow("Vocabulary budget:", self.bias_budget)
        self.hallucination_guard = QCheckBox(
            "Hallucination guard — do not carry context between windows"
        )
        self.hallucination_guard.setChecked(settings.hallucination_guard)
        self.hallucination_guard.setToolTip(
            "Stops one invented passage seeding the next. Worth having on noisy "
            "audio; costs a little continuity on clean audio."
        )
        bf.addRow(self.hallucination_guard)
        audio.addWidget(bias)
        audio.addStretch()

        # --- channels belong with speakers, not with output formatting ---
        chan = QGroupBox("Channels")
        cf = QFormLayout(chan)
        self.channel_mode = QComboBox()
        for label, data in (("Downmix to mono", "downmix"),
                            ("Transcribe each channel separately", "per_channel")):
            self.channel_mode.addItem(label, data)
        index = self.channel_mode.findData(settings.channel_mode)
        self.channel_mode.setCurrentIndex(index if index >= 0 else 0)
        self.channel_mode.setToolTip(
            "Per-channel suits a recorder that gave each speaker their own track; "
            "speakers then come from the channels and diarization is not needed."
        )
        cf.addRow("Channels:", self.channel_mode)
        self.channel_names = QLineEdit(settings.channel_names)
        self.channel_names.setPlaceholderText("e.g. Interviewer, Guest")
        cf.addRow("Channel names:", self.channel_names)
        speakers.addWidget(chan)
        speakers.addStretch()

        # --- everything about how the file reads and what it is called ---
        lines = QGroupBox("Line formatting")
        lf = QFormLayout(lines)
        self.line_mode = QComboBox()
        # "segment" is the stored value for one-line-per-turn; the formatter
        # knows these three strings and nothing else.
        for label, data in (("One line per speaker turn", "segment"),
                            ("One line per sentence", "sentence"),
                            ("Wrap at N characters", "wrap")):
            self.line_mode.addItem(label, data)
        index = self.line_mode.findData(settings.line_mode)
        self.line_mode.setCurrentIndex(index if index >= 0 else 0)
        lf.addRow("Split into lines:", self.line_mode)
        self.wrap_chars = QSpinBox()
        self.wrap_chars.setRange(20, 300)
        self.wrap_chars.setValue(settings.wrap_chars)
        lf.addRow("Wrap width:", self.wrap_chars)
        self.include_ts = QCheckBox("Include timestamps in text output")
        self.include_ts.setChecked(settings.include_timestamps)
        lf.addRow(self.include_ts)
        self.newline = QComboBox()
        for label, data in (("Windows (CRLF)", "crlf"), ("Unix (LF)", "lf")):
            self.newline.addItem(label, data)
        index = self.newline.findData(settings.newline)
        self.newline.setCurrentIndex(index if index >= 0 else 0)
        lf.addRow("Line endings:", self.newline)
        output.addWidget(lines)

        names = QGroupBox("File names")
        nf = QFormLayout(names)
        self.template = QLineEdit(settings.filename_template)
        tmpl_row = QHBoxLayout()
        tmpl_row.setContentsMargins(0, 0, 0, 0)
        tmpl_row.addWidget(self.template, stretch=1)
        builder = QPushButton("Build…")
        builder.setAutoDefault(False)
        builder.clicked.connect(self._open_builder)
        tmpl_row.addWidget(builder)
        nf.addRow("Filename template:", self._wrap(tmpl_row))
        self.preview = QLabel("")
        self.preview.setStyleSheet("color: gray;")
        nf.addRow("Example:", self.preview)
        self.overwrite = QCheckBox("Overwrite files that already exist")
        self.overwrite.setChecked(settings.overwrite)
        nf.addRow(self.overwrite)
        self.sanitize = QCheckBox("Replace characters Windows will not accept")
        self.sanitize.setChecked(settings.sanitize_names)
        nf.addRow(self.sanitize)
        self.owner_names = QLineEdit(settings.owner_names)
        self.owner_names.setPlaceholderText("e.g. Chris, Chris L.")
        self.owner_names.setToolTip(
            "Used to leave your own name out of {name}, so a file is named after "
            "the other person in the conversation."
        )
        nf.addRow("Your name(s):", self.owner_names)
        output.addWidget(names)
        output.addStretch()
        self.template.textChanged.connect(self._update_preview)
        self.sanitize.toggled.connect(self._update_preview)

        # --- glossary tuning; which glossary a job uses stays in Options ---
        gloss = QGroupBox("Glossary extraction (tuning)")
        gf = QFormLayout(gloss)
        self.glossary_model = QLineEdit(settings.glossary_model)
        self.glossary_model.setPlaceholderText("blank = same model as cleanup")
        gf.addRow("Glossary model:", self.glossary_model)
        self.glossary_temperature = QDoubleSpinBox()
        self.glossary_temperature.setRange(0.0, 2.0)
        self.glossary_temperature.setSingleStep(0.1)
        self.glossary_temperature.setValue(settings.glossary_temperature)
        gf.addRow("Glossary temperature:", self.glossary_temperature)
        self.glossary_chunk_threshold = QSpinBox()
        self.glossary_chunk_threshold.setRange(1000, 500000)
        self.glossary_chunk_threshold.setSingleStep(1000)
        self.glossary_chunk_threshold.setValue(settings.glossary_chunk_token_threshold)
        self.glossary_chunk_threshold.setToolTip(
            "Transcripts longer than this are extracted in chunks."
        )
        gf.addRow("Chunk threshold (tokens):", self.glossary_chunk_threshold)
        self.force_reextract = QCheckBox("Always re-extract, ignoring a saved glossary")
        self.force_reextract.setChecked(settings.force_reextract)
        gf.addRow(self.force_reextract)
        self.prompt_cache_on = QCheckBox("Use prompt caching where the provider supports it")
        self.prompt_cache_on.setChecked(settings.prompt_cache_enabled)
        self.prompt_cache_on.setToolTip("Cheaper repeat runs; no effect on the result.")
        gf.addRow(self.prompt_cache_on)
        ai_tab.addWidget(gloss)
        ai_tab.addStretch()

        engines.addStretch()
        plaud_tab.addStretch()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        self.components_btn = QPushButton("Components && updates…")
        self.components_btn.setToolTip(
            "What is installed, what is newer, and buttons to update it."
        )
        self.components_btn.setAutoDefault(False)
        self.components_btn.clicked.connect(self._open_components)
        buttons.addButton(self.components_btn, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        self._update_model_note()
        self._update_denoise_status()
        self._update_preview()
        self.select_tab(tab)

    def select_tab(self, title: str) -> bool:
        """Open on a named tab. Unknown or empty names leave it on the first."""
        if not title:
            return False
        for index in range(self.tabs.count()):
            if self.tabs.tabText(index) == title:
                self.tabs.setCurrentIndex(index)
                return True
        return False

    def _tab(self, title: str) -> QVBoxLayout:
        """A scrolling page in the dialog. Returns the layout to fill."""
        page = QWidget()
        layout = QVBoxLayout(page)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        # The notes wrap; nothing here should ever scroll sideways.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(page)
        self.tabs.addTab(scroll, title)
        return layout

    def _open_builder(self):
        from .options_panel import TemplateDialog

        dlg = TemplateDialog(self.template.text(), self.sanitize.isChecked(), self)
        if dlg.exec():
            self.template.setText(dlg.template())

    def _update_preview(self):
        from .. import filename_builder

        stem = filename_builder.render(
            self.template.text(), filename_builder.sample_values(), self.sanitize.isChecked()
        )
        self.preview.setText(f"{stem}.txt")

    def _reset_default_model_items(self):
        """Show the saved default even before any model list is fetched."""
        self.default_model.clear()
        saved = self.s.ai_default_model
        if saved:
            self.default_model.addItem(saved, saved)
        self.default_model.setCurrentText(saved)

    def _on_default_provider_changed(self, _index: int = 0):
        self.default_model.clear()
        self.default_model.setCurrentText("")
        provider = self.default_provider.currentData()
        if provider:
            self.default_model_status.setText(
                f"Click Load models to list "
                f"{ai_providers.PROVIDER_LABELS.get(provider, provider)} models."
            )
        else:
            self.default_model_status.setText(
                "No default — every run picks its own provider and model."
            )

    def _load_default_models(self):
        provider = self.default_provider.currentData()
        if not provider:
            QMessageBox.warning(self, "AI Cleanup defaults", "Pick a default provider first.")
            return
        if self._model_worker and self._model_worker.isRunning():
            return
        self.load_models_btn.setEnabled(False)
        self.default_model_status.setText("Fetching models…")
        self._model_worker = _ModelListWorker(self._draft_settings(), provider, self)
        self._model_worker.ok.connect(self._on_default_models_loaded)
        self._model_worker.failed.connect(self._on_default_models_failed)
        self._model_worker.finished.connect(
            lambda: self.load_models_btn.setEnabled(True)
        )
        self._model_worker.start()

    def _on_default_models_loaded(self, models: list[str]):
        wanted = self.default_model.currentText().strip() or self.s.ai_default_model
        self.default_model.clear()
        for model in models:
            self.default_model.addItem(model, model)
        if wanted:
            idx = self.default_model.findData(wanted)
            if idx >= 0:
                self.default_model.setCurrentIndex(idx)
            else:
                self.default_model.setCurrentText(wanted)
        self.default_model_status.setText(f"{len(models)} model(s) available.")

    def _on_default_models_failed(self, message: str):
        self.default_model_status.setText(message)

    def _open_components(self):
        from .install_help import InstallHelpDialog

        InstallHelpDialog(self, settings=self._draft_settings()).exec()
        self._update_denoise_status()   # they may have just installed the binary

    def _browse_deep_filter(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Locate the deep-filter binary", self.deep_filter_path.text(),
            "Programs (*.exe);;All files (*)",
        )
        if path:
            self.deep_filter_path.setText(path)

    def _clear_denoise_cache(self):
        denoise.clear_cache()
        self._update_denoise_status()

    def _update_denoise_status(self, *_args):
        found = denoise.binary_path(self.deep_filter_path.text())
        if found:
            self.denoise_status.setText(f"DeepFilterNet binary found: {found}")
        else:
            self.denoise_status.setText(
                "No deep-filter binary yet — denoising falls back to ffmpeg's own "
                "filter, which is weaker on background chatter. The binary is a "
                "single download; see Setup → what still needs installing."
            )
        megabytes = denoise.cache_size_bytes() / (1024 * 1024)
        self.denoise_cache_label.setText(
            f"{megabytes:.0f} MB of enhanced audio kept for re-runs"
        )

    def _update_model_note(self, _index: int = 0):
        model_id = self.model.currentData() or whisper_models.GPU_RECOMMENDED
        self.model_note.setText(whisper_models.describe(model_id, cuda_available()))

    def _provider_row(self, field: QLineEdit, provider: str, *, is_url: bool = False) -> QWidget:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        if is_url:
            field.setEchoMode(QLineEdit.EchoMode.Normal)
        row.addWidget(field, stretch=1)
        test_btn = QPushButton("Test")
        test_btn.setFixedWidth(56)
        test_btn.clicked.connect(lambda _checked=False, p=provider: self._test_provider(p))
        self._test_buttons.append(test_btn)
        row.addWidget(test_btn)
        return self._wrap(row)

    def _draft_settings(self) -> Settings:
        data = self.s.to_dict()
        data.update({
            "ai_key_openrouter": self.ai_openrouter.text().strip(),
            "ai_key_openai": self.ai_openai.text().strip(),
            "ai_key_anthropic": self.ai_anthropic.text().strip(),
            "ai_key_google": self.ai_google.text().strip(),
            "ai_key_grok": self.ai_grok.text().strip(),
            "ai_key_ollama_cloud": self.ai_ollama_cloud.text().strip(),
            "ollama_local_url": self.ollama_local_url.text().strip() or "http://localhost:11434",
            # So a path typed but not yet saved still counts when the components
            # window goes looking for the binary.
            "deep_filter_path": self.deep_filter_path.text().strip(),
            "denoise_model_path": self.denoise_model_path.text().strip(),
        })
        known = {f for f in Settings().to_dict()}
        return Settings(**{k: v for k, v in data.items() if k in known})

    def _set_test_buttons_enabled(self, enabled: bool):
        for btn in self._test_buttons:
            btn.setEnabled(enabled)

    def _test_provider(self, provider: str):
        if self._test_worker and self._test_worker.isRunning():
            QMessageBox.information(self, "Test provider", "A provider test is already running.")
            return
        label = ai_providers.PROVIDER_LABELS.get(provider, provider)
        draft = self._draft_settings()
        if provider == "ollama_local":
            if not draft.ollama_local_url.strip():
                QMessageBox.warning(self, "Test provider", "Enter an Ollama local URL first.")
                return
        else:
            key_field = ai_providers.PROVIDERS[provider].key_field
            if key_field and not getattr(draft, key_field, "").strip():
                QMessageBox.warning(self, "Test provider", f"Enter an API key for {label} first.")
                return
        self._set_test_buttons_enabled(False)
        self._test_worker = _ProviderTestWorker(draft, provider, self)
        self._test_worker.ok.connect(self._on_provider_test_ok)
        self._test_worker.failed.connect(self._on_provider_test_failed)
        self._test_worker.finished.connect(lambda: self._set_test_buttons_enabled(True))
        self._test_worker.start()

    def _test_gemini(self):
        key = self._draft_settings().ai_key_google.strip()
        if not key:
            QMessageBox.warning(
                self, "Test Gemini",
                "Add a Google AI key in the AI providers section below first.",
            )
            return
        self.gemini_test.setEnabled(False)
        try:
            QMessageBox.information(self, "Gemini", stt_gemini.test_key(key))
        except Exception as exc:                       # noqa: BLE001 - shown to the user
            QMessageBox.warning(self, "Gemini test failed", str(exc))
        finally:
            self.gemini_test.setEnabled(True)

    def _test_elevenlabs(self):
        if self._el_worker and self._el_worker.isRunning():
            QMessageBox.information(self, "Test ElevenLabs", "A test is already running.")
            return
        key = self.el_key.text().strip()
        if not key:
            QMessageBox.warning(self, "Test ElevenLabs", "Enter an ElevenLabs API key first.")
            return
        self.el_test.setEnabled(False)
        self._el_worker = _ElevenLabsTestWorker(key, self)
        self._el_worker.ok.connect(self._on_provider_test_ok)
        self._el_worker.failed.connect(self._on_provider_test_failed)
        self._el_worker.finished.connect(lambda: self.el_test.setEnabled(True))
        self._el_worker.start()

    def _on_provider_test_ok(self, message: str):
        QMessageBox.information(self, "Provider test", message)

    def _on_provider_test_failed(self, message: str):
        QMessageBox.warning(self, "Provider test failed", message)

    @staticmethod
    def _key_field(value: str) -> QLineEdit:
        field = QLineEdit(value)
        field.setEchoMode(QLineEdit.EchoMode.Password)
        field.setPlaceholderText("optional")
        return field

    @staticmethod
    def _wrap(layout):
        w = QWidget(); w.setLayout(layout); return w

    def _accept(self):
        self.s.elevenlabs_api_key = self.el_key.text().strip()
        self.s.elevenlabs_model = self.el_model.currentText()
        self.s.elevenlabs_tag_audio_events = self.el_audio_events.isChecked()
        self.s.gemini_model = self.gemini_model.currentText()
        self.s.gemini_mode = self.gemini_mode.currentData() or stt_gemini.DEFAULT_MODE
        self.s.model = self.model.currentData() or self.s.model
        self.s.device = self.device.currentText()
        self.s.compute_type = self.compute.currentText()
        self.s.language = self.lang.currentText()
        self.s.diarization_enabled = self.diar_on.isChecked()
        self.s.hf_token = self.hf.text().strip()
        self.s.plaud_page_size = self.plaud_page_size.value()
        self.s.vad_threshold = self.vad_threshold.value()
        self.s.vad_min_silence_ms = self.vad_min_silence.value()
        self.s.vad_speech_pad_ms = self.vad_pad.value()
        self.s.vad_min_speech_ms = self.vad_min_speech.value()
        self.s.vad_max_speech_s = self.vad_max_speech.value()
        self.s.bias_extra_terms = self.bias_terms.toPlainText().strip()
        self.s.bias_max_chars = self.bias_budget.value()
        self.s.hallucination_guard = self.hallucination_guard.isChecked()
        self.s.channel_mode = self.channel_mode.currentData() or "downmix"
        self.s.channel_names = self.channel_names.text().strip()
        self.s.line_mode = self.line_mode.currentData() or "segment"
        self.s.wrap_chars = self.wrap_chars.value()
        self.s.include_timestamps = self.include_ts.isChecked()
        self.s.newline = self.newline.currentData() or "crlf"
        self.s.filename_template = self.template.text().strip() or "{date}_{name}"
        self.s.overwrite = self.overwrite.isChecked()
        self.s.sanitize_names = self.sanitize.isChecked()
        self.s.owner_names = self.owner_names.text().strip()
        self.s.glossary_model = self.glossary_model.text().strip()
        self.s.glossary_temperature = self.glossary_temperature.value()
        self.s.glossary_chunk_token_threshold = self.glossary_chunk_threshold.value()
        self.s.force_reextract = self.force_reextract.isChecked()
        self.s.prompt_cache_enabled = self.prompt_cache_on.isChecked()
        self.s.deep_filter_path = self.deep_filter_path.text().strip()
        self.s.denoise_model_path = self.denoise_model_path.text().strip()
        self.s.denoise_postfilter = self.denoise_postfilter.isChecked()
        self.s.denoise_atten_lim_db = self.denoise_atten.value()
        self.s.ai_key_openrouter = self.ai_openrouter.text().strip()
        self.s.ai_key_openai = self.ai_openai.text().strip()
        self.s.ai_key_anthropic = self.ai_anthropic.text().strip()
        self.s.ai_key_google = self.ai_google.text().strip()
        self.s.ai_key_grok = self.ai_grok.text().strip()
        self.s.ai_key_ollama_cloud = self.ai_ollama_cloud.text().strip()
        self.s.ollama_local_url = self.ollama_local_url.text().strip() or "http://localhost:11434"

        provider = self.default_provider.currentData() or ""
        model = self.default_model.currentText().strip()
        changed = (provider, model) != (self.s.ai_default_provider, self.s.ai_default_model)
        self.s.ai_default_provider = provider
        self.s.ai_default_model = model
        if changed and provider and model:
            # A default nobody sees is not a default: point the current run at
            # it too, rather than leaving the last-used pair in charge.
            self.s.ai_cleanup_provider = provider
            self.s.ai_cleanup_model = model
        self.accept()
