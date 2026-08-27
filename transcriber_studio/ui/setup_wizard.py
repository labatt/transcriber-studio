# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""First-run setup: walk through everything the app needs before Go works.

The settings live in half a dozen places — a Plaud login held by the CLI, a
transcription engine (local or cloud), a HuggingFace token for speakers, an
optional LLM key for cleanup, an output folder — and none of them announce
themselves as missing until a job fails. This wizard asks for them in order,
checks each one where it can, and says plainly what is still missing at the end.

Nothing is written until Finish: every page edits a draft, so cancelling out
leaves the saved settings exactly as they were.
"""

from __future__ import annotations

import shutil
from dataclasses import replace

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPainter
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
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QWizard,
    QWizardPage,
)

from .. import (
    ai_providers,
    audio_utils,
    config,
    denoise,
    diarization,
    filename_builder,
    hf_client,
    stt_elevenlabs,
    vad,
    whisper_models,
)
from ..config import Settings
from ..hardware import cuda_available, cuda_device_name, torch_cuda_available
from ..plaud_client import PLAUD_BIN
from ..transcriber import ENGINE_ELEVENLABS, ENGINE_LABELS, ENGINE_LOCAL, faster_whisper_available
from ..workers import AccountWorker
from . import install_help
from .install_help import InstallHelpDialog
from .options_panel import FORMAT_OPTIONS
from .theme import WIZARD_HEADER_BAND, WrappedNote, bad, dialog_sheet, edge_color, good, hint, muted

OK, WARN, BAD = "✓", "•", "✗"


# A Plaud sign-in can take minutes; the user may close the wizard first. Keeping
# a reference here means the thread is never garbage-collected mid-run — Qt drops
# the signal connections on its own once the page it would call back into is gone.
_RUNNING: set[QThread] = set()


def _keep_alive(worker: QThread) -> QThread:
    _RUNNING.add(worker)
    worker.finished.connect(lambda: _RUNNING.discard(worker))
    return worker


class _TestWorker(QThread):
    """Runs one credential check off the UI thread; `fn` returns the message."""

    ok = Signal(str)
    failed = Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self):
        try:
            self.ok.emit(self._fn())
        except Exception as e:
            self.failed.emit(str(e))


class _Page(QWizardPage):
    """A page that edits the shared draft when the wizard moves on."""

    def __init__(self, draft: Settings, parent=None):
        super().__init__(parent)
        self.draft = draft

    def apply_to(self, s: Settings) -> None:      # overridden where there is input
        pass


# ---------------------------------------------------------------- welcome
class WelcomePage(_Page):
    def __init__(self, draft, parent=None):
        super().__init__(draft, parent)
        self.setTitle(f"Welcome to {config.APP_NAME}")
        self.setSubTitle("A quick pass through what the app needs. Everything is stored on this PC.")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "This takes about a minute. You can skip any step and change all of it "
            "later from the Settings button."
        ))
        self.box = QGroupBox("What is installed on this machine")
        self.box_layout = QVBoxLayout(self.box)
        layout.addWidget(self.box)

        # Detecting a missing dependency is only half of it — "ffmpeg not on
        # PATH" helps nobody who does not already know what to do about it.
        row = QHBoxLayout()
        self.help_btn = QPushButton("Help me install what is missing…")
        self.help_btn.clicked.connect(self._open_help)
        self.recheck_btn = QPushButton("Re-check")
        self.recheck_btn.clicked.connect(self._refresh)
        row.addWidget(self.help_btn)
        row.addWidget(self.recheck_btn)
        row.addStretch()
        layout.addLayout(row)
        layout.addStretch()
        self._refresh()

    def _refresh(self):
        while self.box_layout.count():
            item = self.box_layout.takeAt(0)
            widget = item.widget() if item else None
            if widget:
                widget.deleteLater()
        for text, style in self._environment():
            label = QLabel(text)
            label.setWordWrap(True)
            label.setStyleSheet(style)
            self.box_layout.addWidget(label)
        outstanding = install_help.missing()
        self.help_btn.setVisible(bool(outstanding))
        self.recheck_btn.setVisible(bool(outstanding))
        if outstanding:
            required = [r for r in outstanding if not r.optional]
            self.help_btn.setText(
                f"Help me install what is missing ({len(required)} required)…"
                if required else "Help me install the optional pieces…"
            )

    def _open_help(self):
        InstallHelpDialog(self).exec()
        self._refresh()      # they may have installed something while it was open

    @staticmethod
    def _environment() -> list[tuple[str, str]]:
        """Report the external pieces, since a missing one shows up as a job failure."""
        rows: list[tuple[str, str]] = []
        if shutil.which("plaud"):
            rows.append((f"{OK} Plaud CLI found ({PLAUD_BIN})", good()))
        else:
            rows.append((
                f"{BAD} Plaud CLI not found — install Node.js 20+, then: "
                "npm install -g @plaud-ai/cli  (only needed for cloud recordings)",
                bad(),
            ))
        if audio_utils.have_ffmpeg():
            rows.append((f"{OK} ffmpeg found", good()))
        else:
            rows.append((f"{BAD} ffmpeg not on PATH — needed for decoding and channel splitting", bad()))
        if faster_whisper_available():
            rows.append((f"{OK} faster-whisper installed (local transcription)", good()))
        else:
            rows.append((
                f"{WARN} faster-whisper not installed — either run "
                "`pip install faster-whisper` or use the ElevenLabs engine",
                muted(),
            ))
        if diarization.is_available():
            rows.append((f"{OK} pyannote.audio installed (local speaker detection)", good()))
        else:
            rows.append((
                f"{WARN} pyannote.audio not installed — local speaker detection is off "
                "until `pip install pyannote.audio`",
                muted(),
            ))
        if cuda_available():
            where = "GPU" if torch_cuda_available() else "GPU for Whisper, CPU for speakers"
            rows.append((f"{OK} CUDA GPU: {cuda_device_name()} ({where})", good()))
        else:
            rows.append((f"{WARN} No CUDA GPU detected — local transcription will use the CPU", muted()))
        return rows


# ------------------------------------------------------------------ plaud
class PlaudPage(_Page):
    def __init__(self, draft, parent=None):
        super().__init__(draft, parent)
        self.setTitle("Plaud account")
        self.setSubTitle("Sign in to browse the recordings in your Plaud cloud library.")
        self._worker: AccountWorker | None = None

        layout = QVBoxLayout(self)
        self.status = QLabel("Checking…")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        row = QHBoxLayout()
        self.login_btn = QPushButton("Sign in to Plaud")
        self.login_btn.clicked.connect(self._login)
        row.addWidget(self.login_btn)
        row.addStretch()
        layout.addLayout(row)

        note = QLabel(
            "Sign-in opens your browser; the Plaud CLI holds the token, not this app. "
            "You can skip this entirely and transcribe local audio files instead."
        )
        note.setWordWrap(True)
        note.setStyleSheet(muted())
        layout.addWidget(note)

        form = QFormLayout()
        self.page_size = QSpinBox()
        self.page_size.setRange(10, 200)
        self.page_size.setValue(draft.plaud_page_size)
        form.addRow("Recordings per page:", self.page_size)
        layout.addLayout(form)
        layout.addStretch()

    def initializePage(self):
        self._check("me")

    def _check(self, action: str):
        self.login_btn.setEnabled(False)
        self.status.setText("Opening your browser to sign in…" if action == "login" else "Checking…")
        self._worker = AccountWorker(action)
        _keep_alive(self._worker)
        self._worker.done.connect(self._on_account)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _login(self):
        self._check("login")

    def _on_account(self, account):
        self.login_btn.setEnabled(True)
        if account:
            self.status.setText(f"{OK} Signed in as {account.nickname} ({account.email})")
            self.status.setStyleSheet(good())
            self.login_btn.setText("Sign in as someone else")
        else:
            self.status.setText(f"{WARN} Not signed in — cloud recordings will not be listed.")
            self.status.setStyleSheet(muted())

    def _on_error(self, message: str):
        self.login_btn.setEnabled(True)
        self.status.setText(f"{BAD} {message.splitlines()[0]}")
        self.status.setStyleSheet(bad())

    def apply_to(self, s: Settings) -> None:
        s.plaud_page_size = self.page_size.value()


# ----------------------------------------------------------------- engine
class EnginePage(_Page):
    def __init__(self, draft, parent=None):
        super().__init__(draft, parent)
        self.setTitle("Transcription engine")
        self.setSubTitle("Where the audio is turned into text — on this PC, or in the cloud.")
        self._worker: _TestWorker | None = None

        layout = QVBoxLayout(self)
        self.engine = QComboBox()
        for engine_id in (ENGINE_LOCAL, ENGINE_ELEVENLABS):
            self.engine.addItem(ENGINE_LABELS[engine_id], engine_id)
        idx = self.engine.findData(draft.stt_engine)
        self.engine.setCurrentIndex(idx if idx >= 0 else 0)
        self.engine.currentIndexChanged.connect(self._on_engine_changed)
        top = QFormLayout()
        top.addRow("Engine:", self.engine)
        layout.addLayout(top)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._local_page(draft))
        self.stack.addWidget(self._cloud_page(draft))
        layout.addWidget(self.stack)
        layout.addStretch()
        self._on_engine_changed()

    def _local_page(self, draft) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.model = QComboBox()
        pick = whisper_models.recommended(cuda_available())
        for model_id in whisper_models.ORDER:
            name = whisper_models.label(model_id)
            self.model.addItem(f"{name}  (recommended)" if model_id == pick else name, model_id)
        idx = self.model.findData(draft.model)
        self.model.setCurrentIndex(idx if idx >= 0 else self.model.findData(pick))
        self.model.currentIndexChanged.connect(self._update_model_note)
        form.addRow("Whisper model:", self.model)

        self.device = QComboBox()
        self.device.addItems(["auto", "cuda", "cpu"])
        self.device.setCurrentText(draft.device)
        form.addRow("Device:", self.device)

        # Six opaque names otherwise: nothing in "medium" says whether it fits
        # on this GPU or how much longer a two-hour recording will take.
        # A self-sizing note: the fine-tunes carry several lines of caveats, and
        # a plain wrapped label gets a row sized for the wrong width.
        self.model_note = WrappedNote()
        form.addRow(self.model_note)

        self.local_note = QLabel()
        self.local_note.setWordWrap(True)
        self.local_note.setStyleSheet(muted())
        if not faster_whisper_available():
            self.local_note.setText(
                f"{BAD} faster-whisper is not installed — see the install help on the "
                "first page, or pick the ElevenLabs engine above."
            )
            self.local_note.setStyleSheet(bad())
        else:
            self.local_note.setText(f"{OK} Nothing leaves this PC.")
        form.addRow(self.local_note)
        self._update_model_note()
        return page

    def _update_model_note(self, _index: int = 0):
        model_id = self.model.currentData() or whisper_models.GPU_RECOMMENDED
        self.model_note.setText(whisper_models.describe(model_id, cuda_available()))
        recommended = model_id == whisper_models.recommended(cuda_available())
        self.model_note.setStyleSheet(good() if recommended else muted())

    def _cloud_page(self, draft) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.el_model = QComboBox()
        self.el_model.addItems(stt_elevenlabs.MODELS)
        self.el_model.setCurrentText(draft.elevenlabs_model or stt_elevenlabs.DEFAULT_MODEL)
        form.addRow("Scribe model:", self.el_model)

        self.el_status = QLabel(
            "Audio is uploaded to ElevenLabs and billed to your ElevenLabs account. "
            "Scribe detects speakers itself, so no HuggingFace token or GPU is needed.\n"
            "The API key is asked for on the next step."
        )
        self.el_status.setWordWrap(True)
        self.el_status.setStyleSheet(muted())
        form.addRow(self.el_status)
        return page

    def _on_engine_changed(self, _index: int = 0):
        self.stack.setCurrentIndex(0 if self.engine.currentData() == ENGINE_LOCAL else 1)

    def apply_to(self, s: Settings) -> None:
        s.stt_engine = self.engine.currentData() or ENGINE_LOCAL
        s.model = self.model.currentData() or s.model
        s.device = self.device.currentText()
        s.elevenlabs_model = self.el_model.currentText()


# ------------------------------------------------------------------- keys
class _KeyField:
    """One optional credential: the box, its Test button, and its verdict.

    Tests only ever run against what is actually typed in — an empty field is
    a deliberate "not using this service", not something to go and check.
    """

    def __init__(self, page: QWidget, placeholder: str, tester, note: str):
        self.page = page
        self.tester = tester
        self.edit = QLineEdit()
        self.edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.edit.setPlaceholderText(placeholder)
        self.edit.textChanged.connect(self._on_changed)
        self.test_btn = QPushButton("Test")
        self.test_btn.setFixedWidth(56)
        self.test_btn.clicked.connect(self.test)
        self.status = QLabel(note)
        self.status.setWordWrap(True)
        self.status.setStyleSheet(muted())
        self.note = note
        self._worker: _TestWorker | None = None
        self._on_changed()

    # -- widgets -------------------------------------------------------
    def row(self) -> QWidget:
        wrapper = QWidget()
        layout = QHBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit, stretch=1)
        layout.addWidget(self.test_btn)
        return wrapper

    # -- state ---------------------------------------------------------
    def value(self) -> str:
        return self.edit.text().strip()

    def set_value(self, value: str) -> None:
        self.edit.setText(value or "")

    def _on_changed(self, _text: str = "") -> None:
        self.test_btn.setEnabled(bool(self.value()))
        if not self.value():
            self._set(self.note, muted())

    def _set(self, text: str, style: str) -> None:
        self.status.setText(text)
        self.status.setStyleSheet(style)

    def busy(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def test(self) -> bool:
        """Check this key. Returns False when there is nothing to check."""
        key = self.value()
        if not key or self.busy():
            return False
        self.test_btn.setEnabled(False)
        self._set("Checking…", muted())
        self._worker = _TestWorker(lambda: self.tester(key))
        _keep_alive(self._worker)
        self._worker.ok.connect(lambda m: self._set(f"{OK} {m}", good()))
        self._worker.failed.connect(lambda m: self._set(f"{BAD} {m}", bad()))
        self._worker.finished.connect(lambda: self.test_btn.setEnabled(bool(self.value())))
        self._worker.start()
        return True


class KeysPage(_Page):
    """Both service keys, whichever engine is selected.

    Neither is required — one covers cloud transcription, the other local
    speaker detection — and a user may well set up both today and choose
    between them per run later, so both are asked for either way.
    """

    def __init__(self, draft, parent=None):
        super().__init__(draft, parent)
        self.setTitle("Service keys")
        self.setSubTitle("Both are optional. Fill in the ones you have; the rest can wait.")
        layout = QVBoxLayout(self)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.elevenlabs = _KeyField(
            self,
            "sk_… (stored locally only)",
            stt_elevenlabs.test_key,
            "Needed only for the ElevenLabs Scribe engine. Leave blank to transcribe locally.",
        )
        self.elevenlabs.set_value(draft.elevenlabs_api_key)
        form.addRow("ElevenLabs API key:", self.elevenlabs.row())
        form.addRow("", self.elevenlabs.status)

        self.huggingface = _KeyField(
            self,
            "hf_… (stored locally only)",
            hf_client.test_token,
            "Needed only for local speaker detection (pyannote). Leave blank to skip speakers "
            "or to let ElevenLabs label them.",
        )
        self.huggingface.set_value(draft.hf_token)
        form.addRow("HuggingFace token:", self.huggingface.row())
        form.addRow("", self.huggingface.status)
        layout.addLayout(form)

        row = QHBoxLayout()
        self.test_all_btn = QPushButton("Test the keys I entered")
        self.test_all_btn.clicked.connect(self._test_entered)
        row.addWidget(self.test_all_btn)
        row.addStretch()
        layout.addLayout(row)

        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet(muted())
        layout.addWidget(self.summary)

        note = QLabel(
            "The HuggingFace test also checks that your account has accepted the three "
            "pyannote model licences — a valid token still fails without them."
        )
        note.setWordWrap(True)
        note.setStyleSheet(muted())
        layout.addWidget(note)
        layout.addStretch()

    def _test_entered(self):
        """Test every key that has something in it, and nothing else."""
        tested = [name for name, field in
                  (("ElevenLabs", self.elevenlabs), ("HuggingFace", self.huggingface))
                  if field.test()]
        if not tested:
            self.summary.setText("Nothing to test — both boxes are empty, which is fine.")
        else:
            self.summary.setText(f"Testing {' and '.join(tested)}…")

    def apply_to(self, s: Settings) -> None:
        s.elevenlabs_api_key = self.elevenlabs.value()
        s.hf_token = self.huggingface.value()


# --------------------------------------------------------------- speakers
class SpeakersPage(_Page):
    def __init__(self, draft, parent=None):
        super().__init__(draft, parent)
        self.setTitle("Speakers")
        self.setSubTitle("Labelling who said what.")
        layout = QVBoxLayout(self)

        self.enabled = QCheckBox("Detect speakers and label the transcript")
        self.enabled.setChecked(draft.diarization_enabled)
        self.enabled.toggled.connect(self._update)
        layout.addWidget(self.enabled)

        form = QFormLayout()
        self.maxspk = QSpinBox()
        self.maxspk.setRange(0, 20)
        self.maxspk.setValue(draft.max_speakers)
        self.maxspk.setToolTip("0 = let the model decide how many people are talking.")
        form.addRow("Most speakers expected:", self.maxspk)
        layout.addLayout(form)

        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setStyleSheet(muted())
        layout.addWidget(self.note)
        layout.addStretch()

    def initializePage(self):
        self._update()

    def _cloud(self) -> bool:
        return self.field_engine() == ENGINE_ELEVENLABS

    def field_engine(self) -> str:
        """The engine chosen on the previous page, not the saved one."""
        wizard = self.wizard()
        page = wizard.page(SetupWizard.PAGE_ENGINE) if wizard else None
        return page.engine.currentData() if page else self.draft.stt_engine

    def _update(self, _checked: bool = False):
        on = self.enabled.isChecked()
        cloud = self._cloud()
        self.maxspk.setEnabled(on)
        if not on:
            self.note.setText("Transcripts will be plain text with no speaker names.")
        elif cloud:
            self.note.setText(
                f"{OK} ElevenLabs Scribe detects speakers as part of transcription — "
                "no HuggingFace token needed. The limit above is passed to Scribe."
            )
        elif not diarization.is_available():
            self.note.setText(
                f"{BAD} pyannote.audio is not installed, so local speaker detection cannot run: "
                "`pip install pyannote.audio`."
            )
        else:
            self.note.setText(
                "Local speaker detection uses pyannote with the HuggingFace token from "
                "the previous step. Your account must also have accepted the licences on "
                "pyannote/speaker-diarization-community-1, pyannote/segmentation-3.0 and "
                "pyannote/speaker-diarization-3.1 — the Test button there checks both."
            )

    def apply_to(self, s: Settings) -> None:
        s.diarization_enabled = self.enabled.isChecked()
        s.max_speakers = self.maxspk.value()


# --------------------------------------------------------------- pipeline
class PipelinePage(_Page):
    """The three layers in front of the decoder.

    Worth its own page rather than a line in Options: on hard audio these
    matter more than the model choice, and a first-run user who never finds
    them judges the app on its worst output.
    """

    def __init__(self, draft, parent=None):
        super().__init__(draft, parent)
        self.setTitle("Audio pipeline")
        self.setSubTitle(
            "What happens to the audio before the decoder sees it. On difficult "
            "recordings this is worth more than a bigger model."
        )
        layout = QVBoxLayout(self)

        self.denoise = QCheckBox("Denoise the audio first")
        self.denoise.setChecked(draft.denoise_enabled)
        self.denoise.toggled.connect(self._update)
        layout.addWidget(self.denoise)

        row = QHBoxLayout()
        self.df_path = QLineEdit(draft.deep_filter_path)
        self.df_path.setPlaceholderText("Path to the deep-filter binary (optional)")
        self.df_path.textChanged.connect(self._update)
        browse = QPushButton("Browse…")
        browse.setAutoDefault(False)
        browse.clicked.connect(self._browse)
        row.addWidget(self.df_path, stretch=1)
        row.addWidget(browse)
        layout.addLayout(row)

        self.denoise_note = QLabel()
        self.denoise_note.setWordWrap(True)
        self.denoise_note.setStyleSheet(muted())
        layout.addWidget(self.denoise_note)

        self.vad = QCheckBox("Skip silence and non-speech (voice activity detection)")
        self.vad.setChecked(draft.vad_enabled)
        self.vad.toggled.connect(self._update)
        layout.addWidget(self.vad)

        self.vad_note = QLabel()
        self.vad_note.setWordWrap(True)
        self.vad_note.setStyleSheet(muted())
        layout.addWidget(self.vad_note)

        self.bias = QCheckBox("Tell the decoder which names and terms to expect")
        self.bias.setChecked(draft.bias_enabled)
        self.bias.toggled.connect(self._update)
        layout.addWidget(self.bias)

        self.guard = QCheckBox("Guard against hallucinated passages")
        self.guard.setChecked(draft.hallucination_guard)
        layout.addWidget(self.guard)

        self.bias_note = QLabel(
            "Vocabulary comes from the shared glossary a job uses, so it fills "
            "itself in as you go: what one recording teaches the app, the next "
            "one already knows."
        )
        self.bias_note.setWordWrap(True)
        self.bias_note.setStyleSheet(muted())
        layout.addWidget(self.bias_note)
        layout.addStretch()

    def initializePage(self):
        self._update()

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Locate the deep-filter binary", self.df_path.text(),
            "Programs (*.exe);;All files (*)",
        )
        if path:
            self.df_path.setText(path)

    def _draft_now(self) -> Settings:
        """The settings as this page currently shows them."""
        draft = Settings(**self.draft.to_dict())
        draft.denoise_enabled = self.denoise.isChecked()
        draft.deep_filter_path = self.df_path.text().strip()
        draft.vad_enabled = self.vad.isChecked()
        return draft

    def _update(self, _checked: bool = False):
        on = self.denoise.isChecked()
        self.df_path.setEnabled(on)
        self.denoise_note.setText(denoise.describe(self._draft_now()))
        self.vad_note.setText(vad.describe(self._draft_now()))

    def apply_to(self, s: Settings) -> None:
        s.denoise_enabled = self.denoise.isChecked()
        s.deep_filter_path = self.df_path.text().strip()
        s.vad_enabled = self.vad.isChecked()
        s.bias_enabled = self.bias.isChecked()
        s.hallucination_guard = self.guard.isChecked()


# --------------------------------------------------------------- AI clean
class CleanupPage(_Page):
    def __init__(self, draft, parent=None):
        super().__init__(draft, parent)
        self.setTitle("AI Cleanup (optional)")
        self.setSubTitle("Have an LLM tidy punctuation, names and speaker turns after transcription.")
        self._worker: _TestWorker | None = None

        layout = QVBoxLayout(self)
        self.enabled = QCheckBox("Run AI Cleanup on every job")
        self.enabled.setChecked(draft.ai_cleanup_enabled)
        self.enabled.toggled.connect(self._update)
        layout.addWidget(self.enabled)

        form = QFormLayout()
        self.provider = QComboBox()
        for pid, spec in ai_providers.PROVIDERS.items():
            self.provider.addItem(ai_providers.PROVIDER_LABELS.get(pid, spec.label), pid)
        idx = self.provider.findData(draft.ai_cleanup_provider or "anthropic")
        self.provider.setCurrentIndex(max(0, idx))
        self.provider.currentIndexChanged.connect(self._on_provider_changed)
        form.addRow("Provider:", self.provider)

        key_row = QHBoxLayout()
        key_row.setContentsMargins(0, 0, 0, 0)
        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.EchoMode.Password)
        self.test_btn = QPushButton("Test")
        self.test_btn.setFixedWidth(56)
        self.test_btn.clicked.connect(self._test)
        key_row.addWidget(self.key, stretch=1)
        key_row.addWidget(self.test_btn)
        wrapper = QWidget()
        wrapper.setLayout(key_row)
        form.addRow("API key:", wrapper)
        layout.addLayout(form)

        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setStyleSheet(muted())
        layout.addWidget(self.status)

        note = QLabel(
            "Only the provider you pick here is set up now; the rest can be added in "
            "Settings. Pick the exact model afterwards in the Options panel — the list "
            "is fetched from the provider."
        )
        note.setWordWrap(True)
        note.setStyleSheet(muted())
        layout.addWidget(note)
        layout.addStretch()
        self._on_provider_changed()

    def _spec(self):
        return ai_providers.PROVIDERS[self.provider.currentData()]

    def _on_provider_changed(self, _index: int = 0):
        """Each provider has its own key (and Ollama local has a URL instead)."""
        spec = self._spec()
        if spec.key_field:
            self.key.setEchoMode(QLineEdit.EchoMode.Password)
            self.key.setPlaceholderText("API key (stored locally only)")
            self.key.setText(getattr(self.draft, spec.key_field, ""))
        else:
            self.key.setEchoMode(QLineEdit.EchoMode.Normal)
            self.key.setPlaceholderText("http://localhost:11434")
            self.key.setText(self.draft.ollama_local_url)
        self.status.clear()
        self._update()

    def _update(self, _checked: bool = False):
        on = self.enabled.isChecked()
        for widget in (self.provider, self.key, self.test_btn):
            widget.setEnabled(on)
        if not on:
            self.status.setText("Transcripts are exported exactly as transcribed.")

    def _draft_with_key(self) -> Settings:
        spec = self._spec()
        value = self.key.text().strip()
        if spec.key_field:
            return replace(self.draft, **{spec.key_field: value})
        return replace(self.draft, ollama_local_url=value or "http://localhost:11434")

    def _test(self):
        if self._worker and self._worker.isRunning():
            return
        provider = self.provider.currentData()
        draft = self._draft_with_key()
        self.test_btn.setEnabled(False)
        self.status.setText("Checking with the provider…")
        self.status.setStyleSheet(muted())
        self._worker = _TestWorker(lambda: ai_providers.test_provider(draft, provider))
        _keep_alive(self._worker)
        self._worker.ok.connect(lambda m: self._set_status(f"{OK} {m}", good()))
        self._worker.failed.connect(lambda m: self._set_status(f"{BAD} {m}", bad()))
        self._worker.finished.connect(lambda: self.test_btn.setEnabled(True))
        self._worker.start()

    def _set_status(self, text: str, style: str):
        self.status.setText(text)
        self.status.setStyleSheet(style)

    def apply_to(self, s: Settings) -> None:
        spec = self._spec()
        value = self.key.text().strip()
        if spec.key_field:
            setattr(s, spec.key_field, value)
        else:
            s.ollama_local_url = value or "http://localhost:11434"
        s.ai_cleanup_enabled = self.enabled.isChecked()
        if self.enabled.isChecked():
            s.ai_cleanup_provider = self.provider.currentData()


# ----------------------------------------------------------------- output
class OutputPage(_Page):
    def __init__(self, draft, parent=None):
        super().__init__(draft, parent)
        self.setTitle("Output")
        self.setSubTitle("Where finished transcripts are written, and what they are called.")
        layout = QVBoxLayout(self)

        form = QFormLayout()
        dir_row = QHBoxLayout()
        dir_row.setContentsMargins(0, 0, 0, 0)
        self.out_dir = QLineEdit(draft.output_dir)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        dir_row.addWidget(self.out_dir, stretch=1)
        dir_row.addWidget(browse)
        wrapper = QWidget()
        wrapper.setLayout(dir_row)
        form.addRow("Folder:", wrapper)

        self.template = QLineEdit(draft.filename_template)
        self.template.textChanged.connect(self._update_preview)
        form.addRow("Filename template:", self.template)
        self.preview = QLabel()
        self.preview.setStyleSheet(hint())
        form.addRow("Example:", self.preview)
        layout.addLayout(form)

        box = QGroupBox("Formats to write")
        inner = QVBoxLayout(box)
        self.formats: dict[str, QCheckBox] = {}
        for key, label in FORMAT_OPTIONS:
            cb = QCheckBox(label)
            cb.setChecked(key in draft.formats)
            self.formats[key] = cb
            inner.addWidget(cb)
        layout.addWidget(box)
        layout.addStretch()
        self._update_preview()

    def _browse(self):
        chosen = QFileDialog.getExistingDirectory(self, "Choose output folder", self.out_dir.text())
        if chosen:
            self.out_dir.setText(chosen)

    def _update_preview(self):
        stem = filename_builder.render(
            self.template.text(), filename_builder.sample_values(), self.draft.sanitize_names
        )
        self.preview.setText(f"{stem}.txt")

    def apply_to(self, s: Settings) -> None:
        s.output_dir = self.out_dir.text().strip() or s.output_dir
        s.filename_template = self.template.text().strip() or "{date}_{name}"
        s.formats = [k for k, cb in self.formats.items() if cb.isChecked()] or ["txt"]


# ----------------------------------------------------------------- finish
class FinishPage(_Page):
    def __init__(self, draft, parent=None):
        super().__init__(draft, parent)
        self.setTitle("Ready")
        self.setSubTitle("What is set up, and anything still missing.")
        layout = QVBoxLayout(self)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.summary)
        layout.addStretch()
        tail = QLabel(
            "Press Finish to save. Everything here stays on this PC and can be changed "
            "later with the Settings button — or by running Setup again."
        )
        tail.setWordWrap(True)
        tail.setStyleSheet(muted())
        layout.addWidget(tail)

    def initializePage(self):
        """Built from the draft as it stands now, not from what was saved."""
        wizard = self.wizard()
        preview = Settings(**self.draft.to_dict())
        if isinstance(wizard, SetupWizard):
            wizard.apply_pages(preview)
        rows: list[str] = []

        if preview.stt_engine == ENGINE_ELEVENLABS:
            rows.append(f"{OK} Engine: ElevenLabs Scribe ({preview.elevenlabs_model})")
            if not preview.elevenlabs_api_key:
                rows.append(f"{BAD} No ElevenLabs API key — jobs will fail until one is added.")
        else:
            rows.append(f"{OK} Engine: local Whisper ({preview.model}, device {preview.device})")
            if not faster_whisper_available():
                rows.append(f"{BAD} faster-whisper is not installed — `pip install faster-whisper`.")

        if not preview.diarization_enabled:
            rows.append(f"{WARN} Speaker detection off.")
        elif preview.stt_engine == ENGINE_ELEVENLABS:
            rows.append(f"{OK} Speakers: detected by Scribe.")
        elif not preview.hf_token:
            rows.append(f"{BAD} Speaker detection on, but no HuggingFace token — it will be skipped.")
        elif not diarization.is_available():
            rows.append(f"{BAD} Speaker detection on, but pyannote.audio is not installed.")
        else:
            rows.append(f"{OK} Speakers: pyannote with your HuggingFace token.")

        if preview.ai_cleanup_enabled:
            label = ai_providers.PROVIDER_LABELS.get(
                preview.ai_cleanup_provider, preview.ai_cleanup_provider
            )
            rows.append(f"{OK} AI Cleanup: {label} — pick the model in the Options panel.")
        else:
            rows.append(f"{WARN} AI Cleanup off.")

        rows.append(f"{OK} Output: {preview.output_dir} ({', '.join(preview.formats)})")
        if not audio_utils.have_ffmpeg():
            rows.append(f"{BAD} ffmpeg is not on PATH — decoding and channel splitting will fail.")
        self.summary.setText("<br>".join(rows))


# ----------------------------------------------------------------- wizard
class SetupWizard(QWizard):
    PAGE_WELCOME, PAGE_PLAUD, PAGE_ENGINE, PAGE_KEYS = 0, 1, 2, 3
    PAGE_SPEAKERS, PAGE_PIPELINE, PAGE_CLEANUP = 4, 5, 6
    PAGE_OUTPUT, PAGE_FINISH = 7, 8

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.s = settings
        # A copy, so a cancelled wizard changes nothing the app is using.
        self.draft = Settings(**settings.to_dict())
        self.setWindowTitle(f"{config.APP_NAME} — Setup")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setOption(QWizard.WizardOption.NoBackButtonOnStartPage, True)
        self.setMinimumSize(660, 560)
        self._pages: list[_Page] = [
            WelcomePage(self.draft, self),
            PlaudPage(self.draft, self),
            EnginePage(self.draft, self),
            KeysPage(self.draft, self),
            SpeakersPage(self.draft, self),
            PipelinePage(self.draft, self),
            CleanupPage(self.draft, self),
            OutputPage(self.draft, self),
            FinishPage(self.draft, self),
        ]
        for page_id, page in enumerate(self._pages):
            self.setPage(page_id, page)
        self.setStartId(self.PAGE_WELCOME)

        # Without the sheet the dialog is near-black on a near-black window and
        # its edges are invisible — the only clue is the widgets it covers.
        self._band_named = False
        self.setStyleSheet(dialog_sheet())

    def showEvent(self, event):
        """Style the title band once Qt has actually built it.

        It does not exist until the wizard is shown, so this is the first
        moment it can be found and named.
        """
        super().showEvent(event)
        if not self._band_named and self._name_header_band():
            self._band_named = True
            self.setStyleSheet(dialog_sheet())   # re-polish now the name matches

    def _name_header_band(self) -> bool:
        """Give QWizard's unnamed title band a name so the sheet can style it.

        Qt's other auto-filled plain QWidgets here are scroll-area viewports,
        which it names; the band is the one it leaves anonymous. If a future Qt
        changes that, nothing matches and the band simply keeps the palette
        colour it has today.
        """
        for child in self.findChildren(QWidget):
            if (
                child.metaObject().className() == "QWidget"
                and child.autoFillBackground()
                and not child.objectName()
                and child.height() < 200      # a band, not a page-sized panel
            ):
                child.setObjectName(WIZARD_HEADER_BAND)
                return True
        return False

    def paintEvent(self, event):
        """Outline the dialog.

        A CSS `border` on a top-level dialog is silently ignored, so the edge
        is drawn here — without it the sheet still merges into a dark window
        and the only clue to where the wizard starts is the widgets it covers.
        Only the sides and bottom come from here: the title band covers the
        top edge, and carries that line in its own style instead.
        """
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setPen(edge_color())
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        painter.end()

    def apply_pages(self, target: Settings) -> None:
        """Fold every page's answers into `target`, in page order."""
        for page in self._pages:
            page.apply_to(target)

    def accept(self):
        self.apply_pages(self.s)
        self.s.setup_complete = True
        super().accept()
