# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Main application window."""

from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .. import config, history
from .. import resume as resume_store
from ..config import Settings
from ..hardware import (
    CUDA_TORCH_INSTALL_CMD,
    cuda_available,
    cuda_device_name,
    torch_cuda_available,
)
from ..jobs import (
    JobResult,
    JobRunner,
    apply_speaker_renames,
    ensure_original_snapshot,
    remove_superseded_outputs,
)
from ..models import Recording, Source
from ..queue_store import clear_queue_file, load_queue, save_queue
from ..workers import AccountWorker, CleanupWorker, DiarizationWorker, TranscriptionWorker
from .ai_cleanup_dialog import AICleanupDialog
from .glossary_dialog import GlossaryLibraryDialog
from .local_files_tab import LocalFilesTab
from .options_panel import OptionsPanel
from .recordings_tab import RecordingsTab
from .rename_dialog import SpeakerRenameDialog
from .settings_dialog import SettingsDialog
from .setup_wizard import SetupWizard

QUEUE_COLS = ["Recording", "Source", "Status", "Progress", "Output"]

GO_BTN_STYLE = """
QPushButton#goBtn {
    background-color: #616161;
    color: #bdbdbd;
    font-weight: bold;
    font-size: 14px;
    padding: 8px 28px;
    border: 1px solid #4a4a4a;
    border-radius: 6px;
}
QPushButton#goBtn:enabled {
    background-color: #2e7d32;
    color: white;
    border: 1px solid #1b5e20;
}
QPushButton#goBtn:enabled:hover {
    background-color: #388e3c;
}
QPushButton#goBtn:disabled {
    background-color: #4a4a4a;
    color: #8a8a8a;
    border: 1px solid #3a3a3a;
}
"""

GPU_BADGE_ON = (
    "QLabel#gpuBadge { background-color: #2e7d32; color: white; font-weight: bold;"
    " padding: 4px 10px; border-radius: 5px; }"
)
GPU_BADGE_OFF = (
    "QLabel#gpuBadge { background-color: #555; color: #ccc; font-weight: bold;"
    " padding: 4px 10px; border-radius: 5px; }"
)

QUEUE_TABLE_STYLE = """
QTableWidget {
    gridline-color: #4a4a4a;
    background-color: #2b2b2b;
    alternate-background-color: #303030;
}
QTableWidget::item {
    padding: 5px 8px;
}
QHeaderView::section {
    background-color: #353535;
    color: #e0e0e0;
    padding: 6px 8px;
    border: 1px solid #4a4a4a;
    font-weight: bold;
}
"""

QUEUE_PROGRESS_STYLE = """
QProgressBar {
    border: 1px solid #555;
    border-radius: 4px;
    text-align: center;
    min-height: 20px;
    max-height: 22px;
    padding: 0 4px;
}
QProgressBar::chunk {
    background-color: #42a5f5;
    border-radius: 3px;
}
"""

OUTPUT_OPEN_BTN_STYLE = """
QPushButton {
    padding: 2px 8px;
    min-height: 22px;
    max-height: 24px;
}
"""


class ElidingLabel(QLabel):
    """A one-line filename that shortens with an ellipsis instead of wrapping.

    A word-wrapped label inside a table cell reports a height the row never
    learns about (its wrapped height depends on the column width, which is
    decided later), so the second line ends up clipped — and the sliver of it
    that survives reads as a line struck through the text. One line, elided in
    the middle, keeps both the name and the extension readable; the full path
    stays in the tooltip.
    """

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self._full = text
        self.setWordWrap(False)
        self.setMinimumWidth(48)
        self.setText(text)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        elided = self.fontMetrics().elidedText(
            self._full, Qt.TextElideMode.ElideMiddle, self.width()
        )
        if elided != self.text():
            super().setText(elided)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings: Settings = config.load()
        self.setWindowTitle(config.APP_NAME)
        self.resize(1180, 820)

        self._worker: TranscriptionWorker | None = None
        self._diar_worker: DiarizationWorker | None = None
        self._cleanup_worker: CleanupWorker | None = None
        self._acct_worker: AccountWorker | None = None
        self._results: dict[int, JobResult] = {}
        self._queue_recordings: list[Recording] = []
        self._queue_statuses: list[str] = []
        self._job_errors: list[str] = []
        self._progress_started: dict[int, float] = {}
        self._progress_labels: dict[int, str] = {}
        self._processing_row: int | None = None
        self._auto_listed = False   # recordings auto-loaded once per session

        self._build_header()
        self._build_central()
        self._load_persisted_queue()
        self._refresh_account()
        self._update_go_button()
        self._update_gpu_badge()
        self._update_job_actions()   # reflect restored rows (e.g. Resume availability)
        # After the window is up, so the wizard opens over a drawn app rather
        # than an empty frame.
        QTimer.singleShot(0, self._maybe_run_setup_wizard)

    # ------------------------------------------------------------------
    def _build_header(self):
        header = QWidget()
        header.setObjectName("appHeader")
        row = QHBoxLayout(header)
        row.setContentsMargins(10, 6, 10, 6)
        row.setSpacing(8)

        self.account_label = QLabel("Checking Plaud login…")
        self.account_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.account_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        row.addWidget(self.account_label, stretch=1)

        self.gpu_badge = QLabel()
        self.gpu_badge.setObjectName("gpuBadge")
        row.addWidget(self.gpu_badge)

        self.login_btn = QPushButton("Login")
        self.logout_btn = QPushButton("Logout")
        self.setup_btn = QPushButton("Setup")
        self.setup_btn.setToolTip("Re-run the first-run setup wizard.")
        # "Glossary", not "Glossaries": the header buttons share a fixed width
        # that the longer word does not fit inside without eliding.
        self.glossaries_btn = QPushButton("Glossary")
        self.glossaries_btn.setToolTip(
            "Create and edit the shared glossaries that jobs read from and write to."
        )
        self.settings_btn = QPushButton("Settings")
        for btn in (
            self.login_btn, self.logout_btn, self.setup_btn,
            self.glossaries_btn, self.settings_btn,
        ):
            btn.setFixedWidth(88)
            btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            row.addWidget(btn)

        self.login_btn.clicked.connect(self._login)
        self.logout_btn.clicked.connect(self._logout)
        self.setup_btn.clicked.connect(self._run_setup_wizard)
        self.glossaries_btn.clicked.connect(self._open_glossaries)
        self.settings_btn.clicked.connect(self._open_settings)

        wrapper = QWidget()
        wrap_layout = QVBoxLayout(wrapper)
        wrap_layout.setContentsMargins(0, 0, 0, 0)
        wrap_layout.setSpacing(0)
        wrap_layout.addWidget(header)
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        wrap_layout.addWidget(line)
        self.setMenuWidget(wrapper)

    def _build_central(self):
        central = QWidget(); outer = QVBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # left: sources
        self.tabs = QTabWidget()
        self.recordings_tab = RecordingsTab(self.settings)
        self.local_tab = LocalFilesTab()
        self.tabs.addTab(self.recordings_tab, "Plaud Recordings")
        self.tabs.addTab(self.local_tab, "Local Files")
        self.tabs.currentChanged.connect(lambda _i: self._update_go_button())
        self.recordings_tab.selection_changed.connect(self._update_go_button)
        self.local_tab.selection_changed.connect(self._update_go_button)
        splitter.addWidget(self.tabs)

        # right: options (scrollable)
        self.options = OptionsPanel(self.settings)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        scroll.setFrameShape(QFrame.Shape.StyledPanel)
        scroll.setWidget(self.options)
        scroll.setMinimumWidth(300)
        self._options_scroll = scroll
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([680, 400])
        outer.addWidget(splitter, stretch=3)

        # action row
        action_row = QHBoxLayout()
        self.start_btn = QPushButton("▶  Go")
        self.start_btn.setObjectName("goBtn")
        self.start_btn.setStyleSheet(GO_BTN_STYLE)
        self.start_btn.setEnabled(False)
        # Go spends GPU time and API credit, so it takes a deliberate click:
        # no autoDefault (a stray Enter anywhere in the window would fire it)
        # and no keyboard focus it could pick up on its own.
        self.start_btn.setAutoDefault(False)
        self.start_btn.setDefault(False)
        self.start_btn.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.start_btn.clicked.connect(self._start)
        self.cancel_btn = QPushButton("Cancel"); self.cancel_btn.setEnabled(False)
        self.cancel_btn.setAutoDefault(False)
        self.cancel_btn.clicked.connect(self._cancel)
        action_row.addWidget(self.start_btn)
        action_row.addWidget(self.cancel_btn)
        action_row.addStretch()
        outer.addLayout(action_row)

        jobs_header = QHBoxLayout()
        jobs_header.addWidget(QLabel("Jobs"))
        jobs_header.addStretch()
        self.remove_job_btn = QPushButton("Remove selected")
        self.remove_job_btn.clicked.connect(self._remove_selected_jobs)
        self.clear_jobs_btn = QPushButton("Clear all")
        self.clear_jobs_btn.clicked.connect(self._clear_all_jobs)
        self.detect_speakers_btn = QPushButton("Detect speakers")
        self.detect_speakers_btn.clicked.connect(self._detect_speakers_selected)
        self.rename_speakers_btn = QPushButton("Rename speakers")
        self.rename_speakers_btn.clicked.connect(self._rename_speakers_selected)
        self.ai_cleanup_btn = QPushButton("AI Cleanup")
        self.ai_cleanup_btn.clicked.connect(self._ai_cleanup_selected)
        self.resume_btn = QPushButton("Resume")
        self.resume_btn.setToolTip(
            "Continue an interrupted job from its last saved step "
            "(no re-transcription, no repeated model calls)."
        )
        self.resume_btn.clicked.connect(self._resume_selected)
        for btn in (
            self.remove_job_btn,
            self.clear_jobs_btn,
            self.resume_btn,
            self.detect_speakers_btn,
            self.rename_speakers_btn,
            self.ai_cleanup_btn,
        ):
            btn.setAutoDefault(False)   # Enter must never trigger these either
            jobs_header.addWidget(btn)
        outer.addLayout(jobs_header)

        # jobs table
        self.queue = QTableWidget(0, len(QUEUE_COLS))
        self.queue.setHorizontalHeaderLabels(QUEUE_COLS)
        self.queue.setStyleSheet(QUEUE_TABLE_STYLE)
        self.queue.setShowGrid(True)
        self.queue.setGridStyle(Qt.PenStyle.SolidLine)
        self.queue.setAlternatingRowColors(True)
        self.queue.verticalHeader().setVisible(False)
        self.queue.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.queue.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.queue.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        qh = self.queue.horizontalHeader()
        qh.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        qh.setStretchLastSection(True)
        qh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        qh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        qh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        qh.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        qh.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        qh.resizeSection(3, 210)
        qh.setMinimumSectionSize(72)
        self.queue.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.queue.itemSelectionChanged.connect(self._update_job_actions)
        outer.addWidget(self.queue, stretch=2)

        # log
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(140)
        outer.addWidget(self.log)

        self.setCentralWidget(central)
        self._update_job_actions()

    def _jobs_busy(self) -> bool:
        return (
            (self._worker is not None and self._worker.isRunning())
            or (self._diar_worker is not None and self._diar_worker.isRunning())
            or (self._cleanup_worker is not None and self._cleanup_worker.isRunning())
        )

    def _selected_job_rows(self) -> list[int]:
        return sorted({idx.row() for idx in self.queue.selectedIndexes()})

    def _selected_job_row(self) -> int | None:
        rows = self._selected_job_rows()
        return rows[0] if len(rows) == 1 else None

    def _selected_job(self) -> tuple[int, JobResult] | None:
        row = self._selected_job_row()
        if row is None:
            return None
        result = self._results.get(row)
        if not result or not result.transcript or result.error:
            return None
        return row, result

    def _update_job_actions(self):
        busy = self._jobs_busy()
        row = self._selected_job_row()
        result = self._results.get(row) if row is not None else None
        completed = bool(result and result.transcript and not result.error)

        self.remove_job_btn.setEnabled(
            bool(self._selected_job_rows()) and not busy
            and self._processing_row not in self._selected_job_rows()
        )
        self.clear_jobs_btn.setEnabled(self.queue.rowCount() > 0 and not busy)
        self.detect_speakers_btn.setEnabled(
            completed and not busy and not (result.transcript.speakers if result and result.transcript else False)
        )
        self.rename_speakers_btn.setEnabled(
            completed and not busy and bool(result and result.transcript and result.transcript.speakers)
        )
        self.ai_cleanup_btn.setEnabled(completed and not busy)
        self.resume_btn.setEnabled(bool(self._resumable_rows()) and not busy)
        self.cancel_btn.setEnabled(busy)
        self._update_go_button()

    # ---- account ------------------------------------------------------
    def _refresh_account(self):
        self._acct_worker = AccountWorker("me")
        self._acct_worker.done.connect(self._on_account)
        self._acct_worker.error.connect(lambda m: self.account_label.setText(f"  Plaud: {m}  "))
        self._acct_worker.start()

    def _on_account(self, account):
        if account:
            email = account.email
            if len(email) > 36:
                email = email[:16] + "…" + email[-16:]
            self.account_label.setText(f"✓ {account.nickname}  ({email})")
            self.login_btn.setEnabled(False)
            self.logout_btn.setEnabled(True)
            self._auto_list_recordings()
        else:
            self.account_label.setText("Not logged in to Plaud")
            self.login_btn.setEnabled(True)
            self.logout_btn.setEnabled(False)

    def _auto_list_recordings(self):
        """Load the recordings list as soon as we know the session is still good.

        Starting up logged in and then having to press Refresh to see anything
        is a wasted click, so the first confirmed account does it for you —
        once per session, so a later login/logout round trip does not stomp on
        a search or a page the user has since navigated to.
        """
        if self._auto_listed:
            return
        self._auto_listed = True
        self._log("Plaud session active — loading recordings…")
        self.recordings_tab.load("files")

    def _login(self):
        self._log("Opening Plaud login in your browser…")
        self._acct_worker = AccountWorker("login")
        self._acct_worker.done.connect(self._on_account)
        self._acct_worker.error.connect(self._on_login_error)
        self._acct_worker.start()

    def _on_login_error(self, message: str):
        self._log(f"Login failed: {message.splitlines()[0]}")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Login failed")
        box.setText("Could not complete Plaud login.")
        box.setInformativeText(message)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()

    def _logout(self):
        self._acct_worker = AccountWorker("logout")
        self._acct_worker.done.connect(self._on_account)
        self._acct_worker.error.connect(lambda m: QMessageBox.warning(self, "Logout failed", m))
        self._acct_worker.start()

    # ---- setup wizard ---------------------------------------------------
    def _maybe_run_setup_wizard(self):
        """First launch (or a config that lost its answers) starts with setup."""
        if self.settings.setup_complete:
            return
        self._run_setup_wizard(first_run=True)

    def _run_setup_wizard(self, first_run: bool = False):
        wizard = SetupWizard(self.settings, self)
        if not wizard.exec():
            # Cancelling changes nothing, but the wizard has been seen: don't
            # ambush the user with it again on every launch.
            if first_run:
                self.settings.setup_complete = True
                config.save(self.settings)
                self._log("Setup skipped — press Setup in the header to run it again.")
            return
        config.save(self.settings)
        self._after_settings_changed()
        self._log("Setup saved.")
        if not self._auto_listed:
            self._refresh_account()     # they may have just signed in to Plaud

    def _after_settings_changed(self):
        """Re-read settings into the parts of the UI that were built from them."""
        self.options = OptionsPanel(self.settings)
        self._options_scroll.setWidget(self.options)   # takes ownership of the old panel
        self._update_gpu_badge()
        self._update_go_button()

    def _open_glossaries(self):
        dlg = GlossaryLibraryDialog(self, selected=self.settings.glossary_shared_id)
        dlg.exec()
        # The panel's chooser lists names and sizes, so it has to be re-read.
        self.options.refresh_glossaries()

    def _open_settings(self):
        dlg = SettingsDialog(self.settings, self)
        if dlg.exec():
            config.save(self.settings)
            self._log("Settings saved.")
            self._update_gpu_badge()
            self.options.refresh_ai_providers()    # picks up a new default model too
            self.options.refresh_engine_status()   # a key added just now counts
            self.options.refresh_glossaries()

    def _update_gpu_badge(self):
        if cuda_available():
            name = cuda_device_name() or "GPU"
            if torch_cuda_available():
                text = "GPU enabled"
                tip = f"Whisper and speaker detection will use: {name}"
            else:
                text = "GPU enabled"
                tip = (
                    f"Whisper will use: {name}\n"
                    "Speaker detection uses CPU until CUDA PyTorch is installed."
                    "\n\nFor GPU diarization, run in PowerShell:\n"
                    f"{CUDA_TORCH_INSTALL_CMD}"
                )
            self.gpu_badge.setText(text)
            self.gpu_badge.setStyleSheet(GPU_BADGE_ON)
            self.gpu_badge.setToolTip(tip)
        else:
            self.gpu_badge.setText("CPU mode")
            self.gpu_badge.setStyleSheet(GPU_BADGE_OFF)
            self.gpu_badge.setToolTip("No CUDA GPU detected. Transcription will use CPU.")

    # ---- transcription ------------------------------------------------
    def _collect_selection(self) -> list[Recording]:
        if self.tabs.currentWidget() is self.recordings_tab:
            recs = self.recordings_tab.selected()
        else:
            recs = self.local_tab.selected()
        return recs

    def _start(self):
        recs = self._collect_selection()
        if not recs:
            QMessageBox.information(self, "Nothing selected",
                                    "Select Plaud recordings (checkboxes) or add local files first.")
            return

        already_done = self._already_transcribed(recs)
        if already_done:
            names = "\n".join(f"  • {r.display_name}" for r in already_done)
            answer = QMessageBox.warning(
                self,
                "Re-transcribe?",
                f"{len(already_done)} selected recording(s) are already in the queue "
                f"with completed transcripts:\n\n{names}\n\n"
                "Go will re-transcribe them from scratch (Whisper runs again).\n\n"
                "Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        # persist current options into settings for this run
        self.options.apply_to(self.settings)
        engine_err = self.options.ensure_engine_ready()
        if engine_err:
            QMessageBox.warning(self, "Transcription engine", engine_err)
            return
        ai_err = self.options.ensure_ai_cleanup_ready()
        if ai_err:
            QMessageBox.warning(self, "AI Cleanup", ai_err)
            return
        self.options.apply_to(self.settings)
        config.save(self.settings)

        self._job_errors.clear()
        start_row = self.queue.rowCount()
        self._append_queue_rows(recs, start_row)

        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self._log(f"Starting {len(recs)} job(s). Model={self.settings.model}, "
                  f"formats={','.join(self.settings.formats)}, "
                  f"diarization={'on' if self.settings.diarization_enabled else 'off'}.")

        self._worker = TranscriptionWorker(self.settings, recs, start_row=start_row)
        self._worker.started_item.connect(self._on_item_started)
        self._worker.progress_item.connect(self._on_item_progress)
        self._worker.log_item.connect(self._on_item_log)
        self._worker.finished_item.connect(self._on_item_finished)
        self._worker.skipped_item.connect(self._on_item_skipped)
        self._worker.all_finished.connect(self._on_all_finished)
        self._worker.start()

    def _row_progress(self, row: int | None) -> str:
        """Saved-progress summary for a queue row, or "" when there is none."""
        if row is None or row >= len(self._queue_recordings):
            return ""
        if self._processing_row == row:
            return ""
        return resume_store.describe_progress(self._queue_recordings[row])

    def _resumable_rows(self) -> list[int]:
        return [r for r in range(len(self._queue_recordings)) if self._row_progress(r)]

    def _resume_selected(self):
        """Re-run an interrupted job, reusing everything already done.

        The row does not have to be selected: with a single interrupted job
        Resume just picks it, so the button is never a no-op click.
        """
        candidates = self._resumable_rows()
        row = self._selected_job_row()
        if row not in candidates:
            if len(candidates) == 1:
                row = candidates[0]
            elif not candidates:
                self._log("Resume: nothing to resume — no saved progress for any job.")
                QMessageBox.information(
                    self, "Resume",
                    "No job has saved progress. Use Go to run one from the start.",
                )
                return
            else:
                names = ', '.join(
                    self._queue_recordings[r].display_name for r in candidates
                )
                self._log(
                    'Resume: several jobs have saved progress — select the row you want to continue.'
                )
                QMessageBox.information(
                    self, 'Resume',
                    'Select the row of the job you want to resume. Interrupted jobs: ' + names,
                )
                return
        progress = self._row_progress(row)
        rec = self._queue_recordings[row]

        self.options.apply_to(self.settings)
        engine_err = self.options.ensure_engine_ready()
        if engine_err:
            QMessageBox.warning(self, "Transcription engine", engine_err)
            return
        ai_err = self.options.ensure_ai_cleanup_ready()
        if ai_err:
            QMessageBox.warning(self, "AI Cleanup", ai_err)
            return
        config.save(self.settings)

        self._job_errors.clear()
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self._set_status(row, "Resuming…")
        self._log(f"Resuming [{rec.display_name}] — reusing {progress}.")

        # start_row reuses this row rather than appending a duplicate.
        self._worker = TranscriptionWorker(self.settings, [rec], start_row=row)
        self._worker.started_item.connect(self._on_item_started)
        self._worker.progress_item.connect(self._on_item_progress)
        self._worker.log_item.connect(self._on_item_log)
        self._worker.finished_item.connect(self._on_item_finished)
        self._worker.skipped_item.connect(self._on_item_skipped)
        self._worker.all_finished.connect(self._on_all_finished)
        self._worker.start()

    def _already_transcribed(self, recs: list[Recording]) -> list[Recording]:
        done_ids = {
            r.recording.id
            for r in self._results.values()
            if r.transcript and not r.error
        }
        return [r for r in recs if r.id in done_ids]

    def _make_progress_bar(self) -> QProgressBar:
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setFormat("%p%")
        bar.setTextVisible(True)
        bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bar.setStyleSheet(QUEUE_PROGRESS_STYLE)
        return bar

    def _set_progress_cell(self, row: int) -> None:
        wrap = QWidget()
        layout = QHBoxLayout(wrap)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.addWidget(self._make_progress_bar())
        self.queue.setCellWidget(row, 3, wrap)

    def _progress_bar_at(self, row: int) -> QProgressBar | None:
        wrap = self.queue.cellWidget(row, 3)
        if isinstance(wrap, QProgressBar):
            return wrap
        if wrap is not None:
            bar = wrap.findChild(QProgressBar)
            if isinstance(bar, QProgressBar):
                return bar
        return None

    def _queue_item(self, text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        return item

    def _set_output_cell(self, row: int, paths: list[str]) -> None:
        if not paths:
            self._set_output_empty(row)
            return
        wrap = QWidget()
        row_layout = QHBoxLayout(wrap)
        row_layout.setContentsMargins(6, 4, 6, 4)
        row_layout.setSpacing(8)
        names = " · ".join(Path(p).name for p in paths)
        label = ElidingLabel(names)
        label.setToolTip("\n".join(paths))
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        open_btn = QPushButton("Open")
        open_btn.setStyleSheet(OUTPUT_OPEN_BTN_STYLE)
        open_btn.setFixedWidth(56)
        open_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        open_btn.clicked.connect(
            lambda _checked=False, p=list(paths), btn=open_btn: self._open_output_paths(p, btn)
        )
        row_layout.addWidget(label, stretch=1)
        row_layout.addWidget(open_btn, stretch=0, alignment=Qt.AlignmentFlag.AlignVCenter)
        self.queue.setCellWidget(row, 4, wrap)
        self.queue.resizeRowToContents(row)

    def _set_output_empty(self, row: int, text: str = "") -> None:
        self.queue.removeCellWidget(row, 4)
        item = self._queue_item(text or "—")
        self.queue.setItem(row, 4, item)

    def _open_output_paths(self, paths: list[str], button: QPushButton | None = None) -> None:
        existing = [str(Path(p).resolve()) for p in paths if Path(p).is_file()]
        if not existing:
            QMessageBox.warning(self, "Open file", "Output file not found on disk.")
            return
        if len(existing) == 1:
            QDesktopServices.openUrl(QUrl.fromLocalFile(existing[0]))
            return
        menu = QMenu(self)
        for fp in existing:
            menu.addAction(
                Path(fp).name,
                lambda checked=False, url=fp: QDesktopServices.openUrl(QUrl.fromLocalFile(url)),
            )
        if button is not None:
            menu.exec(button.mapToGlobal(button.rect().bottomLeft()))
        else:
            menu.exec()

    def _append_queue_rows(self, recs: list[Recording], start_row: int):
        for i, rec in enumerate(recs):
            row = start_row + i
            self.queue.insertRow(row)
            self._queue_recordings.append(rec)
            self._queue_statuses.append("Queued")
            self.queue.setItem(row, 0, self._queue_item(rec.display_name))
            self.queue.setItem(row, 1, self._queue_item(rec.source.value))
            self.queue.setItem(row, 2, self._queue_item("Queued"))
            self._set_progress_cell(row)
            self._set_output_empty(row, "")
        history.record_many(recs, history.QUEUED)
        self.recordings_tab.refresh_statuses()

    def _populate_queue(self, recs: list[Recording]):
        """Replace the entire queue table (used when restoring persisted state)."""
        self.queue.setRowCount(0)
        self._queue_recordings = list(recs)
        self._queue_statuses = ["Queued"] * len(recs)
        for rec in recs:
            r = self.queue.rowCount()
            self.queue.insertRow(r)
            self.queue.setItem(r, 0, self._queue_item(rec.display_name))
            self.queue.setItem(r, 1, self._queue_item(rec.source.value))
            self.queue.setItem(r, 2, self._queue_item("Queued"))
            self._set_progress_cell(r)
            self._set_output_empty(r, "")

    def _load_persisted_queue(self):
        items = load_queue()
        if not items:
            return
        recs = [r.recording for r, _ in items]
        self._populate_queue(recs)
        self._results.clear()
        for row, (result, status) in enumerate(items):
            self._results[row] = result
            self._queue_statuses[row] = status
            bar = self._progress_bar_at(row)
            if isinstance(bar, QProgressBar):
                bar.setValue(100 if not result.error else 0)
                bar.setFormat("%p%")
            if result.error:
                self._set_status(row, f"Failed: {result.error}", error=True)
                self._set_output_empty(row)
            else:
                self._set_status(row, status)
                self._set_output_cell(row, result.output_paths)
        self._backfill_history(items)
        self._log(f"Restored {len(items)} queued transcription(s) from last session.")
        self._announce_resumable()

    def _backfill_history(self, items: list[tuple[JobResult, str]]) -> None:
        """Seed history from a queue saved before this app kept one.

        Only fills gaps: a recording the history already knows about keeps the
        richer entry written when the job actually ran.
        """
        for result, _status in items:
            if history.get_for(result.recording) is not None:
                continue
            if result.error:
                history.record(result.recording, history.FAILED, error=result.error)
            elif result.transcript:
                history.record(
                    result.recording,
                    history.DONE,
                    outputs=list(result.output_paths),
                    speakers=result.transcript.speaker_count,
                    ai_cleanup=result.ai_cleanup_applied,
                )

    def _announce_resumable(self):
        """Point out interrupted jobs whose work is still on disk."""
        for row, rec in enumerate(self._queue_recordings):
            progress = resume_store.describe_progress(rec)
            if not progress:
                continue
            result = self._results.get(row)
            if result and result.transcript and not result.error:
                continue        # already finished; the checkpoint is just leftovers
            self._set_status(row, f"Interrupted — resumable ({progress})")
            self._log(
                f"[{rec.display_name}] was interrupted — {progress} already saved. "
                f"Select the row and press Resume to continue without redoing it."
            )

    def _persist_queue(self):
        ordered: list[JobResult] = []
        statuses: list[str] = []
        for i in range(self.queue.rowCount()):
            if i not in self._results:
                continue
            ordered.append(self._results[i])
            statuses.append(self._queue_statuses[i] if i < len(self._queue_statuses) else "Done")
        if ordered:
            save_queue(ordered, statuses)
        else:
            clear_queue_file()

    def _set_queue_status(self, row: int, status: str):
        while len(self._queue_statuses) <= row:
            self._queue_statuses.append("")
        self._queue_statuses[row] = status

    # ---- durable history ----------------------------------------------
    def _record_history(self, row: int, state: str, **fields) -> None:
        """Remember what happened to this row's recording, outside the queue.

        The jobs table is disposable — rows get removed and cleared — so the
        Plaud recordings browser reads this instead when it fills its Status
        column, and keeps showing "already transcribed" long afterwards.
        """
        if row >= len(self._queue_recordings):
            return
        history.record(self._queue_recordings[row], state, **fields)
        self.recordings_tab.refresh_statuses()

    def _forget_unfinished(self, recordings: list[Recording]) -> None:
        """Drop history for removed jobs that never produced anything."""
        history.drop_unfinished(recordings)
        self.recordings_tab.refresh_statuses()

    def _snapshot_jobs(self) -> list[tuple[Recording, JobResult | None, str]]:
        items: list[tuple[Recording, JobResult | None, str]] = []
        for i in range(self.queue.rowCount()):
            rec = self._queue_recordings[i]
            result = self._results.get(i)
            status = self._queue_statuses[i] if i < len(self._queue_statuses) else "Queued"
            items.append((rec, result, status))
        return items

    def _restore_jobs(self, items: list[tuple[Recording, JobResult | None, str]]) -> None:
        self.queue.setRowCount(0)
        self._queue_recordings.clear()
        self._queue_statuses.clear()
        self._results.clear()
        self._progress_started.clear()
        for rec, result, status in items:
            row = self.queue.rowCount()
            self.queue.insertRow(row)
            self._queue_recordings.append(rec)
            self._queue_statuses.append(status)
            self.queue.setItem(row, 0, self._queue_item(rec.display_name))
            self.queue.setItem(row, 1, self._queue_item(rec.source.value))
            self.queue.setItem(row, 2, self._queue_item(status))
            self._set_progress_cell(row)
            bar = self._progress_bar_at(row)
            if result and result.error:
                self._set_status(row, f"Failed: {result.error}", error=True)
                self._set_output_empty(row)
                if isinstance(bar, QProgressBar):
                    bar.setValue(0)
            elif result and result.transcript:
                self._results[row] = result
                if isinstance(bar, QProgressBar):
                    bar.setValue(100)
                    bar.setFormat("%p%")
                self._set_status(row, status)
                self._set_output_cell(row, result.output_paths)
            else:
                self._set_output_empty(row, "")
                if isinstance(bar, QProgressBar):
                    bar.setValue(0)

    def _remove_selected_jobs(self):
        rows = self._selected_job_rows()
        if not rows:
            QMessageBox.information(self, "Remove job", "Select one or more jobs to remove.")
            return
        if self._jobs_busy():
            QMessageBox.information(self, "Busy", "Wait for the current job to finish before removing.")
            return
        if self._processing_row is not None and self._processing_row in rows:
            QMessageBox.information(self, "Busy", "Cannot remove a job that is currently running.")
            return
        count = len(rows)
        answer = QMessageBox.question(
            self,
            "Remove job(s)?",
            f"Remove {count} job(s) from the list?\n\n"
            "Output files on disk are kept.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        items = self._snapshot_jobs()
        remove = set(rows)
        dropped = [item[0] for i, item in enumerate(items) if i in remove]
        kept = [item for i, item in enumerate(items) if i not in remove]
        self._restore_jobs(kept)
        self._forget_unfinished(dropped)
        self._persist_queue()
        self._log(f"Removed {count} job(s).")
        self._update_job_actions()

    def _clear_all_jobs(self):
        if self.queue.rowCount() == 0:
            return
        if self._jobs_busy():
            QMessageBox.information(self, "Busy", "Wait for the current job to finish before clearing.")
            return
        answer = QMessageBox.question(
            self,
            "Clear all jobs?",
            "Remove all jobs from the list? Completed transcripts in memory will be "
            "discarded (output files on disk are kept).",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.queue.setRowCount(0)
        self._results.clear()
        cleared = list(self._queue_recordings)
        self._queue_recordings.clear()
        self._queue_statuses.clear()
        self._progress_started.clear()
        clear_queue_file()
        self._forget_unfinished(cleared)
        self._log("All jobs cleared.")
        self._update_job_actions()

    def _on_item_started(self, row: int, msg: str):
        self._processing_row = row
        self._progress_started.pop(row, None)
        self._set_status(row, msg)
        self._record_history(row, history.RUNNING)
        self._update_job_actions()

    def _on_item_progress(self, row: int, frac: float):
        bar = self._progress_bar_at(row)
        if not isinstance(bar, QProgressBar):
            return
        if bar.maximum() == 0:
            bar.setRange(0, 100)
        now = time.monotonic()
        if frac > 0 and row not in self._progress_started:
            self._progress_started[row] = now
        pct = int(frac * 100)
        bar.setValue(pct)
        label = self._progress_labels.get(row, "")
        if 0 < frac < 1 and row in self._progress_started:
            elapsed = now - self._progress_started[row]
            remaining = elapsed / frac * (1 - frac)
            eta = self._fmt_eta(remaining)
            if label:
                bar.setFormat(f"{pct}% · {eta} · {label}")
            else:
                bar.setFormat(f"{pct}% · {eta}")
        elif label:
            bar.setFormat(f"{pct}% · {label}")
        elif frac >= 1:
            bar.setFormat("%p%")
        else:
            bar.setFormat("%p%")

    def _progress_label_from_log(self, msg: str) -> str:
        text = msg.strip()
        for prefix in ("AI Cleanup:", "Glossary:"):
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()
                break
        if len(text) > 42:
            return text[:39] + "…"
        return text

    def _on_item_log(self, row: int, msg: str):
        name = self._queue_recordings[row].display_name if row < len(self._queue_recordings) else row
        self._log(f"[{name}] {msg}")
        self._set_status(row, msg)
        self._progress_labels[row] = self._progress_label_from_log(msg)
        bar = self._progress_bar_at(row)
        if isinstance(bar, QProgressBar) and bar.maximum() > 0:
            pct = bar.value()
            label = self._progress_labels[row]
            if label:
                bar.setFormat(f"{pct}% · {label}")

    def _on_item_skipped(self, row: int):
        """A queued recording the batch never reached because Cancel was hit."""
        self._set_status(row, "Cancelled")
        self._set_queue_status(row, "Cancelled")
        self._set_output_empty(row)
        self._progress_started.pop(row, None)
        self._record_history(row, history.CANCELLED)

    def _on_item_finished(self, row: int, result: JobResult):
        self._results[row] = result
        bar = self._progress_bar_at(row)
        name = self._queue_recordings[row].display_name if row < len(self._queue_recordings) else str(row)
        if result.cancelled:
            self._set_status(row, "Cancelled")
            self._set_queue_status(row, "Cancelled")
            self._set_output_empty(row)
            self._log(f"[{name}] cancelled — no files written.")
            self._progress_started.pop(row, None)
            self._record_history(row, history.CANCELLED)
            self._persist_queue()
            self._update_job_actions()
            return
        if result.error:
            msg = self._format_job_error(result.error)
            self._set_status(row, f"Failed: {msg}", error=True)
            self._set_queue_status(row, f"Failed: {msg}")
            self._set_output_empty(row)
            self._log(f"ERROR [{name}]: {msg}")
            self._job_errors.append(f"{name}: {msg}")
            self._record_history(row, history.FAILED, error=msg)
        else:
            if isinstance(bar, QProgressBar):
                bar.setValue(100)
                bar.setFormat("%p%")
            spk = f" · {result.transcript.speaker_count} speakers" if result.transcript else ""
            status = f"Done{spk}"
            self._set_status(row, status)
            self._set_queue_status(row, status)
            self._set_output_cell(row, result.output_paths)
            self._log(f"[{self._queue_recordings[row].display_name}] wrote "
                      f"{len(result.output_paths)} file(s).")
            self._record_history(
                row,
                history.DONE,
                outputs=list(result.output_paths),
                speakers=result.transcript.speaker_count if result.transcript else 0,
                ai_cleanup=result.ai_cleanup_applied,
                error="",
            )
        self._progress_started.pop(row, None)
        self._persist_queue()
        if not result.error and self._queue_recordings[row].source == Source.PLAUD:
            self.recordings_tab.refresh_cache_status()
        self._update_job_actions()

    def _on_all_finished(self):
        cancelled = bool(self._worker and self._worker.was_cancelled())
        self._processing_row = None
        self.cancel_btn.setEnabled(False)
        self._update_go_button()
        self._update_job_actions()
        if self._job_errors:
            QMessageBox.warning(
                self,
                "Transcription failed",
                "One or more jobs failed:\n\n" + "\n\n".join(self._job_errors),
            )
        self._log("Cancelled — stopped before writing." if cancelled else "All jobs finished.")

    def _cancel(self):
        if self._cleanup_worker and self._cleanup_worker.isRunning():
            self._cleanup_worker.cancel()
            self._log("Cancelling AI Cleanup…")
            return
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self.cancel_btn.setEnabled(False)
            self._log("Cancelling — stopping the running job, nothing will be written.")

    # ---- job actions on completed transcripts --------------------------
    def _detect_speakers_selected(self):
        selected = self._selected_job()
        if not selected:
            QMessageBox.information(self, "Detect speakers", "Select a completed job first.")
            return
        row, result = selected
        if result.transcript.speakers:
            QMessageBox.information(
                self, "Detect speakers", "This job already has speaker labels. Use Rename speakers instead."
            )
            return
        if not self.settings.hf_token:
            QMessageBox.information(
                self,
                "HuggingFace token needed",
                "Add your HuggingFace token in Settings to run speaker detection.\n\n"
                "Using the ElevenLabs engine? Scribe labels speakers while it "
                "transcribes — turn speaker detection on in Settings and run the "
                "job again instead of detecting afterwards.",
            )
            return
        self._run_diarization(row, result)

    def _rename_speakers_selected(self):
        selected = self._selected_job()
        if not selected:
            QMessageBox.information(self, "Rename speakers", "Select a completed job with speakers first.")
            return
        row, result = selected
        if not result.transcript.speakers:
            QMessageBox.information(
                self, "Rename speakers", "This job has no speaker labels yet. Use Detect speakers first."
            )
            return
        self._rename_speakers(row, result)

    def _rename_speakers(self, row: int, result: JobResult):
        dlg = SpeakerRenameDialog(result.transcript, self)
        if not dlg.exec():
            return
        renames = dlg.renames()
        if not renames:
            return
        apply_speaker_renames(result.transcript, renames)
        try:
            runner = JobRunner(self.settings)
            previous = list(result.output_paths)
            paths = runner.write_outputs(result.transcript, index=row + 1)
            result.output_paths = paths
            self._set_output_cell(row, paths)
            self._log(f"Re-exported with renamed speakers: {len(paths)} file(s).")
            self._log_superseded(remove_superseded_outputs(previous, paths))
            self._record_history(row, history.DONE, outputs=list(paths))
            self._persist_queue()
        except Exception as e:
            QMessageBox.warning(self, "Re-export failed", str(e))

    def _log_superseded(self, removed: list[str]) -> None:
        for path in removed:
            self._log(f"Removed superseded export: {Path(path).name}")

    def _ai_cleanup_selected(self):
        selected = self._selected_job()
        if not selected:
            QMessageBox.information(self, "AI Cleanup", "Select a completed job first.")
            return
        row, result = selected
        ensure_original_snapshot(result)
        dlg = AICleanupDialog(
            self.settings,
            has_original=result.original_transcript is not None,
            has_cleaned=result.ai_cleanup_applied,
            glossary_id=result.glossary_id,
            parent=self,
        )
        if not dlg.exec():
            return
        choice = dlg.result_choice()
        if not choice:
            return
        self._run_ai_cleanup(
            row,
            result,
            provider=choice.provider,
            model=choice.model,
            use_original=choice.use_original,
            glossary_id=choice.glossary_id,
        )

    def _run_diarization(self, row: int, result: JobResult):
        if self._jobs_busy():
            QMessageBox.information(self, "Busy", "Another job action is already running.")
            return
        self._processing_row = row
        self.start_btn.setEnabled(False)
        self._set_status(row, "Detecting speakers…")
        self._diar_worker = DiarizationWorker(self.settings, row, result)
        self._diar_worker.log_item.connect(self._on_item_log)
        self._diar_worker.progress_item.connect(self._on_item_progress)
        self._diar_worker.done.connect(self._on_diarization_done)
        self._diar_worker.error.connect(self._on_diarization_error)
        self._diar_worker.finished.connect(self._update_job_actions)
        self._diar_worker.start()
        self._update_job_actions()

    def _begin_indeterminate_progress(self, row: int, label: str) -> None:
        bar = self._progress_bar_at(row)
        if isinstance(bar, QProgressBar):
            bar.setRange(0, 0)
            bar.setFormat(label)
        self._progress_started.pop(row, None)
        self._progress_labels[row] = label

    def _end_determinate_progress(self, row: int, *, value: int = 100) -> None:
        bar = self._progress_bar_at(row)
        if isinstance(bar, QProgressBar):
            bar.setRange(0, 100)
            bar.setValue(value)
            bar.setFormat("%p%")
        self._progress_started.pop(row, None)
        self._progress_labels.pop(row, None)

    def _run_ai_cleanup(
        self,
        row: int,
        result: JobResult,
        *,
        provider: str,
        model: str,
        use_original: bool,
        glossary_id: str = "",
    ):
        if self._jobs_busy():
            QMessageBox.information(self, "Busy", "Another job action is already running.")
            return
        self._processing_row = row
        self.start_btn.setEnabled(False)
        self._set_status(row, "AI Cleanup…")
        self._begin_indeterminate_progress(row, "AI Cleanup…")
        self._cleanup_worker = CleanupWorker(
            self.settings,
            row,
            result,
            provider=provider,
            model=model,
            use_original=use_original,
            glossary_id=glossary_id,
        )
        self._cleanup_worker.log_item.connect(self._on_item_log)
        self._cleanup_worker.progress_item.connect(self._on_item_progress)
        self._cleanup_worker.done.connect(self._on_cleanup_done)
        self._cleanup_worker.error.connect(self._on_cleanup_error)
        self._cleanup_worker.cancelled.connect(self._on_cleanup_cancelled)
        self._cleanup_worker.finished.connect(self._update_job_actions)
        self._cleanup_worker.start()
        self.cancel_btn.setEnabled(True)
        self._update_job_actions()

    def _on_diarization_done(self, row: int, result: JobResult):
        self._processing_row = None
        self._results[row] = result
        spk = result.transcript.speaker_count if result.transcript else 0
        self._set_status(row, f"Done · {spk} speakers")
        self._set_queue_status(row, f"Done · {spk} speakers")
        self._set_output_cell(row, result.output_paths)
        self._log(f"Speaker detection finished: {spk} speaker(s), re-exported.")
        self._record_history(
            row, history.DONE, outputs=list(result.output_paths), speakers=spk
        )
        self._persist_queue()
        self._update_job_actions()
        dlg = SpeakerRenameDialog(result.transcript, self)
        if dlg.exec():
            renames = dlg.renames()
            if renames:
                apply_speaker_renames(result.transcript, renames)
                try:
                    runner = JobRunner(self.settings)
                    previous = list(result.output_paths)
                    paths = runner.write_outputs(result.transcript, index=row + 1)
                    result.output_paths = paths
                    self._set_output_cell(row, paths)
                    self._log("Re-exported with renamed speakers.")
                    self._log_superseded(remove_superseded_outputs(previous, paths))
                    self._record_history(row, history.DONE, outputs=list(paths))
                    self._persist_queue()
                except Exception as e:
                    QMessageBox.warning(self, "Re-export failed", str(e))

    def _on_diarization_error(self, row: int, message: str):
        self._processing_row = None
        self._set_status(row, f"Failed: {message}", error=True)
        self._log(f"ERROR speaker detection: {message}")
        # The transcript itself survived; only the extra pass failed, so the
        # recording stays "done" and the reason rides along in the tooltip.
        self._record_history(row, history.DONE, error=f"Speaker detection failed: {message}")
        QMessageBox.warning(self, "Speaker detection failed", message)
        self._update_job_actions()

    def _on_cleanup_done(self, row: int, result: JobResult):
        self._processing_row = None
        self._results[row] = result
        self._end_determinate_progress(row, value=100)
        status = self._queue_statuses[row] if row < len(self._queue_statuses) else "Done"
        if "speakers" not in status.lower():
            status = f"{status} · cleaned"
        else:
            status = status.replace("Done", "Done · cleaned", 1)
        self._set_status(row, status)
        self._set_queue_status(row, status)
        self._set_output_cell(row, result.output_paths)
        self._log("AI Cleanup finished and files re-exported.")
        self._record_history(
            row,
            history.DONE,
            outputs=list(result.output_paths),
            ai_cleanup=True,
            speakers=result.transcript.speaker_count if result.transcript else 0,
        )
        self._persist_queue()
        self._update_job_actions()

    def _on_cleanup_cancelled(self, row: int):
        self._processing_row = None
        self._end_determinate_progress(row, value=0)
        prev = self._queue_statuses[row] if row < len(self._queue_statuses) else "Done"
        if "cleaned" in prev.lower():
            status = prev.replace(" · cleaned", "").replace("cleaned", "").strip(" ·") or "Done"
        else:
            status = prev if prev and prev != "Queued" else "Done"
        self._set_status(row, status)
        self._set_queue_status(row, status)
        self._log("AI Cleanup cancelled — transcript unchanged.")
        self._update_job_actions()

    def _on_cleanup_error(self, row: int, message: str):
        self._processing_row = None
        self._end_determinate_progress(row, value=0)
        self._set_status(row, f"Cleanup failed: {message}", error=True)
        self._log(f"ERROR AI Cleanup: {message}")
        self._record_history(row, history.DONE, error=f"AI Cleanup failed: {message}")
        QMessageBox.warning(self, "AI Cleanup failed", message)
        self._update_job_actions()

    # ---- helpers ------------------------------------------------------
    def _update_go_button(self):
        if self._jobs_busy():
            self.start_btn.setEnabled(False)
            return
        count = len(self._collect_selection())
        self.start_btn.setEnabled(count > 0)

    @staticmethod
    def _fmt_eta(seconds: float) -> str:
        seconds = max(0, int(seconds))
        if seconds < 60:
            return f"{seconds}s"
        minutes, secs = divmod(seconds, 60)
        if minutes < 60:
            return f"{minutes}m {secs:02d}s"
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h {minutes:02d}m"

    def _format_job_error(self, error: str) -> str:
        lower = error.lower()
        if "cublas" in lower or "cudnn" in lower or "cudart" in lower:
            return (
                f"{error}\n\n"
                "CUDA libraries are missing on this PC. The app will retry on CPU next time, "
                "or install them with: pip install nvidia-cublas-cu12"
            )
        return error

    def _set_status(self, row: int, text: str, *, error: bool = False):
        if row >= self.queue.rowCount():
            return
        item = self._queue_item(text)
        item.setToolTip(text)
        if error:
            item.setForeground(QColor("#ff8a80"))
        self.queue.setItem(row, 2, item)
        self.queue.resizeRowToContents(row)

    def _log(self, msg: str):
        # Real clock time on every line: without it a 5-second step and a
        # 5-minute stall look identical, and the "[name]" prefix on job lines
        # is the recording's name, not a time.
        self.log.appendPlainText(f"{time.strftime('%H:%M:%S')}  {msg}")
