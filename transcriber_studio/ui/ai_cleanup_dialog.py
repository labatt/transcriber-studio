# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Dialog to choose provider, model, and transcript source for AI cleanup."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from .. import ai_providers, config
from ..config import Settings
from .glossary_dialog import GlossaryLibraryDialog, populate_glossary_combo
from .theme import SheetDialog


@dataclass(frozen=True)
class CleanupDialogResult:
    provider: str
    model: str
    use_original: bool
    glossary_id: str = ""


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


class AICleanupDialog(SheetDialog):
    def __init__(
        self,
        settings: Settings,
        *,
        has_original: bool,
        has_cleaned: bool,
        glossary_id: str | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.settings = settings
        # None => this job has not chosen one, so it follows the app default.
        self._glossary_id = (
            settings.glossary_shared_id if glossary_id is None else glossary_id
        )
        self._model_worker: _ModelListWorker | None = None
        self._result: CleanupDialogResult | None = None
        self.setWindowTitle("AI Cleanup")
        self.setMinimumWidth(480)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        provider_row = QWidget()
        prow = QHBoxLayout(provider_row)
        prow.setContentsMargins(0, 0, 0, 0)
        self.provider = QComboBox()
        providers = ai_providers.configured_providers(settings)
        if not providers:
            self.provider.addItem("(no providers configured)", "")
        else:
            for pid in providers:
                self.provider.addItem(ai_providers.PROVIDER_LABELS.get(pid, pid), pid)
            wanted = config.cleanup_provider(settings)
            if wanted:
                idx = self.provider.findData(wanted)
                if idx >= 0:
                    self.provider.setCurrentIndex(idx)
        self.refresh_models_btn = QPushButton("Refresh")
        self.refresh_models_btn.clicked.connect(self._load_models)
        prow.addWidget(self.provider, stretch=1)
        prow.addWidget(self.refresh_models_btn)
        form.addRow("Provider:", provider_row)

        self.model = QComboBox()
        self.model.setMinimumWidth(280)
        form.addRow("Model:", self.model)

        glossary_row = QWidget()
        grow = QHBoxLayout(glossary_row)
        grow.setContentsMargins(0, 0, 0, 0)
        self.glossary = QComboBox()
        self.glossary.setMinimumWidth(280)
        self.glossary.setToolTip(
            "Read the shared glossary's terms before cleaning, and write back "
            "the ones this transcript turns up."
        )
        populate_glossary_combo(self.glossary, self._glossary_id)
        self.manage_glossaries = QPushButton("Manage…")
        self.manage_glossaries.setAutoDefault(False)
        self.manage_glossaries.clicked.connect(self._manage_glossaries)
        grow.addWidget(self.glossary, stretch=1)
        grow.addWidget(self.manage_glossaries)
        form.addRow("Shared glossary:", glossary_row)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: gray;")
        form.addRow(self.status)

        layout.addLayout(form)

        layout.addWidget(QLabel("Transcript to clean:"))
        self.src_original = QRadioButton("Original transcription (before any AI cleanup)")
        self.src_current = QRadioButton("Current transcription")
        self.src_original.setEnabled(has_original)
        if has_cleaned:
            self.src_current.setChecked(True)
        elif has_original:
            self.src_original.setChecked(True)
        else:
            self.src_current.setChecked(True)
            self.src_original.setEnabled(False)
        layout.addWidget(self.src_original)
        layout.addWidget(self.src_current)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.provider.currentIndexChanged.connect(lambda _i: self._load_models())
        self._load_models()

    def _manage_glossaries(self):
        current = self.glossary.currentData() or ""
        dlg = GlossaryLibraryDialog(self, selected=current)
        dlg.exec()
        populate_glossary_combo(self.glossary, dlg.selected_id() or current)

    def result_choice(self) -> CleanupDialogResult | None:
        return self._result

    def _set_busy(self, busy: bool):
        self.provider.setEnabled(not busy)
        self.model.setEnabled(not busy)
        self.refresh_models_btn.setEnabled(not busy)

    def _load_models(self):
        provider = self.provider.currentData()
        if not provider:
            self.model.clear()
            self.status.setText("Add API keys in Settings first.")
            return
        if self._model_worker and self._model_worker.isRunning():
            return
        self._set_busy(True)
        self.model.clear()
        self.model.addItem("Loading…", "")
        self.status.setText("Fetching models…")
        self._model_worker = _ModelListWorker(self.settings, provider, self)
        self._model_worker.ok.connect(self._on_models_loaded)
        self._model_worker.failed.connect(self._on_models_failed)
        self._model_worker.finished.connect(lambda: self._set_busy(False))
        self._model_worker.start()

    def _on_models_loaded(self, models: list[str]):
        saved = config.cleanup_model(self.settings)
        self.model.clear()
        if not models:
            self.model.addItem("(no models returned)", "")
            self.status.setText("Provider returned no models.")
            return
        for m in models:
            self.model.addItem(m, m)
        for candidate in (saved, self.settings.ai_default_model):
            if not candidate:
                continue
            idx = self.model.findData(candidate)
            if idx >= 0:
                self.model.setCurrentIndex(idx)
                break
        self.status.setText(f"{len(models)} model(s) available.")

    def _on_models_failed(self, message: str):
        self.model.clear()
        self.model.addItem("(failed to load models)", "")
        self.status.setText(message)

    def _accept(self):
        provider = self.provider.currentData()
        model = self.model.currentData()
        if not provider:
            QMessageBox.warning(self, "AI Cleanup", "Select a provider.")
            return
        if not model or str(model).startswith("("):
            QMessageBox.warning(self, "AI Cleanup", "Select a model.")
            return
        use_original = self.src_original.isChecked() and self.src_original.isEnabled()
        self._result = CleanupDialogResult(
            provider, model, use_original, self.glossary.currentData() or ""
        )
        self.accept()
