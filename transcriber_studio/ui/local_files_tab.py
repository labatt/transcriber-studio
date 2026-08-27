# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Import local audio files (drag-and-drop or picker) for transcription."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..audio_utils import AUDIO_EXTS, probe
from ..models import Recording, Source

FILTER = "Audio files (*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.opus *.wma *.mp4 *.m4b);;All files (*.*)"


class LocalFilesTab(QWidget):
    selection_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        root = QVBoxLayout(self)

        root.addWidget(QLabel("Drag audio files here, or use “Add files”. Multiple files supported."))

        bar = QHBoxLayout()
        self.add_btn = QPushButton("Add files…")
        self.remove_btn = QPushButton("Remove selected")
        self.clear_btn = QPushButton("Clear")
        bar.addWidget(self.add_btn)
        bar.addWidget(self.remove_btn)
        bar.addWidget(self.clear_btn)
        bar.addStretch()
        root.addLayout(bar)

        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        root.addWidget(self.list)

        self.status = QLabel("0 files.")
        root.addWidget(self.status)

        self.list.itemSelectionChanged.connect(self.selection_changed.emit)
        self.add_btn.clicked.connect(self._pick)
        self.remove_btn.clicked.connect(self._remove)
        self.clear_btn.clicked.connect(self._clear)

    # ---- drag & drop ----
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls()]
        self._add_paths(paths)

    # ---- actions ----
    def _pick(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Choose audio files", "", FILTER)
        self._add_paths(files)

    def _add_paths(self, paths: list[str]):
        existing = {self.list.item(i).data(Qt.ItemDataRole.UserRole).local_path
                    for i in range(self.list.count())}
        added = 0
        for p in paths:
            path = Path(p)
            if not path.is_file() or path.suffix.lower() not in AUDIO_EXTS:
                continue
            if str(path) in existing:
                continue
            meta = probe(str(path))
            dur = meta.get("duration", 0.0)
            rec = Recording(
                source=Source.LOCAL,
                id=str(path),
                name=path.stem,
                local_path=str(path),
                duration=_fmt_dur(dur),
                duration_seconds=dur,
            )
            item = QListWidgetItem(f"{path.name}   ({_fmt_dur(dur)}, {meta.get('channels',1)} ch)")
            item.setData(Qt.ItemDataRole.UserRole, rec)
            self.list.addItem(item)
            added += 1
        self._update_status()

    def _remove(self):
        for item in self.list.selectedItems():
            self.list.takeItem(self.list.row(item))
        self._update_status()

    def _clear(self):
        self.list.clear()
        self._update_status()

    def _update_status(self):
        self.status.setText(f"{self.list.count()} file(s).")
        self.selection_changed.emit()

    def selected(self) -> list[Recording]:
        """List items highlighted in the local-files tab."""
        return [item.data(Qt.ItemDataRole.UserRole) for item in self.list.selectedItems()]


def _fmt_dur(seconds: float) -> str:
    seconds = int(seconds)
    h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"
