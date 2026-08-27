# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plaud recordings browser: list / recent / search, with multi-select."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFontMetrics
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import config, history
from .. import resume as resume_store
from ..audio_cache import attach_if_cached, audio_status_label
from ..config import Settings
from ..models import Recording
from ..workers import ListWorker
from .theme import qcolor

COLS = ["", "Name", "Date", "Duration", "Audio", "Status"]
# ~680px total: the default width of the left pane. The Status column measures
# its own longest label instead (fonts and DPI make a fixed number a guess).
DEFAULT_COL_WIDTHS = [28, 240, 92, 78, 72, 170]
STATUS_COL = 5
MIN_COL_WIDTH = 24
NAME_MIN_WIDTH = 120

# Status role per state; the actual colour follows the light/dark palette.
STATE_ROLES = {
    history.DONE: "good",
    history.FAILED: "bad",
    history.RUNNING: "info",
    history.QUEUED: "muted",
    history.CANCELLED: "muted",
}


class RecordingsTab(QWidget):
    selection_changed = Signal()

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.s = settings
        self.page = 1
        self._worker: ListWorker | None = None
        self._recordings: list[Recording] = []

        root = QVBoxLayout(self)

        bar = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh")
        self.recent_btn = QPushButton("Last 7 days")
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search by name…")
        self.search_btn = QPushButton("Search")
        bar.addWidget(self.refresh_btn)
        bar.addWidget(self.recent_btn)
        bar.addWidget(self.search_edit, stretch=1)
        bar.addWidget(self.search_btn)
        root.addLayout(bar)

        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._setup_columns()
        root.addWidget(self.table)

        page_row = QHBoxLayout()
        self.select_all = QPushButton("Select all")
        self.select_none = QPushButton("Clear")
        self.prev_btn = QPushButton("‹ Prev")
        self.page_spin = QSpinBox(); self.page_spin.setRange(1, 9999); self.page_spin.setValue(1)
        self.next_btn = QPushButton("Next ›")
        self.status = QLabel("Not loaded.")
        page_row.addWidget(self.select_all)
        page_row.addWidget(self.select_none)
        page_row.addStretch()
        page_row.addWidget(self.status)
        page_row.addStretch()
        page_row.addWidget(self.prev_btn)
        page_row.addWidget(QLabel("Page"))
        page_row.addWidget(self.page_spin)
        page_row.addWidget(self.next_btn)
        root.addLayout(page_row)

        self.table.cellChanged.connect(self._on_cell_changed)
        self.refresh_btn.clicked.connect(lambda: self.load("files"))
        self.recent_btn.clicked.connect(lambda: self.load("recent"))
        self.search_btn.clicked.connect(lambda: self.load("search"))
        self.search_edit.returnPressed.connect(lambda: self.load("search"))
        self.select_all.clicked.connect(lambda: self._set_all(True))
        self.select_none.clicked.connect(lambda: self._set_all(False))
        self.prev_btn.clicked.connect(self._prev)
        self.next_btn.clicked.connect(self._next)
        self.page_spin.valueChanged.connect(self._goto_page)

    # ---- columns ------------------------------------------------------
    def _setup_columns(self):
        """Every column drag-resizable, at the widths this user last chose.

        Interactive (rather than Stretch/ResizeToContents) is what makes a
        header handle draggable; the cost is that nothing auto-fills, so the
        widths are remembered between sessions instead.
        """
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hh.setStretchLastSection(False)   # a stretched last column cannot be dragged
        hh.setMinimumSectionSize(MIN_COL_WIDTH)
        hh.setTextElideMode(Qt.TextElideMode.ElideRight)
        saved = list(self.s.recordings_col_widths or [])
        for col in range(len(COLS)):
            width = saved[col] if col < len(saved) and saved[col] > 0 else self._default_width(col)
            hh.resizeSection(col, max(MIN_COL_WIDTH, int(width)))
        self._sized_to_fit = bool(saved)

        # Dragging a handle fires per pixel; save once the user lets go.
        self._width_save_timer = QTimer(self)
        self._width_save_timer.setSingleShot(True)
        self._width_save_timer.setInterval(600)
        self._width_save_timer.timeout.connect(self._save_col_widths)
        hh.sectionResized.connect(lambda *_: self._width_save_timer.start())

    def _default_width(self, col: int) -> int:
        """Start width for a column the user has not sized yet."""
        if col != STATUS_COL:
            return DEFAULT_COL_WIDTHS[col]
        # Measured, so "✓ Transcribed + AI cleanup" is readable at any DPI or
        # font size rather than arriving elided.
        fm = QFontMetrics(self.table.font())
        widest = max(fm.horizontalAdvance(label) for label in history.possible_labels())
        return max(DEFAULT_COL_WIDTHS[col], widest + 24)

    def resizeEvent(self, event):
        """First real layout with no remembered widths: Name takes the slack.

        Stretch would do this automatically but takes the drag handle away, so
        the fill happens once — on resize rather than show, which is when the
        table finally knows how wide it is — and stays user-resizable after.
        """
        super().resizeEvent(event)
        if self._sized_to_fit:
            return
        viewport = self.table.viewport().width()
        if viewport <= 0:
            return
        self._sized_to_fit = True
        spare = viewport - sum(self.table.columnWidth(c) for c in range(len(COLS)))
        # Negative slack means a narrow pane: give Name less rather than hand
        # the user a horizontal scrollbar on the first look at the list.
        self.table.setColumnWidth(1, max(NAME_MIN_WIDTH, self.table.columnWidth(1) + spare))

    def _save_col_widths(self):
        widths = [self.table.columnWidth(c) for c in range(len(COLS))]
        if widths == list(self.s.recordings_col_widths or []):
            return
        self.s.recordings_col_widths = widths
        config.save(self.s)

    # ------------------------------------------------------------------
    def load(self, mode: str):
        self.status.setText("Loading…")
        self.refresh_btn.setEnabled(False)
        kw = self.search_edit.text().strip()
        page = self.page_spin.value()
        self._worker = ListWorker(mode, self.s, page=page, keyword=kw)
        self._worker.done.connect(self._on_loaded)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_loaded(self, recs: list[Recording]):
        self.refresh_btn.setEnabled(True)
        self._recordings = [attach_if_cached(r) for r in recs]
        self.table.setRowCount(0)
        history.load(force=True)        # pick up anything finished since last look
        for rec in self._recordings:
            self._add_row(rec)
        cached = sum(1 for r in self._recordings if r.local_path)
        extra = f" · {cached} cached locally" if cached else ""
        self.status.setText(f"{len(recs)} recording(s){extra}.")
        self.selection_changed.emit()

    def _on_error(self, msg: str):
        self.refresh_btn.setEnabled(True)
        self.status.setText(f"Error: {msg}")

    def _add_row(self, rec: Recording):
        r = self.table.rowCount()
        self.table.insertRow(r)
        chk = QTableWidgetItem()
        chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
        chk.setCheckState(Qt.CheckState.Unchecked)
        chk.setData(Qt.ItemDataRole.UserRole, rec)
        self.table.setItem(r, 0, chk)
        self.table.setItem(r, 1, QTableWidgetItem(rec.display_name))
        self.table.setItem(r, 2, QTableWidgetItem(rec.date))
        self.table.setItem(r, 3, QTableWidgetItem(rec.duration))
        self.table.setItem(r, 4, QTableWidgetItem(audio_status_label(rec)))
        self.table.setItem(r, 5, self._status_item(rec))

    # ---- status column ------------------------------------------------
    @staticmethod
    def _status_for(rec: Recording) -> tuple[str, str, QColor | None]:
        """(text, tooltip, colour) for everything the app has done with a recording.

        Read from the history store, which outlives the jobs table, so a
        transcript made months ago still shows here after its job row is gone.
        """
        entry = history.get_for(rec)
        if entry is not None:
            return entry.label, entry.tooltip(), qcolor(STATE_ROLES.get(entry.state, "muted"))
        progress = resume_store.describe_progress(rec)
        if progress:
            return (
                "Interrupted",
                f"An earlier run stopped partway: {progress} already saved.\n"
                "Queue it again and press Resume to continue where it left off.",
                qcolor("warn"),
            )
        return "—", "Not processed yet.", None

    def _status_item(self, rec: Recording) -> QTableWidgetItem:
        text, tip, color = self._status_for(rec)
        item = QTableWidgetItem(text)
        item.setToolTip(tip)
        if color is not None:
            item.setForeground(color)
        return item

    def refresh_statuses(self):
        """Re-read the history store and repaint the Status column."""
        history.load(force=True)
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if not item:
                continue
            rec = item.data(Qt.ItemDataRole.UserRole)
            self.table.setItem(r, 5, self._status_item(rec))

    def _on_cell_changed(self, row: int, col: int):
        if col == 0:
            self.selection_changed.emit()

    def _set_all(self, checked: bool):
        self.table.blockSignals(True)
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for r in range(self.table.rowCount()):
            self.table.item(r, 0).setCheckState(state)
        self.table.blockSignals(False)
        self.selection_changed.emit()

    def _prev(self):
        if self.page_spin.value() > 1:
            self.page_spin.setValue(self.page_spin.value() - 1)

    def _next(self):
        self.page_spin.setValue(self.page_spin.value() + 1)

    def _goto_page(self):
        self.load("files")

    def selected(self) -> list[Recording]:
        out = []
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item and item.checkState() == Qt.CheckState.Checked:
                out.append(item.data(Qt.ItemDataRole.UserRole))
        return out

    def refresh_cache_status(self):
        """Update the Audio column after files are downloaded to the local cache."""
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            audio_item = self.table.item(r, 4)
            if not item or not audio_item:
                continue
            rec = item.data(Qt.ItemDataRole.UserRole)
            attach_if_cached(rec)
            item.setData(Qt.ItemDataRole.UserRole, rec)
            audio_item.setText(audio_status_label(rec))
