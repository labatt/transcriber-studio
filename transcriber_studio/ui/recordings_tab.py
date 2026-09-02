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

from .. import config, history, name_store
from .. import resume as resume_store
from ..audio_cache import attach_if_cached, audio_status_label
from ..config import Settings
from ..models import Recording, Source
from ..workers import ListWorker, RenameWorker
from .theme import qcolor

COLS = ["", "Name", "Date", "Duration", "Audio", "Status"]
# ~680px total: the default width of the left pane. The Status column measures
# its own longest label instead (fonts and DPI make a fixed number a guess).
DEFAULT_COL_WIDTHS = [28, 240, 92, 78, 72, 170]
NAME_COL = 1
STATUS_COL = 5
#: Tooltips are plain text; a literal is clearer here than an escape.
LINE_BREAK = chr(10)
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


def _read_only(item: QTableWidgetItem) -> QTableWidgetItem:
    """Take the edit flag off a cell.

    Qt gives every item ItemIsEditable by default and relies on the view's edit
    triggers to hold it back. The Name column needs those triggers, so every
    other column has to say no for itself.
    """
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return item


class RecordingsTab(QWidget):
    selection_changed = Signal()

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.s = settings
        self.page = 1
        self._worker: ListWorker | None = None
        self._recordings: list[Recording] = []
        self._rename_workers: dict[str, RenameWorker] = {}

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
        # Only the Name cell is editable, and only by asking for it: a stray
        # click in a list people mostly tick boxes in should not start an edit.
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
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
        # A name the user gave a recording outranks the one Plaud sends back —
        # including when the push failed, which is exactly when the two differ.
        name_store.load(force=True)
        for rec in self._recordings:
            rec.name = name_store.name_for(rec.id, rec.name)
        self.table.setRowCount(0)
        history.load(force=True)        # pick up anything finished since last look
        # Populating writes into the Name column, and a write there is
        # indistinguishable from someone typing in it.
        self.table.blockSignals(True)
        for rec in self._recordings:
            self._add_row(rec)
        self.table.blockSignals(False)
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
        self.table.setItem(r, NAME_COL, self._name_item(rec))
        self.table.setItem(r, 2, _read_only(QTableWidgetItem(rec.date)))
        self.table.setItem(r, 3, _read_only(QTableWidgetItem(rec.duration)))
        self.table.setItem(r, 4, _read_only(QTableWidgetItem(audio_status_label(rec))))
        self.table.setItem(r, 5, self._status_item(rec))

    # ---- renaming -----------------------------------------------------
    def _name_item(self, rec: Recording) -> QTableWidgetItem:
        """The Name cell, editable, showing whether Plaud has the name yet."""
        item = QTableWidgetItem(rec.display_name)
        editable = rec.source == Source.PLAUD
        flags = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
        if editable:
            flags |= Qt.ItemFlag.ItemIsEditable
        item.setFlags(flags)

        local = name_store.get(rec.id)
        if local is None:
            item.setToolTip(
                "Double-click to rename." if editable else rec.display_name
            )
            return item
        if local.pushed:
            item.setToolTip(
                LINE_BREAK.join([
                    "Renamed — Plaud has this name too.",
                    f"Originally “{local.original or 'unknown'}”.",
                ])
            )
        else:
            # Said plainly rather than shown as an error: the rename worked,
            # it just has not reached Plaud.
            item.setForeground(qcolor("warn"))
            item.setToolTip(
                LINE_BREAK.join([
                    "Renamed here, but not on Plaud yet.",
                    f"Originally “{local.original or 'unknown'}”.",
                    "Rename it again to retry the push.",
                ])
            )
        return item

    def _on_name_edited(self, row: int):
        item = self.table.item(row, NAME_COL)
        holder = self.table.item(row, 0)
        if item is None or holder is None:
            return
        rec: Recording = holder.data(Qt.ItemDataRole.UserRole)
        new_name = item.text().strip()
        previous = rec.display_name
        if not new_name:
            # An empty name is a slip, not an instruction. Put it back.
            self._set_name_cell(row, rec)
            self.status.setText("A recording needs a name.")
            return
        if new_name == previous:
            return

        # Local first, always: the push is the part that can fail, and a rename
        # the user can see is worth more than one that is merely in sync.
        pushing = bool(self.s.plaud_rename_push and self.s.plaud_web_token.strip())
        name_store.record(rec.id, new_name, original=previous, pushed=False)
        rec.name = new_name
        holder.setData(Qt.ItemDataRole.UserRole, rec)
        self._set_name_cell(row, rec)

        if not pushing:
            self.status.setText(
                f"Renamed “{new_name}” here. Turn on Settings → Plaud rename "
                "to send names to Plaud as well."
            )
            return
        self.status.setText(f"Renaming “{new_name}” on Plaud…")
        worker = RenameWorker(self.s, rec.id, new_name, parent=self)
        worker.done.connect(self._on_rename_pushed)
        worker.error.connect(self._on_rename_failed)
        # Held so the thread is not collected mid-flight; dropped when it ends.
        self._rename_workers[rec.id] = worker
        worker.finished.connect(lambda rid=rec.id: self._rename_workers.pop(rid, None))
        worker.start()

    def _set_name_cell(self, row: int, rec: Recording):
        """Repaint one Name cell without the write looking like a fresh edit."""
        self.table.blockSignals(True)
        self.table.setItem(row, NAME_COL, self._name_item(rec))
        self.table.blockSignals(False)

    def _row_for(self, file_id: str) -> int | None:
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item and item.data(Qt.ItemDataRole.UserRole).id == file_id:
                return r
        return None

    def _on_rename_pushed(self, file_id: str):
        row = self._row_for(file_id)
        if row is not None:
            self._set_name_cell(row, self.table.item(row, 0).data(Qt.ItemDataRole.UserRole))
        self.status.setText("Renamed on Plaud.")

    def _on_rename_failed(self, file_id: str, message: str):
        row = self._row_for(file_id)
        if row is not None:
            self._set_name_cell(row, self.table.item(row, 0).data(Qt.ItemDataRole.UserRole))
        # The name is kept either way; only the push failed, and saying which
        # is the difference between a warning and an apparent data loss.
        first_line = message.strip().splitlines()[0] if message.strip() else "unknown error"
        self.status.setText(f"Renamed here, but Plaud refused: {first_line}")

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
        item = _read_only(QTableWidgetItem(text))
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
        elif col == NAME_COL:
            self._on_name_edited(row)

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
