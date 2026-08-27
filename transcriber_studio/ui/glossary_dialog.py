# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Manage the library of shared glossaries.

A shared glossary is a named vocabulary several jobs point at: each run reads
the accumulated terms before cleanup and writes back what its own transcript
turned up. This dialog is where they are created, curated by hand, merged into
each other, and thrown away.

Merges dedupe. Where two sources describe the same thing differently — one
calls a term a product, the other a concept — the entry is tagged instead of
one reading being picked silently, and the tagged rows can be filtered down to
on their own so they can be edited or deleted.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .. import glossary_merge, glossary_store
from ..glossary_merge import Part
from ..glossary_store import SharedGlossary
from .theme import SheetDialog, muted_small, qcolor

# Editable columns, then the read-only conflict column the merges write.
TERM_COLS = ["Term", "Type", "Variants (comma separated)", "Needs review"]
TERM_FIELDS = [("canonical", "text"), ("type", "text"), ("variants", "list")]
SPEAKER_COLS = ["Label", "Name", "Role", "Needs review"]
SPEAKER_FIELDS = [("label", "text"), ("name", "text"), ("role", "text")]
# Hand-entered speakers outrank whatever a single recording guessed, which is
# what merge_speakers scores on.
SPEAKER_DEFAULTS = {"confidence": "high", "raw_intro": ""}

PER_RECORDING_LABEL = "Per recording (no shared glossary)"
NEW_GLOSSARY = "\x00new"


def populate_glossary_combo(combo: QComboBox, selected: str | None) -> None:
    """Fill a chooser with the library, keeping (or restoring) a selection.

    A selection pointing at a deleted glossary is kept as a visible "(missing)"
    entry rather than silently becoming "per recording" — a job quietly losing
    its shared vocabulary is worse than being told the glossary is gone.
    """
    combo.blockSignals(True)
    combo.clear()
    combo.addItem(PER_RECORDING_LABEL, "")
    ids = set()
    for glossary in glossary_store.list_glossaries():
        ids.add(glossary.id)
        combo.addItem(f"{glossary.name}  ({glossary.summary()})", glossary.id)
    if selected and selected not in ids:
        combo.addItem(f"{selected}  (missing)", selected)
    idx = combo.findData(selected or "")
    combo.setCurrentIndex(idx if idx >= 0 else 0)
    combo.blockSignals(False)


class _EntryTable(QWidget):
    """One editable table of glossary entries, with its row tools and filter."""

    changed = Signal()

    def __init__(self, columns: list[str], fields, *, defaults=None, parent=None):
        super().__init__(parent)
        self._fields = fields
        self._defaults = dict(defaults or {})
        self._conflict_col = len(columns) - 1

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget(0, len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        header = self.table.horizontalHeader()
        for i in range(len(columns)):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        self.table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.table, stretch=1)

        row = QHBoxLayout()
        add = QPushButton("Add row")
        add.setAutoDefault(False)
        add.clicked.connect(self.add_row)
        remove = QPushButton("Remove selected rows")
        remove.setAutoDefault(False)
        remove.clicked.connect(self.remove_selected)
        self.clear_tags_btn = QPushButton("Keep as is")
        self.clear_tags_btn.setToolTip(
            "Drop the review tag on the selected rows without changing them."
        )
        self.clear_tags_btn.setAutoDefault(False)
        self.clear_tags_btn.clicked.connect(self.clear_selected_tags)
        self.only_conflicts = QCheckBox("Show only rows needing review")
        self.only_conflicts.toggled.connect(self._apply_filter)
        self.count_label = QLabel("")
        self.count_label.setStyleSheet(muted_small())
        row.addWidget(add)
        row.addWidget(remove)
        row.addWidget(self.clear_tags_btn)
        row.addWidget(self.only_conflicts)
        row.addWidget(self.count_label)
        row.addStretch()
        layout.addLayout(row)

    # ---- contents -----------------------------------------------------
    def set_entries(self, entries: list[dict]) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for entry in entries:
            self._append(entry)
        self.table.blockSignals(False)
        self.table.resizeColumnsToContents()
        self._refresh_conflict_state()

    def entries(self) -> list[dict]:
        """The rows as glossary entries, keeping fields the table never shows."""
        collected = []
        for row in range(self.table.rowCount()):
            first = self._text(row, 0)
            if not first:
                continue
            stored = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole) or {}
            entry = dict(stored)
            for col, (key, kind) in enumerate(self._fields):
                value = self._text(row, col)
                if kind == "list":
                    entry[key] = [v.strip() for v in value.split(",") if v.strip()]
                else:
                    entry[key] = value
            collected.append(entry)
        return collected

    def conflict_count(self) -> int:
        return sum(
            1 for row in range(self.table.rowCount()) if self._conflict_at(row)
        )

    # ---- row actions --------------------------------------------------
    def add_row(self) -> None:
        self.table.blockSignals(True)
        self._append(dict(self._defaults))
        self.table.blockSignals(False)
        row = self.table.rowCount() - 1
        self.table.setRowHidden(row, False)   # a new row must not vanish into a filter
        self.table.editItem(self.table.item(row, 0))
        self._refresh_conflict_state()
        self.changed.emit()

    def remove_selected(self) -> None:
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)
        if rows:
            self._refresh_conflict_state()
            self.changed.emit()

    def clear_selected_tags(self) -> None:
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()})
        cleared = [row for row in rows if self._conflict_at(row)]
        for row in cleared:
            self._clear_tag(row)
        if cleared:
            self._refresh_conflict_state()
            self.changed.emit()

    # ---- internals ----------------------------------------------------
    def _append(self, entry: dict) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        for col, (key, kind) in enumerate(self._fields):
            value = entry.get(key)
            if kind == "list":
                text = ", ".join(str(v) for v in (value or []))
            else:
                text = "" if value is None else str(value)
            self.table.setItem(row, col, QTableWidgetItem(text))
        self.table.item(row, 0).setData(Qt.ItemDataRole.UserRole, dict(entry))
        note = QTableWidgetItem(glossary_merge.describe(entry))
        note.setFlags(note.flags() & ~Qt.ItemFlag.ItemIsEditable)
        note.setForeground(qcolor("warn"))
        self.table.setItem(row, self._conflict_col, note)

    def _text(self, row: int, col: int) -> str:
        item = self.table.item(row, col)
        return item.text().strip() if item else ""

    def _conflict_at(self, row: int) -> bool:
        item = self.table.item(row, 0)
        stored = item.data(Qt.ItemDataRole.UserRole) if item else None
        return bool((stored or {}).get(glossary_store.CONFLICT_KEY))

    def _clear_tag(self, row: int) -> None:
        item = self.table.item(row, 0)
        if item is None:
            return
        stored = dict(item.data(Qt.ItemDataRole.UserRole) or {})
        glossary_merge.clear(stored)
        item.setData(Qt.ItemDataRole.UserRole, stored)
        note = self.table.item(row, self._conflict_col)
        if note is not None:
            self.table.blockSignals(True)
            note.setText("")
            self.table.blockSignals(False)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        # Editing what the sources disagreed about — the first or second column
        # — IS the resolution, so the tag goes with it.
        if item.column() <= 1 and self._conflict_at(item.row()):
            self._clear_tag(item.row())
            self._refresh_conflict_state()
        self.changed.emit()

    def _refresh_conflict_state(self) -> None:
        conflicts = self.conflict_count()
        self.count_label.setText(
            f"{conflicts} row(s) need review" if conflicts else ""
        )
        self.only_conflicts.setEnabled(bool(conflicts) or self.only_conflicts.isChecked())
        if not conflicts and self.only_conflicts.isChecked():
            self.only_conflicts.setChecked(False)    # nothing left to filter down to
        self._apply_filter()

    def _apply_filter(self, *_args) -> None:
        only = self.only_conflicts.isChecked()
        for row in range(self.table.rowCount()):
            self.table.setRowHidden(row, only and not self._conflict_at(row))


class CombineGlossariesDialog(SheetDialog):
    """Pick which glossaries to fold together, and where the result lands."""

    def __init__(self, parent=None, *, current_id: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Combine glossaries")
        self.setMinimumWidth(480)
        self._library = glossary_store.list_glossaries()

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Combine these glossaries:"))
        self.sources = QListWidget()
        for glossary in self._library:
            item = QListWidgetItem(f"{glossary.name}  ({glossary.summary()})")
            item.setData(Qt.ItemDataRole.UserRole, glossary.id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if glossary.id == current_id else Qt.CheckState.Unchecked
            )
            self.sources.addItem(item)
        layout.addWidget(self.sources, stretch=1)

        dest_row = QHBoxLayout()
        dest_row.addWidget(QLabel("Into:"))
        self.destination = QComboBox()
        self.destination.addItem("A new glossary…", NEW_GLOSSARY)
        for glossary in self._library:
            self.destination.addItem(glossary.name, glossary.id)
        idx = self.destination.findData(current_id)
        self.destination.setCurrentIndex(idx if idx >= 0 else 0)
        self.destination.currentIndexChanged.connect(self._update_name_field)
        dest_row.addWidget(self.destination, stretch=1)
        layout.addLayout(dest_row)

        self.new_name = QLineEdit()
        self.new_name.setPlaceholderText("Name for the combined glossary")
        layout.addWidget(self.new_name)

        note = QLabel(
            "Terms and variants are merged and deduped. Where the sources "
            "disagree — the same term with a different type, the same speaker "
            "label with a different name — the entry is kept and tagged for "
            "review rather than one reading being picked for you. The source "
            "glossaries are left alone."
        )
        note.setWordWrap(True)
        note.setStyleSheet(muted_small())
        layout.addWidget(note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._update_name_field()

    def selection(self) -> tuple[list[str], str, str]:
        """(source ids, destination id or "" for new, name for the new one)"""
        return self._source_ids(), self._destination_id(), self.new_name.text().strip()

    def _source_ids(self) -> list[str]:
        return [
            self.sources.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.sources.count())
            if self.sources.item(i).checkState() == Qt.CheckState.Checked
        ]

    def _destination_id(self) -> str:
        data = self.destination.currentData()
        return "" if data == NEW_GLOSSARY else data

    def _update_name_field(self, *_args) -> None:
        creating = self.destination.currentData() == NEW_GLOSSARY
        self.new_name.setEnabled(creating)
        if not creating:
            self.new_name.clear()

    def _accept(self) -> None:
        sources = self._source_ids()
        if not sources:
            QMessageBox.warning(self, "Combine glossaries", "Tick at least one glossary.")
            return
        if not self._destination_id() and not self.new_name.text().strip():
            QMessageBox.warning(
                self, "Combine glossaries", "Name the new glossary, or pick an existing one."
            )
            return
        self.accept()


class GlossaryLibraryDialog(SheetDialog):
    """Create, edit, merge, import, and delete shared glossaries."""

    def __init__(self, parent=None, *, selected: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Shared glossaries")
        self.setMinimumSize(940, 600)
        self._current: SharedGlossary | None = None
        self._dirty = False

        outer = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ---- left: the library ----
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QLabel("Glossaries"))
        self.list = QListWidget()
        self.list.currentItemChanged.connect(self._on_selection_changed)
        self.list.itemDoubleClicked.connect(lambda _item: self._rename())
        ll.addWidget(self.list, stretch=1)

        for label, slot, tip in (
            ("New…", self._new, "Start an empty glossary."),
            ("Rename…", self._rename, "Rename the selected glossary (or double-click it)."),
            ("Duplicate", self._duplicate, "Copy the selected glossary's contents into a new one."),
            ("Combine…", self._combine, "Merge several glossaries into one, deduped."),
            ("Import…", self._import, "Read a glossary file into this one, or into a new one."),
            ("Export…", self._export, "Write the selected glossary out as JSON."),
            ("Delete", self._delete, "Delete the selected glossary permanently."),
        ):
            btn = QPushButton(label)
            btn.setAutoDefault(False)
            btn.setToolTip(tip)
            btn.clicked.connect(slot)
            ll.addWidget(btn)
        splitter.addWidget(left)

        # ---- right: the contents of the selected glossary ----
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)

        self.title = QLabel("No glossary selected")
        self.title.setStyleSheet("font-weight: bold;")
        rl.addWidget(self.title)

        self.tabs = QTabWidget()
        self.terms = _EntryTable(TERM_COLS, TERM_FIELDS)
        self.speakers = _EntryTable(SPEAKER_COLS, SPEAKER_FIELDS, defaults=SPEAKER_DEFAULTS)
        for table in (self.terms, self.speakers):
            table.changed.connect(self._mark_dirty)
        self.tabs.addTab(self.terms, "Terms")
        self.tabs.addTab(self.speakers, "Speakers")
        rl.addWidget(self.tabs, stretch=1)

        self.sources = QLabel("")
        self.sources.setWordWrap(True)
        self.sources.setStyleSheet(muted_small())
        rl.addWidget(self.sources)

        note = QLabel(
            "Terms are shared across every recording that uses this glossary. "
            "Speakers listed here are treated as a standing roster — leave it "
            "empty unless a label means the same person in every recording "
            "(diarization labels like SPEAKER_00 do not)."
        )
        note.setWordWrap(True)
        note.setStyleSheet(muted_small())
        rl.addWidget(note)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([250, 680])
        outer.addWidget(splitter, stretch=1)

        buttons = QDialogButtonBox()
        self.save_btn = buttons.addButton("Save", QDialogButtonBox.ButtonRole.ApplyRole)
        self.save_btn.clicked.connect(lambda: self._save_current())
        close_btn = buttons.addButton("Close", QDialogButtonBox.ButtonRole.AcceptRole)
        close_btn.clicked.connect(self._accept)
        outer.addWidget(buttons)

        self._reload_list(select=selected)

    # ------------------------------------------------------------------
    def selected_id(self) -> str:
        item = self.list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else ""

    def _mark_dirty(self) -> None:
        if self._current is not None:
            self._dirty = True
            self._update_title()

    def _update_title(self) -> None:
        if self._current is None:
            self.title.setText("No glossary selected")
            return
        bits = [self._current.name]
        conflicts = self.terms.conflict_count() + self.speakers.conflict_count()
        if conflicts:
            bits.append(f"⚠ {conflicts} row(s) need review")
        if self._dirty:
            bits.append("unsaved changes")
        self.title.setText("  •  ".join(bits))

    # ---- library ------------------------------------------------------
    def _reload_list(self, select: str = "") -> None:
        self.list.blockSignals(True)
        self.list.clear()
        for glossary in glossary_store.list_glossaries():
            item = QListWidgetItem(f"{glossary.name}\n{glossary.summary()}")
            item.setData(Qt.ItemDataRole.UserRole, glossary.id)
            self.list.addItem(item)
        self.list.blockSignals(False)
        target = 0
        for i in range(self.list.count()):
            if self.list.item(i).data(Qt.ItemDataRole.UserRole) == select:
                target = i
                break
        if self.list.count():
            self.list.setCurrentRow(-1)     # force the reselect to re-read the file
            self.list.setCurrentRow(target)
        else:
            self._show(None)

    def _on_selection_changed(self, current, previous) -> None:
        if previous is not None and self._dirty:
            self._save_current(silent=True)
        gid = current.data(Qt.ItemDataRole.UserRole) if current else ""
        self._show(glossary_store.load(gid) if gid else None)

    def _show(self, glossary: SharedGlossary | None) -> None:
        self._current = glossary
        self._dirty = False
        self.terms.set_entries(glossary.terms if glossary else [])
        self.speakers.set_entries(glossary.speakers if glossary else [])
        self.save_btn.setEnabled(glossary is not None)
        self.sources.setText(self._sources_text(glossary))
        self._update_title()

    @staticmethod
    def _sources_text(glossary: SharedGlossary | None) -> str:
        if glossary is None or not glossary.sources:
            return "No recording has contributed to this glossary yet."
        names = [s.get("name") or s.get("key", "") for s in glossary.sources]
        shown = ", ".join(names[:6])
        more = f" (+{len(names) - 6} more)" if len(names) > 6 else ""
        return f"Fed by {len(names)} recording(s): {shown}{more}"

    # ---- actions ------------------------------------------------------
    def _save_current(self, *, silent: bool = False) -> None:
        if self._current is None:
            return
        self._current.terms = sorted(
            self.terms.entries(), key=lambda t: str(t.get("canonical", "")).lower()
        )
        self._current.speakers = sorted(
            self.speakers.entries(), key=lambda s: str(s.get("label", "")).lower()
        )
        try:
            glossary_store.save(self._current)
        except Exception as e:
            QMessageBox.warning(self, "Shared glossaries", f"Could not save: {e}")
            return
        self._dirty = False
        self._update_title()
        self.sources.setText(self._sources_text(self._current))
        if not silent:
            self._reload_list(select=self._current.id)

    def _new(self) -> None:
        name, ok = QInputDialog.getText(self, "New glossary", "Name:")
        if not ok or not name.strip():
            return
        created = glossary_store.create(name)
        self._reload_list(select=created.id)

    def _rename(self) -> None:
        if self._current is None:
            return
        name, ok = QInputDialog.getText(
            self, "Rename glossary", "Name:", text=self._current.name
        )
        if not ok or not name.strip() or name.strip() == self._current.name:
            return
        if self._dirty:
            self._save_current(silent=True)
        self._current.name = name.strip()
        glossary_store.save(self._current)
        self._reload_list(select=self._current.id)

    def _duplicate(self) -> None:
        if self._current is None:
            return
        if self._dirty:
            self._save_current(silent=True)
        copy = glossary_store.duplicate(self._current.id)
        if copy:
            self._reload_list(select=copy.id)

    def _delete(self) -> None:
        if self._current is None:
            return
        answer = QMessageBox.question(
            self,
            "Delete glossary",
            f"Delete '{self._current.name}' permanently?\n\n"
            "Jobs pointed at it will fall back to a per-recording glossary.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        glossary_store.delete(self._current.id)
        self._current = None
        self._dirty = False
        self._reload_list()

    def _combine(self) -> None:
        if self._dirty:
            self._save_current(silent=True)
        dlg = CombineGlossariesDialog(self, current_id=self.selected_id())
        if not dlg.exec():
            return
        source_ids, dest_id, new_name = dlg.selection()
        sources = [g for g in (glossary_store.load(gid) for gid in source_ids) if g]
        if not sources:
            return

        parts = [Part.of(g) for g in sources]
        destination = glossary_store.load(dest_id) if dest_id else None
        if destination is not None and destination.id not in source_ids:
            # The destination's own contents are one of the things being merged.
            parts.insert(0, Part.of(destination))
        if destination is None:
            destination = glossary_store.create(new_name)
        for glossary in sources:
            for entry in glossary.sources:
                destination.record_source(entry.get("key", ""), entry.get("name", ""))

        result = self._merge_into(destination, parts)
        names = ", ".join(g.name for g in sources)
        self._announce(
            "Combine glossaries",
            f"Combined {names} into '{destination.name}' — {result.summary()}.",
            result,
        )

    def _import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import glossary", "", "Glossary files (*.json);;All files (*)"
        )
        if not path:
            return
        try:
            payload = glossary_store.read_payload(path)
        except Exception as e:
            QMessageBox.warning(self, "Import glossary", f"Could not read that file: {e}")
            return

        # A shared export names itself; a per-recording file falls back to its
        # filename. Either way the label is what conflicts get attributed to.
        label = str(payload.get("name") or Path(path).stem)
        into_current = self._ask_import_destination(label)
        if into_current is None:
            return
        if not into_current:
            imported = glossary_store.create(
                label, speakers=payload["speakers"], terms=payload["terms"]
            )
            self._reload_list(select=imported.id)
            return

        if self._dirty:
            self._save_current(silent=True)
        destination = self._current
        if destination is None:
            return
        result = self._merge_into(
            destination, [Part.of(destination), Part.of_payload(label, payload)]
        )
        self._announce(
            "Import glossary",
            f"Merged {label} into '{destination.name}' — {result.summary()}.",
            result,
        )

    def _ask_import_destination(self, label: str) -> bool | None:
        """True = into the open glossary, False = as a new one, None = cancel."""
        if self._current is None:
            return False
        box = QMessageBox(self)
        box.setWindowTitle("Import glossary")
        box.setText(f"Where should {label} go?")
        box.setInformativeText(
            "Merging into the open glossary dedupes the two and tags anything "
            "they disagree about."
        )
        into = box.addButton(
            f"Merge into '{self._current.name}'", QMessageBox.ButtonRole.AcceptRole
        )
        fresh = box.addButton("Add as a new glossary", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is into:
            return True
        if clicked is fresh:
            return False
        return None

    def _merge_into(
        self, destination: SharedGlossary, parts: list[Part]
    ) -> glossary_merge.MergeResult:
        """Fold parts into a glossary, save it, and open it on what needs review."""
        result = glossary_merge.apply_to(destination, parts)
        glossary_store.save(destination)
        self._current = destination
        self._dirty = False
        self._reload_list(select=destination.id)
        self._focus_conflicts()
        return result

    def _focus_conflicts(self) -> None:
        """Land the user on exactly the rows that still need a decision."""
        for table in (self.terms, self.speakers):
            table.only_conflicts.setChecked(bool(table.conflict_count()))
        if self.terms.conflict_count():
            self.tabs.setCurrentWidget(self.terms)
        elif self.speakers.conflict_count():
            self.tabs.setCurrentWidget(self.speakers)

    def _announce(self, title: str, message: str, result) -> None:
        if result.total_conflicts:
            message += (
                f"\n\n{result.total_conflicts} entry(ies) need review: the sources "
                "gave them different values. The tables are filtered down to those "
                "rows — edit them, delete the ones you do not want, or press "
                "\"Keep as is\"."
            )
        QMessageBox.information(self, title, message)

    def _export(self) -> None:
        if self._current is None:
            return
        if self._dirty:
            self._save_current(silent=True)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export glossary",
            f"{self._current.id}.json",
            "Glossary files (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            glossary_store.export_to(self._current.id, path)
        except Exception as e:
            QMessageBox.warning(self, "Export glossary", f"Could not export: {e}")

    def _accept(self) -> None:
        if self._dirty:
            self._save_current(silent=True)
        self.accept()
