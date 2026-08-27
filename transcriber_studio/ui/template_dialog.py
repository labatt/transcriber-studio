# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Filename template builder with clickable tokens and a live preview."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialogButtonBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from .. import filename_builder
from .theme import SheetDialog


class TemplateDialog(SheetDialog):
    def __init__(self, template: str, sanitize: bool, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filename Template Builder")
        self.setMinimumWidth(620)
        self._sanitize = sanitize
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(
            "Build the output filename. Click a token to insert it at the cursor. "
            "The file extension is added automatically per format."
        ))

        self.edit = QLineEdit(template)
        self.edit.textChanged.connect(self._update_preview)
        layout.addWidget(self.edit)

        # Token palette
        box = QGroupBox("Available tokens")
        grid = QGridLayout(box)
        for i, tok in enumerate(filename_builder.TOKENS):
            btn = QPushButton(tok.key)
            btn.setToolTip(f"{tok.description}\nExample: {tok.example}")
            btn.clicked.connect(lambda _=False, t=tok.key: self._insert(t))
            r, c = divmod(i, 3)
            grid.addWidget(btn, r, c)
            lbl = QLabel(f"  {tok.description}")
            lbl.setStyleSheet("color: gray;")
        layout.addWidget(box)

        # Quick presets
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Presets:"))
        for name, tmpl in [
            ("Date + Name", "{date}_{name}"),
            ("Source + Name + Model", "{source}_{name}_{model}"),
            ("Datetime + Name", "{datetime}_{name}"),
            ("Index + Name + Lang", "{index}_{name}_{lang}"),
        ]:
            b = QPushButton(name)
            b.clicked.connect(lambda _=False, t=tmpl: self.edit.setText(t))
            preset_row.addWidget(b)
        preset_row.addStretch()
        layout.addLayout(preset_row)

        self.preview = QLabel()
        self.preview.setStyleSheet(
            "padding:8px; background:#f0f4ff; border:1px solid #c3d0ef; border-radius:4px;"
        )
        self.preview.setWordWrap(True)
        layout.addWidget(QLabel("Preview (sample recording):"))
        layout.addWidget(self.preview)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._update_preview()

    def _insert(self, token: str):
        self.edit.insert(token)
        self.edit.setFocus()

    def _update_preview(self):
        stem = filename_builder.render(
            self.edit.text(), filename_builder.sample_values(), self._sanitize
        )
        self.preview.setText(f"{stem}.txt   /   {stem}.srt   …")

    def template(self) -> str:
        return self.edit.text().strip() or "{date}_{name}"
