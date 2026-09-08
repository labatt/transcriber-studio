# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Manage the voices the app has been taught, and how willing it is to use them.

Voices are enrolled from the rename dialog, at the moment someone puts a name
to a speaker. This is where they are looked at afterwards: who is known, which
recordings each sample came from, and which of them to throw away.

The second tab is the important one. The matching thresholds started as
reasoned guesses — derived from how pyannote clusters speakers *within* one
recording, which is a different problem from recognising someone across months
and microphones — so they are settings rather than constants, and the scores
from real recordings are shown next to them. Tuning a threshold against
somebody else's reasoning is guesswork; tuning it against your own recordings
is not.
"""

from __future__ import annotations

import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .. import config, voiceprints
from ..config import Settings
from .theme import SheetDialog, qcolor

SAMPLE_COLS = ["From", "Speech", "Added"]
LOG_COLS = ["When", "Recording", "Speaker", "Result", "Score", "Runner-up", "Speech"]


def _ago(when: float) -> str:
    """Rough age. Precision here would be noise — the useful question is
    'recently or ages ago', not the minute it happened."""
    if not when:
        return "—"
    seconds = max(0.0, time.time() - when)
    if seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)} h ago"
    days = int(seconds // 86400)
    return "yesterday" if days == 1 else f"{days} days ago"


def _mmss(seconds: float) -> str:
    total = int(max(0.0, seconds))
    return f"{total // 60}m {total % 60:02d}s" if total >= 60 else f"{total}s"


class SpeakersDialog(SheetDialog):
    """Enrolled voices, the scores they have produced, and the bar they clear."""

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Speakers")
        self.setMinimumSize(720, 520)
        self.s = settings

        outer = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._voices_tab(), "Enrolled voices")
        self.tabs.addTab(self._tuning_tab(), "Recognition")
        outer.addWidget(self.tabs)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Close
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("Save settings")
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        self._reload()

    # ---- enrolled voices ----------------------------------------------
    def _voices_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        blurb = QLabel(
            "Voices are added from the Rename speakers dialog — name a speaker "
            "and tick “Remember this voice”. Each person can hold several "
            "samples; a speaker is matched against whichever of them their "
            "recording most resembles, so one capture per microphone helps more "
            "than one long capture."
        )
        blurb.setWordWrap(True)
        blurb.setStyleSheet("color: gray;")
        layout.addWidget(blurb)

        split = QHBoxLayout()
        left = QVBoxLayout()
        self.people = QListWidget()
        self.people.currentItemChanged.connect(lambda *_: self._show_samples())
        left.addWidget(self.people)
        person_row = QHBoxLayout()
        self.rename_btn = QPushButton("Rename…")
        self.delete_btn = QPushButton("Forget person")
        self.rename_btn.clicked.connect(self._rename_person)
        self.delete_btn.clicked.connect(self._delete_person)
        person_row.addWidget(self.rename_btn)
        person_row.addWidget(self.delete_btn)
        left.addLayout(person_row)
        split.addLayout(left, stretch=1)

        right = QVBoxLayout()
        self.samples = QTableWidget(0, len(SAMPLE_COLS))
        self.samples.setHorizontalHeaderLabels(SAMPLE_COLS)
        self.samples.verticalHeader().setVisible(False)
        self.samples.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.samples.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.samples.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        right.addWidget(self.samples)
        self.forget_sample_btn = QPushButton("Forget this sample")
        self.forget_sample_btn.setToolTip(
            "For the one bad capture — a meeting on speakerphone, say — that is "
            "dragging every later match toward it. The person is kept."
        )
        self.forget_sample_btn.clicked.connect(self._forget_sample)
        right.addWidget(self.forget_sample_btn)
        split.addLayout(right, stretch=2)
        layout.addLayout(split)

        self.voices_status = QLabel("")
        self.voices_status.setStyleSheet("color: gray;")
        layout.addWidget(self.voices_status)
        return page

    # ---- recognition ---------------------------------------------------
    def _tuning_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.0, 1.0)
        self.threshold.setSingleStep(0.05)
        self.threshold.setDecimals(2)
        self.threshold.setValue(self.s.voiceprint_threshold)
        self.threshold.setToolTip(
            "How alike two voices must be before a name is used. Lower "
            "recognises more people and starts getting some of them wrong."
        )
        self.margin = QDoubleSpinBox()
        self.margin.setRange(0.0, 0.5)
        self.margin.setSingleStep(0.05)
        self.margin.setDecimals(2)
        self.margin.setValue(self.s.voiceprint_margin)
        self.margin.setToolTip(
            "How far ahead of the second-best candidate the winner must be. "
            "Stops a name going on a voice that resembles two enrolled people "
            "about equally."
        )
        self.min_speech = QDoubleSpinBox()
        self.min_speech.setRange(0.0, 600.0)
        self.min_speech.setSingleStep(5.0)
        self.min_speech.setDecimals(0)
        self.min_speech.setSuffix(" s")
        self.min_speech.setValue(self.s.voiceprint_min_speech_s)
        self.min_speech.setToolTip(
            "A speaker who says less than this is not recognised at all: too "
            "little speech to describe a voice with."
        )
        self.min_enroll = QDoubleSpinBox()
        self.min_enroll.setRange(0.0, 600.0)
        self.min_enroll.setSingleStep(5.0)
        self.min_enroll.setDecimals(0)
        self.min_enroll.setSuffix(" s")
        self.min_enroll.setValue(self.s.voiceprint_min_enroll_s)
        self.min_enroll.setToolTip(
            "How much of someone is needed before they can be enrolled. "
            "Enrolling is a promise about every later recording, so this asks "
            "for more than recognising does."
        )

        for label, widget in (
            ("Match threshold:", self.threshold),
            ("Winning margin:", self.margin),
            ("Recognise from:", self.min_speech),
            ("Enrol from:", self.min_enroll),
        ):
            row = QHBoxLayout()
            caption = QLabel(label)
            caption.setFixedWidth(130)
            row.addWidget(caption)
            row.addWidget(widget)
            row.addStretch()
            layout.addLayout(row)

        self.advice = QLabel("")
        self.advice.setWordWrap(True)
        layout.addWidget(self.advice)

        layout.addWidget(QLabel("Every recognition attempt, newest first:"))
        self.log = QTableWidget(0, len(LOG_COLS))
        self.log.setHorizontalHeaderLabels(LOG_COLS)
        self.log.verticalHeader().setVisible(False)
        self.log.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.log.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.log.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        layout.addWidget(self.log)

        row = QHBoxLayout()
        clear = QPushButton("Clear history")
        clear.setToolTip("Start the record again — after re-enrolling, say.")
        clear.clicked.connect(self._clear_log)
        row.addWidget(clear)
        row.addStretch()
        self.log_status = QLabel("")
        self.log_status.setStyleSheet("color: gray;")
        row.addWidget(self.log_status)
        layout.addLayout(row)
        return page

    # ---- loading -------------------------------------------------------
    def _reload(self):
        wanted = self._selected_name()
        self.people.clear()
        profiles = voiceprints.load_profiles()
        for profile in profiles:
            item = QListWidgetItem(
                f"{profile.name}  ({len(profile.prints)} sample"
                f"{'' if len(profile.prints) == 1 else 's'})"
            )
            item.setData(Qt.ItemDataRole.UserRole, profile.name)
            self.people.addItem(item)
            if profile.name == wanted:
                self.people.setCurrentItem(item)
        if self.people.count() and self.people.currentRow() < 0:
            self.people.setCurrentRow(0)

        has_any = bool(profiles)
        for btn in (self.rename_btn, self.delete_btn, self.forget_sample_btn):
            btn.setEnabled(has_any)
        self.voices_status.setText(
            f"{len(profiles)} voice(s) enrolled."
            if has_any
            else "No voices enrolled yet — every speaker will be Speaker 1, 2, 3."
        )
        self._show_samples()
        self._load_log()

    def _selected_name(self) -> str:
        item = self.people.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else ""

    def _show_samples(self):
        self.samples.setRowCount(0)
        profile = voiceprints.get_profile(self._selected_name())
        if profile is None:
            self.forget_sample_btn.setEnabled(False)
            return
        for row, print_ in enumerate(profile.prints):
            self.samples.insertRow(row)
            self.samples.setItem(
                row, 0, QTableWidgetItem(print_.source or "unknown recording")
            )
            self.samples.setItem(row, 1, QTableWidgetItem(_mmss(print_.seconds)))
            self.samples.setItem(row, 2, QTableWidgetItem(_ago(print_.created)))
        if self.samples.rowCount():
            self.samples.selectRow(0)
        self.forget_sample_btn.setEnabled(bool(profile.prints))

    def _load_log(self):
        entries = voiceprints.load_match_log()
        self.log.setRowCount(0)
        for row, entry in enumerate(entries):
            self.log.insertRow(row)
            result = entry.name if entry.named else (entry.reason or "not recognised")
            cells = [
                _ago(entry.when),
                entry.source or "—",
                entry.label,
                result,
                f"{entry.score:.2f}" if entry.score else "—",
                f"{entry.runner_up:.2f}" if entry.runner_up else "—",
                _mmss(entry.seconds),
            ]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column == 3:
                    item.setForeground(qcolor("good" if entry.named else "muted"))
                self.log.setItem(row, column, item)
        self.log_status.setText(
            f"{len(entries)} attempt(s) recorded." if entries else "Nothing recorded yet."
        )
        self.advice.setText(voiceprints.suggest_threshold(entries))

    # ---- actions -------------------------------------------------------
    def _rename_person(self):
        current = self._selected_name()
        if not current:
            return
        new, ok = QInputDialog.getText(self, "Rename speaker", "Name:", text=current)
        if not ok or not new.strip() or new.strip() == current:
            return
        try:
            voiceprints.rename_profile(current, new.strip())
        except voiceprints.VoiceprintError as e:
            QMessageBox.warning(self, "Rename speaker", str(e))
            return
        self._reload()

    def _delete_person(self):
        current = self._selected_name()
        if not current:
            return
        profile = voiceprints.get_profile(current)
        samples = len(profile.prints) if profile else 0
        if QMessageBox.question(
            self,
            "Forget speaker",
            f"Forget “{current}” and all {samples} sample(s)?\n\n"
            "Transcripts already written keep the name. Later recordings will "
            "call them Speaker 1, 2, 3 again until they are enrolled afresh.",
        ) != QMessageBox.StandardButton.Yes:
            return
        voiceprints.delete_profile(current)
        self._reload()

    def _forget_sample(self):
        current = self._selected_name()
        row = self.samples.currentRow()
        if not current or row < 0:
            return
        try:
            voiceprints.forget_sample(current, row)
        except voiceprints.VoiceprintError as e:
            QMessageBox.warning(self, "Forget sample", str(e))
        self._reload()

    def _clear_log(self):
        if QMessageBox.question(
            self,
            "Clear history",
            "Throw away the recorded match scores?\n\n"
            "The enrolled voices are kept. Only the record of how they have "
            "been scoring goes, which is what the advice above reads.",
        ) != QMessageBox.StandardButton.Yes:
            return
        voiceprints.clear_match_log()
        self._load_log()

    def _save(self):
        self.s.voiceprint_threshold = self.threshold.value()
        self.s.voiceprint_margin = self.margin.value()
        self.s.voiceprint_min_speech_s = self.min_speech.value()
        self.s.voiceprint_min_enroll_s = self.min_enroll.value()
        config.save(self.s)
        self.accept()
