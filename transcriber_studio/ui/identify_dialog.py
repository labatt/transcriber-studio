# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hear each detected speaker, say who it is, and have the app remember them.

Shown after speaker detection and before transcription, when a run asked for
it. Every speaker gets several short samples from different parts of the
recording and a name field. A name typed here goes on the transcript — the
user heard the voice, which beats any score — and, with the box ticked, into
the voiceprint store so the next recording can name that person on its own.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtWidgets import (
    QCheckBox,
    QCompleter,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .. import voice_samples, voiceprints
from ..diarization import DiarizationResult
from ..transcriber import Transcriber
from .theme import SheetDialog, muted_small


class ClipPlayer:
    """Plays one window of a file at a time, through Qt's own media stack.

    Seeking into the original file beats cutting clips with ffmpeg: nothing is
    written, nothing has to be cleaned up, and the first click plays as soon
    as the decoder has found its place. Qt Multimedia is part of PySide6, so
    there is nothing extra to install — but a machine with no audio backend
    gets a disabled player rather than a crash.
    """

    def __init__(self):
        self.available = True
        self._end = 0.0
        try:
            from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

            self._player = QMediaPlayer()
            self._output = QAudioOutput()
            self._player.setAudioOutput(self._output)
            self._player.positionChanged.connect(self._on_position)
        except Exception:
            self.available = False
            self._player = None
        self._on_state = None

    def play(self, path: str, start: float, end: float) -> None:
        if not self._player:
            return
        self._end = end
        self._player.stop()
        if self._player.source().toLocalFile() != path:
            self._player.setSource(QUrl.fromLocalFile(path))
        self._player.setPosition(int(start * 1000))
        self._player.play()
        # Some backends ignore a seek issued before the media is loaded; one
        # more after play() has started is cheap and lands.
        QTimer.singleShot(80, lambda: self._player and self._player.setPosition(int(start * 1000)))

    def stop(self) -> None:
        if self._player:
            self._player.stop()

    def _on_position(self, ms: int) -> None:
        if self._end and ms >= int(self._end * 1000):
            self.stop()

    def shutdown(self) -> None:
        if self._player:
            self.stop()
            self._player.setSource(QUrl())


class SpeakerRow:
    """The widgets for one detected speaker."""

    def __init__(self, label: str, shown_as: str, seconds: float):
        self.label = label          # pyannote's raw label
        self.shown_as = shown_as    # "Speaker 2", as the transcript would say
        self.seconds = seconds
        self.name = QLineEdit()
        self.remember = QCheckBox("Remember this voice")
        self.buttons: list[QPushButton] = []
        self.hint = QLabel("")


class SpeakerIdentifyDialog(SheetDialog):
    """Who is each voice? Listen, type a name, tick to remember."""

    #: Result codes beyond accept/reject: Skip keeps the run going with no
    #: names, Cancel (reject) stops the run for this recording.
    SKIPPED = 2

    def __init__(
        self, audio_path: str, diarized: DiarizationResult, settings, parent=None,
        *, recording_name: str = "",
    ):
        super().__init__(parent)
        self.audio_path = audio_path
        self.diarized = diarized
        self.settings = settings
        self.player = ClipPlayer()
        self.rows: list[SpeakerRow] = []
        self.setWindowTitle(
            f"Who is speaking? — {recording_name}" if recording_name else "Who is speaking?"
        )
        self.setMinimumWidth(640)

        turns = diarized.turns
        mapping = Transcriber._stable_speaker_map(turns)
        seconds = diarized.speech_seconds()
        samples = voice_samples.clips_for(turns)
        recognised = self._recognised()
        enrolled = sorted(p.name for p in voiceprints.load_profiles())
        self.min_enroll = float(
            getattr(settings, "voiceprint_min_enroll_s", voiceprints.MIN_ENROLL_SECONDS)
        )

        layout = QVBoxLayout(self)
        count = len(samples)
        intro = QLabel(
            f"{count} voice{'s were' if count != 1 else ' was'} found. Play the samples, "
            "type each person's name, and the transcript will use those names. Leave a "
            "name blank to keep that speaker as a number."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        if not self.player.available:
            warn = QLabel(
                "Audio playback is not available on this machine (Qt Multimedia could "
                "not start), so the sample buttons are disabled. Names can still be typed."
            )
            warn.setWordWrap(True)
            warn.setStyleSheet(muted_small())
            layout.addWidget(warn)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        grid = QGridLayout(inner)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(2, 1)

        for index, (label, clips) in enumerate(samples.items()):
            row = SpeakerRow(label, mapping.get(label, label), float(seconds.get(label, 0.0)))
            self.rows.append(row)
            who = QLabel(f"<b>{row.shown_as}</b><br>{self._spoke(row.seconds)}")
            who.setTextFormat(Qt.TextFormat.RichText)
            grid.addWidget(who, index, 0, alignment=Qt.AlignmentFlag.AlignTop)

            buttons = QWidget()
            buttons_layout = QHBoxLayout(buttons)
            buttons_layout.setContentsMargins(0, 0, 0, 0)
            buttons_layout.setSpacing(4)
            for n, clip in enumerate(clips, start=1):
                btn = QPushButton(f"▶ {n}  {voice_samples.clock(clip.start)}")
                btn.setToolTip(
                    f"Play {clip.seconds:.0f}s of this speaker from "
                    f"{voice_samples.clock(clip.start)}."
                )
                btn.setAutoDefault(False)
                btn.setEnabled(self.player.available)
                btn.clicked.connect(
                    lambda _checked=False, c=clip: self.player.play(self.audio_path, c.start, c.end)
                )
                row.buttons.append(btn)
                buttons_layout.addWidget(btn)
            if not clips:
                buttons_layout.addWidget(QLabel("no sample long enough"))
            buttons_layout.addStretch()
            grid.addWidget(buttons, index, 1, alignment=Qt.AlignmentFlag.AlignTop)

            name_box = QWidget()
            name_layout = QVBoxLayout(name_box)
            name_layout.setContentsMargins(0, 0, 0, 0)
            name_layout.setSpacing(2)
            row.name.setPlaceholderText("Name…")
            if enrolled:
                completer = QCompleter(enrolled)
                completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
                row.name.setCompleter(completer)
            match = recognised.get(label)
            if match is not None and match.named:
                row.name.setText(match.name)
                row.hint.setText(f"Recognised as {match.name} ({match.score:.2f}) — correct it if wrong.")
            row.hint.setStyleSheet(muted_small())
            row.hint.setWordWrap(True)
            name_layout.addWidget(row.name)
            self._configure_remember(row)
            name_layout.addWidget(row.remember)
            name_layout.addWidget(row.hint)
            grid.addWidget(name_box, index, 2)

        scroll.setWidget(inner)
        layout.addWidget(scroll, stretch=1)

        controls = QHBoxLayout()
        self.stop_btn = QPushButton("■ Stop")
        self.stop_btn.setAutoDefault(False)
        self.stop_btn.setEnabled(self.player.available)
        self.stop_btn.clicked.connect(self.player.stop)
        controls.addWidget(self.stop_btn)
        controls.addStretch()
        layout.addLayout(controls)

        note = QLabel(
            "A remembered voice lets later recordings name this person automatically. "
            "Only a clear match is used; anything doubtful stays a numbered speaker."
        )
        note.setWordWrap(True)
        note.setStyleSheet(muted_small())
        layout.addWidget(note)

        buttons = QDialogButtonBox()
        self.continue_btn = buttons.addButton(
            "Continue", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self.skip_btn = buttons.addButton(
            "Skip naming", QDialogButtonBox.ButtonRole.ActionRole
        )
        self.skip_btn.setToolTip("Transcribe with numbered speakers; remember nobody.")
        self.cancel_btn = buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        self.cancel_btn.setText("Cancel run")
        self.continue_btn.setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.skip_btn.clicked.connect(self._skip)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    def _recognised(self) -> dict[str, voiceprints.Match]:
        """What voiceprint matching would already say, to prefill the names."""
        if not self.diarized.embeddings:
            return {}
        try:
            matches = voiceprints.identify(
                dict(self.diarized.embeddings),
                self.diarized.speech_seconds(),
                threshold=getattr(self.settings, "voiceprint_threshold", None),
                margin=getattr(self.settings, "voiceprint_margin", None),
                min_speech=getattr(self.settings, "voiceprint_min_speech_s", None),
            )
        except Exception:
            return {}
        return {m.label: m for m in matches}

    @staticmethod
    def _spoke(seconds: float) -> str:
        total = int(seconds)
        if total >= 60:
            return f"spoke for {total // 60}m {total % 60:02d}s"
        return f"spoke for {total}s"

    def _configure_remember(self, row: SpeakerRow) -> None:
        vector = self.diarized.embeddings.get(row.label)
        if not vector or not voiceprints.is_usable(vector):
            row.remember.setEnabled(False)
            row.remember.setChecked(False)
            row.remember.setToolTip("No usable voice data came back for this speaker.")
        elif row.seconds < self.min_enroll:
            row.remember.setEnabled(False)
            row.remember.setChecked(False)
            row.remember.setToolTip(
                f"Only {row.seconds:.0f}s of speech — {self.min_enroll:.0f}s needed to "
                "remember a voice."
            )
        else:
            row.remember.setChecked(True)
            row.remember.setToolTip(
                f"{row.seconds:.0f}s of this speaker will be stored under the name you type."
            )

    # ------------------------------------------------------------------
    def _skip(self) -> None:
        self.player.stop()
        self.done(self.SKIPPED)

    def accept(self) -> None:
        self.player.stop()
        super().accept()

    def reject(self) -> None:
        self.player.stop()
        super().reject()

    def closeEvent(self, event) -> None:
        self.player.shutdown()
        super().closeEvent(event)

    def names(self) -> dict[str, str]:
        """Raw pyannote label -> the name typed for it. Blank rows are left out,
        and so is a name that is just the number the speaker already had."""
        out: dict[str, str] = {}
        if self.result() == self.SKIPPED:
            return out
        for row in self.rows:
            name = row.name.text().strip()
            if name and name != row.shown_as:
                out[row.label] = name
        return out

    def enrollments(self) -> list[tuple[str, str]]:
        """(name, raw label) for every ticked, named, enrollable speaker."""
        if self.result() == self.SKIPPED:
            return []
        out: list[tuple[str, str]] = []
        for row in self.rows:
            if not (row.remember.isEnabled() and row.remember.isChecked()):
                continue
            name = row.name.text().strip()
            if not name or name == row.shown_as:
                continue
            out.append((name, row.label))
        return out

    def apply_enrollments(self, log=None, *, source: str = "") -> list[str]:
        """Store the ticked voices. Never raises: a voice that cannot be
        remembered must not stop the transcription the user came for."""
        remembered: list[str] = []
        seconds = self.diarized.speech_seconds()
        for name, label in self.enrollments():
            vector = self.diarized.embeddings.get(label)
            if not vector:
                continue
            try:
                voiceprints.enroll(
                    name, vector,
                    seconds=float(seconds.get(label, 0.0)),
                    source=source,
                    minimum_seconds=self.min_enroll,
                )
            except Exception as e:
                if log:
                    log(f"Could not remember {name}: {e}")
                continue
            remembered.append(name)
            if log:
                log(f"Remembered {name}'s voice — later recordings can name them.")
        return remembered
