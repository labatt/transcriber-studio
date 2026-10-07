# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hearing each speaker and naming them before the transcription runs.

What the dialog hands back is what the run uses: names keyed by pyannote's raw
label, and the voices to remember. The audio itself is not tested — there is
no sound card in CI — only that every speaker gets several samples to press.
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

from transcriber_studio import voiceprints
from transcriber_studio.config import Settings
from transcriber_studio.diarization import DiarizationResult, SpeakerTurn
from transcriber_studio.ui.identify_dialog import SpeakerIdentifyDialog

from .support import isolated_voiceprints

DIM = 8


def _vector(*values: float) -> list[float]:
    padded = list(values) + [0.0] * (DIM - len(values))
    return [float(x) for x in padded[:DIM]]


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _diarized() -> DiarizationResult:
    return DiarizationResult(
        turns=[
            SpeakerTurn(0.0, 60.0, "SPEAKER_00"),
            SpeakerTurn(60.0, 100.0, "SPEAKER_01"),
            SpeakerTurn(300.0, 360.0, "SPEAKER_00"),
            SpeakerTurn(400.0, 408.0, "SPEAKER_02"),     # too brief to remember
        ],
        embeddings={
            "SPEAKER_00": _vector(1, 0), "SPEAKER_01": _vector(0, 1), "SPEAKER_02": _vector(0, 0, 1),
        },
    )


def _dialog(settings=None) -> SpeakerIdentifyDialog:
    return SpeakerIdentifyDialog("audio.mp3", _diarized(), settings or Settings(), recording_name="Sync")


def test_every_speaker_has_at_least_two_samples_where_the_speech_allows(app):
    with isolated_voiceprints():
        dlg = _dialog()
        by_label = {row.label: row for row in dlg.rows}
        assert len(by_label["SPEAKER_00"].buttons) >= 2
        assert len(by_label["SPEAKER_01"].buttons) >= 2
        assert [row.shown_as for row in dlg.rows] == ["Speaker 1", "Speaker 2", "Speaker 3"]


def test_names_are_keyed_by_the_raw_label(app):
    with isolated_voiceprints():
        dlg = _dialog()
        dlg.rows[0].name.setText("Alice")
        dlg.rows[1].name.setText("  ")              # left blank
        dlg.rows[2].name.setText("Speaker 3")       # just the number it already had
        dlg.accept()
        assert dlg.names() == {"SPEAKER_00": "Alice"}


def test_remember_is_on_by_default_but_only_where_there_is_enough_speech(app):
    with isolated_voiceprints():
        dlg = _dialog(Settings(voiceprint_min_enroll_s=30.0))
        assert dlg.rows[0].remember.isChecked() and dlg.rows[0].remember.isEnabled()
        assert not dlg.rows[2].remember.isEnabled(), "8 seconds is not a voiceprint"


def test_enrollments_store_the_named_voices(app):
    with isolated_voiceprints():
        dlg = _dialog()
        dlg.rows[0].name.setText("Alice")
        dlg.rows[1].name.setText("Bob")
        dlg.rows[1].remember.setChecked(False)
        dlg.accept()
        assert dlg.enrollments() == [("Alice", "SPEAKER_00")]
        remembered = dlg.apply_enrollments(source="Sync")
        assert remembered == ["Alice"]
        profile = voiceprints.get_profile("Alice")
        assert profile is not None and profile.total_seconds == 120.0


def test_skipping_hands_back_nothing(app):
    with isolated_voiceprints():
        dlg = _dialog()
        dlg.rows[0].name.setText("Alice")
        dlg._skip()
        assert dlg.result() == SpeakerIdentifyDialog.SKIPPED
        assert dlg.names() == {} and dlg.enrollments() == []


def test_a_recognised_voice_is_prefilled(app):
    with isolated_voiceprints():
        voiceprints.enroll("Alice", _vector(1, 0), seconds=120.0)
        dlg = _dialog(Settings(voiceprint_threshold=0.5, voiceprint_margin=0.1))
        assert dlg.rows[0].name.text() == "Alice"
        assert "Recognised" in dlg.rows[0].hint.text()
        assert dlg.rows[1].name.text() == ""
