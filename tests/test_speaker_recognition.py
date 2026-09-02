# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""An enrolled voice reaches the transcript as a name.

Two things have to line up for that: diarization has to keep the vector it
clustered each speaker around, and the label the reader sees has to come from
the recognised name rather than from counting speakers.
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

from transcriber_studio import voiceprints
from transcriber_studio.diarization import DiarizationResult, SpeakerTurn
from transcriber_studio.models import Recording, Segment, Source, TranscriptResult
from transcriber_studio.transcriber import Transcriber
from transcriber_studio.ui.rename_dialog import SpeakerRenameDialog

from .support import isolated_voiceprints

DIM = 8


def _vector(*values: float) -> list[float]:
    padded = list(values) + [0.0] * (DIM - len(values))
    return [float(x) for x in padded[:DIM]]


def _diarized() -> DiarizationResult:
    """Two speakers, a couple of minutes each."""
    return DiarizationResult(
        turns=[
            SpeakerTurn(start=0.0, end=90.0, speaker="SPEAKER_00"),
            SpeakerTurn(start=90.0, end=180.0, speaker="SPEAKER_01"),
        ],
        embeddings={"SPEAKER_00": _vector(1, 0), "SPEAKER_01": _vector(0, 1)},
    )


def _segments() -> list[Segment]:
    return [
        Segment(start=10.0, end=20.0, text="Hello there."),
        Segment(start=100.0, end=110.0, text="Hello back."),
    ]


def _transcriber() -> Transcriber:
    return Transcriber.__new__(Transcriber)


# ---- naming ----------------------------------------------------------
def test_an_enrolled_voice_becomes_a_name_in_the_transcript():
    with isolated_voiceprints():
        voiceprints.enroll("Alice", _vector(1, 0), seconds=120.0)
        segments, speakers = _transcriber()._apply_speakers(
            _segments(), [], _diarized(), lambda _m: None
        )
        assert speakers[0] == "Alice"
        assert segments[0].speaker == "Alice"


def test_the_unrecognised_speaker_keeps_a_number():
    with isolated_voiceprints():
        voiceprints.enroll("Alice", _vector(1, 0), seconds=120.0)
        _segments_, speakers = _transcriber()._apply_speakers(
            _segments(), [], _diarized(), lambda _m: None
        )
        assert speakers == ["Alice", "Speaker 1"]


def test_numbering_does_not_leave_a_gap_where_a_name_went():
    """Naming the first speaker must not make the second one 'Speaker 2',
    which would leave the reader looking for a Speaker 1 who never existed."""
    with isolated_voiceprints():
        voiceprints.enroll("Alice", _vector(1, 0), seconds=120.0)
        mapping = Transcriber._stable_speaker_map(
            _diarized().turns, {"SPEAKER_00": "Alice"}
        )
        assert mapping == {"SPEAKER_00": "Alice", "SPEAKER_01": "Speaker 1"}


def test_with_nothing_enrolled_everyone_is_numbered():
    with isolated_voiceprints():
        _segments_, speakers = _transcriber()._apply_speakers(
            _segments(), [], _diarized(), lambda _m: None
        )
        assert speakers == ["Speaker 1", "Speaker 2"]


def test_recognition_is_reported_in_the_log():
    with isolated_voiceprints():
        voiceprints.enroll("Alice", _vector(1, 0), seconds=120.0)
        logs: list[str] = []
        _transcriber()._apply_speakers(_segments(), [], _diarized(), logs.append)
        assert any("Recognised" in line and "Alice" in line for line in logs)


def test_a_result_without_embeddings_still_works():
    """Older cached diarizations, and the cloud engines, provide none."""
    with isolated_voiceprints():
        voiceprints.enroll("Alice", _vector(1, 0), seconds=120.0)
        bare = DiarizationResult(turns=_diarized().turns)
        _segments_, speakers = _transcriber()._apply_speakers(
            _segments(), [], bare, lambda _m: None
        )
        assert speakers == ["Speaker 1", "Speaker 2"]


def test_a_plain_list_of_turns_is_still_accepted():
    """The old call shape, from callers that never had embeddings."""
    with isolated_voiceprints():
        _segments_, speakers = _transcriber()._apply_speakers(
            _segments(), [], _diarized().turns, lambda _m: None
        )
        assert speakers == ["Speaker 1", "Speaker 2"]


def test_a_broken_voiceprint_store_costs_names_not_the_transcript():
    with isolated_voiceprints() as directory:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "broken.json").write_text("{ not json", encoding="utf-8")
        segments, speakers = _transcriber()._apply_speakers(
            _segments(), [], _diarized(), lambda _m: None
        )
        assert len(segments) == 2
        assert speakers == ["Speaker 1", "Speaker 2"]


# ---- word-level path -------------------------------------------------
def test_names_survive_the_word_level_regrouping():
    """words_to_segments rebuilds its own numbering, so it has to be told."""
    with isolated_voiceprints():
        voiceprints.enroll("Alice", _vector(1, 0), seconds=120.0)
        words = [
            {"type": "word", "text": "Hello", "start": 10.0, "end": 10.5},
            {"type": "word", "text": "there", "start": 10.5, "end": 11.0},
            {"type": "word", "text": "Hello", "start": 100.0, "end": 100.5},
            {"type": "word", "text": "back", "start": 100.5, "end": 101.0},
        ]
        _segments_, speakers = _transcriber()._apply_speakers(
            _segments(), words, _diarized(), lambda _m: None
        )
        assert speakers == ["Alice", "Speaker 1"]


# ---- the vectors offered to the rename dialog ------------------------
def test_voice_data_is_keyed_by_the_label_the_reader_sees():
    with isolated_voiceprints():
        vectors, seconds = Transcriber._speaker_voice_data(
            _diarized(), {"SPEAKER_00": "Alice"}
        )
        assert set(vectors) == {"Alice", "Speaker 1"}
        assert seconds["Alice"] == pytest.approx(90.0)


# ---- enrolling from the rename dialog --------------------------------
@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _result() -> TranscriptResult:
    return TranscriptResult(
        recording=Recording(source=Source.PLAUD, id="f1", name="Team sync"),
        segments=[
            Segment(start=0.0, end=10.0, text="Hello there.", speaker="Speaker 1"),
            Segment(start=90.0, end=100.0, text="Hello back.", speaker="Speaker 2"),
        ],
        speakers=["Speaker 1", "Speaker 2"],
        speaker_embeddings={"Speaker 1": _vector(1, 0), "Speaker 2": _vector(0, 1)},
        speaker_seconds={"Speaker 1": 90.0, "Speaker 2": 90.0},
    )


def test_naming_and_remembering_a_speaker(app):
    with isolated_voiceprints():
        dlg = SpeakerRenameDialog(_result())
        dlg.edits["Speaker 1"].setText("Alice")
        dlg.remember["Speaker 1"].setChecked(True)
        assert dlg.apply_enrollments() == ["Alice"]
        stored = voiceprints.get_profile("Alice")
        assert stored is not None
        assert stored.prints[0].source == "Team sync"


def test_remembering_without_naming_stores_nothing(app):
    """Enrolling somebody as 'Speaker 2' would be useless and confusing later."""
    with isolated_voiceprints():
        dlg = SpeakerRenameDialog(_result())
        dlg.remember["Speaker 1"].setChecked(True)
        assert dlg.apply_enrollments() == []
        assert voiceprints.load_profiles() == []


def test_a_speaker_with_too_little_speech_cannot_be_remembered(app):
    with isolated_voiceprints():
        result = _result()
        result.speaker_seconds["Speaker 1"] = voiceprints.MIN_ENROLL_SECONDS - 1
        dlg = SpeakerRenameDialog(result)
        assert not dlg.remember["Speaker 1"].isEnabled()
        assert "needed to remember" in dlg.remember["Speaker 1"].toolTip()


def test_a_speaker_with_no_voice_data_cannot_be_remembered(app):
    """Cloud engines diarize without exposing anything to compare."""
    with isolated_voiceprints():
        result = _result()
        result.speaker_embeddings = {}
        dlg = SpeakerRenameDialog(result)
        assert not dlg.remember["Speaker 1"].isEnabled()


def test_renaming_still_works_when_remembering_fails(app):
    """The rename is what the user came for; the voiceprint is a bonus."""
    with isolated_voiceprints():
        result = _result()
        dlg = SpeakerRenameDialog(result)
        dlg.edits["Speaker 1"].setText("Alice")
        dlg.remember["Speaker 1"].setChecked(True)
        result.speaker_embeddings["Speaker 1"] = [0.0] * DIM   # unusable
        logs: list[str] = []
        assert dlg.apply_enrollments(log=logs.append) == []
        assert dlg.renames() == {"Speaker 1": "Alice"}


def test_enrolling_the_same_person_from_another_recording(app):
    """A second capture from a different microphone is added, not merged."""
    with isolated_voiceprints():
        voiceprints.enroll("Alice", _vector(1, 0), seconds=120.0, source="first.mp3")
        dlg = SpeakerRenameDialog(_result())
        dlg.edits["Speaker 1"].setText("Alice")
        dlg.remember["Speaker 1"].setChecked(True)
        assert "adds another sample" in dlg.remember["Speaker 1"].toolTip()
        dlg.apply_enrollments()
        assert len(voiceprints.get_profile("Alice").prints) == 2
