# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Detecting speakers again, with different limits, on a finished transcript.

The case that prompted this: a six-person meeting transcribed with "at most 2"
speakers. The two turns pyannote drew under that limit each merge several real
people, so relabelling them can never recover the other four. The words have
to be kept and regrouped.
"""

from __future__ import annotations

import pytest

from transcriber_studio import jobs, queue_store
from transcriber_studio import resume as resume_store
from transcriber_studio.config import Settings
from transcriber_studio.diarization import DiarizationResult, SpeakerTurn
from transcriber_studio.jobs import JobRunner, copy_transcript, describe_bounds
from transcriber_studio.models import Recording, Segment, Source, TranscriptResult
from transcriber_studio.transcriber import Transcriber

from .support import isolated_voiceprints

WORDS = [
    {"type": "word", "text": "Hello", "start": 0.0, "end": 0.5},
    {"type": "spacing", "text": " "},
    {"type": "word", "text": "there.", "start": 0.6, "end": 1.0},
    {"type": "word", "text": "Hi.", "start": 5.0, "end": 5.4},
    {"type": "word", "text": "Yo.", "start": 10.0, "end": 10.3},
]
# What a run limited to two speakers produced: one turn, one label.
TWO_SPEAKER_TURNS = [SpeakerTurn(0.0, 11.0, "SPEAKER_00")]
# What the audio actually contains.
THREE_SPEAKER_TURNS = [
    SpeakerTurn(0.0, 2.0, "SPEAKER_00"),
    SpeakerTurn(4.5, 6.0, "SPEAKER_01"),
    SpeakerTurn(9.5, 11.0, "SPEAKER_02"),
]


def _recording() -> Recording:
    return Recording(source=Source.LOCAL, id="r1", name="Team", local_path="audio.mp3")


def _finished_under_two() -> TranscriptResult:
    return TranscriptResult(
        recording=_recording(),
        segments=[Segment(0.0, 10.3, "Hello there. Hi. Yo.", speaker="Speaker 1")],
        speakers=["Speaker 1"],
        words=[dict(w) for w in WORDS],
    )


class _FakeDiarizer:
    calls: list[tuple[int, int]] = []
    turns = THREE_SPEAKER_TURNS

    def __init__(self, token, device="auto"):
        pass

    def diarize(self, audio_path, min_speakers=0, max_speakers=0, progress_cb=None,
                log_cb=None, should_cancel=None):
        _FakeDiarizer.calls.append((min_speakers, max_speakers))
        return DiarizationResult(turns=list(self.turns), embeddings={})


@pytest.fixture
def runner(monkeypatch):
    _FakeDiarizer.calls = []
    monkeypatch.setattr(jobs.diarization, "Diarizer", _FakeDiarizer)
    monkeypatch.setattr(jobs.diarization, "is_available", lambda: True)
    r = JobRunner.__new__(JobRunner)
    r.s = Settings(hf_token="hf_x", min_speakers=0, max_speakers=2)
    r.transcriber = Transcriber()
    r.client = None
    r._ensure_audio = lambda *a, **k: "audio.mp3"
    return r


# ---- the words survive everything a finished job goes through ---------------
def test_the_transcriber_keeps_the_words_on_the_result(monkeypatch):
    t = Transcriber()
    monkeypatch.setattr(
        t, "_decode_or_restore",
        lambda *a, **k: ([Segment(0.0, 1.0, "Hello there.")], "en", [dict(w) for w in WORDS[:3]]),
    )
    from transcriber_studio.transcriber import TranscribeOptions

    result = t._transcribe_single(
        _recording(), "audio.mp3", None, None,
        TranscribeOptions(diarization_enabled=False), None, lambda _m: None,
    )
    assert [w["text"] for w in result.words if w["type"] == "word"] == ["Hello", "there."]


def test_the_queue_store_round_trips_the_words():
    original = _finished_under_two()
    back = queue_store._transcript_from_dict(
        _recording(), queue_store._transcript_to_dict(original)
    )
    assert back.words == original.words
    assert back.words is not original.words


def test_the_resume_store_round_trips_the_words():
    original = _finished_under_two()
    back = resume_store.transcript_from_dict(
        _recording(), resume_store.transcript_to_dict(original)
    )
    assert back.words == original.words


def test_a_queue_written_before_words_existed_still_loads():
    td = queue_store._transcript_to_dict(_finished_under_two())
    del td["words"]
    assert queue_store._transcript_from_dict(_recording(), td).words == []


def test_copying_a_transcript_copies_the_words_not_the_list():
    original = _finished_under_two()
    copy = copy_transcript(original)
    copy.words[0]["text"] = "changed"
    assert original.words[0]["text"] == "Hello"


# ---- detecting again ---------------------------------------------------------
def test_new_limits_reach_pyannote_and_beat_the_saved_ones(runner):
    with isolated_voiceprints():
        runner.apply_diarization(_finished_under_two(), min_speakers=3, max_speakers=6)
    assert _FakeDiarizer.calls == [(3, 6)]


def test_without_an_override_the_saved_limits_are_used(runner):
    with isolated_voiceprints():
        runner.apply_diarization(_finished_under_two())
    assert _FakeDiarizer.calls == [(0, 2)]


def test_a_two_speaker_transcript_regroups_into_the_three_people_who_spoke(runner):
    """The whole point: the one merged turn becomes three, at the word level."""
    result = _finished_under_two()
    with isolated_voiceprints():
        runner.apply_diarization(result, min_speakers=0, max_speakers=6)

    assert result.speakers == ["Speaker 1", "Speaker 2", "Speaker 3"]
    assert [s.text for s in result.segments] == ["Hello there.", "Hi.", "Yo."]
    assert [s.speaker for s in result.segments] == ["Speaker 1", "Speaker 2", "Speaker 3"]
    # The words now carry the new raw labels, ready for another round.
    assert [w.get("speaker_id") for w in result.words if w["type"] == "word"] == [
        "SPEAKER_00", "SPEAKER_00", "SPEAKER_01", "SPEAKER_02",
    ]


def test_without_words_they_are_recovered_before_regrouping(runner, monkeypatch):
    """A transcript from before words were kept: the words are recovered from
    the audio first, so the merged turn can still be split. Even the evenly
    spaced fallback is enough to find the three people here."""
    from transcriber_studio import word_timings

    monkeypatch.setattr(
        jobs.word_timings, "recover_words",
        lambda audio, segments, language="", log=None, should_cancel=None:
            word_timings.estimate_words(segments),
    )
    result = _finished_under_two()
    result.words = []
    lines = []
    with isolated_voiceprints():
        runner.apply_diarization(result, log_cb=lines.append, max_speakers=6)

    assert any("recovering them from the audio" in line for line in lines)
    assert result.speakers == ["Speaker 1", "Speaker 2", "Speaker 3"]
    assert len(result.segments) == 3
    assert result.words and all(w.get("estimated") for w in result.words if w["type"] == "word")


def test_the_log_says_which_limits_ran(runner):
    lines = []
    with isolated_voiceprints():
        runner.apply_diarization(_finished_under_two(), log_cb=lines.append,
                                 min_speakers=3, max_speakers=6)
    assert any("at least 3, at most 6 speakers" in line for line in lines)


@pytest.mark.parametrize("lo,hi,text", [
    (0, 0, "any number of speakers"),
    (0, 2, "at most 2 speakers"),
    (3, 0, "at least 3 speakers"),
    (3, 6, "at least 3, at most 6 speakers"),
    (4, 4, "exactly 4 speakers"),
    (1, 1, "exactly 1 speaker"),
])
def test_bounds_read_like_a_person_wrote_them(lo, hi, text):
    assert describe_bounds(lo, hi) == text


# ---- the dialog --------------------------------------------------------------
@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_the_dialog_starts_from_the_saved_limits_and_explains_itself(app):
    from transcriber_studio.ui.speaker_count_dialog import SpeakerCountDialog

    dlg = SpeakerCountDialog(_finished_under_two(), Settings(min_speakers=1, max_speakers=2))
    assert dlg.bounds() == (1, 2)
    text = dlg.findChildren(type(dlg.layout().itemAt(0).widget()))[0].text()
    assert "already has 1 speaker" in text
    assert "regrouped word by word" in text


def test_the_dialog_warns_when_there_are_no_words_to_regroup(app):
    from transcriber_studio.ui.speaker_count_dialog import SpeakerCountDialog

    result = _finished_under_two()
    result.words = []
    dlg = SpeakerCountDialog(result, Settings(), cleanup_applied=True)
    text = dlg.layout().itemAt(0).widget().text()
    assert "recovered first by aligning" in text
    assert "AI cleanup pass will be undone" in text


def test_a_maximum_below_the_minimum_is_raised_to_meet_it(app):
    from transcriber_studio.ui.speaker_count_dialog import SpeakerCountDialog

    dlg = SpeakerCountDialog(_finished_under_two(), Settings())
    dlg.min_speakers.setValue(5)
    dlg.max_speakers.setValue(2)
    assert dlg.bounds() == (5, 5)
