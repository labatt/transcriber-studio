# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Speakers are assigned per word, not per decoded segment.

Whisper cuts segments on pauses and punctuation, never on speaker changes, so
a segment routinely holds two people. Labelling the whole segment by whoever
had the most overlap files one person's words under the other's name.
"""

from __future__ import annotations

from transcriber_studio.diarization import SpeakerTurn
from transcriber_studio.models import Segment
from transcriber_studio.transcriber import Transcriber

# One handover, mid-segment.
TURNS = [
    SpeakerTurn(start=0.0, end=2.0, speaker="SPEAKER_00"),
    SpeakerTurn(start=2.0, end=4.0, speaker="SPEAKER_01"),
]
SPANNING_SEGMENT = [Segment(start=0.0, end=4.0, text="Hello there General Kenobi")]
WORDS = [
    {"type": "word", "text": "Hello", "start": 0.0, "end": 0.5},
    {"type": "word", "text": " there", "start": 0.5, "end": 1.9},
    {"type": "word", "text": " General", "start": 2.1, "end": 3.0},
    {"type": "word", "text": " Kenobi", "start": 3.0, "end": 4.0},
]


def _apply(segments, words):
    return Transcriber.__new__(Transcriber)._apply_speakers(
        segments, words, TURNS, lambda m: None
    )


def test_a_segment_spanning_a_handover_is_split_at_the_handover():
    segments, speakers = _apply(list(SPANNING_SEGMENT), [dict(w) for w in WORDS])

    assert len(segments) == 2, "the segment held two people and stayed whole"
    assert segments[0].text == "Hello there"
    assert segments[1].text == "General Kenobi"
    assert segments[0].speaker != segments[1].speaker
    assert speakers == ["Speaker 1", "Speaker 2"]


def test_the_old_whole_segment_behaviour_gets_it_wrong():
    """Why the change was needed: with no words, both halves go to one name."""
    segments, _ = _apply(list(SPANNING_SEGMENT), [])

    assert len(segments) == 1
    assert segments[0].text == "Hello there General Kenobi"
    # Everything Speaker 2 said is filed under Speaker 1.
    assert segments[0].speaker == "Speaker 1"


def test_a_segment_with_one_speaker_in_it_is_left_alone():
    only_first = [Segment(start=0.0, end=1.9, text="Hello there")]
    words = [dict(w) for w in WORDS[:2]]
    segments, speakers = _apply(only_first, words)

    assert len(segments) == 1
    assert segments[0].text == "Hello there"
    assert speakers == ["Speaker 1"]


def test_falling_back_to_segments_when_the_decoder_gave_no_word_timings():
    """word_timestamps can be off; the old path must still label the transcript."""
    segments = [
        Segment(start=0.0, end=1.9, text="Hello there"),
        Segment(start=2.1, end=4.0, text="General Kenobi"),
    ]
    labelled, speakers = _apply(segments, [])

    assert [s.speaker for s in labelled] == ["Speaker 1", "Speaker 2"]
    assert speakers == ["Speaker 1", "Speaker 2"]


def test_speaker_numbers_follow_who_spoke_first():
    later_first = [
        SpeakerTurn(start=0.0, end=2.0, speaker="SPEAKER_07"),
        SpeakerTurn(start=2.0, end=4.0, speaker="SPEAKER_03"),
    ]
    segments, speakers = Transcriber.__new__(Transcriber)._apply_speakers(
        list(SPANNING_SEGMENT), [dict(w) for w in WORDS], later_first, lambda m: None
    )
    assert [s.speaker for s in segments] == ["Speaker 1", "Speaker 2"]
    assert speakers == ["Speaker 1", "Speaker 2"]
