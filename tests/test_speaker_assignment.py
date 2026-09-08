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


# ---- words in a pause between turns ----------------------------------------
def test_a_word_in_a_pause_between_turns_takes_the_nearest_speaker():
    """Diarization turns are cut at speech. A word timed inside the breath
    between two clauses overlaps no turn; live, that left a one-speaker clip as
    five segments, two of them attributed to nobody."""
    from transcriber_studio import diarization
    from transcriber_studio.diarization import SpeakerTurn

    turns = [SpeakerTurn(0.0, 1.0, "SPEAKER_00"), SpeakerTurn(1.4, 3.0, "SPEAKER_00")]
    assert diarization.assign_speaker(1.1, 1.3, turns) is None
    assert diarization.nearest_speaker(1.1, 1.3, turns) == "SPEAKER_00"


def test_the_nearest_turn_wins_when_two_speakers_flank_the_gap():
    from transcriber_studio import diarization
    from transcriber_studio.diarization import SpeakerTurn

    turns = [SpeakerTurn(0.0, 1.0, "SPEAKER_00"), SpeakerTurn(2.0, 3.0, "SPEAKER_01")]
    assert diarization.nearest_speaker(1.05, 1.2, turns) == "SPEAKER_00"
    assert diarization.nearest_speaker(1.8, 1.95, turns) == "SPEAKER_01"


def test_a_word_far_from_any_turn_still_gets_no_speaker():
    """A long silence is a long silence; nobody said the word that pyannote
    heard nothing around."""
    from transcriber_studio import diarization
    from transcriber_studio.diarization import SpeakerTurn

    turns = [SpeakerTurn(0.0, 1.0, "SPEAKER_00")]
    assert diarization.nearest_speaker(10.0, 10.5, turns) is None


def test_a_single_speaker_clip_stays_one_speaker_through_its_pauses():
    """The live case: three MAI segments over one pyannote speaker with pauses
    between turns must not fragment into speakerless pieces."""
    from transcriber_studio.diarization import DiarizationResult, SpeakerTurn
    from transcriber_studio.models import Segment
    from transcriber_studio.transcriber import Transcriber

    # Spacing entries interleaved, as the cloud engines actually send them.
    words = [
        {"type": "word", "text": "First", "start": 0.4, "end": 0.9},
        {"type": "spacing", "text": " "},
        {"type": "word", "text": "book", "start": 1.0, "end": 1.2},   # in a pause
        {"type": "spacing", "text": " "},
        {"type": "word", "text": "for", "start": 1.5, "end": 1.7},
    ]
    diarized = DiarizationResult(turns=[
        SpeakerTurn(0.3, 0.95, "SPEAKER_00"), SpeakerTurn(1.4, 2.0, "SPEAKER_00"),
    ])
    segments, speakers = Transcriber.__new__(Transcriber)._apply_speakers(
        [Segment(0.4, 1.7, "First book for")], words, diarized, lambda _m: None, {}
    )
    assert speakers == ["Speaker 1"]
    assert [s.speaker for s in segments] == ["Speaker 1"]
    assert segments[0].text == "First book for"
