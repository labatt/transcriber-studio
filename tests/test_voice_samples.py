# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Which seconds of each speaker a person gets to hear.

The rule that matters most is "never just one": a single clip can land on an
"mm-hm", and then the listener has nothing to go on. The rest is spread —
samples from far apart in the recording beat two from the same minute.
"""

from __future__ import annotations

from transcriber_studio import voice_samples as vs
from transcriber_studio.diarization import SpeakerTurn


def _turn(start, end, who="SPEAKER_00"):
    return SpeakerTurn(start, end, who)


# ---- stretches -------------------------------------------------------
def test_breath_gaps_are_closed_into_one_stretch():
    turns = [_turn(0, 10), _turn(10.3, 20), _turn(20.4, 25)]
    assert vs.stretches(turns, "SPEAKER_00") == [(0, 25)]


def test_a_gap_someone_else_spoke_in_is_not_closed():
    turns = [_turn(0, 10), _turn(10.1, 10.5, "SPEAKER_01"), _turn(10.5, 20)]
    assert vs.stretches(turns, "SPEAKER_00") == [(0, 10), (10.5, 20)]


def test_a_long_pause_starts_a_new_stretch():
    turns = [_turn(0, 10), _turn(15, 20)]
    assert vs.stretches(turns, "SPEAKER_00") == [(0, 10), (15, 20)]


# ---- picking ---------------------------------------------------------
def test_samples_come_from_stretches_spread_through_the_recording():
    turns = [_turn(0, 30), _turn(100, 130), _turn(1000, 1030), _turn(2000, 2030), _turn(2100, 2130)]
    clips = vs.pick_clips(turns, "SPEAKER_00", count=3)
    starts = [c.start for c in clips]
    assert len(clips) == 3
    assert starts == sorted(starts), "buttons read as a timeline"
    # The longest is first (all equal here, so the first), then the two
    # furthest from what is already chosen: the far end, then the middle —
    # never the stretch a minute and a half after the first.
    assert starts == [0, 1000, 2100]


def test_every_clip_is_at_most_the_sample_length():
    turns = [_turn(0, 600), _turn(900, 1500)]
    for clip in vs.pick_clips(turns, "SPEAKER_00"):
        assert clip.seconds <= vs.CLIP_SECONDS + 1e-6


def test_a_speaker_with_one_long_stretch_still_gets_several_samples():
    """One monologue must not mean one sample."""
    clips = vs.pick_clips([_turn(0, 120)], "SPEAKER_00")
    assert len(clips) >= vs.MIN_CLIPS
    starts = [c.start for c in clips]
    assert len(set(starts)) == len(starts)
    assert all(c.end <= 120 for c in clips)


def test_a_speaker_with_one_short_stretch_gets_what_there_is():
    clips = vs.pick_clips([_turn(100, 105)], "SPEAKER_00")
    assert clips == [vs.Clip("SPEAKER_00", 100.0, 105.0)]


def test_tiny_stretches_are_passed_over_when_longer_ones_exist():
    turns = [_turn(0, 1.0), _turn(50, 70), _turn(200, 1.5 + 200), _turn(400, 430)]
    clips = vs.pick_clips(turns, "SPEAKER_00")
    assert all(c.start in (50, 400) for c in clips), [c.start for c in clips]


def test_a_speaker_who_never_spoke_has_no_clips():
    assert vs.pick_clips([_turn(0, 10, "SPEAKER_01")], "SPEAKER_00") == []


def test_clips_for_follows_first_speech_order():
    turns = [_turn(50, 80, "SPEAKER_02"), _turn(0, 30, "SPEAKER_00"), _turn(100, 130, "SPEAKER_01")]
    assert list(vs.clips_for(turns)) == ["SPEAKER_00", "SPEAKER_02", "SPEAKER_01"]


def test_clock():
    assert vs.clock(0) == "0:00"
    assert vs.clock(754) == "12:34"
    assert vs.clock(3723) == "1:02:03"
