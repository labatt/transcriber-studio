# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Recovering word timings for a transcript that has none, so its speakers
can be detected again and regrouped word by word."""

from __future__ import annotations

import pytest

from transcriber_studio import jobs, word_timings
from transcriber_studio.config import Settings
from transcriber_studio.diarization import DiarizationResult, SpeakerTurn
from transcriber_studio.jobs import JobRunner
from transcriber_studio.models import Recording, Segment, Source, TranscriptResult
from transcriber_studio.transcriber import Transcriber

from .support import isolated_voiceprints


def _words(entries):
    return [e for e in entries if e["type"] == "word"]


# ---- the fallback -------------------------------------------------------------
def test_estimated_words_share_the_segment_evenly_and_say_so():
    words = word_timings.estimate_words([Segment(10.0, 14.0, "one two three four")])
    spoken = _words(words)
    assert [w["text"] for w in spoken] == ["one", "two", "three", "four"]
    assert [(w["start"], w["end"]) for w in spoken] == [
        (10.0, 11.0), (11.0, 12.0), (12.0, 13.0), (13.0, 14.0)
    ]
    assert all(w["estimated"] for w in spoken)
    assert [e["type"] for e in words] == ["word", "spacing"] * 3 + ["word"]


def test_an_empty_segment_yields_no_words():
    assert word_timings.estimate_words([Segment(0.0, 1.0, "   ")]) == []


# ---- putting aligned and unalignable words together ---------------------------
def test_unalignable_tokens_take_the_gap_between_their_neighbours():
    """Numbers and symbols are not in the aligner's alphabet; they get the
    time between the words around them, in order."""
    seg = Segment(0.0, 10.0, "sold 400 units in 2020 today")
    tokens = seg.text.split()
    alignable = [True, False, True, True, False, True]
    times = [(0.0, 0.5), (2.5, 3.0), (3.0, 3.2), (6.0, 6.5)]
    words = _words(word_timings.assemble(seg, tokens, alignable, times))
    assert [w["text"] for w in words] == tokens
    assert (words[1]["start"], words[1]["end"]) == (0.5, 2.5)
    assert (words[4]["start"], words[4]["end"]) == (3.2, 6.0)
    assert words[1]["estimated"] and words[4]["estimated"]
    assert "estimated" not in words[0]


def test_unalignable_tokens_at_the_edges_lean_on_the_segment_bounds():
    seg = Segment(5.0, 9.0, "2020 was good 100%")
    tokens = seg.text.split()
    alignable = [False, True, True, False]
    times = [(6.0, 6.4), (6.5, 7.0)]
    words = _words(word_timings.assemble(seg, tokens, alignable, times))
    assert (words[0]["start"], words[0]["end"]) == (5.0, 6.0)
    assert (words[3]["start"], words[3]["end"]) == (7.0, 9.0)


def test_a_run_of_unalignable_tokens_shares_its_gap_evenly():
    seg = Segment(0.0, 10.0, "a 1 2 3 b")
    alignable = [True, False, False, False, True]
    words = _words(word_timings.assemble(seg, seg.text.split(), alignable,
                                         [(0.0, 1.0), (7.0, 8.0)]))
    assert [(w["start"], w["end"]) for w in words[1:4]] == [(1.0, 3.0), (3.0, 5.0), (5.0, 7.0)]


# ---- recover_words chooses honestly -------------------------------------------
def test_a_language_the_aligner_does_not_know_is_estimated_and_logged():
    lines = []
    words = word_timings.recover_words(
        "audio.mp3", [Segment(0.0, 2.0, "Guten Tag")], language="de", log=lines.append
    )
    assert all(w["estimated"] for w in _words(words))
    assert any("English only" in line for line in lines)


def test_an_unavailable_aligner_falls_back_rather_than_failing(monkeypatch):
    def broken(_log):
        raise ImportError("no torchaudio here")

    monkeypatch.setattr(word_timings, "_Aligner", broken)
    lines = []
    words = word_timings.recover_words(
        "audio.mp3", [Segment(0.0, 2.0, "Hello there")], language="en", log=lines.append
    )
    assert [w["text"] for w in _words(words)] == ["Hello", "there"]
    assert all(w["estimated"] for w in _words(words))
    assert any("aligner unavailable" in line for line in lines)


def test_aligned_segments_and_estimated_ones_are_stitched_in_order(monkeypatch):
    class _FakeAligner:
        def __init__(self, log):
            pass

        def align(self, samples, seg):
            if seg.text.startswith("Hi"):
                return None          # the aligner gave up on this one
            return word_timings.assemble(
                seg, seg.text.split(), [True] * len(seg.text.split()),
                [(seg.start + 0.1 * k, seg.start + 0.1 * k + 0.05)
                 for k in range(len(seg.text.split()))],
            )

    monkeypatch.setattr(word_timings, "_Aligner", _FakeAligner)
    monkeypatch.setattr(word_timings, "load_mono", lambda _p: object())
    lines = []
    words = word_timings.recover_words(
        "audio.mp3",
        [Segment(0.0, 1.0, "Hello there"), Segment(5.0, 6.0, "Hi you"), Segment(9.0, 9.5, "Yo")],
        language="", log=lines.append,
    )
    spoken = _words(words)
    assert [w["text"] for w in spoken] == ["Hello", "there", "Hi", "you", "Yo"]
    assert spoken[0]["start"] == 0.0 and "estimated" not in spoken[0]
    assert spoken[2]["estimated"] and spoken[2]["start"] == 5.0
    assert any("2 turn(s) aligned, 1 spaced evenly" in line for line in lines)


def test_cancel_is_honoured_between_segments(monkeypatch):
    from transcriber_studio.job_cancel import JobCancelled

    class _Aligner:
        def __init__(self, log):
            pass

        def align(self, samples, seg):
            return word_timings.estimate_words([seg])

    monkeypatch.setattr(word_timings, "_Aligner", _Aligner)
    monkeypatch.setattr(word_timings, "load_mono", lambda _p: object())
    with pytest.raises(JobCancelled):
        word_timings.recover_words("a.mp3", [Segment(0, 1, "x")], should_cancel=lambda: True)


# ---- wired into Detect speakers ------------------------------------------------
class _ThreeSpeakerDiarizer:
    def __init__(self, token, device="auto"):
        pass

    def diarize(self, audio_path, min_speakers=0, max_speakers=0, progress_cb=None,
                log_cb=None, should_cancel=None):
        return DiarizationResult(turns=[
            SpeakerTurn(0.0, 2.0, "SPEAKER_00"),
            SpeakerTurn(4.5, 6.0, "SPEAKER_01"),
            SpeakerTurn(9.5, 11.0, "SPEAKER_02"),
        ], embeddings={})


def test_detecting_speakers_on_a_wordless_transcript_recovers_the_words_first(monkeypatch):
    """The user's case: a transcript from before words were kept, diarized
    under the wrong limit. Detect speakers must still split it properly."""
    monkeypatch.setattr(jobs.diarization, "Diarizer", _ThreeSpeakerDiarizer)
    monkeypatch.setattr(jobs.diarization, "is_available", lambda: True)
    recovered = [
        {"type": "word", "text": "Hello", "start": 0.0, "end": 0.5},
        {"type": "spacing", "text": " "},
        {"type": "word", "text": "there.", "start": 0.6, "end": 1.0},
        {"type": "spacing", "text": " "},
        {"type": "word", "text": "Hi.", "start": 5.0, "end": 5.4},
        {"type": "spacing", "text": " "},
        {"type": "word", "text": "Yo.", "start": 10.0, "end": 10.3},
    ]
    asked = {}

    def fake_recover(audio_path, segments, language="", log=None, should_cancel=None):
        asked["segments"] = [s.text for s in segments]
        return recovered

    monkeypatch.setattr(jobs.word_timings, "recover_words", fake_recover)

    runner = JobRunner.__new__(JobRunner)
    runner.s = Settings(hf_token="hf_x")
    runner.transcriber = Transcriber()
    runner.client = None
    runner._ensure_audio = lambda *a, **k: "audio.mp3"
    result = TranscriptResult(
        recording=Recording(source=Source.LOCAL, id="r1", name="Team", local_path="audio.mp3"),
        segments=[Segment(0.0, 10.3, "Hello there. Hi. Yo.", speaker="Speaker 1")],
        speakers=["Speaker 1"], language="en",
    )
    with isolated_voiceprints():
        runner.apply_diarization(result, max_speakers=6)

    assert asked["segments"] == ["Hello there. Hi. Yo."]
    assert [s.text for s in result.segments] == ["Hello there.", "Hi.", "Yo."]
    assert result.speakers == ["Speaker 1", "Speaker 2", "Speaker 3"]
    assert result.words, "the recovered words stay on the transcript for next time"
