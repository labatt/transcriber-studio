# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""The decoder's own opinion of each segment, kept rather than discarded.

Whisper computes avg_logprob, no_speech_prob and a compression ratio for every
segment, and a probability for every word. All of it used to be dropped on the
floor, so a transcript could not say which lines it was unsure of and a reader
had to check all of them or none.

None has to keep meaning "the engine did not say" the whole way through. A
missing score silently becoming zero would read as "certainly wrong", which is
the opposite of unknown.
"""

from __future__ import annotations

import json
import math

import pytest

from transcriber_studio import formatters, transcriber
from transcriber_studio import resume as resume_store
from transcriber_studio.models import Recording, Segment, Source, TranscriptResult
from transcriber_studio.word_segments import word_probability, words_to_segments


def _segment(text: str = "Hello.", **kwargs) -> Segment:
    return Segment(start=kwargs.pop("start", 0.0), end=kwargs.pop("end", 1.0),
                   text=text, **kwargs)


# ---- the readable score ----------------------------------------------
def test_confidence_is_the_probability_behind_the_log():
    seg = _segment(avg_logprob=math.log(0.42))
    assert seg.confidence == pytest.approx(0.42)


def test_confidence_is_unknown_when_the_engine_said_nothing():
    """Not zero. Zero would read as 'certainly wrong'."""
    assert _segment().confidence is None


def test_a_perfect_decode_is_one():
    assert _segment(avg_logprob=0.0).confidence == pytest.approx(1.0)


# ---- what the decoder hands over -------------------------------------
class _FakeWord:
    def __init__(self, word, start, end, probability):
        self.word, self.start, self.end = word, start, end
        self.probability = probability


class _FakeSegment:
    def __init__(self, start, end, text, avg_logprob, no_speech_prob,
                 compression_ratio, words=()):
        self.start, self.end, self.text = start, end, text
        self.avg_logprob = avg_logprob
        self.no_speech_prob = no_speech_prob
        self.compression_ratio = compression_ratio
        self.words = list(words)


class _FakeInfo:
    language = "en"
    duration = 60.0
    duration_after_vad = 60.0


def _decode(fake_segments):
    """Run _run_whisper against a stand-in decoder."""
    class _Model:
        @staticmethod
        def transcribe(_path, **_kwargs):
            return iter(fake_segments), _FakeInfo()

    logs: list[str] = []
    from transcriber_studio.transcriber import TranscribeOptions, Transcriber

    segments, language, words = Transcriber.__new__(Transcriber)._run_whisper(
        _Model(), "a.wav", None, TranscribeOptions(), logs.append, None, 60.0
    )
    return segments, words, logs


def test_the_scores_survive_the_decode_loop():
    segments, _words, _logs = _decode([
        _FakeSegment(0.0, 2.0, " Hello.", math.log(0.9), 0.01, 1.2),
    ])
    assert segments[0].confidence == pytest.approx(0.9)
    assert segments[0].no_speech_prob == pytest.approx(0.01)
    assert segments[0].compression_ratio == pytest.approx(1.2)


def test_word_probabilities_are_kept():
    _segments, words, _logs = _decode([
        _FakeSegment(0.0, 2.0, " Hi.", math.log(0.9), 0.01, 1.2,
                     words=[_FakeWord(" Hi", 0.0, 0.5, 0.87)]),
    ])
    assert words[0]["probability"] == pytest.approx(0.87)


def test_an_engine_reporting_nothing_yields_unknown_not_zero():
    segments, _words, _logs = _decode([
        _FakeSegment(0.0, 2.0, " Hello.", None, None, None),
    ])
    assert segments[0].confidence is None
    assert segments[0].no_speech_prob is None


def test_a_nonsense_score_is_treated_as_unknown():
    segments, _words, _logs = _decode([
        _FakeSegment(0.0, 2.0, " Hello.", float("nan"), float("inf"), "junk"),
    ])
    assert segments[0].avg_logprob is None
    assert segments[0].no_speech_prob is None
    assert segments[0].compression_ratio is None


# ---- the run log -----------------------------------------------------
def test_a_clean_transcript_says_nothing():
    """One shaky line in an hour is speech, not a finding."""
    segments = [_segment(avg_logprob=math.log(0.95)) for _ in range(10)]
    assert transcriber.confidence_report(segments) == []


def test_a_run_of_shaky_segments_is_reported_with_a_timestamp():
    segments = [_segment(avg_logprob=math.log(0.95)) for _ in range(6)]
    segments += [
        _segment(start=125.0, end=130.0, avg_logprob=math.log(0.2)),
        _segment(avg_logprob=math.log(0.3)),
        _segment(avg_logprob=math.log(0.4)),
    ]
    report = " ".join(transcriber.confidence_report(segments))
    assert "3 of 9 segment(s)" in report
    assert "2:05" in report          # points at the worst one


def test_a_likely_decode_loop_is_called_out():
    segments = [_segment(start=60.0, avg_logprob=math.log(0.8), compression_ratio=2.3)]
    report = " ".join(transcriber.confidence_report(segments))
    assert "repeating itself" in report
    assert "1:00" in report


def test_text_over_non_speech_is_called_out():
    segments = [
        _segment(avg_logprob=math.log(0.8), no_speech_prob=0.9) for _ in range(4)
    ]
    report = " ".join(transcriber.confidence_report(segments))
    assert "probably not speech" in report


def test_nothing_is_reported_for_an_engine_without_scores():
    assert transcriber.confidence_report([_segment() for _ in range(10)]) == []


# ---- word-level regrouping -------------------------------------------
def test_a_rebuilt_segment_keeps_a_confidence():
    """Diarization rebuilds segments from words, and the decoder's own score
    describes segments that no longer exist."""
    words = [
        {"type": "word", "text": "Hello", "start": 0.0, "end": 0.4,
         "probability": 0.9, "speaker_id": "a"},
        {"type": "word", "text": " there", "start": 0.4, "end": 0.8,
         "probability": 0.7, "speaker_id": "a"},
    ]
    segments, _speakers = words_to_segments(words, diarized=True)
    assert segments[0].confidence == pytest.approx(math.sqrt(0.9 * 0.7))


def test_each_rebuilt_segment_scores_only_its_own_words():
    words = [
        {"type": "word", "text": "Hello", "start": 0.0, "end": 0.4,
         "probability": 0.9, "speaker_id": "a"},
        {"type": "word", "text": "Yes", "start": 0.5, "end": 0.9,
         "probability": 0.3, "speaker_id": "b"},
    ]
    segments, _speakers = words_to_segments(words, diarized=True)
    assert segments[0].confidence == pytest.approx(0.9)
    assert segments[1].confidence == pytest.approx(0.3)


def test_words_without_probabilities_leave_confidence_unknown():
    words = [{"type": "word", "text": "Hello", "start": 0.0, "end": 0.4,
              "speaker_id": "a"}]
    segments, _speakers = words_to_segments(words, diarized=True)
    assert segments[0].confidence is None


def test_a_log_probability_from_scribe_is_understood():
    """faster-whisper reports a probability; Scribe reports its log."""
    assert word_probability({"logprob": math.log(0.6)}) == pytest.approx(0.6)
    assert word_probability({"probability": 0.6}) == pytest.approx(0.6)
    assert word_probability({}) is None
    assert word_probability({"probability": "junk"}) is None


# ---- it reaches the reader -------------------------------------------
def _result(segments) -> TranscriptResult:
    return TranscriptResult(
        recording=Recording(source=Source.LOCAL, id="a.wav", name="a"),
        segments=segments,
        language="en",
        model="large-v3",
    )


def test_json_export_carries_the_scores():
    data = json.loads(formatters.render(
        _result([_segment(avg_logprob=math.log(0.42), no_speech_prob=0.03,
                          compression_ratio=1.4)]),
        "json", formatters_opts(),
    ))
    segment = data["segments"][0]
    assert segment["confidence"] == pytest.approx(0.42, abs=1e-3)
    assert segment["no_speech_prob"] == pytest.approx(0.03)
    assert segment["compression_ratio"] == pytest.approx(1.4)


def test_json_export_omits_scores_the_engine_never_gave():
    """Absent means 'this engine does not say'; null would blur that into
    'the decoder was unsure'."""
    data = json.loads(formatters.render(_result([_segment()]), "json", formatters_opts()))
    segment = data["segments"][0]
    assert "confidence" not in segment
    assert "no_speech_prob" not in segment
    assert segment["text"] == "Hello."


def formatters_opts():
    from transcriber_studio.config import Settings

    return Settings()


# ---- resume ----------------------------------------------------------
def test_a_restored_decode_keeps_its_scores():
    """Otherwise resuming a run silently produces a transcript that cannot say
    what it was unsure of, while a straight-through run can."""
    original = [_segment(avg_logprob=math.log(0.5), no_speech_prob=0.2,
                         compression_ratio=1.9)]
    restored, language, _words = resume_store.decode_from_dict(
        json.loads(json.dumps(resume_store.decode_to_dict(original, "en")))
    )
    assert restored[0].confidence == pytest.approx(0.5)
    assert restored[0].no_speech_prob == pytest.approx(0.2)
    assert restored[0].compression_ratio == pytest.approx(1.9)
    assert language == "en"


def test_a_restored_transcript_keeps_its_scores():
    original = _result([_segment(avg_logprob=math.log(0.5), speaker="Alice")])
    restored = resume_store.transcript_from_dict(
        original.recording,
        json.loads(json.dumps(resume_store.transcript_to_dict(original))),
    )
    assert restored.segments[0].confidence == pytest.approx(0.5)
    assert restored.segments[0].speaker == "Alice"


def test_a_checkpoint_from_before_this_change_still_loads():
    older = {"language": "en", "segments": [{"start": 0.0, "end": 1.0, "text": "Hi."}]}
    segments, _language, _words = resume_store.decode_from_dict(older)
    assert segments[0].text == "Hi."
    assert segments[0].confidence is None


def test_a_checkpoint_from_a_later_version_does_not_crash_the_resume():
    newer = {
        "language": "en",
        "segments": [{"start": 0.0, "end": 1.0, "text": "Hi.", "invented_field": 7}],
    }
    segments, _language, _words = resume_store.decode_from_dict(newer)
    assert segments[0].text == "Hi."
