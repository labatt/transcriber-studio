# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Scribe's word list has to come back as the speaker turns the app exports."""

from __future__ import annotations

from transcriber_studio import stt_elevenlabs as el
from transcriber_studio.models import Recording, Source
from transcriber_studio.resume import transcript_key
from transcriber_studio.transcriber import ENGINE_ELEVENLABS, ENGINE_LOCAL, TranscribeOptions


def word(text, start, end, speaker="speaker_0", kind="word"):
    return {"text": text, "start": start, "end": end, "type": kind, "speaker_id": speaker}


def space(start, end, speaker="speaker_0"):
    return word(" ", start, end, speaker, "spacing")


def test_a_speaker_change_ends_the_turn():
    words = [
        word("Morning.", 0.0, 0.6, "speaker_0"),
        space(0.6, 0.8, "speaker_0"),
        word("Morning.", 0.8, 1.4, "speaker_1"),
    ]
    segments, speakers = el.words_to_segments(words, diarized=True)
    assert [s.text for s in segments] == ["Morning.", "Morning."]
    assert [s.speaker for s in segments] == ["Speaker 1", "Speaker 2"]
    assert speakers == ["Speaker 1", "Speaker 2"]


def test_speakers_are_numbered_by_first_appearance():
    words = [word("b", 0.0, 0.2, "speaker_3"), word("a", 1.0, 1.2, "speaker_1")]
    segments, _ = el.words_to_segments(words, diarized=True)
    assert [s.speaker for s in segments] == ["Speaker 1", "Speaker 2"]


def test_without_diarization_no_speaker_is_invented():
    words = [word("hello", 0.0, 0.5, "speaker_0")]
    segments, speakers = el.words_to_segments(words, diarized=False)
    assert speakers == []
    assert segments[0].speaker is None


def test_a_long_silence_starts_a_new_turn():
    words = [
        word("first", 0.0, 0.5),
        word("half", 0.5, 1.0),
        word("second", 30.0, 30.5),        # 29s of nothing in between
    ]
    segments, _ = el.words_to_segments(words, diarized=True)
    assert len(segments) == 2
    assert segments[0].end == 1.0 and segments[1].start == 30.0


def test_a_long_monologue_breaks_at_a_sentence_end():
    words = []
    t = 0.0
    for i in range(60):                    # ~60s of one speaker, no pauses
        words.append(word(f"word{i}." if i % 10 == 9 else f"word{i}", t, t + 0.9))
        words.append(space(t + 0.9, t + 1.0))
        t += 1.0
    segments, _ = el.words_to_segments(words, diarized=True)
    assert len(segments) > 1, "a minute of speech should not be one segment"
    assert all(s.text.rstrip().endswith(".") for s in segments[:-1]), \
        "breaks land on sentence ends, not mid-thought"


def test_spacing_between_words_is_kept_as_text():
    words = [word("we", 0.0, 0.2), space(0.2, 0.3), word("shipped", 0.3, 0.8)]
    segments, _ = el.words_to_segments(words, diarized=True)
    assert segments[0].text == "we shipped"


def test_audio_events_ride_along_in_the_text():
    words = [
        word("that", 0.0, 0.3),
        space(0.3, 0.4),
        word("(laughter)", 0.4, 1.0, kind="audio_event"),
    ]
    segments, _ = el.words_to_segments(words, diarized=True)
    assert "(laughter)" in segments[0].text


def _opts(**kw):
    base = dict(
        engine=ENGINE_ELEVENLABS,
        elevenlabs_model="scribe_v1",
        diarization_enabled=True,
        language="auto",
        max_speakers=0,
        tag_audio_events=False,
    )
    base.update(kw)
    return TranscribeOptions(**base)


def test_auto_language_is_left_for_the_model_to_detect():
    fields = el._form_fields(_opts())
    assert "language_code" not in fields
    assert fields["diarize"] == "true"
    assert fields["timestamps_granularity"] == "word"


def test_an_explicit_language_and_speaker_cap_are_sent():
    fields = el._form_fields(_opts(language="en", max_speakers=3))
    assert fields["language_code"] == "en"
    assert fields["num_speakers"] == "3"


def test_the_speaker_cap_is_clamped_to_what_the_api_accepts():
    assert el._form_fields(_opts(max_speakers=99))["num_speakers"] == "32"


def test_no_speaker_cap_is_sent_when_diarization_is_off():
    fields = el._form_fields(_opts(diarization_enabled=False, max_speakers=4))
    assert fields["diarize"] == "false"
    assert "num_speakers" not in fields


def test_transcribing_without_a_key_says_so_before_uploading():
    rec = Recording(source=Source.LOCAL, id="x", name="x")
    try:
        el.transcribe(rec, __file__, _opts(elevenlabs_api_key=""))
    except el.ElevenLabsError as e:
        assert "API key" in str(e)
    else:
        raise AssertionError("a missing key must not reach the network")


def test_switching_engines_invalidates_the_saved_transcript():
    """Resuming must never hand a Whisper transcript to an ElevenLabs run."""
    rec = Recording(source=Source.LOCAL, id="rec", name="rec")
    local = transcript_key(rec, _opts(engine=ENGINE_LOCAL))
    cloud = transcript_key(rec, _opts(engine=ENGINE_ELEVENLABS))
    assert local != cloud


def test_switching_scribe_model_invalidates_it_too():
    rec = Recording(source=Source.LOCAL, id="rec", name="rec")
    v1 = transcript_key(rec, _opts(elevenlabs_model="scribe_v1"))
    v2 = transcript_key(rec, _opts(elevenlabs_model="scribe_v2"))
    assert v1 != v2
