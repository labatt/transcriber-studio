# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Scribe for the words, pyannote for the speakers.

Scribe's own speaker labels carry no voice data, so a voice enrolled once can
never be matched against them. Running the local speaker stage on top of
Scribe's words — exactly as MAI already does — is what lets an ElevenLabs
transcript come back with people's names on it.
"""

from __future__ import annotations

from transcriber_studio import resume as resume_store
from transcriber_studio import stt_elevenlabs
from transcriber_studio.models import Recording, Segment, Source
from transcriber_studio.transcriber import (
    ENGINE_ELEVENLABS,
    TranscribeOptions,
    Transcriber,
    decodes_locally,
    elevenlabs_decodes_locally,
    with_known_speakers,
)

from .support import isolated_resume_dir


def _recording() -> Recording:
    return Recording(source=Source.PLAUD, id="f1", name="Team sync")


def _opts(**overrides) -> TranscribeOptions:
    base = dict(engine=ENGINE_ELEVENLABS, elevenlabs_api_key="key", hf_token="hf_x",
                elevenlabs_speakers="local", diarization_enabled=True)
    base.update(overrides)
    return TranscribeOptions(**base)


# ---- the routing decision --------------------------------------------
def test_local_speakers_is_the_default_when_pyannote_can_run():
    assert TranscribeOptions().elevenlabs_speakers == "local"
    assert elevenlabs_decodes_locally(_opts())
    assert decodes_locally(_opts())


def test_without_a_huggingface_token_scribe_keeps_its_own_speakers():
    """Nobody labelling speakers is worse than Scribe doing it."""
    assert not elevenlabs_decodes_locally(_opts(hf_token=""))


def test_with_speakers_off_there_is_nothing_to_detect_locally():
    assert not elevenlabs_decodes_locally(_opts(diarization_enabled=False))


def test_scribes_own_diarization_is_still_available_when_asked_for():
    assert not elevenlabs_decodes_locally(_opts(elevenlabs_speakers="scribe"))


def test_other_engines_are_untouched():
    assert not elevenlabs_decodes_locally(_opts(engine="local"))
    assert not elevenlabs_decodes_locally(_opts(engine="mai"))


# ---- Scribe as the decoder -------------------------------------------
def test_decode_asks_scribe_for_words_without_speakers(monkeypatch):
    sent = {}

    def fake_body(recording, audio_path, opts, progress_cb, log_cb, should_cancel):
        sent["diarize"] = opts.diarization_enabled
        from transcriber_studio.models import TranscriptResult
        words = [{"type": "word", "text": "Hi", "start": 0.0, "end": 0.4}]
        return TranscriptResult(recording=recording, segments=[Segment(0.0, 0.4, "Hi")],
                                language="en", model="scribe_v2", speakers=[]), words

    monkeypatch.setattr(stt_elevenlabs, "_transcribe_with_words", fake_body)
    segments, language, words = stt_elevenlabs.decode(_recording(), "a.mp3", _opts())
    assert sent["diarize"] is False
    assert [s.text for s in segments] == ["Hi"] and language == "en"
    assert words and words[0]["text"] == "Hi"


def test_the_transcriber_routes_scribe_through_the_local_speaker_stage(monkeypatch):
    seen = {}

    def fake_single(self, recording, audio_path, model, language, opts, progress_cb, log,
                    should_cancel=None, resume=None, decoder=None):
        seen["decoder"] = decoder
        from transcriber_studio.models import TranscriptResult
        return TranscriptResult(recording=recording, model="scribe_v2")

    monkeypatch.setattr(Transcriber, "_transcribe_single", fake_single)
    monkeypatch.setattr(Transcriber, "_transcribe_cloud",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("cloud path taken")))
    with isolated_resume_dir():
        Transcriber().transcribe(_recording(), "a.mp3", _opts(), log_cb=lambda m: None)
    assert seen["decoder"] is stt_elevenlabs.decode


def test_without_a_token_the_cloud_path_is_taken_and_says_why(monkeypatch):
    logged = []
    from transcriber_studio.models import TranscriptResult

    monkeypatch.setattr(
        Transcriber, "_transcribe_cloud",
        lambda self, recording, *a, **k: TranscriptResult(recording=recording),
    )
    Transcriber().transcribe(_recording(), "a.mp3", _opts(hf_token=""), log_cb=logged.append)
    assert any("No HuggingFace token" in line for line in logged)


# ---- the resume key ---------------------------------------------------
def test_who_draws_the_speakers_is_part_of_the_transcript_identity():
    rec = _recording()
    local = resume_store.transcript_key(rec, _opts())
    scribe = resume_store.transcript_key(rec, _opts(elevenlabs_speakers="scribe"))
    assert local != scribe


def test_local_speakers_make_both_bounds_matter():
    rec = _recording()
    assert resume_store.transcript_key(rec, _opts(min_speakers=2)) != \
        resume_store.transcript_key(rec, _opts(min_speakers=3))


def test_the_decode_key_ignores_who_draws_the_speakers():
    rec = _recording()
    assert resume_store.decode_key(rec, _opts()) == \
        resume_store.decode_key(rec, _opts(elevenlabs_speakers="scribe"))


# ---- names heard beat names measured ---------------------------------
def test_known_speakers_override_recognised_ones():
    logged = []
    merged = with_known_speakers(
        {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"},
        {"SPEAKER_01": "Robert", "SPEAKER_02": "Carol", "SPEAKER_03": "  "},
        logged.append,
    )
    assert merged == {"SPEAKER_00": "Alice", "SPEAKER_01": "Robert", "SPEAKER_02": "Carol"}
    assert len(logged) == 2


def test_known_speakers_may_be_absent():
    assert with_known_speakers({"SPEAKER_00": "Alice"}, None) == {"SPEAKER_00": "Alice"}
