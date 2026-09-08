# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""MAI for the words, pyannote for the speakers.

Measured on a real 73-minute recording: MAI transcribes it in one request
without complaint, and MAI's own speaker detection fails on anything past
roughly half an hour - a relayed 408 that reads like a network timeout, then a
500, then a 503 wrapping a diarization 400. The pipeline is split at exactly
that seam: MAI stands in for Whisper as the decoder, and the local speaker
stage runs on top of its words as it does for Whisper.
"""

from __future__ import annotations

import pytest

from transcriber_studio import resume as resume_store
from transcriber_studio import stt_mai
from transcriber_studio.models import Recording, Segment, Source
from transcriber_studio.transcriber import (
    ENGINE_MAI,
    TranscribeOptions,
    Transcriber,
    mai_decodes_locally,
)

from .support import isolated_resume_dir


def _recording() -> Recording:
    return Recording(source=Source.PLAUD, id="f1", name="Team sync")


def _opts(**overrides) -> TranscribeOptions:
    base = dict(engine=ENGINE_MAI, mai_api_key="key", mai_region="eastus",
                mai_speakers="local", diarization_enabled=False)
    base.update(overrides)
    return TranscribeOptions(**base)


# ---- the routing decision --------------------------------------------
def test_local_speakers_is_the_default():
    assert mai_decodes_locally(_opts())
    assert TranscribeOptions().mai_speakers == "local"


def test_mai_diarization_is_still_available_when_asked_for():
    assert not mai_decodes_locally(_opts(mai_speakers="mai"))


def test_other_engines_are_untouched():
    assert not mai_decodes_locally(_opts(engine="local"))
    assert not mai_decodes_locally(_opts(engine="gemini"))


# ---- MAI as the decoder ----------------------------------------------
def _fake_decode(recording, audio_path, opts, progress_cb, log, should_cancel):
    return (
        [Segment(0.0, 2.0, "Hello there."), Segment(2.5, 4.0, "Hi.")],
        "en",
        [
            {"type": "word", "text": "Hello", "start": 0.0, "end": 0.9},
            {"type": "word", "text": "there.", "start": 1.0, "end": 2.0},
            {"type": "word", "text": "Hi.", "start": 2.5, "end": 4.0},
        ],
    )


def test_the_local_pipeline_takes_mai_words_without_loading_whisper(monkeypatch):
    """No Whisper model is loaded: the decoder is MAI, the rest is unchanged."""
    monkeypatch.setattr(stt_mai, "decode", _fake_decode)
    loaded = []
    monkeypatch.setattr(Transcriber, "_get_model", lambda *a, **k: loaded.append(a) or None)

    with isolated_resume_dir():
        result = Transcriber().transcribe(_recording(), "audio.mp3", _opts())

    assert loaded == [], "a Whisper model was loaded for a MAI decode"
    assert [s.text for s in result.segments] == ["Hello there.", "Hi."]
    assert result.language == "en"
    assert result.model == "azure-mai-transcribe-2"


def test_a_mai_decode_is_banked_and_restored_like_a_whisper_one(monkeypatch):
    """The expensive part is the upload; a crash during speaker detection must
    not throw it away, exactly as for Whisper."""
    calls = []

    def counting_decode(*args):
        calls.append(1)
        return _fake_decode(*args)

    monkeypatch.setattr(stt_mai, "decode", counting_decode)
    monkeypatch.setattr(Transcriber, "_get_model", lambda *a, **k: None)
    with isolated_resume_dir():
        opts = _opts(diarization_enabled=True)
        rec = _recording()
        log = resume_store.log_for(rec, None)
        Transcriber().transcribe(rec, "audio.mp3", opts, resume=log)
        Transcriber().transcribe(rec, "audio.mp3", opts, resume=log)
    assert len(calls) == 1, "the second run should have restored the decode"


def test_mai_diarization_setting_still_goes_to_the_cloud_path(monkeypatch):
    seen = []
    monkeypatch.setattr(
        Transcriber, "_transcribe_cloud",
        lambda self, *a, **k: seen.append("cloud") or None,
    )
    monkeypatch.setattr(stt_mai, "decode", lambda *a: pytest.fail("decode should not run"))
    Transcriber().transcribe(_recording(), "audio.mp3", _opts(mai_speakers="mai"))
    assert seen == ["cloud"]


# ---- the decoder itself ----------------------------------------------
def test_decode_returns_the_whisper_shaped_triple(monkeypatch, tmp_path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x" * 100)

    class _Response:
        status_code, ok = 200, True

        @staticmethod
        def json():
            return {"durationMilliseconds": 4000, "combinedPhrases": [], "phrases": [{
                "offsetMilliseconds": 0, "durationMilliseconds": 2000, "text": "Hello there.",
                "words": [{"text": "Hello", "offsetMilliseconds": 0, "durationMilliseconds": 900},
                          {"text": "there.", "offsetMilliseconds": 1000, "durationMilliseconds": 1000}],
                "locale": "en", "confidence": 0,
            }]}

    sent = {}

    def fake_post(url, headers=None, data=None, timeout=None):
        sent["body"] = data
        return _Response()

    monkeypatch.setattr(stt_mai.requests, "post", fake_post)
    segments, language, words = stt_mai.decode(_recording(), str(audio), _opts())
    assert [s.text for s in segments] == ["Hello there."]
    assert language == "en"
    assert [w["text"] for w in words if w["type"] == "word"] == ["Hello", "there."]


def test_decode_never_asks_mai_to_diarize(monkeypatch, tmp_path):
    """That is the whole point of the split."""
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x" * 100)
    captured = {}

    def fake_post(url, headers=None, data=None, timeout=None):
        captured["fields"] = data.encoder.fields

        class _R:
            status_code, ok = 200, True

            @staticmethod
            def json():
                return {"phrases": [], "combinedPhrases": []}
        return _R()

    monkeypatch.setattr(stt_mai.requests, "post", fake_post)
    stt_mai.decode(_recording(), str(audio), _opts(diarization_enabled=True))
    import json as _json

    definition = _json.loads(captured["fields"]["definition"][1])
    assert "diarization" not in definition


# ---- the resume key --------------------------------------------------
def test_a_clean_decode_is_not_restored_for_a_verbatim_run():
    rec = _recording()
    verbatim = resume_store.decode_key(rec, _opts(mai_style="verbatim"))
    clean = resume_store.decode_key(rec, _opts(mai_style="clean"))
    assert verbatim != clean


def test_a_mai_decode_is_not_confused_with_a_whisper_one():
    rec = _recording()
    assert resume_store.decode_key(rec, _opts()) != resume_store.decode_key(
        rec, TranscribeOptions(engine="local")
    )


# ---- no more re-uploading a whole file for an answer that will not change ----
@pytest.mark.parametrize("body", [
    'MAI service returned an error: ServiceUnavailable - {"error":{"code":"diarization_unavailable",'
    '"message":"Speaker diarization service is unavailable: Diarization service returned error code 400"}}',
    'MAI service returned an error: RequestTimeout - { "error": { "code": "Timeout", '
    '"message": "The operation was timeout." } }',
])
def test_deterministic_backend_errors_are_not_retried(monkeypatch, tmp_path, body):
    """Probe A re-uploaded 17.5 MB three times for three differently-worded
    versions of the same refusal. Once is enough."""
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x" * 100)
    attempts = []

    class _R:
        status_code, ok = 503, False
        text = body

        @staticmethod
        def json():
            raise ValueError("prose, not json")

    monkeypatch.setattr(stt_mai.requests, "post",
                        lambda *a, **k: attempts.append(1) or _R())
    monkeypatch.setattr(stt_mai.time, "sleep", lambda *_: None)
    with pytest.raises(stt_mai.MaiError):
        stt_mai._post(str(audio), "key", "eastus", {"enhancedMode": {}}, None, None)
    assert len(attempts) == 1


def test_a_genuinely_transient_error_is_still_retried(monkeypatch, tmp_path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x" * 100)
    attempts = []

    class _R:
        status_code, ok = 503, False
        text = "Service temporarily unavailable"

        @staticmethod
        def json():
            raise ValueError("no json")

    monkeypatch.setattr(stt_mai.requests, "post",
                        lambda *a, **k: attempts.append(1) or _R())
    monkeypatch.setattr(stt_mai.time, "sleep", lambda *_: None)
    with pytest.raises(stt_mai.MaiError):
        stt_mai._post(str(audio), "key", "eastus", {"enhancedMode": {}}, None, None)
    assert len(attempts) == stt_mai.MAX_ATTEMPTS


# ---- faults the first live run found --------------------------------------
def test_spacing_entries_do_not_crash_speaker_assignment():
    """MAI interleaves spacing entries between words; they carry no timing.
    Live, this raised KeyError('start') and every speaker was silently lost."""
    from transcriber_studio.diarization import DiarizationResult, SpeakerTurn

    words = [
        {"type": "word", "text": "Hello", "start": 0.0, "end": 0.5},
        {"type": "spacing", "text": " "},
        {"type": "word", "text": "there", "start": 0.6, "end": 1.0},
    ]
    diarized = DiarizationResult(turns=[SpeakerTurn(0.0, 2.0, "SPEAKER_00")])
    segments, speakers = Transcriber.__new__(Transcriber)._apply_speakers(
        [Segment(0.0, 1.0, "Hello there")], words, diarized, lambda _m: None, {}
    )
    assert speakers == ["Speaker 1"]
    assert segments[0].text == "Hello there"


def test_sparse_words_do_not_rebuild_the_transcript():
    """Clean style: the words covered a fraction of the phrase text, and a
    transcript rebuilt from them read 'First' then 'time prepping…', with
    everything between gone."""
    phrases = [{
        "offsetMilliseconds": 0, "durationMilliseconds": 5000,
        "text": "First book for these things. I spent a lot of time prepping.",
        "words": [{"text": "First", "offsetMilliseconds": 0, "durationMilliseconds": 300}],
        "locale": "en", "confidence": 0,
    }]
    assert not stt_mai.words_cover_text(stt_mai._words_from(phrases), phrases)
    segments, _ = stt_mai._segments_from_phrases(phrases)
    assert segments[0].text.startswith("First book for these things")


def test_full_word_coverage_is_still_trusted():
    phrases = [{
        "offsetMilliseconds": 0, "durationMilliseconds": 1000, "text": "Hello there.",
        "words": [{"text": "Hello", "offsetMilliseconds": 0, "durationMilliseconds": 400},
                  {"text": "there.", "offsetMilliseconds": 500, "durationMilliseconds": 400}],
    }]
    assert stt_mai.words_cover_text(stt_mai._words_from(phrases), phrases)
