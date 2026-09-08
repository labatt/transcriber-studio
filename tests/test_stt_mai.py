# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""MAI-Transcribe through Azure Speech's enhanced mode.

The request shape is the part worth pinning down: ``enhancedMode.enabled`` is
what routes the call to a MAI model at all, and without it Azure serves the
request with its older recogniser and returns something plausible but
different. A test that only checked "a transcript came back" would not notice.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import pytest

from transcriber_studio import stt_mai
from transcriber_studio.models import Recording, Source


@dataclass
class _Opts:
    """Just the fields the engine reads."""

    mai_api_key: str = "key"
    mai_region: str = "eastus"
    mai_model: str = "MAI-Transcribe-2"
    mai_style: str = "verbatim"
    mai_send_phrases: bool = True
    diarization_enabled: bool = True
    language: str = "auto"
    hotwords: str = ""


def _recording() -> Recording:
    return Recording(source=Source.PLAUD, id="f1", name="Team sync")


# ---- the definition that selects MAI ---------------------------------
def test_enhanced_mode_is_what_selects_the_model():
    """Without this Azure quietly answers with a different recogniser."""
    definition = stt_mai.build_definition(_Opts())
    assert definition["enhancedMode"]["enabled"] is True
    assert definition["enhancedMode"]["model"] == "MAI-Transcribe-2"


def test_word_timestamps_are_always_asked_for():
    """They cost nothing and the per-word speaker assignment is built on them."""
    definition = stt_mai.build_definition(_Opts())
    assert definition["enhancedMode"]["modelOptions"]["timestamps"] == "word"


def test_verbatim_is_the_default_style():
    """Tidying belongs in AI Cleanup, where it can be reviewed."""
    assert stt_mai.build_definition(
        _Opts()
    )["enhancedMode"]["modelOptions"]["transcribeStyle"] == "verbatim"


def test_clean_style_is_passed_through():
    assert stt_mai.build_definition(
        _Opts(mai_style="clean")
    )["enhancedMode"]["modelOptions"]["transcribeStyle"] == "clean"


def test_an_unknown_style_falls_back_to_verbatim():
    assert stt_mai.build_definition(
        _Opts(mai_style="whatever")
    )["enhancedMode"]["modelOptions"]["transcribeStyle"] == "verbatim"


def test_diarization_is_only_sent_when_wanted():
    assert "diarization" in stt_mai.build_definition(_Opts())
    assert "diarization" not in stt_mai.build_definition(_Opts(diarization_enabled=False))


def test_the_glossary_becomes_the_phrase_list():
    """The same terms the glossary builds for Whisper's hotwords."""
    definition = stt_mai.build_definition(_Opts(hotwords="Plaud, Suredone, Labatt-Simon"))
    assert definition["phraseList"]["phrases"] == ["Plaud", "Suredone", "Labatt-Simon"]


def test_phrase_hints_can_be_turned_off():
    definition = stt_mai.build_definition(
        _Opts(hotwords="Plaud, Suredone", mai_send_phrases=False)
    )
    assert "phraseList" not in definition


def test_duplicate_terms_are_sent_once():
    definition = stt_mai.build_definition(_Opts(hotwords="Plaud, Plaud, Suredone"))
    assert definition["phraseList"]["phrases"] == ["Plaud", "Suredone"]


def test_the_phrase_list_is_capped():
    """An over-long list of hints dilutes every entry in it."""
    terms = ", ".join(f"term{i}" for i in range(stt_mai.MAX_PHRASES + 50))
    definition = stt_mai.build_definition(_Opts(hotwords=terms))
    assert len(definition["phraseList"]["phrases"]) == stt_mai.MAX_PHRASES


def test_auto_language_sends_no_locale():
    """Documented as a very strong hint, so it is only sent when chosen."""
    assert "locales" not in stt_mai.build_definition(_Opts(language="auto"))


def test_a_chosen_language_is_sent():
    assert stt_mai.build_definition(_Opts(language="en"))["locales"] == ["en"]


# ---- the endpoint ----------------------------------------------------
def test_the_endpoint_carries_the_region_and_a_pinned_api_version():
    url = stt_mai.endpoint("westus2")
    assert url.startswith("https://westus2.api.cognitive.microsoft.com/")
    assert "speechtotext/transcriptions:transcribe" in url
    assert f"api-version={stt_mai.API_VERSION}" in url


def test_only_the_six_supported_regions_are_offered():
    """A resource anywhere else authenticates and then has no model."""
    assert set(stt_mai.REGIONS) == {
        "eastus", "westus", "westus2", "northeurope", "southeastasia", "centralindia",
    }


# ---- the response ----------------------------------------------------
class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 300

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _azure_answer() -> dict:
    """The documented shape: phrases, each with words and a confidence."""
    return {
        "durationMilliseconds": 6000,
        "combinedPhrases": [{"text": "Good afternoon. Hello there."}],
        "phrases": [
            {
                "speaker": 1,
                "offsetMilliseconds": 960,
                "durationMilliseconds": 640,
                "text": "Good afternoon.",
                "words": [
                    {"text": "Good", "offsetMilliseconds": 960, "durationMilliseconds": 240},
                    {"text": "afternoon.", "offsetMilliseconds": 1200, "durationMilliseconds": 400},
                ],
                "locale": "en-US",
                "confidence": 0.93554276,
            },
            {
                "speaker": 2,
                "offsetMilliseconds": 4000,
                "durationMilliseconds": 900,
                "text": "Hello there.",
                "words": [
                    {"text": "Hello", "offsetMilliseconds": 4000, "durationMilliseconds": 400},
                    {"text": "there.", "offsetMilliseconds": 4400, "durationMilliseconds": 500},
                ],
                "locale": "en-US",
                "confidence": 0.81,
            },
        ],
    }


@pytest.fixture
def audio(tmp_path):
    """A real file on disk. The engine measures it before uploading, and
    faking Path.stat breaks pytest's own traceback machinery."""
    path = tmp_path / "audio.mp3"
    path.write_bytes(b"not really an mp3, but it has a size")
    return str(path)


def _fields(data):
    """The multipart fields, from the streaming encoder the engine now sends.

    The upload moved from requests' in-memory ``files=`` to a streaming
    ``MultipartEncoderMonitor`` so progress could be reported; the fields it
    carries are the same, one level down.
    """
    encoder = getattr(data, "encoder", data)
    return getattr(encoder, "fields", {}) or {}


def _transcribe(monkeypatch, payload, audio_path, opts=None, seen=None):
    def fake_post(url, headers=None, data=None, files=None, timeout=None):
        if seen is not None:
            seen.update(url=url, headers=headers, files=_fields(data) if data is not None else files)
        return _Response(payload)

    monkeypatch.setattr(stt_mai.requests, "post", fake_post)
    return stt_mai.transcribe(_recording(), audio_path, opts or _Opts())


def test_a_transcript_comes_back_with_speakers(monkeypatch, audio):
    result = _transcribe(monkeypatch, _azure_answer(), audio)
    assert [s.text for s in result.segments] == ["Good afternoon.", "Hello there."]
    assert result.speakers == ["Speaker 1", "Speaker 2"]
    assert result.language == "en"
    assert result.model == "azure-mai-transcribe-2"


def test_timings_are_converted_from_milliseconds(monkeypatch, audio):
    result = _transcribe(monkeypatch, _azure_answer(), audio)
    assert result.segments[0].start == pytest.approx(0.96)
    assert result.segments[0].end == pytest.approx(1.6)


def test_the_confidence_is_kept(monkeypatch, audio):
    """Azure reports 0..1; a Segment stores the log, like every other engine."""
    result = _transcribe(monkeypatch, _azure_answer(), audio)
    assert result.segments[0].confidence == pytest.approx(0.93554276, abs=1e-4)
    assert result.segments[0].avg_logprob == pytest.approx(math.log(0.93554276))


def test_words_are_spaced_apart(monkeypatch, audio):
    """Azure sends no spacing of its own; without it the words run together."""
    result = _transcribe(monkeypatch, _azure_answer(), audio)
    assert result.segments[0].text == "Good afternoon."


def test_the_request_is_multipart_with_audio_and_definition(monkeypatch, audio):
    seen: dict = {}
    _transcribe(monkeypatch, _azure_answer(), audio, seen=seen)
    assert set(seen["files"]) == {"audio", "definition"}
    definition = json.loads(seen["files"]["definition"][1])
    assert definition["enhancedMode"]["model"] == "MAI-Transcribe-2"
    assert seen["headers"]["Ocp-Apim-Subscription-Key"] == "key"
    assert seen["url"].startswith("https://eastus.")


def test_a_response_without_words_still_yields_segments(monkeypatch, audio):
    """timestamps can come back empty; the phrases are still a transcript."""
    payload = _azure_answer()
    for phrase in payload["phrases"]:
        phrase.pop("words")
    result = _transcribe(monkeypatch, payload, audio)
    assert [s.text for s in result.segments] == ["Good afternoon.", "Hello there."]
    assert result.speakers == ["Speaker 1", "Speaker 2"]
    assert result.segments[1].confidence == pytest.approx(0.81)


def test_only_a_combined_phrase_is_better_than_nothing(monkeypatch, audio):
    payload = {"durationMilliseconds": 5000, "combinedPhrases": [{"text": "Just this."}],
               "phrases": []}
    result = _transcribe(monkeypatch, payload, audio)
    assert [s.text for s in result.segments] == ["Just this."]
    assert result.segments[0].end == pytest.approx(5.0)


def test_diarization_off_leaves_speakers_unset(monkeypatch, audio):
    payload = _azure_answer()
    for phrase in payload["phrases"]:
        phrase.pop("speaker")
    result = _transcribe(monkeypatch, payload, audio, opts=_Opts(diarization_enabled=False))
    assert result.speakers == []


# ---- refusals --------------------------------------------------------
def test_a_missing_key_is_refused_before_the_network(monkeypatch):
    monkeypatch.setattr(
        stt_mai.requests, "post",
        lambda *a, **k: pytest.fail("should not have made a request"),
    )
    with pytest.raises(stt_mai.MaiError, match="No Azure Speech key"):
        stt_mai.transcribe(_recording(), "audio.mp3", _Opts(mai_api_key=""))


def test_an_unsupported_region_is_refused_with_the_list(monkeypatch):
    monkeypatch.setattr(
        stt_mai.requests, "post",
        lambda *a, **k: pytest.fail("should not have made a request"),
    )
    with pytest.raises(stt_mai.MaiError) as excinfo:
        stt_mai.transcribe(_recording(), "audio.mp3", _Opts(mai_region="uksouth"))
    assert "uksouth" in str(excinfo.value)
    assert "eastus" in str(excinfo.value)


def test_an_oversized_file_is_refused_before_upload(monkeypatch, audio):
    monkeypatch.setattr(stt_mai, "MAX_UPLOAD_BYTES", 4)      # the file is bigger
    monkeypatch.setattr(
        stt_mai.requests, "post",
        lambda *a, **k: pytest.fail("should not have made a request"),
    )
    with pytest.raises(stt_mai.MaiError, match="over the"):
        stt_mai.transcribe(_recording(), audio, _Opts())


def test_a_rejected_key_says_so(monkeypatch, audio):
    monkeypatch.setattr(
        stt_mai.requests, "post",
        lambda *a, **k: _Response({"error": {"message": "Access denied"}}, 401),
    )
    with pytest.raises(stt_mai.MaiError, match="Access denied"):
        stt_mai.transcribe(_recording(), audio, _Opts())


def test_test_key_rejects_a_wrong_region_without_calling_azure(monkeypatch):
    monkeypatch.setattr(
        stt_mai.requests, "post",
        lambda *a, **k: pytest.fail("should not have made a request"),
    )
    with pytest.raises(stt_mai.MaiError, match="not available"):
        stt_mai.test_key("key", "uksouth")


def test_test_key_reports_a_bad_key(monkeypatch):
    monkeypatch.setattr(stt_mai.requests, "post", lambda *a, **k: _Response({}, 401))
    with pytest.raises(stt_mai.MaiError, match="rejected that key"):
        stt_mai.test_key("key", "eastus")


def test_test_key_accepts_a_working_one(monkeypatch):
    """A key that works answers 400 for the missing audio, not 401."""
    monkeypatch.setattr(stt_mai.requests, "post", lambda *a, **k: _Response({}, 400))
    assert "eastus" in stt_mai.test_key("key", "eastus")


# ---- the engine is registered ----------------------------------------
def test_the_transcriber_routes_to_this_module():
    from transcriber_studio.transcriber import (
        CLOUD_ENGINES,
        ENGINE_MAI,
        Transcriber,
    )

    assert ENGINE_MAI in CLOUD_ENGINES
    assert Transcriber.__new__(Transcriber)._cloud_engine(ENGINE_MAI) is stt_mai


def test_the_settings_reach_the_engine():
    from transcriber_studio.config import Settings
    from transcriber_studio.jobs import JobRunner

    settings = Settings()
    settings.mai_api_key = "abc"
    settings.mai_region = "westus2"
    settings.mai_style = "clean"
    opts = JobRunner(settings, client=object())._opts()
    assert opts.mai_api_key == "abc"
    assert opts.mai_region == "westus2"
    assert opts.mai_style == "clean"


# ---- what the live service actually sends ----------------------------
def test_a_zero_confidence_means_unknown_not_certainly_wrong(monkeypatch, audio):
    """Measured against the live API: MAI fills every phrase with
    ``"confidence": 0``. Storing that as a real score would mark the whole
    transcript as maximally doubtful."""
    payload = _azure_answer()
    for phrase in payload["phrases"]:
        phrase["confidence"] = 0
    result = _transcribe(monkeypatch, payload, audio)
    assert all(s.confidence is None for s in result.segments)
    assert [s.text for s in result.segments] == ["Good afternoon.", "Hello there."]


def test_speakers_are_numbered_from_a_zero_based_index(monkeypatch, audio):
    """The live service numbers speakers from 0; the transcript counts from 1."""
    payload = _azure_answer()
    payload["phrases"][0]["speaker"] = 0
    payload["phrases"][1]["speaker"] = 1
    result = _transcribe(monkeypatch, payload, audio)
    assert result.speakers == ["Speaker 1", "Speaker 2"]


def test_a_bare_locale_is_understood(monkeypatch, audio):
    """The docs show "en-US"; the live service sends "en"."""
    payload = _azure_answer()
    for phrase in payload["phrases"]:
        phrase["locale"] = "en"
    assert _transcribe(monkeypatch, payload, audio).language == "en"


def test_the_hint_limit_is_fifty():
    """The service's own limit, found by exceeding it — the docs state none.
    Sixty-four glossary terms made every run fail with a 400."""
    assert stt_mai.MAX_PHRASES == 50


def test_overflowing_hints_are_counted_for_the_log():
    terms = ", ".join(f"term{i}" for i in range(64))
    assert stt_mai.phrase_overflow(_Opts(hotwords=terms)) == 14
    assert stt_mai.phrase_overflow(_Opts(hotwords="one, two")) == 0


def test_an_unparseable_error_body_is_still_reported():
    """Azure wraps a JSON error in prose. Reporting only "HTTP 400" threw away
    the sentence that explained the failure."""
    class _Raw:
        status_code = 400
        text = ('MAI service returned an error: BadRequest - {"error":{"code":'
                '"invalid_request","message":"Context list cannot have more than 50 items."}}')

        @staticmethod
        def json():
            raise ValueError("not json")

    message = stt_mai._error_message(_Raw())
    assert "Context list cannot have more than 50 items" in message
    assert "400" in message
