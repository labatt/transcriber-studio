# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""What a banked transcript is keyed by, and how it is found again.

Live, 2026-09-08: a Scribe job was quit during AI cleanup. Resume re-uploaded
the whole hour to ElevenLabs. The transcript had been banked, but under a key
that included the vocabulary hint text, and the job's own glossary stage had
added 46 terms to the shared glossary before the quit. New hints, new key, no
match. Scribe is never sent the hints at all.
"""

from __future__ import annotations

import json

from transcriber_studio import resume as resume_store
from transcriber_studio.config import Settings
from transcriber_studio.jobs import JobRunner
from transcriber_studio.models import Recording, Segment, Source, TranscriptResult
from transcriber_studio.transcriber import TranscribeOptions, expected_model_label

from .support import isolated_glossary_dir, isolated_resume_dir


def _rec() -> Recording:
    return Recording(source=Source.LOCAL, id="r1", name="Call", local_path="a.mp3")


def _scribe(**kw) -> TranscribeOptions:
    base = dict(engine="elevenlabs", elevenlabs_model="scribe_v2", diarization_enabled=True)
    base.update(kw)
    return TranscribeOptions(**base)


def _local(**kw) -> TranscribeOptions:
    base = dict(engine="local", model="large-v3", hotwords="Tyrus, Rhett")
    base.update(kw)
    return TranscribeOptions(**base)


# ---- the hint text is not part of the identity ----------------------------------
def test_a_glossary_that_grew_during_the_job_does_not_change_the_keys():
    rec = _rec()
    before, after = _local(hotwords="Tyrus, Rhett"), _local(hotwords="Tyrus, Rhett, Anne Smarty")
    assert resume_store.transcript_key(rec, before) == resume_store.transcript_key(rec, after)
    assert resume_store.decode_key(rec, before) == resume_store.decode_key(rec, after)


def test_turning_hints_off_entirely_does_change_a_whisper_key():
    rec = _rec()
    assert resume_store.decode_key(rec, _local(hotwords="Tyrus")) != \
        resume_store.decode_key(rec, _local(hotwords=""))


# ---- an engine is keyed by what it reads --------------------------------------------
def test_scribe_ignores_the_whisper_only_settings():
    rec = _rec()
    base = _scribe()
    for changed in (
        _scribe(vad_enabled=False),
        _scribe(hallucination_guard=False),
        _scribe(model="medium"),
        _scribe(min_speakers=3),
        _scribe(hotwords="Tyrus"),
    ):
        assert resume_store.transcript_key(rec, base) == resume_store.transcript_key(rec, changed)


def test_scribe_honours_what_it_is_actually_sent():
    rec = _rec()
    base = _scribe()
    assert resume_store.transcript_key(rec, base) != resume_store.transcript_key(rec, _scribe(max_speakers=2))
    assert resume_store.transcript_key(rec, base) != resume_store.transcript_key(rec, _scribe(elevenlabs_model="scribe_v1"))
    assert resume_store.transcript_key(rec, base) != resume_store.transcript_key(rec, _scribe(diarization_enabled=False))
    assert resume_store.transcript_key(rec, base) != resume_store.transcript_key(rec, _scribe(denoise="deep_filter"))


def test_gemini_ignores_speaker_bounds_it_cannot_take():
    rec = _rec()
    a = TranscribeOptions(engine="gemini", gemini_model="gemini-3.5-transcribe", max_speakers=0)
    b = TranscribeOptions(engine="gemini", gemini_model="gemini-3.5-transcribe", max_speakers=4)
    assert resume_store.transcript_key(rec, a) == resume_store.transcript_key(rec, b)
    c = TranscribeOptions(engine="gemini", gemini_model="gemini-3.5-transcribe", gemini_mode="smart")
    assert resume_store.transcript_key(rec, a) != resume_store.transcript_key(rec, c)


def test_whisper_still_notices_the_settings_that_change_its_words():
    rec = _rec()
    base = _local()
    for changed in (_local(model="medium"), _local(vad_enabled=False), _local(hallucination_guard=False),
                    _local(denoise="deep_filter"), _local(language="de")):
        assert resume_store.decode_key(rec, base) != resume_store.decode_key(rec, changed)


def test_the_engines_do_not_collide():
    rec = _rec()
    keys = {resume_store.transcript_key(rec, o) for o in (
        _local(), _scribe(), TranscribeOptions(engine="gemini"), TranscribeOptions(engine="mai"),
    )}
    assert len(keys) == 4


# ---- finding a transcript banked under an older key ------------------------------------
def test_expected_labels_match_what_the_engines_write():
    assert expected_model_label(_scribe()) == "elevenlabs-scribe_v2"
    assert expected_model_label(_local()) == "large-v3"
    assert expected_model_label(TranscribeOptions(engine="mai")).startswith("azure-mai")


def _runner(settings: Settings) -> JobRunner:
    runner = JobRunner.__new__(JobRunner)
    runner.s = settings
    runner.client = None

    class _NeverTranscribe:
        def transcribe(self, *a, **k):
            raise AssertionError("re-transcribed instead of restoring")

    runner.transcriber = _NeverTranscribe()
    return runner


def test_a_transcript_banked_under_an_old_key_is_restored_by_engine_and_model():
    rec = _rec()
    banked = TranscriptResult(
        recording=rec, model="elevenlabs-scribe_v2",
        segments=[Segment(0, 1, "Hello", speaker="Speaker 1")], speakers=["Speaker 1"],
    )
    with isolated_resume_dir(), isolated_glossary_dir():
        log = resume_store.log_for(rec, None)
        log.record("an-old-formula-key", json.dumps(resume_store.transcript_to_dict(banked)),
                   stage=resume_store.TRANSCRIPT_STAGE, segments=1)
        lines = []
        runner = _runner(Settings(stt_engine="elevenlabs", elevenlabs_model="scribe_v2"))
        out = runner._transcribe_or_restore(rec, 1, None, lines.append, None, log)

    assert [s.text for s in out.segments] == ["Hello"]
    assert any("matched by engine and model" in line for line in lines)


def test_a_transcript_from_a_different_engine_is_not_mistaken_for_ours(monkeypatch):
    """A Whisper transcript in the bank must not be handed to a Scribe run."""
    rec = _rec()
    banked = TranscriptResult(recording=rec, model="large-v3",
                              segments=[Segment(0, 1, "Hello")], speakers=[])
    fresh = TranscriptResult(recording=rec, model="elevenlabs-scribe_v2",
                             segments=[Segment(0, 1, "Fresh")], speakers=[])
    with isolated_resume_dir(), isolated_glossary_dir():
        log = resume_store.log_for(rec, None)
        log.record("old", json.dumps(resume_store.transcript_to_dict(banked)),
                   stage=resume_store.TRANSCRIPT_STAGE, segments=1)
        runner = _runner(Settings(stt_engine="elevenlabs", elevenlabs_model="scribe_v2"))
        monkeypatch.setattr(runner, "transcribe_only", lambda *a, **k: fresh)
        out = runner._transcribe_or_restore(rec, 1, None, None, None, log)
        # And the fresh one is banked with its engine and model on the entry.
        entry = [json.loads(line) for line in resume_store.resume_path(rec).read_text(encoding="utf-8").splitlines()][-1]

    assert [s.text for s in out.segments] == ["Fresh"]
    assert entry["engine"] == "elevenlabs" and entry["model"] == "elevenlabs-scribe_v2"


def test_latest_prefers_the_newest_entry_the_check_accepts(tmp_path):
    log = resume_store.ResumeLog(tmp_path / "r.jsonl")
    log.record("a", '{"model": "x"}', stage="transcript")
    log.record("b", '{"model": "y"}', stage="transcript")
    log.record("c", '{"model": "x"}', stage="cleanup")
    assert log.latest("transcript", lambda raw: json.loads(raw)["model"] == "x") == '{"model": "x"}'
    assert log.latest("transcript", lambda raw: json.loads(raw)["model"] == "y") == '{"model": "y"}'
    assert log.latest("transcript", lambda raw: json.loads(raw)["model"] == "z") is None
