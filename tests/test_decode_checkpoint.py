# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""The Whisper pass must survive a crash during speaker detection.

Diarizing an hour of audio takes minutes. Before this, dying inside it threw
away the decode that came before it too — the far more expensive half.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tests.support import isolated_resume_dir
from transcriber_studio import diarization
from transcriber_studio import resume as resume_store
from transcriber_studio.job_cancel import JobCancelled
from transcriber_studio.models import Recording, Segment, Source
from transcriber_studio.transcriber import TranscribeOptions, Transcriber


def _recording():
    return Recording(source=Source.LOCAL, id="rec-1", name="talk", date="2026-08-27",
                     local_path="talk.mp3", duration_seconds=60)


class _FakeModel:
    """Yields two segments, and complains loudly if asked to decode twice."""

    def __init__(self):
        self.calls = 0

    def transcribe(self, path, **kwargs):
        self.calls += 1
        segments = [
            SimpleNamespace(start=0.0, end=2.0, text=" Hello there. "),
            SimpleNamespace(start=2.0, end=4.0, text=" General Kenobi. "),
        ]
        return iter(segments), SimpleNamespace(language="en", duration=60.0,
                                               duration_after_vad=60.0)


def _decode(transcriber, bank, model, opts, log=None):
    return transcriber._transcribe_single(
        _recording(), "talk.wav", model, None, opts, None, log or (lambda m: None),
        None, bank,
    )


@pytest.fixture
def opts():
    return TranscribeOptions(model="large-v3", language="auto", diarization_enabled=True)


def test_the_decode_is_banked_before_diarization_starts(opts, monkeypatch):
    """Recorded first, so a crash inside pyannote cannot take it down."""
    monkeypatch.setattr(diarization, "is_available", lambda: True)

    def _explode(*a, **k):
        raise RuntimeError("pyannote fell over")

    monkeypatch.setattr(diarization, "Diarizer", _explode)

    with isolated_resume_dir():
        bank = resume_store.ResumeLog(resume_store.resume_path(_recording()))
        model = _FakeModel()
        result = _decode(Transcriber(), bank, model, opts)

        assert model.calls == 1
        assert result.segments[0].text == "Hello there."
        # The decode is on disk even though diarization never finished.
        saved = bank.get(resume_store.decode_key(_recording(), opts))
        assert saved is not None
        assert len(json.loads(saved)["segments"]) == 2


def test_a_rerun_after_that_crash_does_not_decode_again(opts, monkeypatch):
    monkeypatch.setattr(diarization, "is_available", lambda: False)

    with isolated_resume_dir():
        path = resume_store.resume_path(_recording())
        bank = resume_store.ResumeLog(path)
        bank.record(
            resume_store.decode_key(_recording(), opts),
            json.dumps(resume_store.decode_to_dict(
                [Segment(start=0.0, end=2.0, text="Hello there.")], "en")),
            stage=resume_store.DECODE_STAGE, segments=1,
        )

        model = _FakeModel()
        logs: list[str] = []
        result = _decode(Transcriber(), resume_store.ResumeLog(path).load(),
                         model, opts, logs.append)

        assert model.calls == 0, "the whole point is not to decode a second time"
        assert [s.text for s in result.segments] == ["Hello there."]
        assert any("Restored 1 transcribed segment" in line for line in logs)


def test_changing_the_speaker_bounds_still_reuses_the_decode(opts):
    """Speakers are attached after the fact, so the words are unaffected."""
    a = resume_store.decode_key(_recording(), opts)
    b = resume_store.decode_key(
        _recording(),
        TranscribeOptions(model="large-v3", language="auto",
                          diarization_enabled=False, min_speakers=3),
    )
    assert a == b


def test_changing_what_the_decoder_hears_does_not_reuse_it(opts):
    changed = TranscribeOptions(model="large-v3", language="auto", hotwords="Kenobi")
    assert resume_store.decode_key(_recording(), opts) != \
        resume_store.decode_key(_recording(), changed)


def test_the_ui_offers_a_resume_when_only_the_decode_is_banked():
    with isolated_resume_dir():
        rec = _recording()
        opts_ = TranscribeOptions()
        bank = resume_store.ResumeLog(resume_store.resume_path(rec))
        bank.record(resume_store.decode_key(rec, opts_), json.dumps({"segments": [], "language": "en"}),
                    stage=resume_store.DECODE_STAGE, segments=2)
        assert resume_store.describe_progress(rec) == "transcribed audio (speakers still to do)"


def test_cancelling_at_the_diarization_boundary_still_banks_the_decode(opts, monkeypatch):
    """Re-running Whisper costs GPU time however the last run ended."""
    monkeypatch.setattr(diarization, "is_available", lambda: True)

    # The decode loop checks once per segment; the third check is the one
    # guarding diarization, which is the boundary this is about.
    checks = [0]

    def should_cancel():
        checks[0] += 1
        return checks[0] > 2

    with isolated_resume_dir():
        bank = resume_store.ResumeLog(resume_store.resume_path(_recording()))
        with pytest.raises(JobCancelled):
            Transcriber()._transcribe_single(
                _recording(), "talk.wav", _FakeModel(), None, opts, None,
                lambda m: None, should_cancel, bank,
            )
        assert bank.get(resume_store.decode_key(_recording(), opts)) is not None
