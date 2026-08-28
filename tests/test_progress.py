# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""The progress bar is the only sign of life during the long stages.

Each stage owns a slice of it. A stage that reports its own 0..1 over the top
of the slices before it sends the bar backwards, which reads as stuck.
"""

from __future__ import annotations

import tempfile

from tests.support import isolated_resume_dir
from transcriber_studio.config import Settings
from transcriber_studio.jobs import JobRunner
from transcriber_studio.models import Recording, Segment, Source, TranscriptResult


def _recording() -> Recording:
    return Recording(source=Source.LOCAL, id="call.wav", name="Weekly call",
                     date="2026-08-19", local_path="call.wav", duration_seconds=120.0)


class _FakeTranscriber:
    """Reports progress the way faster-whisper does: its own 0..1."""

    def transcribe(self, recording, audio_path, opts, progress_cb=None, log_cb=None,
                   should_cancel=None, resume=None):
        for fraction in (0.0, 0.1, 0.5, 0.95):
            if progress_cb:
                progress_cb(fraction)
        return TranscriptResult(
            recording=recording,
            segments=[Segment(start=0.0, end=1.0, text="hi")],
            language="en", model="fake", speakers=[],
        )


def _runner(tmp: str) -> JobRunner:
    s = Settings()
    s.output_dir = tmp
    s.formats = ["txt"]
    s.ai_cleanup_enabled = False
    s.denoise_enabled = False
    r = JobRunner.__new__(JobRunner)
    r.s = s
    r.client = None
    r.transcriber = _FakeTranscriber()
    return r


def test_the_bar_never_goes_backwards():
    """The whole sequence, with the audio stages reporting as they really do."""
    seen: list[float] = []
    with isolated_resume_dir(), tempfile.TemporaryDirectory() as tmp:
        runner = _runner(tmp)

        def fake_audio(recording, progress_cb, log_cb, should_cancel=None):
            for fraction in (0.1, 0.3):       # downloading owns 0.00-0.30
                progress_cb(fraction)
            for fraction in (0.34, 0.40):     # denoising owns 0.30-0.40
                progress_cb(fraction)
            return "call.wav"

        runner._ensure_audio = fake_audio
        runner.run(_recording(), progress_cb=seen.append)

    assert seen, "the job reported no progress at all"
    assert seen == sorted(seen), f"progress went backwards: {seen}"
    assert max(seen) > 0.4, "the decode never reported anything"


def test_transcription_reports_inside_its_own_slice():
    """Decoding owns 40%-92%: after the audio is ready, before AI cleanup."""
    seen: list[float] = []
    with isolated_resume_dir(), tempfile.TemporaryDirectory() as tmp:
        _runner(tmp).transcribe_only(_recording(), progress_cb=seen.append)

    decode = [f for f in seen if f > 0.4 or f == 0.4]
    assert decode, f"nothing landed in the decode slice: {seen}"
    assert min(decode) >= 0.40 and max(decode) <= 0.92, seen
