# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cancel must stop the job in flight and write nothing."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

from tests.support import isolated_resume_dir
from transcriber_studio.config import Settings
from transcriber_studio.job_cancel import JobCancelled
from transcriber_studio.jobs import JobRunner
from transcriber_studio.models import Recording, Segment, Source, TranscriptResult
from transcriber_studio.transcriber import TranscribeOptions, Transcriber


def _recording() -> Recording:
    return Recording(
        source=Source.LOCAL,
        id="call.wav",
        name="Weekly call",
        date="2026-08-19",
        local_path="call.wav",
        duration_seconds=120.0,
    )


def _transcript(rec: Recording) -> TranscriptResult:
    return TranscriptResult(
        recording=rec,
        segments=[Segment(0.0, 2.0, "Hello.", "Chris")],
        language="en",
        model="large-v3",
    )


class _FakeTranscriber:
    """Stands in for Whisper: records calls, optionally cancels mid-run."""

    def __init__(self, cancel_during: bool = False):
        self.calls = 0
        self.cancel_during = cancel_during

    def transcribe(self, recording, audio_path, opts, progress_cb=None, log_cb=None,
                   should_cancel=None):
        self.calls += 1
        if self.cancel_during:
            raise JobCancelled("Cancelled — stopping transcription.")
        return _transcript(recording)


def _runner(tmp: str, transcriber, cleanup_enabled: bool = False) -> JobRunner:
    s = Settings()
    s.output_dir = tmp
    s.formats = ["txt"]
    s.ai_cleanup_enabled = cleanup_enabled
    r = JobRunner.__new__(JobRunner)   # skip PlaudClient construction
    r.s = s
    r.client = None
    r.transcriber = transcriber
    return r


def test_cancel_before_start_never_touches_whisper():
    with isolated_resume_dir(), tempfile.TemporaryDirectory() as tmp:
        fake = _FakeTranscriber()
        result = _runner(tmp, fake).run(_recording(), should_cancel=lambda: True)
        assert result.cancelled is True
        assert result.error is None
        assert fake.calls == 0
        assert list(Path(tmp).iterdir()) == []


def test_cancel_during_transcription_writes_nothing():
    with isolated_resume_dir(), tempfile.TemporaryDirectory() as tmp:
        fake = _FakeTranscriber(cancel_during=True)
        result = _runner(tmp, fake).run(_recording(), should_cancel=lambda: False)
        assert result.cancelled is True
        assert result.output_paths == []
        assert list(Path(tmp).iterdir()) == []


def test_cancel_between_transcription_and_export_writes_nothing():
    """Flag flips only after Whisper returns — the export must still be skipped."""
    with isolated_resume_dir(), tempfile.TemporaryDirectory() as tmp:
        flag = {"cancelled": False}
        fake = _FakeTranscriber()

        def transcribe(recording, audio_path, opts, progress_cb=None, log_cb=None,
                       should_cancel=None):
            fake.calls += 1
            flag["cancelled"] = True          # user hits Cancel right about now
            return _transcript(recording)

        fake.transcribe = transcribe
        result = _runner(tmp, fake).run(
            _recording(), should_cancel=lambda: flag["cancelled"]
        )
        assert fake.calls == 1
        assert result.cancelled is True
        assert list(Path(tmp).iterdir()) == []


def test_uncancelled_job_still_writes_its_output():
    with isolated_resume_dir(), tempfile.TemporaryDirectory() as tmp:
        result = _runner(tmp, _FakeTranscriber()).run(_recording())
        assert result.cancelled is False
        assert result.error is None
        assert [Path(p).name for p in result.output_paths] == ["2026-08-19_Weekly call.txt"]


def test_whisper_segment_loop_stops_on_cancel():
    """The decode loop is the interrupt point for a long transcription."""
    decoded = []

    def segments():
        for i in range(100):
            decoded.append(i)
            yield SimpleNamespace(start=float(i), end=float(i + 1), text=f"seg {i}")

    model = SimpleNamespace(
        transcribe=lambda *a, **k: (segments(), SimpleNamespace(language="en"))
    )
    cancel_after = 3

    try:
        Transcriber()._run_whisper(
            model, "call.wav", "en", TranscribeOptions(), lambda m: None, None, 120.0,
            lambda: len(decoded) > cancel_after,
        )
        raise AssertionError("expected JobCancelled")
    except JobCancelled:
        pass

    # Stopped early instead of decoding all 100 segments.
    assert len(decoded) <= cancel_after + 2


def test_cancelled_download_leaves_no_partial_file_behind():
    import transcriber_studio.plaud_client as pc

    class _FakeResponse:
        headers = {"Content-Length": "1000"}

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size=0):
            for _ in range(100):
                yield b"x" * 10

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    real_get, real_url = pc.requests.get, pc.PlaudClient.audio_url
    try:
        pc.requests.get = lambda *a, **k: _FakeResponse()
        pc.PlaudClient.audio_url = lambda self, fid, log_cb=None: "https://example.invalid/a.mp3"
        with tempfile.TemporaryDirectory() as tmp:
            dest = str(Path(tmp) / "audio.mp3")
            client = pc.PlaudClient.__new__(pc.PlaudClient)
            try:
                client.download_audio("abc", dest, should_cancel=lambda: True)
                raise AssertionError("expected JobCancelled")
            except JobCancelled:
                pass
            # Neither the final file nor a .part stub may survive a cancel.
            assert list(Path(tmp).iterdir()) == []
    finally:
        pc.requests.get, pc.PlaudClient.audio_url = real_get, real_url
