# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cancel must stop the job in flight and write nothing."""

from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

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
                   should_cancel=None, resume=None):
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
                       should_cancel=None, resume=None):
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


class _FakeResponse:
    """Enough of a requests response to stream from."""

    def __init__(self, body: bytes, status_code: int = 200, chunk: int = 10):
        self.status_code = status_code
        self.headers = {"Content-Length": str(len(body))}
        self._body, self._chunk = body, chunk

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=0):
        for i in range(0, len(self._body), self._chunk):
            yield self._body[i:i + self._chunk]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@contextlib.contextmanager
def _fake_download(respond):
    """Point plaud_client at a canned server. Yields the recorded requests."""
    import transcriber_studio.plaud_client as pc

    calls: list[dict] = []
    real_get, real_url = pc.requests.get, pc.PlaudClient.audio_url

    def get(url, **kwargs):
        calls.append(kwargs.get("headers") or {})
        return respond(len(calls))

    pc.requests.get = get
    pc.PlaudClient.audio_url = lambda self, fid, log_cb=None: "https://example.invalid/a.mp3"
    try:
        yield calls
    finally:
        pc.requests.get, pc.PlaudClient.audio_url = real_get, real_url


def test_cancelled_download_keeps_its_partial_so_the_next_run_resumes():
    """The bytes already on disk are the whole point of resuming.

    Deleting them made every interruption — a closed lid, a dropped link —
    start an hour-long download again from nothing.
    """
    import transcriber_studio.plaud_client as pc

    stop_after = [0]

    def should_cancel():
        stop_after[0] += 1
        return stop_after[0] > 2

    with _fake_download(lambda n: _FakeResponse(b"x" * 1000)):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "audio.mp3"
            client = pc.PlaudClient.__new__(pc.PlaudClient)
            with pytest.raises(JobCancelled):
                client.download_audio("abc", str(dest), should_cancel=should_cancel)
            assert not dest.exists(), "an interrupted download is not a finished one"
            partial = Path(f"{dest}.part")
            assert partial.exists() and partial.stat().st_size == 20


def test_download_resumes_from_where_it_stopped():
    import transcriber_studio.plaud_client as pc

    with _fake_download(lambda n: _FakeResponse(b"tail", status_code=206)) as calls:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "audio.mp3"
            Path(f"{dest}.part").write_bytes(b"head")
            client = pc.PlaudClient.__new__(pc.PlaudClient)
            client.download_audio("abc", str(dest))
            assert calls[0].get("Range") == "bytes=4-"
            assert dest.read_bytes() == b"headtail"


def test_download_starts_over_when_the_server_will_not_resume():
    """A 200 to a ranged request means the body starts from byte zero.

    Appending it to what we had would corrupt the file silently, which is worse
    than the wasted bandwidth of downloading it again.
    """
    import transcriber_studio.plaud_client as pc

    with _fake_download(lambda n: _FakeResponse(b"whole file", status_code=200)):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "audio.mp3"
            Path(f"{dest}.part").write_bytes(b"stale")
            client = pc.PlaudClient.__new__(pc.PlaudClient)
            client.download_audio("abc", str(dest))
            assert dest.read_bytes() == b"whole file"
