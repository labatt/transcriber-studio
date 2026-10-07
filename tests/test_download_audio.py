# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fetching a recording's audio without transcribing it.

The download goes through the cache, so a later transcription costs no second
download, and the copy the user asked for is named after the recording and
never overwrites anything.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from transcriber_studio import audio_cache, diarization, jobs
from transcriber_studio.config import Settings
from transcriber_studio.diarization import DiarizationResult, SpeakerTurn
from transcriber_studio.jobs import JobRunner
from transcriber_studio.models import Recording, Source


class _FakeClient:
    def __init__(self):
        self.downloads = 0

    def download_audio(self, file_id, dest_path, progress_cb=None, should_cancel=None,
                       label="", log_cb=None):
        self.downloads += 1
        Path(dest_path).write_bytes(b"ID3fake")
        return dest_path


def _runner(client) -> JobRunner:
    r = JobRunner.__new__(JobRunner)
    r.s = Settings(sanitize_names=True)
    r.client = client
    return r


def _recording() -> Recording:
    return Recording(source=Source.PLAUD, id="abc123", name="Weekly: sync?", date="2026-10-06")


def test_the_copy_is_named_after_the_recording_and_the_cache_is_filled(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "cache"
        monkeypatch.setattr(jobs, "CACHE_DIR", cache)
        monkeypatch.setattr(jobs, "cache_path", lambda pid: cache / f"{pid}.mp3")
        monkeypatch.setattr(audio_cache, "CACHE_DIR", cache)
        client = _FakeClient()
        logged = []
        saved = _runner(client).download_audio(
            _recording(), str(Path(tmp) / "out"), log_cb=logged.append,
        )
        assert Path(saved).name == "2026-10-06_Weekly sync.mp3"
        assert Path(saved).read_bytes() == b"ID3fake"
        assert (cache / "abc123.mp3").exists(), "the cache is what later runs read"
        assert client.downloads == 1
        assert any("Saved audio" in line for line in logged)


def test_a_second_download_reuses_the_cache_and_never_overwrites(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "cache"
        monkeypatch.setattr(jobs, "CACHE_DIR", cache)
        monkeypatch.setattr(jobs, "cache_path", lambda pid: cache / f"{pid}.mp3")
        monkeypatch.setattr(audio_cache, "CACHE_DIR", cache)
        client = _FakeClient()
        out = str(Path(tmp) / "out")
        first = _runner(client).download_audio(_recording(), out)
        second = _runner(client).download_audio(_recording(), out)
        assert client.downloads == 1, "the cache answered the second time"
        assert first != second and Path(second).name.endswith("(2).mp3")


# ---- detecting speakers before a run ---------------------------------
class _FakeDiarizer:
    calls: list[tuple[int, int]] = []

    def __init__(self, token, device="auto"):
        pass

    def diarize(self, audio_path, min_speakers=0, max_speakers=0, progress_cb=None,
                log_cb=None, should_cancel=None):
        _FakeDiarizer.calls.append((min_speakers, max_speakers))
        if progress_cb:
            progress_cb(1.0)
        return DiarizationResult(turns=[SpeakerTurn(0, 10, "SPEAKER_00")], embeddings={})


def test_detect_speakers_runs_pyannote_with_the_given_count(monkeypatch):
    _FakeDiarizer.calls = []
    monkeypatch.setattr(diarization, "Diarizer", _FakeDiarizer)
    monkeypatch.setattr(diarization, "is_available", lambda: True)
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / "a.wav"
        audio.write_bytes(b"RIFF")
        r = _runner(_FakeClient())
        r.s.hf_token = "hf_x"
        r.s.denoise_enabled = False
        rec = Recording(source=Source.LOCAL, id=str(audio), name="a", local_path=str(audio))
        progress = []
        path, diarized = r.detect_speakers(
            rec, min_speakers=3, max_speakers=3, progress_cb=progress.append,
        )
        assert path == str(audio)
        assert _FakeDiarizer.calls == [(3, 3)]
        assert [t.speaker for t in diarized.turns] == ["SPEAKER_00"]
        assert progress and progress[-1] == 1.0


def test_detect_speakers_needs_a_token():
    r = _runner(_FakeClient())
    r.s.hf_token = ""
    try:
        r.detect_speakers(_recording())
    except RuntimeError as e:
        assert "HuggingFace" in str(e)
    else:
        raise AssertionError("should have refused without a token")
