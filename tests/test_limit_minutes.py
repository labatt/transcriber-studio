# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Running only the first few minutes of a recording.

The cut happens before any stage sees the audio, so denoising, speaker
detection and every engine work on the same short file; and a transcript of
ten minutes must never be banked, or restored, as a transcript of the whole.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from transcriber_studio import audio_utils, components, jobs
from transcriber_studio import resume as resume_store
from transcriber_studio.config import Settings
from transcriber_studio.jobs import JobRunner
from transcriber_studio.models import Recording, Source
from transcriber_studio.transcriber import TranscribeOptions

components.refresh_path()
needs_ffmpeg = pytest.mark.skipif(not audio_utils.have_ffmpeg(), reason="ffmpeg not on PATH")


def _tone(path: Path, seconds: float, *extra: str) -> Path:
    subprocess.run(
        [audio_utils.FFMPEG, "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-ar", "16000", "-ac", "1", *extra, str(path)],
        capture_output=True, check=True, timeout=60,
    )
    return path


def _opus_named_mp3(path: Path, seconds: float) -> Path:
    """What Plaud's cloud actually serves: Opus in Ogg, called ``.mp3``."""
    return _tone(path, seconds, "-c:a", "libopus", "-f", "ogg")


def _runner(limit: int) -> JobRunner:
    r = JobRunner.__new__(JobRunner)
    r.s = Settings(limit_minutes=limit)
    return r


@needs_ffmpeg
def test_a_long_file_is_cut_to_the_first_minutes(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(jobs, "CACHE_DIR", Path(tmp) / "cache")
        source = _tone(Path(tmp) / "long.wav", 150.0)
        logged = []
        cut = _runner(1)._first_minutes(str(source), 1, logged.append)
        assert cut != str(source) and Path(cut).name == "long.first1m.wav"
        assert abs(audio_utils.probe(cut)["duration"] - 60.0) < 0.5
        assert any("first 1 minutes" in line for line in logged)
        # Cut once, reused after.
        stamp = Path(cut).stat().st_mtime_ns
        assert _runner(1)._first_minutes(str(source), 1, None) == cut
        assert Path(cut).stat().st_mtime_ns == stamp


@needs_ffmpeg
def test_an_ogg_file_named_mp3_is_cut_as_ogg(monkeypatch):
    """Regression: the cut used to trust the extension, write Opus into an
    MP3 container, leave a zero-byte file, and reuse it on the next run —
    which the engine then reported as an empty or corrupted upload."""
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(jobs, "CACHE_DIR", Path(tmp) / "cache")
        source = _opus_named_mp3(Path(tmp) / "plaud.mp3", 150.0)
        assert audio_utils.probe(str(source))["format"] == "ogg"
        assert audio_utils.container_extension(str(source)) == ".ogg"
        cut = _runner(1)._first_minutes(str(source), 1, None)
        assert Path(cut).name == "plaud.first1m.ogg"
        assert Path(cut).stat().st_size > 0
        assert abs(audio_utils.probe(cut)["duration"] - 60.0) < 0.5


@needs_ffmpeg
def test_an_empty_leftover_cut_is_never_reused(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "cache"
        monkeypatch.setattr(jobs, "CACHE_DIR", cache)
        source = _tone(Path(tmp) / "long.wav", 150.0)
        cache.mkdir()
        leftover = cache / "long.first1m.mp3"
        leftover.write_bytes(b"")
        cut = _runner(1)._first_minutes(str(source), 1, None)
        assert Path(cut).stat().st_size > 0
        assert not leftover.exists(), "the stale empty file is cleared away"


@needs_ffmpeg
def test_gemini_describes_the_file_by_its_bytes():
    from transcriber_studio import stt_gemini

    with tempfile.TemporaryDirectory() as tmp:
        source = _opus_named_mp3(Path(tmp) / "plaud.mp3", 2.0)
        assert stt_gemini.mime_type_for(str(source)) == "audio/ogg"
        assert stt_gemini.mime_type_for(str(_tone(Path(tmp) / "t.wav", 1.0))) == "audio/wav"


@needs_ffmpeg
def test_a_file_within_the_limit_is_used_whole(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(jobs, "CACHE_DIR", Path(tmp) / "cache")
        source = _tone(Path(tmp) / "short.wav", 20.0)
        logged = []
        assert _runner(1)._first_minutes(str(source), 1, logged.append) == str(source)
        assert any("using all of it" in line for line in logged)


def test_no_limit_means_no_cut(monkeypatch):
    calls = []
    monkeypatch.setattr(JobRunner, "_first_minutes", lambda self, *a, **k: calls.append(a) or a[0])
    monkeypatch.setattr(JobRunner, "_source_audio", lambda self, *a, **k: "orig.mp3")
    monkeypatch.setattr(jobs.denoise, "enhance", lambda source, *a, **k: source)
    r = _runner(0)
    assert r._ensure_audio(Recording(source=Source.LOCAL, id="x", name="x"), None, None) == "orig.mp3"
    assert calls == []
    r = _runner(10)
    r._ensure_audio(Recording(source=Source.LOCAL, id="x", name="x"), None, None)
    assert calls and calls[0][1] == 10


def test_the_limit_reaches_the_options_and_the_resume_key():
    r = _runner(10)
    r.s.channel_names = ""
    assert r._opts().limit_minutes == 10
    rec = Recording(source=Source.LOCAL, id="r", name="r")
    whole = TranscribeOptions()
    part = TranscribeOptions(limit_minutes=10)
    assert resume_store.transcript_key(rec, whole) != resume_store.transcript_key(rec, part)
    assert resume_store.decode_key(rec, whole) != resume_store.decode_key(rec, part)
