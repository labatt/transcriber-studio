# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""ffmpeg-backed audio helpers: probing and channel splitting."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def _tool(name: str) -> str:
    """Resolve an ffmpeg tool at the moment it is needed, not at import.

    Resolving once at import outlives its own truth: winget installs ffmpeg into
    a version-stamped folder, so an upgrade deletes the directory this process
    put on its PATH at startup and the app reports ffmpeg as missing while a
    newer one sits right there. See components.refresh_path().
    """
    return shutil.which(name) or name


class _Tool(str):
    """The path to a tool, re-resolved every time it is read as a string.

    Existing code holds ``FFMPEG`` as a module constant and passes it straight
    into subprocess; keeping that shape while making the value current means the
    call sites do not all have to change.
    """

    def __new__(cls, name: str):
        self = super().__new__(cls, _tool(name))
        self._name = name
        return self

    def __str__(self) -> str:
        return _tool(self._name)

    def __fspath__(self) -> str:
        return _tool(self._name)


FFMPEG = _Tool("ffmpeg")
FFPROBE = _Tool("ffprobe")

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".mp4", ".m4b"}

#: Ceiling on any single ffmpeg conversion. Generous — a two-hour recording
#: converts in a couple of minutes — but bounded, because a subprocess with no
#: timeout is a job that can sit forever with nothing to show and no way out.
FFMPEG_TIMEOUT = 1800

DIARIZATION_SAMPLE_RATE = 16000


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def ffmpeg_path() -> str:
    """Where ffmpeg is right now, or "" when it cannot be found."""
    return shutil.which("ffmpeg") or ""


def probe(path: str) -> dict:
    """Return {'channels': int, 'duration': float} (best effort)."""
    info = {"channels": 1, "duration": 0.0}
    try:
        out = subprocess.run(
            [FFPROBE, "-v", "quiet", "-print_format", "json",
             "-show_streams", "-show_format", path],
            capture_output=True, text=True, timeout=30,
        ).stdout
        data = json.loads(out)
        for s in data.get("streams", []):
            if s.get("codec_type") == "audio":
                info["channels"] = int(s.get("channels", 1))
                break
        info["duration"] = float(data.get("format", {}).get("duration", 0.0) or 0.0)
    except Exception:
        pass
    return info


def split_channels(path: str, names: list[str] | None = None) -> list[tuple[str, str]]:
    """Split a multi-channel file into mono wavs, one per channel.

    Returns a list of (channel_label, wav_path). For a mono file returns a
    single ("mono", <converted wav>) entry.
    """
    meta = probe(path)
    channels = max(1, meta["channels"])
    tmpdir = Path(tempfile.mkdtemp(prefix="pws_chan_"))
    results: list[tuple[str, str]] = []

    if channels == 1:
        out = tmpdir / "mono.wav"
        subprocess.run(
            [FFMPEG, "-y", "-i", path, "-ac", "1", "-ar", "16000", str(out)],
            capture_output=True, check=True, timeout=FFMPEG_TIMEOUT,
        )
        return [("mono", str(out))]

    for ch in range(channels):
        label = (names[ch] if names and ch < len(names) else f"Channel {ch + 1}")
        out = tmpdir / f"ch{ch}.wav"
        # pan filter extracts a single channel to mono.
        subprocess.run(
            [FFMPEG, "-y", "-i", path,
             "-filter_complex", f"pan=mono|c0=c{ch}",
             "-ar", "16000", str(out)],
            capture_output=True, check=True, timeout=FFMPEG_TIMEOUT,
        )
        results.append((label, str(out)))
    return results


def load_waveform_for_diarization(path: str) -> dict[str, Any]:
    """Load audio as a pyannote-compatible dict without torchcodec/torchaudio.decode."""
    import soundfile as sf
    import torch

    if not have_ffmpeg():
        raise RuntimeError(
            "ffmpeg is required for speaker detection. Install ffmpeg and ensure it is on PATH."
        )

    tmp = Path(tempfile.mkdtemp(prefix="pws_diar_")) / "mono.wav"
    try:
        subprocess.run(
            [FFMPEG, "-y", "-i", path, "-ac", "1", "-ar", str(DIARIZATION_SAMPLE_RATE), str(tmp)],
            capture_output=True,
            check=True,
            timeout=FFMPEG_TIMEOUT,
        )
        data, sr = sf.read(str(tmp), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T.copy())
        return {"waveform": waveform, "sample_rate": int(sr)}
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)
