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

#: How far either side of a target boundary to hunt for a pause to cut on.
SPLIT_SEARCH_WINDOW = 90.0
#: Anything quieter than this for long enough counts as a gap between words.
SILENCE_DB = -30
SILENCE_SECONDS = 0.35


def silence_midpoints(path: str, timeout: float = FFMPEG_TIMEOUT) -> list[float]:
    """Middle of every detected silence, in seconds, in order.

    Used to choose where to cut a long recording. ffmpeg reports these on
    stderr as it decodes; there is no cheaper way to ask for them.
    """
    result = subprocess.run(
        [FFMPEG, "-i", path, "-af",
         f"silencedetect=noise={SILENCE_DB}dB:d={SILENCE_SECONDS}", "-f", "null", "-"],
        capture_output=True, text=True, errors="replace", timeout=timeout,
    )
    starts: list[float] = []
    midpoints: list[float] = []
    for line in (result.stderr or "").splitlines():
        if "silence_start:" in line:
            try:
                starts.append(float(line.split("silence_start:")[1].split()[0]))
            except (ValueError, IndexError):
                continue
        elif "silence_end:" in line and starts:
            try:
                end = float(line.split("silence_end:")[1].split()[0])
            except (ValueError, IndexError):
                continue
            midpoints.append((starts.pop() + end) / 2)
    return sorted(midpoints)


def split_points(duration: float, target: float, quiet: list[float]) -> list[float]:
    """Cut times for a recording, nudged onto a pause where one is near.

    Cutting on a fixed clock lands mid-word twice an hour and garbles a word
    at every seam. A pause within a minute and a half of the target is worth
    far more than an exact boundary.
    """
    points: list[float] = []
    position = 0.0
    while duration - position > target:
        target_at = position + target
        nearby = [
            q for q in quiet
            if abs(q - target_at) <= SPLIT_SEARCH_WINDOW and q > position + 60
        ]
        cut = min(nearby, key=lambda q: abs(q - target_at)) if nearby else target_at
        points.append(cut)
        position = cut
    return points


def split_for_upload(
    path: str, target_seconds: float, out_dir: str, log=None, overlap: float = 0.0,
    timeout: float = FFMPEG_TIMEOUT,
) -> list[tuple[str, float, float]]:
    """Cut a recording into uploadable parts. Returns (path, start, seam).

    No part is longer than ``target_seconds``, lead-in included. ``start`` puts
    the part's timestamps back on the original timeline. ``seam`` is where this
    part's words actually belong: every part after the first begins ``overlap``
    seconds early, and that lead-in is transcribed twice on purpose. Hearing
    the same speech in two parts is what lets an engine that numbers speakers
    per request be matched up across the join — the duplicate words themselves
    are thrown away.
    """
    info = probe(path)
    duration = float(info.get("duration") or 0.0)
    if duration <= target_seconds:
        return [(path, 0.0, 0.0)]

    try:
        quiet = silence_midpoints(path, timeout=timeout)
    except (subprocess.SubprocessError, OSError):
        quiet = []          # no pauses found is not fatal; cut on the clock
    spans = part_spans(duration, target_seconds, overlap, quiet)
    if log:
        cuts = [seam for _, seam, _ in spans[1:]]
        landed = sum(1 for c in cuts if any(abs(c - q) < 0.01 for q in quiet))
        log(f"Splitting into {len(spans)} part(s) — {landed} of {len(cuts)} "
            f"cut(s) landed on a pause.")

    suffix = Path(path).suffix or ".wav"
    parts: list[tuple[str, float, float]] = []
    for index, (start, seam, end) in enumerate(spans):
        part = Path(out_dir) / f"part{index:03d}{suffix}"
        subprocess.run(
            [FFMPEG, "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
             "-i", path, "-c", "copy", str(part)],
            capture_output=True, check=True, timeout=timeout,
        )
        parts.append((str(part), start, seam))
    return parts


def part_spans(
    duration: float, target: float, overlap: float, quiet: list[float]
) -> list[tuple[float, float, float]]:
    """(start, seam, end) for each part, none longer than ``target``.

    The seams are placed ``target - overlap`` apart so that once the lead-in
    is prepended the part is at most ``target`` long. Sizing the seams at the
    full target and then adding the lead-in made every middle part
    ``overlap`` too long, and for Gemini that was the difference between a
    clean transcript and one with its timings in pieces.
    """
    stride = max(60.0, target - overlap) if overlap else target
    cuts = split_points(duration, stride, quiet)
    bounds = [0.0, *cuts, duration]
    spans: list[tuple[float, float, float]] = []
    for index in range(len(bounds) - 1):
        seam, end = bounds[index], bounds[index + 1]
        start = max(0.0, seam - overlap) if index else 0.0
        spans.append((start, seam, end))
    return spans


# ---- shrinking a recording before it goes to the cloud ----------------
#: Speech, mono, at the sample rate every recogniser resamples to anyway.
#: 32 kbit/s is generous for voice — the models take 16 kHz input, so nothing
#: above that band survives the trip regardless of what was uploaded.
UPLOAD_SAMPLE_RATE = 16_000
UPLOAD_BITRATE_KBPS = 32

#: Below this there is nothing to gain: re-encoding costs CPU and a few
#: seconds, and a small file already uploads in moments.
COMPRESS_ABOVE_BYTES = 8 * 1024 * 1024


def compress_for_upload(path: str, out_dir: str, log=None,
                        timeout: float = FFMPEG_TIMEOUT) -> str:
    """A smaller copy for uploading, or the original when that is pointless.

    A Plaud recording arrives as a 128 kbit/s stereo-ish MP3, and denoising
    turns it into uncompressed WAV — 73 minutes of speech became 70 MB one way
    and 140 MB the other. Neither is anything to do with how much information
    the recogniser can use: it resamples to 16 kHz mono before the first layer.

    Sending the smaller file is not only faster. Azure's endpoint applies its
    own request timeout, and a 70 MB upload that took two minutes came back as
    "RequestTimeout" having never been transcribed at all — the audio was well
    inside the documented five-hour limit, but the *transfer* was not inside
    the service's patience.

    Never fatal: if ffmpeg is missing or fails, the original path is returned
    and the upload proceeds as it did before.
    """
    source = Path(path)
    try:
        original = source.stat().st_size
    except OSError:
        return path
    if original <= COMPRESS_ABOVE_BYTES or not have_ffmpeg():
        return path

    dest = Path(out_dir) / f"{source.stem}.upload.mp3"
    try:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [FFMPEG, "-y", "-i", str(source),
             "-vn", "-map_metadata", "-1",
             "-ac", "1", "-ar", str(UPLOAD_SAMPLE_RATE),
             "-b:a", f"{UPLOAD_BITRATE_KBPS}k", str(dest)],
            capture_output=True, check=True, timeout=timeout,
        )
    except (subprocess.SubprocessError, OSError) as e:
        if log:
            log(f"Could not compress before upload ({e}) — sending the original.")
        return path
    try:
        reduced = dest.stat().st_size
    except OSError:
        return path
    if reduced >= original:
        return path         # already smaller than anything we would make
    if log:
        log(
            f"Compressed for upload: {original / 1e6:.0f} MB to "
            f"{reduced / 1e6:.0f} MB ({UPLOAD_SAMPLE_RATE // 1000} kHz mono) — "
            "the recogniser resamples to this anyway."
        )
    return str(dest)
