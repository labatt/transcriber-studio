# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Denoising must not quietly collapse a per-channel recording to mono.

Every denoiser here is mono, so the plain path downmixes. That is right for a
single decoder and wrong for per_channel mode, where each channel *is* a
speaker: a downmixed file reaches split_channels() looking like mono and the
whole recording is filed under one speaker.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from transcriber_studio import denoise
from transcriber_studio.audio_utils import FFMPEG, have_ffmpeg, probe, split_channels
from transcriber_studio.config import Settings

from .support import isolated_denoise_cache

pytestmark = pytest.mark.skipif(not have_ffmpeg(), reason="ffmpeg not installed")


def _stereo_file(directory: Path) -> str:
    """Two seconds of two different tones, one per channel."""
    dest = directory / "stereo.wav"
    subprocess.run(
        [
            FFMPEG, "-y",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-f", "lavfi", "-i", "sine=frequency=880:duration=2",
            "-filter_complex", "[0:a][1:a]amerge=inputs=2[a]",
            "-map", "[a]", str(dest),
        ],
        capture_output=True, check=True, timeout=60,
    )
    assert probe(str(dest))["channels"] == 2
    return str(dest)


def _settings() -> Settings:
    s = Settings()
    s.denoise_enabled = True
    s.denoise_backend = "ffmpeg"    # always available; the others need installs
    return s


def test_per_channel_denoise_keeps_the_channels_apart():
    with tempfile.TemporaryDirectory() as tmp, isolated_denoise_cache():
        source = _stereo_file(Path(tmp))
        out = denoise.enhance(source, _settings(), preserve_channels=True)
        assert out != source, "the audio should have been denoised"
        assert probe(out)["channels"] == 2, "per-channel denoise collapsed to mono"
        assert len(split_channels(out)) == 2


def test_downmix_denoise_still_returns_mono():
    """The default path is unchanged: one decoder wants one mixed signal."""
    with tempfile.TemporaryDirectory() as tmp, isolated_denoise_cache():
        source = _stereo_file(Path(tmp))
        out = denoise.enhance(source, _settings())
        assert probe(out)["channels"] == 1


def test_the_two_modes_do_not_share_a_cache_entry():
    """Same audio, same settings — but one file is mono and one is not."""
    with tempfile.TemporaryDirectory() as tmp, isolated_denoise_cache():
        source = _stereo_file(Path(tmp))
        mono = denoise.enhance(source, _settings())
        stereo = denoise.enhance(source, _settings(), preserve_channels=True)
        assert mono != stereo
        assert probe(mono)["channels"] == 1
        assert probe(stereo)["channels"] == 2


def test_mono_source_is_unaffected_by_preserve_channels():
    with tempfile.TemporaryDirectory() as tmp, isolated_denoise_cache():
        dest = Path(tmp) / "mono.wav"
        subprocess.run(
            [FFMPEG, "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
             "-ac", "1", str(dest)],
            capture_output=True, check=True, timeout=60,
        )
        out = denoise.enhance(str(dest), _settings(), preserve_channels=True)
        assert probe(out)["channels"] == 1
