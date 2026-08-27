# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Voice activity detection in front of the decoder.

Cutting non-speech is not only a speed win. Whisper's worst failure mode is
hallucinating fluent text over silence or noise — it was trained on speech, so
given something that is not speech it produces the most likely speech anyway.
Never showing it those stretches removes the failure at the source.

faster-whisper bundles Silero VAD and, importantly, maps the timestamps back
onto the original timeline for us, so segment times still line up with the
audio and with diarization. This module is the settings-to-parameters seam and
the "which version is actually installed" reporting around it.
"""

from __future__ import annotations

import re
from pathlib import Path

from .config import Settings

#: The asset faster-whisper ships, newest first. The name carries the version.
_ASSET_PATTERNS = (
    re.compile(r"silero.*?v(\d+)\.onnx$", re.IGNORECASE),
    re.compile(r"silero_vad\.onnx$", re.IGNORECASE),
)


def bundled_version() -> str:
    """The Silero VAD version faster-whisper will use, e.g. "v6"; "" if unknown."""
    try:
        import faster_whisper
    except Exception:
        return ""
    assets = Path(faster_whisper.__file__).parent / "assets"
    if not assets.is_dir():
        return ""
    best = 0
    plain = False
    for path in assets.glob("*.onnx"):
        match = _ASSET_PATTERNS[0].search(path.name)
        if match:
            best = max(best, int(match.group(1)))
        elif _ASSET_PATTERNS[1].search(path.name):
            plain = True
    if best:
        return f"v{best}"
    return "v4" if plain else ""


def engine_version() -> str:
    try:
        from importlib.metadata import version

        return version("faster-whisper")
    except Exception:
        return ""


def describe(settings: Settings) -> str:
    """One line for the UI: which VAD runs, and what it is set to do."""
    if not settings.vad_enabled:
        return (
            "VAD off — the decoder sees silence and noise too, which is where "
            "Whisper invents text."
        )

    silero = bundled_version()
    engine = engine_version()
    where = f"Silero VAD {silero}" if silero else "Silero VAD"
    if engine:
        where += f" (bundled with faster-whisper {engine})"
    return (
        f"{where} — threshold {settings.vad_threshold:.2f}, "
        f"{settings.vad_min_silence_ms} ms of silence splits, "
        f"{settings.vad_speech_pad_ms} ms kept either side."
    )


def parameters(settings: Settings) -> dict[str, float | int]:
    """Silero parameters for this run, filtered to what the installed VAD takes.

    faster-whisper turns this dict into its own VadOptions, and that dataclass
    has gained and lost fields across releases — passing one it does not know
    is a TypeError in the middle of a job.
    """
    wanted: dict[str, float | int] = {
        "threshold": float(settings.vad_threshold),
        "min_speech_duration_ms": int(settings.vad_min_speech_ms),
        "min_silence_duration_ms": int(settings.vad_min_silence_ms),
        "speech_pad_ms": int(settings.vad_speech_pad_ms),
    }
    if settings.vad_max_speech_s > 0:
        wanted["max_speech_duration_s"] = float(settings.vad_max_speech_s)
    return {k: v for k, v in wanted.items() if k in supported_fields()}


def supported_fields() -> set[str]:
    """Field names the installed faster-whisper's VadOptions accepts."""
    try:
        import dataclasses

        from faster_whisper.vad import VadOptions

        return {f.name for f in dataclasses.fields(VadOptions)}
    except Exception:
        # Nothing to filter against: hand back the names every version has had.
        return {
            "threshold",
            "min_speech_duration_ms",
            "max_speech_duration_s",
            "min_silence_duration_ms",
            "speech_pad_ms",
        }
