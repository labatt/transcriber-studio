# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Transcriber Studio — a local transcription workbench.

Turns recordings into clean, speaker-labelled transcripts on your own machine:
a denoise/VAD/vocabulary-biasing front end, Whisper (or ElevenLabs Scribe) for
the words, pyannote for who said them, and an optional LLM cleanup pass. Audio
sources include local files and PLAUD cloud recorders.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
