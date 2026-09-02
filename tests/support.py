# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared test helpers."""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path

from transcriber_studio import (
    denoise,
    diarization,
    glossary_store,
    history,
    name_store,
    voiceprints,
)
from transcriber_studio import resume as resume_store


@contextmanager
def isolated_resume_dir():
    """Keep checkpoints out of the real app directory during tests.

    Without this, a test that completes a transcription leaves a checkpoint
    that a later test restores instead of running its own fake — the tests
    would pass or fail depending on the order they ran in.
    """
    original = resume_store.RESUME_DIR
    with tempfile.TemporaryDirectory() as tmp:
        resume_store.RESUME_DIR = Path(tmp)
        try:
            yield Path(tmp)
        finally:
            resume_store.RESUME_DIR = original


@contextmanager
def isolated_history_dir():
    """Keep the processing history out of the real app directory during tests.

    The store caches what it read, so the cache is dropped on the way in and
    out — otherwise a test would see entries written by the previous one.
    """
    original_dir, original_path = history.APP_DIR, history.HISTORY_PATH
    with tempfile.TemporaryDirectory() as tmp:
        history.APP_DIR = Path(tmp)
        history.HISTORY_PATH = Path(tmp) / "history.json"
        history.clear()
        try:
            yield Path(tmp)
        finally:
            history.APP_DIR, history.HISTORY_PATH = original_dir, original_path
            history._cache = None           # forget the temp dir's entries
            history._cache_stamp = None


@contextmanager
def isolated_glossary_dir():
    """Keep the shared-glossary library out of the real app directory.

    A test that creates a glossary would otherwise add it to the library the
    running app reads, and every later run would list it.
    """
    original = glossary_store.GLOSSARY_DIR
    with tempfile.TemporaryDirectory() as tmp:
        glossary_store.GLOSSARY_DIR = Path(tmp)
        try:
            yield Path(tmp)
        finally:
            glossary_store.GLOSSARY_DIR = original


@contextmanager
def isolated_denoise_cache():
    """Keep enhanced audio out of the real cache during tests.

    The cache is keyed by the source file's path and mtime, so a test writing
    into the real one would leave megabytes of wav behind and could hand a
    later run a hit it never earned.
    """
    original = denoise.CACHE_DIR
    with tempfile.TemporaryDirectory() as tmp:
        denoise.CACHE_DIR = Path(tmp) / "denoise"
        try:
            yield denoise.CACHE_DIR
        finally:
            denoise.CACHE_DIR = original


@contextmanager
def isolated_diarization_cache():
    """Keep cached speaker turns out of the real cache during tests."""
    original = diarization.CACHE_DIR
    with tempfile.TemporaryDirectory() as tmp:
        diarization.CACHE_DIR = Path(tmp) / "diarization"
        try:
            yield diarization.CACHE_DIR
        finally:
            diarization.CACHE_DIR = original


@contextmanager
def isolated_name_store():
    """Keep renamed-recording names out of the real app directory."""
    original = name_store.STORE_PATH
    with tempfile.TemporaryDirectory() as tmp:
        name_store.STORE_PATH = Path(tmp) / "plaud_names.json"
        name_store.load(force=True)      # drop the previous test's names
        try:
            yield name_store.STORE_PATH
        finally:
            name_store.STORE_PATH = original
            name_store.load(force=True)


@contextmanager
def isolated_voiceprints():
    """Keep enrolled voices out of the real app directory during tests."""
    original = voiceprints.PROFILE_DIR
    with tempfile.TemporaryDirectory() as tmp:
        voiceprints.PROFILE_DIR = Path(tmp) / "voiceprints"
        try:
            yield voiceprints.PROFILE_DIR
        finally:
            voiceprints.PROFILE_DIR = original
