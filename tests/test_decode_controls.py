# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""The two decoder knobs that guard against a decode loop.

The hallucination guard stops an invented passage seeding the *next* window.
Neither it nor the VAD does anything about a decoder stuck repeating itself
inside one window, and these are the levers for that. Both are off by default,
because both can degrade an ordinary transcript: people really do repeat
themselves, and nothing here can tell that apart from a loop.
"""

from __future__ import annotations

import pytest

from transcriber_studio.config import Settings
from transcriber_studio.jobs import JobRunner
from transcriber_studio.transcriber import (
    TranscribeOptions,
    pipeline_summary,
    transcribe_kwargs,
)


def _kwargs(**overrides) -> dict:
    return transcribe_kwargs(TranscribeOptions(**overrides), None)


def test_neither_knob_is_sent_when_both_are_off():
    """Passing the library's own defaults would read, at the call site, like a
    deliberate choice about repetition that nobody made."""
    kwargs = _kwargs()
    assert "repetition_penalty" not in kwargs
    assert "no_repeat_ngram_size" not in kwargs


def test_the_penalty_is_sent_once_turned_on():
    assert _kwargs(repetition_penalty=1.1)["repetition_penalty"] == pytest.approx(1.1)


def test_the_ngram_block_is_sent_once_turned_on():
    assert _kwargs(no_repeat_ngram_size=4)["no_repeat_ngram_size"] == 4


def test_an_explicit_default_is_still_treated_as_off():
    """1.0 and 0 mean 'off' whether they arrived from a default or a spinbox."""
    kwargs = _kwargs(repetition_penalty=1.0, no_repeat_ngram_size=0)
    assert "repetition_penalty" not in kwargs
    assert "no_repeat_ngram_size" not in kwargs


def test_faster_whisper_accepts_both_names():
    """Guards against a rename in the library turning these into silent no-ops.

    faster-whisper ignores nothing — it would raise — but the names are worth
    pinning because a typo here would be invisible in the transcript.
    """
    pytest.importorskip("faster_whisper")
    import inspect

    from faster_whisper import WhisperModel

    params = inspect.signature(WhisperModel.transcribe).parameters
    assert params["repetition_penalty"].default == 1
    assert params["no_repeat_ngram_size"].default == 0


def test_the_settings_reach_the_decoder():
    settings = Settings()
    settings.repetition_penalty = 1.15
    settings.no_repeat_ngram_size = 3
    opts = JobRunner(settings, client=object())._opts()
    assert opts.repetition_penalty == pytest.approx(1.15)
    assert opts.no_repeat_ngram_size == 3


def test_they_default_to_off_in_settings():
    settings = Settings()
    assert settings.repetition_penalty == 1.0
    assert settings.no_repeat_ngram_size == 0


# ---- the run log -----------------------------------------------------
def test_the_summary_says_nothing_when_they_are_off():
    text = " ".join(pipeline_summary(TranscribeOptions()))
    assert "Repetition penalty" not in text
    assert "Repeat block" not in text


def test_the_summary_reports_the_penalty():
    text = " ".join(pipeline_summary(TranscribeOptions(repetition_penalty=1.1)))
    assert "Repetition penalty: 1.10" in text


def test_the_summary_warns_about_the_ngram_block():
    """It deletes real repetition too, and the log should say so."""
    text = " ".join(pipeline_summary(TranscribeOptions(no_repeat_ngram_size=3)))
    assert "Repeat block: no 3-word sequence" in text
    assert "genuinely were said twice" in text
