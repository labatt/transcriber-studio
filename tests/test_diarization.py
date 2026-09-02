# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Diarization must be interruptible, and must not redo work it already did."""

from __future__ import annotations

import pytest

from tests.support import isolated_diarization_cache
from transcriber_studio import diarization
from transcriber_studio.diarization import DiarizationResult, SpeakerTurn
from transcriber_studio.job_cancel import JobCancelled

TURNS = [
    SpeakerTurn(start=0.0, end=2.5, speaker="SPEAKER_00"),
    SpeakerTurn(start=2.5, end=6.0, speaker="SPEAKER_01"),
]
# The centroid pyannote clustered each speaker around, kept so the speaker can
# be recognised in a later recording instead of being Speaker 2 again.
EMBEDDINGS = {"SPEAKER_00": [1.0, 0.0, 0.0], "SPEAKER_01": [0.0, 1.0, 0.0]}
RESULT = DiarizationResult(turns=TURNS, embeddings=EMBEDDINGS)


def test_the_progress_hook_is_where_cancel_gets_noticed():
    """pyannote runs as one long call; its hook is the only way in.

    Before this, pressing Cancel during a ten-minute diarization did nothing
    at all until the pipeline finished on its own.
    """
    hook = diarization._UiProgressHook(should_cancel=lambda: True)
    with pytest.raises(JobCancelled):
        hook("segmentation", None, total=10, completed=3)


def test_the_hook_runs_normally_when_nothing_asked_it_to_stop():
    seen: list[float] = []
    hook = diarization._UiProgressHook(progress_cb=seen.append, should_cancel=lambda: False)
    hook("segmentation", None, total=10, completed=5)
    assert seen and 0.0 <= seen[0] <= 1.0


def test_cached_turns_survive_a_round_trip(tmp_path):
    path = tmp_path / "turns.json"
    diarization.save_cached(path, RESULT)
    restored = diarization.load_cached(path)
    assert restored.turns == TURNS
    assert restored.embeddings == EMBEDDINGS


def test_a_cached_result_without_embeddings_is_still_usable(tmp_path):
    """Older pipelines return no embeddings; that is not a broken cache."""
    path = tmp_path / "turns.json"
    diarization.save_cached(path, DiarizationResult(turns=TURNS))
    restored = diarization.load_cached(path)
    assert restored.turns == TURNS
    assert restored.embeddings == {}


def test_the_cache_key_is_versioned(tmp_path):
    """A cache written before embeddings were stored must not be reused.

    It would come back without them, and speaker recognition would silently
    never run again on any recording already diarized once.
    """
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"pretend audio")
    assert diarization.CACHE_VERSION in "".join(
        [diarization.CACHE_VERSION]
    )
    material_now = diarization.cache_key(str(audio), 0, 0)
    original = diarization.CACHE_VERSION
    try:
        diarization.CACHE_VERSION = "v1"
        assert diarization.cache_key(str(audio), 0, 0) != material_now
    finally:
        diarization.CACHE_VERSION = original


def test_speech_seconds_totals_each_speaker():
    """How long someone spoke decides whether they can be recognised at all."""
    totals = RESULT.speech_seconds()
    assert totals["SPEAKER_00"] == pytest.approx(2.5)
    assert totals["SPEAKER_01"] == pytest.approx(3.5)


def test_an_unreadable_cache_is_ignored_rather_than_fatal(tmp_path):
    path = tmp_path / "turns.json"
    path.write_text("{ this is not json", encoding="utf-8")
    assert diarization.load_cached(path) is None


def test_the_key_changes_with_the_speaker_bounds(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"pretend audio")
    assert diarization.cache_key(str(audio), 0, 0) != diarization.cache_key(str(audio), 2, 2)


def test_a_second_run_reuses_the_turns_instead_of_the_gpu(tmp_path):
    """The expensive part is minutes of GPU; the result is a few kilobytes."""
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"pretend audio")

    with isolated_diarization_cache():
        diarization.save_cached(diarization.cache_path(str(audio), 0, 0), RESULT)

        diarizer = diarization.Diarizer.__new__(diarization.Diarizer)

        def _fail(*a, **k):
            raise AssertionError("the pipeline must not be loaded for a cache hit")

        diarizer._load = _fail
        logs: list[str] = []
        restored = diarizer.diarize(str(audio), 0, 0, log_cb=logs.append)

    assert restored.turns == TURNS
    assert restored.embeddings == EMBEDDINGS
    assert any("Reusing speaker detection" in line for line in logs)


def test_prune_keeps_only_the_most_recent(tmp_path):
    with isolated_diarization_cache() as cache:
        cache.mkdir(parents=True, exist_ok=True)
        for i in range(5):
            diarization.save_cached(cache / f"{i}.json", RESULT)
        diarization.prune(keep=2)
        assert len(list(cache.glob("*.json"))) == 2
