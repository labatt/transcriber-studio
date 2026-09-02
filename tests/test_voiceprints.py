# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Recognising an enrolled speaker, and — mostly — declining to.

The set is open: most recordings contain someone who was never enrolled. The
two mistakes cost very different amounts. An unrecognised speaker is one edit
in the rename dialog. A wrongly named one is treated as authoritative by the
glossary and by AI Cleanup, which then rewrite the transcript around a person
who was never in the room. Nearly every test here is about the second one.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from transcriber_studio import voiceprints
from transcriber_studio.voiceprints import SpeakerProfile, Voiceprint

from .support import isolated_voiceprints

DIM = 8


def _vector(*values: float) -> list[float]:
    """A DIM-length vector from its leading values."""
    padded = list(values) + [0.0] * (DIM - len(values))
    return [float(x) for x in padded[:DIM]]


def _profile(name: str, *vectors: list[float], seconds: float = 120.0) -> SpeakerProfile:
    return SpeakerProfile(
        name=name,
        prints=[Voiceprint(vector=v, seconds=seconds) for v in vectors],
        embedding_id=voiceprints.EMBEDDING_ID,
        dim=DIM,
    )


def _tilted(base: list[float], toward: list[float], amount: float) -> list[float]:
    """``base`` nudged toward ``toward``, to make a similarity land near a value."""
    mixed = np.asarray(base) * (1 - amount) + np.asarray(toward) * amount
    return [float(x) for x in mixed]


# ---- the maths -------------------------------------------------------
def test_similarity_is_cosine():
    assert voiceprints.similarity(_vector(1, 0), _vector(1, 0)) == pytest.approx(1.0)
    assert voiceprints.similarity(_vector(1, 0), _vector(0, 1)) == pytest.approx(0.0)
    assert voiceprints.similarity(_vector(1, 0), _vector(-1, 0)) == pytest.approx(-1.0)


def test_scale_does_not_change_similarity():
    """Centroids are not unit-length; only direction carries the identity."""
    assert voiceprints.similarity(_vector(3, 4), _vector(30, 40)) == pytest.approx(1.0)


def test_a_zero_vector_is_not_usable_and_scores_nothing():
    """pyannote pads its centroid array with zero rows for speakers it never
    clustered. Those describe nobody and must never match."""
    assert not voiceprints.is_usable(_vector(0, 0))
    assert voiceprints.similarity(_vector(0, 0), _vector(1, 0)) == 0.0
    assert not any(math.isnan(x) for x in voiceprints.normalize(_vector(0, 0)))


# ---- recognising -----------------------------------------------------
def test_an_enrolled_speaker_is_recognised():
    alice = _profile("Alice", _vector(1, 0))
    matches = voiceprints.identify(
        {"SPEAKER_00": _vector(1, 0.05)}, {"SPEAKER_00": 120.0}, [alice]
    )
    assert [m.name for m in matches] == ["Alice"]
    assert matches[0].score > voiceprints.MATCH_THRESHOLD


def test_a_stranger_is_left_unnamed():
    alice = _profile("Alice", _vector(1, 0))
    matches = voiceprints.identify(
        {"SPEAKER_00": _vector(0, 1)}, {"SPEAKER_00": 120.0}, [alice]
    )
    assert matches[0].name == ""
    assert "closest match" in matches[0].reason


def test_a_close_call_between_two_people_names_neither():
    """A cluster that fits two enrolled voices about equally is describing a
    kind of voice, not a person."""
    base = _vector(1, 0)
    other = _vector(0.9, 0.44)
    matches = voiceprints.identify(
        {"SPEAKER_00": _tilted(base, other, 0.5)},
        {"SPEAKER_00": 120.0},
        [_profile("Alice", base), _profile("Bob", other)],
    )
    assert matches[0].name == ""
    assert "too close to call" in matches[0].reason


def test_a_brief_speaker_is_not_recognised():
    """A centroid pooled from a few seconds is one or two noisy chunks."""
    alice = _profile("Alice", _vector(1, 0))
    matches = voiceprints.identify(
        {"SPEAKER_00": _vector(1, 0)},
        {"SPEAKER_00": voiceprints.MIN_SPEECH_SECONDS - 1},
        [alice],
    )
    assert matches[0].name == ""
    assert "not enough speech" in matches[0].reason


def test_one_person_cannot_be_two_speakers_at_once():
    """Diarization sometimes splits one voice in two. Both halves match the
    same person, and handing the name to both would be a claim the recording
    contains two Alices."""
    alice = _profile("Alice", _vector(1, 0))
    matches = voiceprints.identify(
        {"SPEAKER_00": _vector(1, 0.02), "SPEAKER_01": _vector(1, 0.03)},
        {"SPEAKER_00": 120.0, "SPEAKER_01": 120.0},
        [alice],
    )
    named = [m.name for m in matches if m.named]
    assert named == ["Alice"]


def test_two_speakers_get_their_own_names():
    matches = voiceprints.identify(
        {"SPEAKER_00": _vector(1, 0), "SPEAKER_01": _vector(0, 1)},
        {"SPEAKER_00": 120.0, "SPEAKER_01": 120.0},
        [_profile("Alice", _vector(1, 0)), _profile("Bob", _vector(0, 1))],
    )
    assert {m.label: m.name for m in matches} == {
        "SPEAKER_00": "Alice",
        "SPEAKER_01": "Bob",
    }


def test_the_best_of_several_captures_wins_not_the_average():
    """Enrolments from different microphones are meant to differ; averaging
    them lands between the two and resembles neither."""
    alice = _profile("Alice", _vector(1, 0), _vector(0, 1))
    matches = voiceprints.identify(
        {"SPEAKER_00": _vector(1, 0.02)}, {"SPEAKER_00": 120.0}, [alice]
    )
    assert matches[0].name == "Alice"
    assert matches[0].score > 0.99


def test_nothing_enrolled_means_nothing_named():
    matches = voiceprints.identify(
        {"SPEAKER_00": _vector(1, 0)}, {"SPEAKER_00": 99.0}, []
    )
    assert matches[0].name == ""
    assert matches[0].reason == "no voices enrolled"


def test_a_profile_from_another_model_is_ignored_not_trusted():
    """Vectors from a different embedding are meaningless here, and matching
    them anyway would produce confident nonsense."""
    stale = _profile("Alice", _vector(1, 0))
    stale.embedding_id = "some/other-model"
    matches = voiceprints.identify(
        {"SPEAKER_00": _vector(1, 0)}, {"SPEAKER_00": 120.0}, [stale]
    )
    assert matches[0].name == ""


def test_a_profile_of_the_wrong_size_is_ignored():
    wrong = _profile("Alice", [1.0, 0.0])
    wrong.dim = 2
    matches = voiceprints.identify(
        {"SPEAKER_00": _vector(1, 0)}, {"SPEAKER_00": 120.0}, [wrong]
    )
    assert matches[0].name == ""


def test_a_padded_zero_centroid_is_never_named():
    alice = _profile("Alice", _vector(1, 0))
    matches = voiceprints.identify(
        {"SPEAKER_00": _vector(0, 0)}, {"SPEAKER_00": 120.0}, [alice]
    )
    assert matches[0].name == ""
    assert matches[0].reason == "no usable voice data"


# ---- storage ---------------------------------------------------------
def test_enrolling_and_reading_back():
    with isolated_voiceprints():
        voiceprints.enroll("Alice", _vector(1, 0), seconds=90.0, source="meeting.mp3")
        profile = voiceprints.get_profile("Alice")
        assert profile is not None
        assert profile.dim == DIM
        assert profile.total_seconds == pytest.approx(90.0)
        assert [p.source for p in profile.prints] == ["meeting.mp3"]


def test_enrolling_the_same_person_twice_keeps_both_captures():
    with isolated_voiceprints():
        voiceprints.enroll("Alice", _vector(1, 0), seconds=90.0)
        voiceprints.enroll("Alice", _vector(0, 1), seconds=90.0)
        assert len(voiceprints.get_profile("Alice").prints) == 2


def test_a_short_sample_is_refused():
    with isolated_voiceprints():
        with pytest.raises(voiceprints.VoiceprintError, match="seconds"):
            voiceprints.enroll(
                "Alice", _vector(1, 0),
                seconds=voiceprints.MIN_ENROLL_SECONDS - 1,
            )


def test_an_empty_name_is_refused():
    with isolated_voiceprints():
        with pytest.raises(voiceprints.VoiceprintError):
            voiceprints.enroll("  ", _vector(1, 0), seconds=90.0)


def test_a_zero_vector_cannot_be_enrolled():
    with isolated_voiceprints():
        with pytest.raises(voiceprints.VoiceprintError):
            voiceprints.enroll("Alice", _vector(0, 0), seconds=90.0)


def test_names_that_differ_only_by_case_are_one_person():
    with isolated_voiceprints():
        voiceprints.enroll("Alice", _vector(1, 0), seconds=90.0)
        voiceprints.enroll("alice", _vector(0, 1), seconds=90.0)
        assert len(voiceprints.load_profiles()) == 1
        assert len(voiceprints.get_profile("Alice").prints) == 2


def test_deleting_a_profile():
    with isolated_voiceprints():
        voiceprints.enroll("Alice", _vector(1, 0), seconds=90.0)
        assert voiceprints.delete_profile("Alice")
        assert voiceprints.get_profile("Alice") is None


def test_an_unreadable_profile_is_skipped_not_fatal():
    with isolated_voiceprints() as directory:
        voiceprints.enroll("Alice", _vector(1, 0), seconds=90.0)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "broken.json").write_text("{ not json", encoding="utf-8")
        assert [p.name for p in voiceprints.load_profiles()] == ["Alice"]


def test_enrolled_voices_are_used_by_default():
    """identify() reads the store when it is not handed profiles."""
    with isolated_voiceprints():
        voiceprints.enroll("Alice", _vector(1, 0), seconds=90.0)
        matches = voiceprints.identify(
            {"SPEAKER_00": _vector(1, 0.02)}, {"SPEAKER_00": 120.0}
        )
        assert matches[0].name == "Alice"
