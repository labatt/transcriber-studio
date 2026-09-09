# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""One spelling per person after AI cleanup.

Live, 2026-09-08: a two-speaker call came back from cleanup with "Chris
Labatt-Simon" on 272 sentences and "Chris Labat-Simon" on 88, and the rename
dialog then produced a speaker list with "Mark Villalobos" in it twice.
"""

from __future__ import annotations

from transcriber_studio.ai_cleanup import canonicalize_speakers
from transcriber_studio.jobs import apply_speaker_renames
from transcriber_studio.models import Recording, Segment, Source, TranscriptResult


def _segs(*names: str) -> list[Segment]:
    return [Segment(i, i + 1, f"line {i}", speaker=n) for i, n in enumerate(names)]


def _speakers(segments):
    return list(dict.fromkeys(s.speaker for s in segments if s.speaker))


def test_a_misspelling_of_the_commonest_name_is_folded_into_it():
    segs = _segs(*(["Chris Labatt-Simon"] * 5 + ["Chris Labat-Simon"] * 2 + ["Mark Villalobos"] * 4))
    out, merges = canonicalize_speakers(segs, [], ["Speaker 1", "Speaker 2"])
    assert _speakers(out) == ["Chris Labatt-Simon", "Mark Villalobos"]
    assert merges == ["unified speaker spelling — 'Chris Labat-Simon' → 'Chris Labatt-Simon' (2 sentence(s))"]


def test_the_roster_spelling_wins_even_when_the_misspelling_is_more_frequent():
    segs = _segs(*(["Chris Labat-Simon"] * 6 + ["Chris Labatt-Simon"] * 2))
    out, _ = canonicalize_speakers(segs, ["Chris Labatt-Simon"], [])
    assert _speakers(out) == ["Chris Labatt-Simon"]


def test_case_and_spacing_variants_are_one_person():
    segs = _segs("Mark Villalobos", "mark  villalobos", "Mark Villalobos ", "Mark Villalobos.")
    out, merges = canonicalize_speakers(segs, [], [])
    assert _speakers(out) == ["Mark Villalobos"]
    assert len(merges) == 3


def test_different_people_with_similar_names_stay_apart():
    segs = _segs("Brad Pottinger", "Brian Ottinger", "Greg Jackson", "Greg Johnson",
                 "Brad Pottinger", "Greg Jackson")
    out, merges = canonicalize_speakers(segs, [], [])
    assert _speakers(out) == ["Brad Pottinger", "Brian Ottinger", "Greg Jackson", "Greg Johnson"]
    assert merges == []


def test_two_roster_names_are_never_folded_into_each_other():
    segs = _segs("Anna Lee", "Anne Lee", "Anna Lee")
    out, merges = canonicalize_speakers(segs, ["Anna Lee", "Anne Lee"], [])
    assert _speakers(out) == ["Anna Lee", "Anne Lee"]
    assert merges == []


def test_generic_labels_are_never_merged_with_each_other():
    """"Speaker 1" and "Speaker 2" are 0.89 alike and are different people."""
    segs = _segs("Speaker 1", "Speaker 2", "Speaker 1", "Speaker 3")
    out, merges = canonicalize_speakers(segs, [], ["Speaker 1", "Speaker 2", "Speaker 3"])
    assert _speakers(out) == ["Speaker 1", "Speaker 2", "Speaker 3"]
    assert merges == []


def test_a_single_speaker_is_left_alone():
    segs = _segs("Chris", "Chris")
    assert canonicalize_speakers(segs, [], []) == (segs, [])


def test_renaming_two_labels_to_one_person_yields_one_speaker():
    result = TranscriptResult(
        recording=Recording(source=Source.LOCAL, id="r", name="n"),
        segments=_segs("Speaker 1", "Speaker 2", "Speaker 3"),
        speakers=["Speaker 1", "Speaker 2", "Speaker 3"],
    )
    apply_speaker_renames(result, {"Speaker 1": "Mark Villalobos", "Speaker 2": "Mark Villalobos"})
    assert result.speakers == ["Mark Villalobos", "Speaker 3"]
    assert [s.speaker for s in result.segments] == ["Mark Villalobos", "Mark Villalobos", "Speaker 3"]
