# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Combining glossaries: dedupe, and the disagreements that survive it."""

from __future__ import annotations

from transcriber_studio import glossary_merge
from transcriber_studio.glossary import merge_terms
from transcriber_studio.glossary_merge import Part, merge_parts
from transcriber_studio.glossary_store import CONFLICT_KEY


def _term(canonical: str, type_: str = "product", variants=None) -> dict:
    return {"canonical": canonical, "variants": list(variants or []), "type": type_}


def _by_name(entries: list[dict]) -> dict[str, dict]:
    return {e["canonical"]: e for e in entries}


def test_same_term_from_two_sources_becomes_one_entry():
    result = merge_parts(
        [
            Part("Acme", terms=[_term("GrowthMark", variants=["growth mark"])]),
            Part("Vendor", terms=[_term("GrowthMark", variants=["growth market"])]),
        ]
    )

    assert len(result.terms) == 1
    assert result.terms[0]["variants"] == ["growth mark", "growth market"]
    assert result.new_conflicts == 0


def test_spelling_variants_of_the_canonical_fold_together():
    result = merge_parts(
        [
            Part("Acme", terms=[_term("GrowthMark")]),
            Part("Vendor", terms=[_term("growth-mark")]),
        ]
    )

    assert len(result.terms) == 1
    assert result.terms[0]["canonical"] == "GrowthMark"
    assert "growth-mark" in result.terms[0]["variants"]


def test_same_term_with_a_different_type_is_tagged_not_silently_picked():
    result = merge_parts(
        [
            Part("Acme", terms=[_term("Scribe", "product")]),
            Part("Vendor", terms=[_term("Scribe", "concept")]),
        ]
    )

    assert result.new_conflicts == 1
    tagged = result.terms[0][CONFLICT_KEY]
    assert tagged["field"] == "type"
    assert sorted(tagged["values"]) == ["concept", "product"]
    assert sorted(tagged["sources"]) == ["Acme", "Vendor"]
    assert "Acme" in glossary_merge.describe(result.terms[0])


def test_same_speaker_label_with_a_different_name_is_tagged():
    result = merge_parts(
        [
            Part("Acme", speakers=[{"label": "Host", "name": "Dana Reyes"}]),
            Part("Vendor", speakers=[{"label": "Host", "name": "Dana Ruiz"}]),
        ]
    )

    assert result.new_conflicts == 1
    assert result.speakers[0][CONFLICT_KEY]["field"] == "name"


def test_a_missing_second_column_is_not_a_disagreement():
    """An extractor with no opinion should not make work for a person."""
    result = merge_parts(
        [
            Part("Acme", speakers=[{"label": "Host", "name": "Dana Reyes"}]),
            Part("Vendor", speakers=[{"label": "Host", "name": None}]),
        ]
    )

    assert result.new_conflicts == 0
    assert CONFLICT_KEY not in result.speakers[0]


def test_case_only_differences_are_not_a_disagreement():
    result = merge_parts(
        [
            Part("Acme", terms=[_term("Scribe", "Product")]),
            Part("Vendor", terms=[_term("Scribe", "product")]),
        ]
    )

    assert result.new_conflicts == 0


def test_only_the_clashing_entries_are_tagged():
    result = merge_parts(
        [
            Part("Acme", terms=[_term("Scribe", "product"), _term("GrowthMark")]),
            Part("Vendor", terms=[_term("Scribe", "concept"), _term("GrowthMark")]),
        ]
    )

    by_name = _by_name(result.terms)
    assert CONFLICT_KEY in by_name["Scribe"]
    assert CONFLICT_KEY not in by_name["GrowthMark"]
    assert result.total_conflicts == 1


def test_an_unresolved_tag_survives_a_later_merge():
    """A row nobody has fixed must not be un-flagged by the next import."""
    first = merge_parts(
        [
            Part("Acme", terms=[_term("Scribe", "product")]),
            Part("Vendor", terms=[_term("Scribe", "concept")]),
        ]
    )

    second = merge_parts(
        [
            Part("Combined", terms=first.terms),
            Part("Third", terms=[_term("GrowthMark")]),
        ]
    )

    assert second.new_conflicts == 0        # nothing new disagreed
    assert second.total_conflicts == 1      # but the old tag is still there
    assert CONFLICT_KEY in _by_name(second.terms)["Scribe"]


def test_a_job_contributing_terms_keeps_existing_tags():
    """merge_terms runs on every cleanup; it must not quietly clear a tag."""
    tagged = _term("Scribe", "product")
    tagged[CONFLICT_KEY] = {"field": "type", "values": ["product", "concept"], "sources": ["a", "b"]}

    merged = merge_terms([[tagged], [_term("Scribe", "product", ["scribes"])]])

    assert CONFLICT_KEY in merged[0]
    assert "scribes" in merged[0]["variants"]


def test_clearing_a_tag_leaves_the_entry_intact():
    entry = _term("Scribe", "product")
    entry[CONFLICT_KEY] = {"field": "type", "values": ["product", "concept"], "sources": []}

    glossary_merge.clear(entry)

    assert not glossary_merge.has_conflict(entry)
    assert entry["canonical"] == "Scribe" and entry["type"] == "product"
