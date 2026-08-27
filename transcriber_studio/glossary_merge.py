# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Combining glossaries: one union, deduped, with the disagreements kept visible.

Merging two glossaries is mostly boring — the same term from two sources is one
term, and their variant spellings pile up under it. What is not boring is the
case where both sources name the same thing and describe it differently: one
calls "Scribe" a product, the other a concept; one has SPEAKER_00 as Dana Reyes,
the other as Dana R. Picking a winner silently would bury exactly the rows a
person needs to look at, so those entries keep both readings and carry a tag
until someone settles it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .glossary import _term_key, merge_speakers, merge_terms
from .glossary_store import CONFLICT_KEY, SharedGlossary


@dataclass(frozen=True)
class Part:
    """One glossary going into a merge, named so conflicts can be attributed."""

    name: str
    speakers: list[dict[str, Any]] = field(default_factory=list)
    terms: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def of(cls, glossary: SharedGlossary) -> Part:
        return cls(glossary.name, list(glossary.speakers), list(glossary.terms))

    @classmethod
    def of_payload(cls, name: str, payload: dict[str, list]) -> Part:
        return cls(
            name,
            list(payload.get("speakers") or []),
            list(payload.get("terms") or []),
        )


@dataclass(frozen=True)
class MergeResult:
    speakers: list[dict[str, Any]]
    terms: list[dict[str, Any]]
    new_conflicts: int      # disagreements this merge uncovered
    total_conflicts: int    # including tags carried in from earlier merges

    def summary(self) -> str:
        parts = [f"{len(self.terms)} term(s)", f"{len(self.speakers)} speaker(s)"]
        if self.total_conflicts:
            parts.append(f"{self.total_conflicts} tagged for review")
        return ", ".join(parts)


def merge_parts(parts: Sequence[Part]) -> MergeResult:
    """Union every part, deduped, tagging entries the parts disagree about."""
    term_conflicts = _detect(parts, "terms", "canonical", "type")
    speaker_conflicts = _detect(parts, "speakers", "label", "name")

    terms = merge_terms([p.terms for p in parts])
    speakers = merge_speakers([p.speakers for p in parts])

    new = _apply(terms, "canonical", term_conflicts)
    new += _apply(speakers, "label", speaker_conflicts)
    total = sum(1 for e in (*terms, *speakers) if e.get(CONFLICT_KEY))
    return MergeResult(speakers=speakers, terms=terms, new_conflicts=new, total_conflicts=total)


def apply_to(destination: SharedGlossary, parts: Sequence[Part]) -> MergeResult:
    """Merge parts into a glossary in place. Saving is the caller's business."""
    result = merge_parts(parts)
    destination.terms = result.terms
    destination.speakers = result.speakers
    return result


def _detect(
    parts: Sequence[Part], list_name: str, key_field: str, value_field: str
) -> dict[str, dict[str, Any]]:
    """Keys where the parts gave the same first column different second columns.

    A part that leaves the second column empty is not disagreeing with anything
    — an extractor that had no opinion should not tag a row a person then has
    to clear.
    """
    seen: dict[str, dict[str, str]] = {}    # key -> normalized value -> source
    display: dict[str, dict[str, str]] = {}  # key -> normalized value -> as written
    for part in parts:
        for entry in getattr(part, list_name):
            key = _term_key(str(entry.get(key_field) or "").strip())
            if not key:
                continue
            value = str(entry.get(value_field) or "").strip()
            if not value:
                continue
            norm = _term_key(value)
            seen.setdefault(key, {}).setdefault(norm, part.name)
            display.setdefault(key, {}).setdefault(norm, value)

    conflicts: dict[str, dict[str, Any]] = {}
    for key, values in seen.items():
        if len(values) < 2:
            continue
        conflicts[key] = {
            "field": value_field,
            "values": [display[key][norm] for norm in values],
            "sources": list(values.values()),
        }
    return conflicts


def _apply(
    entries: Iterable[dict[str, Any]], key_field: str, conflicts: dict[str, dict[str, Any]]
) -> int:
    tagged = 0
    for entry in entries:
        payload = conflicts.get(_term_key(str(entry.get(key_field) or "")))
        if payload and entry.get(CONFLICT_KEY) != payload:
            entry[CONFLICT_KEY] = payload
            tagged += 1
    return tagged


def describe(entry: dict[str, Any]) -> str:
    """One line naming what disagrees, and who said what."""
    payload = entry.get(CONFLICT_KEY)
    if not payload:
        return ""
    values = [str(v) for v in (payload.get("values") or [])]
    sources = [str(s) for s in (payload.get("sources") or [])]
    field_name = str(payload.get("field") or "value")
    shown = [
        f"{value} ({sources[i]})" if i < len(sources) else value
        for i, value in enumerate(values)
    ]
    return f"{field_name}: " + "  vs  ".join(shown) if shown else field_name


def has_conflict(entry: dict[str, Any]) -> bool:
    return bool(entry.get(CONFLICT_KEY))


def clear(entry: dict[str, Any]) -> dict[str, Any]:
    """Drop the tag — the row has been edited or accepted as it stands."""
    entry.pop(CONFLICT_KEY, None)
    return entry


def count(entries: Iterable[dict[str, Any]]) -> int:
    return sum(1 for entry in entries if entry.get(CONFLICT_KEY))
