# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""A library of named glossaries that several jobs can share.

The per-recording glossary (``transcriber_studio.glossary``) is extracted from one transcript
and lives next to that transcript's output. A *shared* glossary lives in the
app directory under a name the user chose, and any number of jobs can be
pointed at it: each run reads the accumulated terms before cleanup and writes
back whatever new ones its own transcript turned up. Point every call with the
same customer, product, or team at one glossary and the vocabulary compounds
instead of being re-learned per recording.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import APP_DIR

GLOSSARY_DIR = APP_DIR / "glossaries"

# Settings/job value meaning "no shared glossary — keep this recording's own".
PER_RECORDING = ""

# Entries whose merge left a disagreement carry this key until someone settles
# it: two sources gave the same term a different type, or the same speaker
# label a different name. See transcriber_studio.glossary_merge.
CONFLICT_KEY = "conflict"

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _slug(name: str) -> str:
    slug = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    return slug or "glossary"


@dataclass
class SharedGlossary:
    """One named glossary in the library."""

    id: str
    name: str
    speakers: list[dict[str, Any]] = field(default_factory=list)
    terms: list[dict[str, Any]] = field(default_factory=list)
    # Which recordings have already contributed, so a re-run of the same job
    # does not re-extract what this glossary already learned from it.
    sources: list[dict[str, str]] = field(default_factory=list)
    created: str = ""
    updated: str = ""

    def payload(self) -> dict[str, list]:
        """The shape the cleanup prompt renderer expects."""
        return {"speakers": list(self.speakers), "terms": list(self.terms)}

    def has_source(self, key: str) -> bool:
        return any(s.get("key") == key for s in self.sources)

    def record_source(self, key: str, name: str) -> None:
        for entry in self.sources:
            if entry.get("key") == key:
                entry["name"] = name
                entry["updated"] = _now()
                return
        self.sources.append({"key": key, "name": name, "updated": _now()})

    def conflict_count(self) -> int:
        """Entries still tagged as a disagreement between two sources."""
        return sum(
            1
            for entry in (*self.terms, *self.speakers)
            if entry.get(CONFLICT_KEY)
        )

    def summary(self) -> str:
        parts = [
            f"{len(self.terms)} term(s)",
            f"{len(self.speakers)} speaker(s)",
            f"{len(self.sources)} recording(s)",
        ]
        conflicts = self.conflict_count()
        if conflicts:
            parts.append(f"⚠ {conflicts} to resolve")
        return ", ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "speakers": self.speakers,
            "terms": self.terms,
            "sources": self.sources,
            "created": self.created,
            "updated": self.updated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, fallback_id: str = "") -> SharedGlossary:
        return cls(
            id=str(data.get("id") or fallback_id),
            name=str(data.get("name") or data.get("id") or fallback_id),
            speakers=list(data.get("speakers") or []),
            terms=list(data.get("terms") or []),
            sources=list(data.get("sources") or []),
            created=str(data.get("created") or ""),
            updated=str(data.get("updated") or ""),
        )


def path_for(gid: str) -> Path:
    return GLOSSARY_DIR / f"{gid}.json"


def list_glossaries() -> list[SharedGlossary]:
    """Every glossary in the library, by name. Unreadable files are skipped."""
    if not GLOSSARY_DIR.exists():
        return []
    found: list[SharedGlossary] = []
    for path in sorted(GLOSSARY_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        found.append(SharedGlossary.from_dict(data, fallback_id=path.stem))
    return sorted(found, key=lambda g: g.name.lower())


def load(gid: str) -> SharedGlossary | None:
    if not gid:
        return None
    path = path_for(gid)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return SharedGlossary.from_dict(data, fallback_id=gid)


def save(glossary: SharedGlossary) -> SharedGlossary:
    GLOSSARY_DIR.mkdir(parents=True, exist_ok=True)
    glossary.created = glossary.created or _now()
    glossary.updated = _now()
    path_for(glossary.id).write_text(
        json.dumps(glossary.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return glossary


def _unique_id(name: str) -> str:
    base = _slug(name)
    gid = base
    n = 2
    while path_for(gid).exists():
        gid = f"{base}-{n}"
        n += 1
    return gid


def create(name: str, *, speakers: list | None = None, terms: list | None = None) -> SharedGlossary:
    """Add a glossary to the library. Names may repeat; ids never do."""
    name = name.strip() or "Untitled glossary"
    return save(
        SharedGlossary(
            id=_unique_id(name),
            name=name,
            speakers=list(speakers or []),
            terms=list(terms or []),
        )
    )


def duplicate(gid: str, new_name: str = "") -> SharedGlossary | None:
    source = load(gid)
    if source is None:
        return None
    return create(
        new_name or f"{source.name} copy",
        speakers=source.speakers,
        terms=source.terms,
    )


def delete(gid: str) -> bool:
    path = path_for(gid)
    if not path.exists():
        return False
    path.unlink()
    return True


def display_name(gid: str) -> str:
    """Name for a stored id, or "" when it no longer exists."""
    glossary = load(gid)
    return glossary.name if glossary else ""


def export_to(gid: str, dest: Path | str) -> Path:
    glossary = load(gid)
    if glossary is None:
        raise FileNotFoundError(f"No shared glossary with id {gid!r}.")
    dest = Path(dest)
    dest.write_text(
        json.dumps(glossary.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return dest


def read_payload(src: Path | str) -> dict[str, list]:
    """Read a glossary JSON file — a shared export or a per-recording one."""
    data = json.loads(Path(src).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("A glossary file must contain a JSON object.")
    return {
        "name": str(data.get("name") or Path(src).stem),
        "speakers": list(data.get("speakers") or []),
        "terms": list(data.get("terms") or []),
    }


def import_from(src: Path | str, name: str = "") -> SharedGlossary:
    """Add a glossary file to the library as a new glossary."""
    payload = read_payload(src)
    return create(
        name or str(payload["name"]),
        speakers=payload["speakers"],
        terms=payload["terms"],
    )
