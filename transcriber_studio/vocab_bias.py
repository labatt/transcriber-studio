# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Telling the decoder which words to expect.

A speech model decoding mumbled or noisy audio is leaning hard on its language
prior — that is where "NorthGate" becomes "north gate" and a surname becomes a
common noun that sounds like it. Whisper takes a prompt, so the prior can be
corrected: hand it the names, products, and jargon this recording is going to
contain and the decoder stops guessing at them.

The words come from the shared glossary the job is pointed at (which is exactly
the vocabulary earlier recordings in that account already taught it), from the
recording's own glossary if it has been transcribed before, and from whatever
was typed into the Options panel.

There is a hard budget: faster-whisper truncates a prompt longer than half the
decoder context, and a truncated list drops its tail silently. Better to spend
the budget deliberately, on the terms most likely to be got wrong.
"""

from __future__ import annotations

import re

from . import glossary_store
from .config import Settings

#: Whisper's prompt window is 448 tokens and faster-whisper keeps at most half
#: for the prompt. At roughly four characters per token that is ~880 characters;
#: the default budget in Settings stays under it with room for the separator.
HARD_CHAR_CEILING = 850

#: A term the model was never going to get wrong is a term wasting the budget.
_COMMON = {
    "the", "and", "for", "you", "your", "our", "with", "this", "that", "team",
    "call", "meeting", "project", "company", "customer", "client", "product",
    "other", "people", "person", "thing", "stuff", "okay", "yeah",
}
_GENERIC_LABEL = re.compile(
    r"^(speaker|spk|voice|participant|channel|unknown)[\s_\-]*[0-9a-z]{0,3}$",
    re.IGNORECASE,
)


def split_terms(text: str) -> list[str]:
    """Free text from the UI: commas, semicolons, or one term per line."""
    return [part.strip() for part in re.split(r"[,;\n]+", text or "") if part.strip()]


def _worth_biasing(term: str) -> bool:
    term = term.strip()
    if len(term) < 2 or _GENERIC_LABEL.match(term):
        return False
    return term.casefold() not in _COMMON


def collect_terms(
    settings: Settings,
    *,
    glossary_id: str | None = None,
    extra_payloads: list[dict] | None = None,
) -> list[str]:
    """Every term worth biasing, best first, deduped case-insensitively.

    Order is the priority order for the budget: what the user typed by hand
    beats what a model extracted, and a name beats a piece of jargon.
    """
    typed = split_terms(settings.bias_extra_terms)
    people: list[str] = []
    other: list[str] = []

    payloads: list[dict] = list(extra_payloads or [])
    gid = settings.glossary_shared_id if glossary_id is None else (glossary_id or "")
    shared = glossary_store.load(gid) if gid else None
    if shared is not None:
        payloads.append(shared.payload())

    for payload in payloads:
        for speaker in payload.get("speakers") or []:
            name = str(speaker.get("name") or "").strip()
            if name:
                people.append(name)
        for term in payload.get("terms") or []:
            canonical = str(term.get("canonical") or "").strip()
            if not canonical:
                continue
            if str(term.get("type") or "").lower() == "person":
                people.append(canonical)
            else:
                other.append(canonical)

    ordered = [*typed, *people, *other]
    seen: dict[str, str] = {}
    for term in ordered:
        if _worth_biasing(term):
            seen.setdefault(term.casefold(), term.strip())
    return list(seen.values())


def build(terms: list[str], max_chars: int) -> str:
    """A comma-separated vocabulary list that fits the budget.

    Whisper is prompted with text, not a word list, so the terms are joined the
    way they would appear in a sentence. Terms are dropped from the tail, which
    is why the caller orders them by how much the bias is worth.
    """
    budget = max(0, min(int(max_chars or 0), HARD_CHAR_CEILING))
    if budget <= 0:
        return ""
    kept: list[str] = []
    length = 0
    for term in terms:
        addition = len(term) + (2 if kept else 0)
        if length + addition > budget:
            continue        # a long term late in the list should not evict short ones
        kept.append(term)
        length += addition
    return ", ".join(kept)


def hotwords(
    settings: Settings,
    *,
    glossary_id: str | None = None,
    extra_payloads: list[dict] | None = None,
) -> str:
    """The biasing string for this run, or "" when there is nothing to say."""
    if not settings.bias_enabled:
        return ""
    return build(
        collect_terms(settings, glossary_id=glossary_id, extra_payloads=extra_payloads),
        settings.bias_max_chars,
    )


def summarize(terms: list[str], prompt: str) -> str:
    """One line for the log: what went in, and what did not fit."""
    if not prompt:
        return "Vocabulary biasing: nothing to bias with."
    used = len([t for t in prompt.split(", ") if t])
    dropped = len(terms) - used
    line = f"Vocabulary biasing: {used} term(s), {len(prompt)} chars"
    if dropped > 0:
        line += f" — {dropped} did not fit the budget"
    return line + f" ({prompt[:80]}{'…' if len(prompt) > 80 else ''})"
