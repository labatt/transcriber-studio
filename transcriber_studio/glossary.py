# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Glossary extraction, merge, persistence, and prompt rendering for AI cleanup."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from . import ai_providers, filename_builder, glossary_store
from .ai_store import load_profile, save_profile, suggest_profile_fix
from .config import Settings
from .glossary_store import CONFLICT_KEY
from .job_cancel import JobCancelled, ShouldCancel, check_cancel
from .models import Segment, TranscriptResult
from .resume import ResumeLog, send_key

CHARS_PER_TOKEN = 3.5
MAX_RETRIES = 4

EXTRACTION_SYSTEM_PROMPT = """You extract a glossary from a raw speech-to-text transcript of a multi-person
conversation. The transcript is noisy: names, product names, company names, and
jargon are often mis-transcribed and appear in several spellings.

Read the entire input, then produce two lists.

SPEAKERS: for each distinct speaker label, resolve the real name and role from
self-introductions and consistent context. If a name was clearly stated but came
out garbled, record the garbled text in raw_intro, set name to null, and mark
confidence low. Never replace a garbled name with a plausible-sounding real name.

TERMS: recurring proper nouns, product names, company names, acronyms, and domain
jargon. For each, choose the canonical spelling (the one that recurs most, or that
matches a known real-world term) and list the garbled variants you saw. Exclude
common English words. Do not invent anything that does not appear in the input.

Return ONLY one JSON object, no markdown, no code fences:
{"speakers":[{"label":"Greg","name":"Gregory Jackson","role":"","confidence":"high","raw_intro":""}],"terms":[{"canonical":"GrowthMark","variants":["growth mark","growth market"],"type":"product"}]}
- name is null when unresolved; confidence is high, medium, or low.
- type is one of person, company, product, acronym, concept, other.
- Output nothing except the JSON object."""

EMPTY_GLOSSARY: dict[str, list] = {"speakers": [], "terms": []}

CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1, "": 0}

# "SPEAKER_00" and friends are positions in one recording, not people.
GENERIC_SPEAKER_LABEL = re.compile(r"(speaker|spk|channel)[\s_\-]*\d+", re.IGNORECASE)


def recording_fingerprint(recording) -> str:
    """Stable short id so each recording gets its own glossary file."""
    return hashlib.sha256(recording.id.encode("utf-8")).hexdigest()[:12]


def glossary_path(settings: Settings, result: TranscriptResult) -> Path:
    """One glossary file per recording, not per queue row or cleanup run."""
    values = filename_builder.build_values(result, index=1, sanitize_names=settings.sanitize_names)
    stem = filename_builder.render(settings.filename_template, values, settings.sanitize_names)
    tag = recording_fingerprint(result.recording)
    return Path(settings.output_dir) / f"{stem}__{tag}.glossary.json"


def legacy_glossary_path(settings: Settings, result: TranscriptResult) -> Path:
    """Pre-per-recording path (stem only); used when loading older glossary files."""
    values = filename_builder.build_values(result, index=1, sanitize_names=settings.sanitize_names)
    stem = filename_builder.render(settings.filename_template, values, settings.sanitize_names)
    return Path(settings.output_dir) / f"{stem}.glossary.json"


def load_glossary(path: Path) -> dict[str, list]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "speakers": list(data.get("speakers") or []),
        "terms": list(data.get("terms") or []),
    }


def save_glossary(path: Path, glossary: dict[str, list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(glossary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def resolve_glossary(
    result: TranscriptResult,
    settings: Settings,
    *,
    provider: str,
    model: str,
    glossary_id: str | None = None,
    log_cb: Callable[[str], None] | None = None,
    progress_cb: Callable[[float], None] | None = None,
    should_cancel: ShouldCancel = None,
    resume: ResumeLog | None = None,
) -> dict[str, list]:
    """The roster and terms this cleanup run should work from.

    ``glossary_id`` names a glossary in the shared library that this job reads
    from and writes back to; None falls back to the app-wide default in
    settings, and "" keeps the glossary private to this recording.
    """
    shared = _shared_glossary(settings, glossary_id, log_cb)

    if not settings.glossary_enabled:
        if shared is not None:
            if log_cb:
                log_cb(
                    f"Glossary: extraction off — using shared glossary "
                    f"'{shared.name}' as-is ({shared.summary()})."
                )
            if progress_cb:
                progress_cb(1.0)
            return shared.payload()
        if log_cb:
            log_cb("Glossary: disabled — cleanup will use batch context only.")
        return dict(EMPTY_GLOSSARY)

    own = _recording_glossary(
        result,
        settings,
        provider=provider,
        model=model,
        log_cb=log_cb,
        progress_cb=progress_cb,
        should_cancel=should_cancel,
        resume=resume,
    )
    if shared is None:
        return own
    return contribute_to_shared(shared, own, result, log_cb=log_cb)


def _shared_glossary(
    settings: Settings,
    glossary_id: str | None,
    log_cb: Callable[[str], None] | None = None,
) -> glossary_store.SharedGlossary | None:
    gid = (settings.glossary_shared_id if glossary_id is None else glossary_id) or ""
    if not gid:
        return None
    shared = glossary_store.load(gid)
    if shared is None and log_cb:
        log_cb(
            f"Glossary: shared glossary '{gid}' is gone — "
            f"falling back to this recording's own."
        )
    return shared


def contribute_to_shared(
    shared: glossary_store.SharedGlossary,
    own: dict[str, list],
    result: TranscriptResult,
    *,
    log_cb: Callable[[str], None] | None = None,
) -> dict[str, list]:
    """Fold this recording's findings into the shared glossary and read it back.

    Terms are shared; the speaker roster is not. A diarization label
    ("SPEAKER_00") points at a different person in every recording, so pushing
    one recording's roster into a shared glossary would mislabel the next one.
    Names the roster did resolve travel across as person terms instead, which is
    what a later recording can actually use.
    """
    before = len(shared.terms)
    shared.terms = merge_terms(
        [shared.terms, own.get("terms") or [], speakers_as_terms(own.get("speakers") or [])]
    )
    shared.record_source(
        recording_fingerprint(result.recording), result.recording.display_name
    )
    try:
        glossary_store.save(shared)
        if log_cb:
            log_cb(
                f"Glossary: shared '{shared.name}' updated — "
                f"{len(shared.terms) - before} new term(s), {len(shared.terms)} total."
            )
    except Exception as e:
        if log_cb:
            log_cb(f"Glossary: could not write shared glossary '{shared.name}' ({e}).")

    merged = {
        "speakers": merge_speakers([shared.speakers, own.get("speakers") or []]),
        "terms": shared.terms,
    }
    if log_cb:
        log_cb(
            f"Glossary: cleanup will use shared '{shared.name}' — "
            f"{len(merged['speakers'])} speaker(s), {len(merged['terms'])} term(s)."
        )
    return merged


def speakers_as_terms(speakers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolved speaker names as person terms, so they survive the label change."""
    terms: list[dict[str, Any]] = []
    for entry in speakers:
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        variants = set()
        raw_intro = str(entry.get("raw_intro") or "").strip()
        if raw_intro and _term_key(raw_intro) != _term_key(name):
            variants.add(raw_intro)
        label = str(entry.get("label") or "").strip()
        if (
            label
            and not GENERIC_SPEAKER_LABEL.fullmatch(label)
            and _term_key(label) != _term_key(name)
        ):
            variants.add(label)
        terms.append({"canonical": name, "variants": sorted(variants), "type": "person"})
    return terms


def _recording_glossary(
    result: TranscriptResult,
    settings: Settings,
    *,
    provider: str,
    model: str,
    log_cb: Callable[[str], None] | None = None,
    progress_cb: Callable[[float], None] | None = None,
    should_cancel: ShouldCancel = None,
    resume: ResumeLog | None = None,
) -> dict[str, list]:
    """This recording's own glossary: loaded from its file, or extracted now."""
    path = glossary_path(settings, result)
    if log_cb:
        log_cb(f"Glossary: file {path.name}")

    if path.exists() and not settings.force_reextract:
        try:
            glossary = load_glossary(path)
            if log_cb:
                log_cb(
                    f"Glossary: loaded existing file — "
                    f"{len(glossary['speakers'])} speaker(s), {len(glossary['terms'])} term(s)"
                )
            if progress_cb:
                progress_cb(1.0)
            return glossary
        except Exception as e:
            if log_cb:
                log_cb(f"Glossary: could not load {path.name} ({e}) — re-extracting…")
    elif settings.force_reextract and log_cb:
        log_cb("Glossary: force re-extract enabled.")
    elif not settings.force_reextract:
        legacy = legacy_glossary_path(settings, result)
        if legacy.exists():
            try:
                glossary = load_glossary(legacy)
                save_glossary(path, glossary)
                if log_cb:
                    log_cb(
                        f"Glossary: migrated legacy file — "
                        f"{len(glossary['speakers'])} speaker(s), {len(glossary['terms'])} term(s)"
                    )
                if progress_cb:
                    progress_cb(1.0)
                return glossary
            except Exception as e:
                if log_cb:
                    log_cb(f"Glossary: could not load legacy {legacy.name} ({e}) — re-extracting…")

    try:
        if log_cb:
            log_cb("Glossary: extracting from full transcript…")
        glossary = extract_glossary(
            result.segments,
            settings,
            provider=provider,
            model=model,
            cache_key=f"glossary-{recording_fingerprint(result.recording)}-{provider}-{model}",
            log_cb=log_cb,
            progress_cb=progress_cb,
            should_cancel=should_cancel,
            resume=resume,
        )
        save_glossary(path, glossary)
        if log_cb:
            log_cb(
                f"Glossary: saved {path.name} — "
                f"{len(glossary['speakers'])} speaker(s), {len(glossary['terms'])} term(s)"
            )
        return glossary
    except JobCancelled:
        raise
    except Exception as e:
        if log_cb:
            log_cb(f"Glossary extraction failed ({e}) — continuing without glossary.")
        return dict(EMPTY_GLOSSARY)


def extract_glossary(
    segments: list[Segment],
    settings: Settings,
    *,
    provider: str,
    model: str,
    cache_key: str | None = None,
    log_cb: Callable[[str], None] | None = None,
    progress_cb: Callable[[float], None] | None = None,
    should_cancel: ShouldCancel = None,
    resume: ResumeLog | None = None,
) -> dict[str, list]:
    glossary_model = (settings.glossary_model or model).strip()
    if not glossary_model:
        raise RuntimeError("No glossary model configured.")

    text = transcript_text_for_extraction(segments)
    if not text.strip():
        if log_cb:
            log_cb("Glossary: transcript empty — skipping extraction.")
        return dict(EMPTY_GLOSSARY)

    if log_cb:
        log_cb(
            f"Glossary: model {glossary_model} — "
            f"{len(segments)} segment(s), ~{len(text):,} chars"
        )
        if settings.prompt_cache_enabled and ai_providers.prompt_cache_mode(provider):
            log_cb(
                f"Glossary: prompt caching enabled "
                f"({ai_providers.prompt_cache_mode(provider)})"
            )

    chunks = chunk_text_for_extraction(text, settings.glossary_chunk_token_threshold)
    total = len(chunks)
    if log_cb:
        if total > 1:
            log_cb(f"Glossary: transcript large — {total} extraction chunk(s)")
        else:
            log_cb("Glossary: single-pass extraction")

    partials: list[dict[str, list]] = []
    for i, chunk in enumerate(chunks):
        check_cancel(should_cancel, log_cb, message="Glossary: cancelled.")
        if log_cb:
            log_cb(f"Glossary: chunk {i + 1}/{total} — contacting model…")
        partial = _extract_glossary_chunk(
            chunk,
            settings,
            provider=provider,
            model=glossary_model,
            cache_key=cache_key,
            log_cb=log_cb,
            resume=resume,
        )
        partials.append(partial)
        if log_cb:
            log_cb(
                f"Glossary: chunk {i + 1}/{total} done — "
                f"{len(partial['speakers'])} speaker(s), {len(partial['terms'])} term(s)"
            )
        if progress_cb:
            progress_cb((i + 1) / total)

    merged = merge_glossaries(partials)
    if log_cb and total > 1:
        log_cb(
            f"Glossary: merged {total} chunk(s) — "
            f"{len(merged['speakers'])} speaker(s), {len(merged['terms'])} term(s)"
        )
    return merged


def transcript_text_for_extraction(segments: list[Segment]) -> str:
    lines: list[str] = []
    for seg in segments:
        speaker = seg.speaker or "Unknown"
        text = seg.text.strip()
        if text:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def chunk_text_for_extraction(text: str, token_threshold: int) -> list[str]:
    if token_threshold <= 0:
        return [text]
    est_tokens = max(1, int(len(text) / CHARS_PER_TOKEN))
    if est_tokens <= token_threshold:
        return [text]
    chars_per_chunk = max(1, int(token_threshold * CHARS_PER_TOKEN))
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chars_per_chunk)
        if end < len(text):
            break_at = text.rfind("\n", start, end)
            if break_at > start:
                end = break_at + 1
        chunks.append(text[start:end])
        start = end
    return chunks or [text]


def _output_token_ceiling(provider: str, model: str) -> int:
    m = model.lower()
    if provider == "anthropic":
        if "opus" in m:
            return 32_000
        if re.search(r"sonnet-([45]|4[\-.])", m) or "sonnet-5" in m:
            return 64_000
        return 8_192
    if provider in ("openai", "openrouter", "grok"):
        return 16_384
    if provider in ("ollama_cloud", "ollama_local"):
        return 4_096
    return 8_192


def merge_glossaries(partials: list[dict[str, list]]) -> dict[str, list]:
    speaker_lists = [p.get("speakers", []) for p in partials]
    term_lists = [p.get("terms", []) for p in partials]
    return {
        "speakers": merge_speakers(speaker_lists),
        "terms": merge_terms(term_lists),
    }


def _term_key(canonical: str) -> str:
    return re.sub(r"[\s\-_]+", "", canonical).lower()


def _dedupe_variants(variants: Iterable[str], canonical: str) -> list[str]:
    """Distinct spellings worth showing the model, the canonical itself aside.

    Only a case-insensitive repeat of the canonical is dropped. Spacing and
    hyphen differences are exactly what a segmenter gets wrong, so "growth
    mark" earns its place under "GrowthMark" even though the two collapse to
    the same key for bucketing.
    """
    kept: dict[str, str] = {}       # folded spelling -> the first one seen
    skip = canonical.strip().casefold()
    for variant in variants:
        variant = str(variant).strip()
        folded = variant.casefold()
        if not variant or folded == skip:
            continue
        kept.setdefault(folded, variant)
    return sorted(kept.values(), key=str.lower)


def merge_terms(term_lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for terms in term_lists:
        for term in terms:
            canonical = str(term.get("canonical", "")).strip()
            if not canonical:
                continue
            key = _term_key(canonical)
            if key not in buckets:
                buckets[key] = {
                    "canonical_counts": Counter(),
                    # A list, not a set: which of two spellings that differ
                    # only in case is kept has to be the same on every run.
                    "variants": [],
                    "type_counts": Counter(),
                }
            bucket = buckets[key]
            bucket["canonical_counts"][canonical] += 1
            bucket["type_counts"][str(term.get("type") or "other")] += 1
            # An unresolved conflict tag outlives the merges that follow it:
            # dropping it here would quietly un-flag a term nobody has fixed.
            if not bucket.get(CONFLICT_KEY) and term.get(CONFLICT_KEY):
                bucket[CONFLICT_KEY] = term[CONFLICT_KEY]
            for variant in term.get("variants") or []:
                variant = str(variant).strip()
                if variant and variant not in bucket["variants"]:
                    bucket["variants"].append(variant)

    merged: list[dict[str, Any]] = []
    for bucket in buckets.values():
        canonical = bucket["canonical_counts"].most_common(1)[0][0]
        term_type = bucket["type_counts"].most_common(1)[0][0]
        variants = _dedupe_variants(
            (*bucket["variants"], *bucket["canonical_counts"]), canonical
        )
        entry = {
            "canonical": canonical,
            "variants": variants,
            "type": term_type,
        }
        if bucket.get(CONFLICT_KEY):
            entry[CONFLICT_KEY] = bucket[CONFLICT_KEY]
        merged.append(entry)
    return sorted(merged, key=lambda item: item["canonical"].lower())


def _speaker_score(entry: dict[str, Any]) -> tuple[int, int]:
    has_name = 1 if entry.get("name") else 0
    conf = CONFIDENCE_RANK.get(str(entry.get("confidence") or "").lower(), 0)
    return has_name, conf


def merge_speakers(speaker_lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    by_label: dict[str, dict[str, Any]] = {}
    for speakers in speaker_lists:
        for entry in speakers:
            label = str(entry.get("label", "")).strip()
            if not label:
                continue
            normalized = {
                "label": label,
                "name": entry.get("name"),
                "role": str(entry.get("role") or ""),
                "confidence": str(entry.get("confidence") or ""),
                "raw_intro": str(entry.get("raw_intro") or ""),
            }
            if entry.get(CONFLICT_KEY):
                normalized[CONFLICT_KEY] = entry[CONFLICT_KEY]
            if label not in by_label:
                by_label[label] = normalized
                continue
            current = by_label[label]
            if _speaker_score(normalized) > _speaker_score(current):
                winner, loser = normalized, current
            else:
                winner, loser = current, normalized
            merged = dict(winner)
            if not merged.get("raw_intro") and loser.get("raw_intro"):
                merged["raw_intro"] = loser["raw_intro"]
            if not merged.get(CONFLICT_KEY) and loser.get(CONFLICT_KEY):
                merged[CONFLICT_KEY] = loser[CONFLICT_KEY]
            by_label[label] = merged
    return sorted(by_label.values(), key=lambda item: item["label"].lower())


def render_glossary_for_prompt(glossary: dict[str, list]) -> str:
    speakers = glossary.get("speakers") or []
    terms = glossary.get("terms") or []
    if not speakers and not terms:
        return ""

    parts: list[str] = []
    if speakers:
        roster_lines: list[str] = []
        for sp in speakers:
            label = sp.get("label", "")
            name = sp.get("name")
            role = str(sp.get("role") or "").strip()
            conf = str(sp.get("confidence") or "").strip()
            raw_intro = str(sp.get("raw_intro") or "").strip()
            if name:
                line = f"{label} -> {name}"
            else:
                line = f"{label} -> unresolved"
                if raw_intro:
                    line += f", raw intro: {raw_intro}"
            if role:
                line += f", role: {role}"
            if conf:
                line += f" [{conf} confidence]"
            roster_lines.append(line)
        parts.append("SPEAKER ROSTER:\n" + "\n".join(roster_lines))

    if terms:
        term_lines: list[str] = []
        for term in terms:
            canonical = term.get("canonical", "")
            term_type = str(term.get("type") or "other")
            variants = [str(v) for v in (term.get("variants") or []) if str(v).strip()]
            if variants:
                term_lines.append(
                    f"{canonical} ({term_type}): {', '.join(variants)}"
                )
            else:
                term_lines.append(f"{canonical} ({term_type})")
        parts.append("GLOSSARY:\n" + "\n".join(term_lines))

    return "\n\n".join(parts) + "\n\nUse the roster and glossary above when cleaning the transcript below.\n\n"


def _extract_glossary_chunk(
    text: str,
    settings: Settings,
    *,
    provider: str,
    model: str,
    cache_key: str | None = None,
    log_cb: Callable[[str], None] | None = None,
    resume: ResumeLog | None = None,
) -> dict[str, list]:
    profile = load_profile(provider, model).with_changes(
        temperature=settings.glossary_temperature
    )
    token_ceiling = min(_output_token_ceiling(provider, model), 16_384)
    last_error = ""
    current_profile = profile

    key = send_key(
        "glossary", provider, model, EXTRACTION_SYSTEM_PROMPT,
        str(settings.glossary_temperature), text,
    )
    saved = resume.get(key) if resume else None
    if saved:
        try:
            parsed = _parse_glossary_response(saved)
            if log_cb:
                log_cb("Glossary: chunk restored from an earlier run — no tokens spent.")
            return parsed
        except Exception as e:
            if log_cb:
                log_cb(f"Glossary: saved response unusable ({e}) — re-sending.")

    def _usage_cb(usage: dict[str, int]) -> None:
        line = ai_providers.format_prompt_cache_usage(usage)
        if line and log_cb:
            log_cb(f"Glossary: {line}")

    for _attempt in range(MAX_RETRIES):
        try:
            api_profile = current_profile
            if current_profile.max_tokens < token_ceiling:
                api_profile = current_profile.with_changes(max_tokens=token_ceiling)
            raw = ai_providers.chat_completion(
                settings,
                provider,
                model,
                EXTRACTION_SYSTEM_PROMPT,
                text,
                api_profile,
                cache_key=cache_key,
                use_prompt_cache=settings.prompt_cache_enabled,
                usage_cb=_usage_cb if log_cb else None,
            )
            parsed = _parse_glossary_response(raw)
            if resume:
                resume.record(
                    key, raw, stage="glossary", chars=len(text),
                    model=f"{provider}/{model}",
                )
            save_profile(api_profile)
            return parsed
        except Exception as e:
            last_error = str(e)
            fix = suggest_profile_fix(last_error, current_profile, provider=provider)
            if fix and fix != current_profile:
                current_profile = fix
                continue
            raise RuntimeError(f"Glossary extraction failed: {last_error}") from e
    raise RuntimeError(f"Glossary extraction failed after retries: {last_error}")


def _parse_glossary_response(raw: str) -> dict[str, list]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("Glossary model returned empty response.")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Glossary response must be a JSON object.")
    return {
        "speakers": list(data.get("speakers") or []),
        "terms": list(data.get("terms") or []),
    }
