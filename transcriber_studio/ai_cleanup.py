# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Post-transcription cleanup via an LLM."""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

from . import ai_providers, config
from .ai_store import ModelProfile, load_profile, save_profile, suggest_profile_fix
from .config import Settings
from .glossary import (
    EMPTY_GLOSSARY,
    recording_fingerprint,
    render_glossary_for_prompt,
    resolve_glossary,
)
from .job_cancel import JobCancelled, ShouldCancel, check_cancel, sleep_cancellable
from .models import Segment, TranscriptResult
from .resume import DECODE_STAGE, TRANSCRIPT_STAGE, ResumeLog, log_for, send_key

SYSTEM_PROMPT = """You clean up raw speech-to-text transcripts produced by a Whisper-style segmenter, where each input segment is usually one short phrase or clause tagged with a speaker label and a start time.

Your job is to turn these fragments into a readable, accurate transcript: complete sentences, correct speaker attribution, filler removed, and any spot that does not make sense clearly flagged. You reorganize and lightly correct. You never summarize, paraphrase away detail, or invent content.

INPUT
- A list of segments, indexed from 0. Each has a speaker label and text. The index positions are authoritative.
- Usually a SPEAKER ROSTER (label -> real name and/or role) and a GLOSSARY (canonical term -> the garbled variants seen elsewhere in the transcript), both extracted from the full transcript ahead of time. Treat these as authoritative. They may occasionally be empty or missing; if so, fall back to context and self-introductions within this batch, and do not fail.

USING THE ROSTER AND GLOSSARY
- Roster: when a segment's label is wrong and the roster or a self-introduction shows who is actually speaking, correct it. A segment whose text names a different person than its label is a red flag, not a fact.
- Glossary: when the text contains a variant spelling listed under a canonical term, normalize it to the canonical spelling. Apply this consistently across the whole batch.
- The glossary is your safety net for terms introduced elsewhere in the meeting. Trust it over your own guess when they disagree.

CORE CLEANUP
1. Merge adjacent segments from the SAME speaker into full sentences with proper capitalization and punctuation. One sentence may span many segments. Start a new sentence at a real boundary, not at every line break.
2. Remove fillers (um, uh, er) and empty discourse crutches (like, you know, I mean, sort of) only when they carry no meaning. Keep them when they change tone or meaning.
3. Collapse false starts, stutters, and verbatim repetition into clean phrasing (five repeats of the same clause become one). Preserve repetition only when it reads as deliberate emphasis; if unsure, collapse and add a note.
4. Fix obvious transcription errors (wrong homophone, split or fused words, mangled term) using the glossary, context, and internal consistency. When a probable proper noun is NOT in the glossary and cannot be resolved from context, ESPECIALLY a person's name, keep the raw text and flag it. Never replace it with a plausible-sounding guess.
5. Preserve all substantive content exactly: figures, money, percentages, dates, names, jargon. Do not round, restate, or improve numbers. Do not soften, euphemize, or censor wording, including profanity. This is an internal working transcript.

SPEAKERS
6. Resolve a speaker for EVERY output sentence and put it in the "speaker" field, using the roster label. Correct a label when the roster or strong context shows the segmenter got it wrong.
7. Never merge two different speakers into one sentence.
8. Handle boundary bleed: segmenters often glue the tail of one speaker's utterance onto the next speaker's segment, or the reverse. Assign each index to the speaker who actually says most of it. If a short stray fragment (a word or two) at the start or end clearly belongs to the neighboring speaker, drop those words from this sentence's text (the index stays assigned here) and add a note. Do not force it in, and do not drop anything substantive.
9. Non-human sources (a played "Demo Video", a voicemail, an announcement) keep their own label. Clean their text lightly but never attribute it to a person in the room.

FLAGGING PROBLEMS
10. When a passage does not make sense, is likely mis-transcribed, has an uncertain speaker, or reads as garbled, add a short "note" on that segment stating the problem plainly. Set "confidence" to "medium" or "low" for any sentence you are not confident reads correctly. Do not annotate clean, obvious sentences.
11. If the first or last segment of the batch looks like a sentence fragment continuing from a previous batch or into the next one, keep it as its own segment, do not fabricate the missing half, and flag it so it can be stitched later.

OUTPUT
Return ONLY one JSON object. No markdown, no code fences, no commentary:
{"segments":[{"from_index":0,"to_index":4,"speaker":"Greg","text":"Complete sentence here.","confidence":"low","note":"garbled name not in glossary, kept raw"}]}
- from_index and to_index are inclusive indices into the input list.
- Cover every index from 0 through the last exactly once: ascending, contiguous, no gaps, no overlaps.
- "speaker" is required on every segment (the resolved roster label).
- "text" is a single line of prose: never put a line break inside it.
- "confidence" is optional; include only when not high ("medium" or "low").
- "note" is optional; include only when there is a real problem to flag.
- Do not include timestamps. Output nothing except the JSON object."""

CLEANUP_USER_INSTRUCTIONS = (
    "Return ONLY the JSON object described in the system prompt.\n"
    '{"segments":[{"from_index":0,"to_index":4,"speaker":"Greg","text":"Complete sentence here.",'
    '"confidence":"low","note":"garbled name not in glossary, kept raw"}]}\n\n'
)

MAX_RETRIES = 5
CHARS_PER_TOKEN = 3.5
OUTPUT_SAFETY = 0.80
JSON_OVERHEAD_PER_SEGMENT = 32
INPUT_OVERHEAD_PER_SEGMENT = 45
ABSOLUTE_MAX_SEGMENTS_PER_BATCH = 3_000
RELIABLE_CLEANUP_BATCH_SEGMENTS = 600
CLEANUP_READ_TIMEOUT_SEC = 600
NETWORK_RETRIES_BEFORE_SPLIT = 2

_TRANSIENT_NETWORK_MARKERS = (
    "connection aborted",
    "connection reset",
    "connectionreseterror",
    "forcibly closed",
    "timed out",
    "timeout",
    "read timed out",
    "remote end closed",
    "broken pipe",
    "502",
    "503",
    "504",
    "429",
    "rate limit",
)


@dataclass(frozen=True)
class ChunkBudget:
    """Per-batch limits derived from model output capacity."""
    max_output_chars: int
    max_input_chars: int
    max_segments: int
    output_token_ceiling: int

    @property
    def summary(self) -> str:
        return (
            f"~{self.output_token_ceiling:,} output tokens "
            f"(~{self.max_output_chars // 1000}k chars/batch)"
        )


def cleanup_transcript(
    result: TranscriptResult,
    settings: Settings,
    *,
    provider: str | None = None,
    model: str | None = None,
    index: int = 1,
    glossary_id: str | None = None,
    progress_cb=None,
    log_cb=None,
    should_cancel: ShouldCancel = None,
    resume: ResumeLog | None = None,
) -> TranscriptResult:
    provider = (provider or config.cleanup_provider(settings)).strip()
    model = (model or config.cleanup_model(settings)).strip()
    if not provider or not model:
        raise RuntimeError("No AI provider/model selected for cleanup.")
    if not ai_providers.is_provider_configured(settings, provider):
        raise RuntimeError(f"AI provider '{provider}' is not configured in Settings.")

    check_cancel(should_cancel, log_cb, message="AI Cleanup: cancelled.")

    if log_cb:
        log_cb(
            f"AI Cleanup: {ai_providers.PROVIDER_LABELS.get(provider, provider)} / {model}"
        )

    if log_cb:
        log_cb("AI Cleanup: verifying provider connection…")
    try:
        ai_providers.list_models(settings, provider)
        if log_cb:
            log_cb("AI Cleanup: provider connection OK.")
    except Exception as exc:
        if log_cb:
            log_cb(f"AI Cleanup: model list skipped ({exc}) — continuing anyway.")

    profile = load_profile(provider, model)
    resume = resume if resume is not None else log_for(result.recording, log_cb)

    def _glossary_progress(frac: float) -> None:
        if progress_cb:
            progress_cb(0.01 + frac * 0.09)

    if log_cb:
        log_cb("AI Cleanup: glossary stage…")
    glossary = resolve_glossary(
        result,
        settings,
        provider=provider,
        model=model,
        glossary_id=glossary_id,
        log_cb=log_cb,
        progress_cb=_glossary_progress if progress_cb else None,
        should_cancel=should_cancel,
        resume=resume,
    )
    spk_n = len(glossary.get("speakers") or [])
    term_n = len(glossary.get("terms") or [])
    if log_cb:
        if spk_n or term_n:
            log_cb(f"AI Cleanup: glossary ready — {spk_n} speaker(s), {term_n} term(s)")
        else:
            log_cb("AI Cleanup: no glossary — batch context only")
    if progress_cb:
        progress_cb(0.10)

    budget = chunk_budget(provider, model)
    # If an earlier run had to split its batches, chunk at that proven size so
    # this run's boundaries line up with the answers already on disk.
    saved_batch = resume.largest_batch("cleanup")
    chunks = _chunk_segments(result.segments, budget, saved_batch or None)
    if log_cb and saved_batch:
        log_cb(
            f"AI Cleanup: earlier run settled on {saved_batch}-segment batches — "
            f"matching that so its saved work can be reused."
        )
    updated: list[Segment] = []
    total = len(chunks)
    cache_key = (
        f"cleanup-{recording_fingerprint(result.recording)}-{provider}-{model}"
    )
    cacheable_user_prefix = (
        render_glossary_for_prompt(glossary or EMPTY_GLOSSARY) + CLEANUP_USER_INSTRUCTIONS
    )
    if log_cb and settings.prompt_cache_enabled and ai_providers.prompt_cache_mode(provider):
        log_cb(
            f"AI Cleanup: prompt caching enabled "
            f"({ai_providers.prompt_cache_mode(provider)})"
        )

    if log_cb:
        log_cb(
            f"AI Cleanup: cleaning {len(result.segments)} segment(s) in {total} batch(es) "
            f"({budget.summary})"
        )
        already = resume.hits([
            _cleanup_send_key(provider, model, cacheable_user_prefix, chunk)
            for chunk in chunks
        ])
        if already:
            log_cb(
                f"AI Cleanup: resuming — {already}/{total} batch(es) already "
                f"completed in an earlier run (no tokens spent on those)"
            )
    if progress_cb:
        progress_cb(0.11)

    try:
        for i, chunk in enumerate(chunks):
            check_cancel(should_cancel, log_cb, message="AI Cleanup: cancelled.")
            batch_no = i + 1
            if log_cb:
                payload_k = _estimate_cleanup_payload_chars(chunk) // 1000
                log_cb(
                    f"AI Cleanup: batch {batch_no}/{total} — "
                    f"{len(chunk)} segment(s), ~{payload_k}k payload…"
                )
            if progress_cb:
                progress_cb(0.10 + (i / total) * 0.85)
            cleaned = _cleanup_chunk(
                chunk,
                result.speakers,
                settings,
                provider,
                model,
                profile,
                cacheable_user_prefix,
                log_cb,
                cache_key=cache_key,
                should_cancel=should_cancel,
                resume=resume,
            )
            updated.extend(cleaned)
            if log_cb:
                log_cb(
                    f"AI Cleanup: batch {batch_no}/{total} done — "
                    f"{len(chunk)} segment(s) → {len(cleaned)} sentence(s)"
                )
            if progress_cb:
                progress_cb(0.10 + ((i + 1) / total) * 0.85)
    except JobCancelled:
        # Stopping on purpose means starting over next time, so the saved
        # sends go. The transcript stays — re-running Whisper costs time for
        # an identical result.
        dropped = resume.discard(keep_stages=(TRANSCRIPT_STAGE, DECODE_STAGE))
        if log_cb and dropped:
            log_cb(f"AI Cleanup: cancelled — discarded {dropped} saved send(s).")
        raise

    result.segments = updated
    result.speakers = list(dict.fromkeys(s.speaker for s in updated if s.speaker))
    # Every send is now reflected in the result; the transcript entry stays so
    # a failure while exporting still can't cost a re-transcription.
    resume.discard(keep_stages=(TRANSCRIPT_STAGE, DECODE_STAGE))
    if log_cb:
        log_cb(
            f"AI Cleanup: finished — {len(updated)} sentence(s), "
            f"{result.speaker_count} speaker(s)"
        )
    if progress_cb:
        progress_cb(1.0)
    return result


def output_token_ceiling(provider: str, model: str) -> int:
    """Best-effort max output tokens the model/API supports."""
    m = model.lower()
    if provider == "anthropic":
        if "opus" in m:
            return 32_000
        if re.search(r"sonnet-([45]|4[\-.])", m) or "sonnet-5" in m:
            return 64_000
        return 8_192
    if provider == "google":
        return 8_192
    if provider in ("openai", "openrouter", "grok"):
        if any(x in m for x in ("o1", "o3", "o4")):
            return 32_768
        if any(x in m for x in ("gpt-4o", "gpt-4.1", "gpt-5", "gpt-4-")):
            return 16_384
        return 16_384
    if provider in ("ollama_cloud", "ollama_local"):
        return 4_096
    return 8_192


def input_char_ceiling(provider: str, model: str) -> int:
    """Rough per-request input size limit (context minus headroom)."""
    m = model.lower()
    if provider == "anthropic":
        return 180_000
    if provider == "google":
        return 900_000 if "1.5" in m or "2" in m else 120_000
    if provider in ("openai", "openrouter", "grok"):
        if any(x in m for x in ("128k", "200k", "1m")):
            return 500_000
        return 120_000
    if provider in ("ollama_cloud", "ollama_local"):
        # Ollama's window is whatever num_ctx we ask for, so the real limit is
        # the largest one worth allocating on the user's own machine, minus the
        # answer it has to fit alongside the prompt. Anything past this is not
        # rejected, it is silently dropped — so the batch has to stay inside it.
        input_tokens = (
            ai_providers.OLLAMA_MAX_NUM_CTX
            - output_token_ceiling(provider, model)
            - ai_providers.OLLAMA_CTX_HEADROOM_TOKENS
        )
        # chars-per-token is an average, and transcript JSON tokenizes worse
        # than prose. Undershoot deliberately: the penalty for overshooting is
        # silent truncation, and the penalty for undershooting is one more batch.
        return int(input_tokens * ai_providers.OLLAMA_CHARS_PER_TOKEN * OUTPUT_SAFETY)
    return 60_000


def chunk_budget(provider: str, model: str) -> ChunkBudget:
    """Size batches from expected JSON output, not a fixed input char cap."""
    tokens = output_token_ceiling(provider, model)
    max_output_chars = int(tokens * CHARS_PER_TOKEN * OUTPUT_SAFETY)
    max_input_chars = input_char_ceiling(provider, model)
    max_segments = min(
        ABSOLUTE_MAX_SEGMENTS_PER_BATCH,
        max(1, max_output_chars // JSON_OVERHEAD_PER_SEGMENT),
    )
    return ChunkBudget(
        max_output_chars=max_output_chars,
        max_input_chars=max_input_chars,
        max_segments=max_segments,
        output_token_ceiling=tokens,
    )


def _estimate_cleanup_payload_chars(segments: list[Segment]) -> int:
    """Rough JSON payload size sent to the cleanup model."""
    total = 64
    for i, seg in enumerate(segments):
        total += 28 + len(str(i)) + len(seg.text or "") + len(seg.speaker or "")
    return total


def _estimate_segment_output_chars(seg: Segment) -> int:
    """Merged sentences produce far less JSON than one row per input segment."""
    text = seg.text or ""
    return 8 + len(text) // 4


def _estimate_segment_input_chars(seg: Segment) -> int:
    text = seg.text or ""
    speaker = seg.speaker or ""
    return INPUT_OVERHEAD_PER_SEGMENT + len(text) + len(speaker)


def _chunk_segments(
    segments: list[Segment], budget: ChunkBudget, max_segments: int | None = None
) -> list[list[Segment]]:
    if not segments:
        return [[]]
    chunks: list[list[Segment]] = []
    current: list[Segment] = []
    output_size = 0
    input_size = 0
    for seg in segments:
        out_len = _estimate_segment_output_chars(seg)
        in_len = _estimate_segment_input_chars(seg)
        seg_limit = min(budget.max_segments, RELIABLE_CLEANUP_BATCH_SEGMENTS,
                        max_segments or RELIABLE_CLEANUP_BATCH_SEGMENTS)
        over_output = current and output_size + out_len > budget.max_output_chars
        over_input = current and input_size + in_len > budget.max_input_chars
        over_count = current and len(current) >= seg_limit
        if over_output or over_input or over_count:
            chunks.append(current)
            current = []
            output_size = 0
            input_size = 0
        current.append(seg)
        output_size += out_len
        input_size += in_len
    if current:
        chunks.append(current)
    return chunks


def _usage_logger(log_cb) -> Any:
    if not log_cb:
        return None

    def _usage_cb(usage: dict[str, int]) -> None:
        line = ai_providers.format_prompt_cache_usage(usage)
        if line:
            log_cb(f"AI Cleanup: {line}")

    return _usage_cb


def _chat_completion_cancellable(
    settings: Settings,
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    profile: ModelProfile,
    *,
    timeout: float | tuple[float, float],
    should_cancel: ShouldCancel,
    log_cb=None,
    cacheable_user_prefix: str | None = None,
    cache_key: str | None = None,
) -> str:
    use_stream = provider == "anthropic"
    last_logged = 0
    last_beat = time.monotonic()
    usage_cb = _usage_logger(log_cb)
    cache_kwargs = dict(
        cacheable_user_prefix=cacheable_user_prefix,
        cache_key=cache_key,
        use_prompt_cache=settings.prompt_cache_enabled,
        usage_cb=usage_cb,
    )

    def _stream_cb(n_chars: int) -> None:
        """Report progress by volume, and at least every 20s so a slow batch
        is visibly alive instead of silent for minutes."""
        nonlocal last_logged, last_beat
        if not log_cb:
            return
        now = time.monotonic()
        if n_chars - last_logged < 8000 and now - last_beat < 20:
            return
        last_logged = n_chars
        last_beat = now
        log_cb(f"AI Cleanup: receiving model output… ~{n_chars // 1000}k chars")

    if use_stream:
        if log_cb:
            log_cb("AI Cleanup: streaming response from model…")
        return ai_providers.chat_completion(
            settings,
            provider,
            model,
            system_prompt,
            user_prompt,
            profile,
            timeout=timeout,
            stream=True,
            should_cancel=should_cancel,
            stream_cb=_stream_cb if log_cb else None,
            **cache_kwargs,
        )

    if not should_cancel:
        return ai_providers.chat_completion(
            settings,
            provider,
            model,
            system_prompt,
            user_prompt,
            profile,
            timeout=timeout,
            **cache_kwargs,
        )

    result: list[str | None] = [None]
    error: list[BaseException | None] = [None]

    def _run() -> None:
        try:
            result[0] = ai_providers.chat_completion(
                settings,
                provider,
                model,
                system_prompt,
                user_prompt,
                profile,
                timeout=timeout,
                **cache_kwargs,
            )
        except BaseException as exc:
            error[0] = exc

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    while thread.is_alive():
        check_cancel(should_cancel, message="AI Cleanup: cancelled.")
        thread.join(0.25)
    if error[0] is not None:
        raise error[0]
    if result[0] is None:
        raise RuntimeError("AI Cleanup: model call returned no response.")
    return result[0]



def _saved_subrange_exists(
    segments: list[Segment], provider: str, model: str, prefix: str,
    resume: ResumeLog | None, depth: int = 8,
) -> bool:
    """True when a previous run split this batch and saved part of the result.

    Splitting always halves, so walking the same tree finds work recorded
    under sub-batch keys that the parent key alone would never match.
    """
    if not resume or depth <= 0 or len(segments) <= 1:
        return False
    mid = len(segments) // 2
    for half in (segments[:mid], segments[mid:]):
        if resume.get(_cleanup_send_key(provider, model, prefix, half)):
            return True
        if _saved_subrange_exists(half, provider, model, prefix, resume, depth - 1):
            return True
    return False


def _cleanup_chunk(
    segments: list[Segment],
    speakers: list[str],
    settings: Settings,
    provider: str,
    model: str,
    profile,
    cacheable_user_prefix: str,
    log_cb,
    *,
    cache_key: str | None = None,
    should_cancel: ShouldCancel = None,
    resume: ResumeLog | None = None,
) -> list[Segment]:
    if not segments:
        return []
    check_cancel(should_cancel, log_cb, message="AI Cleanup: cancelled.")

    # Before paying for this batch, see whether a previous run already
    # answered it in pieces — if so, split the same way and reuse them.
    if (
        resume
        and len(segments) > 1
        and not resume.get(
            _cleanup_send_key(provider, model, cacheable_user_prefix, segments)
        )
        and _saved_subrange_exists(
            segments, provider, model, cacheable_user_prefix, resume
        )
    ):
        mid = len(segments) // 2
        if log_cb:
            log_cb(
                f"AI Cleanup: an earlier run split this {len(segments)}-segment batch "
                f"— following the same split to reuse it."
            )
        return (
            _cleanup_chunk(
                segments[:mid], speakers, settings, provider, model, profile,
                cacheable_user_prefix, log_cb, cache_key=cache_key,
                should_cancel=should_cancel, resume=resume,
            )
            + _cleanup_chunk(
                segments[mid:], speakers, settings, provider, model, profile,
                cacheable_user_prefix, log_cb, cache_key=cache_key,
                should_cancel=should_cancel, resume=resume,
            )
        )
    try:
        return _cleanup_chunk_once(
            segments,
            speakers,
            settings,
            provider,
            model,
            profile,
            cacheable_user_prefix,
            log_cb,
            cache_key=cache_key,
            should_cancel=should_cancel,
            resume=resume,
        )
    except JobCancelled:
        raise
    except RuntimeError as e:
        if len(segments) <= 1 or not _should_split_chunk(str(e)):
            raise
        mid = len(segments) // 2
        if log_cb:
            log_cb(
                f"AI Cleanup: batch failed — splitting {len(segments)} segments "
                f"into {mid} + {len(segments) - mid}…"
            )
        # Each half is a send in its own right, checkpointed under its own key.
        left = _cleanup_chunk(
            segments[:mid],
            speakers,
            settings,
            provider,
            model,
            profile,
            cacheable_user_prefix,
            log_cb,
            cache_key=cache_key,
            should_cancel=should_cancel,
            resume=resume,
        )
        right = _cleanup_chunk(
            segments[mid:],
            speakers,
            settings,
            provider,
            model,
            profile,
            cacheable_user_prefix,
            log_cb,
            cache_key=cache_key,
            should_cancel=should_cancel,
            resume=resume,
        )
        return left + right


def _is_transient_network_error(error: str) -> bool:
    msg = error.lower()
    return any(marker in msg for marker in _TRANSIENT_NETWORK_MARKERS)


def _should_split_chunk(error: str) -> bool:
    msg = error.lower()
    if _is_transient_network_error(msg):
        return True
    return any(
        k in msg
        for k in (
            "invalid json",
            "truncated json",
            "expecting value",
            "empty response",
            "missing 'segments'",
            "stop_reason=max_tokens",
            "finish_reason=length",
        )
    )


def _cleanup_payload(segments: list[Segment]) -> str:
    """The exact user message sent for a batch — also its checkpoint identity."""
    payload = {
        "segments": [
            {
                "index": i,
                "speaker": seg.speaker,
                "text": seg.text,
            }
            for i, seg in enumerate(segments)
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


def _cleanup_send_key(
    provider: str, model: str, cacheable_user_prefix: str, segments: list[Segment]
) -> str:
    return send_key(
        "cleanup",
        provider,
        model,
        SYSTEM_PROMPT,
        cacheable_user_prefix,
        _cleanup_payload(segments),
    )


def _cleanup_chunk_once(
    segments: list[Segment],
    speakers: list[str],
    settings: Settings,
    provider: str,
    model: str,
    profile,
    cacheable_user_prefix: str,
    log_cb,
    *,
    cache_key: str | None = None,
    should_cancel: ShouldCancel = None,
    resume: ResumeLog | None = None,
) -> list[Segment]:
    check_cancel(should_cancel, log_cb, message="AI Cleanup: cancelled.")
    user_prompt = _cleanup_payload(segments)
    key = _cleanup_send_key(provider, model, cacheable_user_prefix, segments)

    saved = resume.get(key) if resume else None
    if saved:
        try:
            merged = _apply_cleanup(segments, _parse_response(saved))
            if log_cb:
                log_cb(
                    f"AI Cleanup: restored {len(segments)} segment(s) from an "
                    f"earlier run — no tokens spent."
                )
            return merged
        except Exception as e:
            if log_cb:
                log_cb(f"AI Cleanup: saved response unusable ({e}) — re-sending.")

    last_error = ""
    current_profile = profile
    token_ceiling = output_token_ceiling(provider, model)
    network_retries = 0
    request_timeout = (30, CLEANUP_READ_TIMEOUT_SEC)
    for attempt in range(MAX_RETRIES):
        check_cancel(should_cancel, log_cb, message="AI Cleanup: cancelled.")
        try:
            api_profile = current_profile
            if current_profile.max_tokens < token_ceiling:
                api_profile = current_profile.with_changes(max_tokens=token_ceiling)
            raw = _chat_completion_cancellable(
                settings,
                provider,
                model,
                SYSTEM_PROMPT,
                user_prompt,
                api_profile,
                timeout=request_timeout,
                should_cancel=should_cancel,
                log_cb=log_cb,
                cacheable_user_prefix=cacheable_user_prefix,
                cache_key=cache_key,
            )
            parsed = _parse_response(raw)
            merged = _apply_cleanup(segments, parsed)
            # Checkpoint only once the answer is known-good, and before
            # anything else can fail — this send is now paid for for good.
            if resume:
                resume.record(
                    key,
                    raw,
                    stage="cleanup",
                    segments=len(segments),
                    model=f"{provider}/{model}",
                )
            save_profile(api_profile)
            return merged
        except JobCancelled:
            raise
        except Exception as e:
            last_error = str(e)
            if _is_transient_network_error(last_error):
                network_retries += 1
                if network_retries >= NETWORK_RETRIES_BEFORE_SPLIT and len(segments) > 1:
                    if log_cb:
                        log_cb(
                            f"AI Cleanup: connection dropped on {len(segments)}-segment batch "
                            f"— splitting…"
                        )
                    raise RuntimeError(last_error) from e
                if attempt < MAX_RETRIES - 1:
                    wait = min(30, 2 ** (network_retries - 1))
                    if log_cb:
                        log_cb(
                            f"AI Cleanup: connection dropped — retrying in {wait}s "
                            f"({network_retries}/{NETWORK_RETRIES_BEFORE_SPLIT})…"
                        )
                    sleep_cancellable(wait, should_cancel, log_cb)
                    continue
            fix = suggest_profile_fix(last_error, current_profile, provider=provider)
            if fix and fix != current_profile:
                if log_cb:
                    log_cb(
                        f"AI Cleanup: {_describe_profile_fix(current_profile, fix)} "
                        f"— retrying ({attempt + 2}/{MAX_RETRIES})…"
                    )
                current_profile = fix
                continue
            if _should_split_chunk(last_error) and len(segments) > 1:
                raise RuntimeError(last_error) from e
            raise RuntimeError(f"AI Cleanup failed: {last_error}") from e
    raise RuntimeError(f"AI Cleanup failed after retries: {last_error}")


def _describe_profile_fix(before, after) -> str:
    if after.omit_temperature and not before.omit_temperature:
        return "model rejected temperature — retrying without it"
    if after.omit_json_response_format and not before.omit_json_response_format:
        return "retrying without JSON mode"
    if after.max_tokens > before.max_tokens:
        return f"increasing output limit to {after.max_tokens} tokens"
    if after.use_max_completion_tokens and not before.use_max_completion_tokens:
        return "switching to max_completion_tokens"
    return "adjusting model settings"


def _parse_response(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("Model returned empty response (no JSON to parse).")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        preview = text[:120].replace("\n", " ")
        truncated = text.count("{") > text.count("}") or not text.rstrip().endswith("}")
        kind = "truncated JSON" if truncated else "invalid JSON"
        raise ValueError(f"Model returned {kind}: {e} — preview: {preview!r}") from e
    if "segments" not in data:
        raise ValueError("Model response missing 'segments' array.")
    return data


def _apply_cleanup(original: list[Segment], data: dict[str, Any]) -> list[Segment]:
    items = data.get("segments", [])
    if not items:
        raise ValueError("Model response missing 'segments' array.")
    if any("from_index" in item or "to_index" in item for item in items):
        return _apply_merged_cleanup(original, items)
    return _apply_indexed_cleanup(original, items)


def _apply_merged_cleanup(
    original: list[Segment], items: list[dict[str, Any]]
) -> list[Segment]:
    n = len(original)
    starts: dict[int, tuple[int, dict[str, Any]]] = {}
    for item in items:
        if "from_index" not in item or "to_index" not in item:
            continue
        start = int(item["from_index"])
        end = int(item["to_index"])
        if start < 0 or end < start or start >= n:
            continue
        end = min(end, n - 1)
        starts[start] = (end, item)

    out: list[Segment] = []
    i = 0
    while i < n:
        if i in starts:
            end, item = starts[i]
            text = str(item.get("text", "")).strip()
            if not text:
                text = _join_fragment_text(original, i, end)
            text = _finalize_cleanup_text(text, item)
            speaker = _speaker_from_item(item, original, i, end)
            out.append(
                Segment(
                    start=original[i].start,
                    end=original[end].end,
                    text=text,
                    speaker=speaker,
                    channel=original[i].channel,
                )
            )
            i = end + 1
        else:
            out.append(
                Segment(
                    start=original[i].start,
                    end=original[i].end,
                    text=original[i].text,
                    speaker=original[i].speaker,
                    channel=original[i].channel,
                )
            )
            i += 1
    return out


def _finalize_cleanup_text(text: str, item: dict[str, Any]) -> str:
    text = text.strip()
    note = str(item.get("note") or "").strip()
    if note and note not in text:
        text = f"{text} ({note})"
    return text


def _speaker_from_item(
    item: dict[str, Any], original: list[Segment], start: int, end: int
) -> str | None:
    if "speaker" in item and item["speaker"] is not None:
        speaker = str(item["speaker"]).strip()
        if speaker:
            return speaker
    return _resolve_speaker(original, start, end, _MISSING)


_MISSING = object()


def _join_fragment_text(original: list[Segment], start: int, end: int) -> str:
    parts = [original[j].text.strip() for j in range(start, end + 1) if original[j].text.strip()]
    return " ".join(parts)


def _resolve_speaker(
    original: list[Segment], start: int, end: int, override: Any
) -> str | None:
    if override is not _MISSING:
        if override == "":
            return None
        return str(override)
    speakers = {original[j].speaker for j in range(start, end + 1)}
    speakers.discard(None)
    if len(speakers) == 1:
        return next(iter(speakers))
    return original[start].speaker


def _apply_indexed_cleanup(
    original: list[Segment], items: list[dict[str, Any]]
) -> list[Segment]:
    by_index: dict[int, dict[str, Any]] = {}
    for item in items:
        if "index" in item:
            by_index[int(item["index"])] = item

    out: list[Segment] = []
    for i, seg in enumerate(original):
        patch = by_index.get(i, {})
        text = str(patch.get("text", seg.text)).strip() or seg.text
        text = _finalize_cleanup_text(text, patch)
        speaker = patch.get("speaker", seg.speaker)
        if speaker == "":
            speaker = None
        out.append(
            Segment(
                start=seg.start,
                end=seg.end,
                text=text,
                speaker=speaker,
                channel=seg.channel,
            )
        )
    return out
