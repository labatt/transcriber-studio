# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Turning a list of words with timings back into readable speaker turns.

Both cloud engines hand back words, not sentences: ElevenLabs Scribe and Gemini
alike. Where the turn boundaries go is the same problem for both, and getting it
wrong the same way twice is worse than sharing one answer.

The input shape is deliberately plain, so an engine only has to translate its
own response into it:

    {"type": "word", "text": "Hello", "start": 0.1, "end": 0.4, "speaker_id": "spk:0"}

``type`` may also be ``"spacing"`` for whitespace an engine reports separately.
"""

from __future__ import annotations

import math

from .models import Segment

# Segment shaping. Scribe hands back words; these bounds turn them back into
# readable turns without cutting mid-sentence.
GAP_SECONDS = 2.0           # a silence this long ends the turn
SOFT_SECONDS, SOFT_CHARS = 30.0, 420    # break here at the next sentence end
HARD_SECONDS, HARD_CHARS = 60.0, 900    # break here regardless
SENTENCE_END = (".", "!", "?", "…")



def _speaker_map(words: list[dict], names: dict[str, str] | None = None) -> dict[str, str]:
    """An engine's own speaker ids become Speaker 1/2/… in order of first speech.

    ``names`` overrides that for speakers recognised from an enrolled voice.
    The numbering then runs over the ones left over, so a conversation between
    Alice and one stranger reads "Alice" and "Speaker 1" rather than skipping
    to "Speaker 2" and inviting the question of who Speaker 1 was.
    """
    names = names or {}
    order: list[str] = []
    for w in words:
        sid = w.get("speaker_id")
        if w.get("type") == "word" and sid and sid not in order:
            order.append(sid)
    mapping: dict[str, str] = {}
    unnamed = 0
    for sid in order:
        named = (names.get(sid) or "").strip()
        if named:
            mapping[sid] = named
            continue
        unnamed += 1
        mapping[sid] = f"Speaker {unnamed}"
    return mapping


def word_probability(word: dict) -> float | None:
    """How sure the engine was about one word, 0..1, or None if it did not say.

    Two spellings are accepted because two engines are: faster-whisper reports
    a probability directly, and Scribe reports the log of one.
    """
    value = word.get("probability")
    if value is None:
        logprob = word.get("logprob")
        if logprob is None:
            return None
        try:
            value = math.exp(float(logprob))
        except (TypeError, ValueError, OverflowError):
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return min(1.0, max(0.0, number))


def _mean_logprob(probabilities: list[float]) -> float | None:
    """The segment-level avg_logprob these word probabilities imply.

    Stored in log space, because that is what ``Segment.confidence`` expects
    and what the decoder's own segments carry — one representation, whichever
    path produced the segment.
    """
    usable = [p for p in probabilities if p > 0.0]
    if not usable:
        return None
    return float(sum(math.log(p) for p in usable) / len(usable))


def _should_break(text: str, start: float, end: float, word: dict) -> bool:
    duration = end - start
    if duration >= HARD_SECONDS or len(text) >= HARD_CHARS:
        return True
    if float(word.get("start", end)) - end >= GAP_SECONDS:
        return True
    if (duration >= SOFT_SECONDS or len(text) >= SOFT_CHARS) and text.rstrip().endswith(SENTENCE_END):
        return True
    return False


def words_to_segments(
    words: list[dict], diarized: bool, names: dict[str, str] | None = None
) -> tuple[list[Segment], list[str]]:
    """Group a word list into speaker turns.

    A turn ends when the speaker changes, when the silence between words is
    long enough to read as a pause, or when the text has simply run long —
    preferring a sentence boundary before forcing the break.

    ``names`` maps raw speaker ids to real names for anyone recognised from an
    enrolled voice; everyone else keeps a Speaker N label.
    """
    mapping = _speaker_map(words, names) if diarized else {}
    segments: list[Segment] = []
    text = ""
    start = end = 0.0
    speaker: str | None = None
    open_seg = False
    probs: list[float] = []

    def flush():
        nonlocal text, open_seg, probs
        cleaned = text.strip()
        if cleaned:
            segments.append(
                Segment(
                    start=start, end=end, text=cleaned, speaker=speaker,
                    # Rebuilt segments do not inherit the decoder's own score —
                    # they are not the segments it scored. Averaging the words
                    # it did score keeps a confidence on every line rather than
                    # losing it the moment diarization splits a turn.
                    avg_logprob=_mean_logprob(probs),
                )
            )
        text = ""
        probs = []
        open_seg = False

    for word in words:
        kind = word.get("type", "word")
        content = word.get("text", "")
        if kind == "spacing":
            if open_seg:
                text += content or " "
            continue
        who = mapping.get(word.get("speaker_id") or "") if diarized else None
        w_start = float(word.get("start", end))
        w_end = float(word.get("end", w_start))
        if open_seg and (who != speaker or _should_break(text, start, end, word)):
            flush()
        if not open_seg:
            start, speaker, open_seg = w_start, who, True
            text = ""
            probs = []
        text += content
        end = w_end
        probability = word_probability(word)
        if probability is not None:
            probs.append(probability)
    flush()

    speakers = list(dict.fromkeys(s.speaker for s in segments if s.speaker))
    return segments, speakers
