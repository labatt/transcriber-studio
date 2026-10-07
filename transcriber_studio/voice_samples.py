# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Choosing which few seconds of each speaker a person should listen to.

Speaker detection says who spoke when, but not who they are. The fastest way to
find out is to hear them — a few seconds each is enough for anyone who knows
the people in the room. What this module decides is *which* few seconds.

Two or more samples per speaker, never one. A single clip can land on a cough,
a cross-talk moment or an "mm-hm", and then the person listening has nothing
to go on. Where the speaker has several stretches of speech the samples come
from stretches far apart in the recording, since a voice at the start of an
hour and the same voice near the end is more convincing than two clips a
minute apart. A speaker who only ever spoke once still gets several windows,
cut from different parts of that one stretch.
"""

from __future__ import annotations

from dataclasses import dataclass

from .diarization import SpeakerTurn

#: How long a sample is. Long enough to hear a voice properly; short enough
#: that four of them for each of six people is still a quick job.
CLIP_SECONDS = 8.0
#: A stretch shorter than this is a word or two — useless as a sample — unless
#: the speaker has nothing longer, in which case it is still better than nothing.
MIN_CLIP_SECONDS = 2.5
#: The user asked for at least two. Four is where listening stops helping.
MIN_CLIPS = 2
MAX_CLIPS = 4
#: pyannote splits one person's monologue at every breath. Turns this close
#: together are the same stretch of speech, and a sample may run across them.
MERGE_GAP_SECONDS = 0.6


@dataclass(frozen=True)
class Clip:
    """A window of the recording worth playing to identify one speaker."""

    speaker: str
    start: float
    end: float

    @property
    def seconds(self) -> float:
        return max(0.0, self.end - self.start)


def stretches(turns: list[SpeakerTurn], speaker: str) -> list[tuple[float, float]]:
    """Continuous spans of one speaker, with breath-length gaps closed.

    Only gaps where nobody else spoke are closed: a half-second answer from
    someone else between two of this speaker's turns means a sample across
    the join would have two voices in it.
    """
    ordered = sorted(turns, key=lambda t: t.start)
    spans: list[tuple[float, float]] = []
    for index, turn in enumerate(ordered):
        if turn.speaker != speaker or turn.end <= turn.start:
            continue
        if spans:
            last_start, last_end = spans[-1]
            gap = turn.start - last_end
            interrupted = any(
                other.speaker != speaker and other.start < turn.start and other.end > last_end
                for other in ordered[:index]
            )
            if 0 <= gap <= MERGE_GAP_SECONDS and not interrupted:
                spans[-1] = (last_start, max(last_end, turn.end))
                continue
        spans.append((turn.start, turn.end))
    return spans


def _windows_within(span: tuple[float, float], count: int, length: float) -> list[tuple[float, float]]:
    """``count`` windows of ``length`` spread evenly through one span.

    A span shorter than the window gives one clip of the whole span. The
    first window starts where the speech starts, since the opening of a turn
    is where a sentence begins; the rest are spaced so the last ends with it.
    """
    start, end = span
    total = end - start
    if total <= length:
        return [(start, end)]
    room = total - length
    if count <= 1:
        return [(start, start + length)]
    step = room / (count - 1)
    return [(start + i * step, start + i * step + length) for i in range(count)]


def pick_clips(
    turns: list[SpeakerTurn], speaker: str, *,
    count: int = MAX_CLIPS, length: float = CLIP_SECONDS, minimum: float = MIN_CLIP_SECONDS,
) -> list[Clip]:
    """Up to ``count`` samples of one speaker, at least MIN_CLIPS where possible.

    Preference order: distinct stretches far apart in the recording, longest
    first; then, if the speaker has too few stretches, extra windows cut from
    the longest one. Returned in recording order, so the buttons read as a
    timeline.
    """
    spans = stretches(turns, speaker)
    if not spans:
        return []
    usable = [s for s in spans if s[1] - s[0] >= minimum] or [max(spans, key=lambda s: s[1] - s[0])]
    by_length = sorted(usable, key=lambda s: s[1] - s[0], reverse=True)

    chosen: list[tuple[float, float]] = [by_length[0]]
    remaining = by_length[1:]
    while remaining and len(chosen) < count:
        # Furthest from everything already chosen, so the samples are spread
        # through the recording rather than clustered where the speaker was
        # chattiest. Length breaks ties.
        def distance(span: tuple[float, float]) -> tuple[float, float]:
            mid = (span[0] + span[1]) / 2
            return (
                min(abs(mid - (c[0] + c[1]) / 2) for c in chosen),
                span[1] - span[0],
            )

        best = max(remaining, key=distance)
        remaining.remove(best)
        chosen.append(best)

    windows: list[tuple[float, float]] = []
    for span in chosen:
        windows.extend(_windows_within(span, 1, length))
    wanted = max(MIN_CLIPS, min(count, len(windows)))
    if len(windows) < wanted:
        # Too few stretches: the longest one has to supply the rest. Only
        # worth it when that stretch is long enough to hold distinct windows.
        longest = by_length[0]
        extra = wanted - len(windows)
        if longest[1] - longest[0] >= length * 1.5:
            more = _windows_within(longest, extra + 1, length)[1:]
            windows.extend(more)
    windows.sort()
    deduped: list[tuple[float, float]] = []
    for window in windows:
        if not deduped or window[0] - deduped[-1][0] >= 1.0:
            deduped.append(window)
    return [Clip(speaker, round(s, 3), round(e, 3)) for s, e in deduped[:count]]


def clips_for(turns: list[SpeakerTurn], **kwargs) -> dict[str, list[Clip]]:
    """Samples for every speaker in the turns, keyed by raw label, in order of
    first speech — the same order the transcript numbers them in."""
    order: list[str] = []
    for turn in sorted(turns, key=lambda t: t.start):
        if turn.speaker not in order:
            order.append(turn.speaker)
    return {label: pick_clips(turns, label, **kwargs) for label in order}


def clock(seconds: float) -> str:
    """12:34 or 1:02:03, for the sample buttons."""
    total = int(max(0.0, seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"
