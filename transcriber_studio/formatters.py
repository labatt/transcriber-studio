# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render a TranscriptResult into various output formats.

Every text-based line ends with a real carriage-return newline (CRLF by
default, per the requirement). Speaker names and timestamps are optional.

In .txt and .md, consecutive segments from the same speaker are joined into a
single speaker turn rendered as "Name: everything they said" on one line — the
next line break only comes when a different speaker starts. Subtitle formats
(.srt/.vtt) and .json stay one entry per segment, since their timings need it.
"""

from __future__ import annotations

import json
import re

from .models import Segment, TranscriptResult

EXT = {"txt": "txt", "srt": "srt", "vtt": "vtt", "json": "json", "md": "md"}
FORMAT_LABELS = {
    "txt": "Plain text (.txt)",
    "srt": "SubRip subtitles (.srt)",
    "vtt": "WebVTT (.vtt)",
    "json": "Structured JSON (.json)",
    "md": "Markdown (.md)",
}

_SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+")
_WHITESPACE_RE = re.compile(r"\s+")


def _nl(newline: str) -> str:
    return "\r\n" if newline == "crlf" else "\n"


def _ts(seconds: float, srt: bool = False) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    sep = "," if srt else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _clock(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _flatten(text: str) -> str:
    """Collapse every run of whitespace — newlines included — to one space."""
    return _WHITESPACE_RE.sub(" ", text or "").strip()


def _speaker_turns(result: TranscriptResult, opts) -> list[list[Segment]]:
    """Group consecutive segments spoken by the same person into one turn.

    Segments with no speaker label (and every segment when speaker names are
    turned off) stand alone, so an undiarized transcript keeps its own lines
    instead of collapsing into a single paragraph.
    """
    turns: list[list[Segment]] = []
    for seg in result.segments:
        if not _flatten(seg.text):
            continue
        joinable = opts.include_speakers and seg.speaker
        if turns and joinable and turns[-1][-1].speaker == seg.speaker:
            turns[-1].append(seg)
        else:
            turns.append([seg])
    return turns


def _expand_lines(turn: list[Segment], line_mode: str, wrap_chars: int) -> list[str]:
    """Render one speaker turn as one-or-more display lines."""
    text = " ".join(t for t in (_flatten(s.text) for s in turn) if t)
    if not text:
        return []
    if line_mode == "sentence":
        return [p.strip() for p in _SENTENCE_RE.split(text) if p.strip()]
    if line_mode == "wrap":
        words, lines, cur = text.split(), [], ""
        for w in words:
            if cur and len(cur) + 1 + len(w) > wrap_chars:
                lines.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        if cur:
            lines.append(cur)
        return lines
    return [text]  # "segment"


def render(result: TranscriptResult, fmt: str, opts) -> str:
    fmt = fmt.lower()
    if fmt == "json":
        return _render_json(result, opts)
    if fmt == "srt":
        return _render_srt(result, opts)
    if fmt == "vtt":
        return _render_vtt(result, opts)
    if fmt == "md":
        return _render_md(result, opts)
    return _render_txt(result, opts)


def _prefix(turn: list[Segment], opts) -> str:
    """Build `[00:12] Name: ` — the turn's start time, the name once."""
    parts = []
    if opts.include_timestamps:
        parts.append(f"[{_clock(turn[0].start)}]")
    if opts.include_speakers and turn[0].speaker:
        parts.append(f"{turn[0].speaker}:")
    return (" ".join(parts) + " ") if parts else ""


def _render_txt(result: TranscriptResult, opts) -> str:
    nl = _nl(opts.newline)
    out = []
    for turn in _speaker_turns(result, opts):
        lines = _expand_lines(turn, opts.line_mode, opts.wrap_chars)
        if not lines:
            continue
        # The name and everything said stay on one line; the break lands only
        # once this speaker is done and the next one starts.
        out.append(_prefix(turn, opts) + lines[0])
        out.extend(lines[1:])
    return nl.join(out) + nl


def _render_md(result: TranscriptResult, opts) -> str:
    nl = _nl(opts.newline)
    rec = result.recording
    head = [
        f"# {rec.display_name}",
        "",
        f"- **Source:** {rec.source.value}",
        f"- **Date:** {rec.date}",
        f"- **Duration:** {rec.duration}",
        f"- **Model:** {result.model}",
        f"- **Language:** {result.language}",
    ]
    if result.speakers:
        head.append(f"- **Speakers:** {', '.join(result.speakers)}")
    head += ["", "---", ""]
    body = []
    for turn in _speaker_turns(result, opts):
        lines = _expand_lines(turn, opts.line_mode, opts.wrap_chars)
        if not lines:
            continue
        if body:
            body.append("")  # Markdown needs the blank line to keep turns apart.
        ts = f"`{_clock(turn[0].start)}` " if opts.include_timestamps else ""
        name = ""
        if opts.include_speakers and turn[0].speaker:
            name = f"**{turn[0].speaker}:** "
        body.append(ts + name + lines[0])
        body.extend(lines[1:])
    return nl.join(head + body) + nl


def _render_srt(result: TranscriptResult, opts) -> str:
    nl = _nl(opts.newline)
    blocks = []
    for i, seg in enumerate(result.segments, 1):
        text = _flatten(seg.text)
        if not text:
            continue
        if opts.include_speakers and seg.speaker:
            text = f"{seg.speaker}: {text}"
        blocks.append(
            f"{i}{nl}{_ts(seg.start, True)} --> {_ts(seg.end, True)}{nl}{text}"
        )
    return (nl + nl).join(blocks) + nl


def _render_vtt(result: TranscriptResult, opts) -> str:
    nl = _nl(opts.newline)
    blocks = ["WEBVTT", ""]
    for seg in result.segments:
        text = _flatten(seg.text)
        if not text:
            continue
        if opts.include_speakers and seg.speaker:
            text = f"<v {seg.speaker}>{text}"
        blocks.append(f"{_ts(seg.start)} --> {_ts(seg.end)}{nl}{text}{nl}")
    return nl.join(blocks)


def _render_json(result: TranscriptResult, opts) -> str:
    data = {
        "recording": {
            "id": result.recording.id,
            "name": result.recording.name,
            "source": result.recording.source.value,
            "date": result.recording.date,
            "duration": result.recording.duration,
        },
        "model": result.model,
        "language": result.language,
        "speakers": result.speakers,
        "segments": [
            {
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "speaker": s.speaker,
                "channel": s.channel,
                "text": s.text.strip(),
            }
            for s in result.segments
        ],
    }
    return json.dumps(data, indent=2, ensure_ascii=False)
