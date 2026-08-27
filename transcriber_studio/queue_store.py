# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Persist the transcription queue across app restarts."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from .config import APP_DIR
from .jobs import JobResult
from .models import Recording, Segment, Source, TranscriptResult

QUEUE_PATH = APP_DIR / "queue.json"


def _recording_to_dict(rec: Recording) -> dict[str, Any]:
    d = asdict(rec)
    d["source"] = rec.source.value
    return d


def _recording_from_dict(d: dict[str, Any]) -> Recording:
    d = dict(d)
    d["source"] = Source(d["source"])
    return Recording(**d)


def _transcript_to_dict(transcript: TranscriptResult) -> dict[str, Any]:
    return {
        "language": transcript.language,
        "model": transcript.model,
        "speakers": transcript.speakers,
        "segments": [asdict(s) for s in transcript.segments],
    }


def _transcript_from_dict(rec: Recording, td: dict[str, Any]) -> TranscriptResult:
    segments = [Segment(**s) for s in td.get("segments", [])]
    return TranscriptResult(
        recording=rec,
        segments=segments,
        language=td.get("language", ""),
        model=td.get("model", ""),
        speakers=td.get("speakers", []),
    )


def _result_to_dict(result: JobResult, status: str) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "recording": _recording_to_dict(result.recording),
        "output_paths": result.output_paths,
        "error": result.error,
        "cancelled": result.cancelled,
        "status": status,
        "ai_cleanup_applied": result.ai_cleanup_applied,
        "glossary_id": result.glossary_id,
        "transcript": None,
        "original_transcript": None,
    }
    if result.transcript:
        entry["transcript"] = _transcript_to_dict(result.transcript)
    if result.original_transcript:
        entry["original_transcript"] = _transcript_to_dict(result.original_transcript)
    return entry


def _result_from_dict(d: dict[str, Any]) -> tuple[JobResult, str]:
    rec = _recording_from_dict(d["recording"])
    transcript = None
    if d.get("transcript"):
        transcript = _transcript_from_dict(rec, d["transcript"])
    original = None
    if d.get("original_transcript"):
        original = _transcript_from_dict(rec, d["original_transcript"])
    elif transcript:
        # Older saved jobs: treat current transcript as the original baseline.
        from .jobs import copy_transcript
        original = copy_transcript(transcript)
    result = JobResult(
        recording=rec,
        output_paths=d.get("output_paths", []),
        transcript=transcript,
        original_transcript=original,
        ai_cleanup_applied=bool(d.get("ai_cleanup_applied", False)),
        glossary_id=d.get("glossary_id"),
        error=d.get("error"),
        cancelled=bool(d.get("cancelled", False)),
    )
    return result, d.get("status", "Done")


def save_queue(results: list[JobResult], statuses: list[str]) -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    items = [
        _result_to_dict(r, statuses[i] if i < len(statuses) else "Done")
        for i, r in enumerate(results)
    ]
    QUEUE_PATH.write_text(
        json.dumps({"items": items}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_queue() -> list[tuple[JobResult, str]]:
    if not QUEUE_PATH.exists():
        return []
    try:
        data = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
        return [_result_from_dict(item) for item in data.get("items", [])]
    except Exception:
        return []


def clear_queue_file() -> None:
    if QUEUE_PATH.exists():
        QUEUE_PATH.unlink()
