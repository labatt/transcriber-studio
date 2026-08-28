# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Crash-resumable checkpoints for the expensive stages of a job.

Every model call is written to an append-only JSONL file the moment it returns,
so a run that dies — crash, kill, power cut, network drop — resumes from the
last completed send instead of paying for the same tokens twice.

A run the user cancels on purpose is different: "stop" means stop, so the
model-call checkpoints are thrown away and the next run starts fresh. The
transcript is kept even then, because re-running Whisper costs GPU time and
produces the same text anyway.

Checkpoints live outside the output folder (they are machinery, not results)
and are deleted once the job they belong to completes.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from .config import APP_DIR
from .models import Recording, Segment, TranscriptResult

RESUME_DIR = APP_DIR / "resume"
MAX_AGE_DAYS = 30

TRANSCRIPT_STAGE = "transcript"
#: The bare Whisper decode, banked before speakers are assigned to it.
DECODE_STAGE = "decode"


def send_key(*parts: str) -> str:
    """Content hash identifying one model call.

    Built from the exact prompts and payload, so any change to the transcript,
    the glossary, the model, or the prompt text yields a different key and the
    call is made again rather than restored from a stale answer.
    """
    h = hashlib.sha256()
    for part in parts:
        h.update((part or "").encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:24]


def recording_key(recording: Recording) -> str:
    return hashlib.sha256(recording.id.encode("utf-8")).hexdigest()[:12]


def resume_path(recording: Recording) -> Path:
    return RESUME_DIR / f"{recording_key(recording)}.jsonl"


def prune(max_age_days: int = MAX_AGE_DAYS) -> None:
    """Drop checkpoints left behind by jobs that were never finished or retried."""
    if not RESUME_DIR.exists():
        return
    cutoff = time.time() - max_age_days * 86400
    for path in RESUME_DIR.glob("*.jsonl"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            continue


class ResumeLog:
    """Append-only record of completed model calls for one recording.

    Pass path=None to disable checkpointing entirely (every call is a miss and
    nothing is written), which keeps callers free of `if resume:` branches.
    """

    def __init__(self, path: Path | None, log_cb=None):
        self.path = Path(path) if path else None
        self.log_cb = log_cb
        self._entries: dict[str, str] = {}
        self._stages: dict[str, str] = {}
        self._meta: dict[str, dict] = {}

    # -- reading -------------------------------------------------------
    def load(self) -> ResumeLog:
        """Read what previous runs completed. A torn final line is ignored."""
        self._entries.clear()
        self._stages.clear()
        self._meta.clear()
        if not self.path or not self.path.exists():
            return self
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError:
            return self
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                key = entry["key"]
            except (json.JSONDecodeError, KeyError, TypeError):
                continue    # a kill mid-write truncates one line; skip it
            self._entries[key] = entry.get("raw", "")
            self._stages[key] = entry.get("stage", "")
            self._meta[key] = entry
        return self

    def get(self, key: str) -> str | None:
        return self._entries.get(key)

    def count(self, stage: str | None = None) -> int:
        if stage is None:
            return len(self._entries)
        return sum(1 for s in self._stages.values() if s == stage)

    def largest_batch(self, stage: str = "cleanup") -> int:
        """Biggest segment count a previous run completed for this stage.

        A run that had to split its batches proves a smaller size works; the
        next run starts there so its boundaries line up with what was saved.
        """
        sizes = [
            int(m.get("segments") or 0)
            for k, m in self._meta.items()
            if self._stages.get(k) == stage
        ]
        return max(sizes) if sizes else 0

    def hits(self, keys: list[str]) -> int:
        return sum(1 for k in keys if k in self._entries)

    # -- writing -------------------------------------------------------
    def record(self, key: str, raw: str, *, stage: str, **meta: Any) -> None:
        """Persist one completed call. Flushed to disk before returning."""
        self._entries[key] = raw
        self._stages[key] = stage
        self._meta[key] = {"stage": stage, **meta}
        if not self.path:
            return
        entry = {"key": key, "stage": stage, "at": time.time(), **meta, "raw": raw}
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8", newline="\n") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())   # survive a hard kill, not just a clean exit
        except OSError as e:
            if self.log_cb:
                self.log_cb(f"Checkpoint: could not save progress ({e}).")

    # -- clearing ------------------------------------------------------
    def discard(self, keep_stages: tuple[str, ...] = ()) -> int:
        """Drop checkpoints, optionally keeping some stages. Returns count dropped."""
        dropped = sum(1 for s in self._stages.values() if s not in keep_stages)
        kept = [
            (k, self._entries[k], self._stages[k])
            for k in self._entries
            if self._stages.get(k) in keep_stages
        ]
        self._entries = {k: raw for k, raw, _ in kept}
        self._stages = {k: stage for k, _, stage in kept}
        if not self.path:
            return dropped
        try:
            if not kept:
                self.path.unlink(missing_ok=True)
                return dropped
            tmp = self.path.with_suffix(".jsonl.tmp")
            with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
                for key, raw, stage in kept:
                    fh.write(json.dumps(
                        {"key": key, "stage": stage, "at": time.time(), "raw": raw},
                        ensure_ascii=False,
                    ) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            tmp.replace(self.path)
        except OSError as e:
            if self.log_cb:
                self.log_cb(f"Checkpoint: could not clear progress ({e}).")
        return dropped


def log_for(recording: Recording, log_cb=None) -> ResumeLog:
    """Open (and load) the checkpoint file for a recording."""
    return ResumeLog(resume_path(recording), log_cb=log_cb).load()


# -- transcript stage ---------------------------------------------------

def transcript_key(recording: Recording, opts: Any) -> str:
    """Identifies a transcript by everything that would change its content."""
    return send_key(
        "transcript",
        recording.id,
        # The engine is part of the identity: a Whisper transcript must never
        # be restored for a run the user switched to ElevenLabs, or the other
        # way round.
        str(getattr(opts, "engine", "local")),
        str(getattr(opts, "elevenlabs_model", "")),
        str(getattr(opts, "gemini_model", "")),
        str(getattr(opts, "gemini_mode", "")),
        str(opts.model),
        str(opts.language),
        str(opts.diarization_enabled),
        str(opts.min_speakers),
        str(opts.max_speakers),
        str(opts.channel_mode),
        ",".join(opts.channel_names or []),
        # The layers in front of the decoder change the words that come out of
        # it, so a saved transcript from before they were switched on must not
        # be restored over the run that switched them on.
        str(getattr(opts, "denoise", "")),
        str(getattr(opts, "vad_enabled", True)),
        repr(sorted((getattr(opts, "vad_parameters", None) or {}).items())),
        str(getattr(opts, "hotwords", "")),
        str(getattr(opts, "hallucination_guard", False)),
    )


def decode_key(recording: Recording, opts: Any) -> str:
    """Identifies a Whisper decode by everything that changes the words.

    Deliberately excludes the diarization settings that transcript_key
    includes: speaker labels are attached to the segments afterwards, so a
    decode stays valid however diarization is configured — or whether it ran
    at all. That is the whole point of saving it separately. Diarizing an hour
    of audio can take minutes, and a crash in the middle of it used to throw
    away the far more expensive decode that came before.
    """
    return send_key(
        "decode",
        recording.id,
        str(getattr(opts, "engine", "local")),
        str(opts.model),
        str(opts.language),
        str(opts.channel_mode),
        ",".join(opts.channel_names or []),
        str(getattr(opts, "denoise", "")),
        str(getattr(opts, "vad_enabled", True)),
        repr(sorted((getattr(opts, "vad_parameters", None) or {}).items())),
        str(getattr(opts, "hotwords", "")),
        str(getattr(opts, "hallucination_guard", False)),
    )


def decode_to_dict(segments: list[Segment], language: str) -> dict[str, Any]:
    return {
        "language": language,
        "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in segments],
    }


def decode_from_dict(data: dict[str, Any]) -> tuple[list[Segment], str]:
    return (
        [Segment(**s) for s in data.get("segments", [])],
        data.get("language", ""),
    )


def transcript_to_dict(transcript: TranscriptResult) -> dict[str, Any]:
    return {
        "language": transcript.language,
        "model": transcript.model,
        "speakers": list(transcript.speakers),
        "segments": [
            {
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "speaker": s.speaker,
                "channel": s.channel,
            }
            for s in transcript.segments
        ],
    }


def transcript_from_dict(recording: Recording, data: dict[str, Any]) -> TranscriptResult:
    return TranscriptResult(
        recording=recording,
        segments=[Segment(**s) for s in data.get("segments", [])],
        language=data.get("language", ""),
        model=data.get("model", ""),
        speakers=list(data.get("speakers") or []),
    )


# -- inspection ---------------------------------------------------------

_progress_cache: dict[str, tuple[float, dict[str, int]]] = {}


def saved_progress(recording: Recording) -> dict[str, int]:
    """How much of an interrupted run is banked, by stage. Empty when none.

    The UI asks this on every selection change, so results are cached against
    the file's mtime rather than re-parsing the log each time.
    """
    path = resume_path(recording)
    if not path.exists():
        _progress_cache.pop(str(path), None)
        return {}
    stamp = path.stat().st_mtime
    cached = _progress_cache.get(str(path))
    if cached and cached[0] == stamp:
        return cached[1]
    log = ResumeLog(path).load()
    counts = {
        stage: log.count(stage)
        for stage in (TRANSCRIPT_STAGE, DECODE_STAGE, "glossary", "cleanup")
    }
    counts = {stage: n for stage, n in counts.items() if n}
    _progress_cache[str(path)] = (stamp, counts)
    return counts


def describe_progress(recording: Recording) -> str:
    """One-line summary for the UI, or "" when there is nothing saved."""
    counts = saved_progress(recording)
    if not counts:
        return ""
    parts = []
    if counts.get(TRANSCRIPT_STAGE):
        parts.append("transcript")
    elif counts.get(DECODE_STAGE):
        # Only worth mentioning on its own: with a full transcript saved, the
        # decode behind it is implied and saying both would just be noise.
        parts.append("transcribed audio (speakers still to do)")
    if counts.get("glossary"):
        parts.append(f"{counts['glossary']} glossary chunk(s)")
    if counts.get("cleanup"):
        parts.append(f"{counts['cleanup']} cleanup batch(es)")
    return ", ".join(parts)
