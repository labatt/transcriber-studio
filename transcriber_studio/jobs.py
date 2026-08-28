# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Job orchestration: turn a Recording + options into written output files.

Kept UI-agnostic so it can be driven from a QThread worker or a CLI/test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import denoise, diarization, filename_builder, formatters, glossary, vad, vocab_bias
from . import resume as resume_store
from .ai_cleanup import cleanup_transcript
from .audio_cache import CACHE_DIR, attach_if_cached, cache_path
from .config import Settings
from .job_cancel import JobCancelled, ShouldCancel, check_cancel
from .models import Recording, Source, TranscriptResult
from .plaud_client import PlaudClient
from .transcriber import TranscribeOptions, Transcriber


@dataclass
class JobResult:
    recording: Recording
    output_paths: list[str] = field(default_factory=list)
    transcript: TranscriptResult | None = None
    original_transcript: TranscriptResult | None = None
    ai_cleanup_applied: bool = False
    # Shared glossary this job reads from and writes back to (an id in
    # transcriber_studio.glossary_store). "" keeps the glossary private to the recording;
    # None means the job has not chosen and follows the app default.
    glossary_id: str | None = None
    error: str | None = None
    cancelled: bool = False     # stopped by the user, not a failure


def copy_transcript(transcript: TranscriptResult) -> TranscriptResult:
    """Deep copy of a transcript (segments + speaker labels)."""
    from .models import Segment

    return TranscriptResult(
        recording=transcript.recording,
        segments=[
            Segment(s.start, s.end, s.text, s.speaker, s.channel)
            for s in transcript.segments
        ],
        language=transcript.language,
        model=transcript.model,
        speakers=list(transcript.speakers),
    )


def ensure_original_snapshot(job: JobResult) -> None:
    """Keep an immutable pre-cleanup copy for re-runs from the original."""
    if job.transcript and job.original_transcript is None:
        job.original_transcript = copy_transcript(job.transcript)


def apply_speaker_renames(result: TranscriptResult, renames: dict[str, str]) -> None:
    """renames maps current label -> new name. Mutates the result in place."""
    if not renames:
        return
    for seg in result.segments:
        if seg.speaker in renames:
            seg.speaker = renames[seg.speaker]
    result.speakers = [renames.get(s, s) for s in result.speakers]


def remove_superseded_outputs(old_paths: list[str], new_paths: list[str]) -> list[str]:
    """Delete earlier exports of this job that the new ones replace.

    Renaming the speakers renames the file, so the pre-rename export would
    otherwise sit next to it as a stale duplicate. Only files this job wrote
    are touched, and only when a file with the new name exists.
    """
    keep = {str(Path(p).resolve()) for p in new_paths}
    removed = []
    for old in old_paths:
        path = Path(old)
        if str(path.resolve()) in keep or not path.is_file():
            continue
        try:
            path.unlink()
            removed.append(old)
        except OSError:
            continue
    return removed


class JobRunner:
    def __init__(self, settings: Settings, client: PlaudClient | None = None):
        self.s = settings
        self.client = client or PlaudClient()
        self.transcriber = Transcriber()

    def _opts(self, recording: Recording | None = None) -> TranscribeOptions:
        names = [n.strip() for n in self.s.channel_names.split(",") if n.strip()]
        return TranscribeOptions(
            denoise=denoise.resolve(self.s),
            vad_enabled=self.s.vad_enabled,
            vad_parameters=vad.parameters(self.s),
            hotwords=self._hotwords(recording),
            hallucination_guard=self.s.hallucination_guard,
            model=self.s.model,
            device=self.s.device,
            compute_type=self.s.compute_type,
            language=self.s.language,
            diarization_enabled=self.s.diarization_enabled,
            hf_token=self.s.hf_token,
            min_speakers=self.s.min_speakers,
            max_speakers=self.s.max_speakers,
            channel_mode=self.s.channel_mode,
            channel_names=names or None,
            engine=self.s.stt_engine,
            elevenlabs_api_key=self.s.elevenlabs_api_key,
            elevenlabs_model=self.s.elevenlabs_model,
            gemini_api_key=self.s.ai_key_google,
            gemini_model=self.s.gemini_model,
            gemini_mode=self.s.gemini_mode,
            tag_audio_events=self.s.elevenlabs_tag_audio_events,
        )

    def _hotwords(self, recording: Recording | None) -> str:
        """Vocabulary to bias the decoder with, from the glossaries this job knows.

        The shared glossary is the interesting one: it is the vocabulary every
        earlier recording in that account already taught the app, available
        before this recording has been decoded even once.
        """
        payloads = []
        prior = self._prior_glossary(recording)
        if prior:
            payloads.append(prior)
        return vocab_bias.hotwords(self.s, extra_payloads=payloads)

    def _prior_glossary(self, recording: Recording | None) -> dict | None:
        """This recording's own glossary from a previous run, if it has one."""
        if recording is None:
            return None
        try:
            path = glossary.glossary_path(self.s, TranscriptResult(recording=recording))
            if path.exists():
                return glossary.load_glossary(path)
        except Exception:
            # A filename template using fields only a finished transcript has
            # (model, language) cannot be resolved yet; there is simply no
            # prior glossary to find in that case.
            return None
        return None

    def transcribe_only(
        self,
        recording: Recording,
        index: int = 1,
        progress_cb=None,
        log_cb=None,
        should_cancel: ShouldCancel = None,
        resume=None,
    ) -> TranscriptResult:
        """Produce a transcript result without writing files (for the rename step)."""
        audio_path = self._ensure_audio(recording, progress_cb, log_cb, should_cancel)
        return self.transcriber.transcribe(
            recording,
            audio_path,
            self._opts(recording),
            progress_cb,
            log_cb,
            should_cancel=should_cancel,
            resume=resume,
        )

    def apply_diarization(
        self, result: TranscriptResult, progress_cb=None, log_cb=None, should_cancel=None
    ) -> TranscriptResult:
        """Label speakers on an existing transcript without re-running Whisper."""
        if not self.s.hf_token:
            raise RuntimeError(
                "Speaker diarization needs a HuggingFace token. Add one in Settings.\n\n"
                "With the ElevenLabs engine, speakers come back from the transcription "
                "itself — enable speaker detection in Settings and re-run the job instead."
            )
        if not diarization.is_available():
            raise RuntimeError(
                "pyannote.audio is not installed. Run: pip install pyannote.audio"
            )
        audio_path = self._ensure_audio(result.recording, progress_cb, log_cb, should_cancel)
        if log_cb:
            log_cb("Running speaker diarization (existing transcript kept)…")
        diar = diarization.Diarizer(self.s.hf_token, self.s.device)
        turns = diar.diarize(
            audio_path,
            self.s.min_speakers,
            self.s.max_speakers,
            progress_cb,
            log_cb,
            should_cancel=should_cancel,
        )
        mapping = Transcriber._stable_speaker_map(turns)
        for seg in result.segments:
            raw = diarization.assign_speaker(seg.start, seg.end, turns)
            seg.speaker = mapping.get(raw) if raw else None
        result.speakers = list(dict.fromkeys(
            s.speaker for s in result.segments if s.speaker
        ))
        return result

    def apply_ai_cleanup(
        self,
        result: TranscriptResult,
        progress_cb=None,
        log_cb=None,
        *,
        force: bool = False,
        provider: str | None = None,
        model: str | None = None,
        index: int = 1,
        glossary_id: str | None = None,
        should_cancel=None,
        resume=None,
    ) -> TranscriptResult:
        if not force and not self.s.ai_cleanup_enabled:
            return result
        return cleanup_transcript(
            result,
            self.s,
            provider=provider,
            model=model,
            index=index,
            glossary_id=glossary_id,
            progress_cb=progress_cb,
            log_cb=log_cb,
            should_cancel=should_cancel,
            resume=resume,
        )

    def write_outputs(
        self,
        result: TranscriptResult,
        index: int = 1,
        *,
        cleanup_provider: str | None = None,
        cleanup_model: str | None = None,
    ) -> list[str]:
        out_dir = Path(self.s.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        values = filename_builder.build_values(result, index, self.s.sanitize_names)
        # Once the other speaker has a real name, that name drives the filename.
        base_stem = filename_builder.person_stem(
            result, self.s.sanitize_names, self.s.owner_names
        ) or (
            filename_builder.render(self.s.filename_template, values, self.s.sanitize_names)
        )
        if cleanup_provider and cleanup_model:
            stem = filename_builder.cleanup_stem(
                base_stem, cleanup_provider, cleanup_model, self.s.sanitize_names
            )
        else:
            stem = base_stem
        written = []
        for fmt in self.s.formats:
            text = formatters.render(result, fmt, self.s)
            path = filename_builder.unique_path(
                str(out_dir), stem, formatters.EXT.get(fmt, fmt), self.s.overwrite
            )
            # newline="" keeps our explicit CRLF/LF intact (no translation).
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(text)
            written.append(str(path))
        return written

    def run(
        self,
        recording: Recording,
        index: int = 1,
        progress_cb=None,
        log_cb=None,
        should_cancel: ShouldCancel = None,
    ) -> JobResult:
        """Transcribe, clean up, and export. Stops promptly when cancelled.

        Nothing is written on cancel: the exception unwinds before
        write_outputs, so a stopped job leaves no half-made transcript.
        """
        try:
            check_cancel(should_cancel, log_cb, message="Cancelled before starting.")
            resume_store.prune()
            resume = resume_store.log_for(recording, log_cb)
            transcript = self._transcribe_or_restore(
                recording, index, progress_cb, log_cb, should_cancel, resume
            )
            original = copy_transcript(transcript)
            cleaned = False
            check_cancel(should_cancel, log_cb, message="Cancelled — nothing written.")
            if self.s.ai_cleanup_enabled:
                if log_cb:
                    log_cb("Running AI Cleanup…")
                transcript = self.apply_ai_cleanup(
                    transcript,
                    progress_cb=(lambda f: progress_cb(0.92 + f * 0.08)) if progress_cb else None,
                    log_cb=log_cb,
                    force=True,
                    index=index,
                    glossary_id=self.s.glossary_shared_id,
                    should_cancel=should_cancel,
                    resume=resume,
                )
                cleaned = True
            check_cancel(should_cancel, log_cb, message="Cancelled — nothing written.")
            paths = self.write_outputs(
                transcript,
                index,
                cleanup_provider=self.s.ai_cleanup_provider if cleaned else None,
                cleanup_model=self.s.ai_cleanup_model if cleaned else None,
            )
            resume.discard()   # exported: there is nothing left to resume
            return JobResult(
                recording,
                paths,
                transcript,
                original_transcript=original,
                ai_cleanup_applied=cleaned,
                glossary_id=self.s.glossary_shared_id,
            )
        except JobCancelled:
            return JobResult(recording, cancelled=True)
        except Exception as e:  # surfaced to the queue row
            return JobResult(recording, error=str(e))

    def _transcribe_or_restore(
        self, recording, index, progress_cb, log_cb, should_cancel, resume
    ) -> TranscriptResult:
        """Reuse the transcript from an interrupted run instead of re-decoding.

        Whisper is deterministic for a given model and options, so a run that
        died during cleanup has no reason to spend the GPU time again.
        """
        opts = self._opts(recording)
        key = resume_store.transcript_key(recording, opts)
        saved = resume.get(key)
        if saved:
            try:
                transcript = resume_store.transcript_from_dict(recording, json.loads(saved))
                if log_cb:
                    log_cb(
                        f"Restored transcript from an interrupted run — "
                        f"{len(transcript.segments)} segment(s), no re-transcription."
                    )
                if progress_cb:
                    progress_cb(0.92)
                return transcript
            except Exception as e:
                if log_cb:
                    log_cb(f"Saved transcript unusable ({e}) — transcribing again.")

        transcript = self.transcribe_only(
            recording, index, progress_cb, log_cb, should_cancel, resume
        )
        resume.record(
            key,
            json.dumps(resume_store.transcript_to_dict(transcript), ensure_ascii=False),
            stage=resume_store.TRANSCRIPT_STAGE,
            segments=len(transcript.segments),
        )
        return transcript

    # ------------------------------------------------------------------
    def _ensure_audio(
        self, recording: Recording, progress_cb, log_cb, should_cancel: ShouldCancel = None
    ) -> str:
        """The audio every stage works from — downloaded if needed, then cleaned.

        Denoising lands here rather than inside the transcriber because the
        enhanced file feeds diarization and the cloud engine as well, and none
        of them should be looking at a different signal than the decoder.
        """
        source = self._source_audio(recording, progress_cb, log_cb, should_cancel)
        # Downloading owns the first 30% of the bar; give denoising the next
        # slice rather than leaving it looking stalled on a long recording.
        return denoise.enhance(
            source,
            self.s,
            log_cb=log_cb,
            progress_cb=(lambda f: progress_cb(0.30 + f * 0.10)) if progress_cb else None,
            should_cancel=should_cancel,
        )

    def _source_audio(
        self, recording: Recording, progress_cb, log_cb, should_cancel: ShouldCancel = None
    ) -> str:
        if recording.source == Source.LOCAL and recording.local_path:
            return recording.local_path
        attach_if_cached(recording)
        if recording.local_path and Path(recording.local_path).exists():
            if log_cb and recording.source == Source.PLAUD:
                log_cb("Using cached audio (already downloaded).")
            return recording.local_path
        dest = cache_path(recording.id)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if log_cb:
            log_cb("Downloading audio from Plaud…")
        self.client.download_audio(
            recording.id,
            str(dest),
            progress_cb=(lambda f: progress_cb(f * 0.3)) if progress_cb else None,
            should_cancel=should_cancel,
            label=recording.display_name,   # the row's name beats a raw hex id
            log_cb=log_cb,
        )
        recording.local_path = str(dest)
        return str(dest)
