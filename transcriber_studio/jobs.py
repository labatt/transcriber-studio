# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Job orchestration: turn a Recording + options into written output files.

Kept UI-agnostic so it can be driven from a QThread worker or a CLI/test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import (
    denoise,
    diarization,
    filename_builder,
    formatters,
    glossary,
    vad,
    vocab_bias,
    word_timings,
)
from . import resume as resume_store
from .ai_cleanup import cleanup_transcript
from .audio_cache import CACHE_DIR, attach_if_cached, cache_path
from .config import Settings
from .job_cancel import JobCancelled, ShouldCancel, check_cancel
from .models import Recording, Source, TranscriptResult
from .plaud_client import PlaudClient
from .transcriber import TranscribeOptions, Transcriber, expected_model_label


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


def describe_bounds(min_speakers: int, max_speakers: int) -> str:
    """"at least 2, at most 6", "at most 2", "exactly 3", or "any number of speakers"."""
    if min_speakers and max_speakers and min_speakers == max_speakers:
        return f"exactly {min_speakers} speaker{'s' if min_speakers != 1 else ''}"
    parts = []
    if min_speakers:
        parts.append(f"at least {min_speakers}")
    if max_speakers:
        parts.append(f"at most {max_speakers}")
    return (", ".join(parts) + " speakers") if parts else "any number of speakers"


def copy_transcript(transcript: TranscriptResult) -> TranscriptResult:
    """Deep copy of a transcript.

    Copies by replacement rather than by listing fields: this used to name the
    five fields a Segment had when it was written, so every field added since —
    the decoder's confidence, the speaker voice vectors — was silently dropped
    on the way through, and a restored copy quietly had less in it than the
    original.
    """
    import dataclasses

    return TranscriptResult(
        recording=transcript.recording,
        segments=[dataclasses.replace(s) for s in transcript.segments],
        language=transcript.language,
        model=transcript.model,
        speakers=list(transcript.speakers),
        speaker_embeddings={
            k: list(v) for k, v in transcript.speaker_embeddings.items()
        },
        speaker_seconds=dict(transcript.speaker_seconds),
        words=[dict(w) for w in transcript.words],
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
    # Two labels renamed to the same person are one speaker, not two.
    result.speakers = list(dict.fromkeys(renames.get(s, s) for s in result.speakers))


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
            repetition_penalty=self.s.repetition_penalty,
            no_repeat_ngram_size=self.s.no_repeat_ngram_size,
            voiceprint_threshold=self.s.voiceprint_threshold,
            voiceprint_margin=self.s.voiceprint_margin,
            voiceprint_min_speech_s=self.s.voiceprint_min_speech_s,
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
            mai_api_key=self.s.mai_api_key,
            mai_region=self.s.mai_region,
            mai_model=self.s.mai_model,
            mai_style=self.s.mai_style,
            mai_send_phrases=self.s.mai_send_phrases,
            mai_speakers=self.s.mai_speakers,
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
        # Fetching and cleaning the audio already spent the first 40% of the
        # bar. Without this the decoder reports its own 0..1 over the top of
        # that, so the bar jumps back to zero and looks stuck for the longest
        # stage of the job. 0.92 is where AI cleanup takes over.
        return self.transcriber.transcribe(
            recording,
            audio_path,
            self._opts(recording),
            (lambda f: progress_cb(0.40 + f * 0.52)) if progress_cb else None,
            log_cb,
            should_cancel=should_cancel,
            resume=resume,
        )

    def apply_diarization(
        self, result: TranscriptResult, progress_cb=None, log_cb=None, should_cancel=None,
        *, min_speakers: int | None = None, max_speakers: int | None = None,
    ) -> TranscriptResult:
        """Label speakers on an existing transcript without re-running the decoder.

        ``min_speakers`` and ``max_speakers`` override the saved settings for
        this run only. That is how a transcript diarized under the wrong limits
        gets done again: a meeting of six people that was told "at most two"
        comes back as two, and no relabelling of those two turns can find the
        other four. When the transcript still carries its words they are
        regrouped into turns exactly as a fresh run would; without words the
        existing segments are relabelled, which cannot split a turn the old
        limits merged, and the log says so.
        """
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
        lo = self.s.min_speakers if min_speakers is None else min_speakers
        hi = self.s.max_speakers if max_speakers is None else max_speakers
        log = log_cb or (lambda _m: None)
        audio_path = self._ensure_audio(result.recording, progress_cb, log_cb, should_cancel)
        log(f"Running speaker diarization ({describe_bounds(lo, hi)}); the words are kept.")
        if not result.words:
            # A transcript from before word timings were kept. Recover them by
            # aligning its text to the audio, so the turns can still be redrawn
            # word by word rather than relabelled whole.
            log("No word timings on this transcript; recovering them from the audio first.")
            result.words = word_timings.recover_words(
                audio_path, result.segments, result.language, log, should_cancel
            )
        diar = diarization.Diarizer(self.s.hf_token, self.s.device)
        diarized = diar.diarize(
            audio_path, lo, hi, progress_cb, log_cb, should_cancel=should_cancel,
        )
        names = Transcriber._recognized_names(
            diarized, log,
            threshold=self.s.voiceprint_threshold,
            margin=self.s.voiceprint_margin,
            min_speech=self.s.voiceprint_min_speech_s,
            source=result.recording.display_name,
        )
        if result.words:
            # Regroup from the words, as a first run does: turns are drawn where
            # the new speaker boundaries fall, not where the old ones did.
            words = [dict(w) for w in result.words]
            segments, speakers = self.transcriber._apply_speakers(
                result.segments, words, diarized, log, names
            )
            result.segments, result.speakers, result.words = segments, speakers, words
        else:
            mapping = Transcriber._stable_speaker_map(diarized.turns, names)
            for seg in result.segments:
                raw = diarization.assign_speaker(seg.start, seg.end, diarized.turns)
                seg.speaker = mapping.get(raw) if raw else None
            result.speakers = list(dict.fromkeys(
                s.speaker for s in result.segments if s.speaker
            ))
        # Without this the rename dialog has labels but nothing to enrol from,
        # so "Remember this voice" sits greyed out after a Detect speakers run.
        result.speaker_embeddings, result.speaker_seconds = (
            Transcriber._speaker_voice_data(diarized, names)
        )
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
        label = expected_model_label(opts)
        saved = resume.get(key)
        matched_by = "the same options"
        if not saved:
            # The key formula has changed more than once as the app learned
            # which settings really shape a transcript. A transcript banked
            # under an older formula is still worth having: accept it when it
            # was made by the same engine and model, which the entry itself
            # records.
            saved = resume.latest(
                resume_store.TRANSCRIPT_STAGE,
                lambda raw: json.loads(raw).get("model") == label,
            )
            matched_by = "engine and model"
        if saved:
            try:
                transcript = resume_store.transcript_from_dict(recording, json.loads(saved))
                if log_cb:
                    log_cb(
                        f"Restored transcript from an interrupted run (matched by "
                        f"{matched_by}) — {len(transcript.segments)} segment(s), no "
                        f"re-transcription."
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
            engine=opts.engine,
            model=label,
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
            # Per-channel mode reads one speaker per channel, so the denoiser
            # must not hand back the downmix it produces by default.
            preserve_channels=self.s.channel_mode == "per_channel",
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
