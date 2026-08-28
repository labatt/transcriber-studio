# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Local Whisper transcription via faster-whisper, with optional diarization.

Heavy deps (faster-whisper / ctranslate2) are imported lazily.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

from . import (
    audio_utils,
    diarization,
    stt_elevenlabs,
    stt_gemini,
    vocab_bias,
    whisper_models,
)
from .hardware import cuda_available
from .job_cancel import JobCancelled, ShouldCancel, check_cancel
from .models import Recording, Segment, TranscriptResult

ENGINE_LOCAL = "local"
ENGINE_ELEVENLABS = "elevenlabs"
ENGINE_GEMINI = "gemini"
ENGINE_LABELS = {
    ENGINE_LOCAL: "Local Whisper (faster-whisper)",
    ENGINE_ELEVENLABS: "ElevenLabs Scribe (cloud)",
    ENGINE_GEMINI: "Gemini 3.5 Transcribe (cloud)",
}
#: Engines that transcribe and separate speakers in one pass, so pyannote and
#: the HuggingFace token play no part.
CLOUD_ENGINES = (ENGINE_ELEVENLABS, ENGINE_GEMINI)


@dataclass
class TranscribeOptions:
    model: str = "large-v3"
    device: str = "auto"
    compute_type: str = "auto"
    language: str = "auto"          # "auto" or ISO code
    diarization_enabled: bool = True
    hf_token: str = ""
    min_speakers: int = 0
    max_speakers: int = 0
    channel_mode: str = "downmix"   # downmix | per_channel
    channel_names: list[str] | None = None
    word_timestamps: bool = True
    # --- pipeline layers in front of the decoder (local engine) ---
    # Denoising happens earlier, in the job runner, because the enhanced file
    # feeds diarization too.
    #: Which denoiser ran before this file reached us — identity only, so a
    #: saved transcript is not restored across a change of front-end.
    denoise: str = ""
    vad_enabled: bool = True
    vad_parameters: dict | None = None
    #: Names, products and jargon to bias the decoder toward. See transcriber_studio.vocab_bias.
    hotwords: str = ""
    hallucination_guard: bool = True
    hallucination_silence_s: float = 2.0
    # --- engine choice ---
    engine: str = ENGINE_LOCAL      # local (faster-whisper) | elevenlabs (Scribe)
    elevenlabs_api_key: str = ""
    elevenlabs_model: str = ""
    tag_audio_events: bool = False  # ElevenLabs only: mark laughter, applause…
    gemini_api_key: str = ""        # the Google AI key, shared with AI Cleanup
    gemini_model: str = ""
    gemini_mode: str = ""           # smart | verbatim


def faster_whisper_available() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


def _resolve_device_compute(device: str, compute_type: str) -> tuple[str, str]:
    has_cuda = cuda_available()
    dev = "cuda" if (device == "cuda" or (device == "auto" and has_cuda)) else "cpu"
    if compute_type != "auto":
        return dev, compute_type
    return dev, ("float16" if dev == "cuda" else "int8")


def transcribe_kwargs(opts: TranscribeOptions, language: str | None) -> dict:
    """Everything the three pipeline layers add to a faster-whisper call.

    ``hotwords`` rather than ``initial_prompt``: faster-whisper re-injects
    hotwords into the prompt for every decoder window, while an initial prompt
    only seeds the first one and then survives on carried-over context — which
    the hallucination guard below deliberately turns off. For a long recording
    the difference is the whole vocabulary, so hotwords it is.
    """
    kwargs: dict = {
        "language": language,
        "word_timestamps": opts.word_timestamps,
        "vad_filter": opts.vad_enabled,
    }
    if opts.vad_enabled and opts.vad_parameters:
        kwargs["vad_parameters"] = dict(opts.vad_parameters)
    if opts.hotwords:
        kwargs["hotwords"] = opts.hotwords
    if opts.hallucination_guard:
        # A hallucinated passage otherwise becomes the prompt for the next
        # window, and the model happily continues it.
        kwargs["condition_on_previous_text"] = False
        if opts.word_timestamps and opts.hallucination_silence_s > 0:
            # Needs word timestamps: it works by spotting speech-free stretches
            # inside a segment the decoder claims is speech.
            kwargs["hallucination_silence_threshold"] = opts.hallucination_silence_s
    return kwargs


def _vad_report(info, log) -> None:
    """Say what the VAD actually removed, in minutes of audio."""
    total = getattr(info, "duration", 0.0) or 0.0
    kept = getattr(info, "duration_after_vad", 0.0) or 0.0
    if not total or not kept or kept >= total - 0.5:
        return
    cut = total - kept
    log(
        f"VAD: kept {kept / 60:.1f} min of {total / 60:.1f} min — "
        f"{cut / 60:.1f} min of non-speech never reached the decoder."
    )


def _is_cuda_runtime_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(token in msg for token in ("cublas", "cudnn", "cudart", ".dll", "cuda"))


def pipeline_summary(opts: TranscribeOptions) -> list[str]:
    """What each layer will do to this run, for the job log."""
    lines = []
    if opts.vad_enabled:
        lines.append("VAD: on — silence and noise are cut before the decoder sees them.")
    else:
        lines.append("VAD: off.")
    if opts.hotwords:
        lines.append(
            vocab_bias.summarize(
                [term for term in opts.hotwords.split(", ") if term], opts.hotwords
            )
        )
    if opts.hallucination_guard:
        lines.append(
            "Hallucination guard: on — no carry-over context between windows."
        )
    return lines


class Transcriber:
    _models: dict = {}

    def _get_model(self, name: str, device: str, compute_type: str, log=None):
        key = (name, device, compute_type)
        if key not in Transcriber._models:
            from faster_whisper import WhisperModel
            try:
                Transcriber._models[key] = WhisperModel(
                    name, device=device, compute_type=compute_type
                )
            except Exception as e:
                if device != "cuda":
                    raise
                if log:
                    log(f"GPU load failed ({e}); falling back to CPU…")
                cpu_key = (name, "cpu", "int8")
                if cpu_key not in Transcriber._models:
                    Transcriber._models[cpu_key] = WhisperModel(
                        name, device="cpu", compute_type="int8"
                    )
                return Transcriber._models[cpu_key]
        return Transcriber._models[key]

    # ------------------------------------------------------------------
    def transcribe(
        self,
        recording: Recording,
        audio_path: str,
        opts: TranscribeOptions,
        progress_cb=None,
        log_cb=None,
        should_cancel: ShouldCancel = None,
    ) -> TranscriptResult:
        def log(msg):
            if log_cb:
                log_cb(msg)

        check_cancel(should_cancel, log, message="Cancelled.")
        if opts.engine in CLOUD_ENGINES:
            return self._transcribe_cloud(
                recording, audio_path, opts, progress_cb, log, should_cancel
            )
        device, compute = _resolve_device_compute(opts.device, opts.compute_type)
        for line in pipeline_summary(opts):
            log(line)
        log(f"Loading Whisper '{whisper_models.label(opts.model)}' on {device} ({compute})…")
        model = self._get_model(opts.model, device, compute, log=log)
        language = None if opts.language == "auto" else opts.language

        try:
            if opts.channel_mode == "per_channel":
                return self._transcribe_per_channel(
                    recording, audio_path, model, language, opts, progress_cb, log,
                    should_cancel,
                )
            return self._transcribe_single(
                recording, audio_path, model, language, opts, progress_cb, log,
                should_cancel,
            )
        except JobCancelled:
            raise  # never mistake a cancel for a CUDA fault worth retrying on CPU
        except Exception as e:
            if device != "cuda" or not _is_cuda_runtime_error(e):
                raise
            log(f"GPU transcription failed ({e}); falling back to CPU…")
            cpu_model = self._get_model(opts.model, "cpu", "int8", log=log)
            if opts.channel_mode == "per_channel":
                return self._transcribe_per_channel(
                    recording, audio_path, cpu_model, language, opts, progress_cb, log,
                    should_cancel,
                )
            return self._transcribe_single(
                recording, audio_path, cpu_model, language, opts, progress_cb, log,
                should_cancel,
            )

    # ---- cloud engines -------------------------------------------------
    def _cloud_engine(self, engine: str):
        """The module for a cloud engine. Both expose the same transcribe()."""
        return stt_gemini if engine == ENGINE_GEMINI else stt_elevenlabs

    def _transcribe_cloud(
        self, recording, audio_path, opts, progress_cb, log, should_cancel,
    ) -> TranscriptResult:
        """Cloud transcription. These engines diarize in the same call, so
        pyannote and the HuggingFace token play no part on this path."""
        if opts.channel_mode == "per_channel":
            return self._cloud_per_channel(
                recording, audio_path, opts, progress_cb, log, should_cancel
            )
        return self._cloud_engine(opts.engine).transcribe(
            recording, audio_path, opts, progress_cb, log, should_cancel
        )

    def _cloud_per_channel(
        self, recording, audio_path, opts, progress_cb, log, should_cancel,
    ) -> TranscriptResult:
        """One upload per channel, so channel == speaker as it does locally."""
        engine = self._cloud_engine(opts.engine)
        name = ENGINE_LABELS.get(opts.engine, opts.engine)
        log("Splitting channels…")
        channels = audio_utils.split_channels(audio_path, opts.channel_names or [])
        all_segments: list[Segment] = []
        language = ""
        n = len(channels)
        for i, (label, wav) in enumerate(channels):
            check_cancel(should_cancel, log, message="Cancelled — stopping transcription.")
            log(f"Transcribing {label} with {name}…")
            part = engine.transcribe(
                recording, wav, opts, None, log, should_cancel
            )
            language = language or part.language
            for s in part.segments:
                s.speaker = label
                s.channel = label
            all_segments.extend(part.segments)
            if progress_cb:
                progress_cb((i + 1) / n)
        all_segments.sort(key=lambda s: s.start)
        speakers = list(dict.fromkeys(s.speaker for s in all_segments if s.speaker))
        return TranscriptResult(
            recording=recording,
            segments=all_segments,
            language=language,
            model=engine.model_label(
                (opts.gemini_model if opts.engine == ENGINE_GEMINI
                 else opts.elevenlabs_model) or engine.DEFAULT_MODEL
            ),
            speakers=speakers,
        )

    # ------------------------------------------------------------------
    def _run_whisper(
        self, model, path, language, opts, log, progress_cb, total_dur,
        should_cancel: ShouldCancel = None,
    ):
        segments_iter, info = model.transcribe(path, **transcribe_kwargs(opts, language))
        _vad_report(info, log)
        out: list[Segment] = []
        # faster-whisper decodes lazily, so this loop is where a long
        # transcription can actually be interrupted.
        for seg in segments_iter:
            check_cancel(should_cancel, log, message="Cancelled — stopping transcription.")
            out.append(Segment(start=seg.start, end=seg.end, text=seg.text.strip()))
            if progress_cb and total_dur:
                progress_cb(min(0.95, seg.end / total_dur))
        return out, info.language

    def _transcribe_single(
        self, recording, audio_path, model, language, opts, progress_cb, log,
        should_cancel: ShouldCancel = None,
    ):
        total_dur = recording.duration_seconds or audio_utils.probe(audio_path)["duration"]
        log("Transcribing audio…")
        segments, detected_lang = self._run_whisper(
            model, audio_path, language, opts, log, progress_cb, total_dur, should_cancel
        )

        speakers_order: list[str] = []
        if opts.diarization_enabled and diarization.is_available():
            check_cancel(should_cancel, log, message="Cancelled — skipping diarization.")
            log("Running speaker diarization…")
            try:
                diar = diarization.Diarizer(opts.hf_token, opts.device)
                turns = diar.diarize(
                    audio_path,
                    opts.min_speakers,
                    opts.max_speakers,
                    progress_cb=(lambda f: progress_cb(0.95 + f * 0.05)) if progress_cb else None,
                    log_cb=log,
                    should_cancel=should_cancel,
                )
                mapping = self._stable_speaker_map(turns)
                for seg in segments:
                    raw = diarization.assign_speaker(seg.start, seg.end, turns)
                    seg.speaker = mapping.get(raw) if raw else None
                speakers_order = list(dict.fromkeys(
                    s.speaker for s in segments if s.speaker
                ))
            except JobCancelled:
                # Diarization failing is survivable and the transcript is still
                # worth having; the user pressing Cancel is neither.
                raise
            except Exception as e:
                log(f"Diarization skipped: {e}")
        elif opts.diarization_enabled:
            log("Diarization libraries not installed — continuing without speakers.")

        if progress_cb:
            progress_cb(1.0)
        return TranscriptResult(
            recording=recording, segments=segments,
            language=detected_lang or (language or ""),
            model=opts.model, speakers=speakers_order,
        )

    def _transcribe_per_channel(
        self, recording, audio_path, model, language, opts, progress_cb, log,
        should_cancel: ShouldCancel = None,
    ):
        log("Splitting channels…")
        names = opts.channel_names or []
        channels = audio_utils.split_channels(audio_path, names)
        all_segments: list[Segment] = []
        n = len(channels)
        for i, (label, wav) in enumerate(channels):
            check_cancel(should_cancel, log, message="Cancelled — stopping transcription.")
            log(f"Transcribing {label}…")
            segs, detected = self._run_whisper(
                model, wav, language, opts, log, None,
                recording.duration_seconds, should_cancel,
            )
            for s in segs:
                s.speaker = label
                s.channel = label
            all_segments.extend(segs)
            if progress_cb:
                progress_cb((i + 1) / n)
        all_segments.sort(key=lambda s: s.start)
        speakers = list(dict.fromkeys(s.speaker for s in all_segments if s.speaker))
        return TranscriptResult(
            recording=recording, segments=all_segments,
            language=language or "", model=opts.model, speakers=speakers,
        )

    @staticmethod
    def _stable_speaker_map(turns) -> dict[str, str]:
        """Map raw pyannote labels to Speaker 1/2/… in order of first appearance."""
        order: list[str] = []
        for t in turns:
            if t.speaker not in order:
                order.append(t.speaker)
        return {raw: f"Speaker {i + 1}" for i, raw in enumerate(order)}
