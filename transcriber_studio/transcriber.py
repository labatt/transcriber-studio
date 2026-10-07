# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Local Whisper transcription via faster-whisper, with optional diarization.

Heavy deps (faster-whisper / ctranslate2) are imported lazily.
"""

from __future__ import annotations

import importlib.util
import json
import math
import time
from dataclasses import dataclass

from . import (
    audio_utils,
    diarization,
    stt_elevenlabs,
    stt_gemini,
    stt_mai,
    vocab_bias,
    whisper_models,
)
from . import resume as resume_store
from .hardware import cuda_available
from .job_cancel import JobCancelled, ShouldCancel, check_cancel
from .models import Recording, Segment, TranscriptResult
from .word_segments import words_to_segments

ENGINE_LOCAL = "local"
ENGINE_ELEVENLABS = "elevenlabs"
ENGINE_GEMINI = "gemini"
ENGINE_MAI = "mai"
ENGINE_LABELS = {
    ENGINE_LOCAL: "Local Whisper (faster-whisper)",
    ENGINE_ELEVENLABS: "ElevenLabs Scribe (cloud)",
    ENGINE_GEMINI: "Gemini 3.5 Transcribe (cloud)",
    ENGINE_MAI: "MAI-Transcribe (Azure, cloud)",
}
#: Engines that transcribe and separate speakers in one pass, so pyannote and
#: the HuggingFace token play no part. None of them return a voice embedding
#: either, so an enrolled voiceprint cannot name anyone on these paths - with
#: two exceptions: MAI and ElevenLabs can each be run as a decoder only, with
#: pyannote separating the speakers locally, and then voiceprints work as they
#: do for Whisper.
CLOUD_ENGINES = (ENGINE_ELEVENLABS, ENGINE_GEMINI, ENGINE_MAI)


def mai_decodes_locally(opts: TranscribeOptions) -> bool:
    """MAI for the words, pyannote for the speakers."""
    return opts.engine == ENGINE_MAI and (opts.mai_speakers or "local") == "local"


def elevenlabs_decodes_locally(opts: TranscribeOptions) -> bool:
    """Scribe for the words, pyannote for the speakers.

    Unlike MAI this needs a reason to leave Scribe's own speaker labels
    behind: they work at any length. The reason is voiceprints, which need
    pyannote's clusters, so the local path is taken only when pyannote could
    actually run — a HuggingFace token is saved and speakers are wanted. With
    neither, Scribe keeps labelling speakers itself rather than nobody doing it.
    """
    return (
        opts.engine == ENGINE_ELEVENLABS
        and (opts.elevenlabs_speakers or "local") == "local"
        and opts.diarization_enabled
        and bool((opts.hf_token or "").strip())
    )


def decodes_locally(opts: TranscribeOptions) -> bool:
    """True when pyannote, not the engine, will separate the speakers."""
    return (
        opts.engine == ENGINE_LOCAL
        or mai_decodes_locally(opts)
        or elevenlabs_decodes_locally(opts)
    )


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
    #: The recording was cut to its first N minutes before reaching us. 0 is
    #: the whole thing. Identity only, like ``denoise``: the cut happens in the
    #: job runner, and this keeps a partial transcript from being restored as
    #: a full one.
    limit_minutes: int = 0
    vad_enabled: bool = True
    vad_parameters: dict | None = None
    #: Names, products and jargon to bias the decoder toward. See transcriber_studio.vocab_bias.
    hotwords: str = ""
    hallucination_guard: bool = True
    hallucination_silence_s: float = 2.0
    # --- how willing to put an enrolled name on a diarized speaker ---
    voiceprint_threshold: float = 0.55
    voiceprint_margin: float = 0.10
    voiceprint_min_speech_s: float = 15.0
    #: Names the user has already put on pyannote's raw labels for this run
    #: (SPEAKER_00 -> "Alice"), from listening to samples before transcribing.
    #: These win over anything voiceprint matching would say: the user heard
    #: the voice, the matcher only measured it.
    known_speakers: dict[str, str] | None = None
    #: Penalty applied to tokens the decoder has already emitted. 1.0 is off,
    #: and off is the library default. The guard above stops a hallucination
    #: *carrying over* between windows; neither it nor the VAD does anything
    #: about a decoder that gets stuck repeating itself inside one window, and
    #: this is the knob for that.
    repetition_penalty: float = 1.0
    #: Forbid any n-gram of this length from repeating in a window. 0 is off.
    #: Blunter than the penalty and easier to regret: real speech repeats
    #: itself, so a small value here will refuse to transcribe things that were
    #: genuinely said twice.
    no_repeat_ngram_size: int = 0
    # --- engine choice ---
    engine: str = ENGINE_LOCAL      # local (faster-whisper) | elevenlabs (Scribe)
    elevenlabs_api_key: str = ""
    elevenlabs_model: str = ""
    tag_audio_events: bool = False  # ElevenLabs only: mark laughter, applause…
    #: Who separates the speakers on ElevenLabs. "local" runs pyannote here on
    #: Scribe's words, which returns the embeddings voiceprints need; "scribe"
    #: keeps Scribe's own labels, which carry no voice data. See
    #: elevenlabs_decodes_locally() for when "local" actually applies.
    elevenlabs_speakers: str = "local"  # local | scribe
    gemini_api_key: str = ""        # the Google AI key, shared with AI Cleanup
    gemini_model: str = ""
    gemini_mode: str = ""           # smart | verbatim
    # --- MAI-Transcribe (Azure Speech, enhanced mode) ---
    mai_api_key: str = ""
    mai_region: str = "eastus"      # only six regions host the model
    mai_model: str = ""             # blank => stt_mai.DEFAULT_MODEL
    mai_style: str = "verbatim"     # verbatim | clean
    #: Send the glossary and vocab-bias terms as Azure phrase-list hints.
    mai_send_phrases: bool = True
    #: Who separates the speakers. "local" runs pyannote here on MAI's words,
    #: which works at any length and returns the embeddings voiceprints need.
    #: "mai" asks Azure to do it in the same request - measured to fail at
    #: 15 minutes and above, with an error that reads like a network timeout.
    mai_speakers: str = "local"     # local | mai


def _cloud_model_id(opts: TranscribeOptions) -> str:
    """Whichever model field belongs to the engine in use."""
    return {
        ENGINE_GEMINI: opts.gemini_model,
        ENGINE_MAI: opts.mai_model,
    }.get(opts.engine, opts.elevenlabs_model)


def expected_model_label(opts: TranscribeOptions) -> str:
    """The ``TranscriptResult.model`` these options would produce, without
    running anything. What a banked transcript is matched against when its
    key no longer lines up with the current options."""
    from . import stt_elevenlabs, stt_gemini

    if opts.engine == ENGINE_MAI:
        return stt_mai.model_label(opts.mai_model)
    if opts.engine == ENGINE_GEMINI:
        return stt_gemini.model_label(opts.gemini_model or stt_gemini.DEFAULT_MODEL)
    if opts.engine == ENGINE_ELEVENLABS:
        return stt_elevenlabs.model_label(opts.elevenlabs_model or stt_elevenlabs.DEFAULT_MODEL)
    return str(opts.model)


def faster_whisper_available() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


def with_known_speakers(
    recognized: dict[str, str], known: dict[str, str] | None, log=None,
) -> dict[str, str]:
    """Recognised names, overridden by the ones the user put on by ear.

    Both are keyed by pyannote's raw label. A name the user typed after
    listening to a speaker is the better authority, so it replaces whatever
    the matcher proposed for that label — and is announced, because a reader
    of the log should be able to tell which names were heard and which were
    measured.
    """
    merged = dict(recognized or {})
    for label, name in (known or {}).items():
        clean = (name or "").strip()
        if not clean:
            continue
        if log and merged.get(label) != clean:
            log(f"Speaker {label} is {clean} (named from the samples you heard).")
        merged[label] = clean
    return merged


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
    # Only sent when actually turned on. Passing the library's own defaults
    # explicitly would be a no-op that still reads, to anyone looking at the
    # call, like a deliberate choice about repetition.
    if opts.repetition_penalty and opts.repetition_penalty != 1.0:
        kwargs["repetition_penalty"] = float(opts.repetition_penalty)
    if opts.no_repeat_ngram_size and opts.no_repeat_ngram_size > 0:
        kwargs["no_repeat_ngram_size"] = int(opts.no_repeat_ngram_size)
    return kwargs


#: How often the decoder says how far it has got. Often enough to prove it is
#: alive, rare enough not to bury the log.
DECODE_REPORT_SECONDS = 30

#: Below this the decoder was, by its own measure, guessing. Not a defect on its
#: own — a quiet aside scores low and is transcribed perfectly well — so nothing
#: is discarded on the strength of it. It is a pointer for a human ear.
LOW_CONFIDENCE = 0.50
#: Text this compressible is the signature of a decode loop: the same phrase
#: emitted until the window ran out. faster-whisper drops segments above 2.4
#: outright, so anything reported here squeaked under that bar.
SUSPICIOUS_COMPRESSION = 2.0
#: Enough of the audio flagged as non-speech that a reader should check whether
#: there was anything there to transcribe.
SUSPICIOUS_NO_SPEECH = 0.60
#: One or two shaky lines in an hour is normal speech, not a finding.
CONFIDENCE_REPORT_MINIMUM = 3


def _as_float(value) -> float | None:
    """A float, or None when the engine gave nothing usable.

    Confidence is optional everywhere: the cloud engines report none, and a
    restored decode from an older run has none either. None has to mean
    "unknown" all the way through rather than silently becoming zero, which
    would read as "certainly wrong".
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def confidence_report(segments: list[Segment]) -> list[str]:
    """What the decoder was unsure about, for the job log.

    Says nothing at all when there is nothing to say. A log line per shaky
    segment would bury the run; a count and the worst offender is enough to
    tell a reader whether this transcript needs an ear on it.
    """
    scored = [s for s in segments if s.confidence is not None]
    if not scored:
        return []
    lines: list[str] = []
    low = [s for s in scored if s.confidence < LOW_CONFIDENCE]
    if len(low) >= CONFIDENCE_REPORT_MINIMUM:
        worst = min(low, key=lambda s: s.confidence)
        share = len(low) / len(scored)
        lines.append(
            f"Confidence: {len(low)} of {len(scored)} segment(s) ({share:.0%}) below "
            f"{LOW_CONFIDENCE:.0%} — worst at {_clock(worst.start)} "
            f"({worst.confidence:.0%}). Worth a listen before trusting those lines."
        )
    looping = [
        s for s in segments
        if s.compression_ratio is not None and s.compression_ratio >= SUSPICIOUS_COMPRESSION
    ]
    if looping:
        worst = max(looping, key=lambda s: s.compression_ratio)
        lines.append(
            f"Repetition: {len(looping)} segment(s) look like the decoder repeating "
            f"itself — worst at {_clock(worst.start)}. If that is what happened, "
            "raise the repetition penalty in Settings and run it again."
        )
    silent = [
        s for s in segments
        if s.no_speech_prob is not None and s.no_speech_prob >= SUSPICIOUS_NO_SPEECH
    ]
    if len(silent) >= CONFIDENCE_REPORT_MINIMUM:
        lines.append(
            f"Non-speech: {len(silent)} segment(s) produced text over audio the "
            "decoder rated as probably not speech."
        )
    return lines


def _clock(seconds: float) -> str:
    total = int(max(0.0, seconds))
    return f"{total // 60}:{total % 60:02d}"


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
    if opts.repetition_penalty and opts.repetition_penalty != 1.0:
        lines.append(
            f"Repetition penalty: {opts.repetition_penalty:.2f} — the decoder is "
            "discouraged from reusing words it has already emitted."
        )
    if opts.no_repeat_ngram_size and opts.no_repeat_ngram_size > 0:
        lines.append(
            f"Repeat block: no {opts.no_repeat_ngram_size}-word sequence may occur "
            "twice in a window — including ones that genuinely were said twice."
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
        resume=None,
    ) -> TranscriptResult:
        def log(msg):
            if log_cb:
                log_cb(msg)

        check_cancel(should_cancel, log, message="Cancelled.")
        if mai_decodes_locally(opts):
            # The local pipeline with MAI standing in for Whisper: no model to
            # load, and the speaker stage that follows is pyannote's.
            for line in pipeline_summary(opts):
                log(line)
            log("MAI-Transcribe decodes the words; speakers are detected here.")
            return self._transcribe_single(
                recording, audio_path, None, None, opts, progress_cb, log,
                should_cancel, resume, decoder=stt_mai.decode,
            )
        if elevenlabs_decodes_locally(opts):
            for line in pipeline_summary(opts):
                log(line)
            log(
                "ElevenLabs Scribe decodes the words; speakers are detected here, "
                "so enrolled voices can be recognised."
            )
            return self._transcribe_single(
                recording, audio_path, None, None, opts, progress_cb, log,
                should_cancel, resume, decoder=stt_elevenlabs.decode,
            )
        if opts.engine in CLOUD_ENGINES:
            if (
                opts.engine == ENGINE_ELEVENLABS and opts.diarization_enabled
                and (opts.elevenlabs_speakers or "local") == "local"
            ):
                log(
                    "No HuggingFace token saved, so Scribe labels the speakers "
                    "itself — voiceprints cannot name anyone on this run."
                )
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
                should_cancel, resume,
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
                should_cancel, resume,
            )

    # ---- cloud engines -------------------------------------------------
    def _cloud_engine(self, engine: str):
        """The module for a cloud engine. They all expose the same transcribe()."""
        return {
            ENGINE_GEMINI: stt_gemini,
            ENGINE_MAI: stt_mai,
        }.get(engine, stt_elevenlabs)

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
            model=engine.model_label(_cloud_model_id(opts) or engine.DEFAULT_MODEL),
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
        # An hour of audio is many minutes of decoding with nothing to show for
        # it. Say something periodically, so the log is evidence of work rather
        # than a gap the user has to interpret.
        last_report = time.monotonic()
        # Word timings are computed anyway when word_timestamps is on (they are
        # what the hallucination guard watches). Keeping them lets speakers be
        # assigned per word instead of per segment, which is the difference
        # between splitting a segment where the floor changes hands and filing
        # both halves under whoever happened to talk longer.
        words: list[dict] = []
        # faster-whisper decodes lazily, so this loop is where a long
        # transcription can actually be interrupted.
        for seg in segments_iter:
            check_cancel(should_cancel, log, message="Cancelled — stopping transcription.")
            out.append(
                Segment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text.strip(),
                    # Computed by the decoder either way. Kept because a
                    # transcript that cannot say which lines it is unsure of
                    # makes the reader check all of them or none.
                    avg_logprob=_as_float(getattr(seg, "avg_logprob", None)),
                    no_speech_prob=_as_float(getattr(seg, "no_speech_prob", None)),
                    compression_ratio=_as_float(getattr(seg, "compression_ratio", None)),
                )
            )
            for w in getattr(seg, "words", None) or []:
                words.append({"type": "word", "text": w.word,
                              "start": float(w.start), "end": float(w.end),
                              "probability": _as_float(getattr(w, "probability", None))})
            if progress_cb and total_dur:
                progress_cb(min(0.95, seg.end / total_dur))
            if total_dur and time.monotonic() - last_report >= DECODE_REPORT_SECONDS:
                last_report = time.monotonic()
                log(
                    f"Transcribed {seg.end / 60:.0f} of {total_dur / 60:.0f} min "
                    f"({seg.end / total_dur:.0%}) — {len(out)} segment(s) so far."
                )
        for line in confidence_report(out):
            log(line)
        return out, info.language, words

    def _transcribe_single(
        self, recording, audio_path, model, language, opts, progress_cb, log,
        should_cancel: ShouldCancel = None, resume=None, decoder=None,
    ):
        # A disabled log rather than an `if resume:` around every use of it.
        bank = resume if resume is not None else resume_store.ResumeLog(None)
        decode_key = resume_store.decode_key(recording, opts)

        segments, detected_lang, words = self._decode_or_restore(
            bank, decode_key, recording, audio_path, model, language, opts,
            progress_cb, log, should_cancel, decoder=decoder,
        )

        speakers_order: list[str] = []
        speaker_vectors: dict[str, list[float]] = {}
        speaker_seconds: dict[str, float] = {}
        if opts.diarization_enabled and diarization.is_available():
            # Bank the decode before starting diarization, not after — and
            # before the cancel check, since re-running Whisper costs GPU time
            # whether the run ended by crash or by choice. Diarizing an hour of
            # audio is minutes of work, and dying inside it used to discard the
            # far more expensive decode along with it.
            bank.record(
                decode_key,
                json.dumps(
                    resume_store.decode_to_dict(segments, detected_lang, words),
                    ensure_ascii=False,
                ),
                stage=resume_store.DECODE_STAGE,
                segments=len(segments),
            )
            check_cancel(should_cancel, log, message="Cancelled — skipping diarization.")
            log("Running speaker diarization…")
            try:
                diar = diarization.Diarizer(opts.hf_token, opts.device)
                diarized = diar.diarize(
                    audio_path,
                    opts.min_speakers,
                    opts.max_speakers,
                    progress_cb=(lambda f: progress_cb(0.95 + f * 0.05)) if progress_cb else None,
                    log_cb=log,
                    should_cancel=should_cancel,
                )
                names = self._recognized_names(
                    diarized, log,
                    threshold=opts.voiceprint_threshold,
                    margin=opts.voiceprint_margin,
                    min_speech=opts.voiceprint_min_speech_s,
                    source=recording.display_name,
                )
                names = with_known_speakers(names, opts.known_speakers, log)
                segments, speakers_order = self._apply_speakers(
                    segments, words, diarized, log, names
                )
                speaker_vectors, speaker_seconds = self._speaker_voice_data(
                    diarized, names
                )
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
            model=expected_model_label(opts) if decoder is not None else opts.model,
            speakers=speakers_order,
            speaker_embeddings=speaker_vectors,
            speaker_seconds=speaker_seconds,
            words=words,
        )

    def _decode_or_restore(
        self, bank, key, recording, audio_path, model, language, opts,
        progress_cb, log, should_cancel, decoder=None,
    ):
        """The decode pass, reused from an interrupted run where possible.

        ``decoder`` swaps Whisper for another source of the same
        ``(segments, language, words)`` triple - MAI-Transcribe, today - so a
        cloud decode gets banked and restored exactly as a local one does.
        """
        saved = bank.get(key)
        if saved:
            try:
                segments, detected_lang, words = resume_store.decode_from_dict(
                    json.loads(saved)
                )
                log(
                    f"Restored {len(segments)} transcribed segment(s) from an interrupted "
                    f"run — going straight to speaker detection."
                )
                if progress_cb:
                    progress_cb(0.95)
                return segments, detected_lang, words
            except Exception as e:
                log(f"Saved transcription unusable ({e}) — transcribing again.")

        if decoder is not None:
            return decoder(
                recording, audio_path, opts, progress_cb, log, should_cancel
            )
        # The file, not the listing: a run limited to the first ten minutes
        # is decoding a ten-minute file, whatever the recording's length.
        total_dur = audio_utils.probe(audio_path)["duration"] or recording.duration_seconds
        log("Transcribing audio…")
        return self._run_whisper(
            model, audio_path, language, opts, log, progress_cb, total_dur, should_cancel
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
            segs, detected, _words = self._run_whisper(
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

    def _apply_speakers(self, segments, words, diarized, log, names=None):
        """Label the transcript from the diarization result.

        Per word where word timings exist, so a segment containing a speaker
        change is split at the change rather than attributed whole. Falls back
        to per-segment overlap when they do not, which is the old behaviour and
        still correct for a segment with only one speaker in it.

        ``names`` is passed in when the caller has already worked out who was
        recognised, so the log says it once rather than once per code path.
        """
        turns = diarized.turns if hasattr(diarized, "turns") else diarized
        if names is None:
            names = self._recognized_names(diarized, log)

        if words:
            for word in words:
                # Cloud engines interleave spacing entries between words; they
                # carry no timing and belong to whoever is speaking around them.
                if word.get("type", "word") != "word" or "start" not in word:
                    continue
                raw = diarization.assign_speaker(word["start"], word["end"], turns)
                if raw is None:
                    # In a pause between turns: spoken by whoever is nearest,
                    # not by nobody.
                    raw = diarization.nearest_speaker(word["start"], word["end"], turns)
                word["speaker_id"] = raw or ""
            regrouped, speakers = words_to_segments(words, diarized=True, names=names)
            if regrouped:
                extra = len(regrouped) - len(segments)
                if extra > 0:
                    log(
                        f"Speakers: {len(regrouped)} turn(s) from {len(segments)} "
                        f"decoded segment(s) — {extra} split where the speaker changed."
                    )
                return regrouped, speakers

        mapping = self._stable_speaker_map(turns, names)
        for seg in segments:
            raw = diarization.assign_speaker(seg.start, seg.end, turns)
            if raw is None:
                raw = diarization.nearest_speaker(seg.start, seg.end, turns)
            seg.speaker = mapping.get(raw) if raw else None
        return segments, list(dict.fromkeys(s.speaker for s in segments if s.speaker))

    @classmethod
    def _speaker_voice_data(cls, diarized, names) -> tuple[dict, dict]:
        """Voice vectors and speaking time, keyed by the label the reader sees.

        Diarization works in SPEAKER_00 terms; everything downstream — the
        rename dialog especially — works in "Speaker 1" and "Alice" terms. This
        is the translation, so naming someone in the dialog can enrol the right
        voice without knowing pyannote's labels exist.
        """
        embeddings = getattr(diarized, "embeddings", None) or {}
        if not embeddings:
            return {}, {}
        turns = diarized.turns if hasattr(diarized, "turns") else diarized
        mapping = cls._stable_speaker_map(turns, names)
        seconds = diarized.speech_seconds() if hasattr(diarized, "speech_seconds") else {}
        vectors = {
            mapping[label]: vector
            for label, vector in embeddings.items()
            if label in mapping
        }
        totals = {
            mapping[label]: value
            for label, value in seconds.items()
            if label in mapping
        }
        return vectors, totals

    @staticmethod
    def _recognized_names(
        diarized, log, *, threshold=None, margin=None, min_speech=None, source="",
    ) -> dict[str, str]:
        """Which diarized speakers are people this app has been taught.

        Never allowed to fail a transcription: an unrecognised speaker is the
        normal case, and a broken voiceprint store should cost names, not the
        transcript.

        Every attempt is written to the match log on the way past — the
        rejections too, since a threshold can only be judged against the scores
        it turned away.
        """
        embeddings = getattr(diarized, "embeddings", None)
        if not embeddings:
            return {}
        try:
            from . import voiceprints

            matches = voiceprints.identify(
                dict(embeddings),
                diarized.speech_seconds(),
                threshold=threshold,
                margin=margin,
                min_speech=min_speech,
            )
        except Exception as e:
            log(f"Speaker recognition skipped: {e}")
            return {}
        voiceprints.record_matches(matches, source=source)
        for line in voiceprints.describe(matches):
            log(line)
        return {m.label: m.name for m in matches if m.named}

    @staticmethod
    def _stable_speaker_map(turns, names: dict[str, str] | None = None) -> dict[str, str]:
        """Map raw pyannote labels to names, or Speaker 1/2/… where unknown.

        The numbering counts only the unrecognised speakers, so naming one
        person does not leave a gap where their number used to be.
        """
        names = names or {}
        order: list[str] = []
        for t in turns:
            if t.speaker not in order:
                order.append(t.speaker)
        mapping: dict[str, str] = {}
        unnamed = 0
        for raw in order:
            named = (names.get(raw) or "").strip()
            if named:
                mapping[raw] = named
                continue
            unnamed += 1
            mapping[raw] = f"Speaker {unnamed}"
        return mapping
