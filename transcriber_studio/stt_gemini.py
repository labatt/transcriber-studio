# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Gemini 3.5 Transcribe: cloud transcription with diarization in one pass.

Google's speech model, reached through the Interactions API. Like ElevenLabs
Scribe it transcribes and separates speakers together, so pyannote and the
HuggingFace token play no part on this path.

Three things the documentation does not tell you, found by asking the API:

* ``output_text`` on the response is null. The transcript lives in
  ``steps[].content[].text``, with word timings in that content's
  ``annotations``.
* ``custom_vocabulary`` is rejected outright alongside either diarization or
  timestamps — "custom_vocabulary is incompatible with diarization". The app
  needs both of those, so vocabulary biasing is not available on this engine
  and the UI says so rather than dropping the terms quietly.
* ``smart`` mode rejects ``diarization_mode`` and ``timestamp_granularities``
  as unknown parameters. It is prose only. ``verbatim`` is therefore the
  default here, being the only mode that fills in speakers and timings.
* Word annotations carry no spacing of their own; the gaps have to be read out
  of the content text using each word's start_index and end_index, or the
  transcript comes back as onerunontogetherstring.
* Offsets come back as strings with a trailing "s", and not always with a
  decimal point: "0.200s" and "11s" both occur.

Audio goes through the Files API first; the Interactions call takes a URI, not
bytes.
"""

from __future__ import annotations

import json
import mimetypes
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .job_cancel import ShouldCancel, check_cancel
from .models import Recording, Segment, TranscriptResult
from .word_segments import words_to_segments

BASE_URL = "https://generativelanguage.googleapis.com"
UPLOAD_URL = f"{BASE_URL}/upload/v1beta/files"
INTERACTIONS_URL = f"{BASE_URL}/v1beta/interactions"
MODELS_URL = f"{BASE_URL}/v1beta/models"

MODELS = ["gemini-3.5-transcribe"]
DEFAULT_MODEL = "gemini-3.5-transcribe"

#: The only two the API accepts — it names them itself when given anything else.
#:
#: They are not two flavours of the same thing. Asked directly, the API rejects
#: `diarization_mode` and `timestamp_granularities` outright in smart mode:
#: it returns one block of punctuated prose with no speakers and no timings.
#: Verbatim is the only mode that fills in this app's data model, which is why
#: it is the default despite being the rougher read.
MODES = ("verbatim", "smart")
DEFAULT_MODE = "verbatim"
MODE_LABELS = {
    "verbatim": "Verbatim — speakers and timestamps, every word as spoken",
    "smart": "Smart — punctuated prose, but no speakers and no timestamps",
}

#: Modes that can carry speaker labels and word timings.
STRUCTURED_MODES = ("verbatim",)

#: The Files API's own ceiling. Well beyond anything this app will send.
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
#: What Google documents for a single request, quoting the Limitations section
#: of https://ai.google.dev/gemini-api/docs/transcribe:
#:
#:   "Standard unary requests support audio files up to 1 hour."
#:   "Audio processing is limited to 30 minutes when features like speaker
#:    diarization or word-level timestamps are enabled."
#:
#: Note "or word-level timestamps": the lower ceiling applies in verbatim mode
#: whether or not speakers were asked for, because that mode always requests
#: word timings. Longer recordings are still sent — the API is the authority —
#: but the log warns first.
MAX_MINUTES_PLAIN = 60
MAX_MINUTES_WITH_FEATURES = 30

#: What the API actually does, which is not what the docs say. Probed against
#: gemini-3.5-transcribe in verbatim mode with diarization on: 35, 46, 51 and
#: 54 minutes were all accepted; 57 and 80 minutes came back
#: "Invalid input received." with no further detail. The documented 30 minute
#: figure is not enforced, so warning at 30 cried wolf while the real wall at
#: ~55 arrived as an opaque 400 after uploading the whole file.
#: Conservative by a couple of minutes, because the boundary was bracketed
#: rather than pinned exactly and may move.
PRACTICAL_MINUTES_WITH_FEATURES = 54

READ_TIMEOUT = 1800         # a long recording takes minutes to come back
UPLOAD_TIMEOUT = 900
RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 3


class GeminiError(RuntimeError):
    """Anything that stops a Gemini transcription, phrased for the job log."""


def model_label(model_id: str) -> str:
    return model_id or DEFAULT_MODEL


def mime_type_for(path: str) -> str:
    guessed, _ = mimetypes.guess_type(path)
    if guessed and guessed.startswith("audio/"):
        return guessed
    # The recorder's own files are mp3; anything unrecognised is sent as wav,
    # which is what the app's own denoiser and channel splitter produce.
    return "audio/mpeg" if Path(path).suffix.lower() == ".mp3" else "audio/wav"


# ---- transport ---------------------------------------------------------
def _request(
    url: str,
    *,
    api_key: str,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 120,
    want_headers: bool = False,
) -> Any:
    head = {"x-goog-api-key": api_key}
    head.update(headers or {})
    request = urllib.request.Request(url, data=data, headers=head)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
        if want_headers:
            return body, dict(response.headers)
        return json.loads(body) if body else {}


def _error_message(exc: urllib.error.HTTPError) -> str:
    """Google's error body, unwrapped, or the plain HTTP status."""
    try:
        payload = json.loads(exc.read())
    except Exception:
        return f"HTTP {exc.code}"
    error = payload.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error.get("status") or f"HTTP {exc.code}")
    return str(error or f"HTTP {exc.code}")


def test_key(api_key: str) -> str:
    """Check a key and confirm the transcription model is available to it."""
    if not api_key.strip():
        raise GeminiError("Enter a Google AI API key first.")
    try:
        data = _request(f"{MODELS_URL}?pageSize=200", api_key=api_key.strip(), timeout=45)
    except urllib.error.HTTPError as exc:
        raise GeminiError(_error_message(exc)) from exc
    except Exception as exc:
        raise GeminiError(f"Could not reach Google AI: {exc}") from exc

    names = {m.get("name", "").split("/")[-1] for m in data.get("models", [])}
    if DEFAULT_MODEL not in names:
        raise GeminiError(
            f"The key works, but {DEFAULT_MODEL} is not available to it "
            f"({len(names)} other models are). It may not be enabled for your project."
        )
    return f"Google AI key works — {DEFAULT_MODEL} is available."


# ---- Files API ---------------------------------------------------------
def upload(audio_path: str, api_key: str, log=None, should_cancel: ShouldCancel = None) -> str:
    """Put the audio in the Files API and return its URI.

    Resumable protocol, because that is the one the API documents; the file is
    small enough to finalise in a single chunk.
    """
    path = Path(audio_path)
    payload = path.read_bytes()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise GeminiError(
            f"{path.name} is {len(payload) / 1e9:.1f} GB; the Files API takes 2 GB."
        )
    check_cancel(should_cancel, log, message="Gemini: cancelled before upload.")
    if log:
        log(f"Gemini: uploading {len(payload) / 1e6:.1f} MB…")

    mime = mime_type_for(audio_path)
    try:
        _, headers = _request(
            UPLOAD_URL,
            api_key=api_key,
            data=json.dumps({"file": {"display_name": path.stem}}).encode(),
            headers={
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(len(payload)),
                "X-Goog-Upload-Header-Content-Type": mime,
                "Content-Type": "application/json",
            },
            timeout=120,
            want_headers=True,
        )
    except urllib.error.HTTPError as exc:
        raise GeminiError(f"Upload could not start: {_error_message(exc)}") from exc

    location = headers.get("X-Goog-Upload-URL") or headers.get("x-goog-upload-url")
    if not location:
        raise GeminiError("The Files API did not return an upload URL.")

    check_cancel(should_cancel, log, message="Gemini: cancelled during upload.")
    try:
        done = _request(
            location,
            api_key=api_key,
            data=payload,
            headers={"X-Goog-Upload-Offset": "0",
                     "X-Goog-Upload-Command": "upload, finalize"},
            timeout=UPLOAD_TIMEOUT,
        )
    except urllib.error.HTTPError as exc:
        raise GeminiError(f"Upload failed: {_error_message(exc)}") from exc

    info = done.get("file", done)
    uri = info.get("uri")
    if not uri:
        raise GeminiError("The upload finished but returned no file URI.")
    if info.get("state") not in (None, "ACTIVE"):
        uri = _wait_until_active(info, api_key, log, should_cancel)
    if log:
        log("Gemini: upload complete.")
    return uri


def _wait_until_active(info: dict, api_key: str, log, should_cancel) -> str:
    """Audio is usually ACTIVE immediately; poll briefly in case it is not."""
    name = info.get("name", "")
    for _ in range(30):
        check_cancel(should_cancel, log, message="Gemini: cancelled while waiting.")
        time.sleep(2)
        try:
            current = _request(f"{BASE_URL}/v1beta/{name}", api_key=api_key, timeout=45)
        except Exception:
            break
        state = current.get("state")
        if state == "ACTIVE":
            return current.get("uri", info.get("uri", ""))
        if state == "FAILED":
            raise GeminiError("Google could not process the uploaded audio.")
    return info.get("uri", "")


# ---- the transcription call --------------------------------------------
def build_config(opts) -> dict[str, Any]:
    """The transcription_config for this run.

    Vocabulary biasing is deliberately absent: the API rejects
    custom_vocabulary together with either diarization or timestamps, and this
    app needs both — timestamps to place segments at all, diarization to say
    who spoke. The Options panel says so where the setting lives.
    """
    requested = (getattr(opts, "gemini_mode", "") or DEFAULT_MODE).strip().lower()
    chosen = requested if requested in MODES else DEFAULT_MODE
    mode: dict[str, Any] = {"type": chosen}
    # Smart mode takes neither of these — the API rejects the parameters
    # rather than ignoring them, so sending either would fail the whole job.
    if chosen in STRUCTURED_MODES:
        mode["timestamp_granularities"] = ["word"]
        if getattr(opts, "diarization_enabled", True):
            mode["diarization_mode"] = "speaker"
    config: dict[str, Any] = {"mode": mode}
    language = getattr(opts, "language", "auto")
    if language and language != "auto":
        config["language_codes"] = [language]
    return config


def length_ceiling(config: dict[str, Any]) -> int:
    """The per-request limit for a config, in minutes — measured, not documented.

    Keyed on what the request actually asks for rather than on diarization
    alone: word timestamps drop the ceiling by themselves, and verbatim mode
    always asks for them.
    """
    mode = config.get("mode") or {}
    if mode.get("diarization_mode") or mode.get("timestamp_granularities"):
        return PRACTICAL_MINUTES_WITH_FEATURES
    return MAX_MINUTES_PLAIN


def _post_interaction(uri: str, mime: str, opts, api_key: str, log, should_cancel) -> dict:
    body = json.dumps({
        "model": (getattr(opts, "gemini_model", "") or DEFAULT_MODEL),
        "input": [{"type": "audio", "uri": uri, "mime_type": mime}],
        "generation_config": {"transcription_config": build_config(opts)},
    }).encode()

    last_error = ""
    for attempt in range(MAX_ATTEMPTS):
        check_cancel(should_cancel, log, message="Gemini: cancelled.")
        try:
            return _request(
                INTERACTIONS_URL,
                api_key=api_key,
                data=body,
                headers={"Content-Type": "application/json"},
                timeout=READ_TIMEOUT,
            )
        except urllib.error.HTTPError as exc:
            last_error = _error_message(exc)
            if exc.code not in RETRY_STATUSES or attempt == MAX_ATTEMPTS - 1:
                raise GeminiError(last_error) from exc
        except Exception as exc:
            last_error = str(exc)
            if attempt == MAX_ATTEMPTS - 1:
                raise GeminiError(f"Gemini request failed: {last_error}") from exc
        wait = 2 ** attempt
        if log:
            log(f"Gemini: {last_error} — retrying in {wait}s…")
        time.sleep(wait)
    raise GeminiError(last_error or "Gemini request failed.")


# ---- response -> segments ----------------------------------------------
def parse_offset(value: Any) -> float:
    """"0.200s", "11s", 1.5 and None all become seconds."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().rstrip("s")
    try:
        return float(text)
    except ValueError:
        return 0.0


def word_annotations(response: dict) -> list[dict]:
    """Every word_info annotation, in order, as the shared grouper wants them.

    The transcript is under steps[].content[], not the output_text the docs
    name — that field comes back null.

    Unlike Scribe, Gemini reports no spacing between words: each annotation
    carries start_index/end_index into the content's own text instead. Reading
    the gap out of that text is what keeps the words from running together —
    and it is more faithful than joining with spaces, since it preserves
    whatever the model actually put between them.
    """
    words: list[dict] = []
    for step in response.get("steps") or []:
        for content in step.get("content") or []:
            full = content.get("text") or ""
            previous_end: int | None = None
            for annotation in content.get("annotations") or []:
                if annotation.get("type") != "word_info":
                    continue
                start_index = annotation.get("start_index")
                if previous_end is not None and isinstance(start_index, int):
                    gap = full[previous_end:start_index]
                    if gap:
                        words.append({"type": "spacing", "text": gap})
                words.append({
                    "type": "word",
                    "text": annotation.get("text", ""),
                    "start": parse_offset(annotation.get("start_offset")),
                    "end": parse_offset(annotation.get("end_offset")),
                    "speaker_id": annotation.get("speaker") or "",
                })
                end_index = annotation.get("end_index")
                previous_end = end_index if isinstance(end_index, int) else None
    return words


def plain_text(response: dict) -> str:
    """The transcript as one string, for when there are no word annotations."""
    parts = [
        content.get("text", "")
        for step in response.get("steps") or []
        for content in step.get("content") or []
        if content.get("type") == "text"
    ]
    return "".join(parts).strip() or str(response.get("output_text") or "").strip()


# ---- the engine entry point --------------------------------------------
def transcribe(
    recording: Recording,
    audio_path: str,
    opts,
    progress_cb=None,
    log_cb=None,
    should_cancel: ShouldCancel = None,
) -> TranscriptResult:
    """Transcribe one file with Gemini, returning the app's own result type."""
    def log(message: str) -> None:
        if log_cb:
            log_cb(message)

    api_key = (getattr(opts, "gemini_api_key", "") or "").strip()
    if not api_key:
        raise GeminiError(
            "No Google AI API key saved. Add one in Settings, or switch engines."
        )
    model = getattr(opts, "gemini_model", "") or DEFAULT_MODEL
    diarize = bool(getattr(opts, "diarization_enabled", True))
    config = build_config(opts)
    mode = config["mode"]["type"]
    # Only claim speaker separation when it was actually asked for: smart mode
    # cannot do it, so saying so there would be a promise the run does not keep.
    separating = "diarization_mode" in config["mode"]
    log(
        f"Gemini: {model}, {mode} mode"
        + (" with speaker separation" if separating else "")
    )
    if diarize and not separating:
        log("Gemini: smart mode cannot separate speakers — the transcript will be unlabelled.")

    minutes = (recording.duration_seconds or 0) / 60
    ceiling = length_ceiling(config)
    if minutes > ceiling:
        # Refusing here rather than uploading a hundred megabytes to be told
        # "Invalid input received." with nothing to act on.
        raise GeminiError(
            f"This recording is {minutes:.0f} minutes and Gemini refuses anything "
            f"over about {ceiling} in {mode} mode — it answers a longer one with "
            f'"Invalid input received." and no explanation.\n\n'
            f"Either split the recording and run the parts separately, or use the "
            f"local Whisper engine, which has no length limit."
        )
    if not recording.duration_seconds:
        # Nothing to check against; the API is the only thing that will say no.
        log("Gemini: recording length unknown — cannot check it against the limit first.")

    if progress_cb:
        progress_cb(0.05)
    uri = upload(audio_path, api_key, log, should_cancel)
    if progress_cb:
        progress_cb(0.35)

    log("Gemini: transcribing…")
    response = _post_interaction(
        uri, mime_type_for(audio_path), opts, api_key, log, should_cancel
    )
    if progress_cb:
        progress_cb(0.9)

    words = word_annotations(response)
    if words:
        segments, speakers = words_to_segments(words, diarized=diarize)
    else:
        # Smart mode, or nothing recognised. Keep the prose rather than lose it,
        # but be explicit that the timeline is not real.
        text = plain_text(response)
        if not text:
            log("Gemini: nothing was transcribed — the audio may contain no speech.")
        elif mode == "smart":
            log(
                "Gemini: smart mode returns prose only — no speaker labels and no "
                "timestamps, so subtitles and per-line times will be empty. Use "
                "verbatim mode if you need either."
            )
        segments = [Segment(start=0.0, end=0.0, text=text)] if text else []
        speakers = []

    usage = response.get("usage") or {}
    if usage.get("total_tokens"):
        log(f"Gemini: {usage['total_tokens']:,} tokens billed.")
    log(f"Gemini: {len(segments)} segment(s), {len(speakers)} speaker(s).")
    if progress_cb:
        progress_cb(1.0)

    return TranscriptResult(
        recording=recording,
        segments=segments,
        language=getattr(opts, "language", "") if getattr(opts, "language", "auto") != "auto" else "",
        model=model_label(model),
        speakers=speakers,
    )
