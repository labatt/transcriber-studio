# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""ElevenLabs Scribe speech-to-text: a cloud alternative to local Whisper.

Scribe transcribes and diarizes in the same call, so choosing this engine
replaces both faster-whisper and pyannote — no GPU, no HuggingFace token, and
no local model download. The audio does leave the machine, which is the whole
trade: pick it in the Options panel per run.

The API returns word-level timings; this module groups those words back into
the speaker-turn segments the rest of the app formats and exports.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import requests

from .job_cancel import ShouldCancel, check_cancel
from .models import Recording, Segment, TranscriptResult

API_URL = "https://api.elevenlabs.io/v1/speech-to-text"
ACCOUNT_URL = "https://api.elevenlabs.io/v1/user/subscription"

#: Scribe models, newest first. The API rejects anything else.
MODELS = ["scribe_v2", "scribe_v1", "scribe_v1_experimental"]
DEFAULT_MODEL = "scribe_v1"

MAX_UPLOAD_BYTES = 5 * 1024 * 1024 * 1024      # 5 GB, per the API docs
MAX_SPEAKERS = 32

CONNECT_TIMEOUT = 30
READ_TIMEOUT = 1800         # a long recording can take minutes to come back
RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 3

# Scribe reports ISO-639-3; the rest of the app shows the 639-1 codes Whisper
# uses. Anything unlisted passes through as-is.
_LANG_3_TO_1 = {
    "eng": "en", "spa": "es", "fra": "fr", "deu": "de", "ita": "it",
    "por": "pt", "nld": "nl", "jpn": "ja", "zho": "zh", "cmn": "zh",
    "kor": "ko", "rus": "ru", "ara": "ar", "hin": "hi", "pol": "pl",
    "tur": "tr", "swe": "sv", "dan": "da", "nor": "no", "fin": "fi",
    "ell": "el", "heb": "he", "tha": "th", "vie": "vi", "ukr": "uk",
    "ces": "cs", "ron": "ro", "hun": "hu", "ind": "id", "msa": "ms",
}


class ElevenLabsError(RuntimeError):
    """A Scribe request that cannot be retried into success."""


def model_label(model_id: str) -> str:
    return f"elevenlabs-{model_id}"


def _headers(api_key: str) -> dict[str, str]:
    return {"xi-api-key": api_key}


def _error_message(response: requests.Response) -> str:
    """Turn an API error body into one line worth showing in the jobs table."""
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, dict):
        text = detail.get("message") or detail.get("status") or str(detail)
    elif isinstance(detail, list) and detail:
        first = detail[0]
        text = first.get("msg", str(first)) if isinstance(first, dict) else str(first)
    elif detail:
        text = str(detail)
    else:
        text = (response.text or "").strip()[:200] or response.reason
    if response.status_code == 401:
        return f"ElevenLabs rejected the API key ({text}). Check it in Settings."
    if response.status_code == 429:
        return f"ElevenLabs rate limit / quota reached ({text})."
    return f"ElevenLabs error {response.status_code}: {text}"


def test_key(api_key: str) -> str:
    """Confirm a key works, without spending any transcription credit."""
    if not api_key.strip():
        raise ElevenLabsError("Enter an ElevenLabs API key first.")
    try:
        r = requests.get(ACCOUNT_URL, headers=_headers(api_key.strip()), timeout=30)
    except requests.RequestException as e:
        raise ElevenLabsError(f"Could not reach ElevenLabs: {e}") from e
    if r.status_code != 200:
        raise ElevenLabsError(_error_message(r))
    data = r.json()
    tier = data.get("tier", "unknown")
    return f"ElevenLabs key OK — {tier} plan."


# ---- request ----------------------------------------------------------
def _form_fields(opts) -> dict[str, str]:
    fields: dict[str, str] = {
        "model_id": opts.elevenlabs_model or DEFAULT_MODEL,
        "timestamps_granularity": "word",
        "diarize": "true" if opts.diarization_enabled else "false",
        "tag_audio_events": "true" if opts.tag_audio_events else "false",
    }
    if opts.language and opts.language != "auto":
        fields["language_code"] = opts.language
    # Scribe takes an upper bound on speakers; the local engine's "max" spin
    # box means the same thing, and 0 there means "let the model decide".
    if opts.diarization_enabled and opts.max_speakers:
        fields["num_speakers"] = str(max(1, min(MAX_SPEAKERS, int(opts.max_speakers))))
    return fields


def _post(audio_path: str, api_key: str, fields: dict[str, str], log, should_cancel) -> dict[str, Any]:
    path = Path(audio_path)
    size = path.stat().st_size
    if size > MAX_UPLOAD_BYTES:
        raise ElevenLabsError(
            f"{path.name} is {size / 1e9:.1f} GB — over the 5 GB ElevenLabs upload limit."
        )
    last_error = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        # The upload itself cannot be interrupted, so this is the last chance
        # to honour a cancel before committing to it.
        check_cancel(should_cancel, log, message="Cancelled — nothing sent to ElevenLabs.")
        try:
            with path.open("rb") as fh:
                response = requests.post(
                    API_URL,
                    headers=_headers(api_key),
                    files={"file": (path.name, fh, "application/octet-stream")},
                    data=fields,
                    timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                )
        except requests.RequestException as e:
            last_error = f"Could not reach ElevenLabs: {e}"
            if attempt == MAX_ATTEMPTS:
                raise ElevenLabsError(last_error) from e
        else:
            if response.status_code == 200:
                return response.json()
            if response.status_code not in RETRY_STATUSES or attempt == MAX_ATTEMPTS:
                raise ElevenLabsError(_error_message(response))
            last_error = _error_message(response)
        delay = min(30, 2 ** attempt)
        if log:
            log(f"ElevenLabs: {last_error} — retrying in {delay}s ({attempt}/{MAX_ATTEMPTS}).")
        time.sleep(delay)
    raise ElevenLabsError(last_error or "ElevenLabs request failed.")


# ---- response -> segments ---------------------------------------------
# Shared with the Gemini engine: both hand back words, and both need the same
# answer to "where does a turn end?".
from .word_segments import (  # noqa: E402,F401  (re-exported for callers and tests)
    GAP_SECONDS,
    HARD_CHARS,
    HARD_SECONDS,
    SENTENCE_END,
    SOFT_CHARS,
    SOFT_SECONDS,
    words_to_segments,
)


def _language(code: str) -> str:
    """Scribe reports ISO-639-3; the app shows the 639-1 codes Whisper uses."""
    return _LANG_3_TO_1.get((code or "").lower(), code or "")


def transcribe(
    recording: Recording,
    audio_path: str,
    opts,
    progress_cb=None,
    log_cb=None,
    should_cancel: ShouldCancel = None,
) -> TranscriptResult:
    """Transcribe one file with Scribe, returning the app's own result type."""

    def log(msg):
        if log_cb:
            log_cb(msg)

    api_key = (opts.elevenlabs_api_key or "").strip()
    if not api_key:
        raise ElevenLabsError(
            "No ElevenLabs API key. Add one in Settings, or switch the engine "
            "back to local Whisper."
        )
    model_id = opts.elevenlabs_model or DEFAULT_MODEL
    fields = _form_fields(opts)
    diarize = fields["diarize"] == "true"

    size_mb = Path(audio_path).stat().st_size / 1e6
    log(
        f"Uploading {size_mb:.1f} MB to ElevenLabs {model_id} "
        f"({'with' if diarize else 'without'} speaker detection)…"
    )
    if progress_cb:
        progress_cb(0.35)
    data = _post(audio_path, api_key, fields, log, should_cancel)
    if progress_cb:
        progress_cb(0.9)

    words = data.get("words") or []
    segments, speakers = words_to_segments(words, diarize)
    if not segments and (data.get("text") or "").strip():
        # No word timings came back (rare): keep the text rather than nothing.
        segments = [Segment(start=0.0, end=float(data.get("audio_duration_secs") or 0.0),
                            text=data["text"].strip())]
    spoken = f", {len(speakers)} speaker(s)" if speakers else ""
    log(f"ElevenLabs returned {len(segments)} segment(s){spoken}.")
    if diarize and not speakers:
        log("ElevenLabs found no speaker labels — the audio may be a single voice.")
    if progress_cb:
        progress_cb(1.0)
    return TranscriptResult(
        recording=recording,
        segments=segments,
        language=_language(data.get("language_code", "")),
        model=model_label(model_id),
        speakers=speakers,
    )
