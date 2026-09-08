# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Microsoft MAI-Transcribe: cloud speech-to-text with diarization built in.

Reached through Azure Speech's fast transcription endpoint in "enhanced mode",
which is what routes a request to the MAI models rather than to Azure's older
recognisers. One multipart POST carries the audio and a JSON ``definition``
describing what to do with it; the answer comes back whole, with speakers
already separated and a confidence on every phrase.

Three of its options line up with things this app already has, and are wired
straight through:

* ``phraseList.phrases`` takes the same vocabulary the glossary and vocab-bias
  layers assemble for Whisper's hotwords, so a name learned on one recording
  biases the next one here too.
* ``modelOptions.timestamps: "word"`` returns per-word timings, which is what
  lets a turn be split where the floor changes hands rather than filed whole.
* ``transcribeStyle`` chooses between the words as spoken and a tidied version.
  Verbatim is the default here for the same reason CrisperWhisper is offered
  locally: tidying belongs in AI Cleanup, after the words are on the page and
  where the decision can be reviewed.

What it cannot do is speaker *recognition*. Diarization happens inside the
model and no voice embedding comes back, so an enrolled voiceprint has nothing
to match against and speakers arrive as Speaker 1, 2, 3 — the same as the
Gemini and ElevenLabs engines. See transcriber_studio.voiceprints.

Public preview at the time of writing: no SLA, and the contract can move.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import requests
from requests_toolbelt.multipart.encoder import (
    MultipartEncoder,
    MultipartEncoderMonitor,
)

from . import uploads
from .job_cancel import ShouldCancel, check_cancel
from .models import Recording, Segment, TranscriptResult
from .word_segments import words_to_segments

#: Only these Azure regions host the MAI models. A Speech resource in any other
#: region authenticates fine and then reports the model as unavailable, which
#: is a confusing way to find out, so the choice is offered as a list.
REGIONS = ("eastus", "westus", "westus2", "northeurope", "southeastasia", "centralindia")
DEFAULT_REGION = "eastus"

#: Pinned rather than "latest": the response shape is parsed below, and a
#: version bump that changes it should be a deliberate edit here.
API_VERSION = "2025-10-15"

MODELS = ("MAI-Transcribe-2", "MAI-Transcribe-1.5")
DEFAULT_MODEL = "MAI-Transcribe-2"

STYLE_VERBATIM = "verbatim"
STYLE_CLEAN = "clean"
STYLES = (STYLE_VERBATIM, STYLE_CLEAN)

#: The MAI documentation states 300 MB; the wider fast-transcription endpoint
#: allows 500 MB. The smaller number is the one that applies to these models.
MAX_UPLOAD_BYTES = 300 * 1024 * 1024
MAX_DURATION_SECONDS = 5 * 3600

#: Only the connection itself; the upload gets its own budget from the file
#: size, because requests applies this value to the whole request body. See
#: transcriber_studio.uploads.
CONNECT_TIMEOUT = 30
READ_TIMEOUT = 1800         # an hour of audio is still minutes of waiting
RETRY_STATUSES = {408, 429, 500, 502, 503, 504}

#: Errors the Speech front end relays from the MAI backend that do not change
#: on a retry. Measured: a 73-minute file with diarization on failed three
#: times in a row — 408, then 500, then a 503 wrapping a diarization 400 — and
#: each retry re-uploaded the full file first. Five minutes of upload for an
#: answer that was never going to differ. Microsoft's own guidance is not to
#: retry 4xx, and the inner code here is a 4xx however it is wrapped.
DETERMINISTIC_MARKERS = (
    "returned error code 400",          # a sub-service rejected the request
    "diarization_unavailable",          # the diarization backend gave up
    "audiolengthlimitexceeded",         # documented: the file is too long
    '"code": "timeout"',                # the MAI backend's own deadline
    '"code":"timeout"',
)


def _is_deterministic(message: str) -> bool:
    """True when sending the identical request again cannot help."""
    lower = (message or "").lower()
    return any(marker in lower for marker in DETERMINISTIC_MARKERS)
MAX_ATTEMPTS = 3

#: How many phrases to send as recognition hints. Fifty is the service's own
#: limit, discovered by exceeding it: a longer list is refused outright with
#: "Context list cannot have more than 50 items", which failed every run whose
#: glossary had grown past it. The documentation does not mention a limit.
MAX_PHRASES = 50

#: The slice of the job's progress bar that sending the audio occupies. The
#: wait for the model that follows has no progress of its own — the response
#: arrives whole — so the bar stops here until it does.
UPLOAD_PROGRESS_BASE = 0.35
UPLOAD_PROGRESS_SPAN = 0.40


class MaiError(RuntimeError):
    """A MAI request that cannot be retried into success."""


def model_label(model_id: str) -> str:
    return f"azure-{(model_id or DEFAULT_MODEL).lower()}"


def endpoint(region: str) -> str:
    region = (region or DEFAULT_REGION).strip().lower()
    return (
        f"https://{region}.api.cognitive.microsoft.com"
        f"/speechtotext/transcriptions:transcribe?api-version={API_VERSION}"
    )


def _headers(api_key: str) -> dict[str, str]:
    return {"Ocp-Apim-Subscription-Key": api_key, "Accept": "application/json"}


def build_definition(opts) -> dict[str, Any]:
    """The JSON that tells the endpoint to use MAI, and how.

    ``enhancedMode`` is what selects a MAI model at all; without it the request
    is served by Azure's older recogniser and quietly returns something else.
    """
    model_options: dict[str, Any] = {
        # Always word-level. The extra data costs nothing and it is what the
        # per-word speaker assignment downstream is built on.
        "timestamps": "word",
        "transcribeStyle": _style(opts),
    }
    definition: dict[str, Any] = {
        "enhancedMode": {
            "enabled": True,
            "model": (getattr(opts, "mai_model", "") or DEFAULT_MODEL),
            "modelOptions": model_options,
        }
    }
    if getattr(opts, "diarization_enabled", True):
        definition["diarization"] = {"enabled": True}
    phrases = _phrases(opts)
    if phrases:
        definition["phraseList"] = {"phrases": phrases}
    language = (getattr(opts, "language", "") or "auto").strip()
    if language and language != "auto":
        # Documented as a very strong hint rather than a filter, so it is only
        # sent when the user has actually chosen a language.
        definition["locales"] = [language]
    return definition


def _style(opts) -> str:
    style = (getattr(opts, "mai_style", "") or STYLE_VERBATIM).strip().lower()
    return style if style in STYLES else STYLE_VERBATIM


def _phrases(opts) -> list[str]:
    """The vocabulary hints, taken from whatever the app already assembled.

    ``opts.hotwords`` is the comma-joined list the glossary and vocab-bias
    layers build for Whisper. The same terms are the right hints here, so the
    work of collecting them is not done twice.
    """
    if not getattr(opts, "mai_send_phrases", True):
        return []
    raw = (getattr(opts, "hotwords", "") or "").strip()
    if not raw:
        return []
    seen: list[str] = []
    for term in raw.split(", "):
        term = term.strip()
        if term and term not in seen:
            seen.append(term)
        if len(seen) >= MAX_PHRASES:
            break
    return seen


def phrase_overflow(opts) -> int:
    """How many glossary terms did not fit in the hint list.

    Worth saying out loud: the glossary grows on its own as recordings are
    processed, so a setup that worked last month can quietly start sending only
    part of its vocabulary.
    """
    if not getattr(opts, "mai_send_phrases", True):
        return 0
    raw = (getattr(opts, "hotwords", "") or "").strip()
    if not raw:
        return 0
    unique = {t.strip() for t in raw.split(", ") if t.strip()}
    return max(0, len(unique) - MAX_PHRASES)


def _error_message(response: requests.Response) -> str:
    """What Azure said, in as much detail as it gave.

    The body is not reliably JSON: a rejection from the MAI service arrives as
    prose with a JSON object embedded in it. Falling back to "HTTP 400" threw
    away the one sentence that explained the failure, so the raw text is kept
    when it cannot be parsed.
    """
    status = response.status_code
    try:
        payload = response.json()
    except ValueError:
        body = (response.text or "").strip()
        return f"{_shorten(body)} (HTTP {status})" if body else f"Azure returned HTTP {status}."
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        message = error.get("message") or error.get("code") or ""
        if message:
            return f"{message} (HTTP {status})"
    return f"Azure returned HTTP {status}."


def _shorten(text: str, limit: int = 300) -> str:
    """Enough of an error to act on, without pasting a wall into the log."""
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[:limit] + "…"


def test_key(api_key: str, region: str) -> str:
    """Prove a key and region work, without transcribing anything.

    Sends a request with no audio: a working key answers "bad request" because
    the audio is missing, and a bad key or wrong region answers 401 or 404
    first. Distinguishing those is the whole point — a key pasted against the
    wrong region otherwise fails much later, on a real recording.
    """
    key = (api_key or "").strip()
    if not key:
        raise MaiError("Paste the Speech resource key first.")
    if region not in REGIONS:
        raise MaiError(
            f"MAI-Transcribe is not available in {region!r}. Supported regions: "
            + ", ".join(REGIONS)
        )
    try:
        response = requests.post(
            endpoint(region), headers=_headers(key), timeout=(CONNECT_TIMEOUT, 60)
        )
    except requests.RequestException as e:
        raise MaiError(f"Could not reach Azure: {e}") from e
    if response.status_code in (401, 403):
        raise MaiError(
            "Azure rejected that key. Check it against Keys and Endpoint on the "
            "Speech resource, and that the region matches the resource."
        )
    if response.status_code == 404:
        raise MaiError(
            f"No Speech endpoint in {region}. Either the resource lives in a "
            "different region, or it is not a Foundry Speech resource."
        )
    return f"Key accepted for {region}."


def _post(
    audio_path: str, api_key: str, region: str, definition: dict[str, Any],
    log, should_cancel, progress_cb=None,
) -> dict[str, Any]:
    path = Path(audio_path)
    size = path.stat().st_size
    if size > MAX_UPLOAD_BYTES:
        raise MaiError(
            f"{path.name} is {size / 1e6:.0f} MB — over the "
            f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB limit for MAI-Transcribe."
        )
    url = endpoint(region)
    payload = json.dumps(definition)
    last_error = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        # The upload cannot be interrupted once it starts, so this is the last
        # honest place to notice a cancel.
        check_cancel(should_cancel, log, message="Cancelled — nothing sent to Azure.")
        try:
            with path.open("rb") as fh:
                encoder = MultipartEncoder(fields={
                    "audio": (path.name, fh, "application/octet-stream"),
                    "definition": (None, payload, "application/json"),
                })
                watcher = uploads.UploadProgress(
                    encoder.len, log=log, progress_cb=progress_cb,
                    base=UPLOAD_PROGRESS_BASE, span=UPLOAD_PROGRESS_SPAN,
                )
                body = MultipartEncoderMonitor(encoder, watcher)
                response = requests.post(
                    url,
                    headers={**_headers(api_key), "Content-Type": body.content_type},
                    data=body,
                    timeout=uploads.timeout_for(path, READ_TIMEOUT),
                )
            if log:
                # The two waits look identical from outside, so name the second
                # one as it begins rather than leaving the bar apparently stuck.
                log(
                    f"Upload finished in {watcher.elapsed():.0f}s. Azure is "
                    "transcribing now — no further progress until it answers."
                )
            if progress_cb:
                progress_cb(UPLOAD_PROGRESS_BASE + UPLOAD_PROGRESS_SPAN)
        except requests.RequestException as e:
            last_error = f"Could not reach Azure: {e}"
            if attempt == MAX_ATTEMPTS:
                raise MaiError(last_error) from e
        else:
            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError as e:
                    raise MaiError("Azure returned something that was not JSON.") from e
            message = _error_message(response)
            if (
                response.status_code not in RETRY_STATUSES
                or attempt == MAX_ATTEMPTS
                or _is_deterministic(message)
            ):
                raise MaiError(message)
            last_error = message
        delay = min(30, 2 ** attempt)
        if log:
            log(f"MAI-Transcribe: {last_error} — retrying in {delay}s ({attempt}/{MAX_ATTEMPTS}).")
        time.sleep(delay)
    raise MaiError(last_error or "The MAI-Transcribe request failed.")


# ---- response -> the app's own shape ----------------------------------
def _confidence_to_logprob(confidence) -> float | None:
    """Azure reports a 0..1 confidence; a Segment stores the log of it.

    One representation for the whole app, whichever engine produced the
    segment — see Segment.confidence.

    Measured against the live service, MAI-Transcribe-2 sends ``"confidence": 0``
    on every phrase: the field is in the schema but the model does not fill it
    in. Zero is returned as None rather than passed through, because a stored
    zero would read as "certainly wrong" everywhere downstream, when what is
    true is "this engine does not say". Kept rather than deleted because the
    field is documented and may start carrying a real number.
    """
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value <= 0.0:
        return None
    return math.log(min(1.0, value))


def _words_from(phrases: list[dict]) -> list[dict]:
    """Every word in the response, in the plain shape word_segments expects."""
    words: list[dict] = []
    for phrase in phrases:
        speaker = phrase.get("speaker")
        speaker_id = "" if speaker is None else f"spk:{speaker}"
        for word in phrase.get("words") or []:
            start = float(word.get("offsetMilliseconds") or 0) / 1000.0
            duration = float(word.get("durationMilliseconds") or 0) / 1000.0
            text = str(word.get("text") or "")
            if not text:
                continue
            if words:
                # Azure gives no spacing entries of its own; without this the
                # words would run together into one unreadable string.
                words.append({"type": "spacing", "text": " "})
            words.append({
                "type": "word",
                "text": text,
                "start": start,
                "end": start + duration,
                "speaker_id": speaker_id,
            })
    return words


#: Share of the phrase text that the word list has to account for before a
#: transcript may be rebuilt from the words. Measured: in the clean style the
#: words covered well under half of what the phrases said.
WORD_COVERAGE_MINIMUM = 0.9


def words_cover_text(words: list[dict], phrases: list[dict]) -> bool:
    """True when rebuilding the transcript from these words would lose nothing.

    Compared by length rather than by matching tokens, because the clean style
    changes spelling and punctuation between the two representations; what it
    must not do is drop clauses, and a length ratio catches that.
    """
    spoken = sum(len(str(p.get("text") or "").split()) for p in phrases)
    timed = sum(1 for w in words if w.get("type", "word") == "word")
    if spoken == 0:
        return bool(timed)
    return timed / spoken >= WORD_COVERAGE_MINIMUM


def _segments_from_phrases(phrases: list[dict]) -> tuple[list[Segment], list[str]]:
    """One segment per returned phrase, used when no word timings came back."""
    segments: list[Segment] = []
    order: list[str] = []
    for phrase in phrases:
        text = str(phrase.get("text") or "").strip()
        if not text:
            continue
        start = float(phrase.get("offsetMilliseconds") or 0) / 1000.0
        duration = float(phrase.get("durationMilliseconds") or 0) / 1000.0
        raw_speaker = phrase.get("speaker")
        speaker = None
        if raw_speaker is not None:
            key = f"spk:{raw_speaker}"
            if key not in order:
                order.append(key)
            speaker = f"Speaker {order.index(key) + 1}"
        segments.append(Segment(
            start=start,
            end=start + duration,
            text=text,
            speaker=speaker,
            avg_logprob=_confidence_to_logprob(phrase.get("confidence")),
        ))
    return segments, [f"Speaker {i + 1}" for i in range(len(order))]


def _apply_phrase_confidence(segments: list[Segment], phrases: list[dict]) -> None:
    """Carry each phrase's confidence onto the segments rebuilt from its words.

    The regrouping downstream cuts turns where the speaker changes, so a
    rebuilt segment is not the phrase Azure scored. Overlap is the honest way
    to decide which score describes it.
    """
    scored = [
        (
            float(p.get("offsetMilliseconds") or 0) / 1000.0,
            (float(p.get("offsetMilliseconds") or 0) + float(p.get("durationMilliseconds") or 0)) / 1000.0,
            _confidence_to_logprob(p.get("confidence")),
        )
        for p in phrases
    ]
    for segment in segments:
        if segment.avg_logprob is not None:
            continue
        best, best_overlap = None, 0.0
        for start, end, logprob in scored:
            if logprob is None:
                continue
            overlap = min(segment.end, end) - max(segment.start, start)
            if overlap > best_overlap:
                best, best_overlap = logprob, overlap
        segment.avg_logprob = best


def _locale_to_language(phrases: list[dict]) -> str:
    """The language, as the two-letter code the rest of the app uses."""
    for phrase in phrases:
        locale = str(phrase.get("locale") or "").strip()
        if locale:
            return locale.split("-")[0].lower()
    return ""


def decode(
    recording: Recording,
    audio_path: str,
    opts,
    progress_cb=None,
    log_cb=None,
    should_cancel: ShouldCancel = None,
) -> tuple[list[Segment], str, list[dict]]:
    """The words, without speakers: MAI standing in for the Whisper decoder.

    Returns the same ``(segments, language, words)`` triple the local decoder
    does, so the rest of the local pipeline — pyannote, per-word speaker
    assignment, voiceprints — runs on top of it unchanged.

    This exists because of what the probes showed: MAI transcribes a 73-minute
    recording in one request without complaint, and its *diarization* fails on
    anything past roughly half an hour, with a relayed 408 that reads like a
    network fault. Splitting the pipeline at exactly that seam keeps the part
    that works and replaces the part that does not with the one this app
    already runs locally.
    """

    def log(msg):
        if log_cb:
            log_cb(msg)

    # No diarization from MAI on this path: that is the point of it.
    plain = type("Opts", (), {})()
    for name in dir(opts):
        if not name.startswith("_"):
            try:
                setattr(plain, name, getattr(opts, name))
            except AttributeError:
                pass
    plain.diarization_enabled = False
    result, words = _transcribe_with_words(
        recording, audio_path, plain, progress_cb, log, should_cancel
    )
    return result.segments, result.language, words


def transcribe(
    recording: Recording,
    audio_path: str,
    opts,
    progress_cb=None,
    log_cb=None,
    should_cancel: ShouldCancel = None,
) -> TranscriptResult:
    """Transcribe one file with MAI-Transcribe, in the app's own result type."""

    def log(msg):
        if log_cb:
            log_cb(msg)

    result, _words = _transcribe_with_words(
        recording, audio_path, opts, progress_cb, log, should_cancel
    )
    return result


def _transcribe_with_words(
    recording: Recording, audio_path: str, opts, progress_cb, log, should_cancel,
) -> tuple[TranscriptResult, list[dict]]:
    """The shared body of transcribe() and decode(): one request, parsed."""

    api_key = (getattr(opts, "mai_api_key", "") or "").strip()
    if not api_key:
        raise MaiError(
            "No Azure Speech key. Add one in Settings → Engines, or switch the "
            "engine back to local Whisper."
        )
    region = (getattr(opts, "mai_region", "") or DEFAULT_REGION).strip().lower()
    if region not in REGIONS:
        raise MaiError(
            f"MAI-Transcribe is not available in {region!r}. Choose one of: "
            + ", ".join(REGIONS)
        )

    definition = build_definition(opts)
    diarize = "diarization" in definition
    style = definition["enhancedMode"]["modelOptions"]["transcribeStyle"]
    model_id = definition["enhancedMode"]["model"]
    phrases_sent = len(definition.get("phraseList", {}).get("phrases", []))

    log(
        f"Uploading {uploads.describe(Path(audio_path).stat().st_size)} to "
        f"{model_id} in {region} — {style} style, "
        f"{'with' if diarize else 'without'} speaker detection"
        + (f", {phrases_sent} vocabulary hint(s)" if phrases_sent else "")
        + "…"
    )
    dropped = phrase_overflow(opts)
    if dropped:
        log(
            f"Vocabulary: {dropped} term(s) beyond the {MAX_PHRASES}-hint limit were "
            "left out — Azure refuses a longer list."
        )
    if progress_cb:
        progress_cb(UPLOAD_PROGRESS_BASE)

    data = _post(
        audio_path, api_key, region, definition, log, should_cancel, progress_cb
    )
    if progress_cb:
        progress_cb(0.9)

    phrases = data.get("phrases") or []
    words = _words_from(phrases)
    if words and words_cover_text(words, phrases):
        segments, speakers = words_to_segments(words, diarize)
        _apply_phrase_confidence(segments, phrases)
    else:
        # The phrases are the text of record. In the clean style MAI rewrites
        # the transcript and the word timings only cover part of it, so a
        # transcript rebuilt from those words loses whole clauses. The words
        # are then kept for alignment only, never as the source of the text.
        if words:
            log(
                "MAI's word timings cover only part of the text (clean style) — "
                "keeping its phrases as the transcript."
            )
        words = []
        segments, speakers = _segments_from_phrases(phrases)

    if not segments:
        combined = " ".join(
            str(c.get("text") or "") for c in (data.get("combinedPhrases") or [])
        ).strip()
        if combined:
            # Nothing structured came back, but there is a transcript. Keeping
            # it beats reporting an empty result.
            duration = float(data.get("durationMilliseconds") or 0) / 1000.0
            segments = [Segment(start=0.0, end=duration, text=combined)]

    spoken = f", {len(speakers)} speaker(s)" if speakers else ""
    log(f"MAI-Transcribe returned {len(segments)} segment(s){spoken}.")
    if diarize and not speakers:
        log("No speaker labels came back — the audio may be a single voice.")

    if progress_cb:
        progress_cb(1.0)
    result = TranscriptResult(
        recording=recording,
        segments=segments,
        language=_locale_to_language(phrases),
        model=model_label(model_id),
        speakers=speakers,
    )
    return result, words
