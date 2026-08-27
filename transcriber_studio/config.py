# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Local, file-based settings persistence.

Every setting — including the API keys for the optional cloud services — lives
on this machine under the user's profile, in plain JSON. Nothing is sent
anywhere except by the services you explicitly turn on: the PLAUD CLI's own
authenticated calls, and whichever transcription or cleanup provider you pick.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

APP_NAME = "Transcriber Studio"


def _default_app_dir() -> Path:
    """Where this platform expects an application to keep its own data.

    Dropping a bare directory in someone's home folder is a Windows habit that
    reads as litter on the other two.
    """
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", Path.home())) / "TranscriberStudio"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "TranscriberStudio"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    return (Path(xdg) if xdg else Path.home() / ".config") / "transcriber-studio"


#: Everything the app stores about itself: settings, queue, glossaries, caches.
APP_DIR = _default_app_dir()
CONFIG_PATH = APP_DIR / "settings.json"

#: What the directory was called before the project was renamed. Kept so an
#: existing install does not silently lose its settings, glossaries and caches.
LEGACY_APP_DIRS = (
    Path(os.environ.get("APPDATA", Path.home())) / "PlaudWhisperStudio",
)


def migrate_legacy_dir() -> Path | None:
    """Adopt a previous install's data directory, once, if ours is not there yet.

    Renaming the app must not orphan the glossaries and settings someone spent
    time building. Returns the directory that was adopted, or None.
    """
    if APP_DIR.exists():
        return None
    for legacy in LEGACY_APP_DIRS:
        if not legacy.is_dir():
            continue
        try:
            APP_DIR.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(legacy, APP_DIR)
        except OSError:
            return None
        _repoint_settings(legacy)
        return legacy
    return None


def _repoint_settings(legacy: Path) -> None:
    """Move settings that name a file inside the old directory to the new one.

    The copy leaves the originals in place, so nothing breaks immediately — but
    a setting still pointing into a directory the user is about to delete is a
    failure waiting to happen.
    """
    if not CONFIG_PATH.exists():
        return
    try:
        raw = CONFIG_PATH.read_text(encoding="utf-8")
    except OSError:
        return
    updated = raw.replace(str(legacy), str(APP_DIR)).replace(
        str(legacy).replace("\\", "\\\\"), str(APP_DIR).replace("\\", "\\\\")
    )
    if updated != raw:
        try:
            CONFIG_PATH.write_text(updated, encoding="utf-8")
        except OSError:
            pass


def _default_output_dir() -> str:
    return str(Path.home() / "Documents" / "Transcripts")


@dataclass
class Settings:
    # --- engine ---
    # Which speech-to-text engine runs a job: the local Whisper install, or
    # ElevenLabs Scribe in the cloud (which also handles diarization).
    stt_engine: str = "local"       # local | elevenlabs
    model: str = "large-v3"
    device: str = "auto"            # auto | cuda | cpu
    compute_type: str = "auto"      # auto | float16 | int8_float16 | int8 | float32
    language: str = "auto"          # auto or ISO code e.g. "en"

    # --- diarization ---
    diarization_enabled: bool = True
    hf_token: str = ""              # HuggingFace access token (local only)
    min_speakers: int = 0           # 0 = unknown / let model decide
    max_speakers: int = 0

    # --- ElevenLabs Scribe (cloud STT; key stored locally only) ---
    elevenlabs_api_key: str = ""
    elevenlabs_model: str = "scribe_v1"
    elevenlabs_tag_audio_events: bool = False   # mark laughter, applause, etc.

    # --- Gemini (cloud STT; key stored locally only) ---
    # Shares the Google AI key with AI Cleanup: it is the same account.
    gemini_model: str = "gemini-3.5-transcribe"
    # verbatim or smart. Not a style preference: smart mode returns prose with
    # no speaker labels and no timestamps, because the API refuses both
    # parameters there. Verbatim is the only mode that fills in the app's own
    # data model, and pairs naturally with AI Cleanup for the tidying.
    gemini_mode: str = "verbatim"

    # --- channels ---
    channel_mode: str = "downmix"   # downmix | per_channel
    channel_names: str = ""         # comma list e.g. "Agent,Customer"

    # --- output ---
    output_dir: str = field(default_factory=_default_output_dir)
    formats: list[str] = field(default_factory=lambda: ["txt"])
    filename_template: str = "{date}_{name}"
    include_speakers: bool = True
    include_timestamps: bool = True
    line_mode: str = "segment"      # segment (one line per speaker turn) | sentence | wrap
    wrap_chars: int = 90
    newline: str = "crlf"           # crlf | lf
    sanitize_names: bool = True
    # Your own name(s) as they appear as speaker labels. Output files are named
    # after the OTHER person on the recording, so the app has to know which one
    # is you. Empty => no owner, and the first named speaker is used.
    owner_names: str = ""
    overwrite: bool = False         # False => add numeric suffix if exists

    # --- misc ---
    plaud_page_size: int = 50

    # --- UI state ---
    # False until the first-run wizard has been through once; the Setup button
    # re-runs it on demand.
    setup_complete: bool = False
    # Column widths of the Plaud recordings table, so a layout the user drags
    # into shape survives a restart. Empty => built-in defaults.
    recordings_col_widths: list[int] = field(default_factory=list)

    # --- AI provider keys (stored locally only) ---
    ai_key_openrouter: str = ""
    ai_key_openai: str = ""
    ai_key_anthropic: str = ""
    ai_key_google: str = ""
    ai_key_grok: str = ""
    ai_key_ollama_cloud: str = ""
    ollama_local_url: str = "http://localhost:11434"

    # --- AI cleanup ---
    ai_cleanup_enabled: bool = False
    ai_cleanup_provider: str = ""   # openrouter | openai | anthropic | google | grok | ollama_cloud | ollama_local
    ai_cleanup_model: str = ""
    # App-wide default, set in Settings. Seeds the Options panel and the AI
    # Cleanup dialog whenever nothing has been picked for the run at hand, so a
    # model chosen once does not have to be re-picked on every job.
    ai_default_provider: str = ""
    ai_default_model: str = ""

    # --- glossary (AI cleanup pre-stage) ---
    glossary_enabled: bool = True
    glossary_chunk_token_threshold: int = 60_000
    glossary_model: str = ""        # empty => use cleanup model
    glossary_temperature: float = 0.0
    force_reextract: bool = False
    # Id of a glossary in the shared library (transcriber_studio.glossary_store) that new jobs
    # read from and write back to. Empty => each recording keeps its own.
    glossary_shared_id: str = ""

    # --- audio front-end: denoise before anything else sees the audio ---
    # On hard audio this buys more than the model choice does; see transcriber_studio.denoise.
    denoise_enabled: bool = False
    denoise_backend: str = "auto"   # auto | deep_filter | deepfilternet | ffmpeg
    deep_filter_path: str = ""      # the deep-filter executable, if not on PATH
    denoise_model_path: str = ""    # optional DeepFilterNet model (.tar.gz / dir)
    denoise_postfilter: bool = False
    # How much noise DeepFilterNet is allowed to remove, in dB. 100 is the
    # tool's own default (full suppression); lowering it mixes some of the
    # original back in, which is the fix when heavy suppression starts eating
    # consonants — a smeared consonant costs more WER than the hiss it removed.
    denoise_atten_lim_db: int = 100

    # --- voice activity detection (local engine) ---
    # Silero, via faster-whisper, which also maps timestamps back for us.
    vad_enabled: bool = True
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 0
    vad_min_silence_ms: int = 2_000
    vad_speech_pad_ms: int = 400
    vad_max_speech_s: float = 0.0   # 0 => no cap

    # --- vocabulary biasing (local engine) ---
    # On mumbled audio the decoder leans on its language prior; these are the
    # priors worth handing it. Terms come from the shared glossary this job
    # uses, plus anything typed in below.
    bias_enabled: bool = True
    bias_extra_terms: str = ""
    bias_max_chars: int = 600
    # Stops a hallucinated passage seeding the next window, and drops segments
    # the decoder produced over long silences.
    hallucination_guard: bool = True

    # --- prompt caching (LLM providers that support it) ---
    prompt_cache_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load() -> Settings:
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            known = {f for f in Settings().to_dict()}
            return Settings(**{k: v for k, v in data.items() if k in known})
        except Exception:
            pass
    return Settings()


def save(settings: Settings) -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(settings.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def cleanup_provider(settings: Settings) -> str:
    """Provider for a cleanup run: what the run picked, else the app default."""
    return (settings.ai_cleanup_provider or settings.ai_default_provider or "").strip()


def cleanup_model(settings: Settings) -> str:
    """Model for a cleanup run: what the run picked, else the app default."""
    return (settings.ai_cleanup_model or settings.ai_default_model or "").strip()
