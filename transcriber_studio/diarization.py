# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Optional speaker diarization via pyannote.audio.

Heavy deps (torch + pyannote.audio) are imported lazily so the app runs even
when they aren't installed yet. Requires a HuggingFace token with the
pyannote/speaker-diarization-community-1 model terms accepted.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import audio_utils

# pyannote 4.x recommended pipeline; pulls in segmentation + community assets.
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"

# Gated models — accept terms on each page (same HuggingFace account as your token).
HF_GATED_MODELS = (
    ("pyannote/speaker-diarization-community-1", "https://huggingface.co/pyannote/speaker-diarization-community-1"),
    ("pyannote/segmentation-3.0", "https://huggingface.co/pyannote/segmentation-3.0"),
    ("pyannote/speaker-diarization-3.1", "https://huggingface.co/pyannote/speaker-diarization-3.1"),
)


def format_hf_access_error(exc: BaseException) -> str:
    """Turn HuggingFace 403 / gated-repo errors into actionable steps."""
    msg = str(exc)
    lower = msg.lower()
    gated = (
        "403",
        "gated",
        "authorized list",
        "cannot access gated",
        "could not download",
        "private or gated",
    )
    if not any(k in lower for k in gated):
        return msg

    lines = [
        "HuggingFace has not granted access to all speaker diarization models yet.",
        "",
        "Do all of the following (free, same HuggingFace account as your token):",
        "",
        "1. Open each model page and click \"Agree and access\":",
    ]
    for _name, url in HF_GATED_MODELS:
        lines.append(f"   {url}")
    lines.extend([
        "",
        "2. Create a Read token: https://huggingface.co/settings/tokens",
        "3. Paste it in Settings → HuggingFace token, save, restart the app.",
        "",
        "If you already accepted, wait a minute and try again — or create a fresh token.",
    ])
    return "\n".join(lines)


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: str  # raw label like SPEAKER_00


def is_available() -> bool:
    try:
        import importlib.util
        return (
            importlib.util.find_spec("pyannote.audio") is not None
            and importlib.util.find_spec("torch") is not None
        )
    except Exception:
        return False


def check_stack() -> None:
    """Raise RuntimeError with actionable text if diarization deps are broken."""
    from .hardware import CPU_TORCH_INSTALL_CMD, CUDA_TORCH_INSTALL_CMD, cuda_available

    torch_fix = CUDA_TORCH_INSTALL_CMD if cuda_available() else CPU_TORCH_INSTALL_CMD
    torch_hint = (
        f"Reinstall matching PyTorch packages (PowerShell):\n  {torch_fix}\n\n"
        "Then restart the app."
    )

    try:
        import torchvision  # noqa: F401 — pyannote/lightning pull this in
    except Exception as e:
        msg = str(e).lower()
        if "torchvision" in msg or "extension" in msg or "circular import" in msg:
            raise RuntimeError(
                "PyTorch and torchvision versions are mismatched.\n\n" + torch_hint
            ) from e
        raise RuntimeError(f"Diarization dependencies failed to load: {e}") from e

    try:
        import torchaudio  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "torchaudio failed to load (often a torch version mismatch).\n\n" + torch_hint
        ) from e

    try:
        from pyannote.audio import Pipeline  # noqa: F401
    except Exception as e:
        raise RuntimeError(f"pyannote.audio failed to load: {e}") from e


class _UiProgressHook:
    """Maps pyannote pipeline hook callbacks to UI progress + log messages."""

    def __init__(
        self,
        progress_cb=None,
        log_cb=None,
        *,
        base: float = 0.12,
        span: float = 0.86,
    ):
        self.progress_cb = progress_cb
        self.log_cb = log_cb
        self.base = base
        self.span = span
        self._steps: list[str] = []
        self._step_name: str | None = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __call__(
        self,
        step_name: str,
        step_artifact,
        file=None,
        total: int | None = None,
        completed: int | None = None,
    ):
        if completed is None:
            completed = total = 1
        total = total or 1
        if step_name not in self._steps:
            self._steps.append(step_name)
        if step_name != self._step_name:
            self._step_name = step_name
            if self.log_cb:
                self.log_cb(f"Diarization: {step_name}…")
        step_idx = self._steps.index(step_name)
        n_steps = len(self._steps)
        step_frac = min(1.0, completed / total)
        overall = (step_idx + step_frac) / n_steps
        if self.progress_cb:
            self.progress_cb(self.base + self.span * overall)


def _turns_from_pipeline_output(output) -> list[SpeakerTurn]:
    """Support pyannote 3.x Annotation and 4.x DiarizeOutput."""
    if hasattr(output, "exclusive_speaker_diarization"):
        annotation = output.exclusive_speaker_diarization
    elif hasattr(output, "speaker_diarization"):
        annotation = output.speaker_diarization
    elif hasattr(output, "itertracks"):
        annotation = output
    else:
        raise RuntimeError(f"Unexpected diarization output type: {type(output).__name__}")

    turns: list[SpeakerTurn] = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        turns.append(SpeakerTurn(segment.start, segment.end, speaker))
    turns.sort(key=lambda t: t.start)
    return turns


class Diarizer:
    """Wraps a pyannote pipeline; caches the loaded pipeline per token."""

    _pipeline = None
    _token_used: str | None = None

    def __init__(self, hf_token: str, device: str = "auto"):
        self.hf_token = hf_token
        self.device = device

    def _resolve_device(self):
        import torch

        from .hardware import torch_cuda_available

        if self.device == "cuda" or (self.device == "auto" and torch_cuda_available()):
            return torch.device("cuda")
        return torch.device("cpu")

    def _load(self, log_cb=None):
        if Diarizer._pipeline is not None and Diarizer._token_used == self.hf_token:
            return Diarizer._pipeline
        if not self.hf_token:
            raise RuntimeError(
                "Speaker diarization needs a HuggingFace token. Add one in Settings, "
                f"and accept the terms for {DIARIZATION_MODEL} on huggingface.co."
            )
        if log_cb:
            log_cb("Loading speaker diarization model…")
        check_stack()
        from pyannote.audio import Pipeline
        try:
            pipeline = Pipeline.from_pretrained(
                DIARIZATION_MODEL,
                token=self.hf_token,
            )
        except Exception as e:
            raise RuntimeError(format_hf_access_error(e)) from e
        device = self._resolve_device()
        pipeline.to(device)
        if log_cb:
            from .hardware import torch_cuda_available
            where = "GPU" if torch_cuda_available() and str(device) != "cpu" else "CPU"
            log_cb(f"Speaker model ready ({where}).")
        Diarizer._pipeline = pipeline
        Diarizer._token_used = self.hf_token
        return pipeline

    def diarize(
        self,
        audio_path: str,
        min_speakers: int = 0,
        max_speakers: int = 0,
        progress_cb=None,
        log_cb=None,
    ) -> list[SpeakerTurn]:
        pipeline = self._load(log_cb=log_cb)
        kwargs = {}
        if min_speakers:
            kwargs["min_speakers"] = min_speakers
        if max_speakers:
            kwargs["max_speakers"] = max_speakers
        if progress_cb:
            progress_cb(0.02)
        if log_cb:
            log_cb("Preparing audio for speaker detection…")
        audio = audio_utils.load_waveform_for_diarization(audio_path)
        if log_cb:
            minutes = audio["waveform"].shape[-1] / audio["sample_rate"] / 60
            log_cb(f"Analyzing {minutes:.1f} min of audio for speakers…")
        if progress_cb:
            progress_cb(0.08)
        hook = _UiProgressHook(progress_cb, log_cb, base=0.10, span=0.88)
        with hook:
            output = pipeline(audio, hook=hook, **kwargs)
        turns = _turns_from_pipeline_output(output)
        if log_cb:
            log_cb(f"Found {len({t.speaker for t in turns})} speaker(s).")
        if progress_cb:
            progress_cb(1.0)
        return turns


def assign_speaker(start: float, end: float, turns: list[SpeakerTurn]) -> str | None:
    """Return the diarization speaker with the most temporal overlap for a span."""
    best, best_overlap = None, 0.0
    for t in turns:
        overlap = max(0.0, min(end, t.end) - max(start, t.start))
        if overlap > best_overlap:
            best_overlap, best = overlap, t.speaker
    return best
