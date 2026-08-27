# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""What the local Whisper sizes actually cost, so the choice can be informed.

The model list is otherwise six opaque names: nothing in "medium" says whether
it fits on this GPU or how much longer a two-hour recording will take. Figures
are for faster-whisper — float16 on a GPU, int8 on CPU — and speed is relative
to large-v3 on the same machine.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    id: str             # what faster-whisper loads: a size name or a HF repo id
    parameters: str
    vram: str           # float16 on GPU
    speed: str          # relative to large-v3
    accuracy: str
    use_when: str
    label: str = ""     # shown instead of the id when the id is a repo path
    caveats: tuple[str, ...] = ()

    @property
    def display(self) -> str:
        return self.label or self.id


#: The CTranslate2 conversion of nyralabs/CrisperWhisper, which is what
#: faster-whisper can load directly. The nyrahealth/* ids redirect here.
CRISPER = "nyralabs/faster_CrisperWhisper"

MODELS: dict[str, ModelInfo] = {
    "tiny": ModelInfo(
        "tiny", "39M", "~1 GB", "~10× faster", "Roughest",
        "Quick drafts, clean single-speaker audio, or a machine with no GPU.",
    ),
    "base": ModelInfo(
        "base", "74M", "~1 GB", "~7× faster", "Rough",
        "Skimming long recordings where the gist is enough.",
    ),
    "small": ModelInfo(
        "small", "244M", "~2 GB", "~4× faster", "Decent",
        "The usual CPU-only choice: readable transcripts without an overnight wait.",
    ),
    "medium": ModelInfo(
        "medium", "769M", "~5 GB", "~2× faster", "Good",
        "A middle ground when large-v3 will not fit or is too slow.",
    ),
    "large-v2": ModelInfo(
        "large-v2", "1550M", "~10 GB", "baseline", "Excellent",
        "Only if large-v3 mis-handles a particular accent or language for you.",
    ),
    "large-v3": ModelInfo(
        "large-v3", "1550M", "~10 GB", "baseline", "Best",
        "The default with a GPU: best accuracy on names, jargon and crosstalk.",
    ),
    CRISPER: ModelInfo(
        CRISPER, "1550M", "~10 GB", "about the same", "Best (verbatim)",
        "Hard audio where you want every word as spoken: it keeps the fillers, "
        "stutters and false starts that stock Whisper quietly tidies away, and "
        "it hallucinates less over noise. Pair it with AI Cleanup, which is "
        "where the tidying should happen — after the words are on the page.",
        label="CrisperWhisper (large-v3 verbatim fine-tune)",
        caveats=(
            "English and German only — the fine-tune was trained on those two, "
            "and other languages are not covered by it.",
            "Word timestamps are less precise here than in the original "
            "CrisperWhisper: this is the CTranslate2 conversion, which computes "
            "them differently. Segment times are unaffected.",
            "Downloaded from HuggingFace on first use (~1.5 GB).",
        ),
    ),
}

ORDER = ["tiny", "base", "small", "medium", "large-v2", "large-v3", CRISPER]

GPU_RECOMMENDED = "large-v3"
CPU_RECOMMENDED = "small"


def recommended(has_gpu: bool) -> str:
    return GPU_RECOMMENDED if has_gpu else CPU_RECOMMENDED


def label(model_id: str) -> str:
    """What to show in a picker for a model id."""
    info = MODELS.get(model_id)
    return info.display if info else model_id


def describe(model_id: str, has_gpu: bool) -> str:
    """A few lines about one model, in the context of this machine."""
    info = MODELS.get(model_id)
    if info is None:
        return ""
    where = "GPU" if has_gpu else "CPU"
    lines = [
        f"{info.parameters} parameters · {info.vram} VRAM (float16) · {info.speed} · {info.accuracy} accuracy",
        info.use_when,
    ]
    lines.extend(f"Note: {caveat}" for caveat in info.caveats)
    pick = recommended(has_gpu)
    if model_id == pick:
        lines.append(f"Recommended for this machine ({where}).")
    else:
        lines.append(f"On this machine ({where}) the usual pick is {pick}.")
    if not has_gpu and model_id in ("large-v2", "large-v3", "medium", CRISPER):
        lines.append(
            "Without a GPU this runs several times slower than the recording itself — "
            "an hour of audio can take hours."
        )
    return "\n".join(lines)
