# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Filter harmless third-party warnings before heavy imports."""

from __future__ import annotations

import os
import warnings


def configure() -> None:
    # Windows often lacks symlink support for the HF cache; caching still works.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    warnings.filterwarnings("ignore", message="Failed to find CUDA.")
    # pyannote emits a long multiline UserWarning when torchcodec DLLs are missing;
    # we preload waveforms via ffmpeg + soundfile instead.
    warnings.filterwarnings("ignore", module=r"pyannote\.audio\.core\.io")
    warnings.filterwarnings("ignore", module=r"pyannote\.audio\.utils\.reproducibility")
    warnings.filterwarnings(
        "ignore",
        message="std\\(\\): degrees of freedom is <= 0.*",
        category=UserWarning,
    )
