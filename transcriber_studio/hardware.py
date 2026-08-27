# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""GPU / device detection shared across Whisper and settings UI."""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path

_cuda_dll_paths_configured = False



def configure_cuda_dll_paths() -> None:
    """On Windows, add pip-installed NVIDIA CUDA DLL folders to the loader search path.

    Registering the directories is enough: CTranslate2 resolves cuBLAS through
    them, and since 4.7 it does not need cuDNN at all. An earlier version of
    this also force-loaded every DLL it found, which Windows' loader answers
    with a fatal 0xc0000139 on libraries that are not meant to be loaded
    standalone — survivable, but it printed a fault dump over the app's output
    on every start.
    """
    global _cuda_dll_paths_configured
    if _cuda_dll_paths_configured or sys.platform != "win32":
        return
    search_roots: list[Path] = []
    for path in site.getsitepackages():
        search_roots.append(Path(path))
    user_site = site.getusersitepackages()
    if user_site:
        search_roots.append(Path(user_site))

    for root in search_roots:
        nvidia_root = root / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for bin_dir in nvidia_root.glob("*/bin"):
            if not bin_dir.is_dir():
                continue
            try:
                os.add_dll_directory(str(bin_dir))
            except OSError:
                pass
    _cuda_dll_paths_configured = True


def cuda_available() -> bool:
    """True when CUDA can be used for Whisper (faster-whisper / CTranslate2)."""
    configure_cuda_dll_paths()
    try:
        import torch

        if torch.cuda.is_available():
            return True
    except Exception:
        pass
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        pass
    return False


def torch_cuda_available() -> bool:
    """True when PyTorch itself can run on CUDA (needed for pyannote diarization)."""
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def cuda_device_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    if cuda_available():
        return "NVIDIA GPU (via CTranslate2)"
    return None


# Install all three together — mixing CPU torch with CUDA torchvision breaks pyannote.
#: PyTorch retires CUDA channels as it adds new ones, so this is a starting
#: point rather than a fact: the Components window asks the index which channel
#: actually publishes the release you are moving to. CUDA 12 is chosen over 13
#: because CTranslate2 — which runs Whisper — is built against CUDA 12.
CUDA_TORCH_INSTALL_CMD = (
    "pip install torch torchvision torchaudio torchcodec "
    "--index-url https://download.pytorch.org/whl/cu126"
)
CPU_TORCH_INSTALL_CMD = (
    "pip install torch torchvision torchaudio torchcodec "
    "--index-url https://download.pytorch.org/whl/cpu"
)


def diarization_device_label() -> str:
    """Short label for settings / tooltips."""
    if torch_cuda_available():
        name = cuda_device_name()
        return f"GPU ({name})" if name else "GPU"
    if cuda_available():
        return "CPU (CUDA PyTorch not installed — see Settings)"
    return "CPU"
