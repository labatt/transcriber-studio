# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Noise suppression in front of transcription.

On hard audio the front-end matters more than the model. Whisper decodes noisy
speech by leaning harder on its language prior, which is exactly when it starts
inventing fluent text, so cleaning the signal first is the cheapest accuracy the
pipeline has.

Three backends, in the order they are picked:

* ``deep_filter`` — the standalone DeepFilterNet binary. The best of the three
  and the one to install. It is a single executable, so it sidesteps the pip
  package's dependency pins entirely.
* ``deepfilternet`` — the pip package, same model family, for Pythons it has
  wheels for (3.11 and older at the time of writing).
* ``ffmpeg`` — an FFT denoiser and a high-pass. Always available because ffmpeg
  is already required. Clearly weaker than DeepFilterNet on babble and
  reverberation, but it costs nothing and it is better than nothing.

Everything degrades: a missing backend, a crash, or a garbled output leaves the
original audio in place and says so in the log. Nothing here is allowed to fail
a job.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .audio_utils import FFMPEG, have_ffmpeg, probe
from .config import APP_DIR, Settings
from .job_cancel import JobCancelled, ShouldCancel, check_cancel

CACHE_DIR = APP_DIR / "denoise_cache"
#: Enhanced audio is big; keep only the most recent few files around.
CACHE_KEEP = 12

#: What the rest of the pipeline wants: Whisper and pyannote both work at 16 kHz.
PIPELINE_SAMPLE_RATE = 16_000
#: What DeepFilterNet operates at.
DF_SAMPLE_RATE = 48_000

#: Denoise this many seconds at a time rather than handing over a whole
#: recording. Measured at roughly 5–8x realtime, so a chunk this size takes
#: about 25 seconds: long enough that the model's start-up cost is noise,
#: short enough to report progress, stay cancellable, and keep memory flat.
#: A whole 80-minute file was 1.8 GB resident and produced nothing at all.
CHUNK_SECONDS = 120
#: A chunk that has not finished in this multiple of its own duration is not
#: slow, it is stuck. Roughly forty times the measured rate, so a genuinely
#: slow machine is never cut off, but a wedged process cannot hang the job.
CHUNK_TIMEOUT_FACTOR = 8
MIN_CHUNK_TIMEOUT = 90.0

AUTO = "auto"
DEEP_FILTER = "deep_filter"
PYTHON_DF = "deepfilternet"
FFMPEG_DN = "ffmpeg"
NONE = "none"

#: Tried in this order when the backend is "auto".
PREFERENCE = (DEEP_FILTER, PYTHON_DF, FFMPEG_DN)

BINARY_NAMES = ("deep-filter", "deep_filter")


@dataclass(frozen=True)
class BackendInfo:
    id: str
    label: str
    detail: str


BACKENDS: dict[str, BackendInfo] = {
    AUTO: BackendInfo(
        AUTO, "Best available",
        "Use the deep-filter binary if it is installed, then the Python package, "
        "then ffmpeg's own denoiser.",
    ),
    DEEP_FILTER: BackendInfo(
        DEEP_FILTER, "DeepFilterNet (deep-filter binary)",
        "A single downloaded executable. The strongest option, and the one that "
        "does not care which Python this app runs on.",
    ),
    PYTHON_DF: BackendInfo(
        PYTHON_DF, "DeepFilterNet (Python package)",
        "pip install deepfilternet. Same model family as the binary; its wheels "
        "lag new Python versions.",
    ),
    FFMPEG_DN: BackendInfo(
        FFMPEG_DN, "ffmpeg (afftdn + high-pass)",
        "Always available. Handles steady hiss and rumble; weaker than "
        "DeepFilterNet on background chatter.",
    ),
}

ORDER = [AUTO, DEEP_FILTER, PYTHON_DF, FFMPEG_DN]


def binary_path(configured: str = "") -> str | None:
    """The deep-filter executable: the configured one, else one on PATH."""
    configured = (configured or "").strip().strip('"')
    if configured:
        path = Path(configured)
        return str(path) if path.is_file() else None
    for name in BINARY_NAMES:
        found = shutil.which(name)
        if found:
            return found
    return None


def python_package_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("df") is not None


def is_available(backend: str, settings: Settings) -> bool:
    if backend == DEEP_FILTER:
        return binary_path(settings.deep_filter_path) is not None
    if backend == PYTHON_DF:
        return python_package_available()
    if backend == FFMPEG_DN:
        return have_ffmpeg()
    return False


def resolve(settings: Settings) -> str:
    """Which backend a run would actually use — NONE when none can run."""
    if not settings.denoise_enabled:
        return NONE
    wanted = settings.denoise_backend or AUTO
    if wanted != AUTO:
        return wanted if is_available(wanted, settings) else NONE
    for backend in PREFERENCE:
        if is_available(backend, settings):
            return backend
    return NONE


def describe(settings: Settings) -> str:
    """One line for the UI: what will run, or what is missing and why."""
    if not settings.denoise_enabled:
        return "Denoising off — the audio goes to the engine as recorded."
    chosen = resolve(settings)
    if chosen == NONE:
        wanted = settings.denoise_backend or AUTO
        if wanted == DEEP_FILTER:
            return (
                "deep-filter is not on PATH and no path is set in Settings — "
                "denoising will be skipped."
            )
        if wanted == PYTHON_DF:
            return (
                "The deepfilternet package is not importable in this Python — "
                "denoising will be skipped."
            )
        return "No denoiser is available (ffmpeg is missing) — denoising will be skipped."
    info = BACKENDS[chosen]
    if chosen == FFMPEG_DN and (settings.denoise_backend or AUTO) == AUTO:
        return (
            f"{info.label} — install the deep-filter binary for a real "
            "DeepFilterNet front-end (see Setup → what still needs installing)."
        )
    return info.label


# ----------------------------------------------------------------------
def enhance(
    path: str,
    settings: Settings,
    *,
    log_cb=None,
    progress_cb=None,
    should_cancel: ShouldCancel = None,
) -> str:
    """Return a denoised copy of ``path``, or ``path`` itself if that is not on.

    Never raises for a denoiser problem: a failed front-end means the job runs
    on the original audio, which is exactly what it did before.
    """
    def log(msg: str) -> None:
        if log_cb:
            log_cb(msg)

    backend = resolve(settings)
    if backend == NONE:
        if settings.denoise_enabled:
            log(f"Denoise: {describe(settings)}")
        return path

    cached = _cached_path(path, backend, settings)
    if cached.exists():
        log(f"Denoise: reusing the enhanced audio from an earlier run ({cached.name}).")
        return str(cached)

    check_cancel(should_cancel, log, message="Denoise: cancelled.")
    duration = probe(path).get("duration") or 0.0
    log(
        f"Denoise: {BACKENDS[backend].label}"
        + (f" — {duration / 60:.1f} min of audio…" if duration else "…")
    )
    started = time.monotonic()
    work = Path(tempfile.mkdtemp(prefix="pws_dn_"))
    try:
        if backend == FFMPEG_DN:
            produced = _ffmpeg_denoise(path, work, should_cancel, log)
        else:
            wav48 = _convert(path, work / "in48.wav", DF_SAMPLE_RATE, should_cancel, log)
            if backend == DEEP_FILTER:
                enhanced = _deep_filter_binary(
                    wav48, work, settings, should_cancel, log, progress_cb,
                    store=chunk_store(path, backend, settings),
                )
            else:
                enhanced = _deep_filter_python(wav48, work, settings, log)
            produced = _convert(
                enhanced, work / "out16.wav", PIPELINE_SAMPLE_RATE, should_cancel, log
            )
        _store(produced, cached)
        shutil.rmtree(chunk_store(path, backend, settings), ignore_errors=True)
    except JobCancelled:
        raise
    except Exception as e:
        log(f"Denoise failed ({e}) — transcribing the original audio instead.")
        return path
    finally:
        shutil.rmtree(work, ignore_errors=True)

    elapsed = time.monotonic() - started
    log(f"Denoise: done in {elapsed:.0f}s — {cached.name}")
    return str(cached)


# ---- backends --------------------------------------------------------
def ffmpeg_filter_chain() -> str:
    """High-pass out the rumble, then ffmpeg's FFT denoiser with noise tracking.

    Deliberately gentle: over-filtering smears consonants, and a smeared
    consonant costs more WER than the hiss it removed.
    """
    return "highpass=f=70,afftdn=nr=12:nf=-25:tn=1"


def _ffmpeg_denoise(path: str, work: Path, should_cancel, log) -> Path:
    out = work / "denoised.wav"
    _run(
        [
            FFMPEG, "-y", "-i", str(path),
            "-af", ffmpeg_filter_chain(),
            "-ac", "1", "-ar", str(PIPELINE_SAMPLE_RATE), str(out),
        ],
        should_cancel, log, "ffmpeg denoise",
    )
    return out


def deep_filter_command(
    binary: str,
    source: Path,
    out_dir: Path,
    model_path: str = "",
    postfilter: bool = False,
    atten_lim_db: int = 100,
) -> list[str]:
    """Build the deep-filter invocation.

    ``-D`` is not optional for us: without delay compensation the enhanced audio
    is shifted against the original by the STFT and model lookahead, and every
    timestamp — and every diarization boundary — shifts with it.
    """
    cmd = [binary, "-D", "-o", str(out_dir)]
    if model_path.strip():
        cmd += ["-m", model_path.strip()]
    if postfilter:
        cmd.append("--pf")
    if 0 <= atten_lim_db < 100:
        # Its own default is 100 (remove everything it can); passing that adds
        # nothing but noise to the command line.
        cmd += ["-a", str(atten_lim_db)]
    cmd.append(str(source))
    return cmd


def chunk_store(path: str, backend: str, settings: Settings) -> Path:
    """Where finished chunks for one recording live between runs.

    Denoising a long recording is minutes of work that a sleeping laptop can
    interrupt, and the tool writes nothing until it has finished the whole
    file — so without this, every interruption starts again from zero.
    """
    return CACHE_DIR / f"{cache_key(path, backend, settings)}.chunks"


def chunk_timeout(seconds: float) -> float:
    """How long a chunk of this length is allowed before it counts as stuck."""
    return max(MIN_CHUNK_TIMEOUT, seconds * CHUNK_TIMEOUT_FACTOR)


def split_into_chunks(source: Path, work: Path, should_cancel, log) -> list[Path]:
    """Cut the 48 kHz audio into fixed-length pieces for the denoiser."""
    parts = work / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    _run(
        [
            FFMPEG, "-y", "-i", str(source),
            "-f", "segment", "-segment_time", str(CHUNK_SECONDS),
            "-c", "copy", str(parts / "part%05d.wav"),
        ],
        should_cancel, log, "splitting the audio",
    )
    return sorted(parts.glob("part*.wav"))


def _deep_filter_binary(
    source: Path, work: Path, settings: Settings, should_cancel, log, progress_cb=None,
    store: Path | None = None,
) -> Path:
    """Denoise in chunks, so progress is visible and a stall costs one chunk.

    A chunk that fails or wedges falls back to its own original audio: the
    output stays complete and correctly timed, just less clean in that stretch.
    Losing two minutes of noise reduction beats losing the recording.

    Finished chunks are kept in ``store`` between runs. deep-filter writes
    nothing until it has processed an entire file, so without this an interrupted
    run — a laptop going to sleep mid-recording, say — throws away everything it
    had done and starts from the beginning.
    """
    binary = binary_path(settings.deep_filter_path)
    if not binary:
        raise RuntimeError("deep-filter binary disappeared between the check and the run.")

    chunks = split_into_chunks(source, work, should_cancel, log)
    if not chunks:
        raise RuntimeError("Splitting the audio produced nothing to denoise.")

    out_dir = work / "df"
    out_dir.mkdir(parents=True, exist_ok=True)
    if store is not None:
        store.mkdir(parents=True, exist_ok=True)
    total = len(chunks)
    if total > 1:
        log(f"Denoise: {total} chunk(s) of up to {CHUNK_SECONDS}s.")

    cleaned: list[Path] = []
    degraded = 0
    reused = 0
    for index, chunk in enumerate(chunks, start=1):
        check_cancel(should_cancel, log, message="Denoise: cancelled.")
        kept = (store / chunk.name) if store is not None else None
        if kept is not None and kept.exists():
            cleaned.append(kept)
            reused += 1
            if progress_cb:
                progress_cb(index / total)
            continue

        seconds = probe(str(chunk)).get("duration") or CHUNK_SECONDS
        try:
            _run(
                deep_filter_command(
                    binary, chunk, out_dir,
                    model_path=settings.denoise_model_path,
                    postfilter=settings.denoise_postfilter,
                    atten_lim_db=settings.denoise_atten_lim_db,
                ),
                should_cancel, log, f"deep-filter chunk {index}/{total}",
                timeout=chunk_timeout(seconds),
            )
            produced = out_dir / chunk.name
            result = produced if produced.exists() else chunk
            if kept is not None and produced.exists():
                shutil.copy2(str(produced), str(kept))
                result = kept
            cleaned.append(result)
        except JobCancelled:
            raise
        except (DenoiseTimeout, RuntimeError) as exc:
            degraded += 1
            log(f"Denoise: chunk {index}/{total} kept as recorded — {exc}")
            cleaned.append(chunk)
        if progress_cb:
            progress_cb(index / total)
        if total > 1 and (index % 5 == 0 or index == total):
            log(f"Denoise: {index}/{total} chunks done.")

    if reused:
        log(f"Denoise: {reused} chunk(s) reused from an interrupted run — no work repeated.")
    if degraded:
        log(f"Denoise: {degraded} of {total} chunk(s) could not be cleaned and were kept as-is.")
    if len(cleaned) == 1:
        return cleaned[0]
    return _concat(cleaned, work, should_cancel, log)


def _concat(parts: list[Path], work: Path, should_cancel, log) -> Path:
    """Join the denoised chunks back into one file, in order."""
    listing = work / "parts.txt"
    listing.write_text(
        "".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8"
    )
    joined = work / "denoised48.wav"
    _run(
        [FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
         "-c", "copy", str(joined)],
        should_cancel, log, "joining the chunks",
    )
    return joined


def _deep_filter_python(source: Path, work: Path, settings: Settings, log) -> Path:
    """The pip package's API. Audio is read and written here rather than through
    torchaudio, which this app avoids everywhere else for the same reason:
    its decoding backends are the flakiest part of the stack."""
    import soundfile as sf
    import torch
    from df.enhance import enhance as df_enhance
    from df.enhance import init_df

    model, df_state, _suffix = init_df(
        model_base_dir=(settings.denoise_model_path.strip() or None),
        post_filter=settings.denoise_postfilter,
    )
    if int(df_state.sr()) != DF_SAMPLE_RATE:
        log(f"Denoise: model runs at {df_state.sr()} Hz, resampling to match.")
        source = _convert(source, work / "in_model.wav", int(df_state.sr()), None, log)
    data, sr = sf.read(str(source), dtype="float32", always_2d=True)
    audio = torch.from_numpy(data.T.copy())
    enhanced = df_enhance(model, df_state, audio)
    out = work / "denoised48.wav"
    sf.write(str(out), enhanced.squeeze(0).cpu().numpy(), sr)
    return out


# ---- plumbing --------------------------------------------------------
def _convert(source, dest: Path, rate: int, should_cancel, log) -> Path:
    _run(
        [FFMPEG, "-y", "-i", str(source), "-ac", "1", "-ar", str(rate), str(dest)],
        should_cancel, log, f"resample to {rate} Hz",
    )
    return dest


class DenoiseTimeout(RuntimeError):
    """A stage that stopped making progress and had to be abandoned."""


def _run(
    cmd: list[str],
    should_cancel: ShouldCancel,
    log,
    what: str,
    timeout: float | None = None,
) -> None:
    """Run a converter, staying interruptible, and never waiting forever.

    The timeout is the important part. Without one, a child that wedges — which
    is exactly what a suspend/resume cycle can do to it — leaves the job sitting
    with no progress, no error and no way out but Cancel. One did precisely
    that for eight hours: it read its whole input, wrote nothing, and sat at one
    percent of a core while the app waited on poll().
    """
    check_cancel(should_cancel, log, message=f"Denoise: cancelled before {what}.")
    # Output goes to a file rather than a pipe nobody drains: a pipe buffer that
    # fills is its own way to deadlock a child.
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as sink:
        process = subprocess.Popen(cmd, stdout=sink, stderr=subprocess.STDOUT)
        deadline = (time.monotonic() + timeout) if timeout else None
        try:
            while process.poll() is None:
                if should_cancel and should_cancel():
                    _stop(process)
                    raise JobCancelled(f"Denoise: cancelled during {what}.")
                if deadline and time.monotonic() > deadline:
                    _stop(process)
                    raise DenoiseTimeout(
                        f"{what} made no progress in {timeout:.0f}s and was stopped."
                    )
                time.sleep(0.2)
        finally:
            if process.poll() is None:
                process.kill()
        if process.returncode != 0:
            sink.seek(0)
            output = sink.read().strip()
            tail = output.splitlines()[-1] if output else "no output"
            raise RuntimeError(f"{what} failed (exit {process.returncode}): {tail}")


def _stop(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def cache_key(path: str, backend: str, settings: Settings) -> str:
    """Same audio, same backend, same settings — same enhanced file."""
    source = Path(path)
    try:
        stat = source.stat()
        stamp = f"{stat.st_size}:{int(stat.st_mtime)}"
    except OSError:
        stamp = "0:0"
    material = "|".join(
        [
            str(source.resolve()),
            stamp,
            backend,
            settings.denoise_model_path.strip(),
            "pf" if settings.denoise_postfilter else "",
            str(settings.denoise_atten_lim_db),
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _cached_path(path: str, backend: str, settings: Settings) -> Path:
    return CACHE_DIR / f"{cache_key(path, backend, settings)}.wav"


def _store(produced: Path, dest: Path) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(produced), str(dest))
    prune()


def prune(keep: int = CACHE_KEEP) -> list[Path]:
    """Drop all but the most recent enhanced files. Returns what was removed."""
    if not CACHE_DIR.exists():
        return []
    files = sorted(CACHE_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    removed = []
    for path in files[keep:]:
        try:
            path.unlink()
            removed.append(path)
        except OSError:
            continue
    # Part-done chunk sets from runs that never finished. Anything still being
    # worked on was touched in the last hour, so it is left alone.
    cutoff = time.time() - 3600
    for directory in CACHE_DIR.glob("*.chunks"):
        try:
            if directory.stat().st_mtime < cutoff:
                shutil.rmtree(directory, ignore_errors=True)
                removed.append(directory)
        except OSError:
            continue
    return removed


def cache_size_bytes() -> int:
    if not CACHE_DIR.exists():
        return 0
    return sum(p.stat().st_size for p in CACHE_DIR.rglob("*.wav"))


def clear_cache() -> None:
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
