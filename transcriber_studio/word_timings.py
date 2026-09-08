# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Word timings for a transcript that has none.

Transcripts finished before word timings were kept have segments only. When
speakers are detected again on one of those, the turns have to be regrouped
from words, so the words are recovered here first: the text of each segment
is force-aligned to the audio under it with the wav2vec2 aligner that ships
with torchaudio. Measured against the AMI corpus's hand-timed words on a
headset mix: every word placed, median start error 59 ms, 78 percent within
100 ms; on the same meeting's distant table mic, median 67 ms and 91 percent
within 250 ms; forty segments in about a second on a laptop GPU. Where that
cannot
be done (a language the aligner does not
know, a segment with nothing alignable in it, the aligner not installed) the
words are spaced evenly across their segment instead and marked
``estimated``, which is still enough to split a turn where the speaker
changed halfway through.

Nothing here changes the transcript's text. The words are the segment's own
words; only their times are new.
"""
from __future__ import annotations

import re
import subprocess

from .audio_utils import FFMPEG, FFMPEG_TIMEOUT, have_ffmpeg
from .job_cancel import JobCancelled

SAMPLE_RATE = 16_000
#: Audio either side of a segment fed to the aligner, so a word the decoder
#: timed a little early or late still lies inside the window.
PAD_SECONDS = 0.25
#: The bundled aligner was trained on English speech. An empty language means
#: the engine did not say, which in this app almost always means English.
ALIGN_LANGUAGES = {"", "en", "en-us", "en-gb", "english"}
MIN_CHUNK_SECONDS = 0.1


def estimate_words(segments) -> list[dict]:
    """Space each segment's words evenly across it. The fallback, and honest
    about it: every word carries ``estimated: True``."""
    out: list[dict] = []
    for seg in segments:
        tokens = seg.text.split()
        if not tokens:
            continue
        step = max(0.0, seg.end - seg.start) / len(tokens)
        for k, tok in enumerate(tokens):
            if k:
                out.append({"type": "spacing", "text": " "})
            out.append({
                "type": "word", "text": tok,
                "start": seg.start + k * step, "end": seg.start + (k + 1) * step,
                "estimated": True,
            })
    return out


def recover_words(audio_path: str, segments, language: str = "", log=None,
                  should_cancel=None) -> list[dict]:
    """Word timings for ``segments``, aligned to ``audio_path`` where possible."""
    log = log or (lambda _m: None)
    if not segments:
        return []
    if (language or "").strip().lower() not in ALIGN_LANGUAGES:
        log(f"Word timings: the aligner knows English only and this transcript is "
            f"'{language}'; spacing words evenly within each turn instead.")
        return estimate_words(segments)
    try:
        aligner = _Aligner(log)
        samples = load_mono(audio_path)
    except Exception as exc:  # the aligner is optional; the estimate is not
        log(f"Word timings: aligner unavailable ({exc}); spacing words evenly "
            f"within each turn instead.")
        return estimate_words(segments)

    out: list[dict] = []
    aligned = estimated = 0
    for seg in segments:
        if should_cancel and should_cancel():
            raise JobCancelled("Cancelled while recovering word timings.")
        words = aligner.align(samples, seg)
        if words is None:
            words = estimate_words([seg])
            estimated += 1
        else:
            aligned += 1
        if out and words:
            out.append({"type": "spacing", "text": " "})
        out.extend(words)
    log(f"Word timings recovered by aligning the text to the audio: {aligned} turn(s) "
        f"aligned, {estimated} spaced evenly.")
    return out


def load_mono(audio_path: str):
    """The whole recording as 16 kHz mono float32 samples, via ffmpeg."""
    import numpy as np

    if not have_ffmpeg():
        raise RuntimeError("ffmpeg is required to read the audio")
    proc = subprocess.run(
        [FFMPEG, "-v", "error", "-i", audio_path, "-ac", "1", "-ar", str(SAMPLE_RATE),
         "-f", "f32le", "-"],
        capture_output=True, check=True, timeout=FFMPEG_TIMEOUT,
    )
    return np.frombuffer(proc.stdout, dtype=np.float32)


def assemble(seg, tokens: list[str], alignable: list[bool],
             times: list[tuple[float, float]]) -> list[dict]:
    """Words for one segment: aligned times for the tokens the aligner placed,
    and the gaps between them shared out over the tokens it could not (numbers,
    symbols), so every word has a time and the order is kept."""
    out: list[dict] = []
    placed = iter(times)
    starts_ends: list[tuple[float, float] | None] = [
        next(placed) if ok else None for ok in alignable
    ]
    n = len(tokens)
    i = 0
    while i < n:
        if starts_ends[i] is not None:
            i += 1
            continue
        j = i
        while j < n and starts_ends[j] is None:
            j += 1
        gap_start = starts_ends[i - 1][1] if i else seg.start
        gap_end = starts_ends[j][0] if j < n else seg.end
        if gap_end < gap_start:
            gap_end = gap_start
        step = (gap_end - gap_start) / (j - i)
        for k in range(i, j):
            starts_ends[k] = (gap_start + (k - i) * step, gap_start + (k - i + 1) * step)
        i = j
    for k, tok in enumerate(tokens):
        if k:
            out.append({"type": "spacing", "text": " "})
        start, end = starts_ends[k]
        word = {"type": "word", "text": tok, "start": start, "end": end}
        if not alignable[k]:
            word["estimated"] = True
        out.append(word)
    return out


class _Aligner:
    """torchaudio's wav2vec2 CTC aligner, English, loaded once per run."""

    def __init__(self, log):
        import torch
        import torchaudio.functional as F
        from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H as bundle

        self.torch, self.F = torch, F
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log("Word timings: loading the wav2vec2 aligner (downloaded on first use)…")
        self.model = bundle.get_model().to(self.device).eval()
        self.labels = bundle.get_labels()
        self.index = {c: i for i, c in enumerate(self.labels)}
        self.blank = 0
        self.sep = self.index["|"]
        self.keep = re.compile("[^" + re.escape("".join(
            c for c in self.labels if c not in ("-", "|")
        )) + "]")

    def _normalise(self, token: str) -> str:
        return self.keep.sub("", token.upper())

    def align(self, samples, seg) -> list[dict] | None:
        """Timed words for one segment, or None when it cannot be aligned."""
        tokens = seg.text.split()
        if not tokens:
            return []
        s0 = max(0.0, seg.start - PAD_SECONDS)
        s1 = seg.end + PAD_SECONDS
        chunk = samples[int(s0 * SAMPLE_RATE):int(s1 * SAMPLE_RATE)]
        if len(chunk) < SAMPLE_RATE * MIN_CHUNK_SECONDS:
            return None

        target: list[int] = []
        alignable: list[bool] = []
        for tok in tokens:
            norm = self._normalise(tok)
            alignable.append(bool(norm))
            if not norm:
                continue
            if target:
                target.append(self.sep)
            target.extend(self.index[c] for c in norm)
        if not any(alignable):
            return None

        torch = self.torch
        with torch.inference_mode():
            wav = torch.from_numpy(chunk.copy()).to(self.device)[None]
            emission, _ = self.model(wav)
            emission = torch.log_softmax(emission, dim=-1).cpu()
        frames = emission.shape[1]
        if len(target) > frames:
            return None
        targets = torch.tensor([target], dtype=torch.int32)
        try:
            aligned, scores = self.F.forced_align(emission, targets, blank=self.blank)
        except Exception:
            return None
        spans = self.F.merge_tokens(aligned[0], scores[0].exp())

        seconds_per_frame = (len(chunk) / SAMPLE_RATE) / frames
        times: list[tuple[float, float]] = []
        current: tuple[float, float] | None = None
        for span in spans:
            if span.token == self.sep:
                if current:
                    times.append(current)
                current = None
                continue
            start = s0 + span.start * seconds_per_frame
            end = s0 + span.end * seconds_per_frame
            current = (start, end) if current is None else (current[0], end)
        if current:
            times.append(current)
        if len(times) != sum(alignable):
            return None
        return assemble(seg, tokens, alignable, times)
