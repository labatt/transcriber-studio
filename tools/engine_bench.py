# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run every speech engine over the same recordings and tabulate what came back.

Each configuration goes through ``Transcriber.transcribe`` exactly as a job
would, with the user's saved settings (glossary hotwords, VAD, diarization)
so the numbers describe the app, not a bare API call.

There is no ground-truth transcript, so nothing here is an accuracy score.
"Agreement" is the word error rate of one engine's words against another's,
which says how far apart they are and not which one is right.

    python tools/engine_bench.py --out DIR REC1.mp3 REC2.mp3
    python tools/engine_bench.py --out DIR --only mai-verbatim,gemini-verbatim REC.mp3
    python tools/engine_bench.py --out DIR --cap elevenlabs-scribe=25 REC1.mp3 REC2.mp3
    python tools/engine_bench.py --out DIR --report-only

Results are saved after every run, and a run that already has a result is
skipped, so the script can be stopped and restarted. ``--cap NAME=MINUTES``
skips a configuration on recordings longer than that, for engines billed
against a monthly quota.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_refs  # noqa: E402  (sibling module; scores against reference transcripts)

from transcriber_studio import config, vad, vocab_bias  # noqa: E402
from transcriber_studio.models import Recording, Source  # noqa: E402
from transcriber_studio.transcriber import (  # noqa: E402
    ENGINE_ELEVENLABS,
    ENGINE_GEMINI,
    ENGINE_LOCAL,
    ENGINE_MAI,
    TranscribeOptions,
    Transcriber,
)

FILLERS = re.compile(r"\b(um+|uh+|hmm+|mm-?hmm|uh-?huh|erm)\b", re.IGNORECASE)
WORD = re.compile(r"[a-z0-9']+")
LOW_CONFIDENCE = 0.50

# Cloud engines first: they are quick, and a crash during a long local decode
# should not cost a re-upload to recover their results.
ORDER = [
    "mai-verbatim",
    "mai-clean",
    "elevenlabs-scribe",
    "gemini-verbatim",
    "whisper-large-v3-turbo",
    "whisper-large-v3",
    # Diagnostic: the same decoder with the VAD front end off, to apportion
    # dropped speech between the two. Run with --only; not part of the default set.
    "whisper-large-v3-turbo-novad",
]
DEFAULT_CONFIGS = ORDER[:-1]


def configs(s: config.Settings, hints: bool = True) -> dict[str, TranscribeOptions]:
    """One TranscribeOptions per configuration. ``hints=False`` sends no
    glossary to any engine, for recordings the user's vocabulary has nothing
    to do with."""
    common: dict[str, Any] = dict(
        device=s.device,
        compute_type=s.compute_type,
        language=s.language,
        diarization_enabled=True,
        hf_token=s.hf_token,
        vad_enabled=s.vad_enabled,
        vad_parameters=vad.parameters(s),
        hotwords=vocab_bias.hotwords(s) if hints else "",
        hallucination_guard=s.hallucination_guard,
        repetition_penalty=s.repetition_penalty,
        no_repeat_ngram_size=s.no_repeat_ngram_size,
        elevenlabs_api_key=s.elevenlabs_api_key,
        elevenlabs_model=s.elevenlabs_model,
        gemini_api_key=s.ai_key_google,
        gemini_model=s.gemini_model,
        gemini_mode="verbatim",
        mai_api_key=s.mai_api_key,
        mai_region=s.mai_region,
        mai_model=s.mai_model,
        mai_send_phrases=s.mai_send_phrases and hints,
        mai_speakers="local",
    )
    return {
        "mai-verbatim": TranscribeOptions(engine=ENGINE_MAI, mai_style="verbatim", **common),
        "mai-clean": TranscribeOptions(engine=ENGINE_MAI, mai_style="clean", **common),
        "elevenlabs-scribe": TranscribeOptions(engine=ENGINE_ELEVENLABS, **common),
        "gemini-verbatim": TranscribeOptions(engine=ENGINE_GEMINI, **common),
        "whisper-large-v3-turbo": TranscribeOptions(
            engine=ENGINE_LOCAL, model="large-v3-turbo", **common
        ),
        "whisper-large-v3": TranscribeOptions(engine=ENGINE_LOCAL, model="large-v3", **common),
        "whisper-large-v3-turbo-novad": TranscribeOptions(
            engine=ENGINE_LOCAL, model="large-v3-turbo",
            **{**common, "vad_enabled": False},
        ),
    }


def duration_of(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def normalise(text: str) -> list[str]:
    return WORD.findall(text.lower().replace("’", "'"))


def metrics(result, duration: float, seconds: float, terms: list[str],
            log_lines: list[str]) -> dict:
    segs = result.segments
    text = " ".join(s.text for s in segs)
    words = normalise(text)
    speech = sum(max(0.0, s.end - s.start) for s in segs)
    pairs = list(zip(segs, segs[1:], strict=False))
    turns = sum(1 for a, b in pairs if a.speaker != b.speaker)
    conf = [s.confidence for s in segs if s.confidence is not None]
    dup = sum(
        1 for a, b in pairs
        if normalise(a.text) and normalise(a.text) == normalise(b.text)
    )
    grams = Counter(tuple(words[i:i + 5]) for i in range(len(words) - 4))
    top_gram, top_n = grams.most_common(1)[0] if grams else ((), 0)
    lowered = text.lower()
    hits = sorted({
        t for t in terms
        if len(t) >= 3 and re.search(r"\b" + re.escape(t.lower()) + r"\b", lowered)
    })
    return {
        "wall_s": round(seconds, 1),
        "realtime_factor": round(duration / seconds, 2) if seconds else None,
        "segments": len(segs),
        "words": len(words),
        "speech_coverage": round(speech / duration, 3) if duration else None,
        "speakers": len(result.speakers),
        "speaker_labels": list(result.speakers),
        "speaker_turns": turns,
        "unlabelled_segments": sum(1 for s in segs if not s.speaker),
        "conf_segments": len(conf),
        "conf_mean": round(sum(conf) / len(conf), 3) if conf else None,
        "conf_low": sum(1 for c in conf if c < LOW_CONFIDENCE),
        "duplicate_adjacent": dup,
        "top_5gram": " ".join(top_gram),
        "top_5gram_count": top_n,
        "fillers": len(FILLERS.findall(text)),
        "numbers": len(re.findall(r"\d[\d,.:%$]*", text)),
        "glossary_hits": len(hits),
        "glossary_hit_terms": hits,
        "glossary_terms_total": len(terms),
        "retries": sum(1 for line in log_lines if "retry" in line.lower()),
        "language": result.language,
        "model": result.model,
        "voiceprint_embeddings": len(result.speaker_embeddings),
        "named_speakers": [
            sp for sp in result.speakers if not sp.lower().startswith("speaker")
        ],
    }


def transcript_lines(result) -> list[str]:
    return [
        f"[{s.start:8.1f} - {s.end:8.1f}] {(s.speaker or '-'):>10}: {s.text}"
        for s in result.segments
    ]


def free_gpu() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _logger(lines: list[str], path: Path):
    """A log callback that keeps the lines and appends them to ``path``."""
    def log(msg: str) -> None:
        lines.append(msg)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"{stamp()} {msg}\n")
    return log


def reference_for(audio: Path) -> dict | None:
    """``NAME.ref.json`` beside the audio, where NAME is the stem or the stem
    up to its first dot (``ES2004b.Array1-01.mp3`` shares ``ES2004b.ref.json``)."""
    for stem in (audio.stem, audio.stem.split(".")[0]):
        ref = audio.with_name(f"{stem}.ref.json")
        if ref.exists():
            return json.loads(ref.read_text(encoding="utf-8"))
    return None


def _skipped(summary: dict, summary_path: Path, key: str, name: str, why: str) -> None:
    summary.setdefault(key, {})[name] = {"error": f"skipped: {why}", "wall_s": 0.0,
                                         "retries": 0}
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run(paths: list[Path], out: Path, only: set[str] | None,
        caps: dict[str, float], skips: dict[str, set[str]], hints: bool = True) -> None:
    s = config.load()
    terms = vocab_bias.collect_terms(s)
    cfgs = configs(s, hints)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    summary = (json.loads(summary_path.read_text(encoding="utf-8"))
               if summary_path.exists() else {})

    recordings = []
    for p in paths:
        d = duration_of(p)
        ref = reference_for(p)
        recordings.append((p, d, ref))
        summary.setdefault(p.stem, {})["_duration_s"] = round(d, 1)
        summary[p.stem]["_reference"] = ref["source"] if ref else None
        print(f"{stamp()} recording {p.name}: {d / 60:.1f} min"
              + (f", reference {ref['source']}" if ref else ", no reference"), flush=True)

    for name in ORDER:
        if (only and name not in only) or (not only and name not in DEFAULT_CONFIGS):
            continue
        opts = cfgs[name]
        for p, d, ref in recordings:
            key = p.stem
            done = summary.get(key, {}).get(name)
            if done and "error" not in done:
                print(f"{stamp()} skip {name} on {p.name} (already done)", flush=True)
                continue
            cap = caps.get(name)
            if cap is not None and d / 60 > cap:
                print(f"{stamp()} skip {name} on {p.name} (over the {cap:g} min cap)",
                      flush=True)
                _skipped(summary, summary_path, key, name,
                         f"recording is over the {cap:g} minute cap set for this engine")
                continue
            if key in skips.get(name, set()):
                print(f"{stamp()} skip {name} on {p.name} (excluded by --skip)", flush=True)
                _skipped(summary, summary_path, key, name, "excluded by --skip")
                continue
            rec_dir = out / key
            rec_dir.mkdir(exist_ok=True)
            log_lines: list[str] = []
            log = _logger(log_lines, rec_dir / f"{name}.log")
            recording = Recording(
                source=Source.LOCAL, id=str(p), name=p.stem,
                local_path=str(p), duration_seconds=d,
            )
            print(f"{stamp()} start {name} on {p.name}", flush=True)
            t0 = time.monotonic()
            try:
                result = Transcriber().transcribe(recording, str(p), opts, None, log)
            except Exception as exc:  # one engine failing is itself a result
                seconds = time.monotonic() - t0
                log(f"FAILED after {seconds:.0f}s: {exc!r}")
                log(traceback.format_exc())
                summary.setdefault(key, {})[name] = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "wall_s": round(seconds, 1),
                    "retries": sum(1 for line in log_lines if "retry" in line.lower()),
                }
                print(f"{stamp()} FAILED {name} on {p.name}: {exc}", flush=True)
            else:
                seconds = time.monotonic() - t0
                m = metrics(result, d, seconds, terms, log_lines)
                if ref:
                    try:
                        m.update(bench_refs.score(result.segments, result.speakers, ref))
                    except Exception as exc:  # scoring must not lose the transcript
                        m["score_error"] = f"{type(exc).__name__}: {exc}"
                        log(f"scoring failed: {exc!r}\n{traceback.format_exc()}")
                summary.setdefault(key, {})[name] = m
                (rec_dir / f"{name}.txt").write_text(
                    "\n".join(transcript_lines(result)), encoding="utf-8"
                )
                (rec_dir / f"{name}.words.json").write_text(
                    json.dumps(normalise(" ".join(sg.text for sg in result.segments))),
                    encoding="utf-8",
                )
                scored = f", WER {m['wer']:.3f}" if "wer" in m else ""
                print(
                    f"{stamp()} done  {name} on {p.name}: {seconds:.0f}s, "
                    f"{m['words']} words, {m['speakers']} speakers, "
                    f"{m['glossary_hits']} glossary hits{scored}",
                    flush=True,
                )
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            free_gpu()
    report(out)


def wer(ref: list[str], hyp: list[str]) -> float:
    return Levenshtein.distance(ref, hyp) / max(1, len(ref))


def agreement(rec_dir: Path, names: list[str]) -> dict[str, dict[str, float]]:
    words = {}
    for n in names:
        f = rec_dir / f"{n}.words.json"
        if f.exists():
            words[n] = json.loads(f.read_text(encoding="utf-8"))
    table: dict[str, dict[str, float]] = {}
    for a in words:
        table[a] = {}
        for b in words:
            if a != b:
                table[a][b] = round(wer(words[a], words[b]), 3)
    return table


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.2f}"
    if isinstance(v, list):
        return ", ".join(v) or "-"
    return str(v)


SCORE_ROWS = [
    ("wer", "Word error rate"),
    ("substitutions", "Substitutions"),
    ("deletions", "Deletions"),
    ("insertions", "Insertions"),
    ("proper_noun_recall", "Proper-noun recall"),
    ("proper_nouns_total", "Proper nouns in reference"),
    ("speaker_attribution", "Speaker attribution accuracy"),
    ("der", "Diarization error rate"),
    ("ref_speakers", "Speakers in reference"),
    ("speaker_count_error", "Speaker count error"),
    ("words_in_silence", "Words during reference silence"),
]

ROWS = [
    ("wall_s", "Wall time (s)"),
    ("realtime_factor", "Audio min per wall min"),
    ("words", "Words"),
    ("segments", "Segments"),
    ("speech_coverage", "Speech coverage"),
    ("speakers", "Speakers"),
    ("speaker_turns", "Speaker turns"),
    ("named_speakers", "Voiceprint names"),
    ("glossary_hits", "Glossary terms found"),
    ("fillers", "Fillers (um, uh, hmm)"),
    ("numbers", "Numbers"),
    ("duplicate_adjacent", "Repeated adjacent segments"),
    ("top_5gram_count", "Most repeated 5-gram (count)"),
    ("conf_segments", "Segments with confidence"),
    ("conf_mean", "Mean confidence"),
    ("conf_low", "Low-confidence segments"),
    ("retries", "Retries"),
    ("voiceprint_embeddings", "Voice embeddings returned"),
]


def report(out: Path) -> None:
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    lines = [
        "# Engine benchmark", "", f"Generated {datetime.now():%Y-%m-%d %H:%M}.", "",
        "Every configuration ran through the app's own transcribe path with the saved "
        "settings (VAD, local pyannote speaker separation for the local and MAI engines). "
        "Where a recording has a reference transcript, the first block of rows scores "
        "against it: WER after Whisper's English normaliser, proper-noun recall, speaker "
        "attribution on aligned words, and diarization error rate. The agreement tables "
        "at the end of each section measure distance between engines, not accuracy.",
        "",
    ]
    for key, data in summary.items():
        names = [n for n in ORDER if n in data]
        ref_name = data.get("_reference")
        lines += [f"## {key}", "", f"Duration: {data.get('_duration_s', 0) / 60:.1f} min"
                  + (f". Reference: {ref_name}." if ref_name else ". No reference transcript."),
                  ""]
        lines.append("| Attribute | " + " | ".join(names) + " |")
        lines.append("|---|" + "---|" * len(names))
        rows = (SCORE_ROWS if ref_name else []) + ROWS
        for field, label in rows:
            cells = []
            for n in names:
                m = data[n]
                if "error" in m and field not in ("wall_s", "retries"):
                    cells.append("failed" if field == "words" else "-")
                else:
                    cells.append(_fmt(m.get(field)))
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        errors = [(n, data[n]["error"]) for n in names if "error" in data[n]]
        if errors:
            lines += ["", "Not run or failed:", ""]
            lines += [f"- {n}: {e}" for n, e in errors]
        missed = {n: data[n].get("missed_proper_nouns") for n in names
                  if data[n].get("missed_proper_nouns")}
        if missed:
            everyone = set.intersection(*(set(v) for v in missed.values()))
            lines += ["", "Proper nouns missed:", ""]
            if everyone:
                lines.append(f"- by every engine: {', '.join(sorted(everyone)[:40])}")
            for n, v in missed.items():
                own = sorted(set(v) - everyone)
                if own:
                    lines.append(f"- {n} only: {', '.join(own[:30])}")
        table = agreement(out / key, names)
        if table:
            ok = [n for n in names if n in table]
            lines += ["", "Word error rate of each row against each column "
                          "(lower means closer):", ""]
            lines.append("| | " + " | ".join(ok) + " | mean |")
            lines.append("|---|" + "---|" * (len(ok) + 1))
            for a in ok:
                vals = [table[a].get(b) for b in ok]
                mean = sum(v for v in vals if v is not None) / max(1, len(ok) - 1)
                lines.append(
                    f"| {a} | "
                    + " | ".join("-" if v is None else f"{v:.3f}" for v in vals)
                    + f" | {mean:.3f} |"
                )
        hit_sets = {
            n: set(data[n].get("glossary_hit_terms", []))
            for n in names if "error" not in data[n]
        }
        if len(hit_sets) > 1:
            union = set().union(*hit_sets.values())
            contested = sorted(
                t for t in union if any(t not in h for h in hit_sets.values())
            )
            if contested:
                lines += ["", "Glossary terms not found by every engine:", ""]
                lines.append("| Term | " + " | ".join(hit_sets) + " |")
                lines.append("|---|" + "---|" * len(hit_sets))
                for t in contested:
                    lines.append(
                        f"| {t} | "
                        + " | ".join("yes" if t in hit_sets[n] else "" for n in hit_sets)
                        + " |"
                    )
        lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{stamp()} report written to {out / 'report.md'}", flush=True)


def _parse_skips(raw: list[str] | None) -> dict[str, set[str]]:
    skips: dict[str, set[str]] = {}
    for item in raw or []:
        name, _, stems = item.partition("=")
        if name not in ORDER or not stems:
            raise SystemExit(f"--skip expects NAME=STEM[,STEM] with a known name, got {item!r}")
        skips.setdefault(name, set()).update(s.strip() for s in stems.split(","))
    return skips


def _parse_caps(raw: list[str] | None) -> dict[str, float]:
    caps: dict[str, float] = {}
    for item in raw or []:
        name, _, minutes = item.partition("=")
        if name not in ORDER or not minutes:
            raise SystemExit(f"--cap expects NAME=MINUTES with a known name, got {item!r}")
        caps[name] = float(minutes)
    return caps


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("recordings", nargs="*", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--only", help="comma-separated configuration names")
    ap.add_argument("--cap", action="append", metavar="NAME=MINUTES",
                    help="skip NAME on recordings longer than MINUTES")
    ap.add_argument("--skip", action="append", metavar="NAME=STEM[,STEM]",
                    help="skip NAME on the recordings with these file stems")
    ap.add_argument("--no-hints", action="store_true",
                    help="send no glossary hints to any engine")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--list", action="store_true", help="print configuration names")
    args = ap.parse_args(argv)
    if args.list:
        print("\n".join(ORDER))
        return 0
    if args.out is None:
        ap.error("--out is required")
    if args.report_only:
        report(args.out)
        return 0
    if not args.recordings:
        ap.error("no recordings given")
    only = set(args.only.split(",")) if args.only else None
    unknown = (only or set()) - set(ORDER)
    if unknown:
        ap.error(f"unknown configuration(s): {', '.join(sorted(unknown))}")
    run(args.recordings, args.out, only, _parse_caps(args.cap), _parse_skips(args.skip),
        hints=not args.no_hints)
    return 0


if __name__ == "__main__":
    sys.exit(main())
