# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Reference transcripts for the engine benchmark, and the scores against them.

Two public corpora are understood:

* AMI Meeting Corpus (CC BY 4.0): NXT ``words/*.words.xml`` and
  ``segments/*.segments.xml`` per speaker, with ``corpusResources/meetings.xml``
  mapping the A-D agents to people.
* Earnings-22 (CC BY-SA 4.0): Rev's pipe-delimited ``.nlp`` token files with a
  speaker and a timestamp per token.

Both become one JSON shape::

    {"words": [{"text", "start", "end", "speaker"}],
     "segments": [{"start", "end", "speaker"}]}

``score()`` compares a transcript against that: word error rate after Whisper's
English normaliser (so number words, fillers and punctuation do not count),
recall on proper nouns, speaker attribution accuracy on aligned words after an
optimal speaker mapping, diarization error rate, and words produced where the
reference has silence, which is what a hallucination looks like.

    python tools/bench_refs.py ami --nxt DIR --meeting ES2004b --out ES2004b.ref.json
    python tools/bench_refs.py earnings --nlp 4474506.aligned.nlp --out 4474506.ref.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET  # the corpora are trusted local files
from collections import Counter
from pathlib import Path

from rapidfuzz.distance import Levenshtein

SEGMENT_GAP_S = 1.0        # Earnings tokens closer than this by one speaker share a segment
SILENCE_PAD_S = 1.0        # a hypothesis word this far from any reference word is "in silence"
SENTENCE_GAP_S = 2.0       # a pause this long starts a new sentence for capitalisation
FIRST_PERSON = {"i", "i'm", "i'll", "i've", "i'd"}
SENTENCE_END = (".", "?", "!")


# ---------------------------------------------------------------- references
def ami_reference(nxt_dir: Path, meeting: str) -> dict:
    names = _ami_speaker_names(nxt_dir / "corpusResources" / "meetings.xml", meeting)
    words: list[dict] = []
    segments: list[dict] = []
    for wfile in sorted((nxt_dir / "words").glob(f"{meeting}.*.words.xml")):
        agent = wfile.name.split(".")[1]
        speaker = names.get(agent, agent)
        # Each speaker's words are one file, so "previous" is that speaker's
        # previous token. AMI capitalises the first word after a full stop and
        # the first word of an utterance; both are marked so they do not pass
        # for names.
        sentence_start = True
        last_end = -1e9
        for el in ET.parse(wfile).getroot():
            text = (el.text or "").strip()
            if el.tag != "w" or not text:
                continue
            if el.get("punc") == "true":
                sentence_start = sentence_start or text in SENTENCE_END
                continue
            start = float(el.get("starttime") or 0.0)
            words.append({
                "text": text,
                "start": start,
                "end": float(el.get("endtime") or start),
                "speaker": speaker,
                "si": sentence_start or start - last_end > SENTENCE_GAP_S,
            })
            sentence_start = False
            last_end = words[-1]["end"]
        sfile = nxt_dir / "segments" / f"{meeting}.{agent}.segments.xml"
        if sfile.exists():
            for seg in ET.parse(sfile).getroot():
                if seg.tag == "segment":
                    segments.append({
                        "start": float(seg.get("transcriber_start") or 0.0),
                        "end": float(seg.get("transcriber_end") or 0.0),
                        "speaker": speaker,
                    })
    words.sort(key=lambda w: w["start"])
    segments.sort(key=lambda s: s["start"])
    return {"source": f"AMI {meeting}", "words": words, "segments": segments}


def _ami_speaker_names(meetings_xml: Path, meeting: str) -> dict[str, str]:
    for m in ET.parse(meetings_xml).getroot():
        if m.get("observation") == meeting:
            return {str(sp.get("nxt_agent")): str(sp.get("global_name") or sp.get("nxt_agent"))
                    for sp in m if sp.tag == "speaker"}
    return {}


def earnings_reference(nlp_path: Path) -> dict:
    words: list[dict] = []
    sentence_start = True
    last_end = -1e9
    with nlp_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="|"):
            token = (row.get("token") or "").strip()
            if not token or not row.get("ts"):
                continue
            start = float(row["ts"])
            words.append({
                "text": token,
                "start": start,
                "end": float(row.get("endTs") or start),
                "speaker": f"S{row.get('speaker') or '0'}",
                "case": row.get("case") or "",
                "tags": row.get("tags") or "",
                "si": sentence_start or start - last_end > SENTENCE_GAP_S,
            })
            # Rev stores the punctuation that follows a token on the token's
            # own row, so this row decides whether the next one opens a sentence.
            sentence_start = (row.get("punctuation") or "").strip() in SENTENCE_END and \
                bool((row.get("punctuation") or "").strip())
            last_end = words[-1]["end"]
    words.sort(key=lambda w: w["start"])
    return {
        "source": f"Earnings-22 {nlp_path.stem.split('.')[0]}",
        "words": words,
        "segments": segments_from_words(words),
    }


def segments_from_words(words: list[dict], gap: float = SEGMENT_GAP_S) -> list[dict]:
    segments: list[dict] = []
    for w in words:
        last = segments[-1] if segments else None
        if last and last["speaker"] == w["speaker"] and w["start"] - last["end"] <= gap:
            last["end"] = max(last["end"], w["end"])
        else:
            segments.append({"start": w["start"], "end": w["end"], "speaker": w["speaker"]})
    return segments


# ---------------------------------------------------------------- normaliser
_normaliser = None


def normaliser():
    """Whisper's English text normaliser, as used by the Open ASR Leaderboard."""
    global _normaliser
    if _normaliser is None:
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
        mapping: dict = {}
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download("openai/whisper-large-v3", "normalizer.json")
            mapping = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            pass  # spelling-variant mapping only; scores still comparable across engines
        _normaliser = EnglishTextNormalizer(mapping)
    return _normaliser


def norm_words(text: str) -> list[str]:
    return normaliser()(text).split()


# ---------------------------------------------------------------- scoring
def proper_nouns(ref_words: list[dict]) -> set[int]:
    """Indexes of reference words that look like names: capitalised, not
    sentence-initial, not the pronoun I. Earnings-22 marks case explicitly."""
    out: set[int] = set()
    for i, w in enumerate(ref_words):
        text = w["text"]
        case = w.get("case") or ""
        looks = case in ("UC", "CA") if case else text[:1].isupper()
        if not looks or w.get("si") or text.lower() in FIRST_PERSON:
            continue
        # Spelled-out letters (AMI writes them "A_", "T_"), lone letters, and
        # number words the normaliser turns into digits are not names.
        if "_" in text or len(text) < 2:
            continue
        normalised = norm_words(text)
        if not normalised or any(t.isdigit() for t in normalised):
            continue
        out.add(i)
    return out


def _hyp_words(segments) -> list[dict]:
    """Normalised words of a transcript, each tagged with its segment's speaker
    and an interpolated time inside the segment."""
    out: list[dict] = []
    for seg in segments:
        toks = norm_words(seg.text)
        if not toks:
            continue
        step = max(0.0, seg.end - seg.start) / len(toks)
        for k, t in enumerate(toks):
            out.append({"text": t, "speaker": seg.speaker or "?",
                        "time": seg.start + step * (k + 0.5)})
    return out


def _ref_tokens(ref_words: list[dict]) -> list[dict]:
    """Reference words after normalisation. One reference word can become
    several tokens or none, so each token remembers its source index."""
    out: list[dict] = []
    for i, w in enumerate(ref_words):
        for t in norm_words(w["text"]):
            out.append({"text": t, "speaker": w["speaker"], "start": w["start"],
                        "end": w["end"], "src": i})
    return out


def score(segments, speakers: list[str], ref: dict) -> dict:
    ref_words = ref["words"]

    # Headline WER on whole texts, normalised in context, as the Open ASR
    # Leaderboard does. Normalising word by word would turn "twenty two" into
    # "20 2" and charge the engine for writing "22".
    r_full = norm_words(" ".join(w["text"] for w in ref_words))
    h_full = norm_words(" ".join(s.text for s in segments))
    sub = dele = ins = 0
    for op in Levenshtein.opcodes(r_full, h_full):
        if op.tag == "replace":
            sub += max(op.src_end - op.src_start, op.dest_end - op.dest_start)
        elif op.tag == "delete":
            dele += op.src_end - op.src_start
        elif op.tag == "insert":
            ins += op.dest_end - op.dest_start
    wer = (sub + dele + ins) / max(1, len(r_full))

    # Everything that needs to know *which* reference word matched uses a
    # per-token alignment instead, where each token remembers its source.
    rt = _ref_tokens(ref_words)
    ht = _hyp_words(segments)
    r = [t["text"] for t in rt]
    h = [t["text"] for t in ht]
    matched: list[tuple[int, int]] = []
    for op in Levenshtein.opcodes(r, h):
        if op.tag == "equal":
            matched.extend(zip(range(op.src_start, op.src_end),
                               range(op.dest_start, op.dest_end), strict=True))

    # Proper-noun recall: a name counts when every one of its tokens aligned.
    names = proper_nouns(ref_words)
    hit_src = Counter(rt[i]["src"] for i, _ in matched)
    need_src = Counter(t["src"] for t in rt)
    name_hits = sum(1 for i in names if hit_src[i] == need_src[i] and need_src[i])
    name_total = sum(1 for i in names if need_src[i])
    missed_names = sorted({ref_words[i]["text"] for i in names
                           if need_src[i] and hit_src[i] < need_src[i]})

    # Speaker attribution on aligned words, after the best one-to-one mapping.
    confusion: Counter = Counter((ht[j]["speaker"], rt[i]["speaker"]) for i, j in matched)
    mapping = _best_mapping(confusion)
    correct = sum(n for (hs, rs), n in confusion.items() if mapping.get(hs) == rs)
    attribution = correct / len(matched) if matched else None

    # Words spoken by nobody: hypothesis words far from every reference word.
    ref_times = sorted((w["start"], w["end"]) for w in ref_words)
    in_silence = sum(1 for t in ht if _in_silence(t["time"], ref_times))

    der = _der(ref["segments"], segments)
    ref_speakers = len({w["speaker"] for w in ref_words})
    return {
        "wer": round(wer, 4),
        "substitutions": sub, "deletions": dele, "insertions": ins,
        "ref_words": len(r_full), "hyp_words": len(h_full),
        "proper_noun_recall": round(name_hits / name_total, 3) if name_total else None,
        "proper_nouns_total": name_total,
        "missed_proper_nouns": missed_names[:60],
        "speaker_attribution": round(attribution, 3) if attribution is not None else None,
        "ref_speakers": ref_speakers,
        "speaker_count_error": len(speakers) - ref_speakers,
        "der": der,
        "words_in_silence": in_silence,
    }


def _best_mapping(confusion: Counter) -> dict[str, str]:
    hyp = sorted({k[0] for k in confusion})
    refs = sorted({k[1] for k in confusion})
    if not hyp or not refs:
        return {}
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        cost = np.zeros((len(hyp), len(refs)))
        for (hs, rs), n in confusion.items():
            cost[hyp.index(hs), refs.index(rs)] = -n
        rows, cols = linear_sum_assignment(cost)
        return {hyp[i]: refs[j] for i, j in zip(rows, cols, strict=True)}
    except Exception:
        # Greedy fallback: each hypothesis speaker takes its most frequent partner.
        out: dict[str, str] = {}
        for hs in hyp:
            best = max(refs, key=lambda rs: confusion.get((hs, rs), 0))
            out[hs] = best
        return out


def _in_silence(t: float, ref_times: list[tuple[float, float]], pad: float = SILENCE_PAD_S) -> bool:
    import bisect
    starts = [s for s, _ in ref_times]
    i = bisect.bisect_right(starts, t)
    for k in (i - 1, i):
        if 0 <= k < len(ref_times):
            s, e = ref_times[k]
            if s - pad <= t <= e + pad:
                return False
    return True


def _der(ref_segments: list[dict], hyp_segments) -> float | None:
    try:
        from pyannote.core import Annotation
        from pyannote.core import Segment as PSeg
        from pyannote.metrics.diarization import DiarizationErrorRate
    except Exception:
        return None
    ref = Annotation()
    for s in ref_segments:
        if s["end"] > s["start"]:
            ref[PSeg(s["start"], s["end"])] = s["speaker"]
    hyp = Annotation()
    for s in hyp_segments:
        if s.end > s.start:
            hyp[PSeg(s.start, s.end)] = s.speaker or "?"
    value = DiarizationErrorRate(collar=0.25)(ref, hyp, detailed=False)
    return round(float(value), 4)  # type: ignore[arg-type]


# ---------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="kind", required=True)
    a = sub.add_parser("ami")
    a.add_argument("--nxt", type=Path, required=True)
    a.add_argument("--meeting", required=True)
    a.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("earnings")
    e.add_argument("--nlp", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    ref = (ami_reference(args.nxt, args.meeting) if args.kind == "ami"
           else earnings_reference(args.nlp))
    args.out.write_text(json.dumps(ref), encoding="utf-8")
    spk = Counter(w["speaker"] for w in ref["words"])
    print(f"{ref['source']}: {len(ref['words'])} words, {len(ref['segments'])} segments, "
          f"speakers {dict(spk)} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
