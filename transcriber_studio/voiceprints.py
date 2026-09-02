# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Putting real names on diarized speakers, from voices enrolled once.

Diarization says how many people spoke and which stretches belong together. It
cannot say who they are: every recording comes back as Speaker 1, Speaker 2,
and someone has to rename them by hand, again, every time.

The missing piece is already computed. pyannote's pipeline clusters speakers by
comparing embeddings, and hands back the centroid it clustered around — one
vector per speaker, pooled over everything they said. Comparing that against a
vector kept from a previous recording is the whole of speaker identification.

So there is no second model here and nothing extra to download. What this module
adds is the bookkeeping: where the vectors live, and when a match is good enough
to put a name on.

**Matching is deliberately reluctant.** The set is open — most recordings
contain someone who was never enrolled — and the two mistakes are not equal. An
unrecognised speaker costs one edit in the rename dialog. A *wrongly* named one
is taken as authoritative by the glossary and by AI Cleanup, which will then
rewrite the transcript around a person who was never in the room. So a match
needs both a good score and a clear win over the runner-up, and anything short
of that stays Speaker N.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import APP_DIR

PROFILE_DIR = APP_DIR / "voiceprints"

#: Which embedding these vectors came from. Comparing across models is
#: meaningless, so a change here makes old profiles refuse to match rather than
#: match badly. It is the embedding bundled with the diarization pipeline, so
#: there is no separate model, download, or HuggingFace gate.
EMBEDDING_ID = "pyannote/speaker-diarization-community-1#embedding"

#: Cosine similarity a cluster must reach before it can be named at all.
#: pyannote clusters *within* one recording at about 0.295, but that is two
#: ten-second chunks on one microphone. These vectors are pooled over minutes
#: and compared across recordings, months and microphones, so the same-speaker
#: scores sit far higher while a stranger's barely move. 0.55 sits in the gap.
MATCH_THRESHOLD = 0.55

#: How far ahead of the runner-up the winner has to be. Two people who both
#: score well against one cluster means the cluster resembles a voice *type*,
#: not a person, and neither answer is worth trusting.
MATCH_MARGIN = 0.10

#: A cluster with less speech than this is one or two noisy chunks; its centroid
#: is not a reliable description of anybody.
MIN_SPEECH_SECONDS = 15.0

#: Enrolling is a promise about every future recording, so it asks for more.
MIN_ENROLL_SECONDS = 30.0
COMFORTABLE_ENROLL_SECONDS = 60.0


class VoiceprintError(RuntimeError):
    pass


@dataclass
class Voiceprint:
    """One captured vector for a person, and where it came from."""

    vector: list[float]
    seconds: float = 0.0
    source: str = ""
    created: float = field(default_factory=time.time)


@dataclass
class SpeakerProfile:
    """Everything known about one enrolled person."""

    name: str
    prints: list[Voiceprint] = field(default_factory=list)
    embedding_id: str = EMBEDDING_ID
    dim: int = 0

    @property
    def total_seconds(self) -> float:
        return sum(p.seconds for p in self.prints)

    def usable_with(self, dim: int) -> bool:
        """False when these vectors cannot be compared with the ones in hand.

        A model change makes old vectors meaningless rather than merely stale,
        and a silent mismatch would produce confident nonsense.
        """
        return self.embedding_id == EMBEDDING_ID and bool(self.prints) and self.dim == dim


# ---- vector maths ----------------------------------------------------
def normalize(vector) -> np.ndarray:
    """Unit-length copy. A zero vector stays zero rather than becoming NaN."""
    arr = np.asarray(vector, dtype=np.float64).ravel()
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0 or not np.isfinite(norm):
        return np.zeros_like(arr)
    return arr / norm


def is_usable(vector) -> bool:
    """pyannote pads its centroid array with zero rows when it has fewer
    clusters than labels. Those rows describe nobody."""
    arr = np.asarray(vector, dtype=np.float64).ravel()
    return bool(arr.size) and bool(np.isfinite(arr).all()) and float(np.linalg.norm(arr)) > 0.0


def similarity(a, b) -> float:
    """Cosine similarity, in [-1, 1]. Zero vectors score 0 against anything."""
    left, right = normalize(a), normalize(b)
    if not left.size or left.size != right.size:
        return 0.0
    return float(np.clip(np.dot(left, right), -1.0, 1.0))


# ---- storage ---------------------------------------------------------
def _slug(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return cleaned or "speaker"


def _profile_path(name: str) -> Path:
    return PROFILE_DIR / f"{_slug(name)}.json"


def _to_dict(profile: SpeakerProfile) -> dict:
    return {
        "name": profile.name,
        "embedding_id": profile.embedding_id,
        "dim": profile.dim,
        "prints": [
            {
                "vector": [float(x) for x in p.vector],
                "seconds": float(p.seconds),
                "source": p.source,
                "created": float(p.created),
            }
            for p in profile.prints
        ],
    }


def _from_dict(data: dict) -> SpeakerProfile | None:
    name = str(data.get("name") or "").strip()
    if not name:
        return None
    prints: list[Voiceprint] = []
    for entry in data.get("prints") or []:
        vector = entry.get("vector")
        if not isinstance(vector, list) or not vector:
            continue
        prints.append(
            Voiceprint(
                vector=[float(x) for x in vector],
                seconds=float(entry.get("seconds") or 0.0),
                source=str(entry.get("source") or ""),
                created=float(entry.get("created") or 0.0),
            )
        )
    return SpeakerProfile(
        name=name,
        prints=prints,
        embedding_id=str(data.get("embedding_id") or ""),
        dim=int(data.get("dim") or (len(prints[0].vector) if prints else 0)),
    )


def load_profiles() -> list[SpeakerProfile]:
    """Every enrolled speaker. An unreadable file is skipped, not fatal."""
    if not PROFILE_DIR.exists():
        return []
    profiles: list[SpeakerProfile] = []
    for path in sorted(PROFILE_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        profile = _from_dict(data)
        if profile is not None:
            profiles.append(profile)
    return profiles


def save_profile(profile: SpeakerProfile) -> Path:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    path = _profile_path(profile.name)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(_to_dict(profile), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp.replace(path)      # never leave a half-written profile to be read
    return path


def get_profile(name: str) -> SpeakerProfile | None:
    path = _profile_path(name)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return _from_dict(data) if isinstance(data, dict) else None


def delete_profile(name: str) -> bool:
    path = _profile_path(name)
    try:
        path.unlink()
        return True
    except OSError:
        return False


def enroll(name: str, vector, *, seconds: float, source: str = "") -> SpeakerProfile:
    """Remember a voice, adding to the person if they are already known.

    Vectors are appended rather than averaged. The same person over a lapel mic
    and over a conference speaker produces genuinely different vectors, and
    averaging them lands between the two, resembling neither. Kept apart, each
    recording is matched by whichever capture it actually resembles.
    """
    clean = (name or "").strip()
    if not clean:
        raise VoiceprintError("Give the speaker a name.")
    if not is_usable(vector):
        raise VoiceprintError(
            "There is not enough of this speaker's voice to recognise them later."
        )
    if seconds < MIN_ENROLL_SECONDS:
        raise VoiceprintError(
            f"Enrolling needs about {MIN_ENROLL_SECONDS:.0f} seconds of one person "
            f"speaking; this stretch has {seconds:.0f}."
        )
    arr = np.asarray(vector, dtype=np.float64).ravel()
    profile = get_profile(clean) or SpeakerProfile(name=clean, dim=int(arr.size))
    if profile.prints and profile.dim != int(arr.size):
        raise VoiceprintError(
            f"“{clean}” was enrolled with a different speaker model. Delete the "
            "existing voiceprint and enrol again."
        )
    profile.name = clean            # keep the capitalisation last supplied
    profile.embedding_id = EMBEDDING_ID
    profile.dim = int(arr.size)
    profile.prints.append(
        Voiceprint(vector=[float(x) for x in arr], seconds=float(seconds), source=source)
    )
    save_profile(profile)
    return profile


# ---- matching --------------------------------------------------------
@dataclass
class Match:
    """One cluster's best answer, named or not."""

    label: str                  # the raw diarization label, e.g. SPEAKER_00
    name: str = ""              # "" when nothing was confident enough
    score: float = 0.0
    runner_up: float = 0.0
    seconds: float = 0.0
    reason: str = ""            # why it was not named, for the log

    @property
    def named(self) -> bool:
        return bool(self.name)


def _score_matrix(
    vectors: dict[str, np.ndarray], profiles: list[SpeakerProfile]
) -> tuple[list[str], np.ndarray]:
    """Best similarity of each cluster to each person.

    Max over a person's vectors, never the mean: their captures are meant to
    differ, so a poor match on the conference-room recording should not drag
    down a good match on the lapel one.
    """
    labels = list(vectors)
    scores = np.zeros((len(labels), len(profiles)), dtype=np.float64)
    for i, label in enumerate(labels):
        centroid = vectors[label]
        for j, profile in enumerate(profiles):
            scores[i, j] = max(
                (similarity(centroid, p.vector) for p in profile.prints), default=0.0
            )
    return labels, scores


def _assign(scores: np.ndarray) -> list[int]:
    """Which person each cluster is provisionally paired with.

    One person cannot be two speakers in the same conversation, so this is an
    assignment problem rather than a per-cluster argmax: picking greedily lets
    a strong match steal the name a slightly stronger one needed.
    """
    if not scores.size:
        return []
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-scores)
        chosen = [-1] * scores.shape[0]
        for row, col in zip(rows, cols, strict=True):
            chosen[row] = int(col)
        return chosen
    except ImportError:
        # Same idea, greedily: strongest pair first, then strike out that row
        # and column. Identical in every case with a clear winner.
        chosen = [-1] * scores.shape[0]
        remaining = scores.copy()
        used_cols: set[int] = set()
        for _ in range(min(remaining.shape)):
            flat = int(np.argmax(remaining))
            row, col = divmod(flat, remaining.shape[1])
            if remaining[row, col] <= -np.inf:
                break
            chosen[row] = int(col)
            used_cols.add(int(col))
            remaining[row, :] = -np.inf
            remaining[:, col] = -np.inf
        return chosen


def identify(
    vectors: dict[str, np.ndarray],
    seconds: dict[str, float] | None = None,
    profiles: list[SpeakerProfile] | None = None,
) -> list[Match]:
    """Name what can be named, and say why for everything else."""
    seconds = seconds or {}
    if profiles is None:
        profiles = load_profiles()

    usable_dim = next((np.asarray(v).ravel().size for v in vectors.values()), 0)
    profiles = [p for p in profiles if p.usable_with(usable_dim)]

    matches = [
        Match(label=label, seconds=float(seconds.get(label, 0.0)))
        for label in vectors
    ]
    if not profiles:
        for match in matches:
            match.reason = "no voices enrolled"
        return matches

    considered = {
        label: normalize(vector)
        for label, vector in vectors.items()
        if is_usable(vector) and seconds.get(label, 0.0) >= MIN_SPEECH_SECONDS
    }
    by_label = {m.label: m for m in matches}
    for match in matches:
        if match.label not in considered:
            match.reason = (
                "not enough speech to recognise"
                if is_usable(vectors[match.label])
                else "no usable voice data"
            )

    if not considered:
        return matches

    labels, scores = _score_matrix(considered, profiles)
    chosen = _assign(scores)
    for i, label in enumerate(labels):
        match = by_label[label]
        row = scores[i]
        column = chosen[i]
        if column < 0:
            match.reason = "more speakers than enrolled voices"
            continue
        match.score = float(row[column])
        others = np.delete(row, column)
        match.runner_up = float(others.max()) if others.size else 0.0
        if match.score < MATCH_THRESHOLD:
            match.reason = f"closest match only {match.score:.2f}"
            continue
        if match.score - match.runner_up < MATCH_MARGIN:
            # Two enrolled people fit about equally well, so the cluster
            # resembles a kind of voice rather than one person.
            match.reason = (
                f"too close to call ({match.score:.2f} vs {match.runner_up:.2f})"
            )
            continue
        match.name = profiles[column].name
    return matches


def describe(matches: list[Match]) -> list[str]:
    """Lines for the job log: what was recognised, and what was not."""
    named = [m for m in matches if m.named]
    lines: list[str] = []
    for match in named:
        lines.append(
            f"Recognised {match.label} as {match.name} "
            f"({match.score:.2f}, next best {match.runner_up:.2f})."
        )
    unnamed = [m for m in matches if not m.named]
    if unnamed and named:
        lines.append(
            f"{len(unnamed)} other speaker(s) left unnamed: "
            + ", ".join(f"{m.label} — {m.reason}" for m in unnamed)
        )
    return lines
