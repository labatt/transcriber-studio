# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Output filename template engine + token catalogue used by the builder UI."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .models import Recording, Source, TranscriptResult

INVALID_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# The person who is on every recording — the one holding the recorder — is
# whoever the user names in Settings. Anyone else with a real name is "the other
# person", and that is who the output file gets named after. With no names
# configured there is no owner, and the first named speaker wins.
# "Speaker 1", "SPEAKER_00", "Speaker A", "Unknown" — a label, not a name.
GENERIC_SPEAKER_RE = re.compile(
    r"^(speaker|spk|voice|participant|unknown|unidentified)[\s_\-]*[0-9a-z]{0,3}$",
    re.IGNORECASE,
)
# Roles and non-human sources the transcript may label but never name a file after.
NON_PERSON_LABELS = {
    "agent", "customer", "caller", "client", "host", "guest", "interviewer",
    "interviewee", "me", "user", "them", "other", "everyone", "group",
    "demo video", "video", "voicemail", "recording", "announcement", "narrator",
}


def _name_parts(label: str) -> list[str]:
    cleaned = re.sub(r"[.,]", " ", (label or "").lower())
    return [p for p in re.split(r"[\s_]+", cleaned) if p]


def _squash(value: str) -> str:
    """Compare names without caring about spacing, hyphens or punctuation."""
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def parse_owner_names(value: str) -> list[str]:
    """The owner's name spellings, from a comma- or newline-separated setting."""
    return [part.strip() for part in re.split(r"[,;\n]+", value or "") if part.strip()]


def owner_forms(owner_names) -> tuple[set[str], set[str]]:
    """Split configured names into (first names, surname forms).

    "Chris Labatt-Simon, Chris L, Chris" becomes first names {chris} and
    surnames {labattsimon, l, ""} — so every spelling the user actually gets
    labelled with is covered without listing each combination.
    """
    if isinstance(owner_names, str):
        owner_names = parse_owner_names(owner_names)
    firsts: set[str] = set()
    surnames: set[str] = set()
    for name in owner_names or ():
        parts = _name_parts(name)
        if not parts:
            continue
        firsts.add(parts[0])
        surnames.add(_squash(" ".join(parts[1:])))
    return firsts, surnames


def is_owner_speaker(label: str, owner_names=()) -> bool:
    """True for the recorder's owner — "Chris", "Chris L", "Chris Smith", …

    A different person who happens to share the first name is entered with their
    own surname, so they do not match unless that surname was configured too.
    With nothing configured, nobody is the owner.
    """
    firsts, surnames = owner_forms(owner_names)
    if not firsts:
        return False
    parts = _name_parts(label)
    if not parts or parts[0] not in firsts:
        return False
    return _squash(" ".join(parts[1:])) in surnames


def is_named_speaker(label: str) -> bool:
    """True once a speaker carries a real name instead of a role or a label."""
    label = (label or "").strip()
    if not label or GENERIC_SPEAKER_RE.match(label):
        return False
    return " ".join(_name_parts(label)) not in NON_PERSON_LABELS


def named_counterpart(speakers: list[str], owner_names=()) -> str | None:
    """The named person on the recording who is not the owner, if there is one."""
    for spk in speakers:
        if is_named_speaker(spk) and not is_owner_speaker(spk, owner_names):
            return spk.strip()
    return None


def person_stem(
    result: TranscriptResult, sanitize_names: bool = True, owner_names=()
) -> str | None:
    """`name-yyyy-mm-dd` once the other speaker has been named; else None."""
    name = named_counterpart(result.speakers, owner_names)
    if not name:
        return None
    date = (result.recording.date or "").strip()
    return sanitize(f"{name}-{date}" if date else name, sanitize_names)


@dataclass
class Token:
    key: str
    label: str
    description: str
    example: str


# The catalogue is the single source of truth for both the builder UI and docs.
TOKENS: list[Token] = [
    Token("{name}", "Name", "Recording name (sanitized)", "Strategy Meeting"),
    Token("{id}", "ID", "Plaud recording id / local stem", "36de218b…"),
    Token("{date}", "Date", "Recording date (YYYY-MM-DD)", "2026-05-29"),
    Token("{datetime}", "Date+Time", "Start datetime (YYYY-MM-DD_HHMMSS)", "2026-05-29_203717"),
    Token("{time}", "Time", "Start time (HHMMSS)", "203717"),
    Token("{source}", "Source", "'plaud' or 'local'", "plaud"),
    Token("{model}", "Model", "Whisper model used", "large-v3"),
    Token("{lang}", "Language", "Detected/selected language", "en"),
    Token("{speakers}", "Speakers", "Number of speakers detected", "2"),
    Token("{duration}", "Duration", "Recording duration", "16m59s"),
    Token("{index}", "Index", "1-based position in the batch (zero-padded)", "03"),
    Token("{orig}", "Original", "Original local filename stem", "interview_raw"),
]


def sanitize(value: str, enabled: bool = True) -> str:
    value = (value or "").strip()
    if not enabled:
        return value
    value = INVALID_CHARS_RE.sub("", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value or "untitled"


def build_values(result: TranscriptResult, index: int, sanitize_names: bool) -> dict[str, str]:
    rec: Recording = result.recording
    dt = rec.datetime or ""
    time_part = ""
    datetime_part = rec.date or ""
    if "T" in dt:
        d, t = dt.split("T", 1)
        time_part = t.replace(":", "")[:6]
        datetime_part = f"{d}_{time_part}"
    orig = ""
    if rec.source == Source.LOCAL and rec.local_path:
        orig = Path(rec.local_path).stem

    raw = {
        "{name}": rec.name,
        "{id}": rec.id if rec.source == Source.PLAUD else Path(rec.local_path or rec.id).stem,
        "{date}": rec.date,
        "{datetime}": datetime_part,
        "{time}": time_part,
        "{source}": rec.source.value,
        "{model}": result.model,
        "{lang}": result.language,
        "{speakers}": str(result.speaker_count),
        "{duration}": rec.duration,
        "{index}": f"{index:02d}",
        "{orig}": orig,
    }
    return {k: sanitize(str(v), sanitize_names) for k, v in raw.items()}


def render(template: str, values: dict[str, str], sanitize_names: bool = True) -> str:
    out = template
    for token, value in values.items():
        out = out.replace(token, value)
    # Strip any unknown {tokens} that were left behind.
    out = re.sub(r"\{[^}]*\}", "", out)
    return sanitize(out, sanitize_names) or "transcript"


def cleanup_stem(
    base_stem: str, provider: str, model: str, sanitize_names: bool = True
) -> str:
    """Build `{original}_cleaned_{provider}_{model}` for AI cleanup exports."""
    safe_provider = sanitize(provider.strip().replace(" ", "_"), sanitize_names)
    safe_model = sanitize(
        model.strip().replace("/", "_").replace(":", "_").replace(" ", "_"),
        sanitize_names,
    )
    return sanitize(f"{base_stem}_cleaned_{safe_provider}_{safe_model}", sanitize_names)


def sample_values() -> dict[str, str]:
    """Values used for the live preview in the builder dialog."""
    return {t.key: t.example for t in TOKENS}


def unique_path(directory: str, stem: str, ext: str, overwrite: bool) -> Path:
    base = Path(directory) / f"{stem}.{ext.lstrip('.')}"
    if overwrite or not base.exists():
        return base
    i = 2
    while True:
        cand = Path(directory) / f"{stem} ({i}).{ext.lstrip('.')}"
        if not cand.exists():
            return cand
        i += 1
