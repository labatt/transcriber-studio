# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared glossaries: the library itself, and jobs reading from / writing to one."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from tests.support import isolated_glossary_dir
from transcriber_studio import glossary, glossary_store
from transcriber_studio.config import Settings
from transcriber_studio.models import Recording, Source, TranscriptResult


def _settings(output_dir: str, **over) -> Settings:
    s = Settings()
    s.output_dir = output_dir
    s.glossary_enabled = True
    for k, v in over.items():
        setattr(s, k, v)
    return s


def _result(rec_id: str = "rec-1", name: str = "Acme kickoff") -> TranscriptResult:
    return TranscriptResult(
        Recording(Source.PLAUD, rec_id, name, date="2026-08-26"),
        language="en",
        model="whisper",
    )


def _write_own_glossary(settings: Settings, result: TranscriptResult, payload: dict) -> Path:
    """Stand in for the extraction pass: leave the file it would have written."""
    path = glossary.glossary_path(settings, result)
    glossary.save_glossary(path, payload)
    return path


# ---- the library ------------------------------------------------------


def test_create_list_and_delete_round_trip():
    with isolated_glossary_dir():
        created = glossary_store.create("Acme Account")
        assert glossary_store.load(created.id) is not None
        assert [g.name for g in glossary_store.list_glossaries()] == ["Acme Account"]
        assert glossary_store.delete(created.id) is True
        assert glossary_store.list_glossaries() == []
        assert glossary_store.load(created.id) is None


def test_same_name_twice_gets_distinct_ids():
    with isolated_glossary_dir():
        first = glossary_store.create("Acme Account")
        second = glossary_store.create("Acme Account")
        assert first.id != second.id
        assert len(glossary_store.list_glossaries()) == 2


def test_duplicate_copies_contents_not_sources():
    with isolated_glossary_dir():
        original = glossary_store.create(
            "Acme", terms=[{"canonical": "GrowthMark", "variants": [], "type": "product"}]
        )
        original.record_source("fingerprint", "Kickoff")
        glossary_store.save(original)

        copy = glossary_store.duplicate(original.id)
        assert copy is not None
        assert [t["canonical"] for t in copy.terms] == ["GrowthMark"]
        assert copy.sources == []


def test_import_accepts_a_per_recording_glossary_file():
    with isolated_glossary_dir(), tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "meeting.glossary.json"
        src.write_text(
            json.dumps(
                {
                    "speakers": [],
                    "terms": [{"canonical": "Scribe", "variants": ["scribes"], "type": "product"}],
                }
            ),
            encoding="utf-8",
        )
        imported = glossary_store.import_from(src, name="From meeting")
        assert imported.name == "From meeting"
        assert [t["canonical"] for t in imported.terms] == ["Scribe"]


# ---- jobs reading from and writing to a shared glossary ---------------


def test_job_writes_its_terms_back_and_reads_the_union():
    with isolated_glossary_dir(), tempfile.TemporaryDirectory() as out:
        shared = glossary_store.create(
            "Acme", terms=[{"canonical": "GrowthMark", "variants": ["growth mark"], "type": "product"}]
        )
        settings = _settings(out, glossary_shared_id=shared.id)
        result = _result()
        _write_own_glossary(
            settings,
            result,
            {
                "speakers": [],
                "terms": [{"canonical": "Scribe", "variants": ["scribed"], "type": "product"}],
            },
        )

        merged = glossary.resolve_glossary(
            result, settings, provider="anthropic", model="claude-opus-5"
        )

        assert sorted(t["canonical"] for t in merged["terms"]) == ["GrowthMark", "Scribe"]
        # …and the shared file on disk now carries both, for the next job.
        reloaded = glossary_store.load(shared.id)
        assert sorted(t["canonical"] for t in reloaded.terms) == ["GrowthMark", "Scribe"]
        assert [s["name"] for s in reloaded.sources] == ["Acme kickoff"]


def test_two_jobs_accumulate_into_one_glossary():
    with isolated_glossary_dir(), tempfile.TemporaryDirectory() as out:
        shared = glossary_store.create("Acme")
        settings = _settings(out, glossary_shared_id=shared.id)

        for rec_id, term in (("rec-1", "GrowthMark"), ("rec-2", "Scribe")):
            result = _result(rec_id, f"Call {rec_id}")
            _write_own_glossary(
                settings,
                result,
                {"speakers": [], "terms": [{"canonical": term, "variants": [], "type": "product"}]},
            )
            glossary.resolve_glossary(
                result, settings, provider="anthropic", model="claude-opus-5"
            )

        reloaded = glossary_store.load(shared.id)
        assert sorted(t["canonical"] for t in reloaded.terms) == ["GrowthMark", "Scribe"]
        assert len(reloaded.sources) == 2


def test_diarization_labels_never_enter_the_shared_roster():
    with isolated_glossary_dir(), tempfile.TemporaryDirectory() as out:
        shared = glossary_store.create("Acme")
        settings = _settings(out, glossary_shared_id=shared.id)
        result = _result()
        _write_own_glossary(
            settings,
            result,
            {
                "speakers": [
                    {
                        "label": "SPEAKER_00",
                        "name": "Gregory Jackson",
                        "role": "CTO",
                        "confidence": "high",
                        "raw_intro": "",
                    }
                ],
                "terms": [],
            },
        )

        merged = glossary.resolve_glossary(
            result, settings, provider="anthropic", model="claude-opus-5"
        )

        # This recording keeps its own label mapping…
        assert [s["label"] for s in merged["speakers"]] == ["SPEAKER_00"]
        reloaded = glossary_store.load(shared.id)
        assert reloaded.speakers == []
        # …while the name it resolved travels on as a person term.
        assert [(t["canonical"], t["type"]) for t in reloaded.terms] == [
            ("Gregory Jackson", "person")
        ]


def test_curated_shared_speakers_join_the_roster():
    with isolated_glossary_dir(), tempfile.TemporaryDirectory() as out:
        shared = glossary_store.create("Acme")
        shared.speakers = [
            {"label": "Host", "name": "Dana Reyes", "role": "AE", "confidence": "high", "raw_intro": ""}
        ]
        glossary_store.save(shared)
        settings = _settings(out, glossary_shared_id=shared.id)
        result = _result()
        _write_own_glossary(settings, result, {"speakers": [], "terms": []})

        merged = glossary.resolve_glossary(
            result, settings, provider="anthropic", model="claude-opus-5"
        )

        assert [s["name"] for s in merged["speakers"]] == ["Dana Reyes"]


def test_extraction_off_still_reads_the_shared_glossary():
    with isolated_glossary_dir(), tempfile.TemporaryDirectory() as out:
        shared = glossary_store.create(
            "Acme", terms=[{"canonical": "GrowthMark", "variants": [], "type": "product"}]
        )
        settings = _settings(out, glossary_enabled=False, glossary_shared_id=shared.id)

        merged = glossary.resolve_glossary(
            _result(), settings, provider="anthropic", model="claude-opus-5"
        )

        assert [t["canonical"] for t in merged["terms"]] == ["GrowthMark"]
        # Nothing was extracted, so nothing was written back.
        assert glossary_store.load(shared.id).sources == []


def test_a_deleted_shared_glossary_falls_back_to_the_recordings_own():
    with isolated_glossary_dir(), tempfile.TemporaryDirectory() as out:
        settings = _settings(out, glossary_shared_id="gone-for-good")
        result = _result()
        _write_own_glossary(
            settings,
            result,
            {"speakers": [], "terms": [{"canonical": "Scribe", "variants": [], "type": "product"}]},
        )
        logged: list[str] = []

        merged = glossary.resolve_glossary(
            result,
            settings,
            provider="anthropic",
            model="claude-opus-5",
            log_cb=logged.append,
        )

        assert [t["canonical"] for t in merged["terms"]] == ["Scribe"]
        assert any("gone-for-good" in line for line in logged)


def test_explicit_per_recording_choice_beats_the_app_default():
    with isolated_glossary_dir(), tempfile.TemporaryDirectory() as out:
        shared = glossary_store.create("Acme")
        settings = _settings(out, glossary_shared_id=shared.id)
        result = _result()
        _write_own_glossary(
            settings,
            result,
            {"speakers": [], "terms": [{"canonical": "Scribe", "variants": [], "type": "product"}]},
        )

        merged = glossary.resolve_glossary(
            result,
            settings,
            provider="anthropic",
            model="claude-opus-5",
            glossary_id="",     # this job opted out
        )

        assert [t["canonical"] for t in merged["terms"]] == ["Scribe"]
        assert glossary_store.load(shared.id).terms == []
