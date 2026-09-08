# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Speaker rosters belong to one recording. Two ways they leaked, both seen
live on 2026-09-08:

* a shared glossary carrying a roster ("Speaker 2" = Greg Jackson) was handed
  to cleanup as authoritative for every recording using that glossary, so Greg
  appeared in meetings he was never in;
* a recording's own saved roster, extracted when it had two speakers, was
  reused after its speakers were re-detected as six, so the new "Speaker 1"
  and "Speaker 2" inherited the old names and a seven-speaker transcript came
  out with fourteen labels.
"""

from __future__ import annotations

from transcriber_studio import glossary, glossary_store
from transcriber_studio.config import Settings
from transcriber_studio.glossary import roster_is_stale
from transcriber_studio.models import Recording, Segment, Source, TranscriptResult

from .support import isolated_glossary_dir

GREG = {"label": "Speaker 2", "name": "Greg Jackson", "role": "Executive"}
CHRIS = {"label": "Speaker 1", "name": "Chris", "role": "incoming CTO"}
UNKNOWN = {"label": "Unknown", "name": "", "role": "Unattributed short interjections"}


def _result(speakers, name="Team call"):
    return TranscriptResult(
        recording=Recording(source=Source.LOCAL, id="r1", name=name, local_path="a.mp3"),
        segments=[Segment(0, 1, "hi", speaker=speakers[0] if speakers else None)],
        speakers=list(speakers),
    )


# ---- the shared glossary never lends its roster -----------------------------------
def test_cleanup_gets_only_this_recordings_roster_from_a_shared_glossary():
    with isolated_glossary_dir():
        shared = glossary_store.create("team", speakers=[CHRIS, GREG], terms=[])
        own = {"speakers": [{"label": "Speaker 1", "name": "Dana", "role": ""}], "terms": []}
        lines = []
        merged = glossary.contribute_to_shared(shared, own, _result(["Speaker 1"]), log_cb=lines.append)
    assert [s["name"] for s in merged["speakers"]] == ["Dana"]
    assert all("Greg" not in s.get("name", "") for s in merged["speakers"])
    assert any("1 from this recording" in line and "left out" in line for line in lines)


def test_a_curated_shared_row_still_joins_but_label_keyed_and_nameless_rows_do_not():
    """A person who typed "Host = Dana Reyes" into the shared glossary meant it
    for every recording. "Speaker 2 = Greg" and "Unknown" did not."""
    rows = [
        {"label": "Host", "name": "Dana Reyes", "role": "AE"},
        GREG, CHRIS, UNKNOWN,
        {"label": "Guest", "name": "", "role": "unnamed"},
    ]
    assert [r["name"] for r in glossary.curated_speakers(rows)] == ["Dana Reyes"]


def test_the_shared_glossary_keeps_its_rows_on_disk_it_just_does_not_lend_them():
    with isolated_glossary_dir():
        shared = glossary_store.create("team", speakers=[CHRIS, GREG], terms=[])
        glossary.contribute_to_shared(shared, {"speakers": [], "terms": []}, _result(["Speaker 1"]))
        again = glossary_store.load(shared.id)
    assert again is not None and len(again.speakers) == 2


def test_with_extraction_off_the_shared_roster_is_still_withheld(monkeypatch):
    with isolated_glossary_dir():
        shared = glossary_store.create("team", speakers=[CHRIS, GREG],
                                       terms=[{"canonical": "HubSpot", "variants": []}])
        settings = Settings(glossary_enabled=False)
        monkeypatch.setattr(glossary, "_shared_glossary", lambda *a, **k: shared)
        payload = glossary.resolve_glossary(_result(["Speaker 1", "Speaker 2"]), settings,
                                            provider="google", model="m")
    assert payload["speakers"] == []
    assert [t["canonical"] for t in payload["terms"]] == ["HubSpot"]


# ---- a recording's own roster goes stale when speakers are re-detected ---------------
def test_the_same_labels_are_not_stale():
    assert not roster_is_stale([CHRIS, GREG, UNKNOWN], ["Speaker 1", "Speaker 2"])


def test_more_speakers_than_the_roster_knew_is_stale():
    """The live case: a two-speaker roster met a six-speaker transcript."""
    assert roster_is_stale([CHRIS, GREG], [f"Speaker {i}" for i in range(1, 7)])


def test_a_roster_label_the_transcript_no_longer_has_is_stale():
    assert roster_is_stale([CHRIS, GREG], ["Speaker 1"])


def test_a_roster_already_applied_by_renaming_is_not_stale():
    """After the rename dialog, "Speaker 2" is gone and "Greg Jackson" is there.
    The roster still describes this transcript."""
    assert not roster_is_stale([CHRIS, GREG], ["Chris", "Greg Jackson"])


def test_an_empty_roster_or_transcript_is_never_stale():
    assert not roster_is_stale([], ["Speaker 1", "Speaker 2"])
    assert not roster_is_stale([CHRIS], [])
    assert not roster_is_stale([UNKNOWN], ["Speaker 1"])


def test_a_stale_roster_is_re_extracted_and_the_terms_are_kept(monkeypatch, tmp_path):
    saved = {
        "speakers": [CHRIS, GREG],
        "terms": [{"canonical": "HubSpot", "variants": ["hub spot"]}],
    }
    path = tmp_path / "rec.glossary.json"
    glossary.save_glossary(path, saved)
    monkeypatch.setattr(glossary, "glossary_path", lambda settings, result: path)
    fresh = {
        "speakers": [{"label": f"Speaker {i}", "name": "", "role": ""} for i in range(1, 7)],
        "terms": [{"canonical": "BOIP", "variants": []}],
    }
    calls = []
    monkeypatch.setattr(glossary, "extract_glossary",
                        lambda *a, **k: calls.append(1) or dict(fresh))
    lines = []

    out = glossary._recording_glossary(
        _result([f"Speaker {i}" for i in range(1, 7)]), Settings(),
        provider="google", model="m", log_cb=lines.append,
    )

    assert calls == [1], "the roster had to be redone"
    assert len(out["speakers"]) == 6
    assert {t["canonical"] for t in out["terms"]} == {"HubSpot", "BOIP"}
    assert any("re-detected" in line for line in lines)
    assert {t["canonical"] for t in glossary.load_glossary(path)["terms"]} == {"HubSpot", "BOIP"}


def test_a_roster_that_still_fits_is_loaded_without_a_model_call(monkeypatch, tmp_path):
    path = tmp_path / "rec.glossary.json"
    glossary.save_glossary(path, {"speakers": [CHRIS, GREG], "terms": []})
    monkeypatch.setattr(glossary, "glossary_path", lambda settings, result: path)
    monkeypatch.setattr(glossary, "extract_glossary",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no call")))
    out = glossary._recording_glossary(
        _result(["Speaker 1", "Speaker 2"]), Settings(), provider="google", model="m",
    )
    assert [s["name"] for s in out["speakers"]] == ["Chris", "Greg Jackson"]
