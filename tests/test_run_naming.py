# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""A run that names its transcript and keeps the audio.

The name typed in "Go with options…" is the name the files get: no
"_cleaned_" suffix, no renaming after the other speaker. The audio copy sits
beside it with the same name, and a failed copy never fails a job whose
transcript is already on disk.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from transcriber_studio import filename_builder as fb
from transcriber_studio.config import Settings
from transcriber_studio.jobs import JobRunner
from transcriber_studio.models import Recording, Segment, Source, TranscriptResult


def _result(speakers=("Alex", "Sam Chen"), source=Source.PLAUD) -> TranscriptResult:
    rec = Recording(source=source, id="abc", name="Weekly sync", date="2026-10-07",
                    local_path="a.wav" if source == Source.LOCAL else None)
    segs = [Segment(start=i, end=i + 1, text=f"line {i}", speaker=s) for i, s in enumerate(speakers)]
    return TranscriptResult(recording=rec, segments=segs, speakers=list(speakers), model="m")


def _runner(tmp: str, **over) -> JobRunner:
    r = JobRunner.__new__(JobRunner)
    r.s = Settings(output_dir=tmp, formats=["txt"], owner_names="Alex", **over)
    return r


def test_strip_output_extension_drops_only_the_apps_own():
    assert fb.strip_output_extension("Board call.txt") == "Board call"
    assert fb.strip_output_extension("Board call.MD") == "Board call"
    assert fb.strip_output_extension("v2.1 planning") == "v2.1 planning"
    assert fb.strip_output_extension(".txt") == ".txt"
    assert fb.strip_output_extension("") == ""


def test_a_typed_name_beats_the_person_rule_and_the_cleanup_suffix():
    with tempfile.TemporaryDirectory() as tmp:
        plain = _runner(tmp)
        assert plain.output_stem(_result()) == "Sam Chen-2026-10-07"
        named = _runner(tmp, filename_override="Board call.txt")
        assert named.output_stem(_result()) == "Board call"
        assert named.output_stem(
            _result(), cleanup_provider="openai", cleanup_model="gpt"
        ) == "Board call"
        paths = named.write_outputs(_result())
        assert [Path(p).name for p in paths] == ["Board call.txt"]


def test_a_typed_template_is_filled_in_and_cleaned_of_bad_characters():
    with tempfile.TemporaryDirectory() as tmp:
        r = _runner(tmp, filename_override="{date} board: {name}")
        assert r.output_stem(_result()) == "2026-10-07 board Weekly sync"


def test_two_recordings_under_one_name_do_not_overwrite_each_other():
    with tempfile.TemporaryDirectory() as tmp:
        r = _runner(tmp, filename_override="Board call")
        first = r.write_outputs(_result(), 1)
        second = r.write_outputs(_result(), 2)
        assert Path(first[0]).name == "Board call.txt"
        assert Path(second[0]).name == "Board call (2).txt"


def test_the_audio_copy_takes_the_transcripts_name(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "cache.mp3"
        src.write_bytes(b"ID3x")
        keep = Path(tmp) / "keep"
        r = _runner(tmp, filename_override="Board call", save_audio_dir=str(keep))
        monkeypatch.setattr(JobRunner, "_source_audio", lambda self, *a, **k: str(src))
        logged = []
        saved = r._keep_audio(_result().recording, _result(), 1, logged.append)
        assert [Path(p).name for p in saved] == ["Board call.mp3"]
        assert (keep / "Board call.mp3").read_bytes() == b"ID3x"
        # Without a name, the recording's own date and name.
        r = _runner(tmp, save_audio_dir=str(keep))
        saved = r._keep_audio(_result().recording, _result(), 1, None)
        assert [Path(p).name for p in saved] == ["2026-10-07_Weekly sync.mp3"]


def test_local_files_and_unticked_runs_keep_nothing(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        calls = []
        monkeypatch.setattr(JobRunner, "_source_audio", lambda self, *a, **k: calls.append(1))
        r = _runner(tmp, save_audio_dir=tmp)
        assert r._keep_audio(_result(source=Source.LOCAL).recording, _result(), 1, None) == []
        r = _runner(tmp)
        assert r._keep_audio(_result().recording, _result(), 1, None) == []
        assert calls == []


def test_a_failed_audio_copy_is_logged_not_raised(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        def boom(self, *a, **k):
            raise OSError("disk full")

        monkeypatch.setattr(JobRunner, "_source_audio", boom)
        r = _runner(tmp, save_audio_dir=tmp)
        logged = []
        assert r._keep_audio(_result().recording, _result(), 1, logged.append) == []
        assert any("Could not save the audio" in line and "disk full" in line for line in logged)
