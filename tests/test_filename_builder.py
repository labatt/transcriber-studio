# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Once the other speaker is named, the export is called name-yyyy-mm-dd."""

from __future__ import annotations

from transcriber_studio import filename_builder as fb
from transcriber_studio.models import Recording, Segment, Source, TranscriptResult

#: The owner as they would configure themselves in Settings.
OWNER = "Alex Rivera, Alex R, Alex"


def _result(speakers: list[str], date: str = "2026-08-19") -> TranscriptResult:
    rec = Recording(source=Source.LOCAL, id="call.wav", name="Weekly call", date=date)
    segs = [Segment(float(i), float(i + 1), "Hello.", s) for i, s in enumerate(speakers)]
    return TranscriptResult(
        recording=rec, segments=segs, language="en", model="whisper", speakers=speakers
    )


def test_the_owner_is_recognised_in_every_spelling_they_configured():
    for label in ("Alex", "alex", "Alex R", "Alex R.", "Alex Rivera", "alex rivera"):
        assert fb.is_owner_speaker(label, OWNER), label


def test_someone_else_with_the_same_first_name_is_not_the_owner():
    for label in ("Alex Vance", "Alex Anderson", "Alex Lewis"):
        assert not fb.is_owner_speaker(label, OWNER), label


def test_with_no_owner_configured_nobody_is_the_owner():
    """A fresh install has no name set; it must not guess at one."""
    for label in ("Alex", "Alex Rivera", "Sam Chen"):
        assert not fb.is_owner_speaker(label, ""), label


def test_punctuation_and_spacing_in_a_surname_do_not_matter():
    owner = "Sam Ortiz-Cole"
    for label in ("Sam Ortiz-Cole", "Sam Ortiz Cole", "Sam ortizcole"):
        assert fb.is_owner_speaker(label, owner), label


def test_detector_labels_do_not_count_as_names():
    for label in ("SPEAKER_00", "Speaker 1", "Speaker A", "spk_02", "Unknown", ""):
        assert not fb.is_named_speaker(label), label


def test_roles_and_non_human_sources_do_not_name_the_file():
    for label in ("Agent", "Customer", "Host", "Demo Video", "Voicemail"):
        assert not fb.is_named_speaker(label), label


def test_stem_uses_the_other_persons_name_and_the_recording_date():
    result = _result(["Alex Rivera", "Sam Chen"])
    assert fb.person_stem(result, owner_names=OWNER) == "Sam Chen-2026-08-19"


def test_speaker_order_does_not_matter():
    result = _result(["Sam Chen", "Alex"])
    assert fb.person_stem(result, owner_names=OWNER) == "Sam Chen-2026-08-19"


def test_a_namesake_still_names_the_file():
    result = _result(["Alex R", "Alex Vance"])
    assert fb.person_stem(result, owner_names=OWNER) == "Alex Vance-2026-08-19"


def test_unnamed_speakers_leave_the_filename_template_alone():
    assert fb.person_stem(_result(["SPEAKER_00", "SPEAKER_01"]), owner_names=OWNER) is None
    assert fb.person_stem(_result(["Alex"]), owner_names=OWNER) is None


def test_missing_date_falls_back_to_the_bare_name():
    assert fb.person_stem(_result(["Alex", "Sam Chen"], date=""), owner_names=OWNER) == "Sam Chen"


def test_write_outputs_prefers_the_person_stem():
    import tempfile
    from pathlib import Path

    from transcriber_studio.config import Settings
    from transcriber_studio.jobs import JobRunner

    with tempfile.TemporaryDirectory() as tmp:
        s = Settings()
        s.output_dir = tmp
        s.formats = ["txt"]
        s.filename_template = "{date}_{name}"
        s.owner_names = OWNER      # the app has to know which speaker is you
        runner = JobRunner.__new__(JobRunner)  # skip PlaudClient/Transcriber setup
        runner.s = s
        paths = runner.write_outputs(_result(["Alex", "Sam Chen"]))
        assert [Path(p).name for p in paths] == ["Sam Chen-2026-08-19.txt"]


def test_without_an_owner_the_first_named_speaker_names_the_file():
    """A fresh install still produces a sensible name, just not a tailored one."""
    import tempfile
    from pathlib import Path

    from transcriber_studio.config import Settings
    from transcriber_studio.jobs import JobRunner

    with tempfile.TemporaryDirectory() as tmp:
        s = Settings()
        s.output_dir, s.formats, s.owner_names = tmp, ["txt"], ""
        runner = JobRunner.__new__(JobRunner)
        runner.s = s
        paths = runner.write_outputs(_result(["Alex", "Sam Chen"]))
        assert [Path(p).name for p in paths] == ["Alex-2026-08-19.txt"]
