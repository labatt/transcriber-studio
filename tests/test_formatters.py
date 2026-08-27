# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Speaker turns render as one unbroken 'Name: what they said' line."""

from __future__ import annotations

from transcriber_studio import formatters
from transcriber_studio.config import Settings
from transcriber_studio.models import Recording, Segment, Source, TranscriptResult


def _result(segments: list[Segment]) -> TranscriptResult:
    rec = Recording(source=Source.LOCAL, id="x.wav", name="Standup", date="2026-08-19")
    speakers = list(dict.fromkeys(s.speaker for s in segments if s.speaker))
    return TranscriptResult(
        recording=rec,
        segments=segments,
        language="en",
        model="whisper",
        speakers=speakers,
    )


def _opts(**over) -> Settings:
    s = Settings()
    s.newline = "lf"
    s.include_timestamps = False
    s.include_speakers = True
    s.line_mode = "segment"
    for k, v in over.items():
        setattr(s, k, v)
    return s


CONVERSATION = [
    Segment(0.0, 2.0, "We pushed the release Tuesday.", "Greg"),
    Segment(2.0, 5.0, "It held up fine under load.", "Greg"),
    Segment(5.0, 7.0, "No rollbacks.", "Greg"),
    Segment(7.0, 9.0, "Good.", "Sarah"),
    Segment(9.0, 12.0, "Did the migration finish?", "Sarah"),
    Segment(12.0, 15.0, "It did, about forty minutes.", "Greg"),
]


def test_txt_gives_one_line_per_speaker_turn():
    text = formatters.render(_result(CONVERSATION), "txt", _opts())
    assert text == (
        "Greg: We pushed the release Tuesday. It held up fine under load. No rollbacks.\n"
        "Sarah: Good. Did the migration finish?\n"
        "Greg: It did, about forty minutes.\n"
    )


def test_txt_never_breaks_between_name_and_speech():
    lines = formatters.render(_result(CONVERSATION), "txt", _opts()).splitlines()
    assert all(line.strip() for line in lines)  # no blank lines mid-transcript
    for line in lines:
        speaker, _, said = line.partition(": ")
        assert speaker in {"Greg", "Sarah"}
        assert said.strip()


def test_newlines_inside_a_segment_are_flattened():
    segs = [Segment(0.0, 3.0, "First part.\nSecond   part.\n", "Greg")]
    text = formatters.render(_result(segs), "txt", _opts())
    assert text == "Greg: First part. Second part.\n"


def test_timestamp_marks_the_start_of_the_turn():
    text = formatters.render(_result(CONVERSATION), "txt", _opts(include_timestamps=True))
    assert text.splitlines()[0].startswith("[00:00] Greg: We pushed")
    assert text.splitlines()[1].startswith("[00:07] Sarah: Good.")


def test_undiarized_transcript_keeps_one_line_per_segment():
    segs = [
        Segment(0.0, 2.0, "First sentence.", None),
        Segment(2.0, 4.0, "Second sentence.", None),
    ]
    text = formatters.render(_result(segs), "txt", _opts())
    assert text == "First sentence.\nSecond sentence.\n"


def test_markdown_keeps_the_name_on_the_same_line_as_the_speech():
    text = formatters.render(_result(CONVERSATION), "md", _opts())
    assert (
        "**Greg:** We pushed the release Tuesday. It held up fine under load. No rollbacks."
        in text
    )
    assert "**Sarah:** Good. Did the migration finish?" in text
    assert "**Greg**\n" not in text  # name never sits alone on its own line


def test_subtitles_stay_one_cue_per_segment():
    srt = formatters.render(_result(CONVERSATION), "srt", _opts())
    assert srt.count(" --> ") == len(CONVERSATION)
    assert "Greg: We pushed the release Tuesday." in srt
