# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Gemini transcription: the request it builds and the response it reads.

The shapes here were taken from the live API, not from the documentation —
which is wrong about `output_text` and silent about the mode restrictions.
"""

from __future__ import annotations

from transcriber_studio import stt_gemini as g
from transcriber_studio.transcriber import TranscribeOptions


def _response(words, text):
    return {"steps": [{"type": "text", "content": [
        {"type": "text", "text": text, "annotations": words}
    ]}]}


def _word(text, start, end, speaker, start_index, end_index):
    return {"type": "word_info", "text": text, "start_offset": start,
            "end_offset": end, "speaker": speaker,
            "start_index": start_index, "end_index": end_index}


# ---- the request -------------------------------------------------------


def test_verbatim_asks_for_speakers_and_word_timings():
    config = g.build_config(TranscribeOptions(gemini_mode="verbatim",
                                              diarization_enabled=True))

    assert config["mode"]["type"] == "verbatim"
    assert config["mode"]["timestamp_granularities"] == ["word"]
    assert config["mode"]["diarization_mode"] == "speaker"


def test_smart_asks_for_neither_because_the_api_rejects_both():
    """Not a preference: sending either parameter fails the whole request."""
    config = g.build_config(TranscribeOptions(gemini_mode="smart",
                                              diarization_enabled=True))

    assert config["mode"] == {"type": "smart"}


def test_diarization_off_still_gets_timestamps():
    config = g.build_config(TranscribeOptions(gemini_mode="verbatim",
                                              diarization_enabled=False))

    assert "diarization_mode" not in config["mode"]
    assert config["mode"]["timestamp_granularities"] == ["word"]


def test_an_unknown_mode_falls_back_rather_than_failing_the_job():
    assert g.build_config(TranscribeOptions(gemini_mode="fancy"))["mode"]["type"] == g.DEFAULT_MODE
    assert g.build_config(TranscribeOptions(gemini_mode=""))["mode"]["type"] == g.DEFAULT_MODE


def test_the_default_mode_is_the_one_that_produces_speakers_and_times():
    assert g.DEFAULT_MODE == "verbatim"
    assert "diarization_mode" in g.build_config(TranscribeOptions())["mode"]


def test_a_chosen_language_is_passed_but_auto_is_not():
    assert g.build_config(TranscribeOptions(language="es"))["language_codes"] == ["es"]
    assert "language_codes" not in g.build_config(TranscribeOptions(language="auto"))


def test_no_vocabulary_is_ever_sent():
    """The API refuses custom_vocabulary alongside speakers or timestamps."""
    config = g.build_config(TranscribeOptions(gemini_mode="verbatim"))

    assert "custom_vocabulary" not in config


# ---- the response ------------------------------------------------------


def test_offsets_parse_with_or_without_a_decimal_point():
    """The API returns both "0.200s" and "11s"."""
    assert g.parse_offset("0.200s") == 0.2
    assert g.parse_offset("11s") == 11.0
    assert g.parse_offset("2.5") == 2.5
    assert g.parse_offset(3) == 3.0
    assert g.parse_offset(None) == 0.0
    assert g.parse_offset("nonsense") == 0.0


def test_words_carry_their_timings_and_speaker():
    text = "Good morning."
    words = g.word_annotations(_response(
        [_word("Good", "0.100s", "0.200s", "spk:0", 0, 4),
         _word("morning.", "0.200s", "0.600s", "spk:0", 5, 13)],
        text,
    ))

    spoken = [w for w in words if w["type"] == "word"]
    assert [w["text"] for w in spoken] == ["Good", "morning."]
    assert spoken[0]["start"] == 0.1 and spoken[0]["end"] == 0.2
    assert spoken[1]["speaker_id"] == "spk:0"


def test_the_gap_between_words_is_recovered_from_the_text():
    """Gemini reports no spacing of its own; without this the transcript
    came back as onerunontogetherstring."""
    text = "Good morning. This is Dana."
    words = g.word_annotations(_response(
        [_word("Good", "0s", "0.2s", "spk:0", 0, 4),
         _word("morning.", "0.2s", "0.6s", "spk:0", 5, 13),
         _word("This", "1s", "1.2s", "spk:0", 14, 18)],
        text,
    ))

    assert "".join(w["text"] for w in words) == "Good morning. This"
    assert [w["type"] for w in words] == ["word", "spacing", "word", "spacing", "word"]


def test_non_word_annotations_are_ignored():
    words = g.word_annotations(_response(
        [{"type": "something_else", "text": "x"},
         _word("Hello", "0s", "1s", "spk:0", 0, 5)],
        "Hello",
    ))

    assert [w["text"] for w in words if w["type"] == "word"] == ["Hello"]


def test_prose_is_found_where_the_docs_say_output_text_would_be():
    """output_text comes back null; the text is under steps[].content[]."""
    response = _response([], "Good morning.")
    response["output_text"] = None

    assert g.plain_text(response) == "Good morning."


def test_an_empty_response_yields_nothing_rather_than_raising():
    assert g.word_annotations({}) == []
    assert g.plain_text({}) == ""


# ---- odds and ends -----------------------------------------------------


def test_mime_type_follows_the_file():
    assert g.mime_type_for("a.mp3") == "audio/mpeg"
    assert g.mime_type_for("a.wav") in ("audio/wav", "audio/x-wav")
    assert g.mime_type_for("a.unknown") == "audio/wav"


def test_every_mode_has_a_label_the_ui_can_show():
    assert set(g.MODE_LABELS) == set(g.MODES)
    assert "no speakers" in g.MODE_LABELS["smart"]


# ---- the documented length limits --------------------------------------


def test_word_timestamps_alone_lower_the_length_ceiling():
    """Google's wording is "diarization OR word-level timestamps", and verbatim
    mode always asks for timestamps — so the lower limit applies even with
    speaker detection off."""
    config = g.build_config(TranscribeOptions(gemini_mode="verbatim",
                                              diarization_enabled=False))

    assert "diarization_mode" not in config["mode"]
    assert g.length_ceiling(config) == g.PRACTICAL_MINUTES_WITH_FEATURES


def test_speakers_also_lower_it():
    config = g.build_config(TranscribeOptions(gemini_mode="verbatim",
                                              diarization_enabled=True))

    assert g.length_ceiling(config) == g.PRACTICAL_MINUTES_WITH_FEATURES


def test_the_ceiling_is_the_measured_one_not_the_documented_one():
    """Google documents 30 minutes and does not enforce it.

    Probed against gemini-3.5-transcribe, verbatim with diarization: 35, 46, 51
    and 54 minutes were accepted, 57 and 80 were refused with a bare
    "Invalid input received.". Warning at 30 cried wolf while the real wall
    arrived as an opaque 400 after the whole file had been uploaded.
    """
    assert g.PRACTICAL_MINUTES_WITH_FEATURES > g.MAX_MINUTES_WITH_FEATURES
    assert 50 <= g.PRACTICAL_MINUTES_WITH_FEATURES <= 56


def test_a_recording_over_the_ceiling_is_refused_before_it_is_uploaded():
    """Uploading 150 MB to be told "Invalid input received." helps nobody."""
    import pytest

    from transcriber_studio.models import Recording, Source

    rec = Recording(source=Source.LOCAL, id="long.mp3", name="long",
                    date="2026-08-28", local_path="long.mp3",
                    duration_seconds=80 * 60)
    opts = TranscribeOptions(gemini_api_key="k", gemini_mode="verbatim",
                             diarization_enabled=True)
    sent = []
    original, g.upload = g.upload, lambda *a, **k: sent.append(a) or "uri"
    try:
        with pytest.raises(g.GeminiError) as caught:
            g.transcribe(rec, "long.mp3", opts)
    finally:
        g.upload = original

    assert not sent, "it uploaded the file before finding out it was too long"
    message = str(caught.value)
    assert "80 minutes" in message
    assert "split" in message.lower()


def test_smart_mode_gets_the_full_hour():
    config = g.build_config(TranscribeOptions(gemini_mode="smart"))

    assert g.length_ceiling(config) == g.MAX_MINUTES_PLAIN == 60
