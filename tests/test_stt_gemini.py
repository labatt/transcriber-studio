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


def test_the_ceiling_is_what_comes_back_sane_not_what_is_accepted():
    """The API accepts up to about 54 minutes with speakers and word timings
    (35, 46, 51 and 54 were accepted; 57 and 80 refused). Scored against
    reference transcripts, though, only requests of 30 minutes or less came
    back with sane timings: a 33-minute part and a 35.7-minute request had
    thousands of word timings out of order and a sentence looped 60 times.
    Accepted is not transcribed. The documented 30 minutes is the ceiling."""
    assert g.ACCEPTED_MINUTES_WITH_FEATURES > g.MAX_MINUTES_WITH_FEATURES
    assert g.PRACTICAL_MINUTES_WITH_FEATURES == g.MAX_MINUTES_WITH_FEATURES == 30


def test_no_part_is_longer_than_the_ceiling_once_the_lead_in_is_added():
    """A 66-minute call was cut 30 / 33 / 9: the middle part ran 30 minutes
    seam to seam plus the 3-minute lead-in, past the ceiling, and that is the
    part whose transcript fell apart."""
    from transcriber_studio.audio_utils import part_spans

    spans = part_spans(66.3 * 60, g.UPLOAD_CHUNK_MINUTES * 60, g.OVERLAP_SECONDS, [])

    assert all(end - start <= g.UPLOAD_CHUNK_MINUTES * 60 + 1e-6 for start, _, end in spans)
    assert spans[0][0] == 0.0
    assert all(seam - start == g.OVERLAP_SECONDS for start, seam, _ in spans[1:])
    # The parts still tile the recording: each seam is where the last one ended.
    assert all(spans[i][2] == spans[i + 1][1] for i in range(len(spans) - 1))
    assert spans[-1][2] == 66.3 * 60


def test_without_a_lead_in_the_target_is_used_as_is():
    from transcriber_studio.audio_utils import part_spans

    assert part_spans(5400, 1800, 0.0, []) == [(0.0, 0.0, 1800), (1800, 1800, 3600), (3600, 3600, 5400)]


def test_a_recording_over_the_ceiling_goes_through_in_parts():
    """Too long for one request is not too long for the engine."""
    from transcriber_studio.models import Recording, Source

    rec = Recording(source=Source.LOCAL, id="long.mp3", name="long",
                    date="2026-08-28", local_path="long.mp3",
                    duration_seconds=80 * 60)
    opts = TranscribeOptions(gemini_api_key="k", gemini_mode="verbatim",
                             diarization_enabled=True)

    # Two parts, the second starting early to overlap the first.
    parts = [("a.mp3", 0.0, 0.0), ("b.mp3", 1620.0, 1800.0)]
    responses = [
        # Part one runs to the seam at 1800s.
        {"steps": [{"content": [{"text": "alpha two", "annotations": [
            {"type": "word_info", "text": "alpha", "start_offset": "1700s",
             "end_offset": "1701s", "speaker": "1", "start_index": 0, "end_index": 5},
            {"type": "word_info", "text": "two", "start_offset": "1795s",
             "end_offset": "1796s", "speaker": "1", "start_index": 6, "end_index": 9},
        ]}]}]},
        # Part two starts at 1620s, so its 175s is 1795s — the repeated word,
        # which it numbers "9" where part one said "1".
        {"steps": [{"content": [{"text": "two three", "annotations": [
            {"type": "word_info", "text": "two", "start_offset": "175s",
             "end_offset": "176s", "speaker": "9", "start_index": 0, "end_index": 3},
            {"type": "word_info", "text": "three", "start_offset": "185s",
             "end_offset": "186s", "speaker": "9", "start_index": 4, "end_index": 9},
        ]}]}]},
    ]
    uploaded = []
    saved = (g.audio_utils.split_for_upload, g.upload, g._post_interaction)
    g.audio_utils.split_for_upload = lambda *a, **k: parts
    g.upload = lambda path, *a, **k: uploaded.append(path) or f"uri:{path}"
    g._post_interaction = lambda *a, **k: responses[len(uploaded) - 1]
    try:
        result = g.transcribe(rec, "long.mp3", opts)
    finally:
        g.audio_utils.split_for_upload, g.upload, g._post_interaction = saved

    assert uploaded == ["a.mp3", "b.mp3"], "it did not send both parts"
    words = " ".join(s.text for s in result.segments)
    assert "alpha" in words and "three" in words
    # 185s into a part starting at 1620s is 1805s on the real timeline.
    assert result.segments[-1].start == 1805.0
    # "two" was transcribed by both parts; only the first part's copy survives.
    assert words.count("two") == 1
    # Part two called the speaker "9" and part one called them "1"; the overlap
    # says they are the same person, so the transcript has one speaker.
    assert result.speakers == ["Speaker 1"]


def test_the_bridge_matches_speakers_by_the_repeated_speech():
    established = [
        {"type": "word", "text": "a", "start": 1700.0, "end": 1701.0, "speaker_id": "p1:1"},
        {"type": "word", "text": "b", "start": 1750.0, "end": 1751.0, "speaker_id": "p1:2"},
    ]
    incoming = [
        {"type": "word", "text": "a", "start": 1700.2, "end": 1701.2, "speaker_id": "p2:7"},
        {"type": "word", "text": "b", "start": 1750.1, "end": 1751.1, "speaker_id": "p2:4"},
        {"type": "word", "text": "c", "start": 1900.0, "end": 1901.0, "speaker_id": "p2:7"},
    ]
    bridge = g._speaker_bridge(established, incoming, seam=1800.0, overlap=180.0)

    assert bridge == {"p2:7": "p1:1", "p2:4": "p1:2"}


def test_the_bridge_declines_to_guess_when_nothing_lines_up():
    """A join where the timings do not correspond must not invent a match."""
    established = [
        {"type": "word", "text": "a", "start": 1700.0, "end": 1701.0, "speaker_id": "p1:1"},
    ]
    incoming = [
        {"type": "word", "text": "z", "start": 1790.0, "end": 1791.0, "speaker_id": "p2:7"},
    ]
    assert g._speaker_bridge(established, incoming, seam=1800.0, overlap=180.0) == {}


def test_the_bridge_takes_the_majority_when_a_word_or_two_disagree():
    established = [
        {"type": "word", "text": w, "start": 1700.0 + i, "end": 1700.5 + i,
         "speaker_id": "p1:1" if i != 2 else "p1:2"}
        for i, w in enumerate("abcd")
    ]
    incoming = [
        {"type": "word", "text": w, "start": 1700.05 + i, "end": 1700.55 + i,
         "speaker_id": "p2:5"}
        for i, w in enumerate("abcd")
    ]
    assert g._speaker_bridge(established, incoming, seam=1800.0, overlap=180.0) == {"p2:5": "p1:1"}


def test_split_points_cuts_on_the_clock_when_there_is_no_pause():
    from transcriber_studio.audio_utils import split_points

    assert split_points(5400, 1800, []) == [1800, 3600]


def test_split_points_prefers_a_nearby_pause():
    """A cut landing mid-word garbles a word at every seam."""
    from transcriber_studio.audio_utils import split_points

    assert split_points(3000, 1800, [1755.0]) == [1755.0]


def test_split_points_ignores_a_pause_that_is_nowhere_near():
    from transcriber_studio.audio_utils import split_points

    assert split_points(3000, 1800, [600.0]) == [1800]


def test_a_recording_under_the_target_is_not_split():
    from transcriber_studio.audio_utils import split_points

    assert split_points(1200, 1800, []) == []


def test_smart_mode_gets_the_full_hour():
    config = g.build_config(TranscribeOptions(gemini_mode="smart"))

    assert g.length_ceiling(config) == g.MAX_MINUTES_PLAIN == 60


def test_two_voices_cannot_both_claim_the_same_speaker():
    """Merging two people into one is how a speaker vanishes mid-transcript.

    Seen for real: the second half of a recording came back with one speaker
    where there were two, and the log reported a clean match.
    """
    established = [
        {"type": "word", "text": "a", "start": 1700.0, "end": 1701.0, "speaker_id": "p1:1"},
        {"type": "word", "text": "b", "start": 1702.0, "end": 1703.0, "speaker_id": "p1:1"},
        {"type": "word", "text": "c", "start": 1704.0, "end": 1705.0, "speaker_id": "p1:2"},
    ]
    # Both incoming voices look most like p1:1 on a naive nearest-word vote.
    incoming = [
        {"type": "word", "text": "a", "start": 1700.1, "end": 1701.1, "speaker_id": "p2:5"},
        {"type": "word", "text": "b", "start": 1702.1, "end": 1703.1, "speaker_id": "p2:5"},
        {"type": "word", "text": "b2", "start": 1702.2, "end": 1703.2, "speaker_id": "p2:6"},
    ]
    bridge = g._speaker_bridge(established, incoming, seam=1800.0, overlap=180.0)

    assert len(set(bridge.values())) == len(bridge), f"two voices merged: {bridge}"
    assert bridge.get("p2:5") == "p1:1"


# ---- timings and loops the live model produced ------------------------------
# Measured on public AMI and Earnings-22 audio: every Gemini run had a few
# words whose end preceded their start, one started 99,711 s into a 66-minute
# file, and one earnings call repeated an eight-word phrase 60 times in a row.


def _w(text, start, end, speaker="spk:0"):
    return {"type": "word", "text": text, "start": start, "end": end, "speaker_id": speaker}


def test_an_end_before_its_start_is_pulled_up_to_the_start():
    words = [_w("a", 10.0, 10.5), _w("b", 11.0, 4.0)]
    assert g.repair_timings(words) == 1
    assert words[1]["end"] == 11.0


def test_a_start_hours_past_the_end_of_the_file_is_clamped():
    words = [_w("a", 10.0, 10.5), _w("b", 99711.6, 99712.0)]
    assert g.repair_timings(words, duration=3980.0) == 1
    assert words[1]["start"] == 10.5 and words[1]["end"] <= 3980.0


def test_a_word_that_jumps_minutes_backwards_is_moved_after_its_predecessor():
    words = [_w("a", 1098.2, 1098.6), _w("b", 122.6, 123.0)]
    assert g.repair_timings(words) == 1
    assert words[1]["start"] == 1098.6 and words[1]["end"] == 1098.6


def test_a_small_overlap_is_left_alone_because_people_talk_over_each_other():
    words = [_w("a", 10.0, 11.0), _w("b", 10.6, 11.4, "spk:1")]
    assert g.repair_timings(words) == 0


def test_sane_timings_are_not_touched():
    words = [_w("a", 0.1, 0.4), _w("b", 0.5, 0.9), _w("c", 2.0, 2.3)]
    assert g.repair_timings(words, duration=60.0) == 0
    assert [(w["start"], w["end"]) for w in words] == [(0.1, 0.4), (0.5, 0.9), (2.0, 2.3)]


def test_a_phrase_repeated_in_a_loop_is_kept_once():
    phrase = ["it", "used", "to", "be", "that", "we", "talk", "about"]
    words = [_w("so", 0, 1)]
    for k in range(6):
        for j, t in enumerate(phrase):
            words.append(_w(t, 10 + k * 8 + j, 10 + k * 8 + j + 0.5))
    words.append(_w("anyway", 100, 101))

    kept, loops = g.collapse_loops(words)

    assert [w["text"] for w in kept] == ["so", *phrase, "anyway"]
    assert loops == [(8, 6)]


def test_short_repetitions_are_real_speech_and_survive():
    words = [_w(t, i, i + 0.4) for i, t in enumerate(
        ["no", "no", "no", "no", "thank", "you", "thank", "you", "thank", "you"]
    )]
    kept, loops = g.collapse_loops(words)
    assert len(kept) == len(words) and loops == []


def test_spacing_after_a_dropped_word_goes_with_it():
    phrase = ["one", "two", "three", "four", "five"]
    words = []
    for k in range(3):
        for j, t in enumerate(phrase):
            words.append(_w(t, k * 5 + j, k * 5 + j + 0.5))
            words.append({"type": "spacing", "text": " "})
    kept, _ = g.collapse_loops(words)
    assert "".join(w["text"] for w in kept) == "one two three four five "


def test_tidy_words_logs_what_it_did():
    lines = []
    phrase = ["a", "b", "c", "d", "e"]
    words = [_w(t, k * 5 + j, k * 5 + j + 0.5) for k in range(3) for j, t in enumerate(phrase)]
    words.append(_w("late", 3.0, 2.0))
    g.tidy_words(words, lines.append)
    assert any("loop" in line for line in lines)
    assert any("timing" in line for line in lines)


def test_tidy_words_is_silent_when_there_is_nothing_to_fix():
    lines = []
    g.tidy_words([_w("a", 0, 0.5), _w("b", 0.6, 1.0)], lines.append)
    assert lines == []


def test_one_absurd_start_does_not_drag_the_words_after_it_along():
    """Live: a start of 99,711 s in part 2 of a 66-minute call. Repairing it by
    chaining on the previous word stacked the next 3,447 words at that instant."""
    words = [_w(str(k), 100 + k, 100.5 + k) for k in range(6)]
    words.insert(3, _w("x", 99711.6, 99711.6))
    tail_before = [(w["start"], w["end"]) for w in words[4:]]

    repaired = g.repair_timings(words)

    assert repaired == 1
    assert 102.5 <= words[3]["start"] <= 103.0 and words[3]["end"] <= 103.0
    assert [(w["start"], w["end"]) for w in words[4:]] == tail_before


def test_a_long_looped_sentence_with_punctuation_is_collapsed():
    """The live loop was seventeen words with commas, repeated 60 times."""
    sentence = ("I mean, it used to be that we talk about when we had 400 "
                "warehouses and the average").split()
    words = [_w("so", 0, 0.5)]
    t = 1.0
    for _ in range(4):
        for tok in sentence:
            words.append(_w(tok, t, t + 0.3))
            t += 0.4
    kept, loops = g.collapse_loops(words)
    assert [w["text"] for w in kept] == ["so", *sentence]
    assert loops == [(len(sentence), 4)]
