# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plaud's audio endpoint lies under load; the client must not take it at face value.

All four recordings in the 2026-08-25 batch were in the cloud — phone app, web
UI and `plaud file` all said so — yet the audio endpoint answered one with a
500 and the others with "Audio not available", then served every one of them
minutes later. These cover that behaviour.
"""

from __future__ import annotations

from transcriber_studio import plaud_client
from transcriber_studio.plaud_client import PlaudClient, PlaudError

URL = "https://plaud-bucket.s3-accelerate.amazonaws.com/audiofiles/abc.mp3?X-Amz-Signature=x"
URL_OUTPUT = f"- Fetching audio URL...\n\nAudio Download URL:\n\n{URL}\n"
NOT_AVAILABLE = "- Fetching audio URL...\nAudio not available for this recording.\n"
FILE_WITH_AUDIO = "  id: abc\n  name: a call\n  duration: 27m17s\n  audio: available\n"
FILE_WITHOUT_AUDIO = "  id: abc\n  name: a call\n  duration: 27m17s\n  audio: unavailable\n"


def client(script: list, file_output: str = FILE_WITH_AUDIO) -> tuple[PlaudClient, list]:
    """A client whose CLI replies come from `script`, one per audio call."""
    calls: list[str] = []
    c = PlaudClient()

    def fake_run(args, timeout=None):
        calls.append(args[0])
        if args[0] == "file":
            return file_output
        step = script.pop(0) if script else URL_OUTPUT
        if isinstance(step, Exception):
            raise step
        return step

    c._run = fake_run                      # type: ignore[method-assign]
    plaud_client.time.sleep = lambda _s: None    # no real backoff in tests
    return c, calls


def test_a_url_is_returned_straight_away():
    c, calls = client([URL_OUTPUT])
    assert c.audio_url("abc") == URL
    assert calls == ["audio"], "a first-try success must not cost extra calls"


def test_a_500_is_retried_rather_than_failing_the_job():
    """One recording died on `API error: 500 Internal Server Error`."""
    c, calls = client([PlaudError("[FETCH_FAILED] API error: 500 Internal Server Error"),
                       URL_OUTPUT])
    assert c.audio_url("abc") == URL
    assert calls.count("audio") == 2


def test_not_available_is_retried_when_the_metadata_says_otherwise():
    c, calls = client([NOT_AVAILABLE, URL_OUTPUT], file_output=FILE_WITH_AUDIO)
    assert c.audio_url("abc") == URL
    assert calls.count("audio") == 2
    assert "file" in calls, "the refusal is checked against the file's own metadata"


def test_not_available_is_believed_when_the_metadata_agrees():
    c, calls = client([NOT_AVAILABLE], file_output=FILE_WITHOUT_AUDIO)
    assert c.audio_url("abc") is None
    assert calls.count("audio") == 1, "no point retrying audio that is genuinely absent"


def test_a_permanent_error_is_not_retried():
    c, calls = client([PlaudError("Recording not found")])
    try:
        c.audio_url("abc")
    except PlaudError as e:
        assert "not found" in str(e)
    else:
        raise AssertionError("a hard error must surface, not be swallowed")
    assert calls.count("audio") == 1


def test_retries_are_capped():
    script = [PlaudError("API error: 500 Internal Server Error")] * 10
    c, calls = client(script)
    try:
        c.audio_url("abc")
    except PlaudError:
        pass
    assert calls.count("audio") == plaud_client.AUDIO_URL_ATTEMPTS


def test_the_failure_message_says_which_recording_and_which_case():
    c, _ = client([NOT_AVAILABLE] * 5, file_output=FILE_WITH_AUDIO)
    try:
        c.download_audio("abc", "out.mp3", label="2026-08-25 11:01:54")
    except PlaudError as e:
        message = str(e)
        assert "2026-08-25 11:01:54" in message, "name the row, not a hex id"
        assert "hiccup" in message and "again" in message
    else:
        raise AssertionError("expected the download to report a failure")


def test_a_genuinely_missing_recording_says_so_instead():
    c, _ = client([NOT_AVAILABLE], file_output=FILE_WITHOUT_AUDIO)
    try:
        c.download_audio("abc", "out.mp3", label="a call")
    except PlaudError as e:
        assert "no cloud audio" in str(e)
        assert "sync" in str(e)
    else:
        raise AssertionError("expected the download to report a failure")
