# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""The upload budget has to come from the size of the file being uploaded.

`requests` takes a `(connect, read)` timeout pair, which reads as "reaching the
server" and "waiting for its answer". It is not what happens: urllib3 keeps the
*connect* value on the socket while the request body is written, and only swaps
in the read value once the body is fully sent.

So the connect timeout governs the whole upload. At the usual 30 seconds a
70 MB recording had to leave the machine at 19 Mbit/s or the socket gave up,
and the failure read as ``Connection aborted: The write operation timed out`` —
which points at the network rather than at a file that is merely large.
"""

from __future__ import annotations

import pytest

from transcriber_studio import uploads


# ---- the budget ------------------------------------------------------
def test_a_small_file_still_gets_a_usable_floor():
    """DNS, a TLS handshake and a slow start all happen before any bytes move."""
    assert uploads.upload_seconds(300_000) == uploads.MIN_UPLOAD_SECONDS


def test_the_budget_grows_with_the_file():
    small = uploads.upload_seconds(5_000_000)
    large = uploads.upload_seconds(70_000_000)
    assert large > small


def test_the_file_that_failed_now_gets_minutes_not_seconds():
    """70 MB was the real case: 30 seconds needed 19 Mbit/s sustained."""
    seconds = uploads.upload_seconds(70_000_000)
    assert seconds > 300
    assert seconds == pytest.approx(70_000_000 / uploads.MIN_UPLOAD_BYTES_PER_SEC, rel=0.01)


def test_a_hung_transfer_cannot_hold_a_job_open_for_ever():
    assert uploads.upload_seconds(10_000_000_000) == uploads.MAX_UPLOAD_SECONDS


def test_the_ceiling_still_covers_the_largest_upload_any_engine_takes():
    from transcriber_studio import stt_mai

    needed = stt_mai.MAX_UPLOAD_BYTES / uploads.MIN_UPLOAD_BYTES_PER_SEC
    assert uploads.MAX_UPLOAD_SECONDS >= needed


def test_a_missing_file_falls_back_to_the_floor_rather_than_raising():
    connect, read = uploads.timeout_for("no-such-file.mp3", 1800)
    assert connect == uploads.MIN_UPLOAD_SECONDS
    assert read == 1800


def test_the_read_timeout_is_passed_through_untouched(tmp_path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x" * 1000)
    assert uploads.timeout_for(audio, 1234)[1] == 1234


def test_the_description_explains_a_long_wait():
    text = uploads.describe(70_000_000)
    assert "70.0 MB" in text
    assert "min to send" in text


# ---- the engines use it ----------------------------------------------
def _sent_timeout(module, monkeypatch, tmp_path, size_bytes: int):
    """The timeout the engine hands requests for a file of this size."""
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x" * size_bytes)
    seen: dict = {}

    class _Response:
        status_code = 200
        ok = True

        @staticmethod
        def json():
            return {"phrases": [], "combinedPhrases": [], "words": [], "text": ""}

    def fake_post(url, headers=None, files=None, data=None, timeout=None):
        seen["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(module.requests, "post", fake_post)
    return seen, audio


def test_mai_sizes_its_upload_budget(monkeypatch, tmp_path):
    from transcriber_studio import stt_mai

    seen, audio = _sent_timeout(stt_mai, monkeypatch, tmp_path, 40_000_000)
    stt_mai._post(str(audio), "key", "eastus", {"enhancedMode": {}}, None, None)
    connect, read = seen["timeout"]
    assert connect == uploads.upload_seconds(40_000_000)
    assert connect > stt_mai.CONNECT_TIMEOUT      # bigger than the old flat value
    assert read == stt_mai.READ_TIMEOUT


def test_elevenlabs_sizes_its_upload_budget(monkeypatch, tmp_path):
    """It allows 5 GB uploads, and had the same 30 second write budget."""
    from transcriber_studio import stt_elevenlabs

    seen, audio = _sent_timeout(stt_elevenlabs, monkeypatch, tmp_path, 40_000_000)
    stt_elevenlabs._post(str(audio), "key", {"model_id": "scribe_v1"}, None, None)
    connect, _read = seen["timeout"]
    assert connect == uploads.upload_seconds(40_000_000)
    assert connect > stt_elevenlabs.CONNECT_TIMEOUT
