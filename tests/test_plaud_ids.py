# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plaud CLI 0.3.11 prefixes file ids with "of_" and rejects the bare form.

Seen live on 2026-09-15: the recordings list came back empty because the row
parser wanted a bare 32-hex id, and `plaud file <bare id>` answered NOT_FOUND.
The app keeps storing the bare id, since every cache file, history row and
checkpoint is named by it; the prefix is a CLI detail.
"""

from __future__ import annotations

from transcriber_studio.plaud_client import PlaudClient, bare_id, cli_id

NEW_LISTING = """\
- Fetching files...

Files on this page: 2

  ID                                  NAME                                  DATE          DURATION
  ──────────────────────────────────────────────────────────────────────────────────────────────────
  of_f8cc089c03fcb36ecb95d56150976e63  2026-09-15 14:25:16                   2026-09-15    1h34m
  of_930fba0474c877f6f940fdd5a9e262a1  2026-09-09 14:10:58  Ann Smarty Mee…  2026-09-09    53m21s
"""

OLD_LISTING = """\
  ID                                NAME                DATE          DURATION
  f8cc089c03fcb36ecb95d56150976e63  Team sync           2026-09-15    1h34m
"""

NEW_DETAILS = """\
- Fetching file...

File Details:

  id:           of_7f2f7ef4cc2e36a983a9f8359355c093
  name:         2026-09-08 15:06:37 Mark
  duration:     1h13m
  audio:        available
"""


def _client(outputs: dict[str, str]) -> tuple[PlaudClient, list[list[str]]]:
    calls: list[list[str]] = []

    def fake_run(args, timeout=None):
        calls.append(list(args))
        return outputs[args[0]]

    c = PlaudClient()
    c._run = fake_run  # type: ignore[method-assign]
    return c, calls


def test_the_new_listing_parses_and_stores_bare_ids():
    c, _ = _client({"files": NEW_LISTING})
    recs = c.list_files(1, 20)
    assert [r.id for r in recs] == [
        "f8cc089c03fcb36ecb95d56150976e63", "930fba0474c877f6f940fdd5a9e262a1",
    ]
    assert recs[1].name == "2026-09-09 14:10:58  Ann Smarty Mee"
    assert recs[0].duration == "1h34m"


def test_the_old_listing_still_parses():
    c, _ = _client({"files": OLD_LISTING})
    assert [r.id for r in c.list_files(1, 20)] == ["f8cc089c03fcb36ecb95d56150976e63"]


def test_the_cli_is_given_the_prefixed_id_for_details_and_audio():
    c, calls = _client({"file": NEW_DETAILS, "audio": "https://example.test/a.mp3"})
    rec = c.get_file("7f2f7ef4cc2e36a983a9f8359355c093")
    c.audio_url("7f2f7ef4cc2e36a983a9f8359355c093")
    assert calls[0] == ["file", "of_7f2f7ef4cc2e36a983a9f8359355c093"]
    assert calls[1] == ["audio", "of_7f2f7ef4cc2e36a983a9f8359355c093"]
    assert rec.id == "7f2f7ef4cc2e36a983a9f8359355c093", "details never re-prefix the stored id"
    assert rec.audio_available


def test_an_already_prefixed_id_is_not_prefixed_twice():
    assert cli_id("of_abc") == "of_abc"
    assert cli_id("abc") == "of_abc"
    assert bare_id("of_abc") == "abc"
    assert bare_id("abc") == "abc"
    assert bare_id("") == ""
