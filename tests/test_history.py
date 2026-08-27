# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""What was done with a recording outlives the jobs list."""

from __future__ import annotations

from tests.support import isolated_history_dir
from transcriber_studio import history
from transcriber_studio.models import Recording, Source


def rec(id_: str = "plaud-1", name: str = "Board call") -> Recording:
    return Recording(source=Source.PLAUD, id=id_, name=name, date="2026-08-01")


def test_a_recording_with_no_past_is_unknown():
    with isolated_history_dir():
        assert history.get_for(rec()) is None


def test_the_state_of_a_job_is_readable_as_a_label():
    with isolated_history_dir():
        r = rec()
        history.record(r, history.QUEUED)
        assert history.get_for(r).label == "Queued"
        history.record(r, history.RUNNING)
        assert history.get_for(r).label == "In progress…"
        history.record(r, history.DONE)
        assert history.get_for(r).label == "✓ Transcribed"
        history.record(r, history.DONE, ai_cleanup=True)
        assert history.get_for(r).label == "✓ Transcribed + AI cleanup"


def test_an_update_keeps_the_fields_it_does_not_mention():
    """Cleanup finishing later must not erase where the transcript landed."""
    with isolated_history_dir():
        r = rec()
        history.record(r, history.DONE, outputs=["board.txt"], speakers=3)
        history.record(r, history.DONE, ai_cleanup=True)
        entry = history.get_for(r)
        assert entry.outputs == ["board.txt"]
        assert entry.speakers == 3
        assert entry.ai_cleanup is True


def test_history_survives_a_reread_from_disk():
    with isolated_history_dir():
        r = rec()
        history.record(r, history.DONE, outputs=["board.txt"])
        history._cache = None           # as if the app had restarted
        history._cache_stamp = None
        assert history.get_for(r).outputs == ["board.txt"]


def test_removing_a_finished_job_keeps_its_status():
    """The point of the store: the Status column still knows after a clear."""
    with isolated_history_dir():
        done, queued = rec("done-1"), rec("queued-1", "Never ran")
        history.record(done, history.DONE, outputs=["a.txt"])
        history.record(queued, history.QUEUED)
        history.drop_unfinished([done, queued])
        assert history.get_for(done).label == "✓ Transcribed"
        assert history.get_for(queued) is None


def test_a_failure_records_why():
    with isolated_history_dir():
        r = rec()
        history.record(r, history.FAILED, error="audio unavailable")
        entry = history.get_for(r)
        assert entry.label == "✗ Failed"
        assert "audio unavailable" in entry.tooltip()


def test_a_torn_file_is_ignored_rather_than_fatal():
    with isolated_history_dir():
        history.record(rec(), history.DONE)
        history.HISTORY_PATH.write_text("{ half a wri", encoding="utf-8")
        assert history.load(force=True) == {}
