# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""A job cannot still be running when the app has only just started.

Reported as: a job that failed still showed "in progress" in the recordings
list after a relaunch. Nothing wrote that state wrongly — the run simply ended
without saying so, because the app was closed or crashed mid-job, and the entry
kept the last thing anyone told it. History is only reconciled when a job row
is removed by hand, so nobody ever corrected it.
"""

from __future__ import annotations

from transcriber_studio import history
from transcriber_studio.models import Recording, Source

from .support import isolated_history_dir


def _recording(rec_id: str = "f1", name: str = "Team sync") -> Recording:
    return Recording(source=Source.PLAUD, id=rec_id, name=name)


def test_a_running_entry_is_retired_on_startup():
    with isolated_history_dir():
        history.record(_recording(), history.RUNNING)
        assert history.reconcile_stale() == 1
        assert history.get("f1").state == history.INTERRUPTED


def test_a_queued_entry_is_retired_too():
    """Queued means "about to start", which is equally impossible at startup."""
    with isolated_history_dir():
        history.record(_recording(), history.QUEUED)
        history.reconcile_stale()
        assert history.get("f1").state == history.INTERRUPTED


def test_finished_work_is_left_alone():
    with isolated_history_dir():
        history.record(_recording("done1"), history.DONE)
        history.record(_recording("failed1"), history.FAILED)
        history.record(_recording("cancelled1"), history.CANCELLED)
        assert history.reconcile_stale() == 0
        assert history.get("done1").state == history.DONE
        assert history.get("failed1").state == history.FAILED
        assert history.get("cancelled1").state == history.CANCELLED


def test_the_output_files_are_kept():
    """A job can die after writing its files, and the entry is the only record
    of where they went — so the state is corrected, not the entry deleted."""
    with isolated_history_dir():
        history.record(
            _recording(), history.RUNNING,
            outputs=["C:/out/transcript.txt"], speakers=5, ai_cleanup=True,
        )
        history.reconcile_stale()
        entry = history.get("f1")
        assert entry.outputs == ["C:/out/transcript.txt"]
        assert entry.speakers == 5


def test_reconciling_twice_changes_nothing_the_second_time():
    with isolated_history_dir():
        history.record(_recording(), history.RUNNING)
        assert history.reconcile_stale() == 1
        assert history.reconcile_stale() == 0


def test_it_survives_a_reload_from_disk():
    with isolated_history_dir():
        history.record(_recording(), history.RUNNING)
        history.reconcile_stale()
        history.load(force=True)
        assert history.get("f1").state == history.INTERRUPTED


def test_the_label_says_interrupted_rather_than_in_progress():
    with isolated_history_dir():
        history.record(_recording(), history.RUNNING)
        assert "In progress" in history.get("f1").label
        history.reconcile_stale()
        assert history.get("f1").label == "⚠ Interrupted"


def test_the_tooltip_says_what_to_do_about_it():
    with isolated_history_dir():
        history.record(_recording(), history.RUNNING)
        history.reconcile_stale()
        assert "Resume" in history.get("f1").tooltip()


def test_the_status_column_can_size_itself_to_the_new_label():
    """possible_labels feeds the column width; a label missing from it arrives
    elided."""
    assert "⚠ Interrupted" in history.possible_labels()


def test_an_interrupted_job_is_still_forgotten_when_its_row_is_removed():
    """It produced no finished result, so removing the row should forget it,
    the same as a queued or running one."""
    with isolated_history_dir():
        history.record(_recording(), history.RUNNING)
        history.reconcile_stale()
        history.drop_unfinished([_recording()])
        assert history.get("f1") is None


def test_the_recordings_list_colours_it_as_a_warning():
    from transcriber_studio.ui.recordings_tab import STATE_ROLES

    assert STATE_ROLES[history.INTERRUPTED] == "warn"


def test_the_main_window_reconciles_before_anything_reads_the_history():
    """The count is taken at construction, so the first paint of the Status
    column already shows the corrected state."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from transcriber_studio.ui.main_window import MainWindow

    QApplication.instance() or QApplication([])
    with isolated_history_dir():
        history.record(_recording(), history.RUNNING)
        window = MainWindow()
        assert window._stale_jobs >= 1
        assert history.get("f1").state == history.INTERRUPTED
        window._live_workers.clear()
