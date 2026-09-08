# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""A running QThread must not be collected — Qt aborts the process if it is.

Reported as: the Plaud token expired, the header never updated to say so, and
clicking Logout crashed the app "after a bit". The account check, the login and
the logout all shared one attribute. The startup check was still waiting on the
dead token when Logout replaced it, which dropped the last reference to a
running thread, and Qt ended the process with exit code 127 — no traceback, no
stderr, nothing in the log to work from.

These run in a subprocess on purpose: the failure being guarded against takes
the whole interpreter down, so an in-process assertion could never report it.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

PRELUDE = """
import gc, os, time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication
app = QApplication([])

class SlowWorker(QThread):
    def run(self):
        time.sleep(4)          # stands in for a CLI call against a dead token
"""


def _run(body: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-u", "-c", PRELUDE + textwrap.dedent(body)],
        capture_output=True, text=True, timeout=120,
    )


def test_the_old_single_slot_pattern_really_does_abort():
    """The bug, reproduced. If this ever stops crashing, the fix below is
    guarding against something that no longer happens and the reasoning in
    _track needs revisiting rather than quietly rotting."""
    result = _run("""
        class Holder:
            def start(self):
                self._worker = SlowWorker()     # one slot, overwritten
                self._worker.start()

        h = Holder()
        h.start()
        print("first started", flush=True)
        h.start()
        gc.collect()
        print("SURVIVED", flush=True)
    """)
    assert "first started" in result.stdout
    assert "SURVIVED" not in result.stdout, "expected Qt to abort; it did not"
    assert result.returncode != 0


def test_tracking_the_worker_keeps_the_process_alive():
    """The fix: hold every running worker until its thread actually ends."""
    result = _run("""
        class Holder:
            def __init__(self):
                self._live = set()
            def track(self, w):
                self._live.add(w)
                w.finished.connect(lambda x=w: self._live.discard(x))
                return w
            def start(self):
                self._worker = self.track(SlowWorker())
                self._worker.start()

        h = Holder()
        h.start()
        first = list(h._live)[0]
        h.start()
        gc.collect()
        app.processEvents()
        print("SURVIVED", flush=True)
        # Wait them out before the interpreter exits: a thread still running at
        # shutdown aborts the process for the same reason, which is what
        # closeEvent now guards against in the real window.
        for w in list(h._live):
            w.wait(10000)
    """)
    assert "SURVIVED" in result.stdout, result.stderr[-500:]
    assert result.returncode == 0


# ---- the real window -------------------------------------------------
def test_the_main_window_tracks_its_account_worker():
    result = _run("""
        from transcriber_studio.ui.main_window import MainWindow
        w = MainWindow()
        before = len(w._live_workers)
        w._acct_worker = w._track(SlowWorker())
        w._acct_worker.start()
        assert len(w._live_workers) == before + 1
        # Overwriting the attribute must not orphan the running thread.
        w._acct_worker = None
        gc.collect()
        app.processEvents()
        print("SURVIVED", flush=True)
        w._live_workers.clear()
    """)
    assert "SURVIVED" in result.stdout, result.stderr[-800:]


def test_a_finished_worker_is_let_go():
    """Tracking must not become a leak: they are dropped when the thread ends."""
    result = _run("""
        from transcriber_studio.ui.main_window import MainWindow

        class QuickWorker(QThread):
            def run(self):
                pass

        w = MainWindow()
        worker = w._track(QuickWorker())
        worker.start()
        worker.wait(5000)
        app.processEvents()
        print("still held:", worker in w._live_workers, flush=True)
    """)
    assert "still held: False" in result.stdout, result.stdout + result.stderr[-500:]


def test_a_second_account_action_is_refused_while_one_runs():
    """Two `plaud` calls at once race over the same token file, and the second
    answer overwrites the first."""
    result = _run("""
        from transcriber_studio.ui.main_window import MainWindow
        w = MainWindow()
        w._acct_worker = w._track(SlowWorker())
        w._acct_worker.start()
        assert w._account_busy()
        logged = []
        w._log = logged.append
        w._logout()
        w._login()
        print("refused:", len(logged), flush=True)
        print("busy still the same worker:", w._acct_worker.isRunning(), flush=True)
        w._live_workers.clear()
    """)
    assert "refused: 2" in result.stdout, result.stdout + result.stderr[-500:]
    assert "busy still the same worker: True" in result.stdout


@pytest.mark.parametrize("attr", ["_live_workers"])
def test_the_window_starts_with_nothing_held(attr):
    result = _run(f"""
        from transcriber_studio.ui.main_window import MainWindow
        w = MainWindow()
        print("held at startup:", len(w.{attr}), flush=True)
    """)
    assert "held at startup:" in result.stdout


def test_closing_the_window_waits_for_its_workers():
    """Closing mid-job used to abort at interpreter shutdown for the same
    reason: Qt destroys a still-running thread rather than waiting for it."""
    result = _run("""
        from transcriber_studio.ui.main_window import MainWindow
        from PySide6.QtGui import QCloseEvent
        w = MainWindow()
        worker = w._track(SlowWorker())
        worker.start()
        assert worker.isRunning()
        w.closeEvent(QCloseEvent())
        print("still running after close:", worker.isRunning(), flush=True)
        print("SURVIVED", flush=True)
    """)
    assert "still running after close: False" in result.stdout, result.stdout
    assert "SURVIVED" in result.stdout
    assert result.returncode == 0


def test_closing_cancels_the_workers_that_can_be_cancelled():
    """A cancellable worker is asked to stop before it is waited on, so closing
    does not sit through the rest of a long job."""
    result = _run("""
        from transcriber_studio.ui.main_window import MainWindow
        from PySide6.QtGui import QCloseEvent

        class Cancellable(QThread):
            def __init__(self):
                super().__init__()
                self._stop = False
                self.was_cancelled = False
            def cancel(self):
                self.was_cancelled = True
                self._stop = True
            def run(self):
                for _ in range(200):
                    if self._stop:
                        return
                    time.sleep(0.05)

        w = MainWindow()
        worker = w._track(Cancellable())
        worker.start()
        time.sleep(0.2)
        started = time.time()
        w.closeEvent(QCloseEvent())
        print("cancelled:", worker.was_cancelled, flush=True)
        print("quick:", (time.time() - started) < 3, flush=True)
    """)
    assert "cancelled: True" in result.stdout, result.stdout
    assert "quick: True" in result.stdout, result.stdout


def test_the_account_buttons_grey_out_while_a_call_is_in_flight():
    """An expired token makes the CLI wait out its full timeout. With the
    buttons live, the header reads "Checking Plaud login…" the whole time and
    Login and Logout look ready to press."""
    result = _run("""
        from transcriber_studio.ui.main_window import MainWindow
        w = MainWindow()
        w._live_workers.clear()
        w.login_btn.setEnabled(True); w.logout_btn.setEnabled(True)
        w._start_account_worker(SlowWorker())
        print("login enabled while busy:", w.login_btn.isEnabled(), flush=True)
        print("logout enabled while busy:", w.logout_btn.isEnabled(), flush=True)
        w._acct_worker.wait(10000)
        app.processEvents()
        print("login enabled after:", w.login_btn.isEnabled(), flush=True)
    """)
    assert "login enabled while busy: False" in result.stdout, result.stdout
    assert "logout enabled while busy: False" in result.stdout
    assert "login enabled after: True" in result.stdout, result.stdout
