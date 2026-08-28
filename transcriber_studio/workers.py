# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""QThread workers so the UI never blocks on network or Whisper work."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from .config import Settings
from .job_cancel import JobCancelled
from .jobs import JobResult, JobRunner, copy_transcript, ensure_original_snapshot
from .models import Recording
from .plaud_client import PlaudClient


class TranscriptionWorker(QThread):
    """Runs a list of recordings sequentially (GPU work is serial anyway)."""

    started_item = Signal(int, str)            # row, message
    progress_item = Signal(int, float)         # row, 0..1
    log_item = Signal(int, str)                # row, message
    finished_item = Signal(int, object)        # row, JobResult
    skipped_item = Signal(int)                 # row — never started, batch cancelled
    all_finished = Signal()

    def __init__(
        self,
        settings: Settings,
        recordings: list[Recording],
        start_row: int = 0,
        parent=None,
    ):
        super().__init__(parent)
        self.settings = settings
        self.recordings = recordings
        self.start_row = start_row
        self._cancel = False

    def cancel(self):
        """Stop the job in flight, not just the ones queued behind it."""
        self._cancel = True

    def was_cancelled(self) -> bool:
        return self._cancel

    def run(self):
        runner = JobRunner(self.settings)
        for i, rec in enumerate(self.recordings):
            row = self.start_row + i
            if self._cancel:
                self.skipped_item.emit(row)
                continue
            self.started_item.emit(row, "Starting…")
            result = runner.run(
                rec,
                index=row + 1,
                progress_cb=lambda f, r=row: self.progress_item.emit(r, f),
                log_cb=lambda m, r=row: self.log_item.emit(r, m),
                should_cancel=lambda: self._cancel,
            )
            self.finished_item.emit(row, result)
        self.all_finished.emit()


class DiarizationWorker(QThread):
    """Runs speaker detection on a finished transcript without re-transcribing."""

    log_item = Signal(int, str)
    progress_item = Signal(int, float)
    done = Signal(int, object)   # row, JobResult
    error = Signal(int, str)     # row, message

    def __init__(self, settings: Settings, row: int, result: JobResult, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.row = row
        self.result = result
        self._cancel = False

    def cancel(self):
        """Stop mid-diarization. pyannote reports progress often enough to act on."""
        self._cancel = True

    def was_cancelled(self) -> bool:
        return self._cancel

    def run(self):
        runner = JobRunner(self.settings)
        row = self.row
        try:
            runner.apply_diarization(
                self.result.transcript,
                progress_cb=lambda f, r=row: self.progress_item.emit(r, f),
                log_cb=lambda m, r=row: self.log_item.emit(r, m),
                should_cancel=lambda: self._cancel,
            )
            paths = runner.write_outputs(self.result.transcript, index=row + 1)
            self.result.output_paths = paths
            self.done.emit(row, self.result)
        except JobCancelled:
            # Asking to stop is not an error; the transcript is untouched.
            self.error.emit(row, "Speaker detection cancelled — the transcript is unchanged.")
        except Exception as e:
            self.error.emit(row, str(e))


class CleanupWorker(QThread):
    """Runs AI cleanup on a finished transcript without re-transcribing."""

    log_item = Signal(int, str)
    progress_item = Signal(int, float)
    done = Signal(int, object)   # row, JobResult
    error = Signal(int, str)     # row, message
    cancelled = Signal(int)      # row

    def __init__(
        self,
        settings: Settings,
        row: int,
        result: JobResult,
        *,
        provider: str,
        model: str,
        use_original: bool,
        glossary_id: str | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.settings = settings
        self.row = row
        self.result = result
        self.provider = provider
        self.model = model
        self.use_original = use_original
        self.glossary_id = glossary_id
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        runner = JobRunner(self.settings)
        row = self.row
        try:
            self.log_item.emit(
                row,
                f"AI Cleanup: starting with {self.provider} / {self.model}…",
            )
            self.progress_item.emit(row, 0.01)
            ensure_original_snapshot(self.result)
            source = "original" if self.use_original else "current"
            self.log_item.emit(row, f"AI Cleanup: using {source} transcript")
            if self.use_original:
                if not self.result.original_transcript:
                    raise RuntimeError("Original transcription is not available for this job.")
                working = copy_transcript(self.result.original_transcript)
            else:
                if not self.result.transcript:
                    raise RuntimeError("No transcript available for this job.")
                working = copy_transcript(self.result.transcript)
            runner.apply_ai_cleanup(
                working,
                progress_cb=lambda f, r=row: self.progress_item.emit(r, f),
                log_cb=lambda m, r=row: self.log_item.emit(r, m),
                force=True,
                provider=self.provider,
                model=self.model,
                index=row + 1,
                glossary_id=self.glossary_id,
                should_cancel=lambda: self._cancel,
            )
            if self._cancel:
                self.cancelled.emit(row)
                return
            self.result.transcript = working
            self.result.ai_cleanup_applied = True
            self.result.glossary_id = self.glossary_id
            self.log_item.emit(row, "AI Cleanup: exporting files…")
            paths = runner.write_outputs(
                self.result.transcript,
                index=row + 1,
                cleanup_provider=self.provider,
                cleanup_model=self.model,
            )
            self.result.output_paths = paths
            names = ", ".join(Path(p).name for p in paths[:2])
            extra = f" (+{len(paths) - 2} more)" if len(paths) > 2 else ""
            self.log_item.emit(row, f"AI Cleanup: wrote {len(paths)} file(s) — {names}{extra}")
            self.done.emit(row, self.result)
        except JobCancelled:
            self.cancelled.emit(row)
        except Exception as e:
            self.error.emit(row, str(e))


class AccountWorker(QThread):
    """Fetches Plaud account info / triggers login off the UI thread."""

    done = Signal(object)       # Account | None
    error = Signal(str)

    def __init__(self, action: str = "me", parent=None):
        super().__init__(parent)
        self.action = action

    def run(self):
        client = PlaudClient()
        try:
            if self.action == "login":
                client.login()
            elif self.action == "logout":
                client.logout()
                self.done.emit(None)
                return
            self.done.emit(client.me())
        except Exception as e:
            self.error.emit(str(e))


class ListWorker(QThread):
    """Loads recordings (files/recent/search) off the UI thread."""

    done = Signal(list)
    error = Signal(str)

    def __init__(self, mode: str, settings: Settings, page: int = 1,
                 keyword: str = "", days: int = 7, parent=None):
        super().__init__(parent)
        self.mode = mode
        self.settings = settings
        self.page = page
        self.keyword = keyword
        self.days = days

    def run(self):
        client = PlaudClient()
        try:
            if self.mode == "search":
                recs = client.search(self.keyword)
            elif self.mode == "recent":
                recs = client.recent(self.days)
            else:
                recs = client.list_files(self.page, self.settings.plaud_page_size)
            self.done.emit(recs)
        except Exception as e:
            self.error.emit(str(e))
