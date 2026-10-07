# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""QThread workers so the UI never blocks on network or Whisper work."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from .config import Settings
from .job_cancel import JobCancelled
from .jobs import (
    JobResult,
    JobRunner,
    copy_transcript,
    ensure_original_snapshot,
    remove_superseded_outputs,
)
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
        *,
        speaker_names: dict[str, dict[str, str]] | None = None,
    ):
        super().__init__(parent)
        self.settings = settings
        self.recordings = recordings
        self.start_row = start_row
        #: recording id -> {raw pyannote label -> name}, from the user having
        #: listened to each speaker before this run (see SpeakerIdentifyWorker).
        self.speaker_names = dict(speaker_names or {})
        self._cancel = False

    def cancel(self):
        """Stop the job in flight, not just the ones queued behind it."""
        self._cancel = True

    def was_cancelled(self) -> bool:
        return self._cancel

    def run(self):
        runner = JobRunner(self.settings, speaker_names=self.speaker_names)
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
    """Runs speaker detection on a finished transcript without re-transcribing.

    Also how speakers get detected *again*: pass ``min_speakers`` and
    ``max_speakers`` to override the saved limits for this run. The work is
    done on a copy, so a cancelled or failed run leaves the job exactly as it
    was, and the outputs the new speaker names supersede are removed.
    """

    log_item = Signal(int, str)
    progress_item = Signal(int, float)
    done = Signal(int, object)   # row, JobResult
    error = Signal(int, str)     # row, message

    def __init__(
        self, settings: Settings, row: int, result: JobResult, parent=None,
        *, min_speakers: int | None = None, max_speakers: int | None = None,
    ):
        super().__init__(parent)
        self.settings = settings
        self.row = row
        self.result = result
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self._cancel = False

    def cancel(self):
        """Stop mid-diarization. pyannote reports progress often enough to act on."""
        self._cancel = True

    def was_cancelled(self) -> bool:
        return self._cancel

    def run(self):
        runner = JobRunner(self.settings)
        row = self.row
        working = copy_transcript(self.result.transcript)
        try:
            runner.apply_diarization(
                working,
                progress_cb=lambda f, r=row: self.progress_item.emit(r, f),
                log_cb=lambda m, r=row: self.log_item.emit(r, m),
                should_cancel=lambda: self._cancel,
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers,
            )
            previous = list(self.result.output_paths)
            paths = runner.write_outputs(working, index=row + 1)
            if working.words and self.result.ai_cleanup_applied:
                # The turns were redrawn from the raw words, so the cleanup
                # pass that rewrote the old turns is gone with them. It can be
                # run again on the new speakers.
                self.result.ai_cleanup_applied = False
                self.log_item.emit(row, "AI cleanup was undone by re-detecting speakers; "
                                        "run it again on the new turns.")
            self.result.transcript = working
            self.result.output_paths = paths
            for gone in remove_superseded_outputs(previous, paths):
                self.log_item.emit(row, f"Removed superseded output: {Path(gone).name}")
            self.done.emit(row, self.result)
        except JobCancelled:
            # Asking to stop is not an error; the transcript is untouched.
            self.error.emit(row, "Speaker detection cancelled — the transcript is unchanged.")
        except Exception as e:
            self.error.emit(row, str(e))


class SpeakerIdentifyWorker(QThread):
    """Finds the speakers in one recording before it is transcribed.

    The first half of a run on its own — download, denoise, pyannote — so the
    window can play a few seconds of each voice and ask who it is. The answers
    ride into the transcription that follows as ``speaker_names``.
    """

    log_item = Signal(int, str)
    progress_item = Signal(int, float)
    done = Signal(int, str, object)     # row, audio path, DiarizationResult
    error = Signal(int, str)            # row, message
    cancelled = Signal(int)             # row

    def __init__(
        self, settings: Settings, row: int, recording: Recording, parent=None,
        *, min_speakers: int = 0, max_speakers: int = 0,
    ):
        super().__init__(parent)
        self.settings = settings
        self.row = row
        self.recording = recording
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        runner = JobRunner(self.settings)
        row = self.row
        try:
            audio_path, diarized = runner.detect_speakers(
                self.recording,
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers,
                progress_cb=lambda f, r=row: self.progress_item.emit(r, f),
                log_cb=lambda m, r=row: self.log_item.emit(r, m),
                should_cancel=lambda: self._cancel,
            )
            self.done.emit(row, audio_path, diarized)
        except JobCancelled:
            self.cancelled.emit(row)
        except Exception as e:
            self.error.emit(row, str(e))


class DownloadWorker(QThread):
    """Downloads recordings' audio and nothing else — no transcription.

    Each file goes through the audio cache, so transcribing it later costs no
    second download, and a copy is placed in the folder the user chose.
    """

    started_item = Signal(int, str)
    progress_item = Signal(int, float)
    log_item = Signal(int, str)
    finished_item = Signal(int, object)    # row, JobResult (output_paths = the saved file)
    skipped_item = Signal(int)
    all_finished = Signal()

    def __init__(
        self, settings: Settings, recordings: list[Recording], dest_dir: str,
        start_row: int = 0, parent=None,
    ):
        super().__init__(parent)
        self.settings = settings
        self.recordings = recordings
        self.dest_dir = dest_dir
        self.start_row = start_row
        self._cancel = False

    def cancel(self):
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
            self.started_item.emit(row, "Downloading…")
            try:
                saved = runner.download_audio(
                    rec, self.dest_dir,
                    progress_cb=lambda f, r=row: self.progress_item.emit(r, f),
                    log_cb=lambda m, r=row: self.log_item.emit(r, m),
                    should_cancel=lambda: self._cancel,
                )
                self.finished_item.emit(row, JobResult(rec, [saved]))
            except JobCancelled:
                self.finished_item.emit(row, JobResult(rec, cancelled=True))
            except Exception as e:
                self.finished_item.emit(row, JobResult(rec, error=str(e)))
        self.all_finished.emit()


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


class RenameWorker(QThread):
    """Pushes one renamed recording to Plaud, off the UI thread.

    The local rename is already committed by the time this starts. This only
    decides whether the name also reaches Plaud, so a failure here is reported
    as a name that did not sync — never as a rename that did not happen.
    """

    done = Signal(str)          # file_id — Plaud accepted the name
    error = Signal(str, str)    # file_id, message
    # Plaud rotates the refresh token as it spends it, and the one it replaces
    # stops working. Losing the replacement means the next run pastes a dead
    # token and gets "re-exchange required", so it has to be saved.
    rotated = Signal(str)       # a refresh token that replaced the saved one

    def __init__(self, settings: Settings, file_id: str, name: str, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.file_id = file_id
        self.name = name

    def run(self):
        from . import name_store, plaud_web

        try:
            client = plaud_web.PlaudWebClient(
                self.settings.plaud_web_token,
                plaud_web.API_HOSTS.get(
                    self.settings.plaud_web_region, plaud_web.DEFAULT_HOST
                ),
            )
            client.on_refresh = self._remember
            client.rename(self.file_id, self.name)
            name_store.mark_pushed(self.file_id)
            self.done.emit(self.file_id)
        except Exception as e:
            self.error.emit(self.file_id, str(e))

    def _remember(self, creds) -> None:
        """Hand a rotated refresh token to the UI thread, which owns saving."""
        if creds.refresh_token and creds.refresh_token != self.settings.plaud_web_token:
            self.rotated.emit(creds.refresh_token)


class TokenCheckWorker(QThread):
    """Proves a pasted Plaud web token works, without changing anything."""

    done = Signal(str)      # a short description of what was accepted
    error = Signal(str)
    rotated = Signal(str)   # a refresh token that replaced the one checked

    def __init__(self, token: str, region: str, parent=None):
        super().__init__(parent)
        self.token = token
        self.region = region

    def run(self):
        from . import plaud_web

        try:
            info = plaud_web.validate_token(self.token)
            client = plaud_web.PlaudWebClient(
                self.token, plaud_web.API_HOSTS.get(self.region, plaud_web.DEFAULT_HOST)
            )
            # Checking a refresh token spends it, and Plaud may hand back a
            # replacement. Dropping that would leave the saved token dead the
            # moment this check succeeds.
            client.on_refresh = lambda creds: (
                self.rotated.emit(creds.refresh_token)
                if creds.refresh_token and creds.refresh_token != self.token
                else None
            )
            client.check()
        except Exception as e:
            self.error.emit(str(e))
            return
        # Not days_left: the user token is minted for 24 hours, so whole days
        # rounds almost every good token down to "0 days left".
        self.done.emit(f"Token accepted — {info.describe_remaining()} left.")


class MaiTestWorker(QThread):
    """Proves an Azure Speech key and region reach MAI-Transcribe."""

    ok = Signal(str)
    failed = Signal(str)

    def __init__(self, api_key: str, region: str, parent=None):
        super().__init__(parent)
        self.api_key = api_key
        self.region = region

    def run(self):
        from . import stt_mai

        try:
            self.ok.emit(stt_mai.test_key(self.api_key, self.region))
        except Exception as e:
            self.failed.emit(str(e))
