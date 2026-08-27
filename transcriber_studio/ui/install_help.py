# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""The components window: what is installed, what is newer, and one-click updates.

The setup wizard can detect a missing dependency, but "ffmpeg not on PATH" is
only useful to someone who already knows what to do about it. And a dependency
that is installed but three releases behind causes a different kind of puzzling
failure, one no "is it installed" check ever reports.

So each component shows the version on this machine next to the current
release, and the exact command that would close the gap. Commands run from
here, in a worker, with their output on screen — but never without being shown
first and confirmed, because they change the user's Python environment.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .. import components
from ..components import (
    MISSING,
    NOT_NEEDED,
    OUTDATED,
    UNCHECKED,
    UNKNOWN,
    UP_TO_DATE,
    Component,
    Status,
)
from ..config import Settings
from .theme import SheetDialog, good, muted, qcolor

STATE_STYLE = {
    UP_TO_DATE: "good",
    OUTDATED: "warn",
    MISSING: "bad",
    UNCHECKED: "muted",
    NOT_NEEDED: "muted",
    UNKNOWN: "warn",
}


class _VersionWorker(QThread):
    """Asks each index what the current release is, off the UI thread."""

    done = Signal(list)     # list[Status]

    def __init__(self, settings: Settings | None, check_network: bool, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.check_network = check_network

    def run(self):
        self.done.emit(
            components.statuses(self.settings, check_network=self.check_network)
        )


class _CommandWorker(QThread):
    """Runs one install/update, streaming its output."""

    line = Signal(str)
    finished_with = Signal(int)

    def __init__(self, cmd: list[str], parent=None):
        super().__init__(parent)
        self.cmd = cmd
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            code = components.run_command(
                self.cmd,
                on_output=self.line.emit,
                should_cancel=lambda: self._cancel,
            )
        except Exception as e:
            self.line.emit(f"Could not run it: {e}")
            code = -1
        self.finished_with.emit(code)


class InstallHelpDialog(SheetDialog):
    """Installs what is missing and updates what is behind, one piece at a time."""

    def __init__(self, parent=None, settings: Settings | None = None, *, auto_check: bool = True):
        super().__init__(parent)
        self.setWindowTitle("Components, versions and updates")
        self.setMinimumSize(760, 620)
        self.settings = settings
        self._statuses: list[Status] = []
        # Checking for updates is the reason anyone opens this window, so the
        # local scan is followed straight away by the live one.
        self._pending_network = auto_check
        self._elevated_log: Path | None = None
        self._version_worker: _VersionWorker | None = None
        self._command_worker: _CommandWorker | None = None
        self._action_buttons: list[QPushButton] = []

        outer = QVBoxLayout(self)
        self.intro = QLabel()
        self.intro.setWordWrap(True)
        outer.addWidget(self.intro)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.body = QWidget()
        self.body_layout = QVBoxLayout(self.body)
        scroll.setWidget(self.body)
        outer.addWidget(scroll, stretch=1)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMaximumHeight(150)
        self.output.setPlaceholderText(
            "Output from an install or update appears here."
        )
        self.output.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        outer.addWidget(self.output)

        buttons = QDialogButtonBox()
        self.check_btn = QPushButton("Check for updates")
        self.check_btn.setToolTip(
            "Ask PyPI, npm and GitHub what the current releases are."
        )
        self.check_btn.clicked.connect(lambda: self.refresh(check_network=True))
        buttons.addButton(self.check_btn, QDialogButtonBox.ButtonRole.ActionRole)
        self.recheck_btn = QPushButton("Re-scan")
        self.recheck_btn.setToolTip("Look at this machine again after installing something.")
        self.recheck_btn.clicked.connect(lambda: self.refresh(check_network=False))
        buttons.addButton(self.recheck_btn, QDialogButtonBox.ButtonRole.ResetRole)
        buttons.addButton(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        outer.addWidget(buttons)

        self.refresh(check_network=False)

    # ------------------------------------------------------------------
    def refresh(self, *, check_network: bool = False):
        """Rebuild the list from a fresh look at the machine (and the indexes)."""
        if self._version_worker and self._version_worker.isRunning():
            # Queue it rather than dropping it: pressing Check for updates while
            # the first scan is still going would otherwise do nothing at all.
            self._pending_network = self._pending_network or check_network
            return
        self._set_busy(True)
        self.intro.setText(
            "Checking the current releases…" if check_network
            else "Looking at what is installed…"
        )
        self._version_worker = _VersionWorker(self.settings, check_network, self)
        self._version_worker.done.connect(self._render)
        self._version_worker.finished.connect(self._on_scan_finished)
        self._version_worker.start()

    def _on_scan_finished(self):
        self._set_busy(False)
        if self._pending_network:
            self._pending_network = False
            self.refresh(check_network=True)

    def _set_busy(self, busy: bool):
        self.check_btn.setEnabled(not busy)
        self.recheck_btn.setEnabled(not busy)
        for button in self._action_buttons:
            button.setEnabled(not busy)

    def _render(self, statuses: list[Status]):
        self._statuses = statuses
        self._action_buttons = []
        while self.body_layout.count():
            item = self.body_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.intro.setStyleSheet("")
        missing = components.missing(statuses)
        required = components.missing(statuses, only_required=True)
        outdated = components.outdated(statuses)
        checked = any(s.latest for s in statuses)

        pending = self._pending_network or (
            self._version_worker is not None and self._version_worker.isRunning()
        )
        if not missing and not outdated:
            self.intro.setStyleSheet(good() if checked else "")
            self.intro.setText(
                "Everything is installed and current."
                if checked
                else "Everything this app needs is installed — checking the current "
                     "releases…" if pending else
                     "Everything this app needs is installed."
            )
        else:
            parts = []
            if missing:
                parts.append(
                    f"{len(missing)} not installed"
                    + (f" ({len(required)} of them required)" if required else " (all optional)")
                )
            if outdated:
                parts.append(f"{len(outdated)} behind the current release")
            tail = (
                "Still checking the current releases…" if pending
                else "Press an Install or Update button, or copy the command and run "
                     "it in a terminal."
            )
            self.intro.setText(" · ".join(parts) + (". " if parts else "") + tail)

        for status in statuses:
            self.body_layout.addWidget(self._card(status))
        self.body_layout.addStretch()

    # ------------------------------------------------------------------
    def _card(self, status: Status) -> QGroupBox:
        component = status.component
        tag = "optional" if component.optional else "required"
        box = QGroupBox(f"{component.title}  ({tag})")
        layout = QVBoxLayout(box)

        version_row = QHBoxLayout()
        state = QLabel(status.summary())
        palette_role = STATE_STYLE.get(status.state, "muted")
        state.setStyleSheet(f"color: {qcolor(palette_role).name()}; font-weight: bold;")
        version_row.addWidget(state)
        if status.detail:
            detail = QLabel(f"· {status.detail}")
            detail.setStyleSheet(muted())
            version_row.addWidget(detail)
        version_row.addStretch()
        version_row.addWidget(self._action_button(status))
        layout.addLayout(version_row)

        why = QLabel(component.why)
        why.setWordWrap(True)
        layout.addWidget(why)

        if status.is_actionable:
            steps = component.install_steps if status.state == MISSING else []
            for step in steps:
                label = QLabel(step)
                label.setWordWrap(True)
                layout.addWidget(label)
            command = self._command_for(status)
            if command:
                layout.addWidget(self._command_row(components.command_text(command)))
            if component.restart_required:
                note = QLabel("Restart the app afterwards for this to take effect.")
                note.setStyleSheet(muted())
                layout.addWidget(note)
            if components.needs_elevation(component):
                note = QLabel(
                    "This will need administrator rights — you will be asked."
                )
                note.setStyleSheet(muted())
                layout.addWidget(note)

        if component.notes:
            note = QLabel(component.notes)
            note.setWordWrap(True)
            note.setStyleSheet(muted())
            layout.addWidget(note)

        if component.url:
            link = QLabel(f'<a href="{component.url}">{component.url}</a>')
            link.setOpenExternalLinks(True)
            link.setStyleSheet(muted())
            layout.addWidget(link)
        return box

    @staticmethod
    def _command_for(status: Status) -> list[str]:
        if status.state == MISSING:
            return components.install_command(status.component, status.latest)
        return components.update_command(status.component, status.latest)

    def _action_button(self, status: Status) -> QWidget:
        command = self._command_for(status)
        if not status.is_actionable:
            spacer = QLabel("")
            return spacer
        if not command:
            # A downloaded binary: the releases page is the action.
            button = QPushButton("Open download page")
            button.clicked.connect(
                lambda _checked=False, url=status.component.url: self._open(url)
            )
        else:
            button = QPushButton("Install" if status.state == MISSING else "Update")
            button.clicked.connect(
                lambda _checked=False, s=status: self._run(s)
            )
        button.setAutoDefault(False)
        self._action_buttons.append(button)
        return button

    @staticmethod
    def _open(url: str):
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl(url))

    def _command_row(self, command: str) -> QWidget:
        """The command as text, selectable and copyable — the escape hatch for
        anyone who would rather run it in their own terminal."""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(12, 0, 0, 0)
        field = QLabel(command)
        field.setWordWrap(True)
        field.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        field.setStyleSheet("font-family: Consolas, monospace;")
        copy_btn = QPushButton("Copy")
        copy_btn.setFixedWidth(64)
        copy_btn.setAutoDefault(False)
        copy_btn.clicked.connect(lambda _checked=False, c=command: self._copy(c, copy_btn))
        layout.addWidget(field, stretch=1)
        layout.addWidget(copy_btn)
        return row

    @staticmethod
    def _copy(command: str, button: QPushButton):
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(command)
            button.setText("Copied")

    def _run(self, status: Status, command: list[str] | None = None, *, elevated: bool = False):
        """Show the exact command, get a yes, then run it with its output visible."""
        if self._command_worker and self._command_worker.isRunning():
            QMessageBox.information(
                self, "Components", "Another install is already running."
            )
            return
        command = command or self._command_for(status)
        text = components.command_text(command)
        verb = "Install" if status.state == MISSING else "Update"
        message = f"{verb} {status.component.title}?\n\nThis will run:\n\n{text}"
        if elevated:
            message += (
                "\n\nWindows will ask for permission, and the command runs in its "
                "own elevated process — its output is collected and shown here "
                "when it finishes."
            )
        if status.component.restart_required:
            message += "\n\nThe app has to be restarted afterwards for it to take effect."
        answer = QMessageBox.question(
            self,
            f"{verb} {status.component.title}",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        self.output.clear()
        self.output.appendPlainText(f"$ {text}\n")
        self._set_busy(True)
        self._elevated_log = None
        if elevated:
            log = Path(tempfile.gettempdir()) / "pws_elevated_install.log"
            log.unlink(missing_ok=True)
            self._elevated_log = log
            self.output.appendPlainText("Waiting for the elevation prompt…")
            command = components.elevated_command(command, str(log))
        self._command_worker = _CommandWorker(command, self)
        self._command_worker.line.connect(self.output.appendPlainText)
        self._command_worker.finished_with.connect(
            lambda code, s=status: self._on_command_finished(code, s)
        )
        self._command_worker.start()

    def _on_command_finished(self, code: int, status: Status):
        self._read_elevated_log()
        if code == 0:
            self.output.appendPlainText("\nDone.")
            # An installer that edits PATH edits the registry, not this
            # process. Without this, something just installed reads as missing —
            # and a winget *upgrade* moves the install into a version-stamped
            # folder, so the entry we started with is dead and a dependency the
            # machine now has reads as gone.
            added = components.refresh_path()
            if added:
                self.output.appendPlainText(
                    f"Picked up {len(added)} new PATH entry(ies) — no restart needed for that."
                )
            if status.component.restart_required:
                self.output.appendPlainText(
                    "Restart the app for the new version to be used."
                )
        else:
            self.output.appendPlainText(
                f"\nExited with code {code}. Nothing was changed if it failed early; "
                "the output above says why."
            )
        if components.looks_like_a_permission_problem(self.output.toPlainText()):
            self._offer_more_rights(status)
        self._set_busy(False)
        self.refresh(check_network=False)

    def _read_elevated_log(self):
        """Collect what the elevated process wrote, since it could not be piped."""
        log = self._elevated_log
        if not log:
            return
        for path in (log, Path(str(log) + ".err")):
            try:
                text = path.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                continue
            if text:
                self.output.appendPlainText(text)
        self._elevated_log = None

    def _offer_more_rights(self, status: Status):
        """It was refused for lack of rights — offer the two ways round that."""
        component = status.component
        box = QMessageBox(self)
        box.setWindowTitle("Not enough rights")
        box.setText(f"{component.title} could not be written where it lives.")
        box.setInformativeText(
            "Windows refused the write. Installing into your own user folder "
            "needs no special rights and is usually the right answer; running as "
            "administrator changes the installation everyone on this machine "
            "shares."
        )
        as_user = None
        if component.kind == "pip" and not components.in_virtualenv():
            as_user = box.addButton(
                "Install for me only (--user)", QMessageBox.ButtonRole.AcceptRole
            )
        as_admin = None
        if components.can_elevate():
            as_admin = box.addButton(
                "Run as administrator", QMessageBox.ButtonRole.DestructiveRole
            )
        else:
            # No tested escalation path off Windows, and an untested one is not
            # something to run on someone's machine. Hand over the command.
            box.setInformativeText(
                box.informativeText()
                + "\n\nTo install system-wide, run the command shown above "
                "yourself with sudo."
            )
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is as_user:
            command = self._command_for(status)
            if "--user" not in command:
                command = command + ["--user"]
            self._run(status, command)
        elif as_admin is not None and clicked is as_admin:
            self._run(status, self._command_for(status), elevated=True)

    def closeEvent(self, event):
        worker = self._command_worker
        if worker and worker.isRunning():
            answer = QMessageBox.question(
                self,
                "Components",
                "An install is still running. Stop it and close?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            worker.cancel()
            worker.wait(3000)
        super().closeEvent(event)


def missing(only_required: bool = False) -> list[Component]:
    """Components not installed right now — a machine-only check, no network."""
    return [
        s.component
        for s in components.missing(components.statuses(), only_required=only_required)
    ]
