# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run with: python -m transcriber_studio (or the transcriber-studio command)."""

from __future__ import annotations

import sys

from .suppress_warnings import configure as configure_warnings

configure_warnings()

from .hardware import configure_cuda_dll_paths

configure_cuda_dll_paths()

from PySide6.QtWidgets import QApplication

from . import components, config
from .ui.main_window import MainWindow

SCROLLBAR_STYLE = """
QScrollBar:vertical {
    background: #2b2b2b;
    width: 16px;
    margin: 2px 0 2px 0;
}
QScrollBar::handle:vertical {
    background: #9a9a9a;
    min-height: 40px;
    border-radius: 6px;
    margin: 2px;
}
QScrollBar::handle:vertical:hover {
    background: #bdbdbd;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
}
"""


def main():
    # A dependency installed after this process's parent shell started is on
    # PATH in the registry but not in our environment. Additive, so an
    # activated virtualenv keeps its own entries.
    components.refresh_path()
    # An install from before the project was renamed keeps its settings,
    # glossaries and caches — they are copied across on first run.
    adopted = config.migrate_legacy_dir()
    app = QApplication(sys.argv)
    app.setApplicationName(config.APP_NAME)
    app.setStyleSheet(app.styleSheet() + SCROLLBAR_STYLE)
    window = MainWindow()
    if adopted:
        window._log(f"Settings and glossaries carried over from {adopted}.")
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
