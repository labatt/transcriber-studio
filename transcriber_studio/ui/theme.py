# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Status colours that stay readable in both the light and dark themes.

Qt follows the Windows app theme, so a hardcoded colour is only ever right
half the time: dark green on a near-black background is unreadable, and the
pale green that fixes it washes out on white. Each role below is a pair —
picked for the current palette at the moment a widget is styled.
"""

from __future__ import annotations

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPalette
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QSizePolicy

# (on light background, on dark background)
_GOOD = ("#1b5e20", "#81c784")
_BAD = ("#b71c1c", "#ef9a9a")
_WARN = ("#8a5a00", "#ffb74d")
_MUTED = ("#5f6368", "#b0b0b0")
_HINT = ("#334455", "#9ec1e8")      # filename previews and other examples

# A dialog needs to read as a sheet sitting ON the window, not a hole in it.
# Dark themes are the hard case: near-black on near-black has no edge at all,
# so the sheet is lifted a step in lightness and given an explicit border.
# The step has to clear the app's own panels (#2b2b2b), not just its window
# (#1e1e1e), or the dialog still reads as part of the background.
_SHEET = ("#ffffff", "#3a3a40")
_EDGE = ("#8f959b", "#8a8a93")
_SHEET_HEADER = ("#f0f2f4", "#45454c")


def is_dark() -> bool:
    """True when the app is running against a dark palette."""
    app = QApplication.instance()
    if app is None:
        return False
    return app.palette().color(QPalette.ColorRole.Window).lightness() < 128


def _color(pair: tuple[str, str]) -> str:
    return pair[1] if is_dark() else pair[0]


def good() -> str:
    return f"color: {_color(_GOOD)};"


def bad() -> str:
    return f"color: {_color(_BAD)};"


def warn() -> str:
    return f"color: {_color(_WARN)};"


def muted() -> str:
    return f"color: {_color(_MUTED)};"


def muted_small() -> str:
    return f"color: {_color(_MUTED)}; font-size: 11px;"


def hint() -> str:
    return f"color: {_color(_HINT)}; font-style: italic;"


#: Object name the app assigns to QWizard's internal title band so it can be
#: styled; Qt leaves it unnamed.
WIZARD_HEADER_BAND = "wizardHeaderBand"


def edge_color() -> QColor:
    """Border colour for a dialog outline, drawn rather than styled."""
    return QColor(_color(_EDGE))


def dialog_sheet() -> str:
    """Stylesheet that separates a dialog from whatever is behind it."""
    sheet, header, edge = _color(_SHEET), _color(_SHEET_HEADER), _color(_EDGE)
    return f"""
        QWizard, QDialog {{
            background-color: {sheet};
        }}
        QWizard QWizardPage {{
            background-color: {sheet};
        }}
        /* ModernStyle's title band: an unnamed, auto-filled child painted
           from the palette, and so the one strip of the dialog that would
           otherwise stay exactly the colour of the app behind it. The app
           names it (see WIZARD_HEADER_BAND) so it can be styled here, and it
           carries the top edge of the outline the rest of the frame draws. */
        QWidget#{WIZARD_HEADER_BAND} {{
            background-color: {header};
            border-top: 1px solid {edge};
            border-left: 1px solid {edge};
            border-right: 1px solid {edge};
            border-bottom: 1px solid {edge};
        }}
    """


#: Status roles used in table cells, where a stylesheet is not available.
_ROLES = {
    "good": _GOOD,
    "bad": _BAD,
    "warn": _WARN,
    "muted": _MUTED,
    "info": ("#0d47a1", "#64b5f6"),
}


def qcolor(role: str) -> QColor:
    """Palette-aware colour for an item view, by status role."""
    return QColor(_color(_ROLES.get(role, _MUTED)))


def draw_dialog_edge(widget) -> None:
    """Outline a dialog. A CSS border on a top-level window is not painted."""
    painter = QPainter(widget)
    painter.setPen(edge_color())
    painter.drawRect(widget.rect().adjusted(0, 0, -1, -1))
    painter.end()


class SheetDialog(QDialog):
    """A dialog that reads as a sheet over the window rather than a hole in it.

    Every dialog in the app sat at the same near-black as the window behind it,
    so its boundaries were invisible in the dark theme — you could only infer
    them from the widgets it covered. Subclasses get the lifted background and
    the drawn outline for free.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(dialog_sheet())

    def paintEvent(self, event):
        super().paintEvent(event)
        draw_dialog_edge(self)


class WrappedNote(QLabel):
    """A multi-line explanatory label that is actually given room to be read.

    A word-wrapped QLabel in a form row reports a height for one width and then
    gets laid out at another: long text widens the row until the form no longer
    fits, and what the user sees is a paragraph clipped at the top and bottom of
    a row sized for a width that is off-screen. The fixes are to refuse to widen
    the parent (a small minimum width) and to report the height the text really
    needs at whatever width it ends up with.
    """

    #: Narrow enough that the note never drives the dialog's width.
    MIN_WIDTH = 120

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.setMinimumWidth(self.MIN_WIDTH)
        policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

    def _wrapped_height(self, width: int) -> int:
        metrics = self.fontMetrics()
        rect = metrics.boundingRect(
            QRect(0, 0, max(self.MIN_WIDTH, width), 0),
            int(Qt.TextFlag.TextWordWrap),
            self.text(),
        )
        return rect.height() + metrics.leading() + 4

    def heightForWidth(self, width: int) -> int:
        return self._wrapped_height(width)

    def hasHeightForWidth(self) -> bool:
        return True

    def minimumSizeHint(self) -> QSize:
        return QSize(self.MIN_WIDTH, self._wrapped_height(self.width() or self.MIN_WIDTH))

    def sizeHint(self) -> QSize:
        width = self.width() or 320
        return QSize(width, self._wrapped_height(width))

    def setText(self, text: str) -> None:
        super().setText(text)
        self._refit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refit()

    def _refit(self) -> None:
        """Claim the height this text needs at the width it actually has."""
        needed = self._wrapped_height(self.width())
        if self.minimumHeight() != needed:
            self.setMinimumHeight(needed)   # guarded, or the layout oscillates
            self.updateGeometry()
