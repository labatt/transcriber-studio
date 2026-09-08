# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""A row of controls that wraps onto another line instead of refusing to fit.

A QHBoxLayout reports the sum of its children as the narrowest it can be, so a
row of six buttons quietly becomes a floor under the whole window: the user
drags the edge inward and nothing moves. Two such rows — the header and the
job actions — were between them holding this app open at 1160px.

Wrapping removes the floor. The row is as wide as the window allows and as tall
as it needs to be, which is the trade a person dragging a window edge is asking
for.
"""

from __future__ import annotations

from PySide6.QtCore import QMargins, QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QLayout, QSizePolicy, QWidgetItem


class FlowLayout(QLayout):
    """Left-to-right, wrapping onto a new line when the width runs out."""

    def __init__(self, parent=None, margin: int = 0, spacing: int = 6):
        super().__init__(parent)
        self._items: list[QWidgetItem] = []
        self._spacing = spacing
        self.setContentsMargins(QMargins(margin, margin, margin, margin))

    # ---- QLayout plumbing ---------------------------------------------
    def addItem(self, item):
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._layout(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QRect):
        super().setGeometry(rect)
        self._layout(rect, apply=True)

    def sizeHint(self) -> QSize:
        """Everything on one line — what the row wants when there is room."""
        margins = self.contentsMargins()
        width = margins.left() + margins.right()
        height = 0
        for i, item in enumerate(self._items):
            hint = item.sizeHint()
            width += hint.width() + (self._spacing if i else 0)
            height = max(height, hint.height())
        return QSize(width, height + margins.top() + margins.bottom())

    def minimumSize(self) -> QSize:
        """One item wide.

        This is the whole point: the row promises only to fit its widest single
        control, not the sum of them, so the window can be dragged narrower and
        the row rearranges to suit.
        """
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(
            margins.left() + margins.right(), margins.top() + margins.bottom()
        )

    # ---- the arrangement ------------------------------------------------
    def _layout(self, rect: QRect, apply: bool) -> int:
        """Place the items, or measure where they would go. Returns the height."""
        margins = self.contentsMargins()
        area = rect.adjusted(
            margins.left(), margins.top(), -margins.right(), -margins.bottom()
        )
        x, y, line_height = area.x(), area.y(), 0
        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + self._spacing
            if line_height and next_x - self._spacing > area.right() + 1:
                # Would overhang: start another line instead of clipping.
                x = area.x()
                y += line_height + self._spacing
                next_x = x + hint.width() + self._spacing
                line_height = 0
            if apply:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y() + margins.bottom()


def make_flow_row(widgets, spacing: int = 8) -> QLayout:
    """A FlowLayout already holding these widgets, in order."""
    layout = FlowLayout(spacing=spacing)
    for widget in widgets:
        widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        layout.addWidget(widget)
    return layout
