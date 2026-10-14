"""2D | 3D pill switch floating over the bottom-right corner of the viewport."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QHBoxLayout, QToolButton, QWidget

ACCENT = "#2563eb"  # checked segment (matches the Build mesh button)
MARGIN_PX = 14  # from the viewport's right and bottom edges
SCROLLBAR_ALLOWANCE_PX = 14  # clear the 2D view's scroll bars

_STYLE = f"""
#ViewSwitch {{
    background: rgba(20, 24, 32, 215);
    border: 1px solid rgba(255, 255, 255, 40);
    border-radius: 15px;
}}
#ViewSwitch QToolButton {{
    color: #c8ccd4;
    background: transparent;
    border: none;
    border-radius: 12px;
    padding: 3px 14px;
    font-weight: bold;
}}
#ViewSwitch QToolButton:hover {{ background: rgba(255, 255, 255, 30); }}
#ViewSwitch QToolButton:checked {{ background: {ACCENT}; color: white; }}
#ViewSwitch QToolButton:disabled {{ color: #6b7079; }}
"""


class ViewSwitch(QWidget):
    """A segmented "2D | 3D" control that floats over a viewport widget.

    Each segment is a button for an action, so the switch, the View menu and the
    keyboard shortcuts all share one checked state. It keeps itself in the
    viewport's bottom-right corner as the viewport is resized.
    """

    def __init__(self, viewport: QWidget, actions: list[QAction]):
        """Create the switch over a viewport.

        Args:
            viewport: The widget to float over (its child, so it moves with it).
            actions: Checkable actions, one per segment, left to right.
        """
        super().__init__(viewport)
        self.setObjectName("ViewSwitch")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setStyleSheet(_STYLE)
        self.buttons: list[QToolButton] = []
        layout = QHBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(2)
        for act in actions:
            button = QToolButton()
            button.setDefaultAction(act)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            layout.addWidget(button)
            self.buttons.append(button)
        viewport.installEventFilter(self)
        self.place()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt override
        """Stay in the corner when the viewport is resized.

        Args:
            watched: The viewport.
            event: The event.

        Returns:
            False (the event is not consumed).
        """
        if watched is self.parent() and event.type() == QEvent.Type.Resize:
            self.place()
        return False

    def place(self) -> None:
        """Move to the viewport's bottom-right corner, on top of its pages."""
        self.adjustSize()
        viewport = self.parentWidget()
        inset = MARGIN_PX + SCROLLBAR_ALLOWANCE_PX
        self.move(viewport.width() - self.width() - inset, viewport.height() - self.height() - inset)
        self.raise_()
