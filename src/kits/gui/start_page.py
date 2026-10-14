"""Pages shown in the main area when there is no board to display."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import __version__

_PATH_ROLE = Qt.ItemDataRole.UserRole
_CONTENT_WIDTH = 560


def _title(text: str, point_delta: int) -> QLabel:
    """Create a bold heading label.

    Args:
        text: Heading text.
        point_delta: Points added to the default font size.

    Returns:
        The label.
    """
    label = QLabel(text)
    font = QFont(label.font())
    font.setPointSize(font.pointSize() + point_delta)
    font.setBold(True)
    label.setFont(font)
    return label


def _centred(content: QWidget) -> QHBoxLayout:
    """Wrap a widget so it sits centred at a fixed width.

    Args:
        content: The widget to centre.

    Returns:
        A layout with stretch on both sides.
    """
    content.setMaximumWidth(_CONTENT_WIDTH)
    row = QHBoxLayout()
    row.addStretch(1)
    row.addWidget(content, 4)
    row.addStretch(1)
    return row


class StartPage(QWidget):
    """Welcome page: new/open buttons and a list of recent projects.

    Signals:
        new_requested(): The user clicked "New Project".
        open_requested(): The user clicked "Open Project".
        recent_requested(Path): The user chose a recent project.
    """

    new_requested = Signal()
    open_requested = Signal()
    recent_requested = Signal(object)

    def __init__(self, parent=None):
        """Create the page.

        Args:
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        new_button = QPushButton("New Project…")
        open_button = QPushButton("Open Project…")
        new_button.clicked.connect(self.new_requested)
        open_button.clicked.connect(self.open_requested)
        buttons = QHBoxLayout()
        buttons.addWidget(new_button)
        buttons.addWidget(open_button)
        buttons.addStretch(1)

        self.recent_list = QListWidget()
        self.recent_list.itemActivated.connect(self._on_activated)  # double-click or Enter
        self.empty_label = QLabel("No recent projects.")
        self.empty_label.setEnabled(False)

        content = QWidget()
        column = QVBoxLayout(content)
        column.addWidget(_title("KITS", 10))
        column.addWidget(QLabel(f"KiCad current density and thermal simulation · version {__version__}"))
        column.addSpacing(16)
        column.addLayout(buttons)
        column.addSpacing(16)
        column.addWidget(_title("Recent projects", 2))
        column.addWidget(self.empty_label)
        column.addWidget(self.recent_list, 1)

        layout = QVBoxLayout(self)
        layout.addSpacing(40)
        layout.addLayout(_centred(content), 1)
        layout.addSpacing(40)

    def set_recent(self, paths: list[Path]) -> None:
        """Show a list of recent projects; files that no longer exist are greyed out.

        Args:
            paths: Project files, most recent first.
        """
        self.recent_list.clear()
        for path in paths:
            item = QListWidgetItem(f"{path.stem}\n{path}")
            item.setData(_PATH_ROLE, str(path))
            if path.is_file():
                item.setToolTip(f"{path}\nDouble-click to open")
            else:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                item.setToolTip(f"File not found: {path}")
            self.recent_list.addItem(item)
        self.recent_list.setVisible(bool(paths))
        self.empty_label.setVisible(not paths)

    def _on_activated(self, item: QListWidgetItem) -> None:
        """Request opening the chosen recent project.

        Args:
            item: The clicked or activated list item.
        """
        if item.flags() & Qt.ItemFlag.ItemIsEnabled:
            self.recent_requested.emit(Path(item.data(_PATH_ROLE)))


class NoBoardPage(QWidget):
    """Shown when the open project has no loaded board (none linked, or it failed to load).

    Signals:
        link_requested(): The user clicked "Link KiCad Board".
    """

    link_requested = Signal()

    def __init__(self, parent=None):
        """Create the page.

        Args:
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self.message = QLabel()
        self.message.setWordWrap(True)
        self.message.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        link_button = QPushButton("Link KiCad Board…")
        link_button.clicked.connect(self.link_requested)
        row = QHBoxLayout()
        row.addWidget(link_button)
        row.addStretch(1)

        content = QWidget()
        column = QVBoxLayout(content)
        column.addWidget(_title("No board loaded", 4))
        column.addWidget(self.message)
        column.addLayout(row)
        column.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addSpacing(40)
        layout.addLayout(_centred(content), 1)

    def set_message(self, text: str) -> None:
        """Explain why no board is loaded.

        Args:
            text: The explanation.
        """
        self.message.setText(text)
