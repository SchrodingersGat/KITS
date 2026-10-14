"""Dock panel listing display layers with visibility checkboxes.

Selecting a copper layer makes it the active layer (drawn on top).
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

_NAME_ROLE = Qt.ItemDataRole.UserRole


def _swatch(color: QColor) -> QIcon:
    """Make a small solid-colour icon for a layer list entry.

    Args:
        color: Fill colour of the swatch.

    Returns:
        A 14x14 pixel icon filled with ``color``.
    """
    pix = QPixmap(14, 14)
    pix.fill(color)
    return QIcon(pix)


class LayerPanel(QWidget):
    """Checkable layer list emitting ``visibility_changed`` and ``active_layer_changed``."""

    visibility_changed = Signal(str, bool)
    active_layer_changed = Signal(str)

    def __init__(self, parent=None):
        """Build the empty layer list and the show/hide-all buttons.

        Args:
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self._list = QListWidget()
        self._list.itemChanged.connect(self._on_item_changed)
        self._list.currentItemChanged.connect(self._on_current_changed)
        self._copper: set[str] = set()

        show_all = QPushButton("Show all")
        hide_all = QPushButton("Hide all")
        show_all.clicked.connect(lambda: self._set_all(True))
        hide_all.clicked.connect(lambda: self._set_all(False))

        buttons = QHBoxLayout()
        buttons.addWidget(show_all)
        buttons.addWidget(hide_all)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._list)
        layout.addLayout(buttons)

    def set_layers(self, layers: list[tuple[str, QColor, bool]]) -> None:
        """Populate from ``(name, colour, is_copper)`` tuples, all visible.

        No signals are emitted while the list is rebuilt.

        Args:
            layers: Layers in display order; an empty list clears the panel.
        """
        self._list.blockSignals(True)
        self._list.clear()
        self._copper = {name for name, _, is_copper in layers if is_copper}
        for name, color, _ in layers:
            item = QListWidgetItem(_swatch(color), name)
            item.setData(_NAME_ROLE, name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self._list.addItem(item)
        self._list.blockSignals(False)

    def _set_all(self, visible: bool) -> None:
        """Check or uncheck every layer.

        Args:
            visible: True to show all layers, False to hide them.
        """
        state = Qt.CheckState.Checked if visible else Qt.CheckState.Unchecked
        for row in range(self._list.count()):
            self._list.item(row).setCheckState(state)

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        """Emit ``visibility_changed`` for a toggled layer checkbox.

        Args:
            item: The list item whose check state changed.
        """
        self.visibility_changed.emit(item.data(_NAME_ROLE), item.checkState() == Qt.CheckState.Checked)

    def _on_current_changed(self, current: QListWidgetItem | None, _previous) -> None:
        """Emit ``active_layer_changed`` when a copper layer is selected.

        Args:
            current: The newly selected item, or ``None``.
            _previous: The previously selected item (unused).
        """
        if current is not None and current.data(_NAME_ROLE) in self._copper:
            self.active_layer_changed.emit(current.data(_NAME_ROLE))
