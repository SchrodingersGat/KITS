"""Dock panel listing display layers with visibility checkboxes, plus overlay toggles.

Selecting a copper layer makes it the active layer (drawn on top, and the layer
whose mesh and results overlays are shown).
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

_NAME_ROLE = Qt.ItemDataRole.UserRole


def swatch(color: QColor) -> QIcon:
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

        self.overlay_boxes: dict[str, QCheckBox] = {}
        self._overlay_layout = QVBoxLayout()
        overlays = QGroupBox("Overlays")
        overlays.setLayout(self._overlay_layout)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._list)
        layout.addLayout(buttons)
        layout.addWidget(overlays)

    def add_overlay_toggle(self, label: str, action: QAction) -> QCheckBox:
        """Add a checkbox mirroring a checkable action (e.g. "Show mesh").

        The checkbox and the action stay in sync in both directions, so menus,
        shortcuts and other panels' checkboxes all agree.

        Args:
            label: Checkbox text.
            action: The checkable action it mirrors.

        Returns:
            The new checkbox.
        """
        box = QCheckBox(label)
        box.setChecked(action.isChecked())
        shortcut = action.shortcut().toString()
        box.setToolTip(
            f"{action.text().replace('&', '')} ({shortcut})" if shortcut else action.text().replace("&", "")
        )
        box.toggled.connect(action.setChecked)
        action.toggled.connect(box.setChecked)
        self._overlay_layout.addWidget(box)
        self.overlay_boxes[label] = box
        return box

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
            item = QListWidgetItem(swatch(color), name)
            item.setData(_NAME_ROLE, name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self._list.addItem(item)
        self._list.blockSignals(False)

    def select_layer(self, name: str) -> None:
        """Select a copper layer in the list, making it the active layer.

        Emits ``active_layer_changed`` if the selection changes.

        Args:
            name: KiCad layer name; ignored unless it is a listed copper layer.
        """
        if name not in self._copper:
            return
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item.data(_NAME_ROLE) == name:
                self._list.setCurrentItem(item)
                return

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
