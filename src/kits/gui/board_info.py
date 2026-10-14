"""Dock panel summarising the loaded board and its stackup."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFormLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..ingest import Project

_STACKUP_COLUMNS = ["Layer", "Type", "Thickness (mm)", "Material", "εr"]
_WARNING_COLOR = "#e0a030"


def _size(bbox) -> str:
    """Format a bounding box as ``"W × H mm"``.

    Args:
        bbox: ``(min_x, min_y, max_x, max_y)`` in millimetres.

    Returns:
        Width and height to two decimal places.
    """
    return f"{bbox[2] - bbox[0]:.2f} × {bbox[3] - bbox[1]:.2f} mm"


def _fmt(value, spec: str = "g") -> str:
    """Format an optional value for a table cell.

    Args:
        value: Value to format, or ``None``.
        spec: Format specification passed to :func:`format`.

    Returns:
        The formatted value, or an empty string for ``None``.
    """
    return "" if value is None else format(value, spec)


class BoardInfoPanel(QWidget):
    """Read-only summary of the board: file, size, object counts and stackup table."""

    def __init__(self, parent=None):
        """Build the summary form and empty stackup table.

        Args:
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self._labels: dict[str, QLabel] = {}
        form = QFormLayout()
        for key in (
            "File",
            "Format version",
            "Size",
            "Thickness",
            "Copper layers",
            "Footprints",
            "Pads",
            "Tracks",
            "Vias",
            "Zones",
            "Nets",
        ):
            label = QLabel("—")
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self._labels[key] = label
            form.addRow(key + ":", label)

        self._stackup = QTableWidget(0, len(_STACKUP_COLUMNS))
        self._stackup.setHorizontalHeaderLabels(_STACKUP_COLUMNS)
        self._stackup.verticalHeader().setVisible(False)
        self._stackup.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._stackup.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addLayout(form)
        layout.addWidget(QLabel("Stackup:"))
        layout.addWidget(self._stackup, 1)

    def set_project(self, project: Project | None) -> None:
        """Show the summary for ``project``, flagging a missing board outline.

        Args:
            project: The loaded project, or ``None`` to clear the panel.
        """
        if project is None:
            for label in self._labels.values():
                label.setText("—")
                label.setStyleSheet("")
                label.setToolTip("")
            self._stackup.setRowCount(0)
            return

        b = project.board
        outline = b.outline_bounding_box()
        values = {
            "File": b.path.name,
            "Format version": str(b.version),
            "Size": _size(outline) if outline else "No board outline",
            "Thickness": f"{b.thickness_mm:g} mm" if b.thickness_mm else "—",
            "Copper layers": str(len(b.copper_layers)),
            "Footprints": str(len(b.footprints)),
            "Pads": str(sum(1 for _ in b.pads())),
            "Tracks": f"{len(b.tracks)} segments, {len(b.arcs)} arcs",
            "Vias": str(len(b.vias)),
            "Zones": str(sum(1 for z in b.zones if not z.is_rule_area)),
            "Nets": str(len(b.nets)),
        }
        for key, text in values.items():
            self._labels[key].setText(text)
        self._labels["File"].setToolTip(str(b.path))

        size_label = self._labels["Size"]
        if outline:
            size_label.setStyleSheet("")
            size_label.setToolTip("Extent of the Edge.Cuts outline")
        else:
            copper = b.copper_bounding_box()
            size_label.setStyleSheet(f"color: {_WARNING_COLOR}; font-weight: bold;")
            size_label.setToolTip(
                "The board has nothing on the Edge.Cuts layer.\n"
                "Draw the board outline on Edge.Cuts in KiCad."
                + (f"\nCopper extent: {_size(copper)}" if copper else "")
            )

        self._stackup.setRowCount(len(b.stackup))
        for row, layer in enumerate(b.stackup):
            cells = [
                layer.name,
                layer.type,
                _fmt(layer.thickness_mm),
                layer.material or "",
                _fmt(layer.epsilon_r),
            ]
            for col, text in enumerate(cells):
                self._stackup.setItem(row, col, QTableWidgetItem(text))
