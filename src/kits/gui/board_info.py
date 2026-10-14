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
from ..ingest.model import COPPER_MM_PER_OZ, format_copper_weight, is_copper_layer

_STACKUP_COLUMNS = ["Layer", "Type", "Thickness (mm)", "Weight", "Material", "εr"]
_WARNING_COLOR = "#e0a030"
_WARNING_STYLE = f"color: {_WARNING_COLOR};"


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
        self._stackup_note = QLabel()
        self._stackup_note.setWordWrap(True)
        self._stackup_note.setStyleSheet(_WARNING_STYLE)
        self._stackup_note.hide()

        layout.addWidget(QLabel("Stackup:"))
        layout.addWidget(self._stackup, 1)
        layout.addWidget(self._stackup_note)

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
            self._stackup_note.hide()
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

        self._fill_stackup(b)

    def _fill_stackup(self, board) -> None:
        """Fill the stackup table, with copper weights, and flag assumed copper thickness.

        Copper layers the board's stackup doesn't describe (e.g. a board whose
        Physical Stackup was never set up in KiCad) are listed with the default
        thickness the solver assumes, marked "assumed".

        Args:
            board: The loaded board.
        """
        copper = board.copper_thicknesses()
        rows = []
        for layer in board.stackup:
            weight, tip = "", ""
            if is_copper_layer(layer.name) and layer.thickness_mm:
                weight = format_copper_weight(layer.thickness_mm)
                microns, oz = layer.thickness_mm * 1000, layer.thickness_mm / COPPER_MM_PER_OZ
                tip = f"{microns:.3g} µm ({oz:.3g} oz at 35 µm/oz)"
            kind = layer.type if layer.sublayers == 1 else f"{layer.type} ×{layer.sublayers}"
            if layer.sublayers > 1:
                tip = f"{layer.sublayers} sheets; thickness is their total"
            cells = [
                layer.name,
                kind,
                _fmt(layer.thickness_mm),
                weight,
                layer.material or "",
                _fmt(layer.epsilon_r),
            ]
            rows.append((cells, tip, False))
        assumed = [name for name, (_t, from_stackup) in copper.items() if not from_stackup]
        for name in assumed:
            thickness = copper[name][0]
            cells = [name, "copper (assumed)", _fmt(thickness), format_copper_weight(thickness), "", ""]
            rows.append((cells, "Not in the board's stackup; the solver assumes this thickness", True))

        self._stackup.setRowCount(len(rows))
        for row, (cells, tip, is_assumed) in enumerate(rows):
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if tip:
                    item.setToolTip(tip)
                if is_assumed:
                    item.setForeground(Qt.GlobalColor.darkYellow)
                self._stackup.setItem(row, col, item)

        if assumed:
            which = "this board's copper" if len(assumed) == len(copper) else ", ".join(assumed)
            thickness = copper[assumed[0]][0]
            self._stackup_note.setText(
                f"The board file has no stackup thickness for {which}: assuming "
                f"{format_copper_weight(thickness)} ({thickness * 1000:g} µm). "
                "Set it in KiCad under Board Setup → Physical Stackup."
            )
            self._stackup_note.show()
        else:
            self._stackup_note.hide()

        weights = list(dict.fromkeys(format_copper_weight(t) for t, _ in copper.values()))
        self._labels["Copper layers"].setText(f"{len(copper)} ({', '.join(weights)})")
        self._labels["Copper layers"].setToolTip(
            "\n".join(
                f"{name}: {format_copper_weight(t)}, {t * 1000:.3g} µm" for name, (t, _) in copper.items()
            )
        )
