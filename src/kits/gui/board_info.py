"""Dock panel summarising the loaded board and its stackup."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QFormLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import MAX_LAYER_THICKNESS_MM, ConfigError
from ..ingest import Project
from ..ingest.model import COPPER_MM_PER_OZ, format_copper_weight, is_copper_layer

_STACKUP_COLUMNS = ["Layer", "Type", "Thickness (mm)", "Weight", "Material", "εr"]
_THICKNESS_COLUMN, _WEIGHT_COLUMN = 2, 3
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
    """Summary of the board: file, size, object counts and stackup table.

    Copper weights and dielectric thicknesses in the stackup table can be
    edited (when given a document); edits need a new solve, not a new mesh.
    """

    def __init__(self, document=None, parent=None):
        """Build the summary form and empty stackup table.

        Args:
            document: The application :class:`~kits.gui.document.Document`, for editing
                layer thicknesses; None for a read-only panel.
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self._document = document
        self._editing = False
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
        self._edits_note = QLabel()
        self._edits_note.setWordWrap(True)
        self._edits_note.hide()
        self.reset_button = QPushButton("Reset to board values")
        self.reset_button.setToolTip("Drop the edited thicknesses and use the board file's stackup")
        self.reset_button.hide()
        layout.addWidget(self._edits_note)
        layout.addWidget(self.reset_button)
        if document is not None:
            self.reset_button.clicked.connect(document.reset_layer_thicknesses)
            document.stackup_changed.connect(self._refresh)

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
            self._edits_note.hide()
            self.reset_button.hide()
            return

        b = self._document.board if self._document is not None and self._document.board else project.board
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
        """Fill the stackup table, with editable copper weights and dielectric thicknesses.

        Copper layers the board's stackup doesn't describe (e.g. a board whose
        Physical Stackup was never set up in KiCad) are listed with the default
        thickness the solver assumes, marked "assumed". Edited values are shown
        in bold with the board file's value in their tooltip.

        Args:
            board: The loaded board (with any thickness edits applied).
        """
        if self._editing:  # a spin box in the table is being edited: keep it
            return
        doc = self._document
        edits = doc.layer_thickness_edits if doc is not None else {}
        copper = board.copper_thicknesses()
        rows = []
        for layer in board.stackup:
            tip = ""
            if is_copper_layer(layer.name) and layer.thickness_mm:
                microns, oz = layer.thickness_mm * 1000, layer.thickness_mm / COPPER_MM_PER_OZ
                tip = f"{microns:.3g} µm ({oz:.3g} oz at 35 µm/oz)"
            kind = layer.type if layer.sublayers == 1 else f"{layer.type} ×{layer.sublayers}"
            if layer.sublayers > 1:
                tip = f"{layer.sublayers} sheets; thickness is their total"
            rows.append(
                (layer.name, kind, layer.thickness_mm, layer.material or "", layer.epsilon_r, tip, False)
            )
        assumed = [name for name, (_t, from_stackup) in copper.items() if not from_stackup]
        overridden = set(board.copper_thickness_overrides)
        for name in [n for n in copper if n in overridden or n in assumed]:
            tip = "Not in the board's stackup; the solver assumes this thickness unless you set it"
            rows.append((name, "copper (assumed)", copper[name][0], "", None, tip, name not in overridden))

        self._stackup.setRowCount(len(rows))
        for row, (name, kind, thickness, material, epsilon_r, tip, is_assumed) in enumerate(rows):
            original = doc.file_layer_thickness(name) if doc is not None else None
            edited = name in edits
            is_copper = is_copper_layer(name)
            weight = format_copper_weight(thickness) if is_copper and thickness else ""
            cells = [name, kind, _fmt(thickness), weight, material, _fmt(epsilon_r)]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if tip:
                    item.setToolTip(tip)
                if is_assumed:
                    item.setForeground(Qt.GlobalColor.darkYellow)
                self._stackup.setItem(row, col, item)
            self._stackup.removeCellWidget(row, _THICKNESS_COLUMN)
            self._stackup.removeCellWidget(row, _WEIGHT_COLUMN)
            if doc is None or thickness is None or original is None:
                continue
            if is_copper:
                editor = self._spin(
                    thickness / COPPER_MM_PER_OZ, " oz", 2, 0.05, MAX_LAYER_THICKNESS_MM / COPPER_MM_PER_OZ
                )
                editor.valueChanged.connect(lambda oz, n=name: self._on_edited(n, oz * COPPER_MM_PER_OZ))
                column, was = _WEIGHT_COLUMN, f"{format_copper_weight(original)} ({original * 1000:.3g} µm)"
            elif kind.startswith(("core", "prepreg")) or "dielectric" in name.lower():
                editor = self._spin(thickness, " mm", 3, 0.001, MAX_LAYER_THICKNESS_MM)
                editor.valueChanged.connect(lambda mm, n=name: self._on_edited(n, mm))
                column, was = _THICKNESS_COLUMN, f"{original:g} mm"
            else:
                continue  # solder mask, silkscreen...: not used by the solver
            if edited:
                font = editor.font()
                font.setBold(True)
                editor.setFont(font)
                editor.setToolTip(f"Edited — the board file says {was}. Re-solve to apply.")
            else:
                editor.setToolTip("Edit to override the board file (needs a new solve, not a new mesh)")
            self._stackup.item(row, column).setText("")  # the editor shows the value
            self._stackup.setCellWidget(row, column, editor)

        if assumed:
            which = "this board's copper" if len(assumed) == len(copper) else ", ".join(assumed)
            thickness = copper[assumed[0]][0]
            self._stackup_note.setText(
                f"The board file has no stackup thickness for {which}: assuming "
                f"{format_copper_weight(thickness)} ({thickness * 1000:g} µm). "
                "Set it here or in KiCad under Board Setup → Physical Stackup."
            )
            self._stackup_note.show()
        else:
            self._stackup_note.hide()
        if edits:
            plural = "es" if len(edits) > 1 else ""
            note = (
                f"{len(edits)} layer thickness{plural} edited (bold); re-solve to apply. "
                "Edits are saved with the project and reset when the board is re-loaded."
            )
        else:
            note = (doc.stackup_notice if doc is not None else None) or ""
        self._edits_note.setText(note)
        self._edits_note.setVisible(bool(note))
        self.reset_button.setVisible(bool(edits))

        weights = list(dict.fromkeys(format_copper_weight(t) for t, _ in copper.values()))
        self._labels["Copper layers"].setText(f"{len(copper)} ({', '.join(weights)})")
        self._labels["Copper layers"].setToolTip(
            "\n".join(
                f"{name}: {format_copper_weight(t)}, {t * 1000:.3g} µm" for name, (t, _) in copper.items()
            )
        )

    @staticmethod
    def _spin(value: float, suffix: str, decimals: int, low: float, high: float) -> QDoubleSpinBox:
        """A compact spin box for a table cell.

        Args:
            value: Initial value.
            suffix: Unit suffix, e.g. ``" oz"``.
            decimals: Decimal places.
            low: Minimum.
            high: Maximum.

        Returns:
            The spin box (commits on Enter or focus loss).
        """
        spin = QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(low, high)
        spin.setSuffix(suffix)
        spin.setKeyboardTracking(False)
        spin.setFrame(False)
        spin.setValue(value)
        return spin

    def _on_edited(self, layer: str, thickness_mm: float) -> None:
        """Store an edited layer thickness.

        Args:
            layer: Layer name.
            thickness_mm: New thickness in millimetres.
        """
        self._editing = True
        try:
            self._document.set_layer_thickness(layer, thickness_mm)
        except ConfigError:
            pass
        finally:
            self._editing = False
        self._refresh()

    def _refresh(self, *_args) -> None:
        """Redraw the stackup table from the document.

        Args:
            *_args: Ignored signal arguments.
        """
        if self._document is not None and self._document.board is not None:
            self._fill_stackup(self._document.board)
