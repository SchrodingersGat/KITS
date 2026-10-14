"""Dock panel listing the project's heat loads (pad heat sources and sinks), with add / edit / remove."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import ABSOLUTE_ZERO_C, ConfigError, HeatLoad, HeatLoadKind
from .document import Document
from .sources_panel import make_enable_box

DEFAULT_POWER_W = 1.0
DEFAULT_RESISTANCE_K_W = 10.0
MAX_POWER_W = 10_000.0
MAX_TEMPERATURE_C = 10_000.0
MIN_RESISTANCE_K_W = 0.001
MAX_RESISTANCE_K_W = 1e6
_COLUMNS = ["Pad", "Net", "Type", "Value"]
_TYPE_COLUMN, _VALUE_COLUMN = 2, 3
_KEY_ROLE = Qt.ItemDataRole.UserRole


class HeatLoadsPanel(QWidget):
    """Edits heat sources and sinks on pads.

    Each pad can carry one heat load: a power source (W, spread over the pad's
    copper), a fixed temperature (°C), or a thermal resistance to ambient (K/W,
    e.g. a heatsink). Loads act on the pad's copper on all of its layers.

    Signals:
        pad_selected(object): A row was selected; carries its ``(ref, pad)`` key, or None.
        show_heat_loads_toggled(bool): The user toggled the heat load markers on the board.
    """

    pad_selected = Signal(object)
    show_heat_loads_toggled = Signal(bool)

    def __init__(self, document: Document, parent=None):
        """Create the panel.

        Args:
            document: The application document holding the project's heat loads.
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self._document = document
        self._selected_pad: tuple[str, str] | None = None
        self._editing = False

        self.selected_label = QLabel()
        self.selected_label.setWordWrap(True)
        self.add_button = QPushButton("Add selected pad")
        self.add_button.setToolTip(
            "Add a heat load on the pad selected on the board (or double-click a pad while this tab is shown)"
        )
        self.add_button.clicked.connect(self.add_selected_pad)
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self.remove_selected_rows)

        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        self.table.horizontalHeaderItem(_TYPE_COLUMN).setToolTip(
            "Power: heat injected over the pad (negative extracts it).\n"
            "Fixed temperature: the pad is held at a temperature.\n"
            "Thermal resistance: the pad is tied to ambient, e.g. through a heatsink."
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(_TYPE_COLUMN, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(_VALUE_COLUMN, QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._on_row_selection)
        self.table.itemChanged.connect(self._on_item_changed)
        QShortcut(QKeySequence.StandardKey.Delete, self.table, self.remove_selected_rows)

        self.show_heat_loads = QCheckBox("Show heat loads")
        self.show_heat_loads.setChecked(True)
        self.show_heat_loads.setToolTip("Show the heat load outlines and labels on the board (H)")
        self.show_heat_loads.toggled.connect(self.show_heat_loads_toggled)
        self.total_label = QLabel()
        self.total_label.setWordWrap(True)

        buttons = QHBoxLayout()
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.remove_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.selected_label)
        layout.addLayout(buttons)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.total_label)
        layout.addWidget(self.show_heat_loads)

        document.project_changed.connect(self._on_project_changed)
        document.board_changed.connect(self.refresh)
        document.heat_loads_changed.connect(self.refresh)
        self.refresh()

    # -- selection ------------------------------------------------------------------

    def set_selected_pad(self, key: tuple[str, str] | None) -> None:
        """Show which pad is selected on the board and select its row, if it has one.

        Args:
            key: ``(ref, pad)`` of the selected pad, or None.
        """
        self._selected_pad = tuple(key) if key is not None else None
        row = self._row_of(self._selected_pad)
        self.table.blockSignals(True)
        if row is None:
            self.table.clearSelection()
        else:
            self.table.selectRow(row)
        self.table.blockSignals(False)
        self._update_selection_ui()

    def add_selected_pad(self) -> None:
        """Add a power source on the selected pad, or select its existing row."""
        key = self._selected_pad
        if key is None or self._document.project is None:
            return
        if self._document.heat_load(*key) is None:
            try:
                self._document.add_heat_load(*key, HeatLoadKind.POWER, DEFAULT_POWER_W)
            except ConfigError:
                return
        row = self._row_of(key)
        if row is not None:
            self.table.selectRow(row)
            spin = self.table.cellWidget(row, _VALUE_COLUMN)
            spin.setFocus()
            spin.selectAll()

    def remove_selected_rows(self) -> None:
        """Remove the heat loads in the selected rows."""
        keys = [self._row_key(index.row()) for index in self.table.selectionModel().selectedRows()]
        for key in keys:
            if self._document.heat_load(*key) is not None:
                self._document.remove_heat_load(*key)

    # -- table ----------------------------------------------------------------------

    def refresh(self, *_args) -> None:
        """Rebuild the table from the document's heat loads.

        Args:
            *_args: Ignored signal arguments.
        """
        if self._editing:  # an editor in the table is being used: keep it
            self._update_total()
            return
        selected = {self._row_key(i.row()) for i in self.table.selectionModel().selectedRows()}
        board = self._document.board
        loads = self._document.heat_loads
        self.table.blockSignals(True)
        # Replaced cell widgets are only deleted once control returns to the event
        # loop; until then they'd still be drawn (at the table's left edge). Hide them.
        for row in range(self.table.rowCount()):
            for col in (_TYPE_COLUMN, _VALUE_COLUMN):
                old = self.table.cellWidget(row, col)
                if old is not None:
                    self.table.removeCellWidget(row, col)
                    old.hide()
        self.table.setRowCount(len(loads))
        for row, load in enumerate(loads):
            pads = board.find_pads(load.ref, load.pad) if board is not None else []
            pad_item = QTableWidgetItem(load.label)
            pad_item.setData(_KEY_ROLE, load.key)
            nets = sorted({p.net for p in pads if p.net})
            net_item = QTableWidgetItem(", ".join(nets) if nets else ("—" if pads else "pad not found"))
            if board is not None and not pads:
                for item in (pad_item, net_item):
                    item.setForeground(Qt.GlobalColor.darkYellow)
                    item.setToolTip(
                        f"{load.label} isn't on the board; the solve will fail until it is removed"
                    )
            make_enable_box(pad_item, load.enabled, net_item)
            self.table.setItem(row, 0, pad_item)
            self.table.setItem(row, 1, net_item)
            self.table.setCellWidget(row, _TYPE_COLUMN, self._kind_combo(load))
            self.table.setCellWidget(row, _VALUE_COLUMN, self._value_spinbox(load))
            if load.key in selected:
                self.table.selectRow(row)
        self.table.blockSignals(False)
        self._update_total()
        self._update_selection_ui()

    def _kind_combo(self, load: HeatLoad) -> QComboBox:
        """Create the editor for one heat load's kind.

        Args:
            load: The heat load.

        Returns:
            A combo box that writes changes back to the document.
        """
        combo = QComboBox()
        for kind in HeatLoadKind:
            combo.addItem(kind.title, kind)
        combo.setCurrentIndex(combo.findData(load.kind))
        combo.currentIndexChanged.connect(
            lambda _i, key=load.key, combo=combo: self._on_kind_edited(key, combo.currentData())
        )
        return combo

    def _value_spinbox(self, load: HeatLoad) -> QDoubleSpinBox:
        """Create the editor for one heat load's value, with the kind's unit and range.

        Args:
            load: The heat load.

        Returns:
            A spin box that writes changes back to the document.
        """
        spin = QDoubleSpinBox()
        spin.setDecimals(3)
        if load.kind is HeatLoadKind.POWER:
            spin.setRange(-MAX_POWER_W, MAX_POWER_W)
        elif load.kind is HeatLoadKind.TEMPERATURE:
            spin.setRange(ABSOLUTE_ZERO_C + 0.01, MAX_TEMPERATURE_C)
        else:
            spin.setRange(MIN_RESISTANCE_K_W, MAX_RESISTANCE_K_W)
        spin.setSuffix(f" {load.kind.unit}")
        spin.setKeyboardTracking(False)
        spin.setValue(load.value)
        spin.valueChanged.connect(
            lambda value, key=load.key, kind=load.kind: self._on_value_edited(key, kind, value)
        )
        return spin

    def default_value(self, kind: HeatLoadKind) -> float:
        """Starting value when a heat load is switched to a kind.

        Args:
            kind: The new kind.

        Returns:
            1 W, the project's ambient temperature, or 10 K/W.
        """
        if kind is HeatLoadKind.TEMPERATURE:
            return self._document.solve_settings.ambient_c
        return DEFAULT_RESISTANCE_K_W if kind is HeatLoadKind.RESISTANCE else DEFAULT_POWER_W

    def _on_kind_edited(self, key: tuple[str, str], kind: HeatLoadKind) -> None:
        """Change a heat load's kind, starting from that kind's default value.

        Args:
            key: ``(ref, pad)`` of the edited load.
            kind: The new kind.
        """
        load = self._document.heat_load(*key)
        if load is None or load.kind is kind:
            return
        try:
            self._document.set_heat_load(*key, kind, self.default_value(kind))
        except ConfigError:
            pass

    def _on_value_edited(self, key: tuple[str, str], kind: HeatLoadKind, value: float) -> None:
        """Store an edited value.

        Args:
            key: ``(ref, pad)`` of the edited load.
            kind: The load's kind (unchanged).
            value: The new value.
        """
        self._editing = True
        try:
            self._document.set_heat_load(*key, kind, value)
        except ConfigError:
            pass
        finally:
            self._editing = False

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """Enable or disable a heat load when its checkbox is toggled.

        Args:
            item: The changed cell.
        """
        if item.column() == 0:
            on = item.checkState() == Qt.CheckState.Checked
            self._document.set_heat_load_enabled(*self._row_key(item.row()), on)

    def _on_row_selection(self) -> None:
        """Report the selected row's pad so the board can highlight it."""
        rows = self.table.selectionModel().selectedRows()
        key = self._row_key(rows[0].row()) if len(rows) == 1 else None
        if key is not None:
            self._selected_pad = key
        self._update_selection_ui()
        self.pad_selected.emit(key)

    def _row_key(self, row: int) -> tuple[str, str]:
        """``(ref, pad)`` of a table row.

        Args:
            row: Row index.

        Returns:
            The row's heat load key.
        """
        return tuple(self.table.item(row, 0).data(_KEY_ROLE))

    def _row_of(self, key: tuple[str, str] | None) -> int | None:
        """Find the row showing a pad's heat load.

        Args:
            key: ``(ref, pad)``, or None.

        Returns:
            The row index, or None if the pad has no heat load.
        """
        if key is None:
            return None
        for row in range(self.table.rowCount()):
            if self._row_key(row) == tuple(key):
                return row
        return None

    # -- status -------------------------------------------------------------------------

    def _on_project_changed(self, *_args) -> None:
        """Clear the selection when a project is opened or closed.

        Args:
            *_args: Ignored signal arguments.
        """
        self._selected_pad = None
        self.refresh()

    def _update_selection_ui(self) -> None:
        """Update the selected-pad text and button states."""
        doc = self._document
        key = self._selected_pad
        board = doc.board
        if doc.project is None:
            text, can_add = "No project open.", False
        elif board is None:
            text, can_add = "No board loaded.", False
        elif key is None:
            text, can_add = "Click a pad on the board to select it; double-click to add a heat load.", False
        else:
            pads = board.find_pads(*key)
            nets = sorted({p.net for p in pads if p.net})
            net = f" (net {', '.join(nets)})" if nets else ""
            text = f"Selected: {key[0]} pad {key[1]}{net}"
            can_add = any(p.layers for p in pads)
            if doc.heat_load(*key) is not None:
                text += " — has a heat load"
        self.selected_label.setText(text)
        self.add_button.setEnabled(can_add)
        self.remove_button.setEnabled(bool(self.table.selectionModel().selectedRows()))

    def _update_total(self) -> None:
        """Summarise the heat loads: injected power and number of sinks."""
        everything = self._document.heat_loads
        loads = [h for h in everything if h.enabled]
        disabled = len(everything) - len(loads)
        if not loads:
            self.total_label.setText(
                "No heat loads. Joule heating from the current sources is always included."
            )
            return
        power = sum(h.value for h in loads if h.kind is HeatLoadKind.POWER)
        sinks = sum(h.kind.is_sink for h in loads)
        note = f" · {disabled} disabled" if disabled else ""
        self.total_label.setText(f"Power in: {power:g} W · sinks: {sinks}{note}")
