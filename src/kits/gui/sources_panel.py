"""Dock panel listing the project's current sources, with add / edit / remove."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
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

from ..config import ConfigError, CurrentSource
from .document import Document

DEFAULT_CURRENT_A = 1.0
MAX_CURRENT_A = 10_000.0
_BALANCE_TOLERANCE_A = 1e-9
_WARNING_STYLE = "color: #e0a030;"
_COLUMNS = ["Pad", "Net", "Current"]
_KEY_ROLE = Qt.ItemDataRole.UserRole


class SourcesPanel(QWidget):
    """Edits the current fed into (or drawn from) the board at pads.

    Positive currents flow into the board (sources), negative ones out of it
    (sinks). For a solve they must sum to zero; the panel warns otherwise.

    Signals:
        pad_selected(object): A row was selected; carries its ``(ref, pad)`` key, or None.
    """

    pad_selected = Signal(object)

    def __init__(self, document: Document, parent=None):
        """Create the panel.

        Args:
            document: The application document holding the project's sources.
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
            "Add a current source on the pad selected on the board (or double-click a pad)"
        )
        self.add_button.clicked.connect(self.add_selected_pad)
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self.remove_selected_rows)

        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        self.table.horizontalHeaderItem(2).setToolTip("Amperes into the board; negative for a sink")
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._on_row_selection)
        QShortcut(QKeySequence.StandardKey.Delete, self.table, self.remove_selected_rows)

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

        document.project_changed.connect(self._on_project_changed)
        document.board_changed.connect(self.refresh)
        document.sources_changed.connect(self.refresh)
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
        """Add a source on the selected pad, or select its existing row."""
        key = self._selected_pad
        if key is None or self._document.project is None:
            return
        if self._document.current_source(*key) is None:
            try:
                self._document.add_current_source(*key, DEFAULT_CURRENT_A)
            except ConfigError:
                return
        row = self._row_of(key)
        if row is not None:
            self.table.selectRow(row)
            spin = self.table.cellWidget(row, 2)
            spin.setFocus()
            spin.selectAll()

    def remove_selected_rows(self) -> None:
        """Remove the sources in the selected rows."""
        keys = [self._row_key(index.row()) for index in self.table.selectionModel().selectedRows()]
        for key in keys:
            if self._document.current_source(*key) is not None:
                self._document.remove_current_source(*key)

    # -- table ----------------------------------------------------------------------

    def refresh(self, *_args) -> None:
        """Rebuild the table from the document's sources.

        Args:
            *_args: Ignored signal arguments.
        """
        if self._editing:  # a spin box in the table is being edited: keep it
            self._update_total()
            return
        selected = {self._row_key(i.row()) for i in self.table.selectionModel().selectedRows()}
        board = self._document.board
        sources = self._document.current_sources
        self.table.blockSignals(True)
        self.table.setRowCount(len(sources))
        for row, source in enumerate(sources):
            pads = board.find_pads(source.ref, source.pad) if board is not None else []
            pad_item = QTableWidgetItem(source.label)
            pad_item.setData(_KEY_ROLE, source.key)
            nets = sorted({p.net for p in pads if p.net})
            net_item = QTableWidgetItem(", ".join(nets) if nets else ("—" if pads else "pad not found"))
            if board is not None and not pads:
                for item in (pad_item, net_item):
                    item.setForeground(Qt.GlobalColor.darkYellow)
                    item.setToolTip(f"{source.label} isn't on the board; it is ignored by the solver")
            self.table.setItem(row, 0, pad_item)
            self.table.setItem(row, 1, net_item)
            self.table.setCellWidget(row, 2, self._current_spinbox(source))
            if source.key in selected:
                self.table.selectRow(row)
        self.table.blockSignals(False)
        self._update_total()
        self._update_selection_ui()

    def _current_spinbox(self, source: CurrentSource) -> QDoubleSpinBox:
        """Create the editor for one source's current.

        Args:
            source: The source.

        Returns:
            A spin box that writes changes back to the document.
        """
        spin = QDoubleSpinBox()
        spin.setRange(-MAX_CURRENT_A, MAX_CURRENT_A)
        spin.setDecimals(3)
        spin.setSuffix(" A")
        spin.setKeyboardTracking(False)
        spin.setValue(source.current_a)
        spin.valueChanged.connect(lambda value, key=source.key: self._on_current_edited(key, value))
        return spin

    def _on_current_edited(self, key: tuple[str, str], value: float) -> None:
        """Store an edited current.

        Args:
            key: ``(ref, pad)`` of the edited source.
            value: The new current in amperes.
        """
        self._editing = True
        try:
            self._document.set_source_current(*key, value)
        except ConfigError:
            pass
        finally:
            self._editing = False

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
            The row's source key.
        """
        return tuple(self.table.item(row, 0).data(_KEY_ROLE))

    def _row_of(self, key: tuple[str, str] | None) -> int | None:
        """Find the row showing a pad's source.

        Args:
            key: ``(ref, pad)``, or None.

        Returns:
            The row index, or None if the pad has no source.
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
            text, can_add = "Click a pad on the board to select it; double-click to add a source.", False
        else:
            pads = board.find_pads(*key)
            nets = sorted({p.net for p in pads if p.net})
            net = f" (net {', '.join(nets)})" if nets else ""
            text = f"Selected: {key[0]} pad {key[1]}{net}"
            can_add = any(p.layers for p in pads)
            if doc.current_source(*key) is not None:
                text += " — has a source"
        self.selected_label.setText(text)
        self.add_button.setEnabled(can_add)
        self.remove_button.setEnabled(bool(self.table.selectionModel().selectedRows()))

    def _update_total(self) -> None:
        """Show the net current and warn if sources and sinks don't balance."""
        sources = self._document.current_sources
        if not sources:
            self.total_label.setStyleSheet("")
            self.total_label.setText("No current sources.")
            return
        total = sum(s.current_a for s in sources)
        into = sum(s.current_a for s in sources if s.current_a > 0)
        text = f"Total in: {into:g} A · net: {total:+g} A"
        if abs(total) > _BALANCE_TOLERANCE_A:
            self.total_label.setStyleSheet(_WARNING_STYLE)
            text += "\nSources and sinks must sum to zero to solve."
        else:
            self.total_label.setStyleSheet("")
        self.total_label.setText(text)
