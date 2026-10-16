"""Mesh areas (refine / exclude rectangles): the editor dialog and the table in the Mesh panel."""

from __future__ import annotations

import dataclasses

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import DEFAULT_CORNER_SIZE_MM, ConfigError, MeshArea
from .document import Document
from .heatsinks_panel import MAX_COORD_MM, GridFrame, spin_box
from .sources_panel import make_enable_box

KIND_LABELS = {"refine": "Refine", "exclude": "Exclude"}
_COLUMNS = ["Name", "Kind", "Size", "Position (mm)", "Area (mm)"]
_NAME_ROLE = Qt.ItemDataRole.UserRole
_ERROR_STYLE = "color: #e05050;"


class MeshAreaDialog(QDialog):
    """Edits one mesh area: name, kind, rectangle (grid coordinates) and element size."""

    def __init__(
        self, area: MeshArea, taken_names: set[str], frame: GridFrame, max_size_mm: float, parent=None
    ):
        """Create the dialog, filled in from ``area``.

        Args:
            area: The area to edit (or a new one with suggested values).
            taken_names: Names other areas already use.
            frame: Converts between board and grid coordinates.
            max_size_mm: The mesh max size, the upper limit of a refine area's size.
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("Mesh area")
        self._taken = taken_names
        self._frame = frame
        self._max_size = max_size_mm
        self._enabled = area.enabled
        self._area: MeshArea | None = None

        gx, gy = frame.to_grid(area)
        self.name = QLineEdit(area.name)
        self.kind = QComboBox()
        self.kind.addItem("Refine: finer elements inside", "refine")
        self.kind.addItem("Exclude: don't mesh (cut out of every layer)", "exclude")
        self.kind.setCurrentIndex(self.kind.findData(area.kind))
        self.x = spin_box(gx, -MAX_COORD_MM, MAX_COORD_MM, " mm")
        self.y = spin_box(gy, -MAX_COORD_MM, MAX_COORD_MM, " mm")
        self.width = spin_box(area.width_mm, 0.01, MAX_COORD_MM, " mm")
        self.height = spin_box(area.height_mm, 0.01, MAX_COORD_MM, " mm")
        self.size = spin_box(area.size_mm, 0.001, MAX_COORD_MM, " mm")
        self.size.setSingleStep(0.05)
        self.size.setToolTip(
            "Element size inside the area, graded back to the max size outside it.\n"
            "May be finer than the mesh min size."
        )
        note = QLabel(
            "X, Y: lower-left corner in grid coordinates (as in the status bar). Areas apply to every "
            "copper layer. An excluded area is removed like a board cutout: no copper, no heat flow, "
            "and holes touching it are left out."
        )
        note.setWordWrap(True)
        note.setEnabled(False)

        form = QFormLayout()
        form.addRow("Name:", self.name)
        form.addRow("Kind:", self.kind)
        form.addRow("X:", self.x)
        form.addRow("Y:", self.y)
        form.addRow("Width:", self.width)
        form.addRow("Height:", self.height)
        form.addRow("Element size:", self.size)
        self._form = form
        self.error = QLabel()
        self.error.setStyleSheet(_ERROR_STYLE)
        self.error.setWordWrap(True)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addWidget(self.error)
        layout.addWidget(self.buttons)

        self.name.textChanged.connect(self._validate)
        self.kind.currentIndexChanged.connect(self._validate)
        for spin in (self.x, self.y, self.width, self.height, self.size):
            spin.valueChanged.connect(self._validate)
        self._validate()

    @property
    def area(self) -> MeshArea | None:
        """The area as entered, or None while the entries are invalid."""
        return self._area

    def _validate(self, *_args) -> None:
        """Rebuild the area from the entries and enable OK only if it is valid.

        Args:
            *_args: Ignored signal arguments.
        """
        self._area = None
        name = self.name.text().strip()
        kind = self.kind.currentData()
        self._form.setRowVisible(self.size, kind == "refine")
        x_mm, y_mm = self._frame.from_grid(self.x.value(), self.y.value(), self.height.value())
        area = MeshArea(
            name=name,
            kind=kind,
            x_mm=x_mm,
            y_mm=y_mm,
            width_mm=self.width.value(),
            height_mm=self.height.value(),
            size_mm=self.size.value(),
            enabled=self._enabled,
        )
        error = ""
        if name in self._taken:
            error = f"Another mesh area is already named {name!r}."
        elif kind == "refine" and area.size_mm > self._max_size:
            error = f"The element size can't exceed the mesh max size ({self._max_size:g} mm)."
        else:
            try:
                area.validate()
                self._area = area
            except ConfigError as exc:
                error = str(exc)
        self.error.setText(error)
        self.error.setVisible(bool(error))
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(self._area is not None)


class MeshAreasWidget(QWidget):
    """Lists the project's mesh areas, with draw / edit / remove and enable checkboxes.

    Signals:
        draw_requested(str): The user wants to draw an area of this kind on the board.
        show_areas_toggled(bool): The user toggled the areas on the board.
    """

    draw_requested = Signal(str)
    show_areas_toggled = Signal(bool)

    def __init__(self, document: Document, parent=None):
        """Create the widget.

        Args:
            document: The application document holding the project's mesh settings.
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self._document = document
        self.frame = GridFrame()

        self.refine_button = QPushButton("Draw refine area")
        self.refine_button.setToolTip("Drag a rectangle on the board to mesh more finely (Esc cancels)")
        self.refine_button.clicked.connect(lambda: self.draw_requested.emit("refine"))
        self.exclude_button = QPushButton("Draw exclusion")
        self.exclude_button.setToolTip("Drag a rectangle on the board to leave out of the mesh (Esc cancels)")
        self.exclude_button.clicked.connect(lambda: self.draw_requested.emit("exclude"))
        self.edit_button = QPushButton("Edit…")
        self.edit_button.clicked.connect(self.edit_selected)
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self.remove_selected_rows)

        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        for column in range(len(_COLUMNS)):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._update_buttons)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.cellDoubleClicked.connect(lambda _row, _col: self.edit_selected())
        QShortcut(QKeySequence.StandardKey.Delete, self.table, self.remove_selected_rows)

        self.show_areas = QCheckBox("Show mesh areas")
        self.show_areas.setChecked(True)
        self.show_areas.toggled.connect(self.show_areas_toggled)

        draw = QHBoxLayout()
        draw.addWidget(self.refine_button)
        draw.addWidget(self.exclude_button)
        edit = QHBoxLayout()
        edit.addWidget(self.edit_button)
        edit.addWidget(self.remove_button)
        edit.addStretch(1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(draw)
        layout.addWidget(self.table, 1)
        layout.addLayout(edit)
        layout.addWidget(self.show_areas)

        document.project_changed.connect(self.refresh)
        document.board_changed.connect(self.refresh)
        document.settings_changed.connect(self.refresh)
        self.refresh()

    @property
    def _areas(self) -> list[MeshArea]:
        """The open project's mesh areas."""
        project = self._document.project
        return list(project.mesh.areas) if project else []

    def _store(self, areas: list[MeshArea]) -> None:
        """Replace the project's mesh areas (in memory; save to persist).

        Args:
            areas: The new areas.

        Raises:
            ConfigError: If the resulting settings are invalid.
        """
        project = self._document.project
        if project is not None:
            self._document.set_mesh_settings(dataclasses.replace(project.mesh, areas=tuple(areas)))

    # -- dialogs (separate so tests can replace them) ------------------------------------

    def run_dialog(self, dialog: MeshAreaDialog) -> MeshArea | None:
        """Show an area dialog modally.

        Args:
            dialog: The dialog.

        Returns:
            The accepted area, or None if cancelled.
        """
        return dialog.area if dialog.exec() == QDialog.DialogCode.Accepted else None

    def _dialog(self, area: MeshArea, taken: set[str]) -> MeshAreaDialog:
        """Create the editor dialog for an area.

        Args:
            area: The area to show.
            taken: Names it may not use.

        Returns:
            The dialog.
        """
        project = self._document.project
        max_size = project.mesh.max_size_mm if project else area.size_mm
        return MeshAreaDialog(area, taken, self.frame, max_size, self)

    # -- actions ----------------------------------------------------------------------

    def _free_name(self, kind: str) -> str:
        """An area name not used yet: R1, R2, ... for refine areas, X1, X2, ... for exclusions.

        Args:
            kind: The area kind.

        Returns:
            The name.
        """
        prefix = "R" if kind == "refine" else "X"
        taken = {a.name for a in self._areas}
        n = 1
        while f"{prefix}{n}" in taken:
            n += 1
        return f"{prefix}{n}"

    def add_from_rect(self, kind: str, rect: QRectF) -> None:
        """Add an area over a rectangle (e.g. drawn on the board), confirmed in the dialog.

        Args:
            kind: ``"refine"`` or ``"exclude"``.
            rect: The rectangle in board coordinates.
        """
        project = self._document.project
        if project is None:
            return
        suggestion = MeshArea(
            name=self._free_name(kind),
            kind=kind,
            x_mm=round(rect.left(), 3),
            y_mm=round(rect.top(), 3),
            width_mm=max(round(rect.width(), 3), 0.01),
            height_mm=max(round(rect.height(), 3), 0.01),
            size_mm=min(DEFAULT_CORNER_SIZE_MM, project.mesh.max_size_mm),
        )
        area = self.run_dialog(self._dialog(suggestion, {a.name for a in self._areas}))
        if area is not None:
            try:
                self._store([*self._areas, area])
            except ConfigError:
                return
            self._select_name(area.name)

    def edit_selected(self) -> None:
        """Edit the selected area in the dialog."""
        rows = self.table.selectionModel().selectedRows()
        if len(rows) == 1:
            self.edit(self._row_name(rows[0].row()))

    def edit(self, name: str) -> None:
        """Edit an area in the dialog.

        Args:
            name: Area name.
        """
        areas = self._areas
        old = next((a for a in areas if a.name == name), None)
        if old is None:
            return
        area = self.run_dialog(self._dialog(old, {a.name for a in areas} - {name}))
        if area is not None and area != old:
            try:
                self._store([area if a.name == name else a for a in areas])
            except ConfigError:
                return
            self._select_name(area.name)

    def remove_selected_rows(self) -> None:
        """Remove the areas in the selected rows."""
        names = {self._row_name(index.row()) for index in self.table.selectionModel().selectedRows()}
        if names:
            self._store([a for a in self._areas if a.name not in names])

    # -- table ----------------------------------------------------------------------

    def refresh(self, *_args) -> None:
        """Rebuild the table from the project's mesh areas.

        Args:
            *_args: Ignored signal arguments.
        """
        selected = {self._row_name(i.row()) for i in self.table.selectionModel().selectedRows()}
        areas = self._areas
        self.table.blockSignals(True)
        self.table.setRowCount(len(areas))
        for row, area in enumerate(areas):
            gx, gy = self.frame.to_grid(area)
            cells = [
                area.name,
                KIND_LABELS[area.kind],
                f"{area.size_mm:g} mm" if area.kind == "refine" else "—",
                f"{gx:.2f}, {gy:.2f}",
                f"{area.width_mm:g} × {area.height_mm:g}",
            ]
            items = []
            for text in cells:
                item = QTableWidgetItem(text)
                item.setData(_NAME_ROLE, area.name)
                items.append(item)
            make_enable_box(items[0], area.enabled, *items[1:])
            for column, item in enumerate(items):
                self.table.setItem(row, column, item)
            if area.name in selected:
                self.table.selectRow(row)
        self.table.blockSignals(False)
        can_draw = self._document.project is not None and self._document.board is not None
        self.refine_button.setEnabled(can_draw)
        self.exclude_button.setEnabled(can_draw)
        self._update_buttons()

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """Enable or disable an area when its checkbox is toggled.

        Args:
            item: The changed cell.
        """
        if item.column() != 0:
            return
        name, on = self._row_name(item.row()), item.checkState() == Qt.CheckState.Checked
        self._store([dataclasses.replace(a, enabled=on) if a.name == name else a for a in self._areas])

    def _row_name(self, row: int) -> str:
        """Area name of a table row.

        Args:
            row: Row index.

        Returns:
            The name.
        """
        return self.table.item(row, 0).data(_NAME_ROLE)

    def _select_name(self, name: str) -> None:
        """Select an area's row.

        Args:
            name: Area name.
        """
        for row in range(self.table.rowCount()):
            if self._row_name(row) == name:
                self.table.selectRow(row)
                return

    def _update_buttons(self) -> None:
        """Enable Edit / Remove for the selection."""
        rows = self.table.selectionModel().selectedRows()
        self.edit_button.setEnabled(len(rows) == 1)
        self.remove_button.setEnabled(bool(rows))
