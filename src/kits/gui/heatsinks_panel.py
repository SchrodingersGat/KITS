"""Dock panel and editor dialog for manual heatsinks (rectangles on a board face)."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
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

from ..config import ConfigError, Heatsink
from .document import Document
from .sources_panel import make_enable_box

DEFAULT_SIZE_MM = 10.0
DEFAULT_BOARD_K_W = 1.0
DEFAULT_AMBIENT_K_W = 10.0
MAX_COORD_MM = 10_000.0
_COLUMNS = ["Name", "Side", "Mount", "Position (mm)", "Size (mm)", "Rθ contact", "Rθ ambient"]
_NAME_ROLE = Qt.ItemDataRole.UserRole
_ERROR_STYLE = "color: #e05050;"


class GridFrame:
    """Converts heatsink rectangles between board coordinates and grid coordinates.

    Board (KiCad) coordinates have Y pointing down; grid coordinates are relative
    to the grid origin (the board's bottom-left) with Y up, as in the status bar.
    A heatsink is shown by its lower-left corner in grid coordinates.
    """

    def __init__(self, origin: tuple[float, float] = (0.0, 0.0)):
        """Create the frame.

        Args:
            origin: Grid origin in board coordinates.
        """
        self.origin = origin

    def to_grid(self, heatsink: Heatsink) -> tuple[float, float]:
        """A heatsink's lower-left corner in grid coordinates.

        Args:
            heatsink: The heatsink.

        Returns:
            ``(x, y)`` in grid millimetres.
        """
        ox, oy = self.origin
        return heatsink.x_mm - ox, oy - (heatsink.y_mm + heatsink.height_mm)

    def from_grid(self, x: float, y: float, height: float) -> tuple[float, float]:
        """Board coordinates of a rectangle's top-left corner from its grid lower-left corner.

        Args:
            x: Grid X of the lower-left corner.
            y: Grid Y of the lower-left corner.
            height: Rectangle height.

        Returns:
            ``(x_mm, y_mm)`` for :class:`~kits.config.Heatsink`.
        """
        ox, oy = self.origin
        return round(x + ox, 6), round(oy - y - height, 6)  # no float noise in the project file


def spin_box(value: float, low: float, high: float, suffix: str, decimals: int = 3) -> QDoubleSpinBox:
    """A spin box for the dialog.

    Args:
        value: Initial value.
        low: Minimum.
        high: Maximum.
        suffix: Unit suffix.
        decimals: Decimal places.

    Returns:
        The spin box.
    """
    spin = QDoubleSpinBox()
    spin.setRange(low, high)
    spin.setDecimals(decimals)
    spin.setSuffix(suffix)
    spin.setValue(value)
    return spin


class HeatsinkDialog(QDialog):
    """Edits one heatsink: name, side, rectangle (grid coordinates) and thermal resistances."""

    def __init__(self, heatsink: Heatsink, taken_names: set[str], frame: GridFrame, parent=None):
        """Create the dialog, filled in from ``heatsink``.

        Args:
            heatsink: The heatsink to edit (or a new one with suggested values).
            taken_names: Names other heatsinks already use.
            frame: Converts between board and grid coordinates.
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("Heatsink")
        self._taken = taken_names
        self._frame = frame
        self._enabled = heatsink.enabled
        self._heatsink: Heatsink | None = None

        gx, gy = frame.to_grid(heatsink)
        self.name = QLineEdit(heatsink.name)
        self.side = QComboBox()
        self.side.addItem("Top", "top")
        self.side.addItem("Bottom", "bottom")
        self.side.setCurrentIndex(self.side.findData(heatsink.side))
        self.mount = QComboBox()
        self.mount.addItem("Board", "board")
        self.mount.addItem("Components (resistive element bodies)", "components")
        self.mount.setCurrentIndex(self.mount.findData(heatsink.mount))
        self.mount.setToolTip(
            "Board: pressed onto the board face; it conducts through the solder mask where the mask\n"
            "covers it (much less than to exposed copper).\n"
            "Components: sitting on the resistive elements under it; heat flows through their\n"
            "bodies, and the heatsink doesn't touch the board."
        )
        self.x = spin_box(gx, -MAX_COORD_MM, MAX_COORD_MM, " mm")
        self.y = spin_box(gy, -MAX_COORD_MM, MAX_COORD_MM, " mm")
        self.width = spin_box(heatsink.width_mm, 0.01, MAX_COORD_MM, " mm")
        self.height = spin_box(heatsink.height_mm, 0.01, MAX_COORD_MM, " mm")
        self.board_theta = spin_box(heatsink.thermal_resistance_board_k_w, 0.001, 1e6, " K/W")
        self.board_theta.setToolTip(
            "Contact: thermal pad, paste or gap filler between the heatsink and the board (or the\n"
            "component bodies, shared between them by covered area)"
        )
        self.ambient_theta = spin_box(heatsink.thermal_resistance_ambient_k_w, 0.001, 1e6, " K/W")
        self.ambient_theta.setToolTip("Heatsink-to-ambient: the heatsink's datasheet value for your airflow")
        note = QLabel(
            "X, Y: lower-left corner in grid coordinates (as in the status bar). On the board, heat "
            "crosses freely where the solder mask is open and through the mask (less) elsewhere."
        )
        note.setWordWrap(True)
        note.setEnabled(False)

        form = QFormLayout()
        form.addRow("Name:", self.name)
        form.addRow("Side:", self.side)
        form.addRow("Mounted on:", self.mount)
        form.addRow("X:", self.x)
        form.addRow("Y:", self.y)
        form.addRow("Width:", self.width)
        form.addRow("Height:", self.height)
        form.addRow("Contact:", self.board_theta)
        form.addRow("To ambient:", self.ambient_theta)
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
        self.side.currentIndexChanged.connect(self._validate)
        self.mount.currentIndexChanged.connect(self._validate)
        for spin in (self.x, self.y, self.width, self.height, self.board_theta, self.ambient_theta):
            spin.valueChanged.connect(self._validate)
        self._validate()

    @property
    def heatsink(self) -> Heatsink | None:
        """The heatsink as entered, or None while the entries are invalid."""
        return self._heatsink

    def _validate(self, *_args) -> None:
        """Rebuild the heatsink from the entries and enable OK only if it is valid.

        Args:
            *_args: Ignored signal arguments.
        """
        self._heatsink = None
        name = self.name.text().strip()
        x_mm, y_mm = self._frame.from_grid(self.x.value(), self.y.value(), self.height.value())
        heatsink = Heatsink(
            name=name,
            side=self.side.currentData(),
            x_mm=x_mm,
            y_mm=y_mm,
            width_mm=self.width.value(),
            height_mm=self.height.value(),
            thermal_resistance_board_k_w=self.board_theta.value(),
            thermal_resistance_ambient_k_w=self.ambient_theta.value(),
            mount=self.mount.currentData(),
            enabled=self._enabled,
        )
        error = ""
        if name in self._taken:
            error = f"Another heatsink is already named {name!r}."
        else:
            try:
                heatsink.validate()
                self._heatsink = heatsink
            except ConfigError as exc:
                error = str(exc)
        self.error.setText(error)
        self.error.setVisible(bool(error))
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(self._heatsink is not None)


class HeatsinksPanel(QWidget):
    """Lists the project's heatsinks, with draw / add / edit / remove and enable checkboxes.

    Signals:
        draw_requested(): The user wants to draw a heatsink on the board.
        show_heatsinks_toggled(bool): The user toggled the heatsinks on the board.
    """

    draw_requested = Signal()
    show_heatsinks_toggled = Signal(bool)

    def __init__(self, document: Document, parent=None):
        """Create the panel.

        Args:
            document: The application document holding the project's heatsinks.
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self._document = document
        self.frame = GridFrame()

        self.draw_button = QPushButton("Draw on board")
        self.draw_button.setToolTip("Drag a rectangle on the board for a new heatsink (Esc cancels)")
        self.draw_button.clicked.connect(self.draw_requested)
        self.add_button = QPushButton("Add…")
        self.add_button.clicked.connect(self.add_heatsink)
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

        self.show_heatsinks = QCheckBox("Show heatsinks")
        self.show_heatsinks.setChecked(True)
        self.show_heatsinks.setToolTip("Show the heatsinks on the board (T)")
        self.show_heatsinks.toggled.connect(self.show_heatsinks_toggled)
        note = QLabel(
            "A heatsink is one body joined to ambient and to what it sits on: the board face "
            "(through the solder mask where it covers, which conducts less) or the bodies of the "
            "resistive elements under it. The board doesn't convect under a board heatsink. "
            "Dashed outline: bottom side; cross-hatched: on components."
        )
        note.setWordWrap(True)
        note.setEnabled(False)

        buttons = QHBoxLayout()
        for button in (self.draw_button, self.add_button, self.edit_button, self.remove_button):
            buttons.addWidget(button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addLayout(buttons)
        layout.addWidget(self.table, 1)
        layout.addWidget(note)
        layout.addWidget(self.show_heatsinks)

        document.project_changed.connect(self.refresh)
        document.board_changed.connect(self.refresh)
        document.heatsinks_changed.connect(self.refresh)
        self.refresh()

    # -- dialogs (separate so tests can replace them) ------------------------------------

    def run_dialog(self, dialog: HeatsinkDialog) -> Heatsink | None:
        """Show a heatsink dialog modally.

        Args:
            dialog: The dialog.

        Returns:
            The accepted heatsink, or None if cancelled.
        """
        return dialog.heatsink if dialog.exec() == QDialog.DialogCode.Accepted else None

    # -- actions ----------------------------------------------------------------------

    def _free_name(self) -> str:
        """A heatsink name not used yet: HS1, HS2, ...

        Returns:
            The name.
        """
        taken = {h.name for h in self._document.heatsinks}
        n = 1
        while f"HS{n}" in taken:
            n += 1
        return f"HS{n}"

    def add_heatsink(self) -> None:
        """Add a heatsink from the dialog, starting as a square at the middle of the board."""
        board = self._document.board
        bbox = board.bounding_box() if board is not None else None
        cx, cy = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2) if bbox else (0.0, 0.0)
        half = DEFAULT_SIZE_MM / 2
        self.add_from_rect(QRectF(cx - half, cy - half, DEFAULT_SIZE_MM, DEFAULT_SIZE_MM))

    def add_from_rect(self, rect: QRectF) -> None:
        """Add a heatsink over a rectangle (e.g. drawn on the board), confirmed in the dialog.

        Args:
            rect: The rectangle in board coordinates.
        """
        if self._document.project is None:
            return
        suggestion = Heatsink(
            name=self._free_name(),
            side="top",
            x_mm=round(rect.left(), 3),
            y_mm=round(rect.top(), 3),
            width_mm=max(round(rect.width(), 3), 0.01),
            height_mm=max(round(rect.height(), 3), 0.01),
            thermal_resistance_board_k_w=DEFAULT_BOARD_K_W,
            thermal_resistance_ambient_k_w=DEFAULT_AMBIENT_K_W,
        )
        taken = {h.name for h in self._document.heatsinks}
        heatsink = self.run_dialog(HeatsinkDialog(suggestion, taken, self.frame, self))
        if heatsink is not None:
            try:
                self._document.add_heatsink(heatsink)
            except ConfigError:
                return
            self._select_name(heatsink.name)

    def edit_selected(self) -> None:
        """Edit the selected heatsink in the dialog."""
        rows = self.table.selectionModel().selectedRows()
        if len(rows) != 1:
            return
        self.edit(self._row_name(rows[0].row()))

    def edit(self, name: str) -> None:
        """Edit a heatsink in the dialog.

        Args:
            name: Heatsink name.
        """
        old = self._document.heatsink(name)
        if old is None:
            return
        taken = {h.name for h in self._document.heatsinks} - {old.name}
        heatsink = self.run_dialog(HeatsinkDialog(old, taken, self.frame, self))
        if heatsink is not None:
            try:
                self._document.update_heatsink(old.name, heatsink)
            except ConfigError:
                return
            self._select_name(heatsink.name)

    def remove_selected_rows(self) -> None:
        """Remove the heatsinks in the selected rows."""
        names = [self._row_name(index.row()) for index in self.table.selectionModel().selectedRows()]
        for name in names:
            if self._document.heatsink(name) is not None:
                self._document.remove_heatsink(name)

    # -- table ----------------------------------------------------------------------

    def refresh(self, *_args) -> None:
        """Rebuild the table from the document's heatsinks.

        Args:
            *_args: Ignored signal arguments.
        """
        selected = {self._row_name(i.row()) for i in self.table.selectionModel().selectedRows()}
        heatsinks = self._document.heatsinks
        self.table.blockSignals(True)
        self.table.setRowCount(len(heatsinks))
        for row, hs in enumerate(heatsinks):
            gx, gy = self.frame.to_grid(hs)
            cells = [
                hs.name,
                hs.side,
                hs.mount,
                f"{gx:.2f}, {gy:.2f}",
                f"{hs.width_mm:g} × {hs.height_mm:g}",
                f"{hs.thermal_resistance_board_k_w:g} K/W",
                f"{hs.thermal_resistance_ambient_k_w:g} K/W",
            ]
            items = []
            for text in cells:
                item = QTableWidgetItem(text)
                item.setData(_NAME_ROLE, hs.name)
                items.append(item)
            make_enable_box(items[0], hs.enabled, *items[1:])
            for column, item in enumerate(items):
                self.table.setItem(row, column, item)
            if hs.name in selected:
                self.table.selectRow(row)
        self.table.blockSignals(False)
        has_project = self._document.project is not None
        self.draw_button.setEnabled(has_project and self._document.board is not None)
        self.add_button.setEnabled(has_project)
        self._update_buttons()

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """Enable or disable a heatsink when its checkbox is toggled.

        Args:
            item: The changed cell.
        """
        if item.column() == 0:
            on = item.checkState() == Qt.CheckState.Checked
            self._document.set_heatsink_enabled(self._row_name(item.row()), on)

    def _row_name(self, row: int) -> str:
        """Heatsink name of a table row.

        Args:
            row: Row index.

        Returns:
            The name.
        """
        return self.table.item(row, 0).data(_NAME_ROLE)

    def _select_name(self, name: str) -> None:
        """Select a heatsink's row.

        Args:
            name: Heatsink name.
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
