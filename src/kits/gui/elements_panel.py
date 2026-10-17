"""Dock panel and editor dialog for resistive elements (lumped resistors between pad groups)."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import ConfigError, PadGroup, ResistiveElement, SolveSettings
from ..ingest.model import Footprint
from ..mesh.geometry import footprint_body
from ..solve.elements import (
    auto_pad_groups,
    default_ambient_conductance,
    default_body_thermal,
    pad_numbers,
    parse_resistance,
    suggest_resistance,
)
from .document import Document
from .field_display import format_resistance
from .sources_panel import make_enable_box

DEFAULT_RESISTANCE_OHM = 0.01  # when the footprint value isn't a resistance (e.g. a MOSFET)
DEFAULT_THETA_K_W = 10.0
MIN_THETA_K_W = 0.001
MAX_THETA_K_W = 1e6
_COLUMNS = ["Name", "Terminals", "R", "Rθ body→A / B", "Rθ ambient"]
_NAME_ROLE = Qt.ItemDataRole.UserRole
_ERROR_STYLE = "color: #e05050;"


def _theta_text(value: float | None, default: str) -> str:
    """Show a thermal resistance, or what its default means.

    Args:
        value: K/W, or None for the default.
        default: Text for the default.

    Returns:
        e.g. ``"20 K/W"`` or ``default``.
    """
    return default if value is None else f"{value:g} K/W"


def _theta_editor(default_label: str) -> tuple[QCheckBox, QDoubleSpinBox, QWidget]:
    """A "use default" checkbox with a K/W spin box that is enabled only when it is off.

    Args:
        default_label: The checkbox text, describing the default.

    Returns:
        ``(checkbox, spin_box, container)``.
    """
    check = QCheckBox(default_label)
    spin = QDoubleSpinBox()
    spin.setRange(MIN_THETA_K_W, MAX_THETA_K_W)
    spin.setDecimals(3)
    spin.setSuffix(" K/W")
    spin.setValue(DEFAULT_THETA_K_W)
    check.toggled.connect(lambda on: spin.setEnabled(not on))
    box = QWidget()
    row = QHBoxLayout(box)
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(check, 1)
    row.addWidget(spin)
    return check, spin, box


class ElementDialog(QDialog):
    """Edits one resistive element: name, the two pad groups, resistance and thermal resistances."""

    def __init__(
        self,
        footprint: Footprint,
        element: ResistiveElement,
        taken_names: set[str],
        settings: SolveSettings,
        parent=None,
    ):
        """Create the dialog, filled in from ``element``.

        Args:
            footprint: The element's footprint (its pads can be assigned to the groups).
            element: The element to edit (or a new one with suggested values).
            taken_names: Names other elements already use.
            settings: Solver settings, for the default ambient thermal resistance.
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle(f"Resistive element on {footprint.reference}")
        self._footprint = footprint
        self._ref = footprint.reference
        self._taken = taken_names
        self._element: ResistiveElement | None = None
        self._enabled = element.enabled
        nets: dict[str, set[str]] = {}
        for pad in footprint.pads:
            if pad.number and pad.net:
                nets.setdefault(pad.number, set()).add(pad.net)

        self.name = QLineEdit(element.name)
        self.resistance = QLineEdit(format_resistance(element.resistance_ohm).replace(" ", ""))
        self.resistance.setToolTip("Ohms, with an optional SI prefix: 10m, 4.7, 4R7, 2k2, 10mΩ")
        suggestion = suggest_resistance(footprint)
        if suggestion is not None:
            hint = f"From the part's {suggestion.source}: {format_resistance(suggestion.ohms)}"
        else:
            hint = (
                f"Value {footprint.value or '—'} isn't a resistance, and no field gives one "
                "(e.g. Resistance or Rds(on)): enter it from the datasheet"
            )
        value_hint = QLabel(hint)
        value_hint.setWordWrap(True)
        value_hint.setEnabled(False)

        groups = QGridLayout()
        self.group_names: list[QLineEdit] = []
        self.group_pads: list[QListWidget] = []
        for column, (title, group) in enumerate(
            (("Terminal A", element.group_a), ("Terminal B", element.group_b))
        ):
            name = QLineEdit(group.name)
            pads = QListWidget()
            for number in pad_numbers(footprint):
                net = ", ".join(sorted(nets.get(number, ())))
                item = QListWidgetItem(f"{number}  ({net})" if net else number)
                item.setData(Qt.ItemDataRole.UserRole, number)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if number in group.pads else Qt.CheckState.Unchecked)
                pads.addItem(item)
            pads.itemChanged.connect(self._validate)
            name.textChanged.connect(self._validate)
            groups.addWidget(QLabel(f"<b>{title}</b>"), 0, column)
            groups.addWidget(name, 1, column)
            groups.addWidget(pads, 2, column)
            self.group_names.append(name)
            self.group_pads.append(pads)

        ambient_default = default_ambient_conductance(footprint, settings)
        ambient_text = (
            f"≈ {1 / ambient_default:.3g} K/W" if ambient_default > 0 else "none: no convection on that side"
        )
        # Body to each terminal's pads: defaulted from the body type (label set in _validate).
        self.terminal_default: list[QCheckBox] = []
        self.terminal_theta: list[QDoubleSpinBox] = []
        terminal_rows = []
        for _ in range(2):
            check, spin, row = _theta_editor("")
            check.setToolTip(
                "Typical value for the footprint's body type (from its library name and reference).\n"
                "Enter a datasheet figure instead where you have one."
            )
            self.terminal_default.append(check)
            self.terminal_theta.append(spin)
            terminal_rows.append(row)
        self.ambient_default, self.ambient_theta, ambient_row = _theta_editor(
            f"Convection from the body ({ambient_text})"
        )
        for check, spin, value in (
            (self.terminal_default[0], self.terminal_theta[0], element.thermal_resistance_a_k_w),
            (self.terminal_default[1], self.terminal_theta[1], element.thermal_resistance_b_k_w),
            (self.ambient_default, self.ambient_theta, element.thermal_resistance_ambient_k_w),
        ):
            check.setChecked(value is None)
            spin.setEnabled(value is not None)
            if value is not None:
                spin.setValue(value)

        thermal = QGroupBox("Thermal")
        thermal_form = QFormLayout(thermal)
        body, source = footprint_body(footprint)
        if body.is_empty:
            body_text = "none (the footprint has no outline or pads)"
        else:
            x0, y0, x1, y1 = body.bounds
            body_text = f"{source}, {x1 - x0:.2f} × {y1 - y0:.2f} mm ({body.area:.3g} mm²)"
        self.body_label = QLabel(body_text)
        self.body_label.setToolTip(
            "The component body's plan area, from the footprint's Fab outline, else its courtyard,\n"
            "else its pads. The body loses heat to ambient from it, and is drawn over it on the\n"
            "temperature map."
        )
        thermal_form.addRow("Body:", self.body_label)
        self._terminal_labels = [QLabel(), QLabel()]
        for label, row in zip(self._terminal_labels, terminal_rows, strict=True):
            thermal_form.addRow(label, row)
        thermal_form.addRow("To ambient:", ambient_row)
        note = QLabel(
            "The element's I²R heats the body, which passes it to the board through each terminal's "
            "pads, and to ambient. Heat also crosses the part from a hotter terminal to a cooler one."
        )
        note.setWordWrap(True)
        note.setEnabled(False)
        thermal_form.addRow(note)

        self.error = QLabel()
        self.error.setStyleSheet(_ERROR_STYLE)
        self.error.setWordWrap(True)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        form = QFormLayout()
        form.addRow("Name:", self.name)
        form.addRow("Resistance:", self.resistance)
        form.addRow("", value_hint)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(groups)
        layout.addWidget(thermal)
        layout.addWidget(self.error)
        layout.addWidget(self.buttons)

        self.name.textChanged.connect(self._validate)
        self.resistance.textChanged.connect(self._validate)
        for widget in (*self.terminal_default, self.ambient_default):
            widget.toggled.connect(self._validate)
        for widget in (*self.terminal_theta, self.ambient_theta):
            widget.valueChanged.connect(self._validate)
        self._validate()

    @property
    def element(self) -> ResistiveElement | None:
        """The element as entered, or None while the entries are invalid."""
        return self._element

    def _checked_pads(self, index: int) -> list[str]:
        """Pad numbers ticked for one terminal.

        Args:
            index: 0 for terminal A, 1 for B.

        Returns:
            The pad numbers, in list order.
        """
        pads = self.group_pads[index]
        return [
            pads.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(pads.count())
            if pads.item(i).checkState() == Qt.CheckState.Checked
        ]

    def _validate(self, *_args) -> None:
        """Rebuild the element from the entries and enable OK only if it is valid.

        Args:
            *_args: Ignored signal arguments.
        """
        self._element = None
        error = ""
        resistance = parse_resistance(self.resistance.text())
        name = self.name.text().strip()
        groups = [PadGroup(self.group_names[i].text().strip(), self._checked_pads(i)) for i in range(2)]
        # The default depends on the groups (a package's die sits on the larger terminal).
        default = default_body_thermal(
            self._footprint, ResistiveElement(name, self._ref, groups[0], groups[1], resistance or 1.0)
        )
        for i, (label, check, value) in enumerate(
            zip(self._terminal_labels, self.terminal_default, (default.a_k_w, default.b_k_w), strict=True)
        ):
            label.setText(f"To {groups[i].name or 'AB'[i]}:")
            check.setText(f"Default: {default.body_type}, {value:g} K/W")
        if resistance is None:
            error = "Enter a positive resistance, e.g. 10m, 4R7 or 2k2."
        elif name in self._taken:
            error = f"Another element is already named {name!r}."
        else:
            element = ResistiveElement(
                name=name,
                ref=self._ref,
                group_a=groups[0],
                group_b=groups[1],
                resistance_ohm=resistance,
                thermal_resistance_a_k_w=(
                    None if self.terminal_default[0].isChecked() else self.terminal_theta[0].value()
                ),
                thermal_resistance_b_k_w=(
                    None if self.terminal_default[1].isChecked() else self.terminal_theta[1].value()
                ),
                thermal_resistance_ambient_k_w=(
                    None if self.ambient_default.isChecked() else self.ambient_theta.value()
                ),
                enabled=self._enabled,
            )
            try:
                element.validate()
                self._element = element
            except ConfigError as exc:
                error = str(exc).replace(f"element {name}: ", "").capitalize()
        self.error.setText(error)
        self.error.setVisible(bool(error))
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(self._element is not None)


class ElementsPanel(QWidget):
    """Lists the project's resistive elements, with add (from a pad's footprint) / edit / remove.

    Signals:
        show_elements_toggled(bool): The user toggled the element markers on the board.
    """

    show_elements_toggled = Signal(bool)

    def __init__(self, document: Document, parent=None):
        """Create the panel.

        Args:
            document: The application document holding the project's elements.
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self._document = document
        self._selected_pad: tuple[str, str] | None = None

        self.selected_label = QLabel()
        self.selected_label.setWordWrap(True)
        self.add_button = QPushButton("Add for selected pad")
        self.add_button.setToolTip(
            "Add an element on the footprint of the pad selected on the board; its terminals are\n"
            "found from the pads' nets (or double-click a pad while this tab is shown)"
        )
        self.add_button.clicked.connect(self.add_for_selected_pad)
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
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._update_selection_ui)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.cellDoubleClicked.connect(lambda _row, _col: self.edit_selected())
        QShortcut(QKeySequence.StandardKey.Delete, self.table, self.remove_selected_rows)

        self.show_elements = QCheckBox("Show resistive elements")
        self.show_elements.setChecked(True)
        self.show_elements.setToolTip("Show the element terminals and labels on the board (E)")
        self.show_elements.toggled.connect(self.show_elements_toggled)
        self.note = QLabel(
            "Each terminal's pads are tied together (like a lead frame). The element's I²R heats its body."
        )
        self.note.setWordWrap(True)
        self.note.setEnabled(False)

        buttons = QHBoxLayout()
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.edit_button)
        buttons.addWidget(self.remove_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.selected_label)
        layout.addLayout(buttons)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.note)
        layout.addWidget(self.show_elements)

        document.project_changed.connect(self._on_project_changed)
        document.board_changed.connect(self.refresh)
        document.elements_changed.connect(self.refresh)
        self.refresh()

    # -- dialogs (separate so tests can replace them) ------------------------------------

    def run_dialog(self, dialog: ElementDialog) -> ResistiveElement | None:
        """Show an element dialog modally.

        Args:
            dialog: The dialog.

        Returns:
            The accepted element, or None if cancelled.
        """
        return dialog.element if dialog.exec() == QDialog.DialogCode.Accepted else None

    # -- actions ----------------------------------------------------------------------

    def set_selected_pad(self, key: tuple[str, str] | None) -> None:
        """Remember the pad selected on the board (its footprint gets new elements).

        Args:
            key: ``(ref, pad)`` of the selected pad, or None.
        """
        self._selected_pad = tuple(key) if key is not None else None
        self._update_selection_ui()

    def _footprint(self, ref: str) -> Footprint | None:
        """Find a footprint on the board.

        Args:
            ref: Footprint reference.

        Returns:
            The footprint, or None.
        """
        board = self._document.board
        return next((fp for fp in board.footprints if fp.reference == ref), None) if board else None

    def add_for_selected_pad(self) -> None:
        """Suggest an element for the selected pad's footprint and add it if the dialog is accepted."""
        if self._selected_pad is None or self._document.project is None:
            return
        footprint = self._footprint(self._selected_pad[0])
        groups = auto_pad_groups(footprint) if footprint is not None else None
        if groups is None:
            return
        taken = {e.name for e in self._document.resistive_elements}
        name, n = footprint.reference, 2
        while name in taken:
            name, n = f"{footprint.reference}_{n}", n + 1
        suggestion = ResistiveElement(
            name=name,
            ref=footprint.reference,
            group_a=groups[0],
            group_b=groups[1],
            resistance_ohm=suggestion.ohms
            if (suggestion := suggest_resistance(footprint))
            else DEFAULT_RESISTANCE_OHM,
        )
        dialog = ElementDialog(footprint, suggestion, taken, self._document.solve_settings, self)
        element = self.run_dialog(dialog)
        if element is not None:
            try:
                self._document.add_resistive_element(element)
            except ConfigError:
                return
            self._select_name(element.name)

    def edit_selected(self) -> None:
        """Edit the selected element in a dialog."""
        rows = self.table.selectionModel().selectedRows()
        if len(rows) != 1:
            return
        old = self._document.resistive_element(self._row_name(rows[0].row()))
        footprint = self._footprint(old.ref) if old is not None else None
        if footprint is None:
            return
        taken = {e.name for e in self._document.resistive_elements} - {old.name}
        dialog = ElementDialog(footprint, old, taken, self._document.solve_settings, self)
        element = self.run_dialog(dialog)
        if element is not None:
            try:
                self._document.update_resistive_element(old.name, element)
            except ConfigError:
                return
            self._select_name(element.name)

    def remove_selected_rows(self) -> None:
        """Remove the elements in the selected rows."""
        names = [self._row_name(index.row()) for index in self.table.selectionModel().selectedRows()]
        for name in names:
            if self._document.resistive_element(name) is not None:
                self._document.remove_resistive_element(name)

    # -- table ----------------------------------------------------------------------

    def refresh(self, *_args) -> None:
        """Rebuild the table from the document's elements.

        Args:
            *_args: Ignored signal arguments.
        """
        selected = {self._row_name(i.row()) for i in self.table.selectionModel().selectedRows()}
        board = self._document.board
        self.table.blockSignals(True)
        elements = self._document.resistive_elements
        self.table.setRowCount(len(elements))
        for row, e in enumerate(elements):
            a, b = e.groups
            missing = board is not None and self._footprint(e.ref) is None
            cells = [
                e.name if e.name == e.ref else f"{e.name} ({e.ref})",
                f"{a.name}: {', '.join(a.pads)}  ↔  {b.name}: {', '.join(b.pads)}",
                format_resistance(e.resistance_ohm),
                " / ".join(_theta_text(v, "default") for v in e.terminal_thermal_resistances),
                _theta_text(e.thermal_resistance_ambient_k_w, "convection"),
            ]
            items = []
            for text in cells:
                item = QTableWidgetItem(text)
                item.setData(_NAME_ROLE, e.name)
                if missing:
                    item.setForeground(Qt.GlobalColor.darkYellow)
                    item.setToolTip(f"{e.ref} isn't on the board; the solve will fail until it is removed")
                items.append(item)
            make_enable_box(items[0], e.enabled, *items[1:])
            for column, item in enumerate(items):
                self.table.setItem(row, column, item)
            if e.name in selected:
                self.table.selectRow(row)
        self.table.blockSignals(False)
        self._update_selection_ui()

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """Enable or disable an element when its checkbox is toggled.

        Args:
            item: The changed cell.
        """
        if item.column() == 0:
            on = item.checkState() == Qt.CheckState.Checked
            self._document.set_element_enabled(self._row_name(item.row()), on)

    def _row_name(self, row: int) -> str:
        """Element name of a table row.

        Args:
            row: Row index.

        Returns:
            The name.
        """
        return self.table.item(row, 0).data(_NAME_ROLE)

    def _select_name(self, name: str) -> None:
        """Select an element's row.

        Args:
            name: Element name.
        """
        for row in range(self.table.rowCount()):
            if self._row_name(row) == name:
                self.table.selectRow(row)
                return

    def _on_project_changed(self, *_args) -> None:
        """Clear the selection when a project is opened or closed.

        Args:
            *_args: Ignored signal arguments.
        """
        self._selected_pad = None
        self.refresh()

    def _update_selection_ui(self) -> None:
        """Update the selected-footprint text and button states."""
        doc = self._document
        key = self._selected_pad
        footprint = self._footprint(key[0]) if key is not None else None
        if doc.project is None:
            text = "No project open."
        elif doc.board is None:
            text = "No board loaded."
        elif footprint is None:
            text = "Click a pad on the board to select its footprint; double-click to add an element."
        else:
            text = f"Selected: {footprint.reference} ({footprint.value or 'no value'})"
            if len(pad_numbers(footprint)) < 2:
                text += " — needs at least two pads"
        self.selected_label.setText(text)
        self.add_button.setEnabled(footprint is not None and len(pad_numbers(footprint)) >= 2)
        rows = self.table.selectionModel().selectedRows()
        self.edit_button.setEnabled(len(rows) == 1)
        self.remove_button.setEnabled(bool(rows))
