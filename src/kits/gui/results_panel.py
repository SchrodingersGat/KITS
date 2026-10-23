"""Dock panel showing the solve results: a summary, how they're drawn on the board, and markers."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QLinearGradient, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..postprocess.limits import check_limits, verdict
from .document import Document
from .field_display import (
    CURRENT_DENSITY_PERCENTILE,
    DisplayMode,
    colormap,
    format_value,
    scale_midpoint,
    supports_log,
)

_WARNING_STYLE = "color: #e0a030;"
_LEGEND_SIZE = (220, 14)
_NUMBER = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
PASS_COLOR = QColor(22, 163, 74)
FAIL_COLOR = QColor(220, 38, 38)

# Results tables: key -> (title, column headings). The first column names the item;
# the rest are right-aligned numbers.
RESULT_TABLES = {
    "limits": ("Pass/fail limits", ["Check", "Result", "Value", "Margin"]),
    "sources": ("Current sources", ["Pad", "Current", "Voltage"]),
    "loads": ("Heat loads", ["Pad", "Heat into board", "Pad temperature"]),
    "elements": (
        "Resistive elements",
        ["Element", "Current", "Power", "Body temp.", "Heat to A", "Heat to B", "Heat to ambient"],
    ),
    "heatsinks": ("Heatsinks", ["Heatsink", "Heat", "Temperature"]),
}
_TABLE_TIPS = {
    "limits": (
        "The results against the project's limits (set on the Solve tab). Current density leaves out\n"
        "corner spikes as set there: sharp inside corners have mesh-dependent spikes."
    ),
    "sources": "Current into the board at each source pad (negative: a sink), and the pad's voltage",
    "loads": "Heat into the board at each heat load's pad (negative: taken out), and its mean temperature",
    "elements": (
        "Current from terminal A to B, I²R power and body temperature. Heat to a terminal is\n"
        "positive where the body heats its pads (negative where the board heats the body)."
    ),
    "heatsinks": "Heat each heatsink takes from the board (and sheds to ambient), and its temperature",
}


def _number(value: float | None, unit: str, digits: int = 4, signed: bool = False) -> str:
    """Format a table number with its unit.

    Args:
        value: The number, or None for a dash.
        unit: Unit text.
        digits: Significant digits.
        signed: Show ``+`` on positive values.

    Returns:
        e.g. ``"+1.234 W"`` or ``"—"``.
    """
    if value is None:
        return "—"
    return f"{value:+.{digits}g} {unit}" if signed else f"{value:.{digits}g} {unit}"


def copy_cells(table: QTableWidget) -> str:
    """Copy a table's selected cells to the clipboard as tab-separated rows.

    Args:
        table: The table.

    Returns:
        The copied text (empty if nothing is selected).
    """
    cells = sorted((i.row(), i.column(), i.text()) for i in table.selectedItems())
    rows: dict[int, list[str]] = {}
    for row, _column, text in cells:
        rows.setdefault(row, []).append(text)
    text = "\n".join("\t".join(values) for _row, values in sorted(rows.items()))
    if text:
        QApplication.clipboard().setText(text)
    return text


class ResultsPanel(QWidget):
    """Summarises the latest results and picks the quantity shown on the board.

    Signals:
        mode_changed(DisplayMode): The displayed quantity changed.
        log_scale_toggled(bool): The log colour scale option changed.
        show_results_toggled(bool): The results overlay was toggled.
        show_vectors_toggled(bool): The flow arrows on the results were toggled.
        show_markers_toggled(bool): The markers were toggled.
        report_requested(): The user asked for an HTML report of the results.
    """

    report_requested = Signal()
    mode_changed = Signal(object)
    log_scale_toggled = Signal(bool)
    show_results_toggled = Signal(bool)
    show_vectors_toggled = Signal(bool)
    show_markers_toggled = Signal(bool)

    def __init__(self, document: Document, parent=None):
        """Create the panel.

        Args:
            document: The application document holding the results.
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self._document = document

        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        # Overall PASS / FAIL against the project's limits (hidden without limits or results).
        self.verdict = QLabel()
        self.verdict.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.verdict.hide()
        self.report_button = QPushButton("Generate report…")
        self.report_button.setToolTip(
            "Save an HTML report: layer-by-layer maps of each result (hover for values) and every "
            "element's figures (Ctrl+Shift+R)"
        )
        self.report_button.setEnabled(False)
        self.report_button.clicked.connect(self.report_requested)
        # One table per kind of simulation element, shown when it has rows.
        self.tables: dict[str, QTableWidget] = {}
        self._table_groups: dict[str, QGroupBox] = {}
        for key, (title, headers) in RESULT_TABLES.items():
            self._table_groups[key], self.tables[key] = self._build_table(title, headers, _TABLE_TIPS[key])

        self.mode = QComboBox()
        for mode in DisplayMode:
            self.mode.addItem(mode.label, mode)
        self.mode.currentIndexChanged.connect(lambda _i: self.mode_changed.emit(self.display_mode))
        self.log_scale = QCheckBox("Log scale")
        self.log_scale.setChecked(True)
        self.log_scale.setToolTip(
            "Spread current density over decades so moderate values (e.g. around vias) stay\n"
            "visible next to a heavily loaded track. Current density only."
        )
        self.log_scale.toggled.connect(self.log_scale_toggled)
        self.mode.currentIndexChanged.connect(lambda _i: self._update_log_enabled())
        self.show_results = QCheckBox("Show results")
        self.show_results.setChecked(True)
        self.show_results.toggled.connect(self.show_results_toggled)
        self.show_vectors = QCheckBox("Show vectors")
        self.show_vectors.setToolTip(
            "Arrows showing where current flows, or heat in the temperature view;\n"
            "longer arrows mean more flow (V)"
        )
        self.show_vectors.toggled.connect(self.show_vectors_toggled)

        self.legend_bar = QLabel()
        self.legend_low = QLabel()
        self.legend_mid = QLabel()
        self.legend_mid.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.legend_high = QLabel()
        self.legend_high.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.legend_note = QLabel()
        self.legend_note.setWordWrap(True)
        self.legend_note.setEnabled(False)
        limits = QHBoxLayout()
        limits.addWidget(self.legend_low)
        limits.addStretch(1)
        limits.addWidget(self.legend_mid)
        limits.addStretch(1)
        limits.addWidget(self.legend_high)
        self.legend = QWidget()
        legend_layout = QVBoxLayout(self.legend)
        legend_layout.setContentsMargins(0, 0, 0, 0)
        legend_layout.addWidget(self.legend_bar)
        legend_layout.addLayout(limits)
        legend_layout.addWidget(self.legend_note)

        display = QGroupBox("Display")
        display_form = QFormLayout(display)
        display_form.addRow("Show:", self.mode)
        display_form.addRow(self.log_scale)
        display_form.addRow(self.show_results)
        display_form.addRow(self.show_vectors)
        display_form.addRow(self.legend)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.summary)
        layout.addWidget(self.verdict)
        layout.addWidget(self.report_button)
        for group in self._table_groups.values():
            layout.addWidget(group)
        layout.addWidget(display)
        layout.addWidget(self._build_markers_group())
        layout.addStretch(1)

        document.project_changed.connect(self.refresh)
        document.board_changed.connect(self.refresh)
        document.results_changed.connect(self.refresh)
        document.limits_changed.connect(self.refresh)
        for signal in (
            document.sources_changed,
            document.heat_loads_changed,
            document.elements_changed,
            document.stackup_changed,
            document.heatsinks_changed,
            document.settings_changed,
        ):
            signal.connect(self.refresh)  # the results may go out of date
        self.refresh()

    @property
    def display_mode(self) -> DisplayMode:
        """The quantity selected for display."""
        return self.mode.currentData()

    @property
    def log_scale_selected(self) -> bool:
        """Whether a log colour scale is requested (applies to current density only)."""
        return self.log_scale.isChecked()

    def _update_log_enabled(self) -> None:
        """Enable the log-scale option only for quantities that support it."""
        self.log_scale.setEnabled(self._document.results is not None and supports_log(self.display_mode))

    # -- markers --------------------------------------------------------------------

    def _build_markers_group(self) -> QGroupBox:
        """Create the Markers list with its buttons and show/hide checkbox.

        Returns:
            The group box.
        """
        self.show_markers = QCheckBox("Show markers")
        self.show_markers.setChecked(True)
        self.show_markers.setToolTip("Show the result markers on the board (K)")
        self.show_markers.toggled.connect(self.show_markers_toggled)
        self.markers_hint = QLabel("Click the board to add a marker.")
        self.markers_hint.setEnabled(False)
        self.markers_table = QTableWidget(0, 3)
        self.markers_table.setHorizontalHeaderLabels(["#", "Position (mm)", "Value"])
        self.markers_table.verticalHeader().setVisible(False)
        self.markers_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.markers_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.markers_table.setMaximumHeight(160)
        header = self.markers_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        QShortcut(QKeySequence.StandardKey.Delete, self.markers_table, self.remove_selected_markers)
        self.remove_markers_button = QPushButton("Remove")
        self.remove_markers_button.clicked.connect(self.remove_selected_markers)
        self.clear_markers_button = QPushButton("Clear all")
        self.clear_markers_button.clicked.connect(self._document.clear_markers)
        buttons = QHBoxLayout()
        buttons.addWidget(self.remove_markers_button)
        buttons.addWidget(self.clear_markers_button)
        group = QGroupBox("Markers")
        column = QVBoxLayout(group)
        column.addWidget(self.markers_hint)
        column.addWidget(self.markers_table)
        column.addLayout(buttons)
        column.addWidget(self.show_markers)
        self.set_markers([])
        return group

    def set_markers(self, rows: list[tuple[str, str]]) -> None:
        """List the markers.

        Args:
            rows: ``(position, value)`` texts per marker, in numbering order.
        """
        self.markers_table.setRowCount(len(rows))
        for row, (position, value) in enumerate(rows):
            for col, text in enumerate((str(row + 1), position, value)):
                self.markers_table.setItem(row, col, QTableWidgetItem(text))
        self.markers_table.setVisible(bool(rows))
        self.remove_markers_button.setEnabled(bool(rows))
        self.clear_markers_button.setEnabled(bool(rows))

    def remove_selected_markers(self) -> None:
        """Remove the markers selected in the list."""
        rows = {index.row() for index in self.markers_table.selectionModel().selectedRows()}
        self._document.remove_markers(rows)

    # -- legend and summary ----------------------------------------------------------

    def set_legend(self, mode: DisplayMode, low: float, high: float, log: bool = False) -> None:
        """Draw the colour scale for the displayed quantity.

        Args:
            mode: The displayed quantity.
            low: Value at the bottom of the scale, in display units.
            high: Value at the top of the scale, in display units.
            log: Whether the scale is logarithmic (the middle label is then the geometric mean).
        """
        width, height = _LEGEND_SIZE
        pixmap = QPixmap(width, height)
        painter = QPainter(pixmap)
        gradient = QLinearGradient(0, 0, width, 0)
        colors = colormap(mode, 16)
        for i, color in enumerate(colors):
            gradient.setColorAt(i / (len(colors) - 1), color)
        painter.fillRect(0, 0, width, height, gradient)
        painter.end()
        self.legend_bar.setPixmap(pixmap)
        self.legend_low.setText(("≤ " if log else "") + format_value(low, mode))
        self.legend_mid.setText(format_value(scale_midpoint(low, high, log), mode))
        high_text = format_value(high, mode)
        if mode is DisplayMode.CURRENT_DENSITY:
            high_text = f"≥ {high_text}"
            self.legend_note.setText(
                ("Log scale. " if log else "")
                + f"Scale tops out at the {CURRENT_DENSITY_PERCENTILE:g}th percentile: sharp inside "
                "corners have mesh-dependent spikes."
            )
        else:
            self.legend_note.setText("Scale shared by all layers. Select a layer in the Layers panel.")
        self.legend_high.setText(high_text)

    # -- results tables ------------------------------------------------------------

    def _build_table(self, title: str, headers: list[str], tip: str) -> tuple[QGroupBox, QTableWidget]:
        """Create one read-only results table in its group box (hidden until it has rows).

        Args:
            title: Group title.
            headers: Column headings; the first column names the item.
            tip: Explanation shown on hover.

        Returns:
            ``(group, table)``.
        """
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # sized to its rows
        table.setWordWrap(False)
        table.setToolTip(f"{tip}\nCtrl+C copies the selected cells (tab-separated).")
        header = table.horizontalHeader()
        # Every column fits its contents (names are never cut short); the last takes any spare
        # width, and a narrow dock scrolls the table sideways.
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        for column in range(1, len(headers)):
            table.horizontalHeaderItem(column).setTextAlignment(_NUMBER)
        # The sideways scrollbar comes and goes with the dock's width: keep every row visible.
        table.horizontalScrollBar().rangeChanged.connect(lambda _low, _high: self._fit_height(table))
        copy = QShortcut(QKeySequence.StandardKey.Copy, table, lambda: copy_cells(table))
        copy.setContext(Qt.ShortcutContext.WidgetShortcut)
        group = QGroupBox(title)
        QVBoxLayout(group).addWidget(table)
        group.hide()
        return group, table

    def _set_rows(self, key: str, rows: list[list[str]]) -> None:
        """Fill one results table, sizing it to its rows; hidden when there are none.

        Args:
            key: The table (a :data:`RESULT_TABLES` key).
            rows: Cell texts per row.
        """
        table = self.tables[key]
        table.setRowCount(len(rows))
        for r, cells in enumerate(rows):
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c:
                    item.setTextAlignment(_NUMBER)
                if text in ("PASS", "FAIL"):
                    item.setForeground(PASS_COLOR if text == "PASS" else FAIL_COLOR)
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                table.setItem(r, c, item)
        table.resizeRowsToContents()
        self._fit_height(table)
        self._table_groups[key].setVisible(bool(rows))

    @staticmethod
    def _fit_height(table: QTableWidget) -> None:
        """Make a table exactly tall enough for its rows (and its sideways scrollbar, if shown).

        Args:
            table: The table.
        """
        height = table.horizontalHeader().height() + 2 * table.frameWidth()
        height += sum(table.rowHeight(r) for r in range(table.rowCount()))
        bar = table.horizontalScrollBar()
        if bar.maximum() > bar.minimum():  # it will be shown
            height += bar.sizeHint().height()
        table.setFixedHeight(height)

    def _fill_tables(self, results) -> None:
        """Show each simulation element's results in its table.

        Args:
            results: The :class:`~kits.solve.results.SolveResults`, or None to empty the tables.
        """
        if results is None:
            for key in self.tables:
                self._set_rows(key, [])
            self._show_verdict(None)
            return
        checks = check_limits(results, self._document.limits)
        self._set_rows(
            "limits",
            [
                [
                    f"{c.quantity} ≤ {c.limit:g} {c.unit}",
                    "PASS" if c.passed else "FAIL",
                    f"{c.value:.4g} {c.unit}",
                    f"{c.margin:+.3g} {c.unit}",
                ]
                for c in checks
            ],
        )
        for row, c in enumerate(checks):  # where each value occurs, and how it's taken
            self.tables["limits"].item(row, 2).setToolTip(f"{c.where}: {c.basis}")
        self._show_verdict(verdict(checks))
        temperature = DisplayMode.TEMPERATURE

        def degrees(value: float | None) -> str:
            """Format a temperature (or a dash).

            Args:
                value: °C, or None.

            Returns:
                The text.
            """
            return "—" if value is None else format_value(value, temperature)

        self._set_rows(
            "sources",
            [
                [
                    label,
                    _number(results.electrode_currents.get(label), "A", signed=True),
                    format_value(voltage, DisplayMode.VOLTAGE),
                ]
                for label, voltage in sorted(results.electrode_voltages.items())
            ],
        )
        self._set_rows(
            "loads",
            [
                [label, _number(heat, "W", signed=True), degrees(results.load_temperatures.get(label))]
                for label, heat in sorted(results.load_heat.items())
            ],
        )
        elements = []
        for name, body in sorted(results.element_temperatures.items()):
            to_a, to_b = results.element_terminal_heat.get(name, (None, None))
            elements.append(
                [
                    name,
                    _number(results.element_currents.get(name), "A", signed=True),
                    _number(results.element_power.get(name), "W"),
                    degrees(body),
                    _number(to_a, "W", signed=True),
                    _number(to_b, "W", signed=True),
                    _number(results.element_ambient_heat.get(name), "W"),
                ]
            )
        self._set_rows("elements", elements)
        self._set_rows(
            "heatsinks",
            [
                [name, _number(results.heatsink_heat.get(name), "W"), degrees(temperature_c)]
                for name, temperature_c in sorted(results.heatsink_temperatures.items())
            ],
        )

    def _show_verdict(self, passed: bool | None) -> None:
        """Show the overall PASS / FAIL banner.

        Args:
            passed: The verdict, or None to hide the banner (no limits, or no results).
        """
        if passed is None:
            self.verdict.hide()
            return
        stale = " (results out of date)" if self._document.results_staleness else ""
        color = PASS_COLOR if passed else FAIL_COLOR
        self.verdict.setText(("PASS" if passed else "FAIL") + stale)
        self.verdict.setStyleSheet(
            f"background: {color.name()}; color: white; font-weight: bold; font-size: 15px;"
            " border-radius: 4px; padding: 4px;"
        )
        self.verdict.setToolTip(
            "All limits met" if passed else "At least one limit is exceeded: see Pass/fail limits below"
        )
        self.verdict.show()

    def table_rows(self, key: str) -> list[list[str]]:
        """The cell texts of one results table.

        Args:
            key: The table (a :data:`RESULT_TABLES` key).

        Returns:
            Rows of cell texts.
        """
        table = self.tables[key]
        return [
            [table.item(r, c).text() for c in range(table.columnCount())] for r in range(table.rowCount())
        ]

    def refresh(self, *_args) -> None:
        """Update the summary text and which controls are usable.

        Args:
            *_args: Ignored signal arguments.
        """
        doc = self._document
        results = doc.results
        style, lines = "", []
        self._fill_tables(results)
        if results is not None:
            lines.append(f"Joule power: {results.total_power_w:.4g} W")
            total = results.total_power_w
            for label, watts in (
                ("Copper losses", results.copper_losses_w),
                ("Resistive element losses", results.element_losses_w),
            ):
                share = f" ({100 * watts / total:.1f}%)" if total > 0 else ""
                lines.append(f"    {label}: {watts:.4g} W{share}")
            lines.append(f"Max temperature: {results.max_temperature_c:.1f} °C")
            if doc.results_from_file:
                lines.append("Reloaded from the project's saved results.")
            staleness = doc.results_staleness
            if staleness:
                style = _WARNING_STYLE
                lines.append(f"{staleness}; solve again to update.")
        elif doc.results_notice and doc.board is not None:
            style = _WARNING_STYLE
            lines.append(f"{doc.results_notice}, so they weren't loaded. Solve again.")
        else:
            lines.append("No results yet: solve the board (Solve tab, or F9).")
        self.summary.setStyleSheet(style)
        self.summary.setText("\n".join(lines))

        has_results = results is not None
        for widget in (self.mode, self.show_results, self.show_vectors, self.legend):
            widget.setEnabled(has_results)
        self._update_log_enabled()
        self.legend.setVisible(has_results)
