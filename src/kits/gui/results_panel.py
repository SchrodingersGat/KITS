"""Dock panel showing the solve results: a summary, how they're drawn on the board, and markers."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QLinearGradient, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
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


class ResultsPanel(QWidget):
    """Summarises the latest results and picks the quantity shown on the board.

    Signals:
        mode_changed(DisplayMode): The displayed quantity changed.
        log_scale_toggled(bool): The log colour scale option changed.
        show_results_toggled(bool): The results overlay was toggled.
        show_markers_toggled(bool): The markers were toggled.
    """

    mode_changed = Signal(object)
    log_scale_toggled = Signal(bool)
    show_results_toggled = Signal(bool)
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
        display_form.addRow(self.legend)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.summary)
        layout.addWidget(display)
        layout.addWidget(self._build_markers_group())
        layout.addStretch(1)

        document.project_changed.connect(self.refresh)
        document.board_changed.connect(self.refresh)
        document.results_changed.connect(self.refresh)
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

    def refresh(self, *_args) -> None:
        """Update the summary text and which controls are usable.

        Args:
            *_args: Ignored signal arguments.
        """
        doc = self._document
        results = doc.results
        style, lines = "", []
        if results is not None:
            lines.append(f"Joule power: {results.total_power_w:.4g} W")
            lines.append(f"Max temperature: {results.max_temperature_c:.1f} °C")
            for label, voltage in sorted(results.electrode_voltages.items()):
                lines.append(f"  {label}: {format_value(voltage, DisplayMode.VOLTAGE)}")
            if results.load_heat:
                lines.append("Heat loads (heat into the board, pad temperature):")
            for label, heat in sorted(results.load_heat.items()):
                temperature = format_value(results.load_temperatures[label], DisplayMode.TEMPERATURE)
                lines.append(f"  {label}: {heat:+.4g} W, {temperature}")
            if results.element_temperatures:
                lines.append("Resistive elements (current, power, body temperature):")
            for name, body in sorted(results.element_temperatures.items()):
                current = results.element_currents.get(name, 0.0)
                power = results.element_power.get(name, 0.0)
                temperature = format_value(body, DisplayMode.TEMPERATURE)
                lines.append(f"  {name}: {current:+.4g} A, {power:.4g} W, {temperature}")
            if results.heatsink_temperatures:
                lines.append("Heatsinks (heat taken from the board, temperature):")
            for name, temperature in sorted(results.heatsink_temperatures.items()):
                shown = format_value(temperature, DisplayMode.TEMPERATURE)
                lines.append(f"  {name}: {results.heatsink_heat.get(name, 0.0):.4g} W, {shown}")
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
        for widget in (self.mode, self.show_results, self.legend):
            widget.setEnabled(has_results)
        self._update_log_enabled()
        self.legend.setVisible(has_results)
