"""Dock panel to run a solve and control how its results are shown."""

from __future__ import annotations

import dataclasses

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QLinearGradient, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
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

from ..config import ConfigError, SolveSettings
from . import colors
from .document import Document
from .field_display import (
    CURRENT_DENSITY_PERCENTILE,
    DisplayMode,
    colormap,
    format_value,
    scale_midpoint,
    supports_log,
)

_BALANCE_TOLERANCE_A = 1e-9
_WARNING_STYLE = "color: #e0a030;"
_LEGEND_SIZE = (220, 14)
_ERROR_STYLE = "color: #e05050;"

# Editable solver settings: (attribute, label, display unit, solver units per display unit, range,
# decimals, tip). Converting by multiplying keeps round numbers exact (58 MS/m -> 58e6 S/m).
_SETTING_FIELDS = (
    ("ambient_c", "Ambient", " °C", 1.0, (-50.0, 200.0), 1, "Ambient air temperature."),
    (
        "convection_top_w_m2k",
        "Convection, top",
        " W/m²K",
        1.0,
        (0.0, 10_000.0),
        1,
        "Heat transfer coefficient from the top surface (≈5-15 still air, 25-100 forced air).",
    ),
    (
        "convection_bottom_w_m2k",
        "Convection, bottom",
        " W/m²K",
        1.0,
        (0.0, 10_000.0),
        1,
        "Heat transfer coefficient from the bottom surface; 0 for an insulated face.",
    ),
    (
        "copper_conductivity_s_m",
        "Copper σ",
        " MS/m",
        1e6,
        (1.0, 100.0),
        2,
        "Electrical conductivity of copper and via plating (annealed copper: 58 MS/m at 20 °C).",
    ),
    (
        "copper_thermal_conductivity_w_mk",
        "Copper k",
        " W/mK",
        1.0,
        (1.0, 1_000.0),
        1,
        "Thermal conductivity of copper.",
    ),
    (
        "plating_thickness_mm",
        "Via plating",
        " µm",
        1e-3,
        (1.0, 200.0),
        1,
        "Barrel plating thickness of vias and plated holes.",
    ),
    (
        "dielectric_thermal_conductivity_w_mk",
        "Dielectric k",
        " W/mK",
        1.0,
        (0.01, 50.0),
        3,
        "Thermal conductivity of the board material (FR-4 ≈ 0.3).",
    ),
)


class SolvePanel(QWidget):
    """Runs the solver and picks the quantity shown on the board.

    Signals:
        solve_requested(): The user asked to solve the board.
        mode_changed(DisplayMode): The displayed quantity changed.
        log_scale_toggled(bool): The log colour scale option changed.
        show_results_toggled(bool): The results overlay was toggled.
        show_markers_toggled(bool): The markers were toggled.
    """

    solve_requested = Signal()
    mode_changed = Signal(object)
    log_scale_toggled = Signal(bool)
    show_results_toggled = Signal(bool)
    show_markers_toggled = Signal(bool)

    def __init__(self, document: Document, parent=None):
        """Create the panel.

        Args:
            document: The application document (project, sources and results).
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self._document = document
        self._busy = False

        self.solve_button = QPushButton("Solve")
        colors.style_action_button(self.solve_button, colors.SOLVE_BUTTON)
        self.solve_button.setToolTip("Solve current flow and temperature for the current sources (F9)")
        self.solve_button.clicked.connect(self.solve_requested)

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

        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.setting_boxes: dict[str, QDoubleSpinBox] = {}
        self._pending = document.solve_settings
        settings_form = QFormLayout()
        for name, label, unit, _scale, (low, high), decimals, tip in _SETTING_FIELDS:
            box = QDoubleSpinBox()
            box.setRange(low, high)
            box.setDecimals(decimals)
            box.setSuffix(unit)
            box.setKeyboardTracking(False)
            box.setToolTip(tip)
            box.valueChanged.connect(lambda value, n=name: self._on_setting_edited(n, value))
            self.setting_boxes[name] = box
            settings_form.addRow(f"{label}:", box)
        self.settings_error = QLabel()
        self.settings_error.setStyleSheet(_ERROR_STYLE)
        self.settings_error.setWordWrap(True)
        self.settings_error.hide()
        settings_form.addRow(self.settings_error)
        self.settings_group = QGroupBox("Materials && environment")
        self.settings_group.setLayout(settings_form)

        form = QFormLayout()
        form.addRow("Display:", self.mode)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.solve_button)
        layout.addWidget(self.status)
        layout.addLayout(form)
        layout.addWidget(self.log_scale)
        layout.addWidget(self.show_results)
        layout.addWidget(self.legend)
        layout.addWidget(self._build_markers_group())
        layout.addWidget(self.settings_group)
        layout.addStretch(1)

        for signal in (
            document.project_changed,
            document.board_changed,
            document.sources_changed,
            document.heat_loads_changed,
            document.elements_changed,
            document.settings_changed,
            document.results_changed,
        ):
            signal.connect(self.refresh)
        document.project_changed.connect(self._load_settings)
        self._load_settings()
        self.refresh()

    @property
    def display_mode(self) -> DisplayMode:
        """The quantity selected for display."""
        return self.mode.currentData()

    def settings(self) -> SolveSettings:
        """Return the material and environment settings entered in the panel.

        Returns:
            The (possibly invalid) settings, in solver units.
        """
        return self._pending

    def _load_settings(self, *_args) -> None:
        """Show the open project's settings.

        Args:
            *_args: Ignored signal arguments.
        """
        settings = self._document.solve_settings
        self._pending = settings
        for name, _label, _unit, factor, _range, _decimals, _tip in _SETTING_FIELDS:
            box = self.setting_boxes[name]
            box.blockSignals(True)
            box.setValue(getattr(settings, name) / factor)
            box.blockSignals(False)
        self.settings_error.hide()

    def _on_setting_edited(self, name: str, value: float) -> None:
        """Validate an edited setting and store it in the project (unsaved until File > Save).

        Only the edited field is converted back to solver units, so the others
        keep their exact stored values.

        Args:
            name: The :class:`SolveSettings` attribute that was edited.
            value: The new value, in display units.
        """
        if self._document.project is None:
            return
        factor = next(f for n, _l, _u, f, _r, _d, _t in _SETTING_FIELDS if n == name)
        self._pending = dataclasses.replace(self._pending, **{name: value * factor})
        try:
            self._document.set_solve_settings(self._pending)
        except ConfigError as exc:
            self.settings_error.setText(str(exc))
            self.settings_error.show()
        else:
            self.settings_error.hide()
        self.refresh()

    def set_busy(self, busy: bool) -> None:
        """Disable solving while a solve runs.

        Args:
            busy: True while the solver is running.
        """
        self._busy = busy
        self.refresh()

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

    @property
    def log_scale_selected(self) -> bool:
        """Whether a log colour scale is requested (applies to current density only)."""
        return self.log_scale.isChecked()

    def _update_log_enabled(self) -> None:
        """Enable the log-scale option only for quantities that support it."""
        self.log_scale.setEnabled(self._document.results is not None and supports_log(self.display_mode))

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
        """Update the button state, status text and legend visibility.

        Args:
            *_args: Ignored signal arguments.
        """
        doc = self._document
        board = doc.board
        sources = [s for s in doc.current_sources if s.enabled]
        balanced = abs(sum(s.current_a for s in sources)) <= _BALANCE_TOLERANCE_A
        valid = self.settings_error.isHidden()
        can_solve = (
            board is not None
            and board.has_outline
            and bool(sources or doc.active_heat_loads)
            and balanced
            and valid
            and not self._busy
        )
        self.solve_button.setEnabled(can_solve)

        style, lines = "", []
        if doc.project is None:
            lines.append("No project open.")
        elif board is None:
            lines.append("No board loaded.")
        elif not board.has_outline:
            style = _WARNING_STYLE
            lines.append("The board has no Edge.Cuts outline, so it can't be solved.")
        elif not sources and not doc.active_heat_loads:
            lines.append(
                "Add or enable current sources or heat loads (Current Sources / Heat Loads tabs) to solve."
            )
        elif not balanced:
            style = _WARNING_STYLE
            lines.append("Current sources and sinks must sum to zero to solve.")

        results = doc.results
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
            if doc.results_from_file:
                lines.append("Reloaded from the project's saved results.")
            staleness = doc.results_staleness
            if staleness:
                style = _WARNING_STYLE
                lines.append(f"{staleness}; solve again to update.")
        elif doc.results_notice and board is not None:
            style = _WARNING_STYLE
            lines.append(f"{doc.results_notice}, so they weren't loaded. Solve again.")
        self.status.setStyleSheet(style)
        self.status.setText("\n".join(lines))

        has_results = results is not None
        for widget in (self.mode, self.show_results, self.legend):
            widget.setEnabled(has_results)
        self._update_log_enabled()
        self.legend.setVisible(has_results)
        self.settings_group.setEnabled(doc.project is not None and not self._busy)
