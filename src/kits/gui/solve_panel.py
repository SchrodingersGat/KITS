"""Dock panel to run a solve and control how its results are shown."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QLinearGradient, QPainter, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .document import Document
from .field_display import CURRENT_DENSITY_PERCENTILE, DisplayMode, colormap, format_value

_BALANCE_TOLERANCE_A = 1e-9
_WARNING_STYLE = "color: #e0a030;"
_LEGEND_SIZE = (220, 14)


class SolvePanel(QWidget):
    """Runs the solver and picks the quantity shown on the board.

    Signals:
        solve_requested(): The user asked to solve the board.
        mode_changed(DisplayMode): The displayed quantity changed.
        show_results_toggled(bool): The results overlay was toggled.
    """

    solve_requested = Signal()
    mode_changed = Signal(object)
    show_results_toggled = Signal(bool)

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
        self.solve_button.setToolTip("Solve current flow and temperature for the current sources (F9)")
        self.solve_button.clicked.connect(self.solve_requested)

        self.mode = QComboBox()
        for mode in DisplayMode:
            self.mode.addItem(mode.label, mode)
        self.mode.currentIndexChanged.connect(lambda _i: self.mode_changed.emit(self.display_mode))
        self.show_results = QCheckBox("Show results")
        self.show_results.setChecked(True)
        self.show_results.toggled.connect(self.show_results_toggled)

        self.legend_bar = QLabel()
        self.legend_low = QLabel()
        self.legend_high = QLabel()
        self.legend_high.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.legend_note = QLabel()
        self.legend_note.setWordWrap(True)
        self.legend_note.setEnabled(False)
        limits = QHBoxLayout()
        limits.addWidget(self.legend_low)
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
        self.settings_note = QLabel()
        self.settings_note.setWordWrap(True)
        self.settings_note.setEnabled(False)

        form = QFormLayout()
        form.addRow("Display:", self.mode)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.solve_button)
        layout.addWidget(self.status)
        layout.addLayout(form)
        layout.addWidget(self.show_results)
        layout.addWidget(self.legend)
        layout.addWidget(self.settings_note)
        layout.addStretch(1)

        for signal in (
            document.project_changed,
            document.board_changed,
            document.sources_changed,
            document.settings_changed,
            document.results_changed,
        ):
            signal.connect(self.refresh)
        self.refresh()

    @property
    def display_mode(self) -> DisplayMode:
        """The quantity selected for display."""
        return self.mode.currentData()

    def set_busy(self, busy: bool) -> None:
        """Disable solving while a solve runs.

        Args:
            busy: True while the solver is running.
        """
        self._busy = busy
        self.refresh()

    def set_legend(self, mode: DisplayMode, low: float, high: float) -> None:
        """Draw the colour scale for the displayed quantity.

        Args:
            mode: The displayed quantity.
            low: Value at the bottom of the scale, in display units.
            high: Value at the top of the scale, in display units.
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
        self.legend_low.setText(format_value(low, mode))
        high_text = format_value(high, mode)
        if mode is DisplayMode.CURRENT_DENSITY:
            high_text = f"≥ {high_text}"
            self.legend_note.setText(
                f"Scale tops out at the {CURRENT_DENSITY_PERCENTILE:g}th percentile: sharp inside "
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
        sources = doc.current_sources
        balanced = abs(sum(s.current_a for s in sources)) <= _BALANCE_TOLERANCE_A
        can_solve = board is not None and board.has_outline and bool(sources) and balanced and not self._busy
        self.solve_button.setEnabled(can_solve)

        style, lines = "", []
        if doc.project is None:
            lines.append("No project open.")
        elif board is None:
            lines.append("No board loaded.")
        elif not board.has_outline:
            style = _WARNING_STYLE
            lines.append("The board has no Edge.Cuts outline, so it can't be solved.")
        elif not sources:
            lines.append("Add current sources (Current Sources tab) to solve.")
        elif not balanced:
            style = _WARNING_STYLE
            lines.append("Current sources and sinks must sum to zero to solve.")

        results = doc.results
        if results is not None:
            lines.append(f"Joule power: {results.total_power_w:.4g} W")
            lines.append(f"Max temperature: {results.max_temperature_c:.1f} °C")
            for label, voltage in sorted(results.electrode_voltages.items()):
                lines.append(f"  {label}: {format_value(voltage, DisplayMode.VOLTAGE)}")
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
        self.legend.setVisible(has_results)
        s = doc.solve_settings
        self.settings_note.setText(
            f"Materials (defaults): copper {s.copper_conductivity_s_m:.3g} S/m, "
            f"plating {s.plating_thickness_mm * 1000:g} µm; "
            f"dielectric {s.dielectric_thermal_conductivity_w_mk:g} W/mK; "
            f"convection {s.convection_w_m2k:g} W/m²K to {s.ambient_c:g} °C ambient."
        )
