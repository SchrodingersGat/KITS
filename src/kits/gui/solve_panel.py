"""Dock panel to run a solve and edit the material and environment settings it uses.

The results themselves are shown in the Results panel (:mod:`kits.gui.results_panel`).
"""

from __future__ import annotations

import dataclasses

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..config import ConfigError, SolveSettings
from . import colors
from .document import Document

_BALANCE_TOLERANCE_A = 1e-9
VIA_FILL_LABELS = {"none": "Unfilled", "resin": "Resin", "copper": "Copper"}
_WARNING_STYLE = "color: #e0a030;"
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
        "via_fill_thermal_conductivity_w_mk",
        "Resin fill k",
        " W/mK",
        1.0,
        (0.01, 50.0),
        2,
        "Thermal conductivity of a resin via fill (plain plugging epoxy ≈ 0.3; thermally\n"
        "conductive fills 1-3 or more). Resin doesn't conduct current.",
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
    (
        "solder_mask_thermal_conductivity_w_mk",
        "Solder mask k",
        " W/mK",
        1.0,
        (0.01, 10.0),
        3,
        "Thermal conductivity of the solder mask (LPI ≈ 0.2-0.3). With the mask thickness from the\n"
        "stackup it adds resistance to convection where the mask covers the board.",
    ),
)


class SolvePanel(QWidget):
    """Runs the solver, says what's needed to solve, and edits the solver settings.

    Signals:
        solve_requested(): The user asked to solve the board.
    """

    solve_requested = Signal()

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
            if name == "plating_thickness_mm":
                settings_form.addRow("Via fill:", self._build_via_fill())
        self.settings_error = QLabel()
        self.settings_error.setStyleSheet(_ERROR_STYLE)
        self.settings_error.setWordWrap(True)
        self.settings_error.hide()
        settings_form.addRow(self.settings_error)
        self.settings_group = QGroupBox("Materials && environment")
        self.settings_group.setLayout(settings_form)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.solve_button)
        layout.addWidget(self.status)
        layout.addWidget(self.settings_group)
        layout.addStretch(1)

        for signal in (
            document.project_changed,
            document.board_changed,
            document.sources_changed,
            document.heat_loads_changed,
            document.elements_changed,
            document.stackup_changed,
            document.heatsinks_changed,
            document.settings_changed,
            document.results_changed,
        ):
            signal.connect(self.refresh)
        document.project_changed.connect(self._load_settings)
        self._load_settings()
        self.refresh()

    def _build_via_fill(self) -> QComboBox:
        """Create the via fill selector.

        Returns:
            The combo box.
        """
        self.via_fill = QComboBox()
        for fill, label in VIA_FILL_LABELS.items():
            self.via_fill.addItem(label, fill)
        self.via_fill.setToolTip(
            "What fills the vias inside their plating. Resin conducts heat (at the resin k below)\n"
            "but not current; copper conducts both, like a solid copper core. Holes of\n"
            "through-hole pads are never filled."
        )
        self.via_fill.currentIndexChanged.connect(self._on_via_fill_changed)
        return self.via_fill

    def _on_via_fill_changed(self, _index: int) -> None:
        """Store the chosen via fill in the project.

        Args:
            _index: The selected index (the fill is read back from the box).
        """
        self._update_resin_enabled()
        if self._document.project is None:
            return
        self._pending = dataclasses.replace(self._pending, via_fill=self.via_fill.currentData())
        try:
            self._document.set_solve_settings(self._pending)
        except ConfigError as exc:
            self.settings_error.setText(str(exc))
            self.settings_error.show()
        else:
            self.settings_error.hide()
        self.refresh()

    def _update_resin_enabled(self) -> None:
        """Enable the resin conductivity only for a resin fill."""
        self.setting_boxes["via_fill_thermal_conductivity_w_mk"].setEnabled(
            self.via_fill.currentData() == "resin"
        )

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
        self.via_fill.blockSignals(True)
        self.via_fill.setCurrentIndex(self.via_fill.findData(settings.via_fill))
        self.via_fill.blockSignals(False)
        self._update_resin_enabled()
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

    def refresh(self, *_args) -> None:
        """Update the button state and the status text.

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
        elif doc.results is not None and doc.results_staleness:
            style = _WARNING_STYLE
            lines.append(f"{doc.results_staleness}; solve again to update the results.")
        elif doc.results is not None:
            lines.append("Solved: see the Results tab.")
        else:
            lines.append("Ready to solve.")
        self.status.setStyleSheet(style)
        self.status.setText("\n".join(lines))
        self.settings_group.setEnabled(doc.project is not None and not self._busy)
