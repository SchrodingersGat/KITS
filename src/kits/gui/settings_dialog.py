"""Preferences dialog editing :class:`AppSettings`."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QVBoxLayout,
)

from .settings import MAX_GRID_SPACING_MM, MIN_GRID_SPACING_MM, AppSettings


class SettingsDialog(QDialog):
    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self._settings = settings

        self.grid_visible = QCheckBox("Show grid")
        self.grid_visible.setChecked(settings.grid_visible)

        self.grid_spacing = QDoubleSpinBox()
        self.grid_spacing.setRange(MIN_GRID_SPACING_MM, MAX_GRID_SPACING_MM)
        self.grid_spacing.setDecimals(3)
        self.grid_spacing.setSuffix(" mm")
        self.grid_spacing.setValue(settings.grid_spacing_mm)

        grid_form = QFormLayout()
        grid_form.addRow(self.grid_visible)
        grid_form.addRow("Spacing:", self.grid_spacing)
        grid_box = QGroupBox("Grid")
        grid_box.setLayout(grid_form)

        self.cursor_visible = QCheckBox("Show crosshair cursor")
        self.cursor_visible.setChecked(settings.cursor_visible)
        cursor_form = QFormLayout()
        cursor_form.addRow(self.cursor_visible)
        cursor_box = QGroupBox("Cursor")
        cursor_box.setLayout(cursor_form)

        button = QDialogButtonBox.StandardButton
        buttons = QDialogButtonBox(button.Ok | button.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(grid_box)
        layout.addWidget(cursor_box)
        layout.addWidget(buttons)

    def accept(self) -> None:
        self._settings.grid_spacing_mm = self.grid_spacing.value()
        self._settings.grid_visible = self.grid_visible.isChecked()
        self._settings.cursor_visible = self.cursor_visible.isChecked()
        super().accept()
