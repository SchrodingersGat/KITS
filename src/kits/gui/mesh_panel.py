"""Dock panel for mesh settings, building the mesh and showing its status."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..config import ConfigError, MeshSettings
from .document import Document

_MIN_SIZE_MM = 0.001
_MAX_SIZE_MM = 100.0
_ERROR_STYLE = "color: #e05050;"
_WARNING_STYLE = "color: #e0a030;"


def _size_spinbox() -> QDoubleSpinBox:
    """Create a millimetre spin box for a mesh size.

    Returns:
        The configured spin box.
    """
    box = QDoubleSpinBox()
    box.setRange(_MIN_SIZE_MM, _MAX_SIZE_MM)
    box.setDecimals(3)
    box.setSingleStep(0.05)
    box.setSuffix(" mm")
    box.setKeyboardTracking(False)  # only commit when editing finishes, not per keystroke
    return box


class MeshPanel(QWidget):
    """Edits the project's mesh settings and triggers mesh builds.

    Edited settings are stored in the open project and written by File > Save
    (building a mesh saves first, since the mesher reads the project file).

    Signals:
        build_requested(): The user asked to build (or rebuild) the mesh.
        show_mesh_toggled(bool): The user toggled the mesh overlay.
    """

    build_requested = Signal()
    show_mesh_toggled = Signal(bool)

    def __init__(self, document: Document, parent=None):
        """Create the panel.

        Args:
            document: The application document whose project settings and mesh are shown.
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self._document = document
        self._busy = False

        self.max_size = _size_spinbox()
        self.min_size = _size_spinbox()
        self.max_size.setToolTip(
            "Largest element size. gmsh treats it as a target, so some edges come out slightly longer."
        )
        self.min_size.setToolTip(
            "Smallest element size gmsh may choose. Geometry finer than this (short outline segments)\n"
            "still gets matching small elements. Also sets the geometry simplification tolerance (10%)."
        )
        self.max_size.valueChanged.connect(self._on_size_changed)
        self.min_size.valueChanged.connect(self._on_size_changed)

        self.build_button = QPushButton("Build mesh")
        self.build_button.clicked.connect(self.build_requested)
        self.show_mesh = QCheckBox("Show mesh")
        self.show_mesh.setChecked(True)
        self.show_mesh.toggled.connect(self.show_mesh_toggled)

        self.settings_error = QLabel()
        self.settings_error.setStyleSheet(_ERROR_STYLE)
        self.settings_error.setWordWrap(True)
        self.settings_error.hide()
        self.status = QLabel()
        self.status.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Max size:", self.max_size)
        form.addRow("Min size:", self.min_size)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addLayout(form)
        layout.addWidget(self.settings_error)
        layout.addWidget(self.build_button)
        layout.addWidget(self.show_mesh)
        layout.addWidget(self.status)
        layout.addStretch(1)

        document.project_changed.connect(self._load_settings)
        document.board_changed.connect(self._refresh)
        document.settings_changed.connect(self._refresh)
        document.mesh_changed.connect(self._refresh)
        self._load_settings()

    def set_busy(self, busy: bool) -> None:
        """Disable editing and building while a mesh job runs.

        Args:
            busy: True while meshing is in progress.
        """
        self._busy = busy
        self._refresh()

    def settings(self) -> MeshSettings:
        """Return the settings currently entered in the panel.

        Returns:
            The (possibly invalid) settings.
        """
        return MeshSettings(max_size_mm=self.max_size.value(), min_size_mm=self.min_size.value())

    def _load_settings(self, *_args) -> None:
        """Show the open project's saved settings.

        Args:
            *_args: Ignored signal arguments.
        """
        project = self._document.project
        settings = project.mesh if project else MeshSettings()
        for box, value in ((self.max_size, settings.max_size_mm), (self.min_size, settings.min_size_mm)):
            box.blockSignals(True)
            box.setValue(value)
            box.blockSignals(False)
        self.settings_error.hide()
        self._refresh()

    def _on_size_changed(self, _value: float) -> None:
        """Validate edited sizes and store them in the project (unsaved until File > Save).

        Args:
            _value: The new spin box value (both boxes are re-read).
        """
        if self._document.project is None:
            return
        try:
            self._document.set_mesh_settings(self.settings())
        except ConfigError as exc:
            self.settings_error.setText(str(exc))
            self.settings_error.show()
        else:
            self.settings_error.hide()
        self._refresh()

    def _refresh(self, *_args) -> None:
        """Update button state and the status text.

        Args:
            *_args: Ignored signal arguments.
        """
        doc = self._document
        board = doc.board
        valid = self.settings_error.isHidden()
        can_build = board is not None and board.has_outline and valid and not self._busy
        self.build_button.setEnabled(can_build)
        self.build_button.setText("Rebuild mesh" if doc.mesh else "Build mesh")
        for box in (self.max_size, self.min_size):
            box.setEnabled(doc.project is not None and not self._busy)

        style, lines = "", []
        if board is None:
            lines.append("No board loaded." if doc.project else "No project open.")
        elif not board.has_outline:
            style = _WARNING_STYLE
            lines.append("The board has no Edge.Cuts outline, so it can't be meshed.")
        elif doc.mesh is None:
            lines.append("No mesh built yet.")
        else:
            m = doc.mesh
            lines.append(f"{m.num_triangles:,} triangles, {m.num_nodes:,} nodes on {len(m.layers)} layers.")
            lines += [f"  {lm.layer}: {lm.num_triangles:,} triangles" for lm in m.layers]
            if m.msh_path:
                lines.append(f"Saved to {m.msh_path}")
            if doc.mesh_is_stale:
                style = _WARNING_STYLE
                lines.append("Settings changed since this mesh was built; rebuild to apply them.")
        self.status.setStyleSheet(style)
        self.status.setText("\n".join(lines))
