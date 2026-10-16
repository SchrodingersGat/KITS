"""Dock panel for mesh settings, building the mesh and showing its status."""

from __future__ import annotations

import dataclasses
from collections import Counter

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..config import MIN_CIRCLE_POINTS_RANGE, ConfigError, MeshSettings
from . import colors
from .document import Document
from .mesh_areas import MeshAreasWidget

TRACK_MODE_LABELS = {
    "all": "All tracks",
    "none": "No tracks (zones, pads, vias only)",
    "nets": "Only tracks on selected nets",
}
_NET_ROLE = Qt.ItemDataRole.UserRole

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

    The track choice and mesh areas (see :attr:`areas`) simplify or focus the
    mesh: leaving out tracks, refining some rectangles, excluding others.
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
        self.circle_points = QSpinBox()
        self.circle_points.setRange(*MIN_CIRCLE_POINTS_RANGE)
        self.circle_points.setKeyboardTracking(False)
        self.circle_points.setToolTip(
            "Minimum number of mesh nodes around each drill hole (vias, plated and non-plated holes).\n"
            "Holes are cut out of the mesh; this takes precedence over the min size for small holes."
        )
        self.circle_points.valueChanged.connect(self._on_size_changed)
        self.corner_size = _size_spinbox()
        self.corner_size.setMinimum(0.0)
        self.corner_size.setSpecialValueText("Off")
        self.corner_size.setToolTip(
            "Element size at concave copper corners (e.g. where a track joins a pour), where\n"
            "current crowds. Graded back to the max size within about a millimetre. 0 = off."
        )
        self.corner_size.valueChanged.connect(self._on_size_changed)

        self.track_mode = QComboBox()
        for mode, label in TRACK_MODE_LABELS.items():
            self.track_mode.addItem(label, mode)
        self.track_mode.setToolTip(
            "Which routed tracks and arcs are meshed. Leaving out signal tracks simplifies the mesh;\n"
            "zone fills, pads, vias and copper graphics are always meshed."
        )
        self.track_mode.currentIndexChanged.connect(self._on_size_changed)
        self.net_filter = QLineEdit()
        self.net_filter.setPlaceholderText("Filter nets…")
        self.net_filter.setClearButtonEnabled(True)
        self.net_filter.textChanged.connect(self._filter_nets)
        self.nets = QListWidget()
        self.nets.setToolTip("Nets whose tracks are meshed (track segment counts in brackets)")
        self.nets.itemChanged.connect(self._on_size_changed)
        self.areas = MeshAreasWidget(document)

        simplify = QGroupBox("Tracks")
        simplify_layout = QVBoxLayout(simplify)
        simplify_layout.addWidget(self.track_mode)
        simplify_layout.addWidget(self.net_filter)
        simplify_layout.addWidget(self.nets)
        areas = QGroupBox("Mesh areas")
        areas.setToolTip("Rectangles meshed more finely, or left out of the mesh, on every copper layer")
        QVBoxLayout(areas).addWidget(self.areas)

        self.build_button = QPushButton("Build mesh")
        colors.style_action_button(self.build_button, colors.MESH_BUTTON)
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
        form.addRow("Hole points:", self.circle_points)
        form.addRow("Corner size:", self.corner_size)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addLayout(form)
        layout.addWidget(simplify)
        layout.addWidget(areas)
        layout.addWidget(self.settings_error)
        layout.addWidget(self.build_button)
        layout.addWidget(self.show_mesh)
        layout.addWidget(self.status)
        layout.addStretch(1)

        document.project_changed.connect(self._load_settings)
        document.board_changed.connect(self._load_nets)
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
            The (possibly invalid) settings, with the project's mesh areas
            (edited through :attr:`areas`).
        """
        project = self._document.project
        return dataclasses.replace(
            project.mesh if project else MeshSettings(),
            max_size_mm=self.max_size.value(),
            min_size_mm=self.min_size.value(),
            min_circle_points=self.circle_points.value(),
            corner_size_mm=self.corner_size.value(),
            tracks=self.track_mode.currentData(),
            track_nets=tuple(self._checked_nets()),
        )

    def _checked_nets(self) -> list[str]:
        """The nets ticked in the list, in list order.

        Returns:
            Net names.
        """
        items = (self.nets.item(i) for i in range(self.nets.count()))
        return [item.data(_NET_ROLE) for item in items if item.checkState() == Qt.CheckState.Checked]

    def _load_nets(self, *_args) -> None:
        """List the board's nets that have tracks, ticking the project's chosen ones.

        Chosen nets the board no longer has are listed too, so they aren't lost.

        Args:
            *_args: Ignored signal arguments.
        """
        board = self._document.board
        project = self._document.project
        chosen = set(project.mesh.track_nets) if project else set()
        counts = Counter(
            t.net for t in [*(board.tracks if board else []), *(board.arcs if board else [])] if t.net
        )
        self.nets.blockSignals(True)
        self.nets.clear()
        for net in sorted(set(counts) | chosen, key=str.casefold):
            text = f"{net} ({counts[net]})" if net in counts else f"{net} (not on the board)"
            item = QListWidgetItem(text)
            item.setData(_NET_ROLE, net)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if net in chosen else Qt.CheckState.Unchecked)
            self.nets.addItem(item)
        self.nets.blockSignals(False)
        self._filter_nets(self.net_filter.text())

    def _filter_nets(self, text: str) -> None:
        """Show only nets whose name contains the filter text (ticked nets always show).

        Args:
            text: The filter, case-insensitive.
        """
        needle = text.strip().casefold()
        for i in range(self.nets.count()):
            item = self.nets.item(i)
            shown = not needle or needle in item.data(_NET_ROLE).casefold()
            item.setHidden(not (shown or item.checkState() == Qt.CheckState.Checked))

    def _load_settings(self, *_args) -> None:
        """Show the open project's saved settings.

        Args:
            *_args: Ignored signal arguments.
        """
        project = self._document.project
        settings = project.mesh if project else MeshSettings()
        values = (
            (self.max_size, settings.max_size_mm),
            (self.min_size, settings.min_size_mm),
            (self.circle_points, settings.min_circle_points),
            (self.corner_size, settings.corner_size_mm),
        )
        for box, value in values:
            box.blockSignals(True)
            box.setValue(value)
            box.blockSignals(False)
        self.track_mode.blockSignals(True)
        self.track_mode.setCurrentIndex(self.track_mode.findData(settings.tracks))
        self.track_mode.blockSignals(False)
        self._load_nets()
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
        editable = doc.project is not None and not self._busy
        for box in (self.max_size, self.min_size, self.circle_points, self.corner_size, self.track_mode):
            box.setEnabled(editable)
        by_net = self.track_mode.currentData() == "nets"
        self.net_filter.setVisible(by_net)
        self.nets.setVisible(by_net)
        self.nets.setEnabled(editable)
        self.areas.setEnabled(editable)

        style, lines = "", []
        if board is None:
            lines.append("No board loaded." if doc.project else "No project open.")
        elif not board.has_outline:
            style = _WARNING_STYLE
            lines.append("The board has no Edge.Cuts outline, so it can't be meshed.")
        elif doc.mesh is None and doc.mesh_notice:
            style = _WARNING_STYLE
            lines.append(f"{doc.mesh_notice}, so it wasn't loaded. Rebuild the mesh.")
        elif doc.mesh is None:
            lines.append("No mesh built yet.")
        else:
            m = doc.mesh
            lines.append(f"{m.num_triangles:,} triangles, {m.num_nodes:,} nodes on {len(m.layers)} layers.")
            lines += [f"  {lm.layer}: {lm.num_triangles:,} triangles" for lm in m.layers]
            if doc.mesh_from_file:
                lines.append("Reloaded from the project's saved mesh.")
            if m.msh_path:
                lines.append(f"Saved to {m.msh_path}")
            if doc.mesh_is_stale:
                style = _WARNING_STYLE
                lines.append("Settings changed since this mesh was built; rebuild to apply them.")
        self.status.setStyleSheet(style)
        self.status.setText("\n".join(lines))
