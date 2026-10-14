"""3D view of the board: the stackup (copper layers and dielectric slabs), holes and plated barrels.

Geometry is built with shapely and turned into PyVista meshes by
:func:`board_parts`, which needs no window (and is what the tests check).
:class:`Board3DView` shows them in a VTK render window (pyvistaqt), created the
first time the view is shown so the 2D app doesn't pay for VTK at startup.
Importing VTK takes a second or more (far longer on a cold start), and building
the solids of a large board takes a while too, so both happen on background
threads: the window stays responsive and the view says what it is waiting for.

Coordinates: x as in KiCad, y flipped (KiCad's y points down the page) and z up
from the board's bottom face, all in millimetres; z can be exaggerated.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import shapely
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from ..config import MeshSettings, SolveSettings
from ..ingest.model import Board
from ..mesh.geometry import GeometryError, board_holes, board_outline, layer_copper, via_layers
from ..mesh.mesher import layer_heights
from . import colors

DIELECTRIC_COLORS = [QColor(70, 120, 60), QColor(95, 140, 70)]  # alternating slabs (FR4 green)
DIELECTRIC_OPACITY = 0.2
BARREL_COLOR = QColor(214, 150, 74)  # plated copper
MAX_Z_SCALE = 50.0
_DEFAULT_COPPER_MM = 0.035


@dataclass
class Part:
    """One coloured solid in the 3D view.

    Attributes:
        key: Unique name, e.g. ``"copper:F.Cu"``, ``"dielectric:F.Cu-B.Cu"`` or ``"barrels"``.
        layer: Copper layer whose visibility toggle hides it, or None if always shown.
        mesh: The surface (a ``pyvista.PolyData``).
        color: Fill colour.
        opacity: 0 (invisible) to 1 (opaque).
    """

    key: str
    layer: str | None
    mesh: object
    color: QColor
    opacity: float = 1.0


def _polygons(geom: BaseGeometry) -> list[Polygon]:
    """The non-empty polygons in a geometry.

    Args:
        geom: Any geometry.

    Returns:
        Its polygon parts.
    """
    return [p for p in shapely.get_parts(geom) if isinstance(p, Polygon) and not p.is_empty and p.area > 0]


def extrude(geom: BaseGeometry, z0: float, z1: float):
    """Extrude a planar area into a closed prism surface.

    Caps are triangulated with constrained Delaunay triangulation (holes are
    respected); walls are one quad per outline edge. Points aren't shared
    between faces, which gives flat shading with crisp edges.

    Args:
        geom: Area in board millimetres (KiCad orientation; y is flipped here).
        z0: Bottom height.
        z1: Top height.

    Returns:
        A ``pyvista.PolyData``, or None if the area is empty.
    """
    import pyvista as pv

    polys = _polygons(geom)
    if not polys:
        return None
    triangles = shapely.get_parts(shapely.constrained_delaunay_triangles(polys))
    tri = shapely.get_coordinates(triangles).reshape(-1, 4, 2)[:, :3]  # drop each ring's closing point
    rings = shapely.get_rings(polys)
    ring_xy, ring_index = shapely.get_coordinates(rings, return_index=True)
    same_ring = ring_index[:-1] == ring_index[1:]
    a, b = ring_xy[:-1][same_ring], ring_xy[1:][same_ring]

    n_tri, n_side = len(tri), len(a)
    flat = tri.reshape(-1, 2)
    points = np.concatenate(
        [
            np.column_stack([flat, np.full(len(flat), z0)]),
            np.column_stack([flat, np.full(len(flat), z1)]),
            np.column_stack([a, np.full(n_side, z0)]),
            np.column_stack([b, np.full(n_side, z0)]),
            np.column_stack([b, np.full(n_side, z1)]),
            np.column_stack([a, np.full(n_side, z1)]),
        ]
    )
    points[:, 1] *= -1  # KiCad y points down

    base = np.arange(n_tri) * 3
    bottom = np.column_stack([np.full(n_tri, 3), base, base + 2, base + 1])
    top_start = 3 * n_tri
    top = np.column_stack([np.full(n_tri, 3), top_start + base, top_start + base + 1, top_start + base + 2])
    side_start = 6 * n_tri
    k = np.arange(n_side)
    sides = np.column_stack(
        [
            np.full(n_side, 4),
            side_start + k,
            side_start + n_side + k,
            side_start + 2 * n_side + k,
            side_start + 3 * n_side + k,
        ]
    )
    faces = np.concatenate([bottom.ravel(), top.ravel(), sides.ravel()])
    return pv.PolyData(points, faces)


def layer_z_ranges(board: Board, z_scale: float = 1.0) -> dict[str, tuple[float, float]]:
    """Bottom and top of each copper layer in the 3D view.

    Args:
        board: The board.
        z_scale: Vertical exaggeration.

    Returns:
        ``{layer: (z0, z1)}`` in scaled millimetres above the board bottom.
    """
    heights = layer_heights(board)
    thickness = {name: mm for name, (mm, _) in board.copper_thicknesses().items()}
    ranges = {}
    for layer, z in heights.items():
        half = thickness.get(layer, _DEFAULT_COPPER_MM) / 2
        ranges[layer] = ((z - half) * z_scale, (z + half) * z_scale)
    return ranges


def _sheets(nodes: np.ndarray, triangles: np.ndarray, levels: tuple[float, ...]):
    """A layer's triangles as flat sheets at the given heights (y flipped, like the solids).

    Args:
        nodes: ``(N, 2)`` node coordinates in board millimetres.
        triangles: ``(M, 3)`` node indices.
        levels: Heights to place a copy at (e.g. just above and just below the copper).

    Returns:
        A ``pyvista.PolyData`` with ``M`` triangles per level, in level order.
    """
    import pyvista as pv

    points, faces = [], []
    for i, z in enumerate(levels):
        points.append(np.column_stack([nodes[:, 0], -nodes[:, 1], np.full(len(nodes), z)]))
        faces.append(np.column_stack([np.full(len(triangles), 3), triangles + i * len(nodes)]))
    return pv.PolyData(np.concatenate(points), np.concatenate(faces).ravel())


def overlay_parts(
    ranges: dict[str, tuple[float, float]],
    mesh=None,
    results=None,
    mode=None,
    log: bool = False,
) -> list[Part]:
    """Mesh wireframes and coloured results for each layer, laid on its copper faces.

    Each sheet is drawn just above the layer's top face and just below its
    bottom face, so it shows from either side and isn't hidden by the solid.
    Results use the same colour scale as the 2D view (shared by all layers).

    Args:
        ranges: :func:`layer_z_ranges` for the current Z scale.
        mesh: The board mesh (``BoardMesh``), or None.
        results: Solve results (``SolveResults``), or None.
        mode: Displayed quantity (``DisplayMode``) for results.
        log: Log colour scale (current density only).

    Returns:
        Parts keyed ``"mesh:<layer>"`` / ``"mesh-bare:<layer>"`` and
        ``"results:<layer>"``; results parts carry per-triangle RGB colours in
        ``cell_data["rgb"]``.
    """
    from .field_display import color_indices, colormap, supports_log, triangle_values, value_range

    if not ranges:
        return []
    span = max(z1 for _, z1 in ranges.values()) - min(z0 for z0, _ in ranges.values())
    gap = max(span, 1.0) * 2e-3  # lift off the copper faces to avoid z-fighting

    def levels(layer: str, lift: float) -> tuple[float, float]:
        """Heights just outside a layer's faces.

        Args:
            layer: Copper layer name.
            lift: Distance outside the faces.

        Returns:
            ``(below, above)``.
        """
        z0, z1 = ranges[layer]
        return (z0 - lift, z1 + lift)

    parts: list[Part] = []
    if results is not None and mode is not None:
        log = log and supports_log(mode)
        low, high = value_range(results, mode, log)
        palette = np.array([[c.red(), c.green(), c.blue()] for c in colormap(mode)], dtype=np.uint8)
        for layer in results.layers:
            if layer.layer not in ranges:
                continue
            values, shown = triangle_values(layer, mode)
            if not shown.any():
                continue
            sheet = _sheets(layer.nodes, layer.triangles[shown], levels(layer.layer, gap))
            rgb = palette[color_indices(values[shown], low, high, steps=len(palette), log=log)]
            sheet.cell_data["rgb"] = np.concatenate([rgb, rgb])
            parts.append(Part(f"results:{layer.layer}", layer.layer, sheet, QColor()))
    if mesh is not None:
        for layer_mesh in mesh.layers:
            if layer_mesh.layer not in ranges:
                continue
            for kind, mask, color in (
                ("mesh", layer_mesh.copper, colors.MESH_COPPER),
                ("mesh-bare", ~layer_mesh.copper, colors.MESH_BARE),
            ):
                if mask.any():
                    sheet = _sheets(
                        layer_mesh.nodes, layer_mesh.triangles[mask], levels(layer_mesh.layer, 2 * gap)
                    )
                    parts.append(
                        Part(f"{kind}:{layer_mesh.layer}", layer_mesh.layer, sheet, color, color.alphaF())
                    )
    return parts


def board_parts(
    board: Board,
    z_scale: float = 1.0,
    mesh_settings: MeshSettings | None = None,
    solve_settings: SolveSettings | None = None,
) -> list[Part]:
    """Build the 3D solids of a board.

    * one solid per copper layer, at its stackup height and thickness, with
      its drill holes cut out;
    * one translucent dielectric slab between each pair of adjacent copper
      layers, with the holes that pass through it cut out;
    * the plated barrels of vias and plated through-holes, between the layers
      they connect.

    Args:
        board: The board.
        z_scale: Vertical exaggeration (boards are thin compared with their size).
        mesh_settings: Hole outline resolution (defaults if None).
        solve_settings: Barrel plating thickness (defaults if None).

    Returns:
        The parts, copper layers top to bottom, then dielectric, then barrels.

    Raises:
        GeometryError: If the board has no usable outline.
    """
    mesh_settings = mesh_settings or MeshSettings()
    plating = (solve_settings or SolveSettings()).plating_thickness_mm
    outline = board_outline(board)
    if outline.is_empty:
        raise GeometryError("the board outline doesn't enclose any area")
    spans = via_layers(board)
    holes = board_holes(board, mesh_settings, spans)
    ranges = layer_z_ranges(board, z_scale)
    heights = ranges  # layers with a known height
    copper = [name for name in board.copper_layers if name in ranges]

    def z_range(layer: str) -> tuple[float, float]:
        """Bottom and top of a copper layer, scaled.

        Args:
            layer: Copper layer name.

        Returns:
            ``(z0, z1)``.
        """
        return ranges[layer]

    def cut(area: BaseGeometry, layers: set[str]) -> BaseGeometry:
        """Remove the holes that pass through all of the given layers.

        Args:
            area: The area to cut.
            layers: Layers the hole must span.

        Returns:
            The area with those holes removed.
        """
        through = [h.polygon for h in holes if layers <= set(h.layers)]
        return area.difference(shapely.union_all(through)) if through else area

    parts: list[Part] = []
    for layer in copper:
        area = cut(layer_copper(board, layer, spans).intersection(outline), {layer})
        mesh = extrude(area, *z_range(layer))
        if mesh is not None:
            parts.append(Part(f"copper:{layer}", layer, mesh, colors.copper_color(layer)))

    for i, (upper, lower) in enumerate(zip(copper, copper[1:], strict=False)):
        z0, z1 = z_range(lower)[1], z_range(upper)[0]
        mesh = extrude(cut(outline, {upper, lower}), z0, z1) if z1 > z0 else None
        if mesh is not None:
            color = DIELECTRIC_COLORS[i % len(DIELECTRIC_COLORS)]
            parts.append(Part(f"dielectric:{upper}-{lower}", None, mesh, color, DIELECTRIC_OPACITY))

    barrels = []
    for hole in holes:
        layers = [name for name in hole.layers if name in heights]
        if not hole.plated or not layers:
            continue
        z0 = min(z_range(name)[0] for name in layers)
        z1 = max(z_range(name)[1] for name in layers)
        ring = hole.polygon.buffer(plating).difference(hole.polygon)
        mesh = extrude(ring, z0, z1)
        if mesh is not None:
            barrels.append(mesh)
    if barrels:
        import pyvista as pv

        parts.append(Part("barrels", None, pv.merge(barrels), BARREL_COLOR))
    return parts


class _ViewerLoader:
    """Imports PyVista and pyvistaqt on a background thread, once."""

    def __init__(self):
        """Create the (not yet started) loader."""
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    def start(self) -> None:
        """Start importing, if not already started."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._load, name="kits-3d-viewer-import", daemon=True)
            self._thread.start()

    def _load(self) -> None:
        """Import the 3D libraries, recording any failure."""
        try:
            import pyvista  # noqa: F401
            import pyvistaqt  # noqa: F401
        except Exception as exc:  # ImportError, or a broken VTK install
            self.error = str(exc) or type(exc).__name__

    @property
    def done(self) -> bool:
        """Whether importing has finished (successfully or not)."""
        return self._thread is not None and not self._thread.is_alive()


_LOADER = _ViewerLoader()


def prime_window_for_3d(window: QWidget) -> QWidget | None:
    """Make a top-level window OpenGL-composited from the start, if the 3D view is available.

    The 3D view is a ``QOpenGLWidget``. Adding the first one to a top-level
    window that is already on screen makes Qt switch the window to OpenGL
    composition by destroying and re-creating its native window, which looks
    like the application closing and re-opening. A hidden, empty
    ``QOpenGLWidget`` added before the window is first shown avoids that; the
    price is that OpenGL is set up at startup (a fraction of a second) rather
    than on the first switch to 3D.

    Call it before ``window.show()``. Does nothing when PyVista/pyvistaqt aren't
    installed (the 3D view then only shows an install hint).

    Args:
        window: The top-level window that will contain the 3D view.

    Returns:
        The hidden placeholder widget, or None if not added.
    """
    import importlib.util

    if importlib.util.find_spec("pyvistaqt") is None:
        return None
    from PySide6.QtOpenGLWidgets import QOpenGLWidget

    primer = QOpenGLWidget(window)
    primer.setObjectName("OpenGLPrimer")
    primer.setFixedSize(0, 0)
    primer.hide()
    return primer


def preload_viewer() -> None:
    """Start loading the 3D libraries in the background, so the 3D view opens quickly."""
    _LOADER.start()


def _rgb(color: QColor) -> tuple[float, float, float]:
    """A QColor as a 0–1 RGB tuple for VTK.

    Args:
        color: The colour.

    Returns:
        ``(r, g, b)``.
    """
    return (color.redF(), color.greenF(), color.blueF())


class Board3DView(QWidget):
    """Interactive 3D view of the board (rotate: left drag, pan: middle drag, zoom: wheel)."""

    def __init__(self, parent=None):
        """Create the view; the render window is made when it is first shown.

        Args:
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self._board: Board | None = None
        self._mesh_settings: MeshSettings | None = None
        self._solve_settings: SolveSettings | None = None
        self._plotter = None
        self._building = False  # creating the render window re-enters showEvent
        self._dirty = False
        self._fresh_camera = True
        self._hidden_layers: set[str] = set()
        self._actors: dict[str, tuple[object, str | None]] = {}
        self.parts: list[Part] = []
        self.overlays: list[Part] = []
        self._overlay_actors: dict[str, tuple[object, str | None]] = {}
        self._overlay_data: dict = {"mesh": None, "results": None, "mode": None, "log": False}
        self._show_mesh = True
        self._show_results = True
        self._pool: ThreadPoolExecutor | None = None
        self._pending: tuple[int, Future] | None = None  # (generation, board_parts job)
        self._generation = 0  # bumped by every rebuild; stale jobs are dropped
        self._waiting_for_viewer = False
        self._poll = QTimer(self)
        self._poll.setInterval(50)
        self._poll.timeout.connect(self._on_poll)

        self.z_scale = QDoubleSpinBox()
        self.z_scale.setRange(1.0, MAX_Z_SCALE)
        self.z_scale.setDecimals(0)
        self.z_scale.setSuffix(" ×")
        self.z_scale.setValue(2.0)
        self.z_scale.setToolTip("Vertical exaggeration: boards are thin next to their size")
        self.z_scale.setKeyboardTracking(False)
        self.z_scale.valueChanged.connect(lambda _v: self._rebuild())
        self.show_dielectric = QCheckBox("Dielectric")
        self.show_dielectric.setChecked(True)
        self.show_dielectric.setToolTip("Show the translucent board material between copper layers")
        self.show_dielectric.toggled.connect(lambda _on: self._apply_visibility())
        self.reset_button = QPushButton("Reset view")
        self.reset_button.clicked.connect(self.reset_camera)
        self.message = QLabel()
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message.setWordWrap(True)
        self.message.hide()
        self.status = QLabel()
        self.status.setEnabled(False)

        controls = QHBoxLayout()
        controls.setContentsMargins(6, 4, 6, 4)
        controls.addWidget(QLabel("Z scale:"))
        controls.addWidget(self.z_scale)
        controls.addWidget(self.show_dielectric)
        controls.addWidget(self.reset_button)
        controls.addWidget(self.status, 1)
        hint = QLabel("Rotate: left drag · Pan: middle drag · Zoom: wheel")
        hint.setEnabled(False)
        for label in (hint, self.status):
            # Clip rather than widen: the stacked viewport would otherwise set the
            # whole window's minimum width, even while the 2D view is showing.
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        controls.addWidget(hint, 1)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._layout.addLayout(controls)
        self._layout.addWidget(self.message, 1)

    # -- inputs ------------------------------------------------------------------------

    def set_board(
        self,
        board: Board | None,
        mesh_settings: MeshSettings | None = None,
        solve_settings: SolveSettings | None = None,
    ) -> None:
        """Show a board (rebuilt now if the view is visible, otherwise when next shown).

        Args:
            board: The board, or None to clear the view.
            mesh_settings: Hole outline resolution.
            solve_settings: Barrel plating thickness.
        """
        if board is None or self._board is None or board.path != self._board.path:
            self._hidden_layers.clear()  # the Layers panel starts with every layer shown
            self._fresh_camera = True
        self._board = board
        self._mesh_settings = mesh_settings
        self._solve_settings = solve_settings
        self._dirty = True
        if self.isVisible():
            self._rebuild()

    def set_overlays(self, mesh=None, results=None, mode=None, log: bool = False) -> None:
        """Set the mesh and results drawn on the copper layers (as in the 2D view).

        Args:
            mesh: The board mesh (``BoardMesh``), or None.
            results: Solve results (``SolveResults``), or None.
            mode: Displayed quantity (``DisplayMode``).
            log: Log colour scale (current density only).
        """
        data = {"mesh": mesh, "results": results, "mode": mode, "log": log}
        old = self._overlay_data
        if all(data[k] is old[k] for k in ("mesh", "results")) and (mode, log) == (old["mode"], old["log"]):
            return  # unchanged (compared by identity: the arrays are large)
        self._overlay_data = data
        if self._plotter is not None and self.parts and not self.busy:
            self._show_overlays()

    def set_overlay_visible(self, mesh: bool, results: bool) -> None:
        """Show or hide the mesh and results overlays (follows the Show mesh / Show results toggles).

        Args:
            mesh: Whether the mesh wireframe is drawn.
            results: Whether the coloured results are drawn.
        """
        self._show_mesh, self._show_results = mesh, results
        self._apply_visibility()

    def set_layer_visible(self, layer: str, visible: bool) -> None:
        """Show or hide a copper layer (follows the Layers panel).

        Args:
            layer: Copper layer name.
            visible: Whether it is drawn.
        """
        if visible:
            self._hidden_layers.discard(layer)
        else:
            self._hidden_layers.add(layer)
        self._apply_visibility()

    @property
    def plotter(self):
        """The pyvistaqt ``QtInteractor``, or None until first shown (or if unavailable)."""
        return self._plotter

    @property
    def busy(self) -> bool:
        """Whether the view is still loading the 3D libraries or building the board."""
        return self._waiting_for_viewer or self._pending is not None

    def visible_part_keys(self) -> list[str]:
        """Keys of the parts currently drawn.

        Returns:
            Part keys in build order.
        """
        actors = {**self._actors, **self._overlay_actors}
        return [key for key, (actor, _) in actors.items() if actor.GetVisibility()]

    # -- rendering ------------------------------------------------------------------------

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Build the render window and scene the first time the view is shown.

        Args:
            event: The show event.
        """
        super().showEvent(event)
        if not self._building and (self._dirty or self._plotter is None):
            self._rebuild()

    def _ensure_plotter(self) -> bool:
        """Create the VTK render window once the 3D libraries have loaded.

        The first call only shows "Loading…" and checks back on a timer: the
        libraries may still be importing in the background, and even when they
        aren't, the graphics driver takes a second to set up OpenGL the first time
        (on this thread), so the message is painted before that pause. The
        rebuild then happens by itself.

        Returns:
            True if a plotter is available now.
        """
        if self._plotter is not None:
            return True
        _LOADER.start()
        if not (_LOADER.done and self._waiting_for_viewer):
            self._waiting_for_viewer = True
            self.message.setText("Loading the 3D viewer…")
            self.message.show()
            self._poll.start()
            return False
        self._waiting_for_viewer = False
        if _LOADER.error is not None:
            self.message.setText(
                f"The 3D view needs PyVista and pyvistaqt ({_LOADER.error}).\n\npip install pyvista pyvistaqt"
            )
            self.message.show()
            return False
        from pyvistaqt import QtInteractor  # already imported by the loader, so instant

        self._building = True
        try:
            plotter = QtInteractor()
            self._layout.addWidget(plotter, 1)
        finally:
            self._building = False
        self._plotter = plotter
        self._plotter.set_background(_rgb(colors.BACKGROUND))
        self._plotter.enable_anti_aliasing()
        self.message.hide()
        return True

    def _rebuild(self) -> None:
        """Start rebuilding the parts for the current board and settings, on a worker thread."""
        if not self._ensure_plotter():
            return
        self._dirty = False
        self._generation += 1
        if self._board is None:
            self._pending = None
            self._show_parts([])
            return
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kits-3d-build")
        job = self._pool.submit(
            board_parts, self._board, self.z_scale.value(), self._mesh_settings, self._solve_settings
        )
        self._pending = (self._generation, job)
        self.status.setText("Building 3D model…")
        self._poll.start()

    def _on_poll(self) -> None:
        """Pick up a finished library load or board build."""
        if self._waiting_for_viewer:
            if _LOADER.done:
                if self.isVisible():
                    self._rebuild()  # creates the render window now that "Loading…" is painted
                else:
                    self._waiting_for_viewer = False
                    self._dirty = True
            return
        if self._pending is None:
            self._poll.stop()
            return
        generation, job = self._pending
        if not job.done():
            return
        self._pending = None
        self._poll.stop()
        if generation != self._generation:  # superseded by a newer rebuild
            return
        try:
            parts = job.result()
        except GeometryError as exc:
            self._show_parts([])
            self.message.setText(f"Can't show the board in 3D: {exc}")
            self.message.show()
            return
        self._show_parts(parts)

    def _show_parts(self, parts: list[Part]) -> None:
        """Replace the scene's solids.

        Args:
            parts: The new parts (from :func:`board_parts`).
        """
        plotter = self._plotter
        plotter.clear()
        self._actors = {}
        self._overlay_actors = {}
        self.parts = parts
        self.status.setText("")
        self.message.hide()
        for part in parts:
            actor = plotter.add_mesh(
                part.mesh, color=_rgb(part.color), opacity=part.opacity, name=part.key, show_edges=False
            )
            self._actors[part.key] = (actor, part.layer)
        self._show_overlays(render=False)
        if parts and self._fresh_camera:
            self.reset_camera()
            self._fresh_camera = False
        plotter.render()

    def _show_overlays(self, render: bool = True) -> None:
        """Replace the mesh and results overlays (the board solids stay).

        Args:
            render: Redraw afterwards.
        """
        plotter = self._plotter
        for actor, _ in self._overlay_actors.values():
            plotter.remove_actor(actor, render=False)
        self._overlay_actors = {}
        self.overlays = []
        if self._board is not None and self.parts:
            data = self._overlay_data
            self.overlays = overlay_parts(
                layer_z_ranges(self._board, self.z_scale.value()),
                data["mesh"],
                data["results"],
                data["mode"],
                data["log"],
            )
        for part in self.overlays:
            if part.key.startswith("results:"):
                actor = plotter.add_mesh(
                    part.mesh, scalars="rgb", rgb=True, name=part.key, show_scalar_bar=False
                )
            else:
                actor = plotter.add_mesh(
                    part.mesh, style="wireframe", color=_rgb(part.color), opacity=part.opacity, name=part.key
                )
            self._overlay_actors[part.key] = (actor, part.layer)
        self._apply_visibility(render)

    def _apply_visibility(self, render: bool = True) -> None:
        """Hide hidden layers, and the dielectric, mesh or results when switched off.

        Args:
            render: Redraw afterwards.
        """
        for key, (actor, layer) in {**self._actors, **self._overlay_actors}.items():
            visible = layer not in self._hidden_layers
            if key.startswith("dielectric:"):
                visible = self.show_dielectric.isChecked()
            elif key.startswith("mesh"):
                visible = visible and self._show_mesh
            elif key.startswith("results:"):
                visible = visible and self._show_results
            actor.SetVisibility(visible)
        if render and self._plotter is not None:
            self._plotter.render()

    def reset_camera(self) -> None:
        """Look at the whole board from above the front edge, tilted to show the stackup."""
        if self._plotter is None:
            return
        self._plotter.view_xy()
        self._plotter.camera.Elevation(-55)  # tilt toward the front edge
        self._plotter.camera.OrthogonalizeViewUp()
        self._plotter.reset_camera()
        self._plotter.render()

    def shutdown(self) -> None:
        """Release the render window (call before the application quits)."""
        self._poll.stop()
        self._pending = None
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
        if self._plotter is not None:
            self._plotter.close()
            self._plotter = None
