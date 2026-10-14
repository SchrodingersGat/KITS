"""QGraphicsScene representation of a board, organised into toggleable layers.

Scene units are board millimetres with Y pointing down, matching KiCad, so
scene coordinates can be used directly as board coordinates.

Each display layer (a copper layer, the board outline, or a pseudo-layer such
as vias) is a :class:`LayerItem` container; showing/hiding or restacking a
layer is a single operation on that container.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QLineF, QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
)

from ..config import CurrentSource
from ..ingest.geometry import arc_points
from ..ingest.model import EDGE_CUTS, Board, Pad
from ..mesh.model import BoardMesh, LayerMesh
from ..solve.results import LayerResult, SolveResults
from . import colors
from .field_display import DisplayMode, color_indices, colormap, triangle_values, value_range
from .settings import DEFAULT_GRID_SPACING_MM
from .shapes import (
    circle_path,
    drill_path,
    graphic_copper,
    graphic_outline,
    pad_path,
    polygon_path,
    polyline_path,
)

TH_PADS = "Through-hole pads"
VIAS = "Vias"
HOLES = "Holes"

# Data role on graphics items holding the board-model object they draw, so
# later click-to-edit tooling can map a scene item back to the board.
MODEL_ROLE = 0

_Z_ZONE, _Z_GRAPHIC, _Z_TRACK, _Z_PAD = 0, 1, 2, 3
_EDGE_CUTS_PX = 1.5
_Z_RESULTS_OVERLAY = 950  # solved fields, above the board, below the mesh
_Z_MESH_OVERLAY = 1000  # above all board layers
_RESULTS_ALPHA = 235
_Z_SOURCES = 1100  # current source markers, above the mesh
_Z_SELECTION = 1200  # selected pad outline, on top of everything
_LABEL_GAP_PX = 4  # gap between a pad and its current label, screen pixels
# Data role holding a pad's (footprint ref, pad number) on its copper and hole items,
# for picking pads with the mouse.
PAD_KEY_ROLE = 1
_MESH_TILES = 16  # overlay tiles per axis
_GRID_MIN_PX = 8  # never draw grid lines closer than this on screen
_GRID_ORIGIN_PX = 8  # half-size of the origin marker


@dataclass
class DisplayLayer:
    """A toggleable display layer and the scene container holding its items."""

    name: str
    color: QColor
    is_copper: bool
    item: LayerItem


class LayerItem(QGraphicsItem):
    """Paint-less container whose children are one display layer's items."""

    def __init__(self):
        """Create an empty container flagged as having no contents of its own."""
        super().__init__()
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemHasNoContents)

    def boundingRect(self) -> QRectF:
        """Return an empty rect; the container itself draws nothing.

        Returns:
            An empty rectangle.
        """
        return QRectF()

    def paint(self, painter, option, widget=None):
        """Draw nothing; only the children are painted.

        Args:
            painter: Painter in item coordinates (unused).
            option: Style options for the item (unused).
            widget: Widget being painted on, if any (unused).
        """
        pass


def _no_pen() -> QPen:
    """Pen that draws nothing, for fill-only items.

    Returns:
        A ``NoPen`` pen.
    """
    return QPen(Qt.PenStyle.NoPen)


def _round_pen(color: QColor, width: float) -> QPen:
    """Round-capped, round-joined pen for copper tracks.

    Args:
        color: Pen colour.
        width: Pen width in millimetres; zero or negative gives a 1 px cosmetic pen.

    Returns:
        The configured pen.
    """
    pen = QPen(color, width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    if width <= 0:
        pen.setCosmetic(True)
        pen.setWidthF(1.0)
    return pen


def _cosmetic_pen(color: QColor, pixels: float) -> QPen:
    """Round-capped cosmetic pen whose width is fixed in screen pixels at any zoom.

    Args:
        color: Pen colour.
        pixels: Pen width in screen pixels.

    Returns:
        The configured pen.
    """
    pen = QPen(color, pixels)
    pen.setCosmetic(True)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


def _net_tip(net: str | None) -> str:
    """Tooltip line describing an item's net.

    Args:
        net: Net name, or ``None``/empty for unconnected items.

    Returns:
        ``"Net: <name>"`` or ``"No net"``.
    """
    return f"Net: {net}" if net else "No net"


class BoardScene(QGraphicsScene):
    """Scene holding a board's copper, pads, vias, holes and outline, grouped by display layer.

    Also draws the configurable background grid and its origin marker.
    """

    def __init__(self, board: Board, parent=None):
        """Build the scene items for ``board``.

        Args:
            board: Parsed board to display.
            parent: Optional Qt parent object.
        """
        super().__init__(parent)
        self.board = board
        self.layers: dict[str, DisplayLayer] = {}
        self._active_layer: str | None = None
        self._mesh_overlays: dict[str, LayerItem] = {}
        self._pad_items: dict[tuple[str, str], list[QGraphicsPathItem]] = {}
        self._source_markers: LayerItem | None = None
        self._selection: LayerItem | None = None
        self._results: SolveResults | None = None
        self._results_mode = DisplayMode.CURRENT_DENSITY
        self._results_range = (0.0, 1.0)
        self._results_visible = True
        self._results_overlays: dict[str, LayerItem] = {}
        self._mesh_visible = True
        self._grid_spacing = DEFAULT_GRID_SPACING_MM
        self._grid_visible = True
        self.setBackgroundBrush(colors.BACKGROUND)
        self._build()

    # -- public API ----------------------------------------------------------

    def layer_names(self) -> list[str]:
        """Display layers in panel order: copper top-to-bottom, then the rest."""
        return list(self.layers)

    def set_layer_visible(self, name: str, visible: bool) -> None:
        """Show or hide a display layer.

        Args:
            name: Display layer name, as returned by :meth:`layer_names`.
            visible: Whether the layer is shown.
        """
        self.layers[name].item.setVisible(visible)
        self._update_mesh_visibility()

    def set_active_layer(self, name: str | None) -> None:
        """Draw ``name`` above the other copper layers (like KiCad's active layer).

        Args:
            name: Copper layer to raise, or ``None`` (or a non-copper name) for the default
                stacking with F.Cu on top.
        """
        copper = [n for n, lay in self.layers.items() if lay.is_copper]
        # Stack copper so that F.Cu ends up on top and B.Cu at the bottom.
        for z, layer_name in enumerate(reversed(copper)):
            self.layers[layer_name].item.setZValue(z)
        if name in copper:
            self.layers[name].item.setZValue(len(copper))
        self._active_layer = name if name in copper else (copper[0] if copper else None)
        self._update_mesh_visibility()

    @property
    def active_layer(self) -> str | None:
        """The copper layer drawn on top, whose mesh is shown; F.Cu unless another was chosen."""
        return self._active_layer

    # -- pads: picking, selection and current sources ------------------------------

    def pad_at(self, pos: QPointF) -> tuple[str, str] | None:
        """The pad (with copper) drawn topmost at a scene position.

        Hidden layers are ignored.

        Args:
            pos: Scene position in millimetres.

        Returns:
            ``(ref, pad number)``, or None if there is no pad there.
        """
        for item in self.items(pos):  # topmost first
            key = item.data(PAD_KEY_ROLE)
            if key is not None and item.isVisible():
                return tuple(key)
        return None

    def has_pad(self, key: tuple[str, str]) -> bool:
        """Whether a pad with copper exists on the board.

        Args:
            key: ``(ref, pad number)``.

        Returns:
            True if the scene has copper items for it.
        """
        return tuple(key) in self._pad_items

    def set_selected_pad(self, key: tuple[str, str] | None) -> None:
        """Outline the selected pad (every pad with that reference and number).

        Args:
            key: ``(ref, pad number)``, or None to clear the selection.
        """
        if self._selection is not None:
            self.removeItem(self._selection)
            self._selection = None
        if key is None or tuple(key) not in self._pad_items:
            return
        self._selection = LayerItem()
        self._selection.setZValue(_Z_SELECTION)
        self.addItem(self._selection)
        for pad_item in self._pad_items[tuple(key)]:
            outline = self._pad_copy(pad_item, self._selection)
            outline.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            outline.setPen(_cosmetic_pen(colors.SELECTION, 2.5))

    def set_sources(self, sources: list[CurrentSource]) -> None:
        """Mark pads that have current sources: tinted pad plus a current label.

        Sources (current into the board) are tinted green, sinks orange. Labels
        keep a fixed on-screen size. Sources on pads missing from the board are
        skipped.

        Args:
            sources: The project's current sources.
        """
        if self._source_markers is not None:
            self.removeItem(self._source_markers)
        self._source_markers = LayerItem()
        self._source_markers.setZValue(_Z_SOURCES)
        self.addItem(self._source_markers)
        font = QFont()
        font.setBold(True)
        for source in sources:
            items = self._pad_items.get(source.key)
            if not items:
                continue
            color = colors.SOURCE if source.current_a >= 0 else colors.SINK
            for pad_item in items:
                tint = self._pad_copy(pad_item, self._source_markers)
                tint.setBrush(QColor(color.red(), color.green(), color.blue(), 150))
                tint.setPen(_cosmetic_pen(color, 1.5))
            label = QGraphicsSimpleTextItem(f"{source.current_a:+g} A")
            label.setFont(font)
            label.setBrush(QColor(255, 255, 255))
            text = label.boundingRect()
            # Screen-pixel box just right of the pad, vertically centred on it.
            box = QGraphicsRectItem(text.adjusted(-3, -1, 3, 1), self._source_markers)
            box.setBrush(QColor(color.red(), color.green(), color.blue(), 220))
            box.setPen(QPen(Qt.PenStyle.NoPen))
            box.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            box.setTransform(box.transform().translate(_LABEL_GAP_PX + 3, -text.height() / 2))
            pad_bounds = items[0].sceneBoundingRect()
            for pad_item in items[1:]:
                pad_bounds = pad_bounds.united(pad_item.sceneBoundingRect())
            box.setPos(QPointF(pad_bounds.right(), pad_bounds.center().y()))
            box.setToolTip(f"{source.label}: {source.current_a:+g} A")
            label.setParentItem(box)

    @property
    def source_marker_count(self) -> int:
        """Number of current labels currently drawn (one per placed source)."""
        if self._source_markers is None:
            return 0
        return sum(isinstance(i, QGraphicsRectItem) for i in self._source_markers.childItems())

    @staticmethod
    def _pad_copy(pad_item: QGraphicsPathItem, parent: QGraphicsItem) -> QGraphicsPathItem:
        """Create an item with a pad's shape, position and rotation under ``parent``.

        Args:
            pad_item: The pad's copper item.
            parent: Container for the copy.

        Returns:
            The new item (no pen or brush set yet).
        """
        copy = QGraphicsPathItem(pad_item.path(), parent)
        copy.setPos(pad_item.pos())
        copy.setRotation(pad_item.rotation())
        return copy

    # -- mesh overlay ----------------------------------------------------------

    def set_mesh(self, mesh: BoardMesh | None) -> None:
        """Replace the mesh overlay.

        One overlay is built per meshed copper layer; only the active layer's
        overlay is shown, since every layer's mesh covers the whole board.

        Args:
            mesh: The board mesh, or None to remove the overlay.
        """
        for overlay in self._mesh_overlays.values():
            self.removeItem(overlay)
        self._mesh_overlays = {}
        if mesh is not None:
            for lm in mesh.layers:
                if lm.layer in self.layers:
                    self._mesh_overlays[lm.layer] = self._build_mesh_overlay(lm)
        self._update_mesh_visibility()

    def set_mesh_visible(self, visible: bool) -> None:
        """Show or hide the mesh overlay.

        Args:
            visible: Whether the active layer's mesh is drawn.
        """
        self._mesh_visible = visible
        self._update_mesh_visibility()

    @property
    def mesh_layers(self) -> list[str]:
        """Copper layers that currently have a mesh overlay."""
        return list(self._mesh_overlays)

    def mesh_overlay_visible(self, layer: str) -> bool:
        """Whether a layer's mesh overlay is currently drawn.

        Args:
            layer: Copper layer name.

        Returns:
            True if the overlay exists and is visible.
        """
        overlay = self._mesh_overlays.get(layer)
        return overlay is not None and overlay.isVisible()

    # -- results overlay --------------------------------------------------------------

    def set_results(self, results: SolveResults | None, mode: DisplayMode | None = None) -> None:
        """Show solve results, coloured by one quantity.

        As with the mesh, only the active layer's overlay is drawn. Overlays are
        built when a layer is first shown and kept until the results or mode change.

        Args:
            results: The results, or None to remove the overlay.
            mode: Quantity to colour by; keeps the current mode if omitted.
        """
        for overlay in self._results_overlays.values():
            self.removeItem(overlay)
        self._results_overlays = {}
        self._results = results
        if mode is not None:
            self._results_mode = mode
        if results is not None:
            self._results_range = value_range(results, self._results_mode)
        self._update_results_visibility()

    def set_results_visible(self, visible: bool) -> None:
        """Show or hide the results overlay.

        Args:
            visible: Whether the active layer's results are drawn.
        """
        self._results_visible = visible
        self._update_results_visibility()

    @property
    def results_range(self) -> tuple[float, float]:
        """Colour scale limits of the current results overlay, in display units."""
        return self._results_range

    def results_overlay_visible(self, layer: str) -> bool:
        """Whether a layer's results overlay is currently drawn.

        Args:
            layer: Copper layer name.

        Returns:
            True if the overlay exists and is visible.
        """
        overlay = self._results_overlays.get(layer)
        return overlay is not None and overlay.isVisible()

    def _update_results_visibility(self) -> None:
        """Build (if needed) and show only the active layer's results overlay."""
        show = self._results is not None and self._results_visible and self._active_layer is not None
        if show and self._active_layer in self.layers:
            show = self.layers[self._active_layer].item.isVisible()
        if show and self._active_layer not in self._results_overlays:
            layer = self._results.layer(self._active_layer)
            if layer is not None:
                self._results_overlays[self._active_layer] = self._build_results_overlay(layer)
        for name, overlay in self._results_overlays.items():
            overlay.setVisible(show and name == self._active_layer)

    def _build_results_overlay(self, layer: LayerResult) -> LayerItem:
        """Create filled, colour-binned triangles for one layer's results.

        Triangles are grouped by colour and by tile (as for the mesh overlay), so
        each group is one path item and partial repaints stay cheap.

        Args:
            layer: The layer's results.

        Returns:
            The overlay container, already added to the scene.
        """
        container = LayerItem()
        container.setZValue(_Z_RESULTS_OVERLAY)
        self.addItem(container)
        values, shown = triangle_values(layer, self._results_mode)
        tris = layer.triangles[shown]
        if not len(tris):
            return container
        colors_ = colormap(self._results_mode)
        index = color_indices(values[shown], *self._results_range, steps=len(colors_))
        corners = layer.nodes[tris]  # (M, 3, 2)
        centroid = corners.mean(axis=1)
        lo, hi = centroid.min(axis=0), centroid.max(axis=0)
        cell = np.minimum(
            ((centroid - lo) / np.maximum(hi - lo, 1e-9) * _MESH_TILES).astype(int), _MESH_TILES - 1
        )
        key = (cell[:, 0] * _MESH_TILES + cell[:, 1]) * len(colors_) + index
        order = np.argsort(key, kind="stable")
        bounds = np.flatnonzero(np.diff(key[order])) + 1
        for group in np.split(order, bounds):
            path = QPainterPath()
            for (x0, y0), (x1, y1), (x2, y2) in corners[group].tolist():
                path.moveTo(x0, y0)
                path.lineTo(x1, y1)
                path.lineTo(x2, y2)
                path.closeSubpath()
            base = colors_[int(index[group[0]])]
            fill = QColor(base.red(), base.green(), base.blue(), _RESULTS_ALPHA)
            item = QGraphicsPathItem(path, container)
            item.setBrush(fill)
            item.setPen(_cosmetic_pen(fill, 1))  # hairline closes antialiasing seams between triangles
        return container

    def _update_mesh_visibility(self) -> None:
        """Show only the active layer's overlay, and only while that layer is visible."""
        self._update_results_visibility()
        for name, overlay in self._mesh_overlays.items():
            overlay.setVisible(
                self._mesh_visible and name == self._active_layer and self.layers[name].item.isVisible()
            )

    def _build_mesh_overlay(self, lm: LayerMesh) -> LayerItem:
        """Create the overlay items for one layer's mesh.

        Edges are split into a grid of tiles, one path item per tile and region,
        so a partial repaint (e.g. the crosshair moving) only redraws nearby tiles.

        Args:
            lm: The layer mesh.

        Returns:
            The overlay container, already added to the scene.
        """
        container = LayerItem()
        container.setZValue(_Z_MESH_OVERLAY)
        self.addItem(container)
        edges, is_copper = lm.edges()
        if not len(edges):
            return container
        p0, p1 = lm.nodes[edges[:, 0]], lm.nodes[edges[:, 1]]
        mid = (p0 + p1) / 2
        lo, hi = mid.min(axis=0), mid.max(axis=0)
        span = np.maximum(hi - lo, 1e-9)
        cell = np.minimum(((mid - lo) / span * _MESH_TILES).astype(int), _MESH_TILES - 1)
        key = (cell[:, 0] * _MESH_TILES + cell[:, 1]) * 2 + is_copper
        order = np.argsort(key, kind="stable")
        bounds = np.flatnonzero(np.diff(key[order])) + 1
        for group in np.split(order, bounds):
            path = QPainterPath()
            for (x0, y0), (x1, y1) in zip(p0[group].tolist(), p1[group].tolist(), strict=True):
                path.moveTo(x0, y0)
                path.lineTo(x1, y1)
            item = QGraphicsPathItem(path, container)
            color = colors.MESH_COPPER if is_copper[group[0]] else colors.MESH_BARE
            item.setPen(_cosmetic_pen(color, 1))
        return container

    def board_rect(self) -> QRectF:
        """Board bounding box in scene coordinates (mm).

        Returns:
            The board's bounding box, or the bounding rect of all items if the board has none.
        """
        bbox = self.board.bounding_box()
        if bbox is None:
            return self.itemsBoundingRect()
        x0, y0, x1, y1 = bbox
        return QRectF(x0, y0, x1 - x0, y1 - y0)

    # -- grid ----------------------------------------------------------------

    def set_grid(self, spacing_mm: float, visible: bool) -> None:
        """Configure the background grid and repaint.

        Args:
            spacing_mm: Configured grid pitch in millimetres.
            visible: Whether the grid and origin marker are drawn.

        Raises:
            ValueError: If ``spacing_mm`` is not positive.
        """
        if spacing_mm <= 0:
            raise ValueError("grid spacing must be positive")
        self._grid_spacing = spacing_mm
        self._grid_visible = visible
        self.update()

    @property
    def grid_spacing(self) -> float:
        """Configured grid pitch in millimetres."""
        return self._grid_spacing

    @property
    def grid_visible(self) -> bool:
        """Whether the grid and origin marker are drawn."""
        return self._grid_visible

    def grid_origin(self) -> QPointF:
        """Bottom-left corner of the board bounding box (Y points down)."""
        return self.board_rect().bottomLeft()

    def to_grid_coords(self, x: float, y: float) -> tuple[float, float]:
        """Convert scene (KiCad) coordinates to grid coordinates.

        Grid coordinates are relative to the grid origin, with Y pointing *up* so
        the board sits in the positive quadrant.

        Args:
            x: Scene X coordinate in millimetres.
            y: Scene Y coordinate in millimetres (Y down).

        Returns:
            ``(x, y)`` in millimetres relative to :meth:`grid_origin`, Y up.
        """
        origin = self.grid_origin()
        return x - origin.x(), origin.y() - y

    def displayed_grid_spacing(self, pixels_per_mm: float) -> float:
        """Grid pitch actually drawn at a zoom level.

        When the configured pitch would put lines closer than ``_GRID_MIN_PX``
        on screen, only every 2nd/5th/10th/20th/... line is drawn, so lines
        stay on the configured grid but never smear into a solid fill.

        Args:
            pixels_per_mm: Current view scale in screen pixels per millimetre.

        Returns:
            The drawn grid pitch in millimetres (a multiple of the configured pitch).
        """
        spacing = self._grid_spacing
        steps = itertools.cycle((2, 2.5, 2))  # x1, x2, x5, x10, x20, x50, ...
        while spacing * pixels_per_mm < _GRID_MIN_PX:
            spacing *= next(steps)
        return spacing

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        """Draw the grid lines (1 px cosmetic) over the background brush.

        Args:
            painter: Painter in scene coordinates.
            rect: Exposed area in scene coordinates.
        """
        super().drawBackground(painter, rect)
        if not self._grid_visible:
            return
        scale = painter.worldTransform().m11()
        if scale <= 0:
            return
        spacing = self.displayed_grid_spacing(scale)
        origin = self.grid_origin()

        first_x = origin.x() + math.ceil((rect.left() - origin.x()) / spacing) * spacing
        first_y = origin.y() + math.ceil((rect.top() - origin.y()) / spacing) * spacing
        xs = [first_x + i * spacing for i in range(int((rect.right() - first_x) // spacing) + 1)]
        ys = [first_y + j * spacing for j in range(int((rect.bottom() - first_y) // spacing) + 1)]
        lines = [QLineF(x, rect.top(), x, rect.bottom()) for x in xs]
        lines += [QLineF(rect.left(), y, rect.right(), y) for y in ys]

        painter.save()
        pen = QPen(colors.GRID, 1)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.drawLines(lines)
        painter.restore()

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        """Draw the grid origin marker, a fixed screen-size cross, above all items.

        Args:
            painter: Painter in scene coordinates.
            rect: Exposed area in scene coordinates.
        """
        super().drawForeground(painter, rect)
        if not self._grid_visible:
            return
        scale = painter.worldTransform().m11()
        if scale <= 0:
            return
        # Origin marker: a fixed-size cross in screen pixels.
        origin = self.grid_origin()
        r = _GRID_ORIGIN_PX / scale
        painter.save()
        pen = QPen(colors.GRID_ORIGIN, 1.5)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawLine(QLineF(origin.x() - r, origin.y(), origin.x() + r, origin.y()))
        painter.drawLine(QLineF(origin.x(), origin.y() - r, origin.x(), origin.y() + r))
        painter.drawEllipse(origin, r / 2, r / 2)
        painter.restore()

    # -- construction ----------------------------------------------------------

    def _add_layer(self, name: str, color: QColor, is_copper: bool, z: float) -> LayerItem:
        """Create a display layer container, add it to the scene and register it.

        Args:
            name: Display layer name.
            color: Base colour for the layer's items.
            is_copper: Whether this is a copper layer (affects active-layer stacking).
            z: Z value of the container relative to other layers.

        Returns:
            The new container item to parent the layer's items to.
        """
        item = LayerItem()
        item.setZValue(z)
        self.addItem(item)
        self.layers[name] = DisplayLayer(name, color, is_copper, item)
        return item

    def _build(self) -> None:
        """Create all layers and items for the board and set a padded scene rect."""
        board = self.board
        copper = board.copper_layers
        for name in copper:
            self._add_layer(name, colors.copper_color(name), True, 0)
        self.set_active_layer(None)

        top = len(copper) + 1
        th_layer = self._add_layer(TH_PADS, colors.TH_PADS, False, top)
        via_layer = self._add_layer(VIAS, colors.VIAS, False, top + 1)
        hole_layer = self._add_layer(HOLES, colors.HOLES, False, top + 2)
        edge_layer = self._add_layer(EDGE_CUTS, colors.EDGE_CUTS, False, top + 3)

        for zone in board.zones:
            if zone.is_rule_area:
                continue
            for layer_name, polygons in zone.fills.items():
                if layer_name not in self.layers:
                    continue
                color = self.layers[layer_name].color.darker(colors.ZONE_DARKEN)
                path = QPainterPath()
                for poly in polygons:
                    if len(poly) >= 3:
                        path.addPath(polygon_path(poly))
                item = self._path_item(path, layer_name, _Z_ZONE, brush=color)
                label = f"Zone '{zone.name}'" if zone.name else "Zone"
                item.setToolTip(f"{label} on {layer_name}\n{_net_tip(zone.net)}")
                item.setData(MODEL_ROLE, zone)

        for g in board.graphics:
            if g.layer == EDGE_CUTS:
                path, _ = graphic_outline(g)
                item = QGraphicsPathItem(path, edge_layer)
                # The outline is a cut line, not copper: draw it at a fixed
                # on-screen width so it stays visible at any zoom level.
                item.setPen(_cosmetic_pen(colors.EDGE_CUTS, _EDGE_CUTS_PX))
            elif g.layer in self.layers:
                item = self._path_item(
                    graphic_copper(g), g.layer, _Z_GRAPHIC, brush=self.layers[g.layer].color
                )
                item.setToolTip(_net_tip(g.net))
            else:
                continue
            item.setData(MODEL_ROLE, g)

        for t in board.tracks:
            if t.layer not in self.layers:
                continue
            item = QGraphicsLineItem(*t.start, *t.end, self.layers[t.layer].item)
            item.setPen(_round_pen(self.layers[t.layer].color, t.width))
            item.setZValue(_Z_TRACK)
            item.setToolTip(f"Track {t.width:g} mm on {t.layer}\n{_net_tip(t.net)}")
            item.setData(MODEL_ROLE, t)

        for a in board.arcs:
            if a.layer not in self.layers:
                continue
            item = QGraphicsPathItem(
                polyline_path(arc_points(a.start, a.mid, a.end)), self.layers[a.layer].item
            )
            item.setPen(_round_pen(self.layers[a.layer].color, a.width))
            item.setZValue(_Z_TRACK)
            item.setToolTip(f"Arc track {a.width:g} mm on {a.layer}\n{_net_tip(a.net)}")
            item.setData(MODEL_ROLE, a)

        for pad in board.pads():
            self._add_pad(pad, th_layer, hole_layer)

        for v in board.vias:
            item = QGraphicsPathItem(circle_path(v.position, v.diameter / 2), via_layer)
            item.setPen(_no_pen())
            item.setBrush(colors.VIAS)
            item.setToolTip(
                f"Via {v.diameter:g}/{v.drill:g} mm ({v.layers[0]}–{v.layers[1]})\n{_net_tip(v.net)}"
            )
            item.setData(MODEL_ROLE, v)
            if v.drill > 0:
                hole = QGraphicsPathItem(circle_path(v.position, v.drill / 2), hole_layer)
                hole.setPen(_no_pen())
                hole.setBrush(colors.HOLES)

        # Leave room around the board for panning.
        rect = self.itemsBoundingRect()
        margin = max(rect.width(), rect.height()) * 0.5
        self.setSceneRect(rect.adjusted(-margin, -margin, margin, margin))

    def _path_item(self, path: QPainterPath, layer: str, z: float, brush: QColor) -> QGraphicsPathItem:
        """Add a filled, unstroked path item to a display layer.

        Args:
            path: Geometry in scene coordinates (mm).
            layer: Name of the display layer to parent the item to.
            z: Z value within the layer.
            brush: Fill colour.

        Returns:
            The new item.
        """
        item = QGraphicsPathItem(path, self.layers[layer].item)
        item.setPen(_no_pen())
        item.setBrush(QBrush(brush))
        item.setZValue(z)
        return item

    def _add_pad(self, pad: Pad, th_layer: LayerItem, hole_layer: LayerItem) -> None:
        """Add a pad's copper and drill hole items, positioned and rotated on the board.

        Through-hole pads go on ``th_layer``, SMD pads on their first copper layer;
        non-plated holes get only a hole item.

        Args:
            pad: Pad to add.
            th_layer: Container for through-hole pad copper.
            hole_layer: Container for drill holes.
        """
        tip = f"{pad.footprint_ref} pad {pad.number}\n{_net_tip(pad.net)}"

        if pad.kind == "thru_hole":
            parent, color = th_layer, colors.TH_PADS
        elif pad.layers and pad.kind != "np_thru_hole":
            parent, color = self.layers[pad.layers[0]].item, self.layers[pad.layers[0]].color
        else:
            parent = color = None

        if parent is not None:
            item = QGraphicsPathItem(pad_path(pad), parent)
            item.setPen(_no_pen())
            item.setBrush(color)
            item.setPos(*pad.position)
            item.setRotation(-pad.rotation)
            item.setZValue(_Z_PAD)
            item.setToolTip(tip)
            item.setData(MODEL_ROLE, pad)
            item.setData(PAD_KEY_ROLE, (pad.footprint_ref, pad.number))
            self._pad_items.setdefault((pad.footprint_ref, pad.number), []).append(item)

        if pad.drill is not None and pad.drill.width > 0:
            hole = QGraphicsPathItem(drill_path(pad.drill), hole_layer)
            hole.setPen(_no_pen())
            hole.setBrush(colors.HOLES)
            hole.setPos(*pad.position)
            hole.setRotation(-pad.rotation)
            hole.setToolTip(tip)
            if parent is not None:  # clicking a plated hole picks its pad
                hole.setData(PAD_KEY_ROLE, (pad.footprint_ref, pad.number))
