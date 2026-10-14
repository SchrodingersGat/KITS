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

from PySide6.QtCore import QLineF, QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsScene,
)

from ..ingest.geometry import arc_points
from ..ingest.model import EDGE_CUTS, Board, Pad
from . import colors
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
_GRID_MIN_PX = 8  # never draw grid lines closer than this on screen
_GRID_ORIGIN_PX = 8  # half-size of the origin marker


@dataclass
class DisplayLayer:
    name: str
    color: QColor
    is_copper: bool
    item: LayerItem


class LayerItem(QGraphicsItem):
    """Paint-less container whose children are one display layer's items."""

    def __init__(self):
        super().__init__()
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemHasNoContents)

    def boundingRect(self) -> QRectF:
        return QRectF()

    def paint(self, painter, option, widget=None):
        pass


def _no_pen() -> QPen:
    return QPen(Qt.PenStyle.NoPen)


def _round_pen(color: QColor, width: float) -> QPen:
    pen = QPen(color, width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    if width <= 0:
        pen.setCosmetic(True)
        pen.setWidthF(1.0)
    return pen


def _cosmetic_pen(color: QColor, pixels: float) -> QPen:
    pen = QPen(color, pixels)
    pen.setCosmetic(True)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


def _net_tip(net: str | None) -> str:
    return f"Net: {net}" if net else "No net"


class BoardScene(QGraphicsScene):
    def __init__(self, board: Board, parent=None):
        super().__init__(parent)
        self.board = board
        self.layers: dict[str, DisplayLayer] = {}
        self._grid_spacing = DEFAULT_GRID_SPACING_MM
        self._grid_visible = True
        self.setBackgroundBrush(colors.BACKGROUND)
        self._build()

    # -- public API ----------------------------------------------------------

    def layer_names(self) -> list[str]:
        """Display layers in panel order: copper top-to-bottom, then the rest."""
        return list(self.layers)

    def set_layer_visible(self, name: str, visible: bool) -> None:
        self.layers[name].item.setVisible(visible)

    def set_active_layer(self, name: str | None) -> None:
        """Draw ``name`` above the other copper layers (like KiCad's active layer)."""
        copper = [n for n, lay in self.layers.items() if lay.is_copper]
        # Stack copper so that F.Cu ends up on top and B.Cu at the bottom.
        for z, layer_name in enumerate(reversed(copper)):
            self.layers[layer_name].item.setZValue(z)
        if name in copper:
            self.layers[name].item.setZValue(len(copper))

    def board_rect(self) -> QRectF:
        bbox = self.board.bounding_box()
        if bbox is None:
            return self.itemsBoundingRect()
        x0, y0, x1, y1 = bbox
        return QRectF(x0, y0, x1 - x0, y1 - y0)

    # -- grid ----------------------------------------------------------------

    def set_grid(self, spacing_mm: float, visible: bool) -> None:
        if spacing_mm <= 0:
            raise ValueError("grid spacing must be positive")
        self._grid_spacing = spacing_mm
        self._grid_visible = visible
        self.update()

    @property
    def grid_spacing(self) -> float:
        return self._grid_spacing

    @property
    def grid_visible(self) -> bool:
        return self._grid_visible

    def grid_origin(self) -> QPointF:
        """Bottom-left corner of the board bounding box (Y points down)."""
        return self.board_rect().bottomLeft()

    def displayed_grid_spacing(self, pixels_per_mm: float) -> float:
        """Grid pitch actually drawn at a zoom level.

        When the configured pitch would put lines closer than ``_GRID_MIN_PX``
        on screen, only every 2nd/5th/10th/20th/... line is drawn, so lines
        stay on the configured grid but never smear into a solid fill.
        """
        spacing = self._grid_spacing
        steps = itertools.cycle((2, 2.5, 2))  # x1, x2, x5, x10, x20, x50, ...
        while spacing * pixels_per_mm < _GRID_MIN_PX:
            spacing *= next(steps)
        return spacing

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
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
        item = LayerItem()
        item.setZValue(z)
        self.addItem(item)
        self.layers[name] = DisplayLayer(name, color, is_copper, item)
        return item

    def _build(self) -> None:
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
        item = QGraphicsPathItem(path, self.layers[layer].item)
        item.setPen(_no_pen())
        item.setBrush(QBrush(brush))
        item.setZValue(z)
        return item

    def _add_pad(self, pad: Pad, th_layer: LayerItem, hole_layer: LayerItem) -> None:
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

        if pad.drill is not None and pad.drill.width > 0:
            hole = QGraphicsPathItem(drill_path(pad.drill), hole_layer)
            hole.setPen(_no_pen())
            hole.setBrush(colors.HOLES)
            hole.setPos(*pad.position)
            hole.setRotation(-pad.rotation)
            hole.setToolTip(tip)
