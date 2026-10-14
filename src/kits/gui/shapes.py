"""Convert board-model geometry into Qt painter paths."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QPainterPath, QPainterPathStroker, QPolygonF

from ..ingest.geometry import arc_points
from ..ingest.model import Drill, Graphic, Pad, Point


def polygon_path(points: list[Point]) -> QPainterPath:
    """Closed polygon path through ``points``.

    Args:
        points: Polygon vertices in millimetres.

    Returns:
        The closed path.
    """
    path = QPainterPath()
    path.addPolygon(QPolygonF([QPointF(x, y) for x, y in points]))
    path.closeSubpath()
    return path


def polyline_path(points: list[Point]) -> QPainterPath:
    """Open path joining ``points`` with straight segments.

    Args:
        points: Vertices in millimetres; must contain at least one point.

    Returns:
        The open (unfilled) path.
    """
    path = QPainterPath(QPointF(*points[0]))
    for x, y in points[1:]:
        path.lineTo(x, y)
    return path


def stroke(path: QPainterPath, width: float) -> QPainterPath:
    """Outline of ``path`` drawn with a round-capped pen of ``width``.

    Args:
        path: Centre-line path to stroke.
        width: Pen width in millimetres; clamped to a tiny positive minimum.

    Returns:
        A fillable path covering the stroked area.
    """
    stroker = QPainterPathStroker()
    stroker.setWidth(max(width, 1e-3))
    stroker.setCapStyle(Qt.PenCapStyle.RoundCap)
    stroker.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return stroker.createStroke(path)


def circle_path(center: Point, radius: float) -> QPainterPath:
    """Closed circular path.

    Args:
        center: Circle centre in millimetres.
        radius: Circle radius in millimetres.

    Returns:
        The circle path.
    """
    path = QPainterPath()
    path.addEllipse(QPointF(*center), radius, radius)
    return path


def graphic_outline(g: Graphic) -> tuple[QPainterPath, bool]:
    """Path for a graphic, and whether it is a closed (fillable) shape.

    Args:
        g: Board graphic (line, arc, circle, bezier or polygon/rect).

    Returns:
        The geometric outline (unstroked) and ``True`` if the shape is closed.
    """
    if g.kind == "line":
        return polyline_path(g.points), False
    if g.kind == "arc":
        return polyline_path(arc_points(*g.points)), False
    if g.kind == "circle":
        (cx, cy), (px, py) = g.points
        return circle_path((cx, cy), math.hypot(px - cx, py - cy)), True
    if g.kind == "bezier" and len(g.points) == 4:
        p0, c1, c2, p3 = (QPointF(*p) for p in g.points)
        path = QPainterPath(p0)
        path.cubicTo(c1, c2, p3)
        return path, False
    return polygon_path(g.points), True


def graphic_copper(g: Graphic) -> QPainterPath:
    """Area covered by a graphic drawn on a copper layer (fill plus stroke).

    Args:
        g: Board graphic on a copper layer.

    Returns:
        A fillable path of the copper area.
    """
    path, closed = graphic_outline(g)
    if closed and g.filled:
        if g.width > 0:
            path = path.united(stroke(path, g.width))
        return path
    return stroke(path, g.width)


def _rounded_rect(w: float, h: float, radius: float) -> QPainterPath:
    """Rounded rectangle centred on the origin.

    Args:
        w: Width in millimetres.
        h: Height in millimetres.
        radius: Corner radius in millimetres; 0 gives square corners.

    Returns:
        The rectangle path.
    """
    path = QPainterPath()
    path.addRoundedRect(QRectF(-w / 2, -h / 2, w, h), radius, radius)
    return path


def pad_path(pad: Pad) -> QPainterPath:
    """Pad copper in the pad's local frame (centred on the anchor, unrotated).

    Args:
        pad: Pad to draw; its shape, size and (for custom pads) primitives are used.

    Returns:
        A fillable path; the caller positions and rotates it.
    """
    w, h = pad.size
    shape = pad.shape
    if shape == "circle":
        return circle_path((0.0, 0.0), w / 2)
    if shape == "oval":
        return _rounded_rect(w, h, min(w, h) / 2)
    if shape == "roundrect":
        return _rounded_rect(w, h, pad.roundrect_ratio * min(w, h))
    if shape == "trapezoid":
        dx, dy = pad.rect_delta
        return polygon_path(
            [
                (-w / 2 - dy / 2, h / 2 + dx / 2),
                (w / 2 + dy / 2, h / 2 - dx / 2),
                (w / 2 - dy / 2, -h / 2 + dx / 2),
                (-w / 2 + dy / 2, -h / 2 - dx / 2),
            ]
        )
    if shape == "custom":
        path = circle_path((0.0, 0.0), w / 2) if w == h else _rounded_rect(w, h, 0)
        path.setFillRule(Qt.FillRule.WindingFill)
        for prim in pad.primitives:
            prim_path, closed = graphic_outline(prim)
            if closed and (prim.filled or prim.width == 0):
                path.addPath(prim_path)
                if prim.width > 0:
                    path.addPath(stroke(prim_path, prim.width))
            else:
                path.addPath(stroke(prim_path, prim.width))
        return path
    # rect, chamfered rect and anything unknown
    return _rounded_rect(w, h, 0)


def drill_path(drill: Drill) -> QPainterPath:
    """Hole outline in the pad's local frame.

    Args:
        drill: Pad drill; round or oval, with an optional offset from the pad anchor.

    Returns:
        A fillable path of the hole.
    """
    ox, oy = drill.offset
    if not drill.is_oval:
        return circle_path((ox, oy), drill.width / 2)
    w, h = drill.width, drill.height
    path = QPainterPath()
    path.addRoundedRect(QRectF(ox - w / 2, oy - h / 2, w, h), min(w, h) / 2, min(w, h) / 2)
    return path
