"""Small 2D geometry helpers shared by ingest, GUI and (later) meshing."""

from __future__ import annotations

import math

from .model import Point


def rotate(point: Point, angle_deg: float) -> Point:
    """Rotate ``point`` about the origin the way KiCad does.

    KiCad angles are counter-clockwise *as displayed*, with Y pointing down.
    """
    if not angle_deg:
        return point
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    x, y = point
    return (x * c + y * s, -x * s + y * c)


def transform(point: Point, origin: Point, angle_deg: float) -> Point:
    """Map a footprint-local point to board coordinates."""
    x, y = rotate(point, angle_deg)
    return (x + origin[0], y + origin[1])


def arc_center(start: Point, mid: Point, end: Point) -> Point | None:
    """Centre of the circle through three points, or None if collinear."""
    (ax, ay), (bx, by), (cx, cy) = start, mid, end
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return None
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return (ux, uy)


def arc_points(start: Point, mid: Point, end: Point, max_step_deg: float = 5.0) -> list[Point]:
    """Approximate a three-point arc as a polyline from ``start`` to ``end``."""
    center = arc_center(start, mid, end)
    if center is None:
        return [start, end]
    cx, cy = center
    r = math.hypot(start[0] - cx, start[1] - cy)
    a0 = math.atan2(start[1] - cy, start[0] - cx)
    am = math.atan2(mid[1] - cy, mid[0] - cx)
    a1 = math.atan2(end[1] - cy, end[0] - cx)

    # Sweep from a0 to a1 in whichever direction passes through mid.
    sweep = (a1 - a0) % (2 * math.pi)
    if (am - a0) % (2 * math.pi) > sweep:
        sweep -= 2 * math.pi
    if abs(sweep) < 1e-12:
        sweep = 2 * math.pi  # start == end: full circle

    steps = max(2, math.ceil(abs(math.degrees(sweep)) / max_step_deg))
    pts = [
        (cx + r * math.cos(a0 + sweep * i / steps), cy + r * math.sin(a0 + sweep * i / steps))
        for i in range(steps + 1)
    ]
    pts[0], pts[-1] = start, end
    return pts
