"""Small 2D geometry helpers shared by ingest, GUI and (later) meshing."""

from __future__ import annotations

import math

Point = tuple[float, float]


def rotate(point: Point, angle_deg: float) -> Point:
    """Rotate ``point`` about the origin the way KiCad does.

    KiCad angles are counter-clockwise *as displayed*, with Y pointing down.

    Args:
        point: Point to rotate, in mm.
        angle_deg: Rotation in degrees, counter-clockwise as displayed.

    Returns:
        The rotated point (``point`` itself if the angle is zero).
    """
    if not angle_deg:
        return point
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    x, y = point
    return (x * c + y * s, -x * s + y * c)


def transform(point: Point, origin: Point, angle_deg: float) -> Point:
    """Map a footprint-local point to board coordinates.

    Args:
        point: Point in the footprint's local frame, in mm.
        origin: Footprint position in board coordinates, in mm.
        angle_deg: Footprint rotation in degrees, counter-clockwise as displayed.

    Returns:
        The point in board coordinates: rotated by ``angle_deg``, then offset by ``origin``.
    """
    x, y = rotate(point, angle_deg)
    return (x + origin[0], y + origin[1])


def arc_center(start: Point, mid: Point, end: Point) -> Point | None:
    """Centre of the circle through three points, or None if collinear.

    Args:
        start: First point on the circle.
        mid: Second point on the circle.
        end: Third point on the circle.

    Returns:
        The circle centre, or None if the points are (numerically) collinear.
    """
    (ax, ay), (bx, by), (cx, cy) = start, mid, end
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return None
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return (ux, uy)


def _arc_sweep(start: Point, mid: Point, end: Point, center: Point) -> tuple[float, float, float]:
    """(radius, start angle, signed sweep) in radians of a three-point arc.

    Angles are measured with ``atan2`` in the raw (Y-down) coordinate frame.

    Args:
        start: Arc start point.
        mid: Any point on the arc between ``start`` and ``end``; selects the sweep direction.
        end: Arc end point. If equal to ``start`` the arc is a full circle.
        center: Circle centre, as returned by :func:`arc_center`.

    Returns:
        ``(radius, start_angle, sweep)``; ``sweep`` is signed so that sweeping from
        ``start_angle`` passes through ``mid``.
    """
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
    return r, a0, sweep


def arc_extent(start: Point, mid: Point, end: Point) -> list[Point]:
    """Points whose bounding box is exactly that of the arc.

    The endpoints plus every axis-extreme point (0/90/180/270 deg) the arc
    passes through — the bulge of an arc can lie well outside its three
    defining points.

    Args:
        start: Arc start point.
        mid: Any point on the arc between ``start`` and ``end``.
        end: Arc end point.

    Returns:
        Points to include in a bounding-box calculation; just ``[start, end]``
        if the three points are collinear.
    """
    center = arc_center(start, mid, end)
    if center is None:
        return [start, end]
    r, a0, sweep = _arc_sweep(start, mid, end, center)
    pts = [start, end]
    for k in range(4):
        cardinal = k * math.pi / 2
        offset = (cardinal - a0) % (2 * math.pi) if sweep > 0 else (a0 - cardinal) % (2 * math.pi)
        if offset <= abs(sweep):
            pts.append((center[0] + r * math.cos(cardinal), center[1] + r * math.sin(cardinal)))
    return pts


def bezier_points(p0: Point, c1: Point, c2: Point, p3: Point, steps: int = 32) -> list[Point]:
    """Sample a cubic Bézier curve.

    Args:
        p0: Start point.
        c1: First control point.
        c2: Second control point.
        p3: End point.
        steps: Number of segments; ``steps + 1`` points are returned.

    Returns:
        Points at uniform parameter steps from ``p0`` to ``p3`` inclusive.
    """
    pts = []
    for i in range(steps + 1):
        t = i / steps
        u = 1 - t
        a, b, c, d = u * u * u, 3 * u * u * t, 3 * u * t * t, t * t * t
        pts.append(
            (a * p0[0] + b * c1[0] + c * c2[0] + d * p3[0], a * p0[1] + b * c1[1] + c * c2[1] + d * p3[1])
        )
    return pts


def arc_points(start: Point, mid: Point, end: Point, max_step_deg: float = 5.0) -> list[Point]:
    """Approximate a three-point arc as a polyline from ``start`` to ``end``.

    Args:
        start: Arc start point; returned exactly as the first vertex.
        mid: Any point on the arc between ``start`` and ``end``.
        end: Arc end point; returned exactly as the last vertex.
        max_step_deg: Maximum angle subtended by each segment, in degrees.

    Returns:
        Polyline vertices (at least three); just ``[start, end]`` if the points
        are collinear.
    """
    center = arc_center(start, mid, end)
    if center is None:
        return [start, end]
    cx, cy = center
    r, a0, sweep = _arc_sweep(start, mid, end, center)
    steps = max(2, math.ceil(abs(math.degrees(sweep)) / max_step_deg))
    pts = [
        (cx + r * math.cos(a0 + sweep * i / steps), cy + r * math.sin(a0 + sweep * i / steps))
        for i in range(steps + 1)
    ]
    pts[0], pts[-1] = start, end
    return pts
