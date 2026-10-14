"""Board model -> planar geometry for meshing, one region map per copper layer.

Each copper layer is meshed over the whole board outline, split into *faces*
that are either copper or bare board (dielectric). The faces form a planar
partition: neighbouring faces share their boundary vertices exactly, so the
mesh is conformal across copper/dielectric boundaries.

Simplifications (2.5D model, see PLAN.md):

* Drill holes are not cut out of the copper or the board; vias and plated
  holes are treated as solid copper discs. Their centres are returned as
  *anchor points* so the mesher can put a node there for inter-layer links.
* Geometry is simplified to a tolerance tied to the minimum mesh size, so
  finely-faceted arcs don't force elements far smaller than requested.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import shapely
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.base import BaseGeometry

from ..config import MeshSettings
from ..ingest.geometry import arc_points, bezier_points
from ..ingest.model import Board, Graphic, Pad

# Coordinates are snapped to this grid (mm) so near-coincident vertices from
# separate shapes merge instead of creating slivers. KiCad's own resolution is 1 nm.
SNAP_GRID_MM = 1e-4
# Simplification tolerance as a fraction of the minimum mesh size.
SIMPLIFY_FRACTION = 0.1
_QUAD_SEGS = 8  # segments per quarter circle when buffering round shapes


class GeometryError(ValueError):
    """The board can't be turned into meshable geometry (e.g. no closed outline)."""


@dataclass
class Face:
    """One region of a layer's planar partition.

    Attributes:
        polygon: The face outline, possibly with holes, in board millimetres.
        is_copper: True for copper, False for bare board.
    """

    polygon: Polygon
    is_copper: bool


@dataclass
class LayerGeometry:
    """Meshable geometry of one copper layer.

    Attributes:
        layer: Copper layer name, e.g. ``"F.Cu"``.
        faces: Planar partition of the board outline into copper/bare faces.
        anchors: Points (via and plated-hole centres) that must become mesh nodes.
    """

    layer: str
    faces: list[Face] = field(default_factory=list)
    anchors: list[tuple[float, float]] = field(default_factory=list)

    @property
    def copper_area(self) -> float:
        """Total copper area on this layer, in mm²."""
        return sum(f.polygon.area for f in self.faces if f.is_copper)

    @property
    def total_area(self) -> float:
        """Total meshed area (the board outline), in mm²."""
        return sum(f.polygon.area for f in self.faces)


# -- shape conversion ----------------------------------------------------------


def _to_board(geom: BaseGeometry, position: tuple[float, float], rotation: float) -> BaseGeometry:
    """Place a pad-local shape on the board.

    KiCad rotates counter-clockwise *as displayed* with Y pointing down, which
    is a clockwise rotation in shapely's Y-up maths convention.

    Args:
        geom: Shape in the pad's local frame.
        position: Pad anchor position in board coordinates.
        rotation: Pad rotation in KiCad degrees.

    Returns:
        The shape in board coordinates.
    """
    if rotation:
        geom = affinity.rotate(geom, -rotation, origin=(0, 0))
    return affinity.translate(geom, position[0], position[1])


def _stroke(points: list[tuple[float, float]], width: float) -> BaseGeometry:
    """Area covered by a round-capped stroke along a polyline.

    Args:
        points: Polyline vertices.
        width: Stroke width; zero-width strokes cover no area.

    Returns:
        The stroke area (empty if ``width`` is not positive).
    """
    if width <= 0 or len(points) < 2:
        return Polygon()
    return LineString(points).buffer(width / 2, quad_segs=_QUAD_SEGS)


def graphic_area(g: Graphic) -> BaseGeometry:
    """Copper area covered by a graphic shape (fill plus stroke).

    Args:
        g: A graphic on a copper layer, or a custom-pad primitive.

    Returns:
        The covered area.
    """
    if g.kind == "circle":
        (cx, cy), (px, py) = g.points
        radius = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
        disc = Point(cx, cy).buffer(radius, quad_segs=_QUAD_SEGS)
        if g.filled:
            return disc.buffer(g.width / 2) if g.width > 0 else disc
        return _stroke(list(disc.exterior.coords), g.width)
    if g.kind == "poly":
        ring = [*g.points, g.points[0]]
        if g.filled or g.width == 0:
            area = shapely.make_valid(Polygon(g.points)) if len(g.points) >= 3 else Polygon()
            return area.union(_stroke(ring, g.width))
        return _stroke(ring, g.width)
    if g.kind == "arc":
        return _stroke(arc_points(*g.points), g.width)
    if g.kind == "bezier" and len(g.points) == 4:
        return _stroke(bezier_points(*g.points), g.width)
    return _stroke(g.points, g.width)


def pad_area(pad: Pad) -> BaseGeometry:
    """Copper area of a pad, in board coordinates.

    Args:
        pad: The pad; its drill is ignored (holes are not cut, see module docs).

    Returns:
        The pad copper shape.
    """
    w, h = pad.size
    if pad.shape == "circle":
        local = Point(0, 0).buffer(w / 2, quad_segs=_QUAD_SEGS)
    elif pad.shape == "oval":
        half = abs(w - h) / 2
        ends = [(-half, 0), (half, 0)] if w >= h else [(0, -half), (0, half)]
        if half:
            local = LineString(ends).buffer(min(w, h) / 2, quad_segs=_QUAD_SEGS)
        else:
            local = Point(0, 0).buffer(w / 2, quad_segs=_QUAD_SEGS)
    elif pad.shape == "roundrect":
        r = min(pad.roundrect_ratio * min(w, h), min(w, h) / 2)
        inner = box(-w / 2 + r, -h / 2 + r, w / 2 - r, h / 2 - r)
        local = inner.buffer(r, quad_segs=_QUAD_SEGS) if r > 0 else box(-w / 2, -h / 2, w / 2, h / 2)
    elif pad.shape == "trapezoid":
        dx, dy = pad.rect_delta
        local = Polygon(
            [
                (-w / 2 - dy / 2, h / 2 + dx / 2),
                (w / 2 + dy / 2, h / 2 - dx / 2),
                (w / 2 - dy / 2, -h / 2 + dx / 2),
                (-w / 2 + dy / 2, -h / 2 - dx / 2),
            ]
        )
    elif pad.shape == "custom":
        if w == h:
            anchor = Point(0, 0).buffer(w / 2, quad_segs=_QUAD_SEGS)
        else:
            anchor = box(-w / 2, -h / 2, w / 2, h / 2)
        local = shapely.union_all([anchor, *(graphic_area(p) for p in pad.primitives)])
    else:  # rect, chamfered rect and anything unknown
        local = box(-w / 2, -h / 2, w / 2, h / 2)
    return _to_board(local, pad.position, pad.rotation)


# -- board outline -------------------------------------------------------------


def _outline_lines(g: Graphic) -> list[tuple[float, float]]:
    """Polyline approximating an Edge.Cuts graphic.

    Args:
        g: A graphic on Edge.Cuts.

    Returns:
        Vertices of the shape's path; closed shapes repeat their first point.
    """
    if g.kind == "arc":
        return arc_points(*g.points)
    if g.kind == "circle":
        (cx, cy), (px, py) = g.points
        radius = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
        return list(Point(cx, cy).buffer(radius, quad_segs=_QUAD_SEGS).exterior.coords)
    if g.kind == "poly":
        return [*g.points, g.points[0]]
    if g.kind == "bezier" and len(g.points) == 4:
        return bezier_points(*g.points)
    return list(g.points)


def board_outline(board: Board) -> BaseGeometry:
    """The board area enclosed by the Edge.Cuts outline.

    Outline segments may be drawn in any order and direction. Nested closed
    loops are treated even-odd: a loop inside the board is a cutout, a loop
    inside a cutout is board again.

    Args:
        board: The board.

    Returns:
        The board area as a (Multi)Polygon.

    Raises:
        GeometryError: If there is no Edge.Cuts outline or it doesn't enclose any area.
    """
    lines = [LineString(pts) for g in board.outline_graphics() if len(pts := _outline_lines(g)) >= 2]
    if not lines:
        raise GeometryError("the board has no outline on Edge.Cuts")
    noded = shapely.union_all(shapely.set_precision(lines, SNAP_GRID_MM))
    faces = list(shapely.get_parts(shapely.polygonize([noded])))
    if not faces:
        raise GeometryError("the Edge.Cuts outline is not closed")
    area: BaseGeometry = Polygon()
    for face in faces:
        area = area.symmetric_difference(Polygon(face.exterior))
    return area


# -- copper per layer ------------------------------------------------------------


def via_layers(board: Board) -> list[list[str]]:
    """Copper layers each via passes through, in board via order.

    Args:
        board: The board.

    Returns:
        For each via, the copper layers between (and including) its two end layers.
    """
    order = {name: i for i, name in enumerate(board.copper_layers)}
    spans = []
    for via in board.vias:
        ends = [order.get(name) for name in via.layers]
        if None in ends:
            spans.append([])
            continue
        lo, hi = sorted(ends)
        spans.append(board.copper_layers[lo : hi + 1])
    return spans


def layer_copper(board: Board, layer: str, spans: list[list[str]] | None = None) -> BaseGeometry:
    """Union of all copper on one layer (zone fills, tracks, pads, vias, graphics).

    Args:
        board: The board.
        layer: Copper layer name.
        spans: Precomputed :func:`via_layers` result, to avoid recomputing per layer.

    Returns:
        The copper area, not yet clipped to the board outline.
    """
    spans = via_layers(board) if spans is None else spans
    parts: list[BaseGeometry] = []
    for zone in board.zones:
        if not zone.is_rule_area:
            parts += [shapely.make_valid(Polygon(p)) for p in zone.fills.get(layer, []) if len(p) >= 3]

    tracks = [t for t in board.tracks if t.layer == layer]
    if tracks:
        lines = shapely.linestrings([[t.start, t.end] for t in tracks])
        parts += list(shapely.buffer(lines, [t.width / 2 for t in tracks], quad_segs=_QUAD_SEGS))
    parts += [_stroke(arc_points(a.start, a.mid, a.end), a.width) for a in board.arcs if a.layer == layer]
    parts += [pad_area(p) for p in board.pads() if layer in p.layers and p.kind != "np_thru_hole"]
    parts += [
        Point(v.position).buffer(v.diameter / 2, quad_segs=_QUAD_SEGS)
        for v, span in zip(board.vias, spans, strict=True)
        if layer in span
    ]
    parts += [graphic_area(g) for g in board.graphics if g.layer == layer]
    return shapely.union_all([p for p in parts if not p.is_empty])


def layer_anchors(
    board: Board, layer: str, spans: list[list[str]] | None = None
) -> list[tuple[float, float]]:
    """Centres of vias and plated through-hole pads present on a layer.

    Args:
        board: The board.
        layer: Copper layer name.
        spans: Precomputed :func:`via_layers` result.

    Returns:
        Unique anchor positions in board coordinates.
    """
    spans = via_layers(board) if spans is None else spans
    points = [v.position for v, span in zip(board.vias, spans, strict=True) if layer in span]
    points += [p.position for p in board.pads() if p.is_through_hole and layer in p.layers]
    return list(dict.fromkeys(points))


def _polygons(geom: BaseGeometry) -> list[Polygon]:
    """Polygon parts of a geometry, dropping points, lines and empties.

    Args:
        geom: Any shapely geometry.

    Returns:
        The non-empty polygons it contains.
    """
    return [g for g in shapely.get_parts(geom) if isinstance(g, Polygon) and not g.is_empty]


def _open(geom: BaseGeometry, distance: float) -> BaseGeometry:
    """Remove parts of a region thinner than ``2 * distance`` (morphological opening).

    Mitred joins keep sharp corners sharp.

    Args:
        geom: The region.
        distance: Half the width below which features are removed, in millimetres.

    Returns:
        The opened region.
    """
    return geom.buffer(-distance, join_style="mitre").buffer(distance, join_style="mitre")


def partition_layer(board_area: BaseGeometry, copper: BaseGeometry, tolerance: float) -> list[Face]:
    """Split the board area into copper and bare faces sharing exact boundaries.

    Args:
        board_area: The board outline area.
        copper: Copper on this layer (any extent; it is clipped to the board).
        tolerance: Simplification tolerance in millimetres.

    Returns:
        The faces of the partition.
    """
    board_s = shapely.set_precision(board_area.simplify(tolerance), SNAP_GRID_MM)
    # Copper running (almost) along the board edge would otherwise leave
    # micron-wide slivers between the two outlines, which mesh as degenerate
    # triangles. Snap copper onto the outline, then drop anything thinner than
    # 2 * tolerance from either region.
    copper_s = shapely.snap(copper.simplify(tolerance), board_s, tolerance).intersection(board_s)
    copper_s = _open(copper_s, tolerance)
    bare = _open(board_s.difference(copper_s), tolerance)
    copper_s = shapely.set_precision(board_s.difference(bare), SNAP_GRID_MM)
    # Snap-rounded noding: a copper vertex lying (within the grid) on a board
    # edge splits that edge, instead of the two nearly-coincident lines
    # enclosing a zero-width sliver face.
    noded = shapely.union_all([board_s.boundary, copper_s.boundary], grid_size=SNAP_GRID_MM)
    faces = []
    for face in shapely.get_parts(shapely.polygonize([noded])):
        inside = face.representative_point()
        if board_s.covers(inside):
            faces.append(Face(face, bool(copper_s.covers(inside))))
    return faces


def extract_layers(
    board: Board,
    settings: MeshSettings,
    progress: Callable[[str, float], None] | None = None,
) -> list[LayerGeometry]:
    """Build the meshable geometry for every copper layer.

    Args:
        board: The board.
        settings: Mesh settings; the minimum size sets the simplification tolerance.
        progress: Optional callback ``(message, fraction_of_this_step)``.

    Returns:
        One :class:`LayerGeometry` per copper layer, top to bottom.

    Raises:
        GeometryError: If the board has no usable outline.
    """
    outline = board_outline(board)
    spans = via_layers(board)
    tolerance = settings.min_size_mm * SIMPLIFY_FRACTION
    layers = []
    for i, layer in enumerate(board.copper_layers):
        if progress:
            progress(f"Extracting geometry: {layer}", i / len(board.copper_layers))
        copper = layer_copper(board, layer, spans)
        layers.append(
            LayerGeometry(
                layer=layer,
                faces=partition_layer(outline, copper, tolerance),
                anchors=layer_anchors(board, layer, spans),
            )
        )
    return layers
