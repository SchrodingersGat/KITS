"""Board model -> planar geometry for meshing, one region map per copper layer.

Each copper layer is meshed over the whole board outline, split into *faces*
that are either copper or bare board (dielectric). The faces form a planar
partition: neighbouring faces share their boundary vertices exactly, so the
mesh is conformal across copper/dielectric boundaries.

Simplifications (2.5D model, see PLAN.md):

* Drill holes (vias, plated and non-plated holes) are cut out of every layer
  they pass through, at the drill radius. Each hole edge is a regular polygon
  with at least ``min_circle_points`` vertices, generated identically for every
  layer, so a hole's edge nodes sit at the same XY on each layer it crosses
  (plated barrels can later link layers through them).
* Other geometry is simplified to a tolerance tied to the minimum mesh size, so
  finely-faceted arcs don't force elements far smaller than requested. Hole
  edges are cut after simplification and keep all their vertices.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import shapely
import shapely.geometry.polygon
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.base import BaseGeometry

from ..config import MeshSettings
from ..ingest.geometry import arc_points, bezier_points
from ..ingest.model import Board, Drill, Footprint, Graphic, Pad

# Coordinates are snapped to this grid (mm) so near-coincident vertices from
# separate shapes merge instead of creating slivers. KiCad's own resolution is 1 nm.
SNAP_GRID_MM = 1e-4
# Simplification tolerance as a fraction of the minimum mesh size.
SIMPLIFY_FRACTION = 0.1
_QUAD_SEGS = 8  # segments per quarter circle when buffering round shapes
# A copper outline vertex turning by at least this much (inwards) is a corner
# to refine around; faceted arcs and hole edges turn far less per vertex.
CORNER_MIN_TURN_DEG = 60.0


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
        faces: Planar partition of the board outline (minus holes) into copper/bare faces.
        holes: Drill-hole outlines cut from this layer.
        pad_edges: Pad outlines running through copper, to be embedded as mesh
            edges so each pad's area is resolved exactly (pads are where current
            enters). Kept a tolerance away from copper and hole edges.
        corners: Concave copper corners (see :func:`concave_copper_corners`), where
            current crowds and the mesh is refined.
    """

    layer: str
    faces: list[Face] = field(default_factory=list)
    holes: list[Polygon] = field(default_factory=list)
    pad_edges: list[LineString] = field(default_factory=list)
    corners: list[tuple[float, float]] = field(default_factory=list)

    @property
    def copper_area(self) -> float:
        """Total copper area on this layer, in mm²."""
        return sum(f.polygon.area for f in self.faces if f.is_copper)

    @property
    def total_area(self) -> float:
        """Total meshed area (the board outline minus holes), in mm²."""
        return sum(f.polygon.area for f in self.faces)


@dataclass
class Hole:
    """A drill hole and the copper layers it passes through.

    Attributes:
        polygon: The hole outline in board millimetres.
        layers: Copper layers the hole is cut from.
        plated: True for vias and plated through-holes.
    """

    polygon: Polygon
    layers: list[str]
    plated: bool


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


# Where a component body's outline is taken from, in order of preference.
BODY_SOURCES = (("Fab outline", ("F.Fab", "B.Fab")), ("courtyard", ("F.CrtYd", "B.CrtYd")))


def footprint_body(footprint: Footprint) -> tuple[BaseGeometry, str]:
    """The plan area of a component body.

    Taken from the footprint's Fab drawing, else its courtyard, else the
    bounding box of its copper pads. Closed loops in the drawing (rectangles,
    polygons, circles, or lines that join up) are filled and merged, so pin-1
    marks inside the outline don't matter; a drawing with no closed loop gives
    the convex hull of its points.

    Args:
        footprint: The footprint.

    Returns:
        ``(area, source)``: the area in board millimetres (empty if the footprint
        has neither drawings nor copper pads) and ``"Fab outline"``,
        ``"courtyard"`` or ``"pads"``.
    """
    for source, layers in BODY_SOURCES:
        lines = [
            LineString(pts)
            for g in footprint.body_graphics
            if g.layer in layers and len(pts := _outline_lines(g)) >= 2
        ]
        if not lines:
            continue
        noded = shapely.union_all(shapely.set_precision(lines, SNAP_GRID_MM))
        faces = [Polygon(face.exterior) for face in shapely.get_parts(shapely.polygonize([noded]))]
        area = shapely.union_all(faces) if faces else noded.convex_hull
        if area.area > 0:
            return area, source
    pads = [pad_area(p) for p in footprint.pads if p.layers]
    if not pads:
        return Polygon(), "pads"
    return box(*shapely.union_all(pads).bounds), "pads"


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


def circle_points(radius: float, settings: MeshSettings) -> int:
    """Number of vertices for a hole edge of the given radius.

    At least ``settings.min_circle_points``, and more if needed to keep edge
    segments no longer than the maximum mesh size.

    Args:
        radius: Hole radius in millimetres.
        settings: Mesh settings.

    Returns:
        The vertex count for a full circle.
    """
    by_size = math.ceil(2 * math.pi * radius / settings.max_size_mm)
    return max(settings.min_circle_points, by_size)


def hole_polygon(
    drill: Drill, position: tuple[float, float], rotation: float, settings: MeshSettings
) -> Polygon:
    """Outline of a round hole or slot as a polygon with vertices on the drill edge.

    Round holes are regular polygons starting at angle 0. Slots (oval drills) are
    two half-circles, each with half the full-circle vertex count, joined by
    straight sides.

    Args:
        drill: The drill size and offset (in the pad's frame).
        position: Pad or via position in board coordinates.
        rotation: Pad rotation in KiCad degrees (0 for vias).
        settings: Mesh settings (``min_circle_points`` and max size).

    Returns:
        The hole outline in board coordinates.
    """
    w, h = drill.width, drill.height
    r = min(w, h) / 2
    n = circle_points(r, settings)
    if not drill.is_oval:
        pts = [(r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n)) for k in range(n)]
    else:
        half = abs(w - h) / 2
        per_end = math.ceil(n / 2)
        pts = []
        for cx, start in ((half, -math.pi / 2), (-half, math.pi / 2)):
            angles = [start + math.pi * k / per_end for k in range(per_end + 1)]
            pts += [(cx + r * math.cos(a), r * math.sin(a)) for a in angles]
        if h > w:  # long axis vertical
            pts = [(-y, x) for x, y in pts]
    local = Polygon([(x + drill.offset[0], y + drill.offset[1]) for x, y in pts])
    # Snap to the same grid as the partition, so the hole's vertices appear
    # unchanged in the meshed faces (the mesher looks them up by coordinate).
    return shapely.set_precision(_to_board(local, position, rotation), SNAP_GRID_MM)


def board_holes(board: Board, settings: MeshSettings, spans: list[list[str]] | None = None) -> list[Hole]:
    """All drill holes on the board with the layers they cut.

    Vias cut the layers they span (blind/buried vias only some); plated and
    non-plated through-hole pads cut every copper layer.

    Args:
        board: The board.
        settings: Mesh settings for the hole outlines.
        spans: Precomputed :func:`via_layers` result.

    Returns:
        The holes.
    """
    spans = via_layers(board) if spans is None else spans
    holes = [
        Hole(hole_polygon(Drill(v.drill, v.drill), v.position, 0.0, settings), span, True)
        for v, span in zip(board.vias, spans, strict=True)
        if v.drill > 0 and span
    ]
    holes += [
        Hole(
            hole_polygon(p.drill, p.position, p.rotation, settings),
            list(board.copper_layers),
            p.kind == "thru_hole",
        )
        for p in board.pads()
        if p.drill is not None and p.drill.width > 0 and p.kind in ("thru_hole", "np_thru_hole")
    ]
    return holes


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


def partition_layer(
    board_area: BaseGeometry,
    copper: BaseGeometry,
    tolerance: float,
    holes: list[Polygon] | None = None,
) -> list[Face]:
    """Split the board area into copper and bare faces sharing exact boundaries.

    Args:
        board_area: The board outline area.
        copper: Copper on this layer (any extent; it is clipped to the board).
        tolerance: Simplification tolerance in millimetres.
        holes: Drill holes to cut out, after simplification so they keep every vertex.

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
    if holes:
        # Holes are cut last, so the offsets above never touch (and round) their
        # edges, and after simplifying, so they keep every vertex. Copper edges
        # within the tolerance of a hole edge (a pad the same size as its drill,
        # a track running into a via) are first snapped onto the hole's
        # vertices, so the two don't leave a sliver or near-duplicate nodes.
        cut = shapely.union_all(holes)
        copper_s = shapely.snap(copper_s, cut, tolerance)
        board_s = board_s.difference(cut)
        copper_s = shapely.set_precision(copper_s, SNAP_GRID_MM).intersection(board_s)
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


def pad_edge_lines(board: Board, layer: str, faces: list[Face], tolerance: float) -> list[LineString]:
    """Pad outlines on a layer that run through the interior of its copper.

    Pad edges that coincide with the copper outline are already mesh edges;
    the rest (where a pad joins a track or pour) are returned so they can be
    embedded. They stop ``tolerance`` short of copper and hole edges so they
    never create near-duplicate nodes there.

    Args:
        board: The board.
        layer: Copper layer name.
        faces: The layer's partition (see :func:`partition_layer`).
        tolerance: Inset from copper/hole edges, and minimum kept line length, in mm.

    Returns:
        Non-overlapping line pieces, noded where pad outlines cross.
    """
    copper = shapely.union_all([f.polygon for f in faces if f.is_copper])
    interior = copper.buffer(-tolerance)
    if interior.is_empty:
        return []
    outlines = [
        pad_area(p).simplify(tolerance).boundary
        for p in board.pads()
        if layer in p.layers and p.kind != "np_thru_hole"
    ]
    if not outlines:
        return []
    clipped = shapely.union_all(outlines, grid_size=SNAP_GRID_MM).intersection(interior)
    lines = [g for g in shapely.get_parts(shapely.line_merge(clipped) if not clipped.is_empty else clipped)]
    return [g for g in lines if isinstance(g, LineString) and g.length > tolerance]


def concave_copper_corners(
    faces: list[Face], min_turn_deg: float = CORNER_MIN_TURN_DEG
) -> list[tuple[float, float]]:
    """Concave (re-entrant) corners of the copper outline.

    At a concave corner copper wraps around the corner, e.g. where a track
    joins a wider pour or on the inside of a track bend. Current crowds there
    (the density is singular at a sharp corner), so the mesh is refined around
    them. Convex corners, where current thins out, are ignored, and so are the
    shallow vertices of faceted arcs and hole edges (below ``min_turn_deg``).

    Args:
        faces: A layer's partition (see :func:`partition_layer`).
        min_turn_deg: Smallest change of direction, in degrees, that counts as a corner.

    Returns:
        Corner positions in board millimetres, without duplicates.
    """
    corners: dict[tuple[float, float], None] = {}
    min_turn = math.radians(min_turn_deg)
    for face in faces:
        if not face.is_copper:
            continue
        polygon = shapely.geometry.polygon.orient(face.polygon, sign=1.0)  # copper on the left
        for ring in (polygon.exterior, *polygon.interiors):
            pts = np.asarray(ring.coords)[:-1]
            if len(pts) < 3:
                continue
            incoming = pts - np.roll(pts, 1, axis=0)
            outgoing = np.roll(pts, -1, axis=0) - pts
            cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
            dot = (incoming * outgoing).sum(axis=1)
            turn = np.arctan2(cross, dot)  # > 0 left (convex), < 0 right (concave)
            for x, y in pts[turn <= -min_turn]:
                corners[(float(x), float(y))] = None
    return list(corners)


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
    holes = board_holes(board, settings, spans)
    tolerance = settings.min_size_mm * SIMPLIFY_FRACTION
    layers = []
    for i, layer in enumerate(board.copper_layers):
        if progress:
            progress(f"Extracting geometry: {layer}", i / len(board.copper_layers))
        copper = layer_copper(board, layer, spans)
        layer_holes = [h.polygon for h in holes if layer in h.layers]
        faces = partition_layer(outline, copper, tolerance, layer_holes)
        layers.append(
            LayerGeometry(
                layer=layer,
                faces=faces,
                holes=layer_holes,
                pad_edges=pad_edge_lines(board, layer, faces, settings.min_size_mm),
                corners=concave_copper_corners(faces),
            )
        )
    return layers
