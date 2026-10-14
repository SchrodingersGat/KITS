"""Neutral board data model shared by the GUI, mesher and solver.

Everything here is plain data in KiCad's board coordinate system: millimetres,
X to the right, Y *down*, angles in degrees counter-clockwise as displayed.
Nothing in this module depends on KiCad file syntax or on Qt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from .geometry import Point, arc_extent, bezier_points

BBox = tuple[float, float, float, float]  # min_x, min_y, max_x, max_y
Polygon = list[Point]

EDGE_CUTS = "Edge.Cuts"


def is_copper_layer(name: str) -> bool:
    """True if ``name`` is a copper layer name (ends with ``.Cu``).

    Args:
        name: KiCad canonical layer name, e.g. ``"In1.Cu"``.

    Returns:
        Whether the layer is copper.
    """
    return name.endswith(".Cu")


def copper_sort_key(name: str) -> tuple[int, int]:
    """Physical stack order: F.Cu, In1.Cu, In2.Cu, ..., B.Cu.

    Args:
        name: Copper layer name.

    Returns:
        A sort key placing F.Cu first, inner layers by number, then B.Cu.
    """
    if name == "F.Cu":
        return (0, 0)
    if name == "B.Cu":
        return (2, 0)
    digits = "".join(ch for ch in name if ch.isdigit())
    return (1, int(digits) if digits else 0)


@dataclass
class Layer:
    """A layer declared in the board's ``(layers ...)`` section."""

    ordinal: int
    name: str
    kind: str  # signal, power, mixed, jumper, user, ...
    user_name: str | None = None

    @property
    def is_copper(self) -> bool:
        """Whether this is a copper layer."""
        return is_copper_layer(self.name)


@dataclass
class StackupLayer:
    """One entry of the physical stackup (copper, dielectric, mask, ...)."""

    name: str
    type: str
    thickness_mm: float | None = None
    material: str | None = None
    epsilon_r: float | None = None
    loss_tangent: float | None = None


@dataclass
class Graphic:
    """A drawn shape on a copper or Edge.Cuts layer.

    ``points`` depend on ``kind``:
      line:   [start, end]
      arc:    [start, mid, end]
      circle: [center, point_on_circle]
      poly:   polygon vertices (arcs flattened; rectangles become polys)
      bezier: [p0, c1, c2, p3]
    """

    kind: str
    layer: str
    points: list[Point]
    width: float = 0.0
    filled: bool = False
    net: str | None = None


@dataclass
class Track:
    """A straight copper track segment; ``width`` in mm."""

    start: Point
    end: Point
    width: float
    layer: str
    net: str | None = None


@dataclass
class ArcTrack:
    """A circular-arc copper track through ``start``, ``mid`` and ``end``; ``width`` in mm."""

    start: Point
    mid: Point
    end: Point
    width: float
    layer: str
    net: str | None = None


@dataclass
class Via:
    """A plated via; ``diameter`` (pad) and ``drill`` (hole) in mm."""

    position: Point
    diameter: float
    drill: float
    layers: tuple[str, str]  # outermost copper layers spanned
    net: str | None = None
    via_type: str = "through"  # through, blind, buried, micro


@dataclass
class Drill:
    """A pad's drill hole, in mm; ``offset`` is relative to the pad centre."""

    width: float
    height: float  # == width for round holes
    offset: Point = (0.0, 0.0)

    @property
    def is_oval(self) -> bool:
        """Whether the hole is a slot (width differs from height)."""
        return self.width != self.height


@dataclass
class Pad:
    """A footprint pad, already placed in absolute board coordinates.

    ``size`` and ``rect_delta`` are in mm in the pad's local frame (before
    ``rotation`` is applied).
    """

    number: str
    kind: str  # smd, thru_hole, np_thru_hole, connect
    shape: str  # circle, rect, oval, roundrect, trapezoid, custom
    position: Point  # absolute board coordinates
    rotation: float  # absolute angle, degrees
    size: tuple[float, float]
    layers: list[str]  # copper layers only, wildcards expanded
    net: str | None = None
    drill: Drill | None = None
    roundrect_ratio: float = 0.25
    rect_delta: tuple[float, float] = (0.0, 0.0)  # trapezoid
    # Custom pad primitives, in the pad's local frame (before rotation).
    primitives: list[Graphic] = field(default_factory=list)
    footprint_ref: str = ""

    @property
    def is_through_hole(self) -> bool:
        """Whether this is a plated through-hole pad."""
        return self.kind == "thru_hole"


@dataclass
class Footprint:
    """A placed footprint and its pads; ``rotation`` in degrees CCW as displayed."""

    reference: str
    value: str
    lib_id: str
    position: Point
    rotation: float
    side: str  # F.Cu or B.Cu
    pads: list[Pad] = field(default_factory=list)


@dataclass
class Zone:
    """A copper pour. ``fills`` holds the filled polygons per copper layer.

    KiCad stores fills "fractured" into hole-free outlines, so each polygon
    can be drawn/meshed as a simple (if self-touching) outline.
    """

    net: str | None
    layers: list[str]
    outline: Polygon
    fills: dict[str, list[Polygon]] = field(default_factory=dict)
    name: str | None = None
    is_rule_area: bool = False
    priority: int = 0


@dataclass
class Board:
    """A loaded ``.kicad_pcb`` board: layers, stackup, copper items and outline.

    ``nets`` lists the net names actually used, in first-seen order.
    ``thickness_mm`` is the board thickness from the ``(general ...)`` section,
    if present.
    """

    path: Path
    version: int
    generator: str
    thickness_mm: float | None
    layers: list[Layer]
    stackup: list[StackupLayer]
    nets: list[str]
    footprints: list[Footprint]
    tracks: list[Track]
    arcs: list[ArcTrack]
    vias: list[Via]
    zones: list[Zone]
    graphics: list[Graphic]  # board-level and footprint graphics on copper/Edge.Cuts

    @property
    def copper_layers(self) -> list[str]:
        """Copper layer names in physical order, top to bottom."""
        names = [layer.name for layer in self.layers if layer.is_copper]
        return sorted(names, key=copper_sort_key)

    def pads(self):
        """Iterate over the pads of every footprint.

        Yields:
            Each :class:`Pad`, footprint by footprint.
        """
        for fp in self.footprints:
            yield from fp.pads

    def outline_graphics(self) -> list[Graphic]:
        """Return the graphics on the Edge.Cuts layer (the board outline)."""
        return [g for g in self.graphics if g.layer == EDGE_CUTS]

    @property
    def has_outline(self) -> bool:
        """Whether the board has any Edge.Cuts graphics."""
        return any(g.layer == EDGE_CUTS for g in self.graphics)

    def outline_bounding_box(self) -> BBox | None:
        """(min_x, min_y, max_x, max_y) of the Edge.Cuts outline, or None if there is none."""
        return _bbox([p for g in self.outline_graphics() for p in _graphic_extent(g)])

    def copper_bounding_box(self) -> BBox | None:
        """Extent of all copper: zones, tracks (with width), pads and vias (with size)."""
        pts: list[Point] = []
        for zone in self.zones:
            if not zone.is_rule_area:
                pts += zone.outline
                pts += [p for polys in zone.fills.values() for poly in polys for p in poly]
        pts += [p for g in self.graphics if g.layer != EDGE_CUTS for p in _graphic_extent(g)]
        for t in self.tracks:
            pts += _grow([t.start, t.end], t.width / 2)
        for a in self.arcs:
            pts += _grow(arc_extent(a.start, a.mid, a.end), a.width / 2)
        for pad in self.pads():
            # Half-diagonal covers the pad at any rotation.
            pts += _grow([pad.position], math.hypot(*pad.size) / 2)
        for v in self.vias:
            pts += _grow([v.position], v.diameter / 2)
        return _bbox(pts)

    def bounding_box(self) -> BBox | None:
        """Board outline extent, falling back to the copper extent if there is no outline."""
        return self.outline_bounding_box() or self.copper_bounding_box()


@dataclass
class Project:
    """A KiCad project: the ``.kicad_pro`` settings plus its board."""

    path: Path
    name: str
    board: Board
    settings: dict = field(default_factory=dict)  # raw .kicad_pro JSON


def _bbox(pts: list[Point]) -> BBox | None:
    """Axis-aligned bounding box of ``pts``.

    Args:
        pts: Points to enclose.

    Returns:
        ``(min_x, min_y, max_x, max_y)``, or None if ``pts`` is empty.
    """
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _grow(pts: list[Point], r: float) -> list[Point]:
    """Corners of the boxes of half-size ``r`` around each point.

    Args:
        pts: Centre points.
        r: Half-size of each box, in mm.

    Returns:
        Two opposite corners (min and max) per input point.
    """
    return [q for x, y in pts for q in ((x - r, y - r), (x + r, y + r))]


def _graphic_extent(g: Graphic) -> list[Point]:
    """Points whose bounding box covers the graphic's geometry (stroke width ignored).

    Args:
        g: Graphic to measure.

    Returns:
        Box corners for circles, axis extremes for arcs, sampled points for
        Béziers, and the raw points otherwise.
    """
    if g.kind == "circle":
        (cx, cy), (px, py) = g.points
        r = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
        return [(cx - r, cy - r), (cx + r, cy + r)]
    if g.kind == "arc":
        return arc_extent(*g.points)
    if g.kind == "bezier" and len(g.points) == 4:
        return bezier_points(*g.points)
    return g.points
