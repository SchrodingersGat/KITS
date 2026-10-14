"""Neutral board data model shared by the GUI, mesher and solver.

Everything here is plain data in KiCad's board coordinate system: millimetres,
X to the right, Y *down*, angles in degrees counter-clockwise as displayed.
Nothing in this module depends on KiCad file syntax or on Qt.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path

from .geometry import Point, arc_extent, bezier_points

BBox = tuple[float, float, float, float]  # min_x, min_y, max_x, max_y
Polygon = list[Point]

EDGE_CUTS = "Edge.Cuts"

# Copper weight convention used by KiCad's stackup presets: 1 oz/ft² = 35 µm.
COPPER_MM_PER_OZ = 0.035
DEFAULT_COPPER_THICKNESS_MM = COPPER_MM_PER_OZ  # used when the board has no stackup entry
# Nominal copper weights (oz) that thicknesses are rounded to for display.
_NOMINAL_WEIGHTS_OZ = (0.25, 1 / 3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)
_NOMINAL_TOLERANCE = 0.08  # within 8% of a nominal weight counts as that weight


def copper_weight_oz(thickness_mm: float) -> float:
    """Copper weight in ounces per square foot for a thickness (35 µm per oz).

    Args:
        thickness_mm: Copper thickness in millimetres.

    Returns:
        The weight in oz.
    """
    return thickness_mm / COPPER_MM_PER_OZ


def format_copper_weight(thickness_mm: float) -> str:
    """Copper weight for display, using the nominal name when close (e.g. 18 µm → "½ oz").

    Args:
        thickness_mm: Copper thickness in millimetres.

    Returns:
        e.g. ``"1 oz"``, ``"½ oz"``, ``"2 oz"``, or ``"1.29 oz"`` for a non-standard thickness.
    """
    oz = copper_weight_oz(thickness_mm)
    for nominal in _NOMINAL_WEIGHTS_OZ:
        if abs(oz - nominal) <= _NOMINAL_TOLERANCE * nominal:
            names = {0.25: "¼", 1 / 3: "⅓", 0.5: "½", 1.5: "1½", 2.5: "2½"}
            return f"{names.get(nominal, f'{nominal:g}')} oz"
    return f"{oz:.3g} oz"


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
    """One entry of the physical stackup (copper, dielectric, mask, ...).

    Attributes:
        name: Layer name, e.g. ``"F.Cu"`` or ``"dielectric 1"``.
        type: KiCad layer type, e.g. ``"copper"``, ``"core"``, ``"prepreg"``.
        thickness_mm: Total thickness (all sub-layers), or None if not given.
        material: Material of the first sub-layer, e.g. ``"FR4"``.
        epsilon_r: Relative permittivity of the first sub-layer.
        loss_tangent: Loss tangent of the first sub-layer.
        sublayers: Number of sub-layers KiCad stacked into this entry (dielectrics
            made of several prepreg sheets are written as one entry with ``addsublayer``).
    """

    name: str
    type: str
    thickness_mm: float | None = None
    material: str | None = None
    epsilon_r: float | None = None
    loss_tangent: float | None = None
    sublayers: int = 1


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
    """A placed footprint and its pads; ``rotation`` in degrees CCW as displayed.

    ``body_graphics`` holds the footprint's Fab and courtyard drawings (board
    coordinates), which outline the component body.
    """

    reference: str
    value: str
    lib_id: str
    position: Point
    rotation: float
    side: str  # F.Cu or B.Cu
    pads: list[Pad] = field(default_factory=list)
    body_graphics: list[Graphic] = field(default_factory=list)


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
    # Thickness of copper layers the stackup doesn't list, set by the user
    # (see with_thickness_overrides); empty for a board as read from its file.
    copper_thickness_overrides: dict[str, float] = field(default_factory=dict)

    @property
    def copper_layers(self) -> list[str]:
        """Copper layer names in physical order, top to bottom."""
        names = [layer.name for layer in self.layers if layer.is_copper]
        return sorted(names, key=copper_sort_key)

    def copper_thicknesses(self) -> dict[str, tuple[float, bool]]:
        """Thickness of each copper layer, and whether it came from the stackup.

        Returns:
            ``{layer: (thickness_mm, from_stackup)}`` in physical order. Layers the
            stackup doesn't describe get :data:`DEFAULT_COPPER_THICKNESS_MM` and
            ``from_stackup=False``.
        """
        stack = {s.name: s.thickness_mm for s in self.stackup if is_copper_layer(s.name) and s.thickness_mm}
        stack.update(self.copper_thickness_overrides)
        return {
            name: (stack[name], True) if name in stack else (DEFAULT_COPPER_THICKNESS_MM, False)
            for name in self.copper_layers
        }

    def with_thickness_overrides(self, overrides: dict[str, float]) -> Board:
        """A copy of the board with some layer thicknesses replaced.

        Stackup entries (copper or dielectric) named in ``overrides`` get the new
        thickness; copper layers the stackup doesn't list are recorded in
        :attr:`copper_thickness_overrides`. Everything else is shared with this
        board, which is left unchanged.

        Args:
            overrides: ``{layer name: thickness in mm}``.

        Returns:
            The edited board (``self`` if there are no overrides).
        """
        if not overrides:
            return self
        listed = {s.name for s in self.stackup}
        stackup = [
            dataclasses.replace(s, thickness_mm=overrides[s.name]) if s.name in overrides else s
            for s in self.stackup
        ]
        extra = {
            name: mm for name, mm in overrides.items() if name in self.copper_layers and name not in listed
        }
        return dataclasses.replace(self, stackup=stackup, copper_thickness_overrides=extra)

    def pads(self):
        """Iterate over the pads of every footprint.

        Yields:
            Each :class:`Pad`, footprint by footprint.
        """
        for fp in self.footprints:
            yield from fp.pads

    def find_pads(self, ref: str, number: str) -> list[Pad]:
        """All pads of footprint ``ref`` numbered ``number``.

        Footprints can repeat a pad number (e.g. several thermal pads all
        numbered "9"); they are electrically one pad, so all are returned.

        Args:
            ref: Footprint reference, e.g. ``"U1"``.
            number: Pad number, e.g. ``"9"``.

        Returns:
            The matching pads, possibly empty.
        """
        number = str(number)
        return [p for fp in self.footprints if fp.reference == ref for p in fp.pads if p.number == number]

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
