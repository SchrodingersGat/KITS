"""Load KiCad projects (``.kicad_pro``) and boards (``.kicad_pcb``).

Only the subset of the board needed for electrical/thermal simulation is
extracted: layers and stackup, copper (tracks, arcs, vias, pads, zone fills,
copper graphics) and the Edge.Cuts outline. Everything else in the file is
ignored, which keeps the reader tolerant of KiCad format additions.

Handles both net styles: KiCad <= 9 (``(net 3)`` + a board-level net table)
and KiCad 10 (``(net "GND")``, no net numbers).
"""

from __future__ import annotations

import json
from pathlib import Path

from . import sexpr
from .geometry import arc_points, transform
from .model import (
    EDGE_CUTS,
    ArcTrack,
    Board,
    Drill,
    Footprint,
    Graphic,
    Layer,
    Pad,
    Point,
    Polygon,
    Project,
    StackupLayer,
    Track,
    Via,
    Zone,
    copper_sort_key,
    is_copper_layer,
)
from .sexpr import Node, Quoted, children, find, has_flag, value

PROJECT_SUFFIX = ".kicad_pro"
BOARD_SUFFIX = ".kicad_pcb"

# Footprint layers outlining the component body (kept per footprint, see Footprint.body_graphics).
BODY_LAYERS = ("F.Fab", "B.Fab", "F.CrtYd", "B.CrtYd")
MASK_LAYERS = ("F.Mask", "B.Mask")

_GRAPHIC_KINDS = {
    "line": "line",
    "arc": "arc",
    "circle": "circle",
    "rect": "rect",
    "poly": "poly",
    "curve": "bezier",
}


class BoardLoadError(Exception):
    """The project or board file could not be read."""


def load_project(path: str | Path) -> Project:
    """Load a ``.kicad_pro`` project and the ``.kicad_pcb`` board next to it.

    A ``.kicad_pcb`` path is also accepted; its project file is read if present.

    Args:
        path: Path to a ``.kicad_pro`` or ``.kicad_pcb`` file.

    Returns:
        The project, with ``settings`` holding the raw project JSON (empty if
        a board was given and no project file exists).

    Raises:
        BoardLoadError: If the path has the wrong suffix, the project file is
            missing or unreadable, or the board is missing or cannot be loaded.
    """
    path = Path(path)
    if path.suffix == BOARD_SUFFIX:
        pro_path, pcb_path = path.with_suffix(PROJECT_SUFFIX), path
    elif path.suffix == PROJECT_SUFFIX:
        pro_path, pcb_path = path, path.with_suffix(BOARD_SUFFIX)
    else:
        raise BoardLoadError(f"Not a KiCad project or board file: {path}")

    settings: dict = {}
    if pro_path.is_file():
        try:
            settings = json.loads(pro_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise BoardLoadError(f"Could not read KiCad project file {pro_path}: {exc}") from exc
    elif path.suffix == PROJECT_SUFFIX:
        raise BoardLoadError(f"KiCad project file not found: {pro_path}")

    if not pcb_path.is_file():
        raise BoardLoadError(f"No board file found for project (expected {pcb_path.name})")

    return Project(path=pro_path, name=pro_path.stem, board=load_board(pcb_path), settings=settings)


def load_board(path: str | Path) -> Board:
    """Load a ``.kicad_pcb`` board file.

    Args:
        path: Path to the ``.kicad_pcb`` file.

    Returns:
        The parsed board.

    Raises:
        BoardLoadError: If the file cannot be read, is not a well-formed
            S-expression, is not a KiCad board, or has unexpected content.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise BoardLoadError(f"Could not read board file {path}: {exc}") from exc
    try:
        root = sexpr.parse(text)
    except sexpr.SExprError as exc:
        raise BoardLoadError(f"Malformed board file {path}: {exc}") from exc
    if root[0] != "kicad_pcb":
        raise BoardLoadError(f"Not a KiCad board file: {path}")
    try:
        return _BoardReader(path, root).read()
    except (ValueError, IndexError, TypeError) as exc:
        raise BoardLoadError(f"Unexpected content in board file {path}: {exc}") from exc


# ---------------------------------------------------------------------------


def _xy(node: Node, keyword: str, default: Point | None = None) -> Point | None:
    """Read a coordinate pair child such as ``(start x y)``.

    Args:
        node: Parent node.
        keyword: Keyword of the child holding the pair, e.g. ``"start"``.
        default: Value returned if the child is missing or has fewer than two arguments.

    Returns:
        ``(x, y)`` in mm, or ``default``.
    """
    child = find(node, keyword)
    if child is None or len(child) < 3:
        return default
    return (float(child[1]), float(child[2]))


def _at(node: Node) -> tuple[Point, float]:
    """Read an ``(at x y [angle])`` child.

    Args:
        node: Node with an optional ``at`` child.

    Returns:
        ``((x, y), angle_deg)``; ``((0, 0), 0)`` if there is no ``at`` child and
        an angle of 0 if it is omitted.
    """
    at = find(node, "at")
    if at is None:
        return (0.0, 0.0), 0.0
    angle = float(at[3]) if len(at) > 3 and not isinstance(at[3], list) else 0.0
    return (float(at[1]), float(at[2])), angle


def _float(node: Node, keyword: str, default: float | None = None) -> float | None:
    """Read the first argument of child ``keyword`` as a float.

    Args:
        node: Parent node.
        keyword: Keyword of the child, e.g. ``"width"``.
        default: Value returned if the child or its argument is missing.

    Returns:
        The parsed value, or ``default``.
    """
    v = value(node, keyword)
    return default if v is None else float(v)


def _pts(node: Node) -> Polygon:
    """Read a ``(pts (xy ..) (arc (start ..) (mid ..) (end ..)) ...)`` list.

    Arc entries are flattened with :func:`arc_points`, dropping a first vertex
    that duplicates the previous point.

    Args:
        node: Node with a ``pts`` child (e.g. a polygon, filled_polygon or gr_poly).

    Returns:
        The vertices in mm; empty if there is no ``pts`` child.
    """
    pts_node = find(node, "pts")
    out: Polygon = []
    if pts_node is None:
        return out
    for item in children(pts_node):
        if item[0] == "xy":
            out.append((float(item[1]), float(item[2])))
        elif item[0] == "arc":
            seg = arc_points(_xy(item, "start"), _xy(item, "mid"), _xy(item, "end"))
            if out and out[-1] == seg[0]:
                seg = seg[1:]
            out.extend(seg)
    return out


def _stroke_width(node: Node) -> float:
    """Line width of a graphic, from ``(stroke (width ..))`` or legacy ``(width ..)``.

    Args:
        node: Graphic node.

    Returns:
        The stroke width in mm, 0 if not specified.
    """
    stroke = find(node, "stroke")
    if stroke is not None:
        return _float(stroke, "width", 0.0)
    return _float(node, "width", 0.0)


def _is_filled(node: Node) -> bool:
    """Whether a graphic is filled.

    Accepts both ``(fill yes)`` / ``(fill solid)`` and the older
    ``(fill (type solid))`` form.

    Args:
        node: Graphic node.

    Returns:
        True for a solid fill.
    """
    fill = find(node, "fill")
    if fill is None or len(fill) < 2:
        return False
    v = fill[1]
    if isinstance(v, list):  # (fill (type solid))
        v = value(fill, "type", "none")
    return v in ("solid", "yes")


class _BoardReader:
    """Converts a parsed ``kicad_pcb`` tree into a :class:`Board`.

    Holds the state needed while reading: the KiCad <= 9 net-number table, the
    ordered set of net names seen, and the board's copper layers.
    """

    def __init__(self, path: Path, root: Node):
        """Create a reader for one parsed board.

        Args:
            path: Board file path, stored on the resulting :class:`Board`.
            root: Root ``kicad_pcb`` node from :func:`sexpr.parse`.
        """
        self.path = path
        self.root = root
        self.net_table: dict[int, str] = {}
        self.nets_seen: dict[str, None] = {}  # ordered set
        self.copper: list[str] = []

    def read(self) -> Board:
        """Read the whole board.

        Returns:
            The board with layers, stackup, nets and all supported copper and
            Edge.Cuts items.
        """
        root = self.root
        layers = self._layers()
        self.copper = sorted((lay.name for lay in layers if lay.is_copper), key=copper_sort_key)

        for net in children(root, "net"):
            if len(net) >= 3:
                self.net_table[int(net[1])] = str(net[2])
                if net[2]:
                    self.nets_seen[str(net[2])] = None

        general = find(root, "general")
        board = Board(
            path=self.path,
            version=int(value(root, "version", 0)),
            generator=str(value(root, "generator", "")),
            thickness_mm=_float(general, "thickness") if general else None,
            layers=layers,
            stackup=self._stackup(),
            nets=[],
            footprints=[],
            tracks=[],
            arcs=[],
            vias=[],
            zones=[],
            graphics=[],
            pad_to_mask_clearance_mm=self._mask_clearance(),
        )

        for item in children(root):
            kw = item[0]
            if kw == "segment":
                board.tracks.append(
                    Track(
                        _xy(item, "start"),
                        _xy(item, "end"),
                        _float(item, "width", 0.0),
                        str(value(item, "layer")),
                        self._net(item),
                    )
                )
            elif kw == "arc":
                board.arcs.append(
                    ArcTrack(
                        _xy(item, "start"),
                        _xy(item, "mid"),
                        _xy(item, "end"),
                        _float(item, "width", 0.0),
                        str(value(item, "layer")),
                        self._net(item),
                    )
                )
            elif kw == "via":
                board.vias.append(self._via(item))
            elif kw == "zone":
                board.zones.append(self._zone(item))
            elif kw == "footprint":
                board.footprints.append(self._footprint(item, board))
            elif kw.startswith("gr_"):
                g = self._graphic(item)
                if g is not None:
                    board.graphics.append(g)
                elif str(value(item, "layer", "")) in MASK_LAYERS:
                    mask = self._graphic(item, any_layer=True)
                    if mask is not None:
                        board.mask_graphics.append(mask)

        board.nets = list(self.nets_seen)
        return board

    # -- layers ------------------------------------------------------------

    def _layers(self) -> list[Layer]:
        """Return the layers declared in the board's ``(layers ...)`` section."""
        node = find(self.root, "layers")
        out = []
        if node is None:
            return out
        for item in children(node):
            user_name = str(item[3]) if len(item) > 3 and not isinstance(item[3], list) else None
            out.append(Layer(int(item[0]), str(item[1]), str(item[2]), user_name))
        return out

    def _stackup(self) -> list[StackupLayer]:
        """Return the ``(setup (stackup ...))`` layers top to bottom, or [] if absent."""
        setup = find(self.root, "setup")
        stackup = find(setup, "stackup") if setup else None
        if stackup is None:
            return []
        layers = []
        for item in children(stackup, "layer"):
            # A dielectric of several sheets is one entry with repeated
            # (thickness ...) children separated by "addsublayer"; sum them.
            thicknesses = [float(t[1]) for t in children(item, "thickness") if len(t) > 1]
            layers.append(
                StackupLayer(
                    name=str(item[1]),
                    type=str(value(item, "type", "")),
                    thickness_mm=sum(thicknesses) if thicknesses else None,
                    material=(str(m) if (m := value(item, "material")) is not None else None),
                    epsilon_r=_float(item, "epsilon_r"),
                    loss_tangent=_float(item, "loss_tangent"),
                    sublayers=max(1, len(thicknesses)),
                )
            )
        return layers

    def _mask_layers(self, names) -> list[str]:
        """Solder mask layers named by a pad's layer list (``*.Mask`` expands to both).

        Args:
            names: Layer name atoms of the pad's ``(layers ...)`` node.

        Returns:
            ``"F.Mask"`` and/or ``"B.Mask"``.
        """
        out: list[str] = []
        for name in map(str, names):
            if name in ("*.Mask", "F&B.Mask"):
                out += list(MASK_LAYERS)
            elif name in MASK_LAYERS:
                out.append(name)
        return list(dict.fromkeys(out))

    def _mask_clearance(self) -> float:
        """The board's pad-to-mask clearance from ``(setup (pad_to_mask_clearance x))``.

        Returns:
            Millimetres (0 if not set, KiCad's default).
        """
        setup = find(self.root, "setup")
        return _float(setup, "pad_to_mask_clearance", 0.0) if setup else 0.0

    def _expand_layers(self, names) -> list[str]:
        """Copper layers named by ``names``, expanding ``*.Cu`` / ``F&B.Cu``.

        Args:
            names: Layer name atoms, e.g. the arguments of a ``(layers ...)`` node.

        Returns:
            De-duplicated copper layer names that exist on this board;
            non-copper names are dropped.
        """
        out: list[str] = []
        for name in names:
            name = str(name)
            if name == "*.Cu":
                out.extend(self.copper)
            elif name == "F&B.Cu":
                out.extend(["F.Cu", "B.Cu"])
            elif is_copper_layer(name):
                out.append(name)
        return [n for n in dict.fromkeys(out) if n in self.copper]

    # -- nets --------------------------------------------------------------

    def _net(self, node: Node) -> str | None:
        """Resolve the net name of an item and record it as seen.

        Handles KiCad 10 ``(net "GND")``, KiCad <= 9 pads ``(net 3 "GND")``,
        tracks/vias ``(net 3)`` (via the net table) and zones ``(net_name ..)``.

        Args:
            node: Item node (track, via, pad, zone or graphic).

        Returns:
            The net name, or None if the item has no net or the unnamed net.
        """
        name = value(node, "net_name")  # zones in KiCad <= 9
        if name is None:
            net = find(node, "net")
            if net is None or len(net) < 2:
                return None
            arg = net[1]
            if isinstance(arg, Quoted):  # KiCad 10: (net "GND")
                name = arg
            elif len(net) >= 3:  # pads in KiCad <= 9: (net 3 "GND")
                name = net[2]
            else:  # tracks/vias in KiCad <= 9: (net 3)
                name = self.net_table.get(int(arg), "")
        name = str(name)
        if not name:
            return None
        self.nets_seen[name] = None
        return name

    # -- items -------------------------------------------------------------

    def _via(self, node: Node) -> Via:
        """Build a :class:`Via` from a ``via`` node.

        Args:
            node: The ``via`` node.

        Returns:
            The via; layers default to F.Cu/B.Cu and type to ``through``.
        """
        pos, _ = _at(node)
        layers_node = find(node, "layers")
        layers = [str(x) for x in layers_node[1:3]] if layers_node else ["F.Cu", "B.Cu"]
        if len(layers) < 2:
            layers = ["F.Cu", "B.Cu"]
        via_type = str(value(node, "type", "through"))
        for flag in ("blind", "buried", "micro"):
            if has_flag(node, flag):
                via_type = flag
        return Via(
            position=pos,
            diameter=_float(node, "size", 0.0),
            drill=_float(node, "drill", 0.0),
            layers=(layers[0], layers[1]),
            net=self._net(node),
            via_type=via_type,
        )

    def _zone(self, node: Node) -> Zone:
        """Build a :class:`Zone` (copper pour or rule area) from a ``zone`` node.

        Args:
            node: The ``zone`` node, at board level or inside a footprint.

        Returns:
            The zone with its outline and per-layer filled polygons.
        """
        if (layer := value(node, "layer")) is not None:
            layer_names = [layer]
        else:
            layers_node = find(node, "layers")
            layer_names = layers_node[1:] if layers_node else []
        layers = self._expand_layers(layer_names)

        polygon = find(node, "polygon")
        fills: dict[str, list[Polygon]] = {}
        for fp in children(node, "filled_polygon"):
            fill_layer = str(value(fp, "layer", layers[0] if layers else ""))
            fills.setdefault(fill_layer, []).append(_pts(fp))

        name = value(node, "name")
        return Zone(
            net=self._net(node),
            layers=layers,
            outline=_pts(polygon) if polygon else [],
            fills=fills,
            name=str(name) if name is not None else None,
            is_rule_area=find(node, "keepout") is not None,
            priority=int(value(node, "priority", 0)),
        )

    def _footprint(self, node: Node, board: Board) -> Footprint:
        """Build a :class:`Footprint` with its pads from a ``footprint`` node.

        Footprint graphics and zones are transformed to board coordinates and
        appended to ``board.graphics`` / ``board.zones`` rather than the footprint.

        Args:
            node: The ``footprint`` node.
            board: Board being built; receives the footprint's graphics and zones.

        Returns:
            The footprint.
        """
        pos, rot = _at(node)
        reference, value_text = "", ""
        self._fp_mask_margin = _float(node, "solder_mask_margin", None)
        for prop in children(node, "property"):
            if len(prop) >= 3 and prop[1] == "Reference":
                reference = str(prop[2])
            elif len(prop) >= 3 and prop[1] == "Value":
                value_text = str(prop[2])
        for text in children(node, "fp_text"):  # KiCad <= 7
            if len(text) >= 3 and text[1] == "reference" and not reference:
                reference = str(text[2])
            elif len(text) >= 3 and text[1] == "value" and not value_text:
                value_text = str(text[2])

        fp = Footprint(
            reference=reference,
            value=value_text,
            lib_id=str(node[1]) if len(node) > 1 and not isinstance(node[1], list) else "",
            position=pos,
            rotation=rot,
            side=str(value(node, "layer", "F.Cu")),
        )
        for item in children(node):
            kw = item[0]
            if kw == "pad":
                fp.pads.append(self._pad(item, fp))
            elif kw.startswith("fp_"):
                g = self._graphic(item, origin=pos, angle=rot)
                if g is not None:
                    board.graphics.append(g)
                elif str(value(item, "layer", "")) in BODY_LAYERS:
                    body = self._graphic(item, origin=pos, angle=rot, any_layer=True)
                    if body is not None:
                        fp.body_graphics.append(body)
                elif str(value(item, "layer", "")) in MASK_LAYERS:
                    mask = self._graphic(item, origin=pos, angle=rot, any_layer=True)
                    if mask is not None:
                        board.mask_graphics.append(mask)
            elif kw == "zone":
                board.zones.append(self._zone(item))
        return fp

    def _pad(self, node: Node, fp: Footprint) -> Pad:
        """Build a :class:`Pad` from a ``pad`` node inside a footprint.

        Args:
            node: The ``pad`` node.
            fp: Owning footprint; its position and rotation place the pad on the board.

        Returns:
            The pad in board coordinates. Custom-pad primitives stay in the
            pad's local frame.
        """
        local, angle = _at(node)
        size = _xy(node, "size", (0.0, 0.0))
        layers_node = find(node, "layers")

        drill = None
        drill_node = find(node, "drill")
        if drill_node is not None:
            nums = [float(a) for a in drill_node[1:] if not isinstance(a, list) and a != "oval"]
            if nums:
                w = nums[0]
                h = nums[1] if len(nums) > 1 and has_flag(drill_node, "oval") else w
                drill = Drill(w, h, _xy(drill_node, "offset", (0.0, 0.0)))

        primitives = []
        prims = find(node, "primitives")
        if prims is not None:
            for item in children(prims):
                g = self._graphic(item, any_layer=True)
                if g is not None:
                    primitives.append(g)

        return Pad(
            number=str(node[1]),
            kind=str(node[2]),
            shape=str(node[3]),
            position=transform(local, fp.position, fp.rotation),
            rotation=angle,
            size=size,
            layers=self._expand_layers(layers_node[1:] if layers_node else []),
            net=self._net(node),
            drill=drill,
            roundrect_ratio=_float(node, "roundrect_rratio", 0.25),
            rect_delta=_xy(node, "rect_delta", (0.0, 0.0)),
            primitives=primitives,
            footprint_ref=fp.reference,
            mask_layers=self._mask_layers(layers_node[1:] if layers_node else []),
            mask_margin=_float(node, "solder_mask_margin", self._fp_mask_margin),
        )

    def _graphic(
        self,
        node: Node,
        origin: Point = (0.0, 0.0),
        angle: float = 0.0,
        any_layer: bool = False,
    ) -> Graphic | None:
        """Parse a gr_*/fp_* shape; returns None for layers/kinds we don't use.

        Rectangles are converted to four-vertex polys.

        Args:
            node: The ``gr_*``, ``fp_*`` or pad-primitive node.
            origin: Footprint position used to transform the points to board coordinates.
            angle: Footprint rotation in degrees, counter-clockwise as displayed.
            any_layer: Accept the shape regardless of its layer (for custom pad primitives).

        Returns:
            The graphic, or None if its kind or layer is not used or a point is missing.
        """
        kind = _GRAPHIC_KINDS.get(node[0].partition("_")[2])
        if kind is None:
            return None
        layer = str(value(node, "layer", ""))
        if not any_layer and layer != EDGE_CUTS and layer not in self.copper:
            return None

        if kind == "line":
            pts = [_xy(node, "start"), _xy(node, "end")]
        elif kind == "arc":
            pts = [_xy(node, "start"), _xy(node, "mid"), _xy(node, "end")]
        elif kind == "circle":
            pts = [_xy(node, "center"), _xy(node, "end")]
        elif kind == "rect":
            (x0, y0), (x1, y1) = _xy(node, "start"), _xy(node, "end")
            kind, pts = "poly", [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        else:  # poly, bezier
            pts = _pts(node)
        if any(p is None for p in pts) or not pts:
            return None

        if origin != (0.0, 0.0) or angle:
            pts = [transform(p, origin, angle) for p in pts]
        return Graphic(
            kind=kind,
            layer=layer,
            points=pts,
            width=_stroke_width(node),
            filled=_is_filled(node),
            net=self._net(node) if layer in self.copper else None,
        )
