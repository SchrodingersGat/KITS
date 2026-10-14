"""Mesh each copper layer's geometry with gmsh and write a combined ``.msh`` file.

gmsh holds global state and is not thread-safe; callers that need a
responsive UI should run :func:`mesh_board` in a separate process (see
:mod:`kits.mesh.cli`).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import gmsh
import numpy as np
import shapely
from shapely.geometry import Point

from ..config import MeshSettings
from ..ingest.model import Board, is_copper_layer
from .fingerprint import MeshFingerprint
from .geometry import LayerGeometry, extract_layers
from .model import BoardMesh, LayerMesh

DEFAULT_BOARD_THICKNESS_MM = 1.6
_TRIANGLE = 2  # gmsh element type: 3-node triangle
_COORD_DECIMALS = 6  # vertex de-duplication precision, mm

Progress = Callable[[str, float], None]


class MeshError(RuntimeError):
    """gmsh failed to mesh the board."""


def layer_heights(board: Board) -> dict[str, float]:
    """Height of each copper layer's mid-plane above the board bottom.

    Uses the stackup's copper and dielectric thicknesses when available,
    otherwise spaces the copper layers evenly through the board thickness.

    Args:
        board: The board.

    Returns:
        Mapping of copper layer name to height in millimetres.
    """
    copper = board.copper_layers
    stack = [s for s in board.stackup if is_copper_layer(s.name) or s.type in ("core", "prepreg")]
    if stack and {s.name for s in stack} >= set(copper) and all(s.thickness_mm for s in stack):
        heights, z = {}, 0.0
        for s in reversed(stack):  # stackup is listed top to bottom
            if is_copper_layer(s.name):
                heights[s.name] = z + s.thickness_mm / 2
            z += s.thickness_mm
        return {name: heights[name] for name in copper}
    thickness = board.thickness_mm or DEFAULT_BOARD_THICKNESS_MM
    if len(copper) == 1:
        return {copper[0]: thickness}
    return {name: thickness * (1 - i / (len(copper) - 1)) for i, name in enumerate(copper)}


def _configure(settings: MeshSettings) -> None:
    """Apply mesh size and algorithm options to the current gmsh session.

    Args:
        settings: Mesh size limits.
    """
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", settings.max_size_mm)
    gmsh.option.setNumber("Mesh.MeshSizeMin", settings.min_size_mm)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    # Don't propagate boundary segment lengths into the interior: with dense
    # copper outlines that makes the whole board as fine as its smallest
    # feature (8x more elements on pic_programmer). Elements still match the
    # boundary segments at edges and grow towards max size inside.
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay


def mesh_layer(geom: LayerGeometry, settings: MeshSettings, z_mm: float) -> LayerMesh:
    """Mesh one layer's faces with gmsh's built-in kernel.

    Faces are built from shared points and lines, so adjacent copper and bare
    faces get identical nodes along their common boundary. Anchor points are
    embedded so each becomes a mesh node.

    Must be called inside an initialised gmsh session.

    Args:
        geom: The layer's planar partition and anchor points.
        settings: Mesh size limits.
        z_mm: Layer height, stored on the result.

    Returns:
        The layer mesh.

    Raises:
        MeshError: If gmsh fails or produces no triangles.
    """
    gmsh.model.add(geom.layer)
    try:
        point_tags: dict[tuple[float, float], int] = {}
        line_tags: dict[tuple[int, int], int] = {}

        def point(xy) -> int:
            """Return the gmsh point at ``xy``, creating it if new.

            Args:
                xy: Coordinates in millimetres.

            Returns:
                The point tag.
            """
            key = (round(xy[0], _COORD_DECIMALS), round(xy[1], _COORD_DECIMALS))
            if key not in point_tags:
                point_tags[key] = gmsh.model.geo.addPoint(key[0], key[1], 0)
            return point_tags[key]

        def loop(coords) -> int | None:
            """Create a curve loop for a closed ring, reusing shared lines.

            Args:
                coords: Ring coordinates (first point repeated at the end).

            Returns:
                The curve-loop tag, or None if the ring is degenerate.
            """
            tags = [point(c) for c in coords]
            curves = []
            for a, b in zip(tags, tags[1:], strict=False):
                if a == b:
                    continue
                key = (min(a, b), max(a, b))
                if key not in line_tags:
                    line_tags[key] = gmsh.model.geo.addLine(*key)
                curves.append(line_tags[key] if a < b else -line_tags[key])
            return gmsh.model.geo.addCurveLoop(curves) if len(curves) >= 3 else None

        surfaces: dict[int, bool] = {}  # surface tag -> is copper
        face_polys = []
        for face in geom.faces:
            outer = loop(face.polygon.exterior.coords)
            if outer is None:
                continue
            holes = [h for ring in face.polygon.interiors if (h := loop(ring.coords)) is not None]
            surfaces[gmsh.model.geo.addPlaneSurface([outer, *holes])] = face.is_copper
            face_polys.append(face.polygon)

        # Anchors: embed a point in whichever face contains it (skip existing vertices).
        tree = shapely.STRtree(face_polys)
        surface_list = list(surfaces)
        embeds: dict[int, list[int]] = defaultdict(list)
        for xy in geom.anchors:
            key = (round(xy[0], _COORD_DECIMALS), round(xy[1], _COORD_DECIMALS))
            if key in point_tags:
                continue
            hits = tree.query(Point(xy), predicate="within")
            if len(hits):
                embeds[surface_list[int(hits[0])]].append(point(xy))

        gmsh.model.geo.synchronize()
        for surface, points in embeds.items():
            gmsh.model.mesh.embed(0, points, 2, surface)
        gmsh.model.mesh.generate(2)

        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        index = np.full(int(node_tags.max()) + 1 if len(node_tags) else 1, -1, dtype=np.int64)
        index[node_tags.astype(np.int64)] = np.arange(len(node_tags))
        nodes = coords.reshape(-1, 3)[:, :2].copy()

        triangles, copper = [], []
        for surface, is_copper in surfaces.items():
            types, _, elem_nodes = gmsh.model.mesh.getElements(2, surface)
            for etype, enodes in zip(types, elem_nodes, strict=True):
                if etype == _TRIANGLE:
                    tri = index[enodes.astype(np.int64)].reshape(-1, 3)
                    triangles.append(tri)
                    copper.append(np.full(len(tri), is_copper))
    except Exception as exc:  # gmsh raises bare Exceptions
        raise MeshError(f"gmsh failed on {geom.layer}: {exc}") from exc
    finally:
        gmsh.model.remove()

    if not triangles:
        raise MeshError(f"gmsh produced no triangles on {geom.layer}")
    return LayerMesh(
        layer=geom.layer,
        z_mm=z_mm,
        nodes=nodes,
        triangles=np.concatenate(triangles).astype(np.int32),
        copper=np.concatenate(copper),
    )


def write_msh(mesh: BoardMesh, path: str | Path) -> None:
    """Write all layer meshes to one gmsh ``.msh`` (format 4.1) file.

    Each layer is placed at its height ``z_mm``. Coordinates are board
    millimetres with KiCad's axes (Y down). Physical groups are named
    ``"<layer>/copper"`` and ``"<layer>/dielectric"``.

    Must be called inside an initialised gmsh session.

    Args:
        mesh: The meshes to write.
        path: Destination file; parent directories are created.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.model.add("kits_board")
    try:
        node_offset = elem_offset = 0
        for lm in mesh.layers:
            node_tags = np.arange(1, lm.num_nodes + 1) + node_offset
            coords = np.column_stack([lm.nodes, np.full(lm.num_nodes, lm.z_mm)]).ravel()
            nodes_added = False
            for region, mask in (("copper", lm.copper), ("dielectric", ~lm.copper)):
                if not mask.any():
                    continue
                entity = gmsh.model.addDiscreteEntity(2)
                if not nodes_added:
                    gmsh.model.mesh.addNodes(2, entity, node_tags, coords)
                    nodes_added = True
                tris = lm.triangles[mask]
                elem_tags = np.arange(1, len(tris) + 1) + elem_offset
                elem_offset += len(tris)
                elem_nodes = (tris + 1 + node_offset).ravel()
                gmsh.model.mesh.addElementsByType(entity, _TRIANGLE, elem_tags, elem_nodes)
                gmsh.model.addPhysicalGroup(2, [entity], name=f"{lm.layer}/{region}")
            node_offset += lm.num_nodes
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.write(str(path))
    finally:
        gmsh.model.remove()


def mesh_board(
    board: Board,
    settings: MeshSettings,
    msh_path: str | Path | None = None,
    progress: Progress | None = None,
) -> BoardMesh:
    """Extract geometry, mesh every copper layer and optionally write a ``.msh`` file.

    Args:
        board: The board to mesh.
        settings: Mesh size limits.
        msh_path: Where to write the combined gmsh mesh; None to skip writing.
        progress: Optional callback ``(message, overall_fraction)``.

    Returns:
        The board mesh.

    Raises:
        GeometryError: If the board has no usable outline.
        MeshError: If gmsh fails.
        OSError: If the board file can't be read for fingerprinting.
    """
    settings.validate()
    # Fingerprint before meshing: if the file changes mid-run, the saved mesh
    # will then (correctly) look stale rather than current.
    fingerprint = MeshFingerprint.of(board.path, settings)
    report = progress or (lambda message, fraction: None)
    n = len(board.copper_layers)
    # Progress budget: geometry 30%, meshing 65%, writing 5%.
    geometry = extract_layers(board, settings, lambda msg, f: report(msg, 0.30 * f))
    heights = layer_heights(board)

    gmsh.initialize(interruptible=False)
    try:
        _configure(settings)
        layers = []
        for i, geom in enumerate(geometry):
            report(f"Meshing {geom.layer} ({i + 1}/{n})", 0.30 + 0.65 * i / n)
            layers.append(mesh_layer(geom, settings, heights[geom.layer]))
        mesh = BoardMesh(layers, settings, Path(msh_path) if msh_path else None, fingerprint)
        if msh_path:
            report("Writing mesh file", 0.95)
            write_msh(mesh, msh_path)
    finally:
        gmsh.finalize()
    report("Done", 1.0)
    return mesh
