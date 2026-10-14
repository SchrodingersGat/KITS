"""In-memory triangle meshes of the board's copper layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import DEFAULT_MIN_CIRCLE_POINTS, MeshSettings
from .fingerprint import MeshFingerprint


@dataclass
class LayerMesh:
    """Triangle mesh of one copper layer over the whole board outline.

    Attributes:
        layer: Copper layer name.
        z_mm: Height of the layer above the board bottom, in millimetres.
        nodes: ``(N, 2)`` node coordinates in board millimetres (KiCad axes, Y down).
        triangles: ``(M, 3)`` node indices of each triangle.
        copper: ``(M,)`` True where a triangle is copper, False for bare board.
    """

    layer: str
    z_mm: float
    nodes: np.ndarray
    triangles: np.ndarray
    copper: np.ndarray

    @property
    def num_nodes(self) -> int:
        """Number of mesh nodes."""
        return len(self.nodes)

    @property
    def num_triangles(self) -> int:
        """Number of triangles."""
        return len(self.triangles)

    def edges(self) -> tuple[np.ndarray, np.ndarray]:
        """Unique triangle edges, flagged by whether they touch copper.

        Returns:
            ``(edges, is_copper)``: a ``(K, 2)`` array of node index pairs and a
            ``(K,)`` bool array, True if the edge belongs to at least one copper triangle.
        """
        tri = self.triangles
        pairs = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
        pairs.sort(axis=1)
        flags = np.tile(self.copper, 3)
        edges, inverse = np.unique(pairs, axis=0, return_inverse=True)
        is_copper = np.zeros(len(edges), dtype=bool)
        np.logical_or.at(is_copper, inverse.ravel(), flags)
        return edges, is_copper

    def triangle_areas(self) -> np.ndarray:
        """Area of each triangle in mm².

        Returns:
            ``(M,)`` array of areas.
        """
        a, b, c = (self.nodes[self.triangles[:, i]] for i in range(3))
        ab, ac = b - a, c - a
        return 0.5 * np.abs(ab[:, 0] * ac[:, 1] - ac[:, 0] * ab[:, 1])


@dataclass
class BoardMesh:
    """Meshes of all copper layers, plus the settings that produced them.

    Attributes:
        layers: Layer meshes, top to bottom.
        settings: Mesh settings used.
        msh_path: The gmsh ``.msh`` file written alongside, if any.
        fingerprint: Fingerprints of the board and settings the mesh was built from.
    """

    layers: list[LayerMesh]
    settings: MeshSettings = field(default_factory=MeshSettings)
    msh_path: Path | None = None
    fingerprint: MeshFingerprint = field(default_factory=MeshFingerprint)

    def layer(self, name: str) -> LayerMesh | None:
        """Return the mesh for a copper layer.

        Args:
            name: Copper layer name.

        Returns:
            The layer's mesh, or None if it wasn't meshed.
        """
        return next((m for m in self.layers if m.layer == name), None)

    @property
    def num_nodes(self) -> int:
        """Total nodes over all layers."""
        return sum(m.num_nodes for m in self.layers)

    @property
    def num_triangles(self) -> int:
        """Total triangles over all layers."""
        return sum(m.num_triangles for m in self.layers)

    def save_npz(self, path: str | Path) -> None:
        """Save the mesh arrays to a ``.npz`` file (no pickled objects).

        Args:
            path: Destination file.
        """
        arrays: dict[str, np.ndarray] = {
            "layers": np.array([m.layer for m in self.layers]),
            "z_mm": np.array([m.z_mm for m in self.layers]),
            "sizes_mm": np.array([self.settings.max_size_mm, self.settings.min_size_mm]),
            "min_circle_points": np.array(self.settings.min_circle_points),
            "msh_path": np.array(str(self.msh_path) if self.msh_path else ""),
            "settings_hash": np.array(self.fingerprint.settings),
            "board_hash": np.array(self.fingerprint.board),
        }
        for i, m in enumerate(self.layers):
            arrays[f"nodes_{i}"] = m.nodes
            arrays[f"triangles_{i}"] = m.triangles
            arrays[f"copper_{i}"] = m.copper
        np.savez_compressed(path, **arrays)

    @classmethod
    def load_npz(cls, path: str | Path) -> BoardMesh:
        """Load a mesh written by :meth:`save_npz`.

        Args:
            path: Source ``.npz`` file.

        Returns:
            The loaded mesh.
        """
        with np.load(path, allow_pickle=False) as data:
            layers = [
                LayerMesh(
                    layer=str(name),
                    z_mm=float(data["z_mm"][i]),
                    nodes=data[f"nodes_{i}"],
                    triangles=data[f"triangles_{i}"],
                    copper=data[f"copper_{i}"],
                )
                for i, name in enumerate(data["layers"])
            ]
            max_size, min_size = (float(v) for v in data["sizes_mm"])
            # Files from before this setting existed are stale anyway (their fingerprint won't match).
            circle_points = int(data.get("min_circle_points", DEFAULT_MIN_CIRCLE_POINTS))
            msh_path = str(data["msh_path"])
            fingerprint = MeshFingerprint(
                settings=str(data["settings_hash"]) if "settings_hash" in data else "",
                board=str(data["board_hash"]) if "board_hash" in data else "",
            )
        return cls(
            layers=layers,
            settings=MeshSettings(max_size, min_size, circle_points),
            msh_path=Path(msh_path) if msh_path else None,
            fingerprint=fingerprint,
        )


def load_saved_mesh(
    path: str | Path, board_file: str | Path, settings: MeshSettings
) -> tuple[BoardMesh | None, str | None]:
    """Load a saved mesh if it was built from the current board and settings.

    Args:
        path: The saved mesh (``.mesh.npz``).
        board_file: The current ``.kicad_pcb`` file.
        settings: The current mesh settings.

    Returns:
        ``(mesh, None)`` if the saved mesh is current; ``(None, reason)`` if it
        exists but is out of date or unreadable; ``(None, None)`` if there is no
        saved mesh.
    """
    path = Path(path)
    if not path.is_file():
        return None, None
    try:
        current = MeshFingerprint.of(board_file, settings)
        mesh = BoardMesh.load_npz(path)
    except (OSError, ValueError, KeyError) as exc:
        return None, f"The saved mesh couldn't be read ({exc})"
    reason = mesh.fingerprint.staleness(current)
    if reason is not None:
        return None, reason
    return mesh, None
