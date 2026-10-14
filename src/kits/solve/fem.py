"""Finite-element building blocks for the 2.5D layered model (scikit-fem + scipy).

Each copper layer is a 2D sheet meshed over the whole board outline (see
:mod:`kits.mesh`). Global unknowns are the nodes of all layer meshes, numbered
layer by layer. Everything here works in SI units (metres); mesh coordinates
arrive in millimetres and are converted by :func:`layer_fem`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import scipy.sparse as sp
import shapely
import skfem as fem
from scipy.spatial import cKDTree
from skfem.helpers import dot, grad

from ..mesh.model import LayerMesh

MM = 1e-3  # metres per millimetre
_INTORDER = 2  # quadrature order: exact for P1 mass matrices


@dataclass
class LayerFEM:
    """One layer's mesh prepared for assembly.

    Attributes:
        layer: Copper layer name.
        z: Layer height above the board bottom, in metres.
        mesh: The scikit-fem triangle mesh (coordinates in metres).
        basis: Linear (P1) basis on the mesh.
        copper: ``(M,)`` True for copper triangles.
        offset: Index of this layer's first node in the global numbering.
    """

    layer: str
    z: float
    mesh: fem.MeshTri
    basis: fem.CellBasis
    copper: np.ndarray
    offset: int

    @property
    def num_nodes(self) -> int:
        """Number of nodes (unknowns) on this layer."""
        return self.mesh.nvertices

    @property
    def nodes_mm(self) -> np.ndarray:
        """``(N, 2)`` node coordinates in millimetres."""
        return self.mesh.p.T / MM

    @cached_property
    def is_copper_node(self) -> np.ndarray:
        """``(N,)`` True for nodes that belong to at least one copper triangle."""
        mask = np.zeros(self.num_nodes, dtype=bool)
        mask[self.mesh.t[:, self.copper].ravel()] = True
        return mask

    @cached_property
    def node_tree(self) -> cKDTree:
        """Spatial index of the node coordinates (millimetres), built once per layer."""
        return cKDTree(self.nodes_mm)


def layer_fem(lm: LayerMesh, offset: int) -> LayerFEM:
    """Wrap a layer mesh for assembly.

    Args:
        lm: The layer mesh (millimetres).
        offset: Global index of the layer's first node.

    Returns:
        The prepared layer.
    """
    mesh = fem.MeshTri(np.ascontiguousarray(lm.nodes.T * MM), np.ascontiguousarray(lm.triangles.T))
    basis = fem.Basis(mesh, fem.ElementTriP1(), intorder=_INTORDER)
    return LayerFEM(lm.layer, lm.z_mm * MM, mesh, basis, lm.copper.copy(), offset)


def _per_element(basis: fem.CellBasis, values: np.ndarray) -> np.ndarray:
    """Broadcast one value per element to every quadrature point.

    Args:
        basis: The basis whose quadrature is used.
        values: ``(M,)`` values per element.

    Returns:
        ``(M, Q)`` array for use as a form coefficient.
    """
    return np.repeat(np.asarray(values, float)[:, None], basis.X.shape[1], axis=1)


@fem.BilinearForm
def _diffusion(u, v, w):
    """Weak form of ``-div(c grad u)`` with a per-element coefficient ``c``.

    Args:
        u: Trial function.
        v: Test function.
        w: Form parameters; ``w.c`` is the coefficient.

    Returns:
        The integrand.
    """
    return w.c * dot(grad(u), grad(v))


@fem.BilinearForm
def _mass(u, v, w):
    """Weighted mass matrix ``c u v`` with a per-element coefficient ``c``.

    Args:
        u: Trial function.
        v: Test function.
        w: Form parameters; ``w.c`` is the coefficient.

    Returns:
        The integrand.
    """
    return w.c * u * v


@fem.LinearForm
def _load(v, w):
    """Load vector of a per-element source density ``c``.

    Args:
        v: Test function.
        w: Form parameters; ``w.c`` is the source per unit area.

    Returns:
        The integrand.
    """
    return w.c * v


def sheet_matrix(lf: LayerFEM, coefficient: np.ndarray) -> sp.csr_matrix:
    """Stiffness of a 2D sheet with a per-element sheet conductance.

    Args:
        lf: The layer.
        coefficient: ``(M,)`` sheet conductance per element (S, or W/K for heat).

    Returns:
        The layer-local stiffness matrix.
    """
    return _diffusion.assemble(lf.basis, c=_per_element(lf.basis, coefficient)).tocsr()


def mass_matrix(lf: LayerFEM, coefficient: np.ndarray) -> sp.csr_matrix:
    """Area-weighted mass matrix, e.g. for convection to ambient.

    Args:
        lf: The layer.
        coefficient: ``(M,)`` per-element coefficient (e.g. W/m²K).

    Returns:
        The layer-local mass matrix.
    """
    return _mass.assemble(lf.basis, c=_per_element(lf.basis, coefficient)).tocsr()


def load_vector(lf: LayerFEM, density: np.ndarray) -> np.ndarray:
    """Nodal loads for a per-element source density.

    Args:
        lf: The layer.
        density: ``(M,)`` source per unit area (e.g. W/m²).

    Returns:
        ``(N,)`` layer-local load vector.
    """
    return _load.assemble(lf.basis, c=_per_element(lf.basis, density))


def element_gradients(lf: LayerFEM, values: np.ndarray) -> np.ndarray:
    """Gradient of a P1 field on each element (constant per element).

    Args:
        lf: The layer.
        values: ``(N,)`` nodal values.

    Returns:
        ``(M, 2)`` gradients in units of value per metre.
    """
    g = lf.basis.interpolate(values).grad  # (2, M, Q), constant over Q for P1
    return g[:, :, 0].T


def block_diag(matrices: list[sp.spmatrix]) -> sp.csr_matrix:
    """Place layer-local matrices on the diagonal of the global matrix.

    Args:
        matrices: One matrix per layer, in layer order.

    Returns:
        The global matrix.
    """
    return sp.block_diag(matrices, format="csr")


def locate(lf: LayerFEM, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find the triangle containing each point, and its barycentric coordinates.

    Args:
        lf: The layer to search.
        points: ``(P, 2)`` points in metres.

    Returns:
        ``(element, weights)``: element index per point (-1 if the point is
        outside the mesh, e.g. inside a hole) and ``(P, 3)`` barycentric weights.
    """
    p, t = lf.mesh.p, lf.mesh.t
    triangles = shapely.polygons(np.stack([p[:, t[0]].T, p[:, t[1]].T, p[:, t[2]].T], axis=1))
    tree = shapely.STRtree(triangles)
    hits = tree.query(shapely.points(points), predicate="intersects")
    element = np.full(len(points), -1)
    first = np.unique(hits[0], return_index=True)
    element[first[0]] = hits[1][first[1]]

    weights = np.zeros((len(points), 3))
    found = element >= 0
    a, b, c = (p[:, t[k, element[found]]].T for k in range(3))
    q = points[found]
    det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    l1 = ((q[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (q[:, 1] - a[:, 1])) / det
    l2 = ((b[:, 0] - a[:, 0]) * (q[:, 1] - a[:, 1]) - (q[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])) / det
    weights[found] = np.column_stack([1 - l1 - l2, l1, l2])
    return element, weights


def interlayer_coupling(upper: LayerFEM, lower: LayerFEM, h: float, size: int) -> sp.csr_matrix:
    """Through-thickness coupling ``∫ h (T_upper - T_lower)^2 / 2`` between two layers.

    The layers have different (non-matching) meshes, so the integral is taken
    with the upper layer's quadrature; the lower layer's field is interpolated
    at those points. Points that fall in a hole of either layer are skipped. The
    result is symmetric and conserves heat exactly.

    Args:
        upper: The layer whose quadrature is used.
        lower: The other layer.
        h: Heat transfer coefficient between the layers, W/m²K.
        size: Total number of global unknowns.

    Returns:
        The global coupling matrix.
    """
    basis = upper.basis
    points = basis.mapping.F(basis.X)  # (2, M, Q) global coordinates of quadrature points
    m, q = points.shape[1:]
    flat = points.reshape(2, -1).T
    weights = (basis.dx).reshape(-1) * h  # quadrature weight x area x h

    # Upper layer basis values at its own quadrature points.
    xi = basis.X  # reference coordinates (2, Q)
    ref = np.stack([1 - xi[0] - xi[1], xi[0], xi[1]])  # (3, Q)
    upper_rows = np.repeat(np.arange(m * q), 3)
    upper_cols = upper.offset + np.repeat(upper.mesh.t.T, q, axis=0).reshape(-1)
    upper_vals = np.tile(ref.T, (m, 1)).reshape(-1)

    element, lam = locate(lower, flat)
    keep = element >= 0
    rows = np.arange(m * q)[keep]
    lower_rows = np.repeat(rows, 3)
    lower_cols = lower.offset + lower.mesh.t[:, element[keep]].T.reshape(-1)
    lower_vals = -lam[keep].reshape(-1)

    difference = sp.csr_matrix(
        (
            np.concatenate([upper_vals, lower_vals]),
            (np.concatenate([upper_rows, lower_rows]), np.concatenate([upper_cols, lower_cols])),
        ),
        shape=(m * q, size),
    )
    w = sp.diags(np.where(np.isin(np.arange(m * q), rows), weights, 0.0))
    return (difference.T @ w @ difference).tocsr()


def ring_links(
    layers: list[LayerFEM], ring_mm: np.ndarray, copper_only: bool = True, tol_mm: float = 1e-6
) -> list[tuple[LayerFEM, LayerFEM, np.ndarray, np.ndarray]]:
    """Pair up a hole's edge nodes on consecutive layers it connects.

    Hole-edge vertices are placed at identical XY on every layer a hole cuts
    (see :mod:`kits.mesh.geometry`), so matching is exact. Layers where the hole
    edge has no copper (e.g. an inner layer with no pad) are skipped: the barrel
    then links the nearest layers above and below that do.

    Args:
        layers: Layers the hole passes through, top to bottom.
        ring_mm: ``(K, 2)`` hole-edge vertices in millimetres.
        copper_only: Only use nodes that touch copper on their layer.
        tol_mm: Matching tolerance.

    Returns:
        ``(upper, lower, upper_nodes, lower_nodes)`` per linked layer pair, with
        layer-local node indices of the vertices present on both layers.
    """
    found = []
    for lf in layers:
        distance, index = lf.node_tree.query(ring_mm)
        present = distance < tol_mm
        if copper_only:
            present &= lf.is_copper_node[index]
        if present.any():
            found.append((lf, np.where(present, index, -1)))
    links = []
    for (upper, ui), (lower, li) in zip(found, found[1:], strict=False):
        both = (ui >= 0) & (li >= 0)
        if both.any():
            links.append((upper, lower, ui[both], li[both]))
    return links


def link_matrix(pairs: list[tuple[int, int, float]], size: int) -> sp.csr_matrix:
    """Stiffness of discrete conductances between global nodes.

    Args:
        pairs: ``(node_a, node_b, conductance)`` triples (global indices).
        size: Total number of global unknowns.

    Returns:
        The global matrix (graph Laplacian of the links).
    """
    if not pairs:
        return sp.csr_matrix((size, size))
    a, b, g = (np.array(x) for x in zip(*pairs, strict=True))
    rows = np.concatenate([a, b, a, b])
    cols = np.concatenate([a, b, b, a])
    vals = np.concatenate([g, g, -g, -g])
    return sp.csr_matrix((vals, (rows, cols)), shape=(size, size))
