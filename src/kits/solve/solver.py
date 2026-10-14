"""Steady DC current and temperature on a 2.5D layered board model.

Model (see PLAN.md):

* **Electrical.** Each copper layer is a sheet with conductance ``σ·t``
  (copper triangles only). Vias and plated holes are barrels of thickness
  ``plating_thickness`` linking the hole-edge nodes of consecutive layers that
  have copper there. Pads are electrodes: fixed voltage, or a current
  injected over the pad area.
* **Thermal.** Each layer conducts heat in-plane through its copper plus half
  of each adjacent dielectric slab. Heat crosses between adjacent layers through
  the dielectric (``k / thickness`` per area) and through plated barrels, and
  leaves by convection from the top and bottom layers. Sources are Joule heat
  in the sheets and barrels.

Temperature dependence of copper resistivity is not modelled yet.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import shapely
from scipy.sparse.linalg import spsolve

from ..ingest.model import Board, Pad, is_copper_layer
from ..mesh.geometry import board_holes, pad_area
from ..mesh.model import BoardMesh
from .fem import (
    MM,
    LayerFEM,
    block_diag,
    element_gradients,
    interlayer_coupling,
    layer_fem,
    link_matrix,
    load_vector,
    mass_matrix,
    ring_links,
    sheet_matrix,
)

DEFAULT_COPPER_THICKNESS_MM = 0.035
# Leakage to 0 V, relative to the mean diagonal, so copper islands not connected
# to any electrode stay solvable. Small enough not to affect driven currents.
_LEAKAGE = 1e-12


class SolveError(ValueError):
    """The problem is ill-posed (e.g. no reference voltage) or an electrode can't be found."""


@dataclass
class SolveSettings:
    """Material and boundary-condition inputs.

    Attributes:
        copper_conductivity_s_m: Electrical conductivity of copper and plating.
        copper_thermal_conductivity_w_mk: Thermal conductivity of copper.
        dielectric_thermal_conductivity_w_mk: Thermal conductivity of the board material.
        plating_thickness_mm: Barrel plating thickness of vias and plated holes.
        convection_w_m2k: Heat transfer coefficient from the top and bottom surfaces.
        ambient_c: Ambient temperature.
    """

    copper_conductivity_s_m: float = 5.8e7
    copper_thermal_conductivity_w_mk: float = 390.0
    dielectric_thermal_conductivity_w_mk: float = 0.3
    plating_thickness_mm: float = 0.025
    convection_w_m2k: float = 10.0
    ambient_c: float = 25.0


@dataclass
class Electrode:
    """A pad held at a voltage, or with a current injected into the board.

    Attributes:
        ref: Footprint reference, e.g. ``"J1"``.
        pad: Pad number, e.g. ``"1"``.
        voltage: Fixed voltage, or None.
        current: Current into the board in amperes (negative for a sink), or None.
    """

    ref: str
    pad: str
    voltage: float | None = None
    current: float | None = None


@dataclass
class DCResult:
    """Solution of the DC conduction problem.

    Attributes:
        layers: Layer names, top to bottom.
        voltage: Nodal potential per layer, volts.
        current_density: Volumetric current density magnitude per triangle, A/m²
            (zero on bare-board triangles).
        sheet_power: Joule power per unit area per triangle, W/m².
        barrel_heat: Global nodal Joule power from via/plated barrels, W.
        electrode_currents: Current into the board at each electrode, A.
        electrode_voltages: Area-weighted mean potential of each electrode's pad, V.
        total_power: Total Joule power, W.
    """

    layers: list[str]
    voltage: dict[str, np.ndarray]
    current_density: dict[str, np.ndarray]
    sheet_power: dict[str, np.ndarray]
    barrel_heat: np.ndarray
    electrode_currents: dict[tuple[str, str], float]
    electrode_voltages: dict[tuple[str, str], float]
    total_power: float

    def resistance(self, a: tuple[str, str], b: tuple[str, str]) -> float:
        """Resistance between two voltage electrodes.

        Args:
            a: ``(ref, pad)`` of one electrode.
            b: ``(ref, pad)`` of the other.

        Returns:
            ``(V_a - V_b) / I_a`` in ohms, using the electrodes' solved potentials.
        """
        return (self.electrode_voltages[a] - self.electrode_voltages[b]) / self.electrode_currents[a]


@dataclass
class ThermalResult:
    """Solution of the heat problem.

    Attributes:
        temperature: Nodal temperature per layer, °C.
        convected_power: Heat leaving by convection, W (equals the input power at steady state).
    """

    temperature: dict[str, np.ndarray]
    convected_power: float

    @property
    def max_temperature(self) -> float:
        """Highest nodal temperature on any layer, °C."""
        return max(float(t.max()) for t in self.temperature.values())


class LayeredModel:
    """The board mesh assembled for solving: layer FEM data, thicknesses and barrels."""

    def __init__(self, board: Board, mesh: BoardMesh, settings: SolveSettings | None = None):
        """Prepare a board mesh for electrical and thermal solves.

        Args:
            board: The board (pads, holes, stackup).
            mesh: Its mesh; layers must be top to bottom as built by :func:`kits.mesh.mesher.mesh_board`.
            settings: Material and boundary settings; defaults if omitted.
        """
        self.board = board
        self.settings = settings or SolveSettings()
        self.layers: list[LayerFEM] = []
        offset = 0
        for lm in mesh.layers:
            self.layers.append(layer_fem(lm, offset))
            offset += lm.num_nodes
        self.size = offset
        self.by_name = {lf.layer: lf for lf in self.layers}
        self.copper_thickness = _copper_thickness(board, [lf.layer for lf in self.layers])
        self.holes = [h for h in board_holes(board, mesh.settings) if h.plated]

    # -- geometry helpers ----------------------------------------------------------

    def dielectric_gap(self, upper: LayerFEM, lower: LayerFEM) -> float:
        """Dielectric thickness between two adjacent layers, in metres.

        Args:
            upper: The upper layer.
            lower: The lower layer.

        Returns:
            Mid-plane separation minus half of each copper thickness.
        """
        half_copper = (self.copper_thickness[upper.layer] + self.copper_thickness[lower.layer]) / 2
        return max(upper.z - lower.z - half_copper, 1e-6)

    def electrode_nodes(self, electrode: Electrode) -> tuple[np.ndarray, np.ndarray]:
        """Global nodes covered by an electrode's pad, with their lumped area weights.

        Args:
            electrode: The electrode.

        Returns:
            ``(nodes, weights)``: global node indices on the pad's copper (all
            its layers) and each node's share of the pad's copper area.

        Raises:
            SolveError: If the pad doesn't exist or has no meshed copper.
        """
        pad = _find_pad(self.board, electrode.ref, electrode.pad)
        area = pad_area(pad).buffer(1e-6)
        nodes, weights = [], []
        for lf in self.layers:
            if lf.layer not in pad.layers:
                continue
            t = lf.mesh.t[:, lf.copper]
            centroids = lf.nodes_mm[t].mean(axis=0)
            inside = shapely.contains_xy(area, centroids[:, 0], centroids[:, 1])
            if not inside.any():
                continue
            tri_area = _triangle_areas(lf.nodes_mm, t[:, inside])
            lumped = np.bincount(t[:, inside].ravel(), np.repeat(tri_area / 3, 3), minlength=lf.num_nodes)
            local = np.flatnonzero(lumped)
            nodes.append(lf.offset + local)
            weights.append(lumped[local])
        if not nodes:
            raise SolveError(f"pad {electrode.ref}.{electrode.pad} has no copper in the mesh")
        weights = np.concatenate(weights)
        return np.concatenate(nodes), weights / weights.sum()

    def barrel_links(self, conductivity: float) -> list[tuple[int, int, float]]:
        """Discrete conductances of all plated barrels between their layers.

        Args:
            conductivity: Barrel material conductivity (electrical S/m or thermal W/mK).

        Returns:
            ``(node_a, node_b, conductance)`` triples in global numbering; each
            barrel segment's conductance is split evenly over its edge nodes.
        """
        plating = self.settings.plating_thickness_mm * MM
        pairs = []
        for hole in self.holes:
            layers = [self.by_name[name] for name in hole.layers if name in self.by_name]
            ring = np.array(hole.polygon.exterior.coords)[:-1]
            perimeter = hole.polygon.exterior.length * MM
            for upper, lower, ui, li in ring_links(layers, ring):
                length = max(upper.z - lower.z, 1e-6)
                g = conductivity * perimeter * plating / length / len(ui)
                pairs += [(upper.offset + a, lower.offset + b, g) for a, b in zip(ui, li, strict=True)]
        return pairs

    # -- electrical ------------------------------------------------------------------

    def sheet_conductance(self, lf: LayerFEM) -> np.ndarray:
        """Electrical sheet conductance ``σ·t`` of each triangle of a layer.

        Args:
            lf: The layer.

        Returns:
            ``(M,)`` conductance in siemens (per square); zero on bare board.
        """
        sigma_t = self.settings.copper_conductivity_s_m * self.copper_thickness[lf.layer]
        return np.where(lf.copper, sigma_t, 0.0)

    def solve_dc(self, electrodes: Sequence[Electrode]) -> DCResult:
        """Solve for potentials and currents with the given electrodes.

        Voltage electrodes are equipotential pads. Current electrodes spread their
        current over the pad's copper area. If no electrode fixes a voltage, the
        first sink (negative current) is grounded and its current follows from the
        others, which must then sum to zero.

        Args:
            electrodes: The pads driving the board.

        Returns:
            The DC solution.

        Raises:
            SolveError: If no reference voltage can be set or currents don't balance.
        """
        s = self.settings
        sheets = [sheet_matrix(lf, self.sheet_conductance(lf)) for lf in self.layers]
        links = self.barrel_links(s.copper_conductivity_s_m)
        k = (block_diag(sheets) + link_matrix(links, self.size)).tocsr()

        electrodes = list(electrodes)
        if not any(e.voltage is not None for e in electrodes):
            sinks = [e for e in electrodes if (e.current or 0) < 0]
            total = sum(e.current or 0 for e in electrodes)
            if not sinks or abs(total) > 1e-9 * max(abs(e.current or 0) for e in electrodes):
                raise SolveError("give a voltage electrode, or current sources and sinks that sum to zero")
            ground = sinks[0]
            electrodes = [Electrode(ground.ref, ground.pad, voltage=0.0)] + [
                e for e in electrodes if e is not ground
            ]

        rhs = np.zeros(self.size)
        fixed = np.zeros(self.size, dtype=bool)
        values = np.zeros(self.size)
        electrode_nodes = {}
        for e in electrodes:
            nodes, weights = self.electrode_nodes(e)
            electrode_nodes[(e.ref, e.pad)] = (nodes, weights)
            if e.voltage is not None:
                fixed[nodes] = True
                values[nodes] = e.voltage
            elif e.current is not None:
                rhs[nodes] += e.current * weights

        voltage = _solve_with_fixed(k, rhs, fixed, values)

        residual = k @ voltage - rhs  # current into the board at fixed nodes
        currents, potentials = {}, {}
        for key, (nodes, weights) in electrode_nodes.items():
            currents[key] = float(residual[nodes].sum() + rhs[nodes].sum())
            potentials[key] = float(voltage[nodes] @ weights)

        by_layer, density, sheet_power = {}, {}, {}
        total_power = 0.0
        for lf, sheet in zip(self.layers, sheets, strict=True):
            v = voltage[lf.offset : lf.offset + lf.num_nodes]
            by_layer[lf.layer] = v
            g = element_gradients(lf, v)
            e2 = (g**2).sum(axis=1)
            sigma_t = self.sheet_conductance(lf)
            density[lf.layer] = np.where(lf.copper, s.copper_conductivity_s_m * np.sqrt(e2), 0)
            sheet_power[lf.layer] = sigma_t * e2
            total_power += float(v @ (sheet @ v))
        barrel_heat = np.zeros(self.size)
        for a, b, g in links:
            p = g * (voltage[a] - voltage[b]) ** 2
            barrel_heat[a] += p / 2
            barrel_heat[b] += p / 2
        total_power += float(barrel_heat.sum())

        result = DCResult(
            layers=[lf.layer for lf in self.layers],
            voltage=by_layer,
            current_density=density,
            sheet_power=sheet_power,
            barrel_heat=barrel_heat,
            electrode_currents=currents,
            electrode_voltages=potentials,
            total_power=total_power,
        )
        return result

    # -- thermal -------------------------------------------------------------------------

    def solve_thermal(self, dc: DCResult) -> ThermalResult:
        """Solve for steady temperature with Joule heating from a DC solution.

        Args:
            dc: The DC solution supplying the heat sources.

        Returns:
            The temperature field.
        """
        s = self.settings
        k_cu, k_d = s.copper_thermal_conductivity_w_mk, s.dielectric_thermal_conductivity_w_mk
        gaps = [self.dielectric_gap(u, lo) for u, lo in zip(self.layers, self.layers[1:], strict=False)]
        sheets, loads, convection = [], [], []
        for i, lf in enumerate(self.layers):
            # In-plane conduction: own copper plus half of each adjacent dielectric slab.
            dielectric_share = k_d * ((gaps[i - 1] if i > 0 else 0) + (gaps[i] if i < len(gaps) else 0)) / 2
            conductance = np.where(lf.copper, k_cu * self.copper_thickness[lf.layer], 0) + dielectric_share
            sheets.append(sheet_matrix(lf, conductance))
            loads.append(load_vector(lf, dc.sheet_power[lf.layer]))
            outer = i == 0 or i == len(self.layers) - 1
            h = np.full(lf.mesh.nelements, s.convection_w_m2k if outer else 0.0)
            convection.append(mass_matrix(lf, h))

        k = block_diag(sheets) + block_diag(convection)
        for upper, lower, gap in zip(self.layers, self.layers[1:], gaps, strict=False):
            k = k + interlayer_coupling(upper, lower, k_d / gap, self.size)
        k = k + link_matrix(self.barrel_links(k_cu), self.size)

        conv = block_diag(convection)
        rhs = np.concatenate(loads) + dc.barrel_heat + conv @ np.full(self.size, s.ambient_c)
        temperature = spsolve(k.tocsc(), rhs)
        convected = float(np.ones(self.size) @ (conv @ (temperature - s.ambient_c)))
        by_layer = {lf.layer: temperature[lf.offset : lf.offset + lf.num_nodes] for lf in self.layers}
        return ThermalResult(temperature=by_layer, convected_power=convected)


# -- helpers ---------------------------------------------------------------------------


def _solve_with_fixed(k: sp.csr_matrix, rhs: np.ndarray, fixed: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Solve ``k x = rhs`` with some unknowns fixed, regularising floating parts.

    Nodes with no conductance at all (bare board) are fixed at 0 V; a tiny
    leakage keeps copper islands that touch no electrode solvable.

    Args:
        k: Global stiffness.
        rhs: Injected nodal currents.
        fixed: True where the value is prescribed.
        values: Prescribed values (used where ``fixed``).

    Returns:
        The full solution vector.
    """
    diagonal = k.diagonal()
    fixed = fixed | (diagonal == 0)
    leak = _LEAKAGE * diagonal[diagonal > 0].mean()
    k = k + sp.diags(np.where(fixed, 0, leak))
    free = ~fixed
    x = np.where(fixed, values, 0.0)
    k_ff = k[free][:, free].tocsc()
    k_fc = k[free][:, fixed]
    x[free] = spsolve(k_ff, rhs[free] - k_fc @ x[fixed])
    return x


def _find_pad(board: Board, ref: str, number: str) -> Pad:
    """Look up a pad by footprint reference and pad number.

    Args:
        board: The board.
        ref: Footprint reference.
        number: Pad number.

    Returns:
        The first matching pad.

    Raises:
        SolveError: If there is no such pad.
    """
    for pad in board.pads():
        if pad.footprint_ref == ref and pad.number == str(number):
            return pad
    raise SolveError(f"no pad {ref}.{number} on the board")


def _triangle_areas(nodes: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Areas of triangles given by node indices.

    Args:
        nodes: ``(N, 2)`` coordinates.
        t: ``(3, M)`` node indices.

    Returns:
        ``(M,)`` areas in the coordinates' units squared.
    """
    a, b, c = nodes[t[0]], nodes[t[1]], nodes[t[2]]
    return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))


def _copper_thickness(board: Board, layers: list[str]) -> dict[str, float]:
    """Copper thickness of each layer from the stackup, in metres.

    Args:
        board: The board.
        layers: Copper layer names.

    Returns:
        Thickness per layer; :data:`DEFAULT_COPPER_THICKNESS_MM` where the stackup has none.
    """
    stack = {s.name: s.thickness_mm for s in board.stackup if is_copper_layer(s.name) and s.thickness_mm}
    return {name: stack.get(name, DEFAULT_COPPER_THICKNESS_MM) * MM for name in layers}
