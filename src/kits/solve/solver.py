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
  in the sheets and barrels, plus user heat loads on pads: injected power,
  fixed temperature, or a thermal resistance to ambient (each spread over the
  pad's copper area on all its layers).

Temperature dependence of copper resistivity is not modelled yet.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import shapely
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve
from shapely.geometry.base import BaseGeometry

from ..config import HeatLoad, HeatLoadKind, ResistiveElement, SolveSettings
from ..ingest.model import Board
from ..mesh.geometry import board_holes, footprint_body, mask_openings, pad_area
from ..mesh.mesher import layer_heights
from ..mesh.model import BoardMesh
from .elements import default_ambient_conductance
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

# KiCad's default solder mask thickness, when the stackup doesn't give one.
DEFAULT_MASK_THICKNESS_MM = 0.01

# Leakage to 0 V, relative to the mean diagonal, so copper islands not connected
# to any electrode stay solvable. Small enough not to affect driven currents.
_LEAKAGE = 1e-12


class SolveError(ValueError):
    """The problem is ill-posed (e.g. no reference voltage) or an electrode can't be found."""


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
        total_power: Total Joule power, W, including resistive elements.
        element_currents: Current through each resistive element from group A to
            group B, A, keyed by element name.
        element_power: ``I²R`` of each resistive element, W, keyed by element name.
    """

    layers: list[str]
    voltage: dict[str, np.ndarray]
    current_density: dict[str, np.ndarray]
    sheet_power: dict[str, np.ndarray]
    barrel_heat: np.ndarray
    electrode_currents: dict[tuple[str, str], float]
    electrode_voltages: dict[tuple[str, str], float]
    total_power: float
    element_currents: dict[str, float] = field(default_factory=dict)
    element_power: dict[str, float] = field(default_factory=dict)

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
        convected_power: Heat leaving the board surfaces by convection, W. At
            steady state, it plus the heat resistive elements shed to ambient
            equals the Joule power plus the net heat of all heat loads.
        load_heat: Heat into the board at each heat load's pad, W (negative where
            a sink takes heat out), keyed ``(ref, pad)``.
        load_temperatures: Area-weighted mean temperature of each heat load's pad, °C.
        element_temperatures: Body temperature of each resistive element, °C (its
            pads' mean temperature when the board contact is ideal), keyed by name.
        element_ambient_heat: Heat each resistive element sheds straight to ambient, W.
    """

    temperature: dict[str, np.ndarray]
    convected_power: float
    load_heat: dict[tuple[str, str], float] = field(default_factory=dict)
    load_temperatures: dict[tuple[str, str], float] = field(default_factory=dict)
    element_temperatures: dict[str, float] = field(default_factory=dict)
    element_ambient_heat: dict[str, float] = field(default_factory=dict)

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
        heights = layer_heights(board)
        for lm in mesh.layers:
            # Heights from the board's stackup (which may have been edited since
            # meshing), so thickness changes need a new solve but not a new mesh.
            lf = layer_fem(lm, offset)
            self.layers.append(dataclasses.replace(lf, z=heights.get(lm.layer, lm.z_mm) * MM))
            offset += lm.num_nodes
        self.size = offset
        self.by_name = {lf.layer: lf for lf in self.layers}
        self.copper_thickness = {name: t * MM for name, (t, _) in board.copper_thicknesses().items()}
        self.holes = [h for h in board_holes(board, mesh.settings) if h.plated]
        # Solder mask openings on the outer layers ({layer: area}); no entry = no mask.
        self.mask_openings: dict[str, BaseGeometry] = {}
        for side, lf in (("F", self.layers[0]), ("B", self.layers[-1])):
            openings = mask_openings(board, side)
            if openings is not None and lf.layer == f"{side}.Cu":
                shapely.prepare(openings)
                self.mask_openings[lf.layer] = openings

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
            See :meth:`pad_nodes`.

        Raises:
            SolveError: If the pad doesn't exist or has no meshed copper.
        """
        return self.pad_nodes(electrode.ref, electrode.pad)

    def pad_nodes(self, ref: str, number: str | Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """Global nodes covered by a pad (or several), with their lumped area weights.

        Args:
            ref: Footprint reference.
            number: Pad number, or several (a pad group); every pad of the
                footprint with one of them is included.

        Returns:
            ``(nodes, weights)``: global node indices on the pads' copper (all
            their layers) and each node's share of the pads' copper area.

        Raises:
            SolveError: If a pad doesn't exist or none has meshed copper.
        """
        numbers = [number] if isinstance(number, str) else list(number)
        label = f"{ref}.{','.join(numbers)}"
        pads = []
        for n in numbers:
            found = [p for p in self.board.find_pads(ref, n) if p.layers]
            if not found:
                raise SolveError(f"no pad {ref}.{n} with copper on the board")
            pads += found
        areas = {
            layer: shapely.union_all([pad_area(p) for p in pads if layer in p.layers])
            for layer in {layer for p in pads for layer in p.layers}
        }
        # Components only touch a pad where the solder mask is open.
        exposed = {
            layer: area.intersection(self.mask_openings[layer]) if layer in self.mask_openings else area
            for layer, area in areas.items()
        }
        found = self.area_nodes(exposed, copper_only=True)
        if found is None:
            if self.area_nodes(areas, copper_only=True) is not None:
                raise SolveError(
                    f"pad {label} is covered by solder mask (no mask opening), so nothing can connect to it"
                )
            raise SolveError(f"pad {label} has no copper in the mesh")
        return found

    def body_nodes(self, element: ResistiveElement) -> tuple[np.ndarray, np.ndarray]:
        """Global nodes under a resistive element's body, with their lumped area weights.

        The body is the footprint's plan area (:func:`kits.mesh.geometry.footprint_body`)
        on the outer copper layer of the footprint's side, copper and bare board
        alike. If no mesh lies under it, the element's pads are used instead.

        Args:
            element: The element.

        Returns:
            ``(nodes, weights)``, the weights summing to 1.

        Raises:
            SolveError: If the footprint or its pads can't be found.
        """
        footprint = next((fp for fp in self.board.footprints if fp.reference == element.ref), None)
        if footprint is None:
            raise SolveError(f"no footprint {element.ref} on the board (resistive element {element.name})")
        area, _source = footprint_body(footprint)
        side = footprint.side if footprint.side in self.by_name else self.layers[0].layer
        found = self.area_nodes({side: area}, copper_only=False) if not area.is_empty else None
        return found if found is not None else self.pad_nodes(element.ref, element.all_pads)

    def area_nodes(
        self, areas: dict[str, BaseGeometry], copper_only: bool
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Global nodes of the triangles inside some areas, with lumped area weights.

        Args:
            areas: ``{layer: area}`` in board millimetres.
            copper_only: Only count copper triangles.

        Returns:
            ``(nodes, weights)`` with the weights summing to 1, or None if no
            triangle's centroid falls inside the areas.
        """
        nodes, weights = [], []
        for lf in self.layers:
            area = areas.get(lf.layer)
            if area is None or area.is_empty:
                continue
            area = area.buffer(1e-6)
            t = lf.mesh.t[:, lf.copper] if copper_only else lf.mesh.t
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
            return None
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

    def solve_dc(
        self, electrodes: Sequence[Electrode], elements: Sequence[ResistiveElement] = ()
    ) -> DCResult:
        """Solve for potentials and currents with the given electrodes.

        Voltage electrodes are equipotential pads. Current electrodes spread their
        current over the pad's copper area. If no electrode fixes a voltage, the
        first sink (negative current) is grounded and its current follows from the
        others, which must then sum to zero.

        A resistive element's pad groups are its terminals: each group is
        equipotential (its pads are soldered to one lead frame or end cap, so its
        nodes are merged into one unknown), and a conductance ``1/R`` joins the two.

        Args:
            electrodes: The pads driving the board.
            elements: Lumped resistors between pad groups.

        Returns:
            The DC solution.

        Raises:
            SolveError: If no reference voltage can be set, currents don't balance,
                or a pad can't be found.
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

        # Merge each element terminal's nodes into one unknown: x = merge @ y.
        terminals = {e.name: tuple(self.pad_nodes(e.ref, g.pads)[0] for g in e.groups) for e in elements}
        merge = _merge_matrix(self.size, [nodes for pair in terminals.values() for nodes in pair])
        label = merge.indices  # reduced unknown of each node (one entry per row)
        size = merge.shape[1]
        k_r = merge.T @ k @ merge
        for e in elements:
            a, b = (label[nodes[0]] for nodes in terminals[e.name])
            k_r = k_r + _rank_one(np.array([a, b]), np.array([1.0, -1.0]), 1.0 / e.resistance_ohm, size)
        k_r = k_r.tocsr()
        rhs_r = merge.T @ rhs
        fixed_r = np.zeros(size, dtype=bool)
        fixed_r[label[fixed]] = True
        values_r = np.zeros(size)
        values_r[label[fixed]] = values[fixed]
        reduced_nodes = {key: (np.unique(label[nodes]), None) for key, (nodes, _) in electrode_nodes.items()}

        _check_connectivity(k_r, rhs_r, fixed_r, reduced_nodes)
        y = _solve_with_fixed(k_r, rhs_r, fixed_r, values_r)
        voltage = y[label]

        residual = k_r @ y - rhs_r  # current into the board at fixed unknowns
        currents, potentials = {}, {}
        for key, (nodes, weights) in electrode_nodes.items():
            currents[key] = float(residual[reduced_nodes[key][0]].sum() + rhs[nodes].sum())
            potentials[key] = float(voltage[nodes] @ weights)

        element_currents, element_power = {}, {}
        for e in elements:
            a, b = (label[nodes[0]] for nodes in terminals[e.name])
            current = float(y[a] - y[b]) / e.resistance_ohm
            element_currents[e.name] = current
            element_power[e.name] = current**2 * e.resistance_ohm

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
        total_power += sum(element_power.values())

        result = DCResult(
            layers=[lf.layer for lf in self.layers],
            voltage=by_layer,
            current_density=density,
            sheet_power=sheet_power,
            barrel_heat=barrel_heat,
            electrode_currents=currents,
            electrode_voltages=potentials,
            total_power=total_power,
            element_currents=element_currents,
            element_power=element_power,
        )
        return result

    # -- thermal -------------------------------------------------------------------------

    def mask_thickness_mm(self, side: str) -> float:
        """Solder mask thickness on one side, from the stackup.

        Args:
            side: ``"F"`` or ``"B"``.

        Returns:
            Millimetres; :data:`DEFAULT_MASK_THICKNESS_MM` if the stackup doesn't say.
        """
        listed = next((s.thickness_mm for s in self.board.stackup if s.name == f"{side}.Mask"), None)
        return listed or DEFAULT_MASK_THICKNESS_MM

    def masked_triangles(self, lf: LayerFEM) -> np.ndarray:
        """Which triangles of an outer layer the solder mask covers.

        Args:
            lf: The layer.

        Returns:
            ``(M,)`` True where covered (all False if the layer has no mask).
        """
        openings = self.mask_openings.get(lf.layer)
        if openings is None:
            return np.zeros(lf.mesh.nelements, dtype=bool)
        centroids = lf.nodes_mm[lf.mesh.t].mean(axis=0)
        return ~shapely.contains_xy(openings, centroids[:, 0], centroids[:, 1])

    def _surface_coefficient(self, lf: LayerFEM, h: float, top: bool, bottom: bool) -> np.ndarray:
        """Convection coefficient per triangle of a layer, through the solder mask where it covers.

        The mask adds a series resistance ``t / k`` per unit area:
        ``h_eff = 1 / (1/h + t/k)``. Exposed copper and openings convect with ``h``.

        Args:
            lf: The layer.
            h: Convection coefficient of the face(s) this layer forms, W/m²K (0 if none).
            top: Whether the layer is the top face.
            bottom: Whether the layer is the bottom face.

        Returns:
            ``(M,)`` W/m²K.
        """
        coefficient = np.full(lf.mesh.nelements, h)
        if h <= 0 or lf.layer not in self.mask_openings:
            return coefficient
        side = "F" if top and lf.layer == "F.Cu" else "B" if bottom and lf.layer == "B.Cu" else None
        if side is None:
            return coefficient
        resistance = self.mask_thickness_mm(side) * MM / self.settings.solder_mask_thermal_conductivity_w_mk
        coefficient[self.masked_triangles(lf)] = 1.0 / (1.0 / h + resistance)
        return coefficient

    def solve_thermal(
        self,
        dc: DCResult | None,
        heat_loads: Sequence[HeatLoad] = (),
        elements: Sequence[ResistiveElement] = (),
    ) -> ThermalResult:
        """Solve for steady temperature with Joule heating and pad heat loads.

        Heat loads act over their pad's copper (all of its layers), shared by
        area: power is injected; a fixed temperature holds every node of the pad;
        a thermal resistance ``R`` ties the pad to ambient with a total
        conductance ``1/R``.

        A resistive element's ``I²R`` (from ``dc``) heats its body, whose plan
        area (Fab outline, courtyard or pads' bounding box) covers part of the
        board on the footprint's side. With ideal board contact the heat goes
        straight into the board under the body, shared by area; otherwise the
        body is a node joined to that area's mean temperature through the
        thermal resistance to the board. The body also loses heat to ambient
        through its thermal resistance to ambient, or by default by convection
        from its plan area.

        Args:
            dc: The DC solution supplying Joule heat, or None for no current flow.
            heat_loads: Heat sources and sinks on pads.
            elements: Resistive elements (the same ones passed to :meth:`solve_dc`).

        Returns:
            The temperature field, with each heat load's heat flow and pad
            temperature and each element's body temperature.

        Raises:
            SolveError: If a heat load's or element's pad doesn't exist or has no meshed copper.
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
            if dc is not None:
                loads.append(load_vector(lf, dc.sheet_power[lf.layer]))
            h_layer = (s.convection_top_w_m2k if i == 0 else 0.0) + (
                s.convection_bottom_w_m2k if i == len(self.layers) - 1 else 0.0
            )
            h = self._surface_coefficient(lf, h_layer, i == 0, i == len(self.layers) - 1)
            convection.append(mass_matrix(lf, h))

        k = block_diag(sheets) + block_diag(convection)
        for upper, lower, gap in zip(self.layers, self.layers[1:], gaps, strict=False):
            k = k + interlayer_coupling(upper, lower, k_d / gap, self.size)
        k = k + link_matrix(self.barrel_links(k_cu), self.size)

        conv = block_diag(convection)
        rhs = conv @ np.full(self.size, s.ambient_c)
        if dc is not None:
            rhs = rhs + np.concatenate(loads) + dc.barrel_heat

        # Pad heat loads.
        fixed = np.zeros(self.size, dtype=bool)
        values = np.zeros(self.size)
        tie = np.zeros(self.size)  # lumped conductance to ambient from thermal resistances, W/K
        load_nodes = {}
        for load in heat_loads:
            nodes, weights = self.pad_nodes(load.ref, load.pad)
            load_nodes[load.key] = (load, nodes, weights)
            if load.kind is HeatLoadKind.POWER:
                rhs[nodes] += load.value * weights
            elif load.kind is HeatLoadKind.TEMPERATURE:
                fixed[nodes] = True
                values[nodes] = load.value
            else:
                tie[nodes] += weights / load.value
        k = k + sp.diags(tie)
        rhs = rhs + tie * s.ambient_c

        # Resistive elements: the body takes I²R and is tied to its pads (directly,
        # or through a junction node — an extra unknown — when a thermal resistance
        # to the board is given) and to ambient.
        junctions = [e for e in elements if e.thermal_resistance_board_k_w is not None]
        n = self.size + len(junctions)
        if junctions:
            k = sp.block_diag((k, sp.csr_matrix((len(junctions), len(junctions)))))
            extra = np.zeros(len(junctions))
            rhs, values, fixed = (
                np.concatenate([rhs, extra]),
                np.concatenate([values, extra]),
                np.append(fixed, extra.astype(bool)),
            )
        junction_index = {e.name: self.size + i for i, e in enumerate(junctions)}
        bodies = {}
        for e in elements:
            nodes, weights = self.body_nodes(e)
            if e.name in junction_index:
                j = junction_index[e.name]
                to_board = np.concatenate([[1.0], -weights])
                k = k + _rank_one(
                    np.concatenate([[j], nodes]), to_board, 1.0 / e.thermal_resistance_board_k_w, n
                )
                nodes, weights = np.array([j]), np.array([1.0])
            if e.thermal_resistance_ambient_k_w is not None:
                g_ambient = 1.0 / e.thermal_resistance_ambient_k_w
            else:
                footprint = next(fp for fp in self.board.footprints if fp.reference == e.ref)
                g_ambient = default_ambient_conductance(footprint, s)
            power = dc.element_power.get(e.name, 0.0) if dc is not None else 0.0
            rhs[nodes] += (power + g_ambient * s.ambient_c) * weights
            k = k + _rank_one(nodes, weights, g_ambient, n)
            bodies[e.name] = (nodes, weights, g_ambient)
        k = k.tocsr()

        temperature = _solve_partitioned(k, rhs, fixed, values) if fixed.any() else spsolve(k.tocsc(), rhs)
        convected = float(np.ones(self.size) @ (conv @ (temperature[: self.size] - s.ambient_c)))
        injected = k @ temperature - rhs  # heat the fixed-temperature constraints put in
        load_heat, load_temperatures = {}, {}
        for key, (load, nodes, weights) in load_nodes.items():
            if load.kind is HeatLoadKind.POWER:
                load_heat[key] = load.value
            elif load.kind is HeatLoadKind.TEMPERATURE:
                load_heat[key] = float(injected[nodes].sum())
            else:
                load_heat[key] = -float(weights @ (temperature[nodes] - s.ambient_c)) / load.value
            load_temperatures[key] = float(weights @ temperature[nodes])
        element_temperatures, element_ambient_heat = {}, {}
        for name, (nodes, weights, g_ambient) in bodies.items():
            element_temperatures[name] = float(weights @ temperature[nodes])
            element_ambient_heat[name] = g_ambient * (element_temperatures[name] - s.ambient_c)
        by_layer = {lf.layer: temperature[lf.offset : lf.offset + lf.num_nodes] for lf in self.layers}
        return ThermalResult(
            temperature=by_layer,
            convected_power=convected,
            load_heat=load_heat,
            load_temperatures=load_temperatures,
            element_temperatures=element_temperatures,
            element_ambient_heat=element_ambient_heat,
        )


# -- helpers ---------------------------------------------------------------------------


def _check_connectivity(
    k: sp.csr_matrix,
    rhs: np.ndarray,
    fixed: np.ndarray,
    electrode_nodes: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
) -> None:
    """Reject current that has no copper path to a fixed-voltage pad.

    Each piece of connected copper (layers joined by barrels) that receives
    current must either contain a fixed-voltage electrode or have its injected
    currents sum to zero; otherwise the current has nowhere to go.

    Args:
        k: Global conductance matrix (without regularisation).
        rhs: Injected nodal currents.
        fixed: True at fixed-voltage nodes.
        electrode_nodes: Nodes of each electrode, keyed ``(ref, pad)``.

    Raises:
        SolveError: If some current can't reach a sink.
    """
    _, component = connected_components(k, directed=False)
    injected = np.bincount(component, weights=rhs)
    grounded = np.zeros(len(injected), dtype=bool)
    grounded[component[fixed]] = True
    scale = max(np.abs(rhs).sum(), 1e-30)
    for c in np.flatnonzero(~grounded & (np.abs(injected) > 1e-9 * scale)):
        pads = sorted(
            f"{ref}.{pad}"
            for (ref, pad), (nodes, _) in electrode_nodes.items()
            if (component[nodes] == c).any()
        )
        raise SolveError(
            f"no copper path balances the current at {', '.join(pads)} "
            f"({injected[c]:+.4g} A would have nowhere to go); check the sources are on connected copper"
        )


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
    return _solve_partitioned(k.tocsr(), rhs, fixed, values)


def _merge_matrix(size: int, groups: list[np.ndarray]) -> sp.csr_matrix:
    """Map each unknown to a reduced one, merging every group of nodes into one.

    Groups that share a node are merged together.

    Args:
        size: Number of unknowns.
        groups: Node index arrays to make equipotential.

    Returns:
        ``(size, reduced)`` 0/1 matrix with one entry per row: ``x = M @ y``.
    """
    rows = [np.repeat(g[0], len(g)) for g in groups if len(g)]
    cols = [g for g in groups if len(g)]
    if rows:
        rows, cols = np.concatenate(rows), np.concatenate(cols)
    else:
        rows = cols = np.zeros(0, dtype=int)
    graph = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(size, size))
    count, label = connected_components(graph, directed=False)
    return sp.csr_matrix((np.ones(size), (np.arange(size), label)), shape=(size, count))


def _rank_one(nodes: np.ndarray, coefficients: np.ndarray, conductance: float, size: int) -> sp.coo_matrix:
    """A lumped conductance acting on a weighted combination of unknowns.

    ``conductance · c cᵀ``: the flow is ``conductance · (c · x)`` and it is
    distributed back over the same unknowns by ``c``.

    Args:
        nodes: Global unknown indices.
        coefficients: The weights ``c`` (e.g. +area shares on one side, -on the other).
        conductance: Siemens or W/K.
        size: Size of the global system.

    Returns:
        A ``(size, size)`` sparse matrix.
    """
    rows = np.repeat(nodes, len(nodes))
    cols = np.tile(nodes, len(nodes))
    data = conductance * np.outer(coefficients, coefficients).ravel()
    return sp.coo_matrix((data, (rows, cols)), shape=(size, size))


def _solve_partitioned(
    k: sp.csr_matrix, rhs: np.ndarray, fixed: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """Solve ``k x = rhs`` for the free unknowns, with the others prescribed.

    Args:
        k: Global stiffness (non-singular on the free unknowns).
        rhs: Nodal loads.
        fixed: True where the value is prescribed.
        values: Prescribed values (used where ``fixed``).

    Returns:
        The full solution vector.
    """
    free = ~fixed
    x = np.where(fixed, values, 0.0)
    k_ff = k[free][:, free].tocsc()
    k_fc = k[free][:, fixed]
    x[free] = spsolve(k_ff, rhs[free] - k_fc @ x[fixed])
    return x


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
