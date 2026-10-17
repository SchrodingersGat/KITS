"""Display quantities of solve results: values per triangle, colour scales and point lookup.

Qt-free, so the GUI (:mod:`kits.gui.field_display`) and the HTML report
(:mod:`kits.report`) share them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import shapely
from scipy.spatial import cKDTree

from ..solve.results import LayerResult, SolveResults

# Perceptually uniform colour maps (matplotlib's), sampled at 9 anchors.
_VIRIDIS = [
    (68, 1, 84),
    (71, 44, 122),
    (59, 81, 139),
    (44, 113, 142),
    (33, 144, 141),
    (39, 173, 129),
    (92, 200, 99),
    (170, 220, 50),
    (253, 231, 37),
]
_INFERNO = [
    (0, 0, 4),
    (31, 12, 72),
    (85, 15, 109),
    (136, 34, 106),
    (186, 54, 85),
    (227, 89, 51),
    (249, 140, 10),
    (249, 201, 50),
    (252, 255, 164),
]
COLOR_STEPS = 64  # distinct colours in the overlay
# Current density spikes at sharp inside corners (mesh-dependent), so its scale
# tops out at this percentile instead of the maximum.
CURRENT_DENSITY_PERCENTILE = 99.5
# Log scale: the bottom of the scale is this percentile of the copper values...
LOG_LOW_PERCENTILE = 1.0
# ...but never more than this many decades below the top.
LOG_MAX_DECADES = 3.0


@dataclass(frozen=True)
class _Mode:
    """How one quantity is shown.

    Attributes:
        label: Name shown in the GUI.
        unit: Display unit.
        anchors: Colour map anchor colours, low to high.
    """

    label: str
    unit: str
    anchors: tuple


class DisplayMode(Enum):
    """Quantity shown by the results overlay."""

    CURRENT_DENSITY = _Mode("Current density", "A/mm²", tuple(_INFERNO))
    VOLTAGE = _Mode("Voltage", "V", tuple(_VIRIDIS))
    TEMPERATURE = _Mode("Temperature", "°C", tuple(_INFERNO))

    @property
    def label(self) -> str:
        """Name shown in the GUI."""
        return self.value.label

    @property
    def unit(self) -> str:
        """Display unit."""
        return self.value.unit


def triangle_values(layer: LayerResult, mode: DisplayMode) -> tuple[np.ndarray, np.ndarray]:
    """Value of the displayed quantity on each triangle of a layer.

    Nodal fields (voltage, temperature) are averaged over each triangle's corners.
    Current density and voltage are only meaningful on copper; temperature
    covers the whole board.

    Args:
        layer: The layer's results.
        mode: Quantity to show.

    Returns:
        ``(values, shown)``: ``(M,)`` values in display units and a mask of the
        triangles to draw.
    """
    if mode is DisplayMode.CURRENT_DENSITY:
        return layer.current_density / 1e6, layer.copper  # A/m² -> A/mm²
    nodal = layer.voltage if mode is DisplayMode.VOLTAGE else layer.temperature
    values = nodal[layer.triangles].mean(axis=1)
    shown = layer.copper if mode is DisplayMode.VOLTAGE else np.ones(len(values), dtype=bool)
    return values, shown


def supports_log(mode: DisplayMode) -> bool:
    """Whether a log colour scale is offered for a quantity.

    Only current density: it spans decades (a hot track next to a lightly loaded
    pour) and is never negative. Voltage is zero at the reference pad and
    temperature varies little relative to its level, so log scales don't help.

    Args:
        mode: The displayed quantity.

    Returns:
        True for current density.
    """
    return mode is DisplayMode.CURRENT_DENSITY


def value_range(results: SolveResults, mode: DisplayMode, log: bool = False) -> tuple[float, float]:
    """Colour scale limits shared by all layers, so layers can be compared.

    For temperature, the element body temperatures are included too.

    Args:
        results: The solve results.
        mode: Quantity shown.
        log: Log scale (current density only, see :func:`supports_log`): the low
            limit is then positive, at :data:`LOG_LOW_PERCENTILE` of the values but
            no more than :data:`LOG_MAX_DECADES` below the high limit.

    Returns:
        ``(low, high)`` in display units; ``high > low`` (and ``low > 0`` for log).
    """
    parts = [values[shown] for values, shown in (triangle_values(r, mode) for r in results.layers)]
    if mode is DisplayMode.TEMPERATURE and results.element_temperatures:
        # Element bodies are drawn on the temperature map, and are often its hottest parts.
        parts.append(np.array(list(results.element_temperatures.values()), float))
    values = np.concatenate(parts) if parts else np.zeros(1)
    if values.size == 0:
        values = np.zeros(1)
    if mode is DisplayMode.CURRENT_DENSITY:
        low, high = 0.0, float(np.percentile(values, CURRENT_DENSITY_PERCENTILE))
        if log and supports_log(mode):
            high = max(high, 1e-12)
            positive = values[values > 0]
            floor = high * 10**-LOG_MAX_DECADES
            low = max(float(np.percentile(positive, LOG_LOW_PERCENTILE)) if positive.size else floor, floor)
            if not high > low:
                low = high / 10
            return low, high
    else:
        low, high = float(values.min()), float(values.max())
    if not high > low:
        high = low + max(abs(low) * 1e-6, 1e-12)
    return low, high


def color_indices(
    values: np.ndarray, low: float, high: float, steps: int = COLOR_STEPS, log: bool = False
) -> np.ndarray:
    """Map values to colour map indices, clamping outside the range.

    Args:
        values: Values in display units.
        low: Value mapped to the first colour (positive for a log scale).
        high: Value mapped to the last colour.
        steps: Number of colours.
        log: Space colours evenly in ``log10(value)``; values at or below zero get
            the first colour.

    Returns:
        Integer indices in ``[0, steps)``.
    """
    if log:
        with np.errstate(divide="ignore", invalid="ignore"):
            logs = np.log10(np.maximum(values, low))
        t = np.clip((logs - np.log10(low)) / (np.log10(high) - np.log10(low)), 0.0, 1.0)
    else:
        t = np.clip((values - low) / (high - low), 0.0, 1.0)
    return np.minimum((t * steps).astype(int), steps - 1)


def scale_midpoint(low: float, high: float, log: bool) -> float:
    """The value at the middle of a colour scale.

    Args:
        low: Bottom of the scale.
        high: Top of the scale.
        log: Whether the scale is logarithmic (then the geometric mean).

    Returns:
        The midpoint value.
    """
    return float(np.sqrt(low * high)) if log else (low + high) / 2


def format_value(value: float, mode: DisplayMode) -> str:
    """Format a value for the legend, with an SI prefix for voltages.

    Args:
        value: Value in display units.
        mode: Quantity, which sets the unit.

    Returns:
        e.g. ``"12.3 mV"``, ``"151 A/mm²"`` or ``"34.4 °C"``.
    """
    if mode is DisplayMode.VOLTAGE:
        magnitude = abs(value)
        for scale, prefix in ((1, ""), (1e-3, "m"), (1e-6, "µ")):
            if magnitude >= scale or scale == 1e-6:
                return f"{value / scale:.3g} {prefix}V"
    return f"{value:.3g} {mode.unit}"


def format_resistance(ohms: float) -> str:
    """Format a resistance with an SI prefix.

    Args:
        ohms: Resistance in ohms.

    Returns:
        e.g. ``"10 mΩ"``, ``"4.7 kΩ"`` or ``"500 µΩ"``.
    """
    for scale, prefix in ((1e6, "M"), (1e3, "k"), (1, ""), (1e-3, "m"), (1e-6, "µ")):
        if abs(ohms) >= scale or scale == 1e-6:
            return f"{ohms / scale:.3g} {prefix}Ω"
    return f"{ohms:.3g} Ω"  # pragma: no cover - the loop always returns


def locate_points(
    nodes: np.ndarray, triangles: np.ndarray, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Find the triangle containing each of many points, with barycentric weights.

    For sampling a whole layer at once (e.g. on a pixel grid); :class:`FieldProbe`
    is quicker for a single point.

    Args:
        nodes: ``(N, 2)`` node coordinates.
        triangles: ``(M, 3)`` node indices.
        points: ``(P, 2)`` points, same units as ``nodes``.

    Returns:
        ``(triangle, weights)``: index per point (-1 off the mesh, e.g. in a hole)
        and ``(P, 3)`` barycentric weights (zero where not found).
    """
    return TriangleLocator(nodes, triangles).locate(points)


class TriangleLocator:
    """A spatial index of a mesh's triangles, for locating many points (built once, reused)."""

    def __init__(self, nodes: np.ndarray, triangles: np.ndarray):
        """Index a mesh.

        Args:
            nodes: ``(N, 2)`` node coordinates.
            triangles: ``(M, 3)`` node indices.
        """
        self._corners = nodes[triangles]  # (M, 3, 2)
        self._tree = shapely.STRtree(shapely.polygons(self._corners))

    def locate(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Find the triangle containing each point, with barycentric weights.

        Args:
            points: ``(P, 2)`` points, same units as the nodes.

        Returns:
            ``(triangle, weights)``: index per point (-1 off the mesh, e.g. in a hole)
            and ``(P, 3)`` barycentric weights (zero where not found).
        """
        corners = self._corners
        hits = self._tree.query(shapely.points(points), predicate="intersects")
        triangle = np.full(len(points), -1)
        first = np.unique(hits[0], return_index=True)
        triangle[first[0]] = hits[1][first[1]]
        weights = np.zeros((len(points), 3))
        found = triangle >= 0
        a, b, c = (corners[triangle[found], k] for k in range(3))
        q = points[found]
        det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
        l1 = ((q[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (q[:, 1] - a[:, 1])) / det
        l2 = ((b[:, 0] - a[:, 0]) * (q[:, 1] - a[:, 1]) - (q[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])) / det
        weights[found] = np.column_stack([1 - l1 - l2, l1, l2])
        return triangle, weights


def sample_layer(layer: LayerResult, points: np.ndarray) -> dict[DisplayMode, np.ndarray]:
    """Every displayed quantity at many points of one layer, as the GUI shows them.

    Voltage and temperature are interpolated; current density is constant per
    triangle. Current density and voltage are only defined on copper.

    Args:
        layer: The layer's results.
        points: ``(P, 2)`` points in board millimetres.

    Returns:
        ``{mode: (P,) values}`` in display units, NaN where the quantity isn't shown.
    """
    triangle, weights = locate_points(layer.nodes, layer.triangles, points)
    found = triangle >= 0
    on_copper = np.zeros(len(points), dtype=bool)
    on_copper[found] = layer.copper[triangle[found]]
    corner_nodes = layer.triangles[triangle[found]]
    out = {}
    for mode, nodal, where in (
        (DisplayMode.TEMPERATURE, layer.temperature, found),
        (DisplayMode.VOLTAGE, layer.voltage, on_copper),
    ):
        values = np.full(len(points), np.nan)
        interpolated = np.zeros(len(points))
        interpolated[found] = (weights[found] * nodal[corner_nodes]).sum(axis=1)
        values[where] = interpolated[where]
        out[mode] = values
    density = np.full(len(points), np.nan)
    density[on_copper] = layer.current_density[triangle[on_copper]] / 1e6  # A/m² -> A/mm²
    out[DisplayMode.CURRENT_DENSITY] = density
    return out


def triangle_gradients(layer: LayerResult, nodal: np.ndarray) -> np.ndarray:
    """Gradient of a nodal (linear) field on each triangle, constant within it.

    Args:
        layer: The layer's mesh.
        nodal: ``(N,)`` values at its nodes.

    Returns:
        ``(M, 2)`` gradients, value per millimetre, in board axes.
    """
    p = layer.nodes[layer.triangles]  # (M, 3, 2)
    f = nodal[layer.triangles]  # (M, 3)
    dx1, dy1 = p[:, 1, 0] - p[:, 0, 0], p[:, 1, 1] - p[:, 0, 1]
    dx2, dy2 = p[:, 2, 0] - p[:, 0, 0], p[:, 2, 1] - p[:, 0, 1]
    df1, df2 = f[:, 1] - f[:, 0], f[:, 2] - f[:, 0]
    det = dx1 * dy2 - dx2 * dy1
    det = np.where(det == 0, np.inf, det)  # degenerate: no gradient
    return np.column_stack([(df1 * dy2 - df2 * dy1) / det, (df2 * dx1 - df1 * dx2) / det])


class VectorSampler:
    """Flow arrows for one layer on a grid of any spacing and extent.

    Current density and voltage show where current flows, along ``-grad V``
    with the current density's magnitude (on copper only). Temperature shows
    where heat flows, along ``-grad T`` with ``|grad T|`` as the magnitude.

    The triangle index and per-triangle flows are built on first use and kept,
    so resampling (e.g. as a view zooms and pans) only locates the new points.
    """

    def __init__(self, layer: LayerResult):
        """Prepare to sample a layer.

        Args:
            layer: The layer's results.
        """
        self.layer = layer
        self._locator: TriangleLocator | None = None
        self._flows: dict[bool, np.ndarray] = {}  # heat? -> (M, 2) flow per triangle

    def flow(self, mode: DisplayMode) -> np.ndarray:
        """Flow vector of each triangle for a displayed quantity.

        Args:
            mode: The displayed quantity.

        Returns:
            ``(M, 2)`` vectors: A/mm² of current (zero off copper), or K/mm down the
            temperature gradient for :attr:`DisplayMode.TEMPERATURE`.
        """
        heat = mode is DisplayMode.TEMPERATURE
        if heat not in self._flows:
            layer = self.layer
            if heat:
                flow = -triangle_gradients(layer, layer.temperature)
            else:
                field = -triangle_gradients(layer, layer.voltage)
                length = np.hypot(field[:, 0], field[:, 1])
                unit = np.divide(field, length[:, None], out=np.zeros_like(field), where=length[:, None] > 0)
                flow = unit * (np.where(layer.copper, layer.current_density, 0.0) / 1e6)[:, None]
            self._flows[heat] = flow
        return self._flows[heat]

    def reference(self, mode: DisplayMode) -> float:
        """A typical large flow on this layer: the 95th percentile of its triangles' flow.

        Args:
            mode: The displayed quantity.

        Returns:
            The magnitude (0 if there's no flow).
        """
        magnitude = np.hypot(*self.flow(mode).T)
        moving = magnitude[magnitude > 0]
        return float(np.percentile(moving, 95)) if moving.size else 0.0

    def sample(
        self, mode: DisplayMode, spacing: float, bounds: tuple[float, float, float, float] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Flow arrows on a grid over (part of) the layer.

        The grid is anchored to the board's origin, at ``(i + 1/2) * spacing``, so
        sampling a moved or resized region keeps the arrows that are still in it
        exactly where they were.

        Args:
            mode: The displayed quantity.
            spacing: Grid spacing, mm.
            bounds: ``(x0, y0, x1, y1)`` region to cover, mm; the whole layer if None.

        Returns:
            ``(points, vectors)``: ``(K, 2)`` arrow positions (mm) and ``(K, 2)`` flow
            vectors there (A/mm² for current, K/mm for heat), for grid points on the
            mesh (and on copper, for current) where there's some flow.
        """
        layer = self.layer
        lo, hi = layer.nodes.min(axis=0), layer.nodes.max(axis=0)
        if bounds is not None:
            lo, hi = np.maximum(lo, bounds[:2]), np.minimum(hi, bounds[2:])
        xs = (np.arange(np.ceil(lo[0] / spacing - 0.5), np.floor(hi[0] / spacing - 0.5) + 1) + 0.5) * spacing
        ys = (np.arange(np.ceil(lo[1] / spacing - 0.5), np.floor(hi[1] / spacing - 0.5) + 1) + 0.5) * spacing
        if not len(xs) or not len(ys):
            return np.zeros((0, 2)), np.zeros((0, 2))
        gx, gy = np.meshgrid(xs, ys)
        points = np.column_stack([gx.ravel(), gy.ravel()])
        if self._locator is None:
            self._locator = TriangleLocator(layer.nodes, layer.triangles)
        triangle, _ = self._locator.locate(points)
        keep = triangle >= 0
        vectors = self.flow(mode)[np.maximum(triangle, 0)]
        keep &= np.hypot(vectors[:, 0], vectors[:, 1]) > 0
        return points[keep], vectors[keep]


def vector_field(layer: LayerResult, mode: DisplayMode, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Flow arrows on a regular grid over a whole layer (see :class:`VectorSampler`).

    Args:
        layer: The layer's results.
        mode: The displayed quantity.
        spacing: Grid spacing, mm.

    Returns:
        ``(points, vectors)``, as :meth:`VectorSampler.sample`.
    """
    return VectorSampler(layer).sample(mode, spacing)


class FieldProbe:
    """Looks up result values at arbitrary points on one layer's mesh.

    Built once per layer (a k-d tree of triangle centroids); each lookup checks
    the nearest few triangles for the one containing the point.
    """

    _CANDIDATES = 12  # nearest centroids checked per lookup
    _WIDE_CANDIDATES = 128  # second, wider search if the first misses

    def __init__(self, layer: LayerResult):
        """Index a layer's triangles.

        Args:
            layer: The layer's results.
        """
        self.layer = layer
        self._corners = layer.nodes[layer.triangles]  # (M, 3, 2)
        self._tree = cKDTree(self._corners.mean(axis=1))

    def locate(self, x: float, y: float) -> tuple[int, np.ndarray] | None:
        """Find the triangle containing a point.

        Args:
            x: X in board millimetres.
            y: Y in board millimetres.

        Returns:
            ``(triangle, barycentric weights)``, or None if the point is off the mesh
            (outside the board or in a hole).
        """
        # Next to a finely meshed hole, a big triangle's centroid can be farther
        # away than many small ones', so widen the search if the first pass misses.
        for k in (self._CANDIDATES, self._WIDE_CANDIDATES):
            _, candidates = self._tree.query((x, y), k=min(k, len(self._corners)))
            for t in np.atleast_1d(candidates):
                a, b, c = self._corners[t]
                det = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
                if det == 0:
                    continue
                l1 = ((x - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (y - a[1])) / det
                l2 = ((b[0] - a[0]) * (y - a[1]) - (x - a[0]) * (b[1] - a[1])) / det
                weights = np.array([1 - l1 - l2, l1, l2])
                if weights.min() >= -1e-9:
                    return int(t), weights
        return None

    def value_at(self, x: float, y: float, mode: DisplayMode) -> float | None:
        """The displayed quantity at a point, in display units.

        Voltage and temperature are interpolated within the triangle; current
        density is constant per triangle. Current density and voltage are only
        defined on copper.

        Args:
            x: X in board millimetres.
            y: Y in board millimetres.
            mode: Quantity to look up.

        Returns:
            The value, or None where the quantity isn't shown (off the mesh, or off copper).
        """
        found = self.locate(x, y)
        if found is None:
            return None
        t, weights = found
        if mode is not DisplayMode.TEMPERATURE and not self.layer.copper[t]:
            return None
        if mode is DisplayMode.CURRENT_DENSITY:
            return float(self.layer.current_density[t] / 1e6)
        nodal = self.layer.voltage if mode is DisplayMode.VOLTAGE else self.layer.temperature
        return float(weights @ nodal[self.layer.triangles[t]])
