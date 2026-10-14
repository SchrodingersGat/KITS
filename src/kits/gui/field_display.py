"""Turn solve results into per-triangle values and colours for display."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from PySide6.QtGui import QColor
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


def colormap(mode: DisplayMode, steps: int = COLOR_STEPS) -> list[QColor]:
    """Sample a mode's colour map.

    Args:
        mode: The display mode.
        steps: Number of colours.

    Returns:
        ``steps`` colours from low to high.
    """
    anchors = np.array(mode.value.anchors, float)
    positions = np.linspace(0, 1, len(anchors))
    t = np.linspace(0, 1, steps)
    rgb = np.column_stack([np.interp(t, positions, anchors[:, c]) for c in range(3)])
    return [QColor(int(r), int(g), int(b)) for r, g, b in rgb.round()]


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
