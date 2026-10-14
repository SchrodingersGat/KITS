"""Turn solve results into per-triangle values and colours for display."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from PySide6.QtGui import QColor

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


def value_range(results: SolveResults, mode: DisplayMode) -> tuple[float, float]:
    """Colour scale limits shared by all layers, so layers can be compared.

    Args:
        results: The solve results.
        mode: Quantity shown.

    Returns:
        ``(low, high)`` in display units; ``high > low``.
    """
    parts = [values[shown] for values, shown in (triangle_values(r, mode) for r in results.layers)]
    values = np.concatenate(parts) if parts else np.zeros(1)
    if values.size == 0:
        values = np.zeros(1)
    if mode is DisplayMode.CURRENT_DENSITY:
        low, high = 0.0, float(np.percentile(values, CURRENT_DENSITY_PERCENTILE))
    else:
        low, high = float(values.min()), float(values.max())
    if not high > low:
        high = low + max(abs(low) * 1e-6, 1e-12)
    return low, high


def color_indices(values: np.ndarray, low: float, high: float, steps: int = COLOR_STEPS) -> np.ndarray:
    """Map values to colour map indices, clamping outside the range.

    Args:
        values: Values in display units.
        low: Value mapped to the first colour.
        high: Value mapped to the last colour.
        steps: Number of colours.

    Returns:
        Integer indices in ``[0, steps)``.
    """
    t = np.clip((values - low) / (high - low), 0.0, 1.0)
    return np.minimum((t * steps).astype(int), steps - 1)


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
