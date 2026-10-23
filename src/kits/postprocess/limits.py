"""Pass/fail checks of solve results against a project's limits.

Qt-free, so the GUI, the command-line tools and the HTML report agree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import DEFAULT_SPIKE_AREA_PCT, Limits
from ..solve.results import LayerResult, SolveResults

# The peak current density ignores the highest values over a small share of the
# copper area (Limits.current_density_spike_area_pct): the true maximum sits at
# sharp inside corners, where the density is singular and grows as the mesh is
# refined, so it would make the peak depend on the mesh rather than the design.


@dataclass(frozen=True)
class LimitCheck:
    """One limit compared with the results.

    Attributes:
        quantity: What is checked, e.g. ``"Current density"``.
        unit: Display unit.
        limit: The threshold.
        value: The results' value compared with it.
        where: Where that value occurs (a layer, or an element body).
        basis: How the value is taken (see :func:`current_density_basis`).
    """

    quantity: str
    unit: str
    limit: float
    value: float
    where: str
    basis: str

    @property
    def passed(self) -> bool:
        """Whether the value is within the limit (equal counts as a pass)."""
        return self.value <= self.limit

    @property
    def margin(self) -> float:
        """How far below the limit the value is (negative: over the limit), in the unit."""
        return self.limit - self.value


def _triangle_areas(layer) -> np.ndarray:
    """Areas of a layer's triangles, mm².

    Args:
        layer: A :class:`~kits.solve.results.LayerResult`.

    Returns:
        ``(M,)`` areas.
    """
    a, b, c = (layer.nodes[layer.triangles[:, k]] for k in range(3))
    return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    """The value below which ``percentile`` % of the total weight lies.

    Args:
        values: The values.
        weights: Their weights (e.g. areas).
        percentile: 0 to 100.

    Returns:
        The weighted percentile (0 if there are no values).
    """
    if not len(values):
        return 0.0
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    index = np.searchsorted(cumulative, percentile / 100 * cumulative[-1])
    return float(values[order][min(index, len(values) - 1)])


def current_density_basis(spike_area_pct: float) -> str:
    """How the peak current density is taken, in words.

    Args:
        spike_area_pct: See :attr:`kits.config.Limits.current_density_spike_area_pct`.

    Returns:
        E.g. ``"peak over 99.5% of the copper area (corner spikes excluded)"``.
    """
    if spike_area_pct > 0:
        return f"peak over {100 - spike_area_pct:g}% of the copper area (corner spikes excluded)"
    return "true maximum (corner spikes included)"


def _copper_density(layer: LayerResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A layer's copper triangles with their current density and area.

    Args:
        layer: The layer's results.

    Returns:
        ``(triangles, A/mm², mm²)``: indices of the copper triangles, and each one's
        current density and area.
    """
    copper = np.flatnonzero(layer.copper)
    return copper, layer.current_density[copper] / 1e6, _triangle_areas(layer)[copper]  # A/m² -> A/mm²


def _peak(values: np.ndarray, areas: np.ndarray, spike_area_pct: float) -> float:
    """The highest value once the top ``spike_area_pct`` % of the area is ignored.

    Args:
        values: Values per triangle.
        areas: Their areas.
        spike_area_pct: Share of the area to ignore, %; 0 for the maximum.

    Returns:
        The peak (0 if there are no values).
    """
    if spike_area_pct <= 0:
        return float(values.max()) if len(values) else 0.0
    return _weighted_percentile(values, areas, 100 - spike_area_pct)


def peak_current_density(
    results: SolveResults, spike_area_pct: float = DEFAULT_SPIKE_AREA_PCT
) -> tuple[float, str]:
    """The peak current density, ignoring corner spikes, and the layer it's highest on.

    Args:
        results: The results.
        spike_area_pct: Share of the copper area whose highest densities are
            ignored, % (see :attr:`kits.config.Limits.current_density_spike_area_pct`);
            0 for the true maximum.

    Returns:
        ``(A/mm², layer)``: the peak over every copper triangle on every layer,
        and the layer whose own peak is highest.
    """
    values, areas, per_layer = [], [], {}
    for layer in results.layers:
        if not layer.copper.any():
            continue
        _, density, area = _copper_density(layer)
        values.append(density)
        areas.append(area)
        per_layer[layer.layer] = _peak(density, area, spike_area_pct)
    if not values:
        return 0.0, ""
    peak = _peak(np.concatenate(values), np.concatenate(areas), spike_area_pct)
    return peak, max(per_layer, key=per_layer.get)


def layer_peak_current_density(
    layer: LayerResult, spike_area_pct: float = DEFAULT_SPIKE_AREA_PCT
) -> tuple[float, int] | None:
    """A layer's own peak current density and a triangle where it occurs.

    With spikes excluded, the triangle is the densest one at or below the peak:
    a real hot spot (e.g. the neck of a track) rather than a corner singularity.

    Args:
        layer: The layer's results.
        spike_area_pct: As for :func:`peak_current_density`.

    Returns:
        ``(A/mm², triangle index)``, or None if the layer carries no current.
    """
    triangles, density, area = _copper_density(layer)
    if not len(density) or density.max() <= 0:
        return None
    peak = _peak(density, area, spike_area_pct)
    at_or_below = np.flatnonzero(density <= peak)
    best = at_or_below[np.argmax(density[at_or_below])]
    return float(density[best]), int(triangles[best])


def hottest(results: SolveResults) -> tuple[float, str]:
    """The highest temperature in the results, and where it is.

    Args:
        results: The results.

    Returns:
        ``(°C, where)``: the hottest layer, or element body if that's hotter.
    """
    layer = max(results.layers, key=lambda r: float(r.temperature.max()))
    value, where = float(layer.temperature.max()), layer.layer
    for name, body in results.element_temperatures.items():
        if body >= value:
            value, where = body, f"{name} body"
    return value, where


def check_limits(results: SolveResults, limits: Limits) -> list[LimitCheck]:
    """Compare results with each limit that is set.

    Args:
        results: The results.
        limits: The project's limits.

    Returns:
        One check per limit set (empty if none are).
    """
    checks = []
    if limits.max_current_density_a_mm2 is not None:
        value, where = peak_current_density(results, limits.current_density_spike_area_pct)
        checks.append(
            LimitCheck(
                "Current density",
                "A/mm²",
                limits.max_current_density_a_mm2,
                value,
                where,
                current_density_basis(limits.current_density_spike_area_pct),
            )
        )
    if limits.max_temperature_c is not None:
        value, where = hottest(results)
        checks.append(
            LimitCheck(
                "Temperature",
                "°C",
                limits.max_temperature_c,
                value,
                where,
                "highest on the board or any element body",
            )
        )
    return checks


def verdict(checks: list[LimitCheck]) -> bool | None:
    """Overall pass/fail.

    Args:
        checks: :func:`check_limits`.

    Returns:
        True if every check passes, False if any fails, None if there are no limits.
    """
    return all(c.passed for c in checks) if checks else None
