"""Pass/fail checks of solve results against a project's limits.

Qt-free, so the GUI, the command-line tools and the HTML report agree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Limits
from ..solve.results import SolveResults

# Current density is compared at this area-weighted percentile of the copper:
# the value exceeded over only 0.5% of the copper area. The true maximum sits
# at sharp inside corners, where the density is singular and grows as the mesh
# is refined, so it would make the check depend on the mesh rather than the design.
CURRENT_DENSITY_AREA_PERCENTILE = 99.5


@dataclass(frozen=True)
class LimitCheck:
    """One limit compared with the results.

    Attributes:
        quantity: What is checked, e.g. ``"Current density"``.
        unit: Display unit.
        limit: The threshold.
        value: The results' value compared with it.
        where: Where that value occurs (a layer, or an element body).
        basis: How the value is taken, e.g. ``"peak over 99.5% of the copper area"``.
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


def peak_current_density(results: SolveResults) -> tuple[float, str]:
    """The current density exceeded over only 0.5% of the copper area, and the layer it's highest on.

    Args:
        results: The results.

    Returns:
        ``(A/mm², layer)``: the area-weighted :data:`CURRENT_DENSITY_AREA_PERCENTILE`
        of every copper triangle on every layer, and the layer whose own
        percentile is highest.
    """
    values, areas, per_layer = [], [], {}
    for layer in results.layers:
        copper = layer.copper
        if not copper.any():
            continue
        density = layer.current_density[copper] / 1e6  # A/m² -> A/mm²
        area = _triangle_areas(layer)[copper]
        values.append(density)
        areas.append(area)
        per_layer[layer.layer] = _weighted_percentile(density, area, CURRENT_DENSITY_AREA_PERCENTILE)
    if not values:
        return 0.0, ""
    peak = _weighted_percentile(
        np.concatenate(values), np.concatenate(areas), CURRENT_DENSITY_AREA_PERCENTILE
    )
    return peak, max(per_layer, key=per_layer.get)


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
        value, where = peak_current_density(results)
        checks.append(
            LimitCheck(
                "Current density",
                "A/mm²",
                limits.max_current_density_a_mm2,
                value,
                where,
                f"peak over {CURRENT_DENSITY_AREA_PERCENTILE:g}% of the copper area (corner spikes excluded)",
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
