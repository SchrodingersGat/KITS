"""Colour maps for the GUI; the display quantities themselves are in :mod:`kits.postprocess.fields`."""

from __future__ import annotations

import numpy as np
from PySide6.QtGui import QColor

from ..postprocess.fields import (  # noqa: F401 (re-exported for the GUI)
    COLOR_STEPS,
    CURRENT_DENSITY_PERCENTILE,
    LOG_LOW_PERCENTILE,
    LOG_MAX_DECADES,
    DisplayMode,
    FieldProbe,
    VectorSampler,
    color_indices,
    format_resistance,
    format_value,
    scale_midpoint,
    supports_log,
    triangle_values,
    value_range,
    vector_field,
)


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
