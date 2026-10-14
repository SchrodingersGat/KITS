"""Layer colours, loosely following KiCad's default theme."""

from __future__ import annotations

from PySide6.QtGui import QColor

BACKGROUND = QColor(0, 16, 35)
EDGE_CUTS = QColor(208, 210, 205)
VIAS = QColor(194, 194, 194)
TH_PADS = QColor(227, 183, 46)
HOLES = QColor(BACKGROUND).lighter(130)
GRID = QColor(58, 76, 100)
GRID_ORIGIN = QColor(220, 220, 220)
CURSOR = QColor(160, 170, 185, 110)  # muted, translucent crosshair
MESH_COPPER = QColor(255, 255, 255, 150)  # mesh edges in copper
MESH_BARE = QColor(120, 150, 190, 110)  # mesh edges in bare board
SELECTION = QColor(255, 255, 255)  # selected pad outline
SOURCE = QColor(46, 204, 113)  # current into the board
SINK = QColor(255, 140, 26)  # current out of the board
MARKER = QColor(80, 220, 255)  # result probe markers
HEAT_SOURCE = QColor(235, 64, 52)  # heat injected at a pad
HEAT_SINK = QColor(66, 135, 245)  # pad held at a temperature or tied to ambient
ZONE_DARKEN = 160  # zones drawn this much darker than tracks on the same layer

_FRONT = QColor(200, 52, 52)
_BACK = QColor(77, 127, 196)
_INNER = [
    QColor(127, 200, 127),
    QColor(206, 125, 44),
    QColor(79, 203, 203),
    QColor(219, 98, 139),
    QColor(167, 165, 198),
    QColor(40, 204, 217),
    QColor(232, 178, 167),
    QColor(242, 237, 161),
    QColor(141, 203, 129),
    QColor(237, 124, 51),
    QColor(91, 195, 235),
    QColor(247, 111, 142),
]


def copper_color(name: str) -> QColor:
    """Display colour for a copper layer.

    F.Cu and B.Cu get fixed colours; inner layers (``In1.Cu``, ``In2.Cu``, ...) cycle
    through a palette indexed by the layer number in the name.

    Args:
        name: KiCad copper layer name, e.g. ``"F.Cu"`` or ``"In3.Cu"``.

    Returns:
        A new colour instance, safe for the caller to modify.
    """
    if name == "F.Cu":
        return QColor(_FRONT)
    if name == "B.Cu":
        return QColor(_BACK)
    digits = "".join(ch for ch in name if ch.isdigit())
    index = int(digits) - 1 if digits else 0
    return QColor(_INNER[index % len(_INNER)])
