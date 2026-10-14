"""Helpers for resistive elements: auto-grouping a footprint's pads and default body cooling."""

from __future__ import annotations

import re

from ..config import PadGroup, SolveSettings
from ..ingest.model import Footprint, Pad
from ..mesh.geometry import footprint_body

MM2 = 1e-6  # m² per mm²


def _natural_key(number: str) -> list:
    """Sort key putting pad "2" before "10".

    Args:
        number: Pad number.

    Returns:
        The key.
    """
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", number)]


def _copper_pads(footprint: Footprint) -> list[Pad]:
    """The footprint's pads that have copper (mechanical holes excluded).

    Args:
        footprint: The footprint.

    Returns:
        The pads.
    """
    return [p for p in footprint.pads if p.layers]


def pad_numbers(footprint: Footprint) -> list[str]:
    """Distinct copper pad numbers of a footprint, in natural order.

    Args:
        footprint: The footprint.

    Returns:
        e.g. ``["1", "2", "3"]``.
    """
    return sorted({p.number for p in _copper_pads(footprint) if p.number}, key=_natural_key)


def auto_pad_groups(footprint: Footprint) -> tuple[PadGroup, PadGroup] | None:
    """Guess an element's two terminals from a footprint's pads.

    Pads are grouped by net, and the two groups with the most pad area are the
    terminals: a MOSFET's drain pins and tab, and its source pins, but not the
    small gate pad. Groups are named after their nets. Without two nets, the
    first pad number is one terminal and the rest the other ("A" and "B").

    Args:
        footprint: The footprint.

    Returns:
        ``(group_a, group_b)``, or None if the footprint has fewer than two pad numbers.
    """
    numbers = pad_numbers(footprint)
    if len(numbers) < 2:
        return None
    by_net: dict[str, set[str]] = {}
    area: dict[str, float] = {}
    for pad in _copper_pads(footprint):
        if pad.net and pad.number:
            by_net.setdefault(pad.net, set()).add(pad.number)
            area[pad.net] = area.get(pad.net, 0.0) + pad.size[0] * pad.size[1]
    # A pad number on two nets can't be split between terminals: drop such nets.
    nets = [n for n in by_net if not any(by_net[n] & by_net[m] for m in by_net if m != n)]
    if len(nets) >= 2:
        largest = sorted(nets, key=lambda n: -area[n])[:2]
        groups = [PadGroup(net, sorted(by_net[net], key=_natural_key)) for net in largest]
        groups.sort(key=lambda g: _natural_key(g.pads[0]))
        return groups[0], groups[1]
    return PadGroup("A", numbers[:1]), PadGroup("B", numbers[1:])


def body_area_mm2(footprint: Footprint) -> float:
    """Plan area of the component body (see :func:`kits.mesh.geometry.footprint_body`).

    Args:
        footprint: The footprint.

    Returns:
        Area in mm² (0 if the footprint has no outline and no copper pads).
    """
    return footprint_body(footprint)[0].area


def default_ambient_conductance(footprint: Footprint, settings: SolveSettings) -> float:
    """Body-to-ambient conductance used when an element has no thermal resistance to ambient.

    Convection from the top of the body: ``h · area``, with ``h`` for the board
    face the footprint is on and the body's plan area (:func:`body_area_mm2`).

    Args:
        footprint: The element's footprint.
        settings: Supplies the convection coefficients.

    Returns:
        Conductance in W/K (0 if that face has no convection).
    """
    h = settings.convection_bottom_w_m2k if footprint.side == "B.Cu" else settings.convection_top_w_m2k
    return h * body_area_mm2(footprint) * MM2


_SI_PREFIX = {
    "p": 1e-12,
    "n": 1e-9,
    "u": 1e-6,
    "µ": 1e-6,
    "m": 1e-3,
    "": 1.0,
    "R": 1.0,
    "r": 1.0,
    "k": 1e3,
    "K": 1e3,
    "M": 1e6,
    "G": 1e9,
}
_RESISTANCE = re.compile(r"\s*(\d*)(?:[.,](\d+))?\s*([pnuµmrRkKMG]?)(\d*)\s*(?:Ω|[oO]hms?|R)?")


def parse_resistance(text: str) -> float | None:
    """Read a resistance from a component value, as KiCad schematics write them.

    Accepts SI prefixes (``"10k"``, ``"100m"``, ``"2.2M"``), the prefix or ``R``
    as decimal point (``"4k7"``, ``"4R7"``, ``"R005"``) and an optional unit
    (``"10mΩ"``, ``"10 ohm"``, ``"10mR"``). Trailing notes after a space or
    punctuation are ignored (``"10k 1%"``); other units (``"10uF"``) and part
    numbers (``"2N7002"``) are rejected.

    Args:
        text: The value text.

    Returns:
        Ohms, or None if the text isn't a positive resistance.
    """
    match = _RESISTANCE.match(text)
    if match is None:
        return None
    whole, fraction, prefix, trailing = match.groups()
    rest = text[match.end() :]
    if not (whole or fraction or trailing) or (rest and rest[0].isalpha()):
        return None
    if fraction and trailing:
        return None
    digits = f"{whole or '0'}.{fraction or trailing or '0'}"
    value = float(digits) * _SI_PREFIX[prefix]
    return value if value > 0 else None
