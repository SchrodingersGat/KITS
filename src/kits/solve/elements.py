"""Helpers for resistive elements: auto-grouping a footprint's pads and default body cooling."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import PadGroup, ResistiveElement, SolveSettings
from ..ingest.model import Footprint, Pad
from ..mesh.geometry import footprint_body

MM2 = 1e-6  # m² per mm²

# Terminal-to-terminal thermal resistance of chip parts by imperial size code,
# K/W: conduction along an alumina body (k ≈ 25 W/mK) between the end caps,
# R = L / (k · W · t) with typical body length between terminations, width and
# thickness. Rough (±50%), but the right order: about 100 K/W for most sizes.
CHIP_TERMINAL_TO_TERMINAL_K_W = {
    "01005": 250.0,
    "0201": 170.0,
    "0402": 115.0,
    "0603": 100.0,
    "0805": 85.0,
    "1206": 100.0,
    "1210": 65.0,
    "1812": 60.0,
    "2010": 100.0,
    "2220": 45.0,
    "2512": 100.0,
    "2725": 70.0,
    "2817": 60.0,
    "3920": 50.0,
}
_CHIP_SIZE = re.compile(r"(?<![0-9])(" + "|".join(CHIP_TERMINAL_TO_TERMINAL_K_W) + r")(?![0-9])")
# Semiconductor packages: the die sits on (is soldered to) one terminal's lead
# frame or tab, the larger by pad area (a MOSFET's drain, a diode's cathode
# tab), and reaches the other through bond wires or a clip and its leads.
POWER_PACKAGE_K_W = (1.5, 15.0)  # (die → tab terminal, die → other terminal)
SMALL_PACKAGE_K_W = (60.0, 150.0)  # SOT-23, SOD-123 and the like
_SMALL_PACKAGE = re.compile(r"SOT-?(23|323|363|523|563|143)|SC-?70|SOD-?(123|323|523|923)", re.IGNORECASE)
_POWER_PACKAGE = re.compile(
    r"TO-?\d|D2?PAK|PAK|SON|DFN|QFN|LFPAK|TOLL|TOLT|TOLG|HSOF|HDSOP|SuperSO|SO-?8|SOIC|PQFN|PG-|"
    r"\bSM[ABC]\b|SM[ABC]_|PowerDI|SOT-?(89|223|669)",
    re.IGNORECASE,
)
INDUCTOR_K_W = 10.0  # each terminal: the copper winding conducts well
GENERIC_K_W = 50.0  # each terminal, for bodies that can't be classified


@dataclass(frozen=True)
class BodyThermal:
    """Default thermal resistances from an element's body to its two terminals.

    Attributes:
        body_type: What the footprint was taken to be, e.g. ``"chip 2512"``.
        a_k_w: Body to terminal A, K/W.
        b_k_w: Body to terminal B, K/W.
    """

    body_type: str
    a_k_w: float
    b_k_w: float


def _group_area(footprint: Footprint, group: PadGroup) -> float:
    """Total pad copper area of a pad group (bounding size of each pad), mm².

    Args:
        footprint: The footprint.
        group: The pad group.

    Returns:
        The area.
    """
    numbers = set(group.pads)
    return sum(p.size[0] * p.size[1] for p in _copper_pads(footprint) if p.number in numbers)


def default_body_thermal(footprint: Footprint, element: ResistiveElement) -> BodyThermal:
    """Default body-to-terminal thermal resistances, from the footprint's body type.

    The type is read from the footprint's library name and reference:

    * chip parts with a size code (``R_2512_6432Metric``, ``C_0603``...): conduction
      along the ceramic body (:data:`CHIP_TERMINAL_TO_TERMINAL_K_W`), split evenly;
    * semiconductor packages (``Q``/``D``/``U`` parts, or names like ``TO-``,
      ``DPAK``, ``QFN``, ``SOT-23``): the die sits on the terminal with more pad
      area (:data:`POWER_PACKAGE_K_W`, :data:`SMALL_PACKAGE_K_W`);
    * inductors: :data:`INDUCTOR_K_W` to each terminal;
    * anything else: :data:`GENERIC_K_W` to each terminal.

    These are typical values for a first estimate; datasheet figures, where
    known, should be entered on the element instead.

    Args:
        footprint: The element's footprint.
        element: The element (its pad groups decide which terminal holds the die).

    Returns:
        The defaults.
    """
    name = footprint.lib_id.split(":")[-1]
    prefix = re.match(r"[A-Za-z]*", footprint.reference).group(0).upper()
    semiconductor = prefix.startswith(("Q", "D", "U", "T")) or bool(_POWER_PACKAGE.search(name))
    chip = _CHIP_SIZE.search(name)
    if chip and not (semiconductor and prefix.startswith(("Q", "U", "T"))):
        size = chip.group(1)
        half = CHIP_TERMINAL_TO_TERMINAL_K_W[size] / 2
        return BodyThermal(f"chip {size}", half, half)
    if prefix.startswith("L") or name.upper().startswith("L_"):
        return BodyThermal("inductor", INDUCTOR_K_W, INDUCTOR_K_W)
    if semiconductor or _SMALL_PACKAGE.search(name):
        small = bool(_SMALL_PACKAGE.search(name))
        near, far = SMALL_PACKAGE_K_W if small else POWER_PACKAGE_K_W
        kind = "small-signal package" if small else "power package"
        a_has_die = _group_area(footprint, element.group_a) >= _group_area(footprint, element.group_b)
        return BodyThermal(kind, near if a_has_die else far, far if a_has_die else near)
    return BodyThermal("unclassified body", GENERIC_K_W, GENERIC_K_W)


def terminal_thermal_resistances(footprint: Footprint, element: ResistiveElement) -> tuple[float, float]:
    """The thermal resistances from an element's body to its terminals, defaults filled in.

    Args:
        footprint: The element's footprint.
        element: The element.

    Returns:
        ``(body → A, body → B)`` in K/W.
    """
    default = default_body_thermal(footprint, element)
    a, b = element.terminal_thermal_resistances
    return (default.a_k_w if a is None else a, default.b_k_w if b is None else b)


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
