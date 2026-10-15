"""Simulation settings stored in a KITS project file (see :mod:`kits.project`).

Each settings class validates itself and converts to/from the plain mapping
stored in its section of the ``.kits`` YAML file.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from enum import Enum

DEFAULT_MESH_MAX_SIZE_MM = 1.0
DEFAULT_MESH_MIN_SIZE_MM = 0.1
DEFAULT_MIN_CIRCLE_POINTS = 36
DEFAULT_CORNER_SIZE_MM = 0.1
# Global scale applied to every current source at solve time.
MAX_CURRENT_MULTIPLIER = 1000.0
ABSOLUTE_ZERO_C = -273.15
MAX_LAYER_THICKNESS_MM = 10.0
MIN_CIRCLE_POINTS_RANGE = (8, 1000)


class ConfigError(ValueError):
    """A project file is unreadable or contains invalid values."""


@dataclass
class MeshSettings:
    """Mesh resolution settings for the gmsh mesher.

    Attributes:
        max_size_mm: Largest allowed element edge length, in millimetres.
        min_size_mm: Smallest element edge length gmsh may choose, in millimetres.
            Geometry finer than this (e.g. very short outline segments) can still
            force smaller elements locally.
        min_circle_points: Minimum number of mesh nodes around each drill hole
            edge. Takes precedence over ``min_size_mm`` for small holes.
        corner_size_mm: Element size at concave copper corners (where current
            crowds), graded back to ``max_size_mm`` nearby; 0 turns corner
            refinement off. Takes precedence over ``min_size_mm`` there.
    """

    max_size_mm: float = DEFAULT_MESH_MAX_SIZE_MM
    min_size_mm: float = DEFAULT_MESH_MIN_SIZE_MM
    min_circle_points: int = DEFAULT_MIN_CIRCLE_POINTS
    corner_size_mm: float = DEFAULT_CORNER_SIZE_MM

    def validate(self) -> None:
        """Check the sizes are positive and ordered, and the circle point count is in range.

        Raises:
            ConfigError: If a size is not positive, the minimum exceeds the maximum, or
                ``min_circle_points`` is not an integer in :data:`MIN_CIRCLE_POINTS_RANGE`.
        """
        if not (0 <= self.corner_size_mm <= self.max_size_mm):
            raise ConfigError("corner size must be between 0 (off) and the max size")
        lo, hi = MIN_CIRCLE_POINTS_RANGE
        if not isinstance(self.min_circle_points, int) or not lo <= self.min_circle_points <= hi:
            raise ConfigError(f"min circle points must be a whole number from {lo} to {hi}")
        if not (self.min_size_mm > 0 and self.max_size_mm > 0):
            raise ConfigError("mesh sizes must be positive")
        if self.min_size_mm > self.max_size_mm:
            raise ConfigError(
                f"mesh min size ({self.min_size_mm} mm) is larger than max size ({self.max_size_mm} mm)"
            )

    def to_dict(self) -> dict:
        """Return the YAML representation of these settings.

        Returns:
            Mapping suitable for the ``mesh:`` section of the config file.
        """
        return {
            "max_size_mm": self.max_size_mm,
            "min_size_mm": self.min_size_mm,
            "min_circle_points": self.min_circle_points,
            "corner_size_mm": self.corner_size_mm,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> MeshSettings:
        """Build settings from the ``mesh:`` section, using defaults for missing keys.

        Args:
            data: The parsed ``mesh:`` mapping, or None if the section is absent.

        Returns:
            The validated settings.

        Raises:
            ConfigError: If the section is not a mapping or holds invalid values.
        """
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ConfigError("'mesh' must be a mapping")
        try:
            settings = cls(
                max_size_mm=float(data.get("max_size_mm", DEFAULT_MESH_MAX_SIZE_MM)),
                min_size_mm=float(data.get("min_size_mm", DEFAULT_MESH_MIN_SIZE_MM)),
                min_circle_points=_whole_number(data.get("min_circle_points", DEFAULT_MIN_CIRCLE_POINTS)),
                corner_size_mm=float(data.get("corner_size_mm", DEFAULT_CORNER_SIZE_MM)),
            )
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"invalid mesh setting: {exc}") from exc
        settings.validate()
        return settings


def _enabled_from(data: dict, what: str) -> bool:
    """Read the optional ``enabled`` flag of a list entry.

    Args:
        data: The parsed entry.
        what: Description for error messages, e.g. ``"current source J1.1"``.

    Returns:
        The flag (True if absent).

    Raises:
        ConfigError: If it isn't true or false.
    """
    value = data.get("enabled", True)
    if not isinstance(value, bool):
        raise ConfigError(f"'enabled' of {what} must be true or false")
    return value


@dataclass
class CurrentSource:
    """A current fed into (or drawn out of) the board at a pad.

    Attributes:
        ref: Footprint reference, e.g. ``"J1"``.
        pad: Pad number, e.g. ``"1"``. Every pad of the footprint with this number
            is included (footprints may repeat a number, e.g. thermal pads).
        current_a: Current into the board in amperes; negative for a sink.
        enabled: False to leave it out of solves and the board view without deleting it.
    """

    ref: str
    pad: str
    current_a: float
    enabled: bool = True

    @property
    def key(self) -> tuple[str, str]:
        """``(ref, pad)``, unique within a project."""
        return (self.ref, self.pad)

    @property
    def label(self) -> str:
        """Short display name, e.g. ``"J1.1"``."""
        return f"{self.ref}.{self.pad}"

    def validate(self) -> None:
        """Check the reference, pad and current.

        Raises:
            ConfigError: If the reference or pad is empty or the current isn't finite.
        """
        if not self.ref or not self.pad:
            raise ConfigError("a current source needs a footprint reference and a pad number")
        if not math.isfinite(self.current_a):
            raise ConfigError(f"current for {self.label} must be a finite number")

    def to_dict(self) -> dict:
        """Return the YAML representation.

        Returns:
            Mapping with ``ref``, ``pad``, ``current_a`` and ``enabled: false`` if disabled.
        """
        data = {"ref": self.ref, "pad": self.pad, "current_a": self.current_a}
        if not self.enabled:
            data["enabled"] = False
        return data

    @classmethod
    def from_dict(cls, data) -> CurrentSource:
        """Build a source from one entry of the ``current_sources:`` list.

        Args:
            data: The parsed entry.

        Returns:
            The validated source.

        Raises:
            ConfigError: If the entry is malformed.
        """
        if not isinstance(data, dict):
            raise ConfigError("each current source must be a mapping with ref, pad and current_a")
        try:
            source = cls(
                ref=str(data["ref"]),
                pad=str(data["pad"]),
                current_a=float(data["current_a"]),
                enabled=_enabled_from(data, f"current source {data.get('ref')}.{data.get('pad')}"),
            )
        except KeyError as exc:
            raise ConfigError(f"current source is missing {exc.args[0]!r}") from exc
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"invalid current source: {exc}") from exc
        source.validate()
        return source


def current_sources_from_list(data) -> list[CurrentSource]:
    """Parse the ``current_sources:`` section.

    Args:
        data: The parsed list, or None if the section is absent.

    Returns:
        The sources, in file order.

    Raises:
        ConfigError: If the section isn't a list, an entry is invalid, or a pad appears twice.
    """
    if data is None:
        return []
    if not isinstance(data, list):
        raise ConfigError("'current_sources' must be a list")
    sources = [CurrentSource.from_dict(item) for item in data]
    seen = set()
    for source in sources:
        if source.key in seen:
            raise ConfigError(f"pad {source.label} has more than one current source")
        seen.add(source.key)
    return sources


def current_multiplier_from_value(data) -> float:
    """Parse and check the ``current_multiplier:`` value.

    Every source's current is multiplied by it when solving, so a whole load case
    can be scaled without editing each source.

    Args:
        data: The parsed value, or None if the key is absent (meaning 1).

    Returns:
        The multiplier.

    Raises:
        ConfigError: If it isn't a number in ``(0, MAX_CURRENT_MULTIPLIER]``.
    """
    if data is None:
        return 1.0
    if isinstance(data, bool):
        raise ConfigError("'current_multiplier' must be a number")
    try:
        value = float(data)
    except (TypeError, ValueError) as exc:
        raise ConfigError("'current_multiplier' must be a number") from exc
    if not (0 < value <= MAX_CURRENT_MULTIPLIER):  # also rejects NaN
        raise ConfigError(f"'current_multiplier' must be above 0 and at most {MAX_CURRENT_MULTIPLIER:g}")
    return value


class HeatLoadKind(Enum):
    """How a heat load acts on its pad.

    The value is the key holding the load's number in the project file.
    """

    POWER = "power_w"  # heat injected into the board (negative extracts it)
    TEMPERATURE = "temperature_c"  # pad held at a fixed temperature
    RESISTANCE = "thermal_resistance_k_w"  # pad tied to ambient through a thermal resistance

    @property
    def is_sink(self) -> bool:
        """Whether this kind can take heat out of the board by itself (any but power)."""
        return self is not HeatLoadKind.POWER

    @property
    def unit(self) -> str:
        """Unit of the load's value: ``"W"``, ``"°C"`` or ``"K/W"``."""
        return {HeatLoadKind.POWER: "W", HeatLoadKind.TEMPERATURE: "°C", HeatLoadKind.RESISTANCE: "K/W"}[self]

    @property
    def title(self) -> str:
        """Human-readable name, e.g. ``"Power"``."""
        return {
            HeatLoadKind.POWER: "Power",
            HeatLoadKind.TEMPERATURE: "Fixed temperature",
            HeatLoadKind.RESISTANCE: "Thermal resistance",
        }[self]


@dataclass
class HeatLoad:
    """A heat source or sink applied to a pad's copper (on all of its layers).

    Attributes:
        ref: Footprint reference, e.g. ``"U1"``.
        pad: Pad number, e.g. ``"9"``. Every pad of the footprint with this number is included.
        kind: Power source, fixed temperature, or thermal resistance to ambient.
        value: Watts, °C, or K/W (°C/W) according to ``kind``.
        enabled: False to leave it out of solves and the board view without deleting it.
    """

    ref: str
    pad: str
    kind: HeatLoadKind
    value: float
    enabled: bool = True

    @property
    def key(self) -> tuple[str, str]:
        """``(ref, pad)``, unique within a project."""
        return (self.ref, self.pad)

    @property
    def label(self) -> str:
        """Short display name, e.g. ``"U1.9"``."""
        return f"{self.ref}.{self.pad}"

    @property
    def value_text(self) -> str:
        """The value with its unit, e.g. ``"2 W"`` or ``"5 K/W"``."""
        return f"{self.value:g} {self.kind.unit}"

    def validate(self) -> None:
        """Check the reference, pad and value.

        Raises:
            ConfigError: If the reference or pad is empty, the value isn't finite,
                a temperature is below absolute zero, or a resistance isn't positive.
        """
        if not self.ref or not self.pad:
            raise ConfigError("a heat load needs a footprint reference and a pad number")
        if not math.isfinite(self.value):
            raise ConfigError(f"heat load value for {self.label} must be a finite number")
        if self.kind is HeatLoadKind.TEMPERATURE and self.value <= ABSOLUTE_ZERO_C:
            raise ConfigError(f"temperature for {self.label} must be above absolute zero")
        if self.kind is HeatLoadKind.RESISTANCE and self.value <= 0:
            raise ConfigError(f"thermal resistance for {self.label} must be above zero")

    def to_dict(self) -> dict:
        """Return the YAML representation.

        Returns:
            Mapping with ``ref``, ``pad`` and one of ``power_w``, ``temperature_c``
            or ``thermal_resistance_k_w``, and ``enabled: false`` if disabled.
        """
        data = {"ref": self.ref, "pad": self.pad, self.kind.value: self.value}
        if not self.enabled:
            data["enabled"] = False
        return data

    @classmethod
    def from_dict(cls, data) -> HeatLoad:
        """Build a heat load from one entry of the ``heat_loads:`` list.

        Args:
            data: The parsed entry.

        Returns:
            The validated heat load.

        Raises:
            ConfigError: If the entry is malformed or doesn't give exactly one value.
        """
        keys = ", ".join(k.value for k in HeatLoadKind)
        if not isinstance(data, dict):
            raise ConfigError(f"each heat load must be a mapping with ref, pad and one of {keys}")
        kinds = [k for k in HeatLoadKind if k.value in data]
        if len(kinds) != 1:
            raise ConfigError(f"each heat load needs exactly one of {keys}")
        try:
            load = cls(
                ref=str(data["ref"]),
                pad=str(data["pad"]),
                kind=kinds[0],
                value=float(data[kinds[0].value]),
                enabled=_enabled_from(data, f"heat load {data.get('ref')}.{data.get('pad')}"),
            )
        except KeyError as exc:
            raise ConfigError(f"heat load is missing {exc.args[0]!r}") from exc
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"invalid heat load: {exc}") from exc
        load.validate()
        return load


def heat_loads_from_list(data) -> list[HeatLoad]:
    """Parse the ``heat_loads:`` section.

    Args:
        data: The parsed list, or None if the section is absent.

    Returns:
        The heat loads, in file order.

    Raises:
        ConfigError: If the section isn't a list, an entry is invalid, or a pad appears twice.
    """
    if data is None:
        return []
    if not isinstance(data, list):
        raise ConfigError("'heat_loads' must be a list")
    loads = [HeatLoad.from_dict(item) for item in data]
    seen = set()
    for load in loads:
        if load.key in seen:
            raise ConfigError(f"pad {load.label} has more than one heat load")
        seen.add(load.key)
    return loads


@dataclass
class PadGroup:
    """A named set of a footprint's pads that act as one terminal of an element.

    Attributes:
        name: Terminal name, e.g. ``"D"`` or a net name.
        pads: Pad numbers; every pad with each number is included.
    """

    name: str
    pads: list[str]

    def to_dict(self) -> dict:
        """Return the YAML representation.

        Returns:
            Mapping with ``name`` and ``pads``.
        """
        return {"name": self.name, "pads": list(self.pads)}

    @classmethod
    def from_dict(cls, data, which: str) -> PadGroup:
        """Build a pad group from its YAML mapping.

        Args:
            data: The parsed mapping.
            which: Key it was read from (``"pads_a"``/``"pads_b"``), for error messages.

        Returns:
            The pad group (not yet validated).

        Raises:
            ConfigError: If the entry is malformed.
        """
        if not isinstance(data, dict) or not isinstance(data.get("pads"), list):
            raise ConfigError(f"'{which}' must be a mapping with a name and a list of pads")
        return cls(name=str(data.get("name", "")), pads=[str(p) for p in data["pads"]])


@dataclass
class ResistiveElement:
    """A lumped resistor between two pad groups of one footprint (resistor, MOSFET, fuse...).

    Current through it is ``(V_a - V_b) / R`` from the area-weighted mean
    potentials of the two groups; its ``I²R`` heats the element body, which is
    tied to its pads and to ambient.

    Attributes:
        name: Unique element name, e.g. ``"R5"`` or ``"Q1 channel"``.
        ref: Footprint reference the pads belong to.
        group_a: One terminal's pads.
        group_b: The other terminal's pads.
        resistance_ohm: Resistance between the groups, Ω.
        thermal_resistance_board_k_w: Body-to-board thermal resistance, K/W; None
            for ideal contact (the heat goes straight into the board under the
            body's plan area: Fab outline, else courtyard, else the pads' extent).
        thermal_resistance_ambient_k_w: Body-to-ambient thermal resistance, K/W;
            None for convection from the body (``1 / (h · area)``, see
            :func:`kits.solve.elements.default_ambient_conductance`).
        enabled: False to leave it out of solves and the board view without deleting it.
    """

    name: str
    ref: str
    group_a: PadGroup
    group_b: PadGroup
    resistance_ohm: float
    thermal_resistance_board_k_w: float | None = None
    thermal_resistance_ambient_k_w: float | None = None
    enabled: bool = True

    @property
    def groups(self) -> tuple[PadGroup, PadGroup]:
        """``(group_a, group_b)``."""
        return (self.group_a, self.group_b)

    @property
    def all_pads(self) -> list[str]:
        """Pad numbers of both groups."""
        return [*self.group_a.pads, *self.group_b.pads]

    def validate(self) -> None:
        """Check the names, pad groups, resistance and thermal resistances.

        Raises:
            ConfigError: If a name or reference is empty, a group has no pads, the
                groups share a pad, or a resistance isn't a positive finite number.
        """
        if not self.name or not self.ref:
            raise ConfigError("a resistive element needs a name and a footprint reference")
        for group in self.groups:
            if not group.name:
                raise ConfigError(f"element {self.name}: each pad group needs a name")
            if not group.pads:
                raise ConfigError(f"element {self.name}: pad group {group.name!r} has no pads")
        shared = set(self.group_a.pads) & set(self.group_b.pads)
        if shared:
            raise ConfigError(f"element {self.name}: pad {', '.join(sorted(shared))} is in both groups")
        values = {
            "resistance": self.resistance_ohm,
            "thermal resistance to board": self.thermal_resistance_board_k_w,
            "thermal resistance to ambient": self.thermal_resistance_ambient_k_w,
        }
        for label, value in values.items():
            if value is not None and not (math.isfinite(value) and value > 0):
                raise ConfigError(f"element {self.name}: {label} must be a positive number")

    def to_dict(self) -> dict:
        """Return the YAML representation.

        Returns:
            Mapping with ``name``, ``ref``, ``resistance_ohm``, ``pads_a``, ``pads_b``
            the thermal resistances that aren't defaulted, and ``enabled: false`` if disabled.
        """
        data = {
            "name": self.name,
            "ref": self.ref,
            "resistance_ohm": self.resistance_ohm,
            "pads_a": self.group_a.to_dict(),
            "pads_b": self.group_b.to_dict(),
        }
        if self.thermal_resistance_board_k_w is not None:
            data["thermal_resistance_board_k_w"] = self.thermal_resistance_board_k_w
        if self.thermal_resistance_ambient_k_w is not None:
            data["thermal_resistance_ambient_k_w"] = self.thermal_resistance_ambient_k_w
        if not self.enabled:
            data["enabled"] = False
        return data

    @classmethod
    def from_dict(cls, data) -> ResistiveElement:
        """Build an element from one entry of the ``resistive_elements:`` list.

        Args:
            data: The parsed entry.

        Returns:
            The validated element.

        Raises:
            ConfigError: If the entry is malformed.
        """
        if not isinstance(data, dict):
            raise ConfigError("each resistive element must be a mapping")

        def optional(key: str) -> float | None:
            """Read an optional number.

            Args:
                key: The mapping key.

            Returns:
                The value, or None if absent.
            """
            return None if data.get(key) is None else float(data[key])

        try:
            element = cls(
                name=str(data["name"]),
                ref=str(data["ref"]),
                group_a=PadGroup.from_dict(data.get("pads_a"), "pads_a"),
                group_b=PadGroup.from_dict(data.get("pads_b"), "pads_b"),
                resistance_ohm=float(data["resistance_ohm"]),
                thermal_resistance_board_k_w=optional("thermal_resistance_board_k_w"),
                thermal_resistance_ambient_k_w=optional("thermal_resistance_ambient_k_w"),
                enabled=_enabled_from(data, f"resistive element {data.get('name')}"),
            )
        except KeyError as exc:
            raise ConfigError(f"resistive element is missing {exc.args[0]!r}") from exc
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"invalid resistive element: {exc}") from exc
        element.validate()
        return element


def resistive_elements_from_list(data) -> list[ResistiveElement]:
    """Parse the ``resistive_elements:`` section.

    Args:
        data: The parsed list, or None if the section is absent.

    Returns:
        The elements, in file order.

    Raises:
        ConfigError: If the section isn't a list, an entry is invalid, or a name repeats.
    """
    if data is None:
        return []
    if not isinstance(data, list):
        raise ConfigError("'resistive_elements' must be a list")
    elements = [ResistiveElement.from_dict(item) for item in data]
    names = [e.name for e in elements]
    for name in names:
        if names.count(name) > 1:
            raise ConfigError(f"more than one resistive element is named {name!r}")
    return elements


HEATSINK_SIDES = ("top", "bottom")
HEATSINK_MOUNTS = ("board", "components")


@dataclass
class Heatsink:
    """A heatsink pressed onto one face of the board over a rectangle.

    It is a lumped body: one temperature, joined to what it sits on through
    ``thermal_resistance_board_k_w`` and to ambient through
    ``thermal_resistance_ambient_k_w``. On the board, the contact spreads over
    the covered area, with the solder mask's resistance in series where it covers.

    Attributes:
        name: Unique name, e.g. ``"HS1"``.
        side: ``"top"`` or ``"bottom"`` face of the board.
        x_mm: Left edge, board (KiCad) coordinates in mm.
        y_mm: Top edge (smallest KiCad Y; KiCad's Y points down), mm.
        width_mm: Size along X, mm.
        height_mm: Size along Y, mm.
        thermal_resistance_board_k_w: Thermal resistance of the heatsink's contact
            (e.g. the thermal pad or paste) to what it sits on, K/W.
        thermal_resistance_ambient_k_w: Heatsink-to-ambient thermal resistance
            (the heatsink's datasheet value), K/W.
        mount: ``"board"``: pressed onto the board face (through the solder mask
            where it covers, which conducts less). ``"components"``: sitting on top of the
            resistive elements under it, touching their bodies and not the board.
        enabled: False to leave it out of solves and the board view without deleting it.
    """

    name: str
    side: str
    x_mm: float
    y_mm: float
    width_mm: float
    height_mm: float
    thermal_resistance_board_k_w: float = 1.0
    thermal_resistance_ambient_k_w: float = 10.0
    mount: str = "board"
    enabled: bool = True

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(x0, y0, x1, y1)`` in board millimetres."""
        return (self.x_mm, self.y_mm, self.x_mm + self.width_mm, self.y_mm + self.height_mm)

    @property
    def copper_layer(self) -> str:
        """The outer copper layer of its face: ``"F.Cu"`` or ``"B.Cu"``."""
        return "F.Cu" if self.side == "top" else "B.Cu"

    def validate(self) -> None:
        """Check the name, side, rectangle and thermal resistances.

        Raises:
            ConfigError: If the name is empty, the side unknown, the size or a
                resistance not a positive finite number, or a coordinate not finite.
        """
        if not self.name:
            raise ConfigError("a heatsink needs a name")
        if self.side not in HEATSINK_SIDES:
            raise ConfigError(f"heatsink {self.name}: side must be 'top' or 'bottom'")
        if self.mount not in HEATSINK_MOUNTS:
            raise ConfigError(f"heatsink {self.name}: mount must be 'board' or 'components'")
        if not (math.isfinite(self.x_mm) and math.isfinite(self.y_mm)):
            raise ConfigError(f"heatsink {self.name}: position must be finite")
        values = {
            "width": self.width_mm,
            "height": self.height_mm,
            "thermal resistance to board": self.thermal_resistance_board_k_w,
            "thermal resistance to ambient": self.thermal_resistance_ambient_k_w,
        }
        for label, value in values.items():
            if not (math.isfinite(value) and value > 0):
                raise ConfigError(f"heatsink {self.name}: {label} must be a positive number")

    def to_dict(self) -> dict:
        """Return the YAML representation.

        Returns:
            Mapping of the fields (``mount`` only for components, ``enabled`` only when false).
        """
        data = {
            "name": self.name,
            "side": self.side,
            "x_mm": self.x_mm,
            "y_mm": self.y_mm,
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
            "thermal_resistance_board_k_w": self.thermal_resistance_board_k_w,
            "thermal_resistance_ambient_k_w": self.thermal_resistance_ambient_k_w,
        }
        if self.mount != "board":
            data["mount"] = self.mount
        if not self.enabled:
            data["enabled"] = False
        return data

    @classmethod
    def from_dict(cls, data) -> Heatsink:
        """Build a heatsink from one entry of the ``heatsinks:`` list.

        Args:
            data: The parsed entry.

        Returns:
            The validated heatsink.

        Raises:
            ConfigError: If the entry is malformed.
        """
        if not isinstance(data, dict):
            raise ConfigError("each heatsink must be a mapping")
        try:
            heatsink = cls(
                name=str(data["name"]),
                side=str(data.get("side", "top")),
                x_mm=float(data["x_mm"]),
                y_mm=float(data["y_mm"]),
                width_mm=float(data["width_mm"]),
                height_mm=float(data["height_mm"]),
                thermal_resistance_board_k_w=float(data.get("thermal_resistance_board_k_w", 1.0)),
                thermal_resistance_ambient_k_w=float(data.get("thermal_resistance_ambient_k_w", 10.0)),
                mount=str(data.get("mount", "board")),
                enabled=_enabled_from(data, f"heatsink {data.get('name')}"),
            )
        except KeyError as exc:
            raise ConfigError(f"heatsink is missing {exc.args[0]!r}") from exc
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"invalid heatsink: {exc}") from exc
        heatsink.validate()
        return heatsink


def heatsinks_from_list(data) -> list[Heatsink]:
    """Parse the ``heatsinks:`` section.

    Args:
        data: The parsed list, or None if the section is absent.

    Returns:
        The heatsinks, in file order.

    Raises:
        ConfigError: If the section isn't a list, an entry is invalid, or a name repeats.
    """
    if data is None:
        return []
    if not isinstance(data, list):
        raise ConfigError("'heatsinks' must be a list")
    heatsinks = [Heatsink.from_dict(item) for item in data]
    names = [h.name for h in heatsinks]
    for name in names:
        if names.count(name) > 1:
            raise ConfigError(f"more than one heatsink is named {name!r}")
    return heatsinks


@dataclass
class StackupEdits:
    """Layer thicknesses changed from what the board file says (copper weight, dielectric).

    They only apply to the board version they were made for: if the board file
    changes (its fingerprint no longer matches), they are dropped.

    Attributes:
        board_hash: :func:`kits.mesh.fingerprint.board_hash` of the board they were made on.
        thickness_mm: ``{layer name: thickness in mm}``.
    """

    board_hash: str = ""
    thickness_mm: dict[str, float] = dataclasses.field(default_factory=dict)

    def validate(self) -> None:
        """Check the thicknesses.

        Raises:
            ConfigError: If a layer name is empty or a thickness isn't positive and plausible.
        """
        for name, mm in self.thickness_mm.items():
            if not name:
                raise ConfigError("stackup edits: layer names can't be empty")
            if not (math.isfinite(mm) and 0 < mm <= MAX_LAYER_THICKNESS_MM):
                raise ConfigError(
                    f"stackup edits: thickness of {name} must be above 0 and at most "
                    f"{MAX_LAYER_THICKNESS_MM:g} mm"
                )

    def to_dict(self) -> dict:
        """Return the YAML representation.

        Returns:
            Mapping with ``board_hash`` and ``thickness_mm``.
        """
        return {"board_hash": self.board_hash, "thickness_mm": dict(self.thickness_mm)}

    @classmethod
    def from_dict(cls, data) -> StackupEdits | None:
        """Read the ``stackup_edits:`` section.

        Args:
            data: The parsed mapping, or None if absent.

        Returns:
            The validated edits, or None if there are none.

        Raises:
            ConfigError: If the section is malformed.
        """
        if data is None:
            return None
        if not isinstance(data, dict) or not isinstance(data.get("thickness_mm", {}), dict):
            raise ConfigError("'stackup_edits' must be a mapping with board_hash and thickness_mm")
        try:
            edits = cls(
                board_hash=str(data.get("board_hash", "")),
                thickness_mm={str(k): float(v) for k, v in data.get("thickness_mm", {}).items()},
            )
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"invalid stackup edit: {exc}") from exc
        edits.validate()
        return edits if edits.thickness_mm else None


@dataclass
class SolveSettings:
    """Material and boundary-condition inputs for the solver.

    Stored in the project file as ``ambient_temperature_c``, ``materials:`` and
    ``convection:`` (see :meth:`to_sections`).

    Attributes:
        copper_conductivity_s_m: Electrical conductivity of copper and plating, S/m.
        copper_thermal_conductivity_w_mk: Thermal conductivity of copper, W/mK.
        dielectric_thermal_conductivity_w_mk: Thermal conductivity of the board material, W/mK.
        plating_thickness_mm: Barrel plating thickness of vias and plated holes, mm.
        convection_top_w_m2k: Heat transfer coefficient from the top surface, W/m²K.
        convection_bottom_w_m2k: Heat transfer coefficient from the bottom surface, W/m²K.
        ambient_c: Ambient temperature, °C.
        solder_mask_thermal_conductivity_w_mk: Thermal conductivity of the solder
            mask, W/mK; with its thickness it adds resistance to convection where
            the mask covers the board.
    """

    copper_conductivity_s_m: float = 5.8e7
    copper_thermal_conductivity_w_mk: float = 390.0
    dielectric_thermal_conductivity_w_mk: float = 0.3
    plating_thickness_mm: float = 0.025
    convection_top_w_m2k: float = 10.0
    convection_bottom_w_m2k: float = 10.0
    ambient_c: float = 25.0
    solder_mask_thermal_conductivity_w_mk: float = 0.25

    def validate(self) -> None:
        """Check the values are physically meaningful.

        Raises:
            ConfigError: If a conductivity or the plating thickness isn't positive, a
                convection coefficient is negative, both are zero, or a value isn't finite.
        """
        for name, value in dataclasses.asdict(self).items():
            if not math.isfinite(value):
                raise ConfigError(f"{name} must be a finite number")
        positive = {
            "copper conductivity": self.copper_conductivity_s_m,
            "copper thermal conductivity": self.copper_thermal_conductivity_w_mk,
            "dielectric thermal conductivity": self.dielectric_thermal_conductivity_w_mk,
            "plating thickness": self.plating_thickness_mm,
            "solder mask thermal conductivity": self.solder_mask_thermal_conductivity_w_mk,
        }
        for label, value in positive.items():
            if value <= 0:
                raise ConfigError(f"{label} must be positive")
        if self.convection_top_w_m2k < 0 or self.convection_bottom_w_m2k < 0:
            raise ConfigError("convection coefficients can't be negative")
        if self.convection_top_w_m2k + self.convection_bottom_w_m2k <= 0:
            raise ConfigError("at least one surface needs convection, or the heat has nowhere to go")
        if self.ambient_c < -273.15:
            raise ConfigError("ambient temperature is below absolute zero")

    def to_sections(self) -> dict:
        """Return the project-file representation (three top-level keys).

        Returns:
            Mapping with ``ambient_temperature_c``, ``materials`` and ``convection``.
        """
        return {
            "ambient_temperature_c": self.ambient_c,
            "materials": {
                "copper": {
                    "conductivity_s_m": self.copper_conductivity_s_m,
                    "thermal_conductivity_w_mk": self.copper_thermal_conductivity_w_mk,
                    "plating_thickness_mm": self.plating_thickness_mm,
                },
                "dielectric": {"thermal_conductivity_w_mk": self.dielectric_thermal_conductivity_w_mk},
                "solder_mask": {"thermal_conductivity_w_mk": self.solder_mask_thermal_conductivity_w_mk},
            },
            "convection": {
                "top_w_m2k": self.convection_top_w_m2k,
                "bottom_w_m2k": self.convection_bottom_w_m2k,
            },
        }

    @classmethod
    def from_sections(cls, ambient=None, materials=None, convection=None) -> SolveSettings:
        """Build settings from the project-file sections, using defaults for anything missing.

        Args:
            ambient: The ``ambient_temperature_c`` value, or None.
            materials: The ``materials:`` mapping, or None.
            convection: The ``convection:`` mapping, or None.

        Returns:
            The validated settings.

        Raises:
            ConfigError: If a section has the wrong shape or a value is invalid.
        """
        defaults = cls()
        materials = _mapping(materials, "materials")
        copper = _mapping(materials.get("copper"), "materials.copper")
        dielectric = _mapping(materials.get("dielectric"), "materials.dielectric")
        mask = _mapping(materials.get("solder_mask"), "materials.solder_mask")
        convection = _mapping(convection, "convection")
        try:
            settings = cls(
                copper_conductivity_s_m=float(
                    copper.get("conductivity_s_m", defaults.copper_conductivity_s_m)
                ),
                copper_thermal_conductivity_w_mk=float(
                    copper.get("thermal_conductivity_w_mk", defaults.copper_thermal_conductivity_w_mk)
                ),
                dielectric_thermal_conductivity_w_mk=float(
                    dielectric.get("thermal_conductivity_w_mk", defaults.dielectric_thermal_conductivity_w_mk)
                ),
                plating_thickness_mm=float(copper.get("plating_thickness_mm", defaults.plating_thickness_mm)),
                convection_top_w_m2k=float(convection.get("top_w_m2k", defaults.convection_top_w_m2k)),
                convection_bottom_w_m2k=float(
                    convection.get("bottom_w_m2k", defaults.convection_bottom_w_m2k)
                ),
                ambient_c=float(defaults.ambient_c if ambient is None else ambient),
                solder_mask_thermal_conductivity_w_mk=float(
                    mask.get("thermal_conductivity_w_mk", defaults.solder_mask_thermal_conductivity_w_mk)
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"invalid material or environment setting: {exc}") from exc
        settings.validate()
        return settings


def _mapping(value, name: str) -> dict:
    """Return a config section as a dict, treating a missing section as empty.

    Args:
        value: The parsed section, or None.
        name: Section name for error messages.

    Returns:
        The mapping.

    Raises:
        ConfigError: If the section isn't a mapping.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"'{name}' must be a mapping")
    return value


def _whole_number(value) -> int:
    """Convert a YAML value to an int, rejecting fractions and booleans.

    Args:
        value: The parsed value, e.g. ``36`` or ``36.0``.

    Returns:
        The integer value.

    Raises:
        ValueError: If the value is not a whole number.
    """
    if isinstance(value, bool) or float(value) != int(float(value)):
        raise ValueError(f"{value!r} is not a whole number")
    return int(float(value))
