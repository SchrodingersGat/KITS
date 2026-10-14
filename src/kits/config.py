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


@dataclass
class CurrentSource:
    """A current fed into (or drawn out of) the board at a pad.

    Attributes:
        ref: Footprint reference, e.g. ``"J1"``.
        pad: Pad number, e.g. ``"1"``. Every pad of the footprint with this number
            is included (footprints may repeat a number, e.g. thermal pads).
        current_a: Current into the board in amperes; negative for a sink.
    """

    ref: str
    pad: str
    current_a: float

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
            Mapping with ``ref``, ``pad`` and ``current_a``.
        """
        return {"ref": self.ref, "pad": self.pad, "current_a": self.current_a}

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
    """

    ref: str
    pad: str
    kind: HeatLoadKind
    value: float

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
            or ``thermal_resistance_k_w``.
        """
        return {"ref": self.ref, "pad": self.pad, self.kind.value: self.value}

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
                ref=str(data["ref"]), pad=str(data["pad"]), kind=kinds[0], value=float(data[kinds[0].value])
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
    """

    copper_conductivity_s_m: float = 5.8e7
    copper_thermal_conductivity_w_mk: float = 390.0
    dielectric_thermal_conductivity_w_mk: float = 0.3
    plating_thickness_mm: float = 0.025
    convection_top_w_m2k: float = 10.0
    convection_bottom_w_m2k: float = 10.0
    ambient_c: float = 25.0

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
