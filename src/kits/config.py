"""Simulation settings stored in a KITS project file (see :mod:`kits.project`).

Each settings class validates itself and converts to/from the plain mapping
stored in its section of the ``.kits`` YAML file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

DEFAULT_MESH_MAX_SIZE_MM = 1.0
DEFAULT_MESH_MIN_SIZE_MM = 0.1
DEFAULT_MIN_CIRCLE_POINTS = 36
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
    """

    max_size_mm: float = DEFAULT_MESH_MAX_SIZE_MM
    min_size_mm: float = DEFAULT_MESH_MIN_SIZE_MM
    min_circle_points: int = DEFAULT_MIN_CIRCLE_POINTS

    def validate(self) -> None:
        """Check the sizes are positive and ordered, and the circle point count is in range.

        Raises:
            ConfigError: If a size is not positive, the minimum exceeds the maximum, or
                ``min_circle_points`` is not an integer in :data:`MIN_CIRCLE_POINTS_RANGE`.
        """
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
