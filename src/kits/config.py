"""Simulation settings stored in a KITS project file (see :mod:`kits.project`).

Each settings class validates itself and converts to/from the plain mapping
stored in its section of the ``.kits`` YAML file.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MESH_MAX_SIZE_MM = 1.0
DEFAULT_MESH_MIN_SIZE_MM = 0.1


class ConfigError(ValueError):
    """A project file is unreadable or contains invalid values."""


@dataclass
class MeshSettings:
    """Mesh element size limits for the gmsh mesher.

    Attributes:
        max_size_mm: Largest allowed element edge length, in millimetres.
        min_size_mm: Smallest element edge length gmsh may choose, in millimetres.
            Geometry finer than this (e.g. very short outline segments) can still
            force smaller elements locally.
    """

    max_size_mm: float = DEFAULT_MESH_MAX_SIZE_MM
    min_size_mm: float = DEFAULT_MESH_MIN_SIZE_MM

    def validate(self) -> None:
        """Check the sizes are positive and ordered.

        Raises:
            ConfigError: If a size is not positive or the minimum exceeds the maximum.
        """
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
        return {"max_size_mm": self.max_size_mm, "min_size_mm": self.min_size_mm}

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
            )
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"invalid mesh size: {exc}") from exc
        settings.validate()
        return settings
