"""Per-project KITS configuration, stored as YAML beside the KiCad project.

The file ``<project>.kits.yaml`` holds simulation inputs that KiCad doesn't
model (see PLAN.md, "Config file schema"). Both the GUI and batch runs read and
write it, so they never disagree. A missing file means "all defaults".

Only the sections KITS currently understands are interpreted; any other keys
in the file are preserved untouched when it is saved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

CONFIG_SUFFIX = ".kits.yaml"

DEFAULT_MESH_MAX_SIZE_MM = 1.0
DEFAULT_MESH_MIN_SIZE_MM = 0.1


class ConfigError(ValueError):
    """The config file is unreadable or contains invalid values."""


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


@dataclass
class ProjectConfig:
    """The contents of a ``.kits.yaml`` file.

    Attributes:
        path: Where the config is (or will be) stored.
        mesh: Mesh element size settings.
        extra: Top-level keys KITS doesn't interpret yet, kept so saving doesn't drop them.
    """

    path: Path
    mesh: MeshSettings = field(default_factory=MeshSettings)
    extra: dict = field(default_factory=dict)

    @property
    def exists(self) -> bool:
        """Whether the config file is present on disk."""
        return self.path.is_file()

    def save(self) -> None:
        """Write the config to :attr:`path`, preserving uninterpreted keys.

        Raises:
            ConfigError: If the settings are invalid or the file can't be written.
        """
        self.mesh.validate()
        data = dict(self.extra)
        data["mesh"] = self.mesh.to_dict()
        try:
            self.path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"could not write {self.path}: {exc}") from exc


def config_path_for(project_path: str | Path) -> Path:
    """Return the config file path for a KiCad project or board file.

    Args:
        project_path: A ``.kicad_pro`` or ``.kicad_pcb`` path.

    Returns:
        ``<dir>/<stem>.kits.yaml``.
    """
    project_path = Path(project_path)
    return project_path.with_name(project_path.stem + CONFIG_SUFFIX)


def load_config(path: str | Path) -> ProjectConfig:
    """Load a ``.kits.yaml`` file; a missing file gives default settings.

    Args:
        path: The config file path (see :func:`config_path_for`).

    Returns:
        The parsed config.

    Raises:
        ConfigError: If the file can't be read, isn't valid YAML, or holds invalid values.
    """
    path = Path(path)
    if not path.is_file():
        return ProjectConfig(path=path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"could not read {path}: {exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    mesh = MeshSettings.from_dict(data.pop("mesh", None))
    return ProjectConfig(path=path, mesh=mesh, extra=data)
