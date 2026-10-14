"""KITS project files (``.kits``): the document the GUI and batch tools open.

A ``.kits`` file is YAML. It links one KiCad board and holds the simulation
settings KiCad doesn't model (see PLAN.md, "Config file schema")::

    board: Simple-In-Out.kicad_pro      # relative to the .kits file
    mesh:
      max_size_mm: 1.0
      min_size_mm: 0.1

The board path is stored relative to the project file whenever possible, so a
project folder can be moved or checked out elsewhere (e.g. in CI). Outputs go
to ``kits_output/`` beside the project file.

Keys KITS doesn't interpret yet are preserved when the file is saved.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import (
    ConfigError,
    CurrentSource,
    HeatLoad,
    MeshSettings,
    ResistiveElement,
    SolveSettings,
    current_multiplier_from_value,
    current_sources_from_list,
    heat_loads_from_list,
    resistive_elements_from_list,
)

PROJECT_SUFFIX = ".kits"
OUTPUT_DIR = "kits_output"


def _normalise(path: str | Path) -> Path:
    """Make a path absolute and collapse ``.``/``..`` without resolving symlinks.

    Args:
        path: Any path.

    Returns:
        The normalised absolute path.
    """
    return Path(os.path.normpath(Path(path).absolute()))


@dataclass
class KitsProject:
    """The contents of a ``.kits`` project file.

    Attributes:
        path: The project file's location (absolute).
        board: The linked KiCad ``.kicad_pro``/``.kicad_pcb`` file (absolute), or None.
        mesh: Mesh element size settings.
        current_sources: Currents fed into / drawn from the board at pads.
        current_multiplier: Scale applied to every source's current when solving.
        heat_loads: Heat sources and sinks applied to pads.
        resistive_elements: Lumped resistors between pad groups (resistors, MOSFETs...).
        solve: Materials, convection and ambient temperature for the solver.
        extra: Top-level keys KITS doesn't interpret yet, kept so saving doesn't drop them.
    """

    path: Path
    board: Path | None = None
    mesh: MeshSettings = field(default_factory=MeshSettings)
    current_sources: list[CurrentSource] = field(default_factory=list)
    current_multiplier: float = 1.0
    heat_loads: list[HeatLoad] = field(default_factory=list)
    resistive_elements: list[ResistiveElement] = field(default_factory=list)
    solve: SolveSettings = field(default_factory=SolveSettings)
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalise paths to absolute ones without ``..`` components."""
        self.path = _normalise(self.path)
        if self.board is not None:
            self.board = _normalise(self.board)

    def set_board(self, board: str | Path | None) -> None:
        """Link a KiCad board file (or unlink with None).

        Args:
            board: The ``.kicad_pro``/``.kicad_pcb`` path, or None.
        """
        self.board = _normalise(board) if board is not None else None

    @property
    def name(self) -> str:
        """Project name: the project file name without its extension."""
        return self.path.stem

    @property
    def output_dir(self) -> Path:
        """Folder for generated files (meshes, results), beside the project file."""
        return self.path.parent / OUTPUT_DIR

    @property
    def msh_path(self) -> Path:
        """Where the board mesh is written (gmsh format, for the solver)."""
        return self.output_dir / f"{self.name}.msh"

    @property
    def saved_mesh_path(self) -> Path:
        """Where the mesh arrays and their input fingerprints are saved for reloading."""
        return saved_mesh_path_for(self.msh_path)

    @property
    def results_path(self) -> Path:
        """Where solve results (fields per layer, with input fingerprints) are saved."""
        return self.output_dir / f"{self.name}.results.npz"

    def scaled_current_sources(self) -> list[CurrentSource]:
        """The enabled current sources as solved: each current times :attr:`current_multiplier`.

        Returns:
            New source objects; the stored ones are unchanged.
        """
        return [
            CurrentSource(s.ref, s.pad, s.current_a * self.current_multiplier)
            for s in self.current_sources
            if s.enabled
        ]

    @property
    def active_heat_loads(self) -> list[HeatLoad]:
        """The enabled heat loads (those that are solved)."""
        return [h for h in self.heat_loads if h.enabled]

    @property
    def active_resistive_elements(self) -> list[ResistiveElement]:
        """The enabled resistive elements (those that are solved)."""
        return [e for e in self.resistive_elements if e.enabled]

    def board_reference(self) -> str | None:
        """The board path as written to the file: relative to the project if possible.

        Returns:
            A POSIX-style relative path, an absolute path if no relative path exists
            (e.g. a different Windows drive), or None if no board is linked.
        """
        if self.board is None:
            return None
        try:
            return Path(os.path.relpath(self.board, self.path.parent)).as_posix()
        except ValueError:  # different drive on Windows
            return self.board.as_posix()

    def to_dict(self) -> dict:
        """Return the YAML representation of the project.

        Returns:
            Mapping with ``board``, ``mesh``, the solver sections, ``current_multiplier``
            (unless 1), ``current_sources``, ``heat_loads`` and
            ``resistive_elements`` (if any) and any preserved extra keys.
        """
        data: dict = {"board": self.board_reference(), "mesh": self.mesh.to_dict()}
        data.update(self.solve.to_sections())
        if self.current_multiplier != 1.0:
            data["current_multiplier"] = self.current_multiplier
        if self.current_sources:
            data["current_sources"] = [s.to_dict() for s in self.current_sources]
        if self.heat_loads:
            data["heat_loads"] = [h.to_dict() for h in self.heat_loads]
        if self.resistive_elements:
            data["resistive_elements"] = [e.to_dict() for e in self.resistive_elements]
        data.update({k: v for k, v in self.extra.items() if k not in data})
        return data

    def save(self, path: str | Path | None = None) -> None:
        """Write the project file, optionally to a new location ("save as").

        Saving elsewhere keeps the same board; its stored relative path is
        recomputed for the new location.

        Args:
            path: New project file location; None saves in place.

        Raises:
            ConfigError: If the settings are invalid or the file can't be written.
        """
        self.mesh.validate()
        self.solve.validate()
        for source in self.current_sources:
            source.validate()
        current_multiplier_from_value(self.current_multiplier)
        for load in self.heat_loads:
            load.validate()
        resistive_elements_from_list([e.to_dict() for e in self.resistive_elements])
        target = _normalise(path) if path is not None else self.path
        try:
            target.write_text(yaml.safe_dump(self._dict_for(target), sort_keys=False), encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"could not write {target}: {exc.strerror or exc}") from exc
        self.path = target

    def _dict_for(self, target: Path) -> dict:
        """Return :meth:`to_dict` as it would be written at ``target``.

        Args:
            target: The location the file will be written to.

        Returns:
            The YAML mapping, with the board path relative to ``target``.
        """
        original, self.path = self.path, target
        try:
            return self.to_dict()
        finally:
            self.path = original


def load_kits_project(path: str | Path) -> KitsProject:
    """Read a ``.kits`` project file.

    The linked board isn't loaded or checked here; see :func:`kits.ingest.load_project`.

    Args:
        path: The project file.

    Returns:
        The parsed project, with the board path resolved to an absolute path.

    Raises:
        ConfigError: If the file can't be read, isn't valid YAML, or holds invalid values.
    """
    path = _normalise(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"could not read {path}: {getattr(exc, 'strerror', None) or exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name}: top level must be a mapping")

    board = data.pop("board", None)
    if board is not None and not isinstance(board, str):
        raise ConfigError(f"{path.name}: 'board' must be a file path")
    mesh = MeshSettings.from_dict(data.pop("mesh", None))
    sources = current_sources_from_list(data.pop("current_sources", None))
    multiplier = current_multiplier_from_value(data.pop("current_multiplier", None))
    heat_loads = heat_loads_from_list(data.pop("heat_loads", None))
    elements = resistive_elements_from_list(data.pop("resistive_elements", None))
    solve = SolveSettings.from_sections(
        data.pop("ambient_temperature_c", None), data.pop("materials", None), data.pop("convection", None)
    )
    return KitsProject(
        path=path,
        board=(path.parent / board) if board else None,
        mesh=mesh,
        current_sources=sources,
        current_multiplier=multiplier,
        heat_loads=heat_loads,
        resistive_elements=elements,
        solve=solve,
        extra=data,
    )


def saved_mesh_path_for(msh_path: str | Path) -> Path:
    """Return the reloadable mesh file stored beside a ``.msh`` file.

    Args:
        msh_path: The gmsh mesh path, e.g. ``kits_output/board.msh``.

    Returns:
        e.g. ``kits_output/board.mesh.npz``.
    """
    msh_path = Path(msh_path)
    return msh_path.with_name(f"{msh_path.stem}.mesh.npz")


def default_project_path(board_path: str | Path) -> Path:
    """Suggested project file for a KiCad board: ``<board dir>/<board stem>.kits``.

    Args:
        board_path: A ``.kicad_pro`` or ``.kicad_pcb`` file.

    Returns:
        The suggested ``.kits`` path.
    """
    board_path = Path(board_path)
    return board_path.with_name(board_path.stem + PROJECT_SUFFIX)
