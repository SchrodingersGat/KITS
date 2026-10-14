"""Application document: the currently open project, shared by all panels.

Panels never load files themselves; they listen to :class:`Document` signals.
Later phases (config, mesh, results) hang off this same object so the GUI
stays a consumer of the batch engine's data model.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal

from ..config import ConfigError, MeshSettings, ProjectConfig, config_path_for, load_config
from ..ingest import Project, load_project
from ..mesh.model import BoardMesh


class Document(QObject):
    """Holds the open project, its KITS config and its mesh.

    Signals:
        project_changed(Project | None): A project was opened, reloaded or closed.
        config_changed(): The project config was edited.
        mesh_changed(BoardMesh | None): A mesh was built, or cleared.
    """

    project_changed = Signal(object)  # Project | None
    config_changed = Signal()
    mesh_changed = Signal(object)  # BoardMesh | None

    def __init__(self, parent=None):
        """Create an empty document with no project open.

        Args:
            parent: Optional Qt parent object.
        """
        super().__init__(parent)
        self._project: Project | None = None
        self._config: ProjectConfig | None = None
        self._config_error: str | None = None
        self._mesh: BoardMesh | None = None
        self._board_mtime: float | None = None

    @property
    def project(self) -> Project | None:
        """The currently open project, or ``None`` if nothing is open."""
        return self._project

    @property
    def config(self) -> ProjectConfig | None:
        """The open project's ``.kits.yaml`` config (defaults if the file is missing)."""
        return self._config

    @property
    def config_error(self) -> str | None:
        """Why the config file couldn't be read at open time, if it couldn't; defaults are used."""
        return self._config_error

    @property
    def board_changed_on_disk(self) -> bool:
        """True if the open board file was modified (or removed) since it was loaded."""
        if self._project is None:
            return False
        try:
            return self._project.board.path.stat().st_mtime != self._board_mtime
        except OSError:
            return True

    @property
    def mesh(self) -> BoardMesh | None:
        """The most recently built mesh for the open board, if any."""
        return self._mesh

    @property
    def mesh_is_stale(self) -> bool:
        """True if a mesh exists but was built with different settings than the current config."""
        if self._mesh is None or self._config is None:
            return False
        return self._mesh.settings != self._config.mesh

    def open(self, path: str | Path) -> Project:
        """Load ``path``; raises :class:`~kits.ingest.BoardLoadError` on failure.

        On failure the previously open project is kept. The project's
        ``.kits.yaml`` is loaded too; if it is invalid, defaults are used and
        :attr:`config_error` says why. Any existing mesh is cleared.

        Args:
            path: A ``.kicad_pro`` or ``.kicad_pcb`` file.

        Returns:
            The newly loaded project.

        Raises:
            BoardLoadError: If the project or board cannot be loaded (propagated from the loader).
        """
        project = load_project(path)
        try:
            board_mtime = project.board.path.stat().st_mtime
        except OSError:
            board_mtime = None
        config_path = config_path_for(project.path)
        try:
            self._config, self._config_error = load_config(config_path), None
        except ConfigError as exc:
            self._config, self._config_error = ProjectConfig(path=config_path), str(exc)
        self._project = project
        self._board_mtime = board_mtime
        self._mesh = None
        self.project_changed.emit(project)
        self.mesh_changed.emit(None)
        return project

    def reload(self) -> Project | None:
        """Re-read the open project's board from disk.

        Returns:
            The reloaded project, or ``None`` if nothing is open.

        Raises:
            BoardLoadError: If the board can no longer be loaded (propagated from :meth:`open`).
        """
        if self._project is None:
            return None
        # The board path always exists; the .kicad_pro may not if a bare
        # .kicad_pcb was opened.
        return self.open(self._project.board.path)

    def close(self) -> None:
        """Close the open project and emit ``project_changed(None)``."""
        self._project = None
        self._config = None
        self._config_error = None
        self._mesh = None
        self.project_changed.emit(None)
        self.mesh_changed.emit(None)

    def set_mesh_settings(self, settings: MeshSettings) -> None:
        """Store new mesh settings and save them to the project's ``.kits.yaml``.

        Args:
            settings: The new settings.

        Raises:
            ConfigError: If no project is open, the settings are invalid, or the file can't be written.
        """
        if self._config is None:
            raise ConfigError("no project is open")
        settings.validate()
        if settings == self._config.mesh and self._config.exists:
            return
        self._config.mesh = settings
        self._config.save()
        self.config_changed.emit()

    def set_mesh(self, mesh: BoardMesh | None) -> None:
        """Replace the current mesh.

        Args:
            mesh: The newly built mesh, or None to clear it.
        """
        self._mesh = mesh
        self.mesh_changed.emit(mesh)
