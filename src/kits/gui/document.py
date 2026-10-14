"""Application document: the open KITS project, its linked board and its mesh.

Panels never load files themselves; they listen to :class:`Document` signals.
The document is a consumer and editor of the same project file and data
model the batch tools use, so the GUI stays in sync with CLI runs.

Edits (linking a board, changing settings) mark the project *dirty*; nothing
is written until :meth:`Document.save` or :meth:`Document.save_as`.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal

from ..config import ConfigError, CurrentSource, MeshSettings
from ..ingest import Board, BoardLoadError, Project, load_project
from ..mesh.model import BoardMesh, load_saved_mesh
from ..project import KitsProject, load_kits_project


class Document(QObject):
    """Holds the open KITS project, the KiCad board it links, and the board's mesh.

    Signals:
        project_changed(KitsProject | None): A project was created, opened or closed.
        board_changed(Project | None): The linked KiCad board was loaded, reloaded,
            replaced or unloaded (``None`` if no board is loaded).
        settings_changed(): Project settings (e.g. mesh sizes) were edited.
        sources_changed(): Current sources were added, edited or removed.
        dirty_changed(bool): The project gained or lost unsaved changes.
        mesh_changed(BoardMesh | None): A mesh was built, or cleared.
    """

    project_changed = Signal(object)
    board_changed = Signal(object)
    settings_changed = Signal()
    sources_changed = Signal()
    dirty_changed = Signal(bool)
    mesh_changed = Signal(object)

    def __init__(self, parent=None):
        """Create an empty document with no project open.

        Args:
            parent: Optional Qt parent object.
        """
        super().__init__(parent)
        self._project: KitsProject | None = None
        self._kicad: Project | None = None
        self._board_error: str | None = None
        self._board_mtime: float | None = None
        self._mesh: BoardMesh | None = None
        self._dirty = False
        self._mesh_notice: str | None = None
        self._mesh_from_file = False

    # -- state -------------------------------------------------------------------

    @property
    def project(self) -> KitsProject | None:
        """The open KITS project, or ``None`` if nothing is open."""
        return self._project

    @property
    def kicad(self) -> Project | None:
        """The loaded KiCad project (board plus ``.kicad_pro`` settings), if any."""
        return self._kicad

    @property
    def board(self) -> Board | None:
        """The loaded board, or ``None`` if no board is linked or it failed to load."""
        return self._kicad.board if self._kicad else None

    @property
    def board_error(self) -> str | None:
        """Why the linked board couldn't be loaded, if it couldn't."""
        return self._board_error

    @property
    def dirty(self) -> bool:
        """True if the project has changes not yet saved to its file."""
        return self._dirty

    @property
    def board_changed_on_disk(self) -> bool:
        """True if the loaded board file was modified (or removed) since it was loaded."""
        if self._kicad is None:
            return False
        try:
            return self._kicad.board.path.stat().st_mtime != self._board_mtime
        except OSError:
            return True

    @property
    def mesh(self) -> BoardMesh | None:
        """The most recently built mesh for the loaded board, if any."""
        return self._mesh

    @property
    def mesh_notice(self) -> str | None:
        """Why the project's saved mesh wasn't loaded (out of date or unreadable), if it wasn't."""
        return self._mesh_notice

    @property
    def mesh_from_file(self) -> bool:
        """True if the current mesh was reloaded from the project's saved mesh file."""
        return self._mesh is not None and self._mesh_from_file

    @property
    def mesh_is_stale(self) -> bool:
        """True if a mesh exists but was built with different settings than the project's."""
        if self._mesh is None or self._project is None:
            return False
        return self._mesh.settings != self._project.mesh

    # -- project lifecycle -----------------------------------------------------

    def new_project(self, path: str | Path, board_path: str | Path) -> KitsProject:
        """Create a project file linking ``board_path`` and open it.

        The board is loaded first, so nothing is written if it can't be read.

        Args:
            path: Where to create the ``.kits`` file (overwritten if it exists).
            board_path: The KiCad ``.kicad_pro`` or ``.kicad_pcb`` to link.

        Returns:
            The new project.

        Raises:
            BoardLoadError: If the board can't be loaded.
            ConfigError: If the project file can't be written.
        """
        kicad = load_project(board_path)
        project = KitsProject(path=path, board=board_path)
        project.save()
        self._set_project(project)
        self._set_kicad(kicad, None)
        return project

    def open_project(self, path: str | Path) -> KitsProject:
        """Open a ``.kits`` file and load the board it links.

        A board that fails to load doesn't stop the project opening: the
        project is shown with no board and :attr:`board_error` says why, so
        the user can re-link it.

        Args:
            path: The project file.

        Returns:
            The opened project.

        Raises:
            ConfigError: If the project file can't be read or is invalid; the
                previously open project is kept.
        """
        project = load_kits_project(path)
        self._set_project(project)
        self._load_linked_board()
        return project

    def close(self) -> None:
        """Close the project without saving."""
        self._project = None
        self._set_dirty(False)
        self.project_changed.emit(None)
        self._set_kicad(None, None)

    def save(self) -> None:
        """Write the project to its file.

        Raises:
            ConfigError: If no project is open, the settings are invalid, or the file
                can't be written.
        """
        if self._project is None:
            raise ConfigError("no project is open")
        self._project.save()
        self._set_dirty(False)

    def save_as(self, path: str | Path) -> None:
        """Write the project to a new file, which becomes the open project's file.

        Args:
            path: The new ``.kits`` location.

        Raises:
            ConfigError: If no project is open, the settings are invalid, or the file
                can't be written.
        """
        if self._project is None:
            raise ConfigError("no project is open")
        self._project.save(path)
        self._set_dirty(False)
        self.project_changed.emit(self._project)

    # -- board ---------------------------------------------------------------------

    def link_board(self, board_path: str | Path) -> None:
        """Link a different KiCad board to the open project and load it.

        Args:
            board_path: The ``.kicad_pro`` or ``.kicad_pcb`` file.

        Raises:
            ConfigError: If no project is open.
            BoardLoadError: If the board can't be loaded; the project is unchanged.
        """
        if self._project is None:
            raise ConfigError("no project is open")
        kicad = load_project(board_path)
        self._project.set_board(board_path)
        self._set_dirty(True)
        self._set_kicad(kicad, None)

    def reload_board(self) -> None:
        """Re-read the linked board from disk.

        Raises:
            BoardLoadError: If the board can no longer be loaded; the previously
                loaded board is kept.
        """
        if self._project is None or self._project.board is None:
            return
        self._set_kicad(load_project(self._project.board), None)

    def _load_linked_board(self) -> None:
        """Load the open project's board, recording (not raising) any failure."""
        board_path = self._project.board if self._project else None
        if board_path is None:
            self._set_kicad(None, "The project doesn't link a KiCad board yet.")
            return
        try:
            kicad = load_project(board_path)
        except BoardLoadError as exc:
            self._set_kicad(None, str(exc))
        else:
            self._set_kicad(kicad, None)

    # -- settings and mesh -----------------------------------------------------------

    @property
    def current_sources(self) -> list[CurrentSource]:
        """The open project's current sources (a copy; use the methods below to edit)."""
        return list(self._project.current_sources) if self._project else []

    def current_source(self, ref: str, pad: str) -> CurrentSource | None:
        """Return the source on a pad, if there is one.

        Args:
            ref: Footprint reference.
            pad: Pad number.

        Returns:
            The source, or None.
        """
        return next((s for s in self.current_sources if s.key == (ref, str(pad))), None)

    def add_current_source(self, ref: str, pad: str, current_a: float) -> CurrentSource:
        """Add a current source on a pad (in memory; save to persist).

        Args:
            ref: Footprint reference.
            pad: Pad number.
            current_a: Current into the board in amperes (negative for a sink).

        Returns:
            The new source.

        Raises:
            ConfigError: If no project is open, the pad already has a source, or the values are invalid.
        """
        project = self._require_project()
        source = CurrentSource(ref, str(pad), float(current_a))
        source.validate()
        if self.current_source(ref, pad) is not None:
            raise ConfigError(f"pad {source.label} already has a current source")
        project.current_sources.append(source)
        self._sources_edited()
        return source

    def set_source_current(self, ref: str, pad: str, current_a: float) -> None:
        """Change the current of an existing source.

        Args:
            ref: Footprint reference.
            pad: Pad number.
            current_a: New current in amperes.

        Raises:
            ConfigError: If there is no such source or the current is invalid.
        """
        project = self._require_project()
        for i, source in enumerate(project.current_sources):
            if source.key == (ref, str(pad)):
                updated = CurrentSource(source.ref, source.pad, float(current_a))
                updated.validate()
                if updated != source:
                    project.current_sources[i] = updated
                    self._sources_edited()
                return
        raise ConfigError(f"pad {ref}.{pad} has no current source")

    def remove_current_source(self, ref: str, pad: str) -> None:
        """Remove the source on a pad.

        Args:
            ref: Footprint reference.
            pad: Pad number.

        Raises:
            ConfigError: If there is no such source.
        """
        project = self._require_project()
        remaining = [s for s in project.current_sources if s.key != (ref, str(pad))]
        if len(remaining) == len(project.current_sources):
            raise ConfigError(f"pad {ref}.{pad} has no current source")
        project.current_sources = remaining
        self._sources_edited()

    def _require_project(self) -> KitsProject:
        """Return the open project.

        Returns:
            The project.

        Raises:
            ConfigError: If no project is open.
        """
        if self._project is None:
            raise ConfigError("no project is open")
        return self._project

    def _sources_edited(self) -> None:
        """Mark the project dirty and announce the change."""
        self._set_dirty(True)
        self.sources_changed.emit()

    def set_mesh_settings(self, settings: MeshSettings) -> None:
        """Change the project's mesh settings (in memory; save to persist).

        Args:
            settings: The new settings.

        Raises:
            ConfigError: If no project is open or the settings are invalid.
        """
        if self._project is None:
            raise ConfigError("no project is open")
        settings.validate()
        if settings == self._project.mesh:
            return
        self._project.mesh = settings
        self._set_dirty(True)
        self.settings_changed.emit()

    def set_mesh(self, mesh: BoardMesh | None) -> None:
        """Replace the current mesh.

        Args:
            mesh: The newly built mesh, or None to clear it.
        """
        self._mesh = mesh
        self._mesh_from_file = False
        self._mesh_notice = None
        self.mesh_changed.emit(mesh)

    # -- internals ---------------------------------------------------------------------

    def _set_project(self, project: KitsProject) -> None:
        """Make ``project`` the open project, with no unsaved changes.

        Args:
            project: The project.
        """
        self._project = project
        self._set_dirty(False)
        self.project_changed.emit(project)

    def _set_kicad(self, kicad: Project | None, error: str | None) -> None:
        """Replace the loaded board, then reload its saved mesh if that is still current.

        Args:
            kicad: The loaded KiCad project, or None.
            error: Why no board is loaded, if ``kicad`` is None.
        """
        self._kicad = kicad
        self._board_error = error
        try:
            self._board_mtime = kicad.board.path.stat().st_mtime if kicad else None
        except OSError:
            self._board_mtime = None
        self.board_changed.emit(kicad)
        self._mesh, self._mesh_notice = self._load_saved_mesh()
        self._mesh_from_file = self._mesh is not None
        self.mesh_changed.emit(self._mesh)

    def _load_saved_mesh(self) -> tuple[BoardMesh | None, str | None]:
        """Load the project's saved mesh if it matches the loaded board and current settings.

        Returns:
            ``(mesh, None)`` if a current saved mesh was loaded, ``(None, reason)`` if
            one exists but is out of date or unreadable, else ``(None, None)``.
        """
        if self._project is None or self._kicad is None:
            return None, None
        return load_saved_mesh(self._project.saved_mesh_path, self._kicad.board.path, self._project.mesh)

    def _set_dirty(self, dirty: bool) -> None:
        """Update the unsaved-changes flag, emitting ``dirty_changed`` if it changed.

        Args:
            dirty: The new state.
        """
        if dirty != self._dirty:
            self._dirty = dirty
            self.dirty_changed.emit(dirty)
