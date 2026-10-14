"""Application document: the currently open project, shared by all panels.

Panels never load files themselves; they listen to :class:`Document` signals.
Later phases (config, mesh, results) hang off this same object so the GUI
stays a consumer of the batch engine's data model.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal

from ..ingest import Project, load_project


class Document(QObject):
    """Holds the currently open project and emits ``project_changed`` when it is replaced."""

    project_changed = Signal(object)  # Project | None

    def __init__(self, parent=None):
        """Create an empty document with no project open.

        Args:
            parent: Optional Qt parent object.
        """
        super().__init__(parent)
        self._project: Project | None = None

    @property
    def project(self) -> Project | None:
        """The currently open project, or ``None`` if nothing is open."""
        return self._project

    def open(self, path: str | Path) -> Project:
        """Load ``path``; raises :class:`~kits.ingest.BoardLoadError` on failure.

        On failure the previously open project is kept.

        Args:
            path: A ``.kicad_pro`` or ``.kicad_pcb`` file.

        Returns:
            The newly loaded project.

        Raises:
            BoardLoadError: If the project or board cannot be loaded (propagated from the loader).
        """
        project = load_project(path)
        self._project = project
        self.project_changed.emit(project)
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
        self.project_changed.emit(None)
