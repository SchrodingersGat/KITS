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
    project_changed = Signal(object)  # Project | None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._project: Project | None = None

    @property
    def project(self) -> Project | None:
        return self._project

    def open(self, path: str | Path) -> Project:
        """Load ``path``; raises :class:`~kits.ingest.BoardLoadError` on failure.

        On failure the previously open project is kept.
        """
        project = load_project(path)
        self._project = project
        self.project_changed.emit(project)
        return project

    def reload(self) -> Project | None:
        if self._project is None:
            return None
        # The board path always exists; the .kicad_pro may not if a bare
        # .kicad_pcb was opened.
        return self.open(self._project.board.path)

    def close(self) -> None:
        self._project = None
        self.project_changed.emit(None)
