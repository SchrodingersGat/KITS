"""Board ingest: KiCad files -> neutral board data model."""

from .kicad import BoardLoadError, load_board, load_project
from .model import Board, Project

__all__ = ["Board", "BoardLoadError", "Project", "load_board", "load_project"]
