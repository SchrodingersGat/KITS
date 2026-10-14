"""Run the mesher (``kits.mesh.cli``) in a child process."""

from __future__ import annotations

from pathlib import Path

from ..mesh.model import BoardMesh
from .process_job import ProcessJob


class MeshJob(ProcessJob):
    """One meshing run of a project; :attr:`succeeded` carries the :class:`BoardMesh`."""

    module = "kits.mesh.cli"
    result_key = "mesh"
    what = "mesher"

    def load_result(self, path: Path) -> BoardMesh:
        """Load the saved mesh the mesher wrote.

        Args:
            path: The ``.mesh.npz`` file.

        Returns:
            The mesh.
        """
        return BoardMesh.load_npz(path)
