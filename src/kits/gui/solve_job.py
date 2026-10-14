"""Run the solver (``kits.solve.cli``) in a child process."""

from __future__ import annotations

from pathlib import Path

from ..solve.results import SolveResults
from .process_job import ProcessJob


class SolveJob(ProcessJob):
    """One solve of a project; :attr:`succeeded` carries the :class:`SolveResults`.

    The child meshes the board first if there is no current saved mesh; the
    ``done`` record's ``"mesh"`` entry then names the (re)saved mesh.
    """

    module = "kits.solve.cli"
    result_key = "results"
    what = "solver"

    def load_result(self, path: Path) -> SolveResults:
        """Load the results the solver wrote.

        Args:
            path: The ``.results.npz`` file.

        Returns:
            The results.
        """
        return SolveResults.load_npz(path)
