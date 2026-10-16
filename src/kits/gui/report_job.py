"""Write a project's HTML report (``kits.report.cli``) in a child process."""

from __future__ import annotations

from pathlib import Path

from .process_job import ProcessJob


class ReportJob(ProcessJob):
    """One report of a project's saved results; :attr:`succeeded` carries the written file's path.

    The ``done`` record's ``"stale"`` entry says why the reported results no
    longer match the project, if they don't.
    """

    module = "kits.report.cli"
    result_key = "report"
    what = "report generator"

    def __init__(self, project_path: str | Path, output: str | Path, parent=None):
        """Prepare (but don't start) a report.

        Args:
            project_path: The ``.kits`` project (saved, with its results).
            output: Where to write the ``.html`` report.
            parent: Optional Qt parent object.
        """
        super().__init__(project_path, parent)
        self._output = Path(output)

    def arguments(self) -> list[str]:
        """Pass the report's destination.

        Returns:
            ``["-o", <output>]``.
        """
        return ["-o", str(self._output)]

    def load_result(self, path: Path) -> Path:
        """Check the report was written.

        Args:
            path: The report file.

        Returns:
            The path.

        Raises:
            OSError: If the file isn't there.
        """
        if not path.is_file():
            raise OSError(f"{path} wasn't written")
        return path
