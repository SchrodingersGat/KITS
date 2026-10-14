"""Run the mesher (``kits.mesh.cli``) in a child process and report progress.

Running gmsh out of process keeps the GUI responsive, lets the user cancel
mid-layer (the process is killed), and means a gmsh crash only fails the job.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, Signal

from ..mesh.model import BoardMesh

_STDERR_TAIL = 2000  # characters of child stderr included in error messages


class MeshJob(QObject):
    """One meshing run of a project in a child Python process.

    Signals:
        progress(str, float): Step description and overall fraction (0-1).
        succeeded(BoardMesh): Meshing finished and the result was loaded.
        failed(str): Meshing failed; the argument explains why.
        cancelled(): The job was cancelled by :meth:`cancel`.
    """

    progress = Signal(str, float)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, project_path: str | Path, parent=None):
        """Prepare (but don't start) a job.

        Args:
            project_path: The ``.kits`` project to mesh. The mesher reads the board link
                and settings from the file, so save any edits before starting.
            parent: Optional Qt parent object.
        """
        super().__init__(parent)
        self._project_path = Path(project_path)
        self._tmpdir: Path | None = None
        self._buffer = ""
        self._error: str | None = None
        self._done = False
        self._cancelled = False
        self._process = QProcess(self)
        self._process.readyReadStandardOutput.connect(self._read_stdout)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(self._on_error)

    @property
    def running(self) -> bool:
        """Whether the child process is still running."""
        return self._process.state() != QProcess.ProcessState.NotRunning

    def start(self) -> None:
        """Launch the mesher process."""
        self._tmpdir = Path(tempfile.mkdtemp(prefix="kits-mesh-"))
        args = ["-m", "kits.mesh.cli", str(self._project_path), "--progress", "--npz", str(self._npz_path)]
        self._process.start(sys.executable, args)

    def cancel(self) -> None:
        """Kill the mesher process; :attr:`cancelled` is emitted once it has exited."""
        if self.running:
            self._cancelled = True
            self._process.kill()

    @property
    def _npz_path(self) -> Path:
        """Temporary file the child writes the mesh arrays to."""
        return self._tmpdir / "mesh.npz"

    def _read_stdout(self) -> None:
        """Parse complete JSON lines from the child's stdout."""
        self._buffer += bytes(self._process.readAllStandardOutput()).decode("utf-8", "replace")
        *lines, self._buffer = self._buffer.split("\n")
        for line in lines:
            try:
                record = json.loads(line)
            except ValueError:
                continue  # stray non-JSON output (e.g. a library warning)
            if "progress" in record:
                self.progress.emit(str(record.get("message", "")), float(record["progress"]))
            elif "error" in record:
                self._error = str(record["error"])
            elif record.get("done"):
                self._done = True

    def _on_error(self, error: QProcess.ProcessError) -> None:
        """Report a process that couldn't be started.

        Args:
            error: The process error; only start failures are handled here, others
                surface through :meth:`_on_finished`.
        """
        if error == QProcess.ProcessError.FailedToStart:
            self._cleanup()
            self.failed.emit(f"Could not start the mesher ({sys.executable}): {self._process.errorString()}")

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        """Load the result, or report why the child failed.

        Args:
            exit_code: The child's exit code.
            exit_status: Whether the child exited normally or crashed.
        """
        self._read_stdout()
        try:
            if self._cancelled:
                self.cancelled.emit()
            elif self._done and exit_code == 0 and self._npz_path.is_file():
                try:
                    mesh = BoardMesh.load_npz(self._npz_path)
                except (OSError, ValueError, KeyError) as exc:
                    self.failed.emit(f"Could not read the mesh result: {exc}")
                else:
                    self.succeeded.emit(mesh)
            elif self._error:
                self.failed.emit(self._error)
            else:
                stderr = bytes(self._process.readAllStandardError()).decode("utf-8", "replace").strip()
                crashed = exit_status == QProcess.ExitStatus.CrashExit
                how = "crashed" if crashed else f"exited with code {exit_code}"
                detail = f"\n\n{stderr[-_STDERR_TAIL:]}" if stderr else ""
                self.failed.emit(f"The mesher {how}.{detail}")
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        """Remove the temporary directory."""
        if self._tmpdir is not None:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None
