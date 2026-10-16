"""Run a KITS command-line tool in a child process and report its progress.

Running gmsh and the solver out of process keeps the GUI responsive, lets the
user cancel (the process is killed), and means a crash only fails the job.
The child speaks the JSON-lines protocol of ``kits-mesh`` / ``kits-solve``
(``--progress``): progress records, then a ``done`` or ``error`` record.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, Signal

_STDERR_TAIL = 2000  # characters of child stderr included in error messages


class ProcessJob(QObject):
    """One run of a ``python -m <module> <project> --progress`` child process.

    Signals:
        progress(str, float): Step description and overall fraction (0-1).
        succeeded(object): The child finished and its result file was loaded.
        failed(str): The run failed; the argument explains why.
        cancelled(): The job was cancelled by :meth:`cancel`.
    """

    progress = Signal(str, float)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    #: Module run with ``python -m``.
    module = ""
    #: Key of the ``done`` record holding the result file path.
    result_key = ""
    #: What the job does, for error messages (e.g. "mesher").
    what = "job"

    def __init__(self, project_path: str | Path, parent=None):
        """Prepare (but don't start) a job.

        Args:
            project_path: The ``.kits`` project. The child reads everything from the
                file, so save any edits before starting.
            parent: Optional Qt parent object.
        """
        super().__init__(parent)
        self._project_path = Path(project_path)
        self._buffer = ""
        self._error: str | None = None
        self._done: dict | None = None
        self._cancelled = False
        self._process = QProcess(self)
        self._process.readyReadStandardOutput.connect(self._read_stdout)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(self._on_error)

    @property
    def running(self) -> bool:
        """Whether the child process is still running."""
        return self._process.state() != QProcess.ProcessState.NotRunning

    @property
    def done_record(self) -> dict | None:
        """The child's final ``done`` record, once it has finished successfully."""
        return self._done

    def arguments(self) -> list[str]:
        """Extra command-line arguments for the child, after the project and ``--progress``.

        Returns:
            None by default; subclasses add their own.
        """
        return []

    def start(self) -> None:
        """Launch the child process."""
        self._process.start(
            sys.executable, ["-m", self.module, str(self._project_path), "--progress", *self.arguments()]
        )

    def cancel(self) -> None:
        """Kill the child process; :attr:`cancelled` is emitted once it has exited."""
        if self.running:
            self._cancelled = True
            self._process.kill()

    def load_result(self, path: Path) -> object:
        """Load the result file named by the ``done`` record.

        Args:
            path: The result file.

        Returns:
            The loaded result, emitted with :attr:`succeeded`.
        """
        raise NotImplementedError

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
                self._done = record

    def _on_error(self, error: QProcess.ProcessError) -> None:
        """Report a process that couldn't be started.

        Args:
            error: The process error; only start failures are handled here, others
                surface through :meth:`_on_finished`.
        """
        if error == QProcess.ProcessError.FailedToStart:
            self.failed.emit(
                f"Could not start the {self.what} ({sys.executable}): {self._process.errorString()}"
            )

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        """Load the result, or report why the child failed.

        Args:
            exit_code: The child's exit code.
            exit_status: Whether the child exited normally or crashed.
        """
        self._read_stdout()
        result_path = self._done.get(self.result_key) if self._done else None
        if self._cancelled:
            self.cancelled.emit()
        elif self._done and exit_code == 0 and result_path:
            try:
                result = self.load_result(Path(result_path))
            except (OSError, ValueError, KeyError) as exc:
                self.failed.emit(f"Could not read the {self.what} result: {exc}")
            else:
                self.succeeded.emit(result)
        elif self._error:
            self.failed.emit(self._error)
        else:
            stderr = bytes(self._process.readAllStandardError()).decode("utf-8", "replace").strip()
            crashed = exit_status == QProcess.ExitStatus.CrashExit
            how = "crashed" if crashed else f"exited with code {exit_code}"
            detail = f"\n\n{stderr[-_STDERR_TAIL:]}" if stderr else ""
            self.failed.emit(f"The {self.what} {how}.{detail}")
