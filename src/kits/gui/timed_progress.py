"""A cancellable progress dialog that also shows how long the job has been running."""

from __future__ import annotations

from PySide6.QtCore import QElapsedTimer, Qt, QTimer
from PySide6.QtGui import QHideEvent
from PySide6.QtWidgets import QProgressBar, QProgressDialog, QWidget

_TICK_MS = 1000  # elapsed-time refresh interval


def format_duration(seconds: float) -> str:
    """Format a duration as ``m:ss``, or ``h:mm:ss`` from an hour.

    Args:
        seconds: The duration (negative counts as zero).

    Returns:
        The formatted duration, e.g. ``"1:05"`` or ``"1:02:03"``.
    """
    total = max(int(seconds), 0)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


class TimedProgressDialog(QProgressDialog):
    """Window-modal progress dialog whose bar shows the percentage and the elapsed time.

    The time ticks every second, so it keeps moving through long steps that
    report no progress (e.g. a direct sparse solve).
    """

    def __init__(self, title: str, label: str, steps: int, parent: QWidget | None = None):
        """Create and start timing the dialog (show it with :meth:`show`).

        Args:
            title: Window title.
            label: Initial step description.
            steps: Range of the progress bar (0 to ``steps``).
            parent: Parent widget.
        """
        super().__init__(label, "Cancel", 0, steps, parent)
        self._bar = QProgressBar(self)
        self._bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setBar(self._bar)
        self.setRange(0, steps)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setMinimumDuration(0)
        self.setAutoClose(False)
        self.setAutoReset(False)
        self.setValue(0)
        self._clock = QElapsedTimer()
        self._clock.start()
        self._tick = QTimer(self)
        self._tick.timeout.connect(self._show_elapsed)
        self._tick.start(_TICK_MS)
        self._show_elapsed()

    @property
    def elapsed_s(self) -> float:
        """Seconds since the dialog was created."""
        return self._clock.elapsed() / 1000

    def _show_elapsed(self) -> None:
        """Put the elapsed time in the bar's text, after the percentage."""
        self._bar.setFormat(f"%p%  ·  {format_duration(self.elapsed_s)}")

    def hideEvent(self, event: QHideEvent) -> None:  # noqa: N802 (Qt override)
        """Stop ticking once the dialog is closed.

        Args:
            event: The hide event.
        """
        self._tick.stop()
        super().hideEvent(event)
