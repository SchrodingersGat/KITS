import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from kits.gui.timed_progress import TimedProgressDialog, format_duration  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize(
    "seconds, text",
    [(0, "0:00"), (7.9, "0:07"), (65, "1:05"), (3599, "59:59"), (3723, "1:02:03"), (-1, "0:00")],
)
def test_format_duration(seconds, text):
    assert format_duration(seconds) == text


def test_dialog_shows_elapsed_time(app):
    dialog = TimedProgressDialog("Solving", "Starting solver…", 1000)
    assert dialog.windowTitle() == "Solving"
    assert dialog.maximum() == 1000 and dialog.value() == 0
    assert dialog._bar.format() == "%p%  ·  0:00"

    dialog._clock = _Fixed(83_000)  # pretend it started 83 s ago
    dialog._show_elapsed()
    assert dialog._bar.format() == "%p%  ·  1:23"
    assert dialog.elapsed_s == pytest.approx(83.0)

    dialog.show()
    assert dialog._tick.isActive()
    dialog.close()
    assert not dialog._tick.isActive()


class _Fixed:
    """Stand-in for QElapsedTimer that reports a fixed elapsed time."""

    def __init__(self, ms: int):
        self._ms = ms

    def elapsed(self) -> int:
        return self._ms
