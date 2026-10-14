"""Persistent application settings (stored via QSettings)."""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QObject, QSettings, Signal

DEFAULT_GRID_SPACING_MM = 10.0
MIN_GRID_SPACING_MM = 0.01
MAX_GRID_SPACING_MM = 1000.0

_GRID_SPACING_KEY = "grid/spacing_mm"
_GRID_VISIBLE_KEY = "grid/visible"
_CURSOR_VISIBLE_KEY = "cursor/visible"
_RECENT_KEY = "projects/recent"
MAX_RECENT_PROJECTS = 10


def _same_path(a: Path, b: Path) -> bool:
    """Compare two paths the way the file system would (case-insensitive on Windows).

    Args:
        a: First path.
        b: Second path.

    Returns:
        True if they name the same file.
    """
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def _to_bool(value) -> bool:
    """Coerce a stored QSettings value to a bool.

    Args:
        value: Raw value from QSettings: a bool, number or string such as ``"true"``.

    Returns:
        The value interpreted as a boolean.
    """
    # QSettings' INI/registry backends may hand back "true"/"false" strings.
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes")
    return bool(value)


class AppSettings(QObject):
    """Typed access to user preferences.

    Signals:
        changed(): A display preference (grid, cursor) changed.
        recent_changed(): The recent-projects list changed.
    """

    changed = Signal()
    recent_changed = Signal()

    def __init__(self, store: QSettings | None = None, parent=None):
        """Wrap a QSettings store.

        Args:
            store: Backing store; defaults to the application's ``QSettings()``.
            parent: Optional Qt parent object.
        """
        super().__init__(parent)
        self._store = store if store is not None else QSettings()

    @property
    def grid_spacing_mm(self) -> float:
        """Grid spacing in millimetres; the default if unset, invalid or out of range."""
        try:
            value = float(self._store.value(_GRID_SPACING_KEY, DEFAULT_GRID_SPACING_MM))
        except (TypeError, ValueError):
            return DEFAULT_GRID_SPACING_MM
        if not MIN_GRID_SPACING_MM <= value <= MAX_GRID_SPACING_MM:
            return DEFAULT_GRID_SPACING_MM
        return value

    @grid_spacing_mm.setter
    def grid_spacing_mm(self, value: float) -> None:
        """Store the grid spacing, clamped to the allowed range.

        Args:
            value: New spacing in millimetres.
        """
        value = min(max(float(value), MIN_GRID_SPACING_MM), MAX_GRID_SPACING_MM)
        if value != self.grid_spacing_mm:
            self._store.setValue(_GRID_SPACING_KEY, value)
            self.changed.emit()

    @property
    def grid_visible(self) -> bool:
        """Whether the board grid is drawn."""
        return self._get_bool(_GRID_VISIBLE_KEY, True)

    @grid_visible.setter
    def grid_visible(self, value: bool) -> None:
        """Store whether the board grid is drawn.

        Args:
            value: True to show the grid.
        """
        self._set_bool(_GRID_VISIBLE_KEY, value, True)

    @property
    def cursor_visible(self) -> bool:
        """Whether the full-view crosshair follows the mouse."""
        return self._get_bool(_CURSOR_VISIBLE_KEY, True)

    @cursor_visible.setter
    def cursor_visible(self, value: bool) -> None:
        """Store whether the crosshair cursor is shown.

        Args:
            value: True to show the crosshair.
        """
        self._set_bool(_CURSOR_VISIBLE_KEY, value, True)

    @property
    def recent_projects(self) -> list[Path]:
        """Recently opened project files, most recent first (missing files included)."""
        value = self._store.value(_RECENT_KEY, [])
        if isinstance(value, str):  # QSettings collapses one-item lists to a string
            value = [value] if value else []
        return [Path(p) for p in value if isinstance(p, str) and p][:MAX_RECENT_PROJECTS]

    def add_recent_project(self, path: str | Path) -> None:
        """Move (or add) a project file to the top of the recent list.

        Args:
            path: The project file.
        """
        path = Path(path).absolute()
        recent = [p for p in self.recent_projects if not _same_path(p, path)]
        self._set_recent([path, *recent])

    def remove_recent_project(self, path: str | Path) -> None:
        """Drop a project file from the recent list.

        Args:
            path: The project file.
        """
        self._set_recent([p for p in self.recent_projects if not _same_path(p, Path(path))])

    def clear_recent_projects(self) -> None:
        """Empty the recent list."""
        self._set_recent([])

    def _set_recent(self, paths: list[Path]) -> None:
        """Store the recent list and emit ``recent_changed`` if it changed.

        Args:
            paths: Project files, most recent first.
        """
        paths = paths[:MAX_RECENT_PROJECTS]
        if paths != self.recent_projects:
            self._store.setValue(_RECENT_KEY, [str(p) for p in paths])
            self.recent_changed.emit()

    def _get_bool(self, key: str, default: bool) -> bool:
        """Read a boolean setting.

        Args:
            key: QSettings key.
            default: Value returned when the key is unset.

        Returns:
            The stored value coerced to bool.
        """
        return _to_bool(self._store.value(key, default))

    def _set_bool(self, key: str, value: bool, default: bool) -> None:
        """Store a boolean setting, emitting ``changed`` only if the value differs.

        Args:
            key: QSettings key.
            value: New value.
            default: Value assumed for the key when it is unset.
        """
        if bool(value) != self._get_bool(key, default):
            self._store.setValue(key, bool(value))
            self.changed.emit()
