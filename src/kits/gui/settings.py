"""Persistent application settings (stored via QSettings)."""

from __future__ import annotations

from PySide6.QtCore import QObject, QSettings, Signal

DEFAULT_GRID_SPACING_MM = 10.0
MIN_GRID_SPACING_MM = 0.01
MAX_GRID_SPACING_MM = 1000.0

_GRID_SPACING_KEY = "grid/spacing_mm"
_GRID_VISIBLE_KEY = "grid/visible"
_CURSOR_VISIBLE_KEY = "cursor/visible"


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
    """Typed access to user preferences; emits ``changed`` when any value changes."""

    changed = Signal()

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
