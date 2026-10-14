"""Main application window."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
)

from .. import __version__
from ..ingest import BoardLoadError, Project
from ..ingest.kicad import BOARD_SUFFIX, PROJECT_SUFFIX
from .board_info import BoardInfoPanel
from .board_scene import BoardScene
from .board_view import BoardView
from .document import Document
from .layer_panel import LayerPanel
from .settings import AppSettings
from .settings_dialog import SettingsDialog

log = logging.getLogger(__name__)

_FILE_FILTER = "KiCad project (*.kicad_pro);;KiCad board (*.kicad_pcb);;All files (*)"
_LAST_DIR_KEY = "paths/last_open_dir"


class MainWindow(QMainWindow):
    """Top-level window: board view, layer and board-info docks, menus and file loading."""

    def __init__(self, document: Document | None = None, settings: AppSettings | None = None):
        """Build the window and wire it to the document and settings.

        Args:
            document: Document to display; a new one owned by the window is created if omitted.
            settings: Application settings; a new QSettings-backed instance is created if omitted.
        """
        super().__init__()
        self.document = document or Document(self)
        self.settings = settings or AppSettings(parent=self)
        self.scene: BoardScene | None = None

        self.view = BoardView()
        self.setCentralWidget(self.view)

        self.layer_panel = LayerPanel()
        self.info_panel = BoardInfoPanel()
        self._layer_dock = self._add_dock("Layers", self.layer_panel, Qt.DockWidgetArea.LeftDockWidgetArea)
        self._info_dock = self._add_dock("Board", self.info_panel, Qt.DockWidgetArea.RightDockWidgetArea)

        self._cursor_label = QLabel()
        self.statusBar().addPermanentWidget(self._cursor_label)

        self._create_actions()
        self._create_menus()

        self.document.project_changed.connect(self._on_project_changed)
        self.layer_panel.visibility_changed.connect(self._on_layer_visibility)
        self.layer_panel.active_layer_changed.connect(self._on_active_layer)
        self.view.cursor_moved.connect(self._on_cursor_moved)
        self.settings.changed.connect(self._apply_settings)
        self.view.set_cursor_visible(self.settings.cursor_visible)

        self.setAcceptDrops(True)
        self.resize(1400, 900)
        self._update_title()

    # -- setup ----------------------------------------------------------------

    def _add_dock(self, title: str, widget, area) -> QDockWidget:
        """Wrap ``widget`` in a dock widget and add it to the window.

        Args:
            title: Dock title; also used to derive its object name for state saving.
            widget: The widget to place in the dock.
            area: The ``Qt.DockWidgetArea`` to dock into.

        Returns:
            The created dock widget.
        """
        dock = QDockWidget(title, self)
        dock.setObjectName(f"dock_{title.lower()}")
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def _create_actions(self) -> None:
        """Create the menu actions; board-dependent ones start disabled."""

        def action(text, slot, shortcut=None, enabled=True):
            """Create a QAction connected to ``slot``.

            Args:
                text: Menu text, with ``&`` marking the mnemonic.
                slot: Callable invoked when the action is triggered.
                shortcut: Optional key sequence, standard key or key.
                enabled: Initial enabled state.

            Returns:
                The new action, parented to the window.
            """
            act = QAction(text, self)
            act.triggered.connect(slot)
            if shortcut is not None:
                act.setShortcut(shortcut)
            act.setEnabled(enabled)
            return act

        self.open_action = action("&Open…", self.open_dialog, QKeySequence.StandardKey.Open)
        self.reload_action = action("&Reload", self.reload, QKeySequence.StandardKey.Refresh, False)
        self.quit_action = action("&Quit", self.close, QKeySequence.StandardKey.Quit)
        keys = QKeySequence.StandardKey
        self.fit_action = action("Zoom to &fit", self.view.zoom_to_fit, Qt.Key.Key_Home, False)
        self.zoom_in_action = action("Zoom &in", self.view.zoom_in, keys.ZoomIn, False)
        self.zoom_out_action = action("Zoom &out", self.view.zoom_out, keys.ZoomOut, False)
        self.grid_action = QAction("Show &grid", self)
        self.grid_action.setCheckable(True)
        self.grid_action.setChecked(self.settings.grid_visible)
        self.grid_action.setShortcut(Qt.Key.Key_G)
        self.grid_action.toggled.connect(self._on_grid_toggled)
        self.cursor_action = QAction("Show &crosshair cursor", self)
        self.cursor_action.setCheckable(True)
        self.cursor_action.setChecked(self.settings.cursor_visible)
        self.cursor_action.toggled.connect(self._on_cursor_toggled)
        self.preferences_action = action("&Preferences…", self.open_preferences, "Ctrl+,")
        self._board_actions = [self.reload_action, self.fit_action, self.zoom_in_action, self.zoom_out_action]

    def _create_menus(self) -> None:
        """Populate the File, Edit and View menus."""
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.reload_action)
        file_menu.addSeparator()
        file_menu.addAction(self.quit_action)

        edit_menu = self.menuBar().addMenu("&Edit")
        edit_menu.addAction(self.preferences_action)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.fit_action)
        view_menu.addAction(self.zoom_in_action)
        view_menu.addAction(self.zoom_out_action)
        view_menu.addSeparator()
        view_menu.addAction(self.grid_action)
        view_menu.addAction(self.cursor_action)
        view_menu.addSeparator()
        view_menu.addAction(self._layer_dock.toggleViewAction())
        view_menu.addAction(self._info_dock.toggleViewAction())

    # -- settings --------------------------------------------------------------

    def open_preferences(self) -> None:
        """Show the modal preferences dialog."""
        SettingsDialog(self.settings, self).exec()

    def _on_grid_toggled(self, checked: bool) -> None:
        """Persist the grid visibility chosen from the View menu.

        Args:
            checked: New state of the "Show grid" action.
        """
        self.settings.grid_visible = checked

    def _on_cursor_toggled(self, checked: bool) -> None:
        """Persist the crosshair visibility chosen from the View menu.

        Args:
            checked: New state of the "Show crosshair cursor" action.
        """
        self.settings.cursor_visible = checked

    def _apply_settings(self) -> None:
        """Sync the menu actions, view and scene with the current settings."""
        self.grid_action.setChecked(self.settings.grid_visible)
        self.cursor_action.setChecked(self.settings.cursor_visible)
        self.view.set_cursor_visible(self.settings.cursor_visible)
        if self.scene is not None:
            self.scene.set_grid(self.settings.grid_spacing_mm, self.settings.grid_visible)

    # -- file handling ---------------------------------------------------------

    def open_dialog(self) -> None:
        """Ask the user for a project or board file and open it, remembering its directory."""
        settings = QSettings()
        start_dir = settings.value(_LAST_DIR_KEY, str(Path.home()))
        path, _ = QFileDialog.getOpenFileName(self, "Open KiCad project", start_dir, _FILE_FILTER)
        if path:
            settings.setValue(_LAST_DIR_KEY, str(Path(path).parent))
            self.open_path(path)

    def open_path(self, path: str | Path) -> bool:
        """Open a project or board file, reporting errors in a message box.

        Args:
            path: A ``.kicad_pro`` or ``.kicad_pcb`` file.

        Returns:
            True if the file loaded, False if an error was shown.
        """
        return self._load(lambda: self.document.open(path), f"Loading {Path(path).name}…")

    def reload(self) -> bool:
        """Reload the open board from disk, reporting errors in a message box.

        Returns:
            True if the reload succeeded, False if an error was shown.
        """
        return self._load(self.document.reload, "Reloading…")

    def _load(self, loader, message: str) -> bool:
        """Run ``loader`` with a busy cursor and status message, showing any error to the user.

        Both :class:`BoardLoadError` and unexpected exceptions are caught and reported
        rather than propagated, so the GUI keeps running.

        Args:
            loader: Zero-argument callable that loads into the document.
            message: Status-bar text shown while loading.

        Returns:
            True on success, False if loading failed.
        """
        self.statusBar().showMessage(message)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            loader()
        except BoardLoadError as exc:
            QApplication.restoreOverrideCursor()
            self.statusBar().clearMessage()
            QMessageBox.critical(self, "Could not open board", str(exc))
            return False
        except Exception as exc:  # unexpected bug: report it rather than crash
            log.exception("Unexpected error while loading board")
            QApplication.restoreOverrideCursor()
            self.statusBar().clearMessage()
            QMessageBox.critical(
                self, "Could not open board", f"Unexpected error while loading the board:\n{exc!r}"
            )
            return False
        QApplication.restoreOverrideCursor()
        return True

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Accept drags carrying a KiCad project or board file.

        Args:
            event: The drag-enter event.
        """
        if self._dropped_path(event) is not None:
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        """Open a dropped KiCad project or board file.

        Args:
            event: The drop event.
        """
        path = self._dropped_path(event)
        if path is not None:
            event.acceptProposedAction()
            self.open_path(path)

    @staticmethod
    def _dropped_path(event) -> Path | None:
        """Find the first KiCad project or board file among a drag/drop event's URLs.

        Args:
            event: A drag-enter or drop event.

        Returns:
            Path of the first ``.kicad_pro``/``.kicad_pcb`` file, or ``None`` if there is none.
        """
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix in (PROJECT_SUFFIX, BOARD_SUFFIX):
                return path
        return None

    # -- document updates ------------------------------------------------------

    def _on_project_changed(self, project: Project | None) -> None:
        """Rebuild the scene and panels for a newly opened (or closed) project.

        Args:
            project: The new project, or ``None`` when the document was closed.
        """
        old_scene = self.scene
        self.scene = BoardScene(project.board, self) if project else None
        self._apply_settings()
        self.view.setScene(self.scene)
        if old_scene is not None:
            old_scene.deleteLater()

        if self.scene is not None:
            self.layer_panel.set_layers(
                [(lay.name, lay.color, lay.is_copper) for lay in self.scene.layers.values()]
            )
            self.view.fit_to_rect(self.scene.board_rect())
            b = project.board
            if b.has_outline:
                self.statusBar().showMessage(
                    f"Loaded {b.path.name}: {len(b.copper_layers)} copper layers, "
                    f"{len(b.footprints)} footprints, {len(b.tracks) + len(b.arcs)} tracks",
                    5000,
                )
            else:
                # Left up until the next message: the board can't be simulated like this.
                self.statusBar().showMessage(
                    f"Warning: {b.path.name} has no board outline — nothing is drawn on Edge.Cuts"
                )
        else:
            self.layer_panel.set_layers([])

        self.info_panel.set_project(project)
        for act in self._board_actions:
            act.setEnabled(project is not None)
        self._update_title()

    def _on_layer_visibility(self, name: str, visible: bool) -> None:
        """Show or hide a layer in the scene.

        Args:
            name: KiCad layer name.
            visible: True to show the layer.
        """
        if self.scene is not None:
            self.scene.set_layer_visible(name, visible)

    def _on_active_layer(self, name: str) -> None:
        """Make a copper layer the active (topmost) layer in the scene.

        Args:
            name: KiCad layer name.
        """
        if self.scene is not None:
            self.scene.set_active_layer(name)

    def _on_cursor_moved(self, x: float, y: float) -> None:
        """Show the cursor position, relative to the grid origin, in the status bar.

        Args:
            x: Cursor X in KiCad board coordinates (mm).
            y: Cursor Y in KiCad board coordinates (mm).
        """
        if self.scene is not None:
            gx, gy = self.scene.to_grid_coords(x, y)
            self._cursor_label.setText(f"X {gx:9.3f} mm   Y {gy:9.3f} mm")
            self._cursor_label.setToolTip(f"Relative to grid origin\nKiCad: X {x:.3f}  Y {y:.3f} mm")

    def _update_title(self) -> None:
        """Set the window title from the open project's name and the app version."""
        project = self.document.project
        app_title = f"KITS {__version__}"
        self.setWindowTitle(f"{project.name} — {app_title}" if project else app_title)
