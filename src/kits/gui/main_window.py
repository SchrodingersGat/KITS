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

log = logging.getLogger(__name__)

_FILE_FILTER = "KiCad project (*.kicad_pro);;KiCad board (*.kicad_pcb);;All files (*)"
_LAST_DIR_KEY = "paths/last_open_dir"


class MainWindow(QMainWindow):
    def __init__(self, document: Document | None = None):
        super().__init__()
        self.document = document or Document(self)
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

        self.setAcceptDrops(True)
        self.resize(1400, 900)
        self._update_title()

    # -- setup ----------------------------------------------------------------

    def _add_dock(self, title: str, widget, area) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(f"dock_{title.lower()}")
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def _create_actions(self) -> None:
        def action(text, slot, shortcut=None, enabled=True):
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
        self._board_actions = [self.reload_action, self.fit_action, self.zoom_in_action, self.zoom_out_action]

    def _create_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.reload_action)
        file_menu.addSeparator()
        file_menu.addAction(self.quit_action)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.fit_action)
        view_menu.addAction(self.zoom_in_action)
        view_menu.addAction(self.zoom_out_action)
        view_menu.addSeparator()
        view_menu.addAction(self._layer_dock.toggleViewAction())
        view_menu.addAction(self._info_dock.toggleViewAction())

    # -- file handling ---------------------------------------------------------

    def open_dialog(self) -> None:
        settings = QSettings()
        start_dir = settings.value(_LAST_DIR_KEY, str(Path.home()))
        path, _ = QFileDialog.getOpenFileName(self, "Open KiCad project", start_dir, _FILE_FILTER)
        if path:
            settings.setValue(_LAST_DIR_KEY, str(Path(path).parent))
            self.open_path(path)

    def open_path(self, path: str | Path) -> bool:
        return self._load(lambda: self.document.open(path), f"Loading {Path(path).name}…")

    def reload(self) -> bool:
        return self._load(self.document.reload, "Reloading…")

    def _load(self, loader, message: str) -> bool:
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
            QMessageBox.critical(self, "Could not open board",
                                 f"Unexpected error while loading the board:\n{exc!r}")
            return False
        QApplication.restoreOverrideCursor()
        return True

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if self._dropped_path(event) is not None:
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        path = self._dropped_path(event)
        if path is not None:
            event.acceptProposedAction()
            self.open_path(path)

    @staticmethod
    def _dropped_path(event) -> Path | None:
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix in (PROJECT_SUFFIX, BOARD_SUFFIX):
                return path
        return None

    # -- document updates ------------------------------------------------------

    def _on_project_changed(self, project: Project | None) -> None:
        old_scene = self.scene
        self.scene = BoardScene(project.board, self) if project else None
        self.view.setScene(self.scene)
        if old_scene is not None:
            old_scene.deleteLater()

        if self.scene is not None:
            self.layer_panel.set_layers(
                [(lay.name, lay.color, lay.is_copper) for lay in self.scene.layers.values()]
            )
            self.view.fit_to_rect(self.scene.board_rect())
            b = project.board
            self.statusBar().showMessage(
                f"Loaded {b.path.name}: {len(b.copper_layers)} copper layers, "
                f"{len(b.footprints)} footprints, {len(b.tracks) + len(b.arcs)} tracks", 5000)
        else:
            self.layer_panel.set_layers([])

        self.info_panel.set_project(project)
        for act in self._board_actions:
            act.setEnabled(project is not None)
        self._update_title()

    def _on_layer_visibility(self, name: str, visible: bool) -> None:
        if self.scene is not None:
            self.scene.set_layer_visible(name, visible)

    def _on_active_layer(self, name: str) -> None:
        if self.scene is not None:
            self.scene.set_active_layer(name)

    def _on_cursor_moved(self, x: float, y: float) -> None:
        if self.scene is not None:
            self._cursor_label.setText(f"X {x:9.3f} mm   Y {y:9.3f} mm")

    def _update_title(self) -> None:
        project = self.document.project
        app_title = f"KITS {__version__}"
        self.setWindowTitle(f"{project.name} — {app_title}" if project else app_title)
