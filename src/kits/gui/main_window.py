"""Main application window."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QDragEnterEvent, QDropEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QScrollArea,
    QStackedWidget,
)

from .. import __version__
from ..config import ConfigError
from ..ingest import BoardLoadError, Project
from ..ingest.kicad import BOARD_SUFFIX
from ..ingest.kicad import PROJECT_SUFFIX as KICAD_PROJECT_SUFFIX
from ..mesh.model import BoardMesh
from ..project import PROJECT_SUFFIX, KitsProject, default_project_path
from .board_info import BoardInfoPanel
from .board_scene import BoardScene
from .board_view import BoardView
from .document import Document
from .layer_panel import LayerPanel
from .mesh_job import MeshJob
from .mesh_panel import MeshPanel
from .settings import AppSettings
from .settings_dialog import SettingsDialog
from .solve_job import SolveJob
from .solve_panel import SolvePanel
from .sources_panel import SourcesPanel
from .start_page import NoBoardPage, StartPage

log = logging.getLogger(__name__)

_KITS_FILTER = f"KITS project (*{PROJECT_SUFFIX})"
_KICAD_FILTER = f"KiCad project or board (*{KICAD_PROJECT_SUFFIX} *{BOARD_SUFFIX})"
_KICAD_SUFFIXES = (KICAD_PROJECT_SUFFIX, BOARD_SUFFIX)
_LAST_DIR_KEY = "paths/last_open_dir"
_PROGRESS_STEPS = 1000


class MainWindow(QMainWindow):
    """Top-level window: project lifecycle, board view, docks, menus and meshing."""

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
        self._mesh_job: MeshJob | None = None
        self._solve_job: SolveJob | None = None
        self._solve_progress: QProgressDialog | None = None
        self._mesh_progress: QProgressDialog | None = None

        self.view = BoardView()
        self.start_page = StartPage()
        self.no_board_page = NoBoardPage()
        self._pages = QStackedWidget()
        for page in (self.start_page, self.no_board_page, self.view):
            self._pages.addWidget(page)
        self.setCentralWidget(self._pages)

        self.layer_panel = LayerPanel()
        self.info_panel = BoardInfoPanel()
        self.mesh_panel = MeshPanel(self.document)
        self.sources_panel = SourcesPanel(self.document)
        self.solve_panel = SolvePanel(self.document)
        self._selected_pad: tuple[str, str] | None = None
        self._layer_dock = self._add_dock("Layers", self.layer_panel, Qt.DockWidgetArea.LeftDockWidgetArea)
        self._info_dock = self._add_dock("Board", self.info_panel, Qt.DockWidgetArea.RightDockWidgetArea)
        self._mesh_dock = self._add_dock("Mesh", self.mesh_panel, Qt.DockWidgetArea.RightDockWidgetArea)
        self._sources_dock = self._add_dock(
            "Current Sources", self.sources_panel, Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.tabifyDockWidget(self._info_dock, self._sources_dock)
        self._sources_dock.raise_()
        # Scrollable: the Solve tab is tall, and a dock's minimum height would
        # otherwise set the whole window's minimum size.
        solve_scroll = QScrollArea()
        solve_scroll.setWidget(self.solve_panel)
        solve_scroll.setWidgetResizable(True)
        solve_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._solve_dock = self._add_dock("Solve", solve_scroll, Qt.DockWidgetArea.RightDockWidgetArea)
        self.tabifyDockWidget(self._mesh_dock, self._solve_dock)
        self._mesh_dock.raise_()

        self._cursor_label = QLabel()
        self.statusBar().addPermanentWidget(self._cursor_label)

        self._create_actions()
        self._create_menus()

        self.document.project_changed.connect(self._on_project_changed)
        self.document.board_changed.connect(self._on_board_changed)
        self.document.dirty_changed.connect(self._update_title)
        self.document.mesh_changed.connect(self._on_mesh_changed)
        self.document.results_changed.connect(self._on_results_changed)
        self.solve_panel.solve_requested.connect(self.solve)
        self.solve_panel.mode_changed.connect(self._on_display_mode_changed)
        self.solve_panel.show_results_toggled.connect(self.show_results_action.setChecked)
        self.mesh_panel.build_requested.connect(self.build_mesh)
        self.mesh_panel.show_mesh_toggled.connect(self.show_mesh_action.setChecked)
        self.layer_panel.visibility_changed.connect(self._on_layer_visibility)
        self.layer_panel.active_layer_changed.connect(self._on_active_layer)
        self.view.cursor_moved.connect(self._on_cursor_moved)
        self.view.clicked.connect(self._on_board_clicked)
        self.view.double_clicked.connect(self._on_board_double_clicked)
        self.sources_panel.pad_selected.connect(self._on_source_row_selected)
        self.document.sources_changed.connect(self._show_sources)
        self.start_page.new_requested.connect(self.new_project)
        self.start_page.open_requested.connect(self.open_project_dialog)
        self.start_page.recent_requested.connect(self.open_path)
        self.no_board_page.link_requested.connect(self.link_board_dialog)
        self.settings.changed.connect(self._apply_settings)
        # Deferred: the change may come from a recent-list click whose widget the
        # refresh would otherwise delete while its signal is still being handled.
        self.settings.recent_changed.connect(lambda: QTimer.singleShot(0, self._update_recent))
        self.view.set_cursor_visible(self.settings.cursor_visible)

        self.setAcceptDrops(True)
        self.resize(1400, 900)
        self._update_recent()
        self._update_state()

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
        """Create the menu actions; project- and board-dependent ones start disabled."""

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
            act.triggered.connect(lambda _checked=False: slot())
            if shortcut is not None:
                act.setShortcut(shortcut)
            act.setEnabled(enabled)
            return act

        def toggle(text, checked, slot, shortcut=None):
            """Create a checkable QAction.

            Args:
                text: Menu text, with ``&`` marking the mnemonic.
                checked: Initial checked state.
                slot: Callable receiving the new checked state.
                shortcut: Optional key.

            Returns:
                The new action, parented to the window.
            """
            act = QAction(text, self)
            act.setCheckable(True)
            act.setChecked(checked)
            if shortcut is not None:
                act.setShortcut(shortcut)
            act.toggled.connect(slot)
            return act

        keys = QKeySequence.StandardKey
        self.new_project_action = action("&New Project…", self.new_project, keys.New)
        self.open_project_action = action("&Open Project…", self.open_project_dialog, keys.Open)
        self.save_action = action("&Save", self.save, keys.Save, False)
        self.save_as_action = action("Save &As…", self.save_as_dialog, keys.SaveAs, False)
        self.close_project_action = action("&Close Project", self.close_project, keys.Close, False)
        self.quit_action = action("&Quit", self.close, keys.Quit)
        self.link_board_action = action("&Link KiCad Board…", self.link_board_dialog, None, False)
        self.reload_action = action("&Reload Board", self.reload_board, keys.Refresh, False)
        self.preferences_action = action("&Preferences…", self.open_preferences, "Ctrl+,")
        self.fit_action = action("Zoom to &fit", self.view.zoom_to_fit, Qt.Key.Key_Home, False)
        self.zoom_in_action = action("Zoom &in", self.view.zoom_in, keys.ZoomIn, False)
        self.zoom_out_action = action("Zoom &out", self.view.zoom_out, keys.ZoomOut, False)
        self.grid_action = toggle(
            "Show &grid", self.settings.grid_visible, self._on_grid_toggled, Qt.Key.Key_G
        )
        self.cursor_action = toggle(
            "Show &crosshair cursor", self.settings.cursor_visible, self._on_cursor_toggled
        )
        self.build_mesh_action = action("&Build mesh", self.build_mesh, "Ctrl+B", False)
        self.show_mesh_action = toggle("Show &mesh", True, self._on_show_mesh_toggled, Qt.Key.Key_M)
        self.solve_action = action("&Solve", self.solve, Qt.Key.Key_F9, False)
        self.show_results_action = toggle("Show &results", True, self._on_show_results_toggled, Qt.Key.Key_R)
        self._project_actions = [
            self.save_action,
            self.save_as_action,
            self.close_project_action,
            self.link_board_action,
        ]
        self._board_actions = [
            self.reload_action,
            self.fit_action,
            self.zoom_in_action,
            self.zoom_out_action,
            self.build_mesh_action,
            self.solve_action,
        ]

    def _create_menus(self) -> None:
        """Populate the File, Edit, Project, View and Mesh menus."""
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.new_project_action)
        file_menu.addAction(self.open_project_action)
        self.recent_menu = QMenu("Open &Recent", self)
        self.recent_menu.aboutToShow.connect(self._populate_recent_menu)
        file_menu.addMenu(self.recent_menu)
        file_menu.addSeparator()
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.save_as_action)
        file_menu.addSeparator()
        file_menu.addAction(self.close_project_action)
        file_menu.addAction(self.quit_action)

        edit_menu = self.menuBar().addMenu("&Edit")
        edit_menu.addAction(self.preferences_action)

        project_menu = self.menuBar().addMenu("&Project")
        project_menu.addAction(self.link_board_action)
        project_menu.addAction(self.reload_action)

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
        view_menu.addAction(self._mesh_dock.toggleViewAction())
        view_menu.addAction(self._sources_dock.toggleViewAction())
        view_menu.addAction(self._solve_dock.toggleViewAction())

        mesh_menu = self.menuBar().addMenu("&Mesh")
        mesh_menu.addAction(self.build_mesh_action)
        mesh_menu.addAction(self.show_mesh_action)

        solve_menu = self.menuBar().addMenu("&Solve")
        solve_menu.addAction(self.solve_action)
        solve_menu.addAction(self.show_results_action)

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

    # -- project lifecycle -------------------------------------------------------

    def new_project(self) -> bool:
        """Create a project: choose where to save the ``.kits`` file, then the KiCad board.

        Returns:
            True if a project was created and opened.
        """
        if not self.maybe_save():
            return False
        path, _ = QFileDialog.getSaveFileName(self, "New KITS project", self._last_dir(), _KITS_FILTER)
        if not path:
            return False
        path = Path(path).with_suffix(PROJECT_SUFFIX)
        board, _ = QFileDialog.getOpenFileName(
            self, "Choose the KiCad board for this project", str(path.parent), _KICAD_FILTER
        )
        if not board:
            return False
        return self._create_project(path, Path(board))

    def open_project_dialog(self) -> bool:
        """Ask for a project file (or a KiCad file to create a project for) and open it.

        Returns:
            True if a project was opened.
        """
        if not self.maybe_save():
            return False
        path, _ = QFileDialog.getOpenFileName(
            self, "Open KITS project", self._last_dir(), f"{_KITS_FILTER};;{_KICAD_FILTER}"
        )
        return bool(path) and self.open_path(path, confirm_unsaved=False)

    def open_path(self, path: str | Path, confirm_unsaved: bool = True) -> bool:
        """Open a ``.kits`` project, or offer to create one for a KiCad file.

        Args:
            path: A ``.kits``, ``.kicad_pro`` or ``.kicad_pcb`` file.
            confirm_unsaved: Ask about unsaved changes in the open project first.

        Returns:
            True if a project was opened (or created and opened).
        """
        path = Path(path)
        if confirm_unsaved and not self.maybe_save():
            return False
        if path.suffix in _KICAD_SUFFIXES:
            return self._open_kicad_file(path)
        if path.suffix != PROJECT_SUFFIX:
            message = f"{path.name} is not a KITS project ({PROJECT_SUFFIX}) or a KiCad file."
            QMessageBox.warning(self, "Cannot open file", message)
            return False
        if not path.is_file():
            self.settings.remove_recent_project(path)
            QMessageBox.warning(self, "Cannot open project", f"Project file not found:\n{path}")
            return False
        if not self._run(lambda: self.document.open_project(path), f"Opening {path.name}…", "open project"):
            return False
        self._remember(path)
        if self.document.board_error:
            self.statusBar().showMessage(f"Opened {path.name}, but its board couldn't be loaded", 8000)
        return True

    def save(self) -> bool:
        """Save the open project to its file.

        Returns:
            True if it was saved.
        """
        if self.document.project is None:
            return False
        if not self._run(self.document.save, "Saving…", "save project"):
            return False
        self._remember(self.document.project.path)
        self.statusBar().showMessage(f"Saved {self.document.project.path.name}", 3000)
        return True

    def save_as_dialog(self) -> bool:
        """Save the open project under a new name.

        Returns:
            True if it was saved.
        """
        project = self.document.project
        if project is None:
            return False
        path, _ = QFileDialog.getSaveFileName(self, "Save KITS project as", str(project.path), _KITS_FILTER)
        if not path:
            return False
        path = Path(path).with_suffix(PROJECT_SUFFIX)
        if not self._run(lambda: self.document.save_as(path), "Saving…", "save project"):
            return False
        self._remember(path)
        return True

    def close_project(self) -> bool:
        """Close the open project, asking about unsaved changes.

        Returns:
            True if the project was closed (or none was open).
        """
        if not self.maybe_save():
            return False
        self._cancel_jobs()
        self.document.close()
        return True

    def maybe_save(self) -> bool:
        """If the project has unsaved changes, ask whether to save them.

        Returns:
            False if the user cancelled (the caller should abort), True otherwise.
        """
        if not self.document.dirty:
            return True
        answer = self._ask_save_changes()
        if answer == QMessageBox.StandardButton.Save:
            return self.save()
        return answer == QMessageBox.StandardButton.Discard

    def _ask_save_changes(self) -> QMessageBox.StandardButton:
        """Ask whether to save, discard or keep unsaved changes.

        Returns:
            ``Save``, ``Discard`` or ``Cancel``.
        """
        buttons = QMessageBox.StandardButton
        return QMessageBox.question(
            self,
            "Unsaved changes",
            f"Save changes to {self.document.project.path.name}?",
            buttons.Save | buttons.Discard | buttons.Cancel,
            buttons.Save,
        )

    def _open_kicad_file(self, board: Path) -> bool:
        """Offer to create (or open the existing) project for a KiCad file.

        Args:
            board: A ``.kicad_pro`` or ``.kicad_pcb`` file.

        Returns:
            True if a project was opened.
        """
        target = default_project_path(board)
        if target.is_file():
            buttons = QMessageBox.StandardButton
            answer = QMessageBox.question(
                self,
                "Open KITS project",
                f"KITS opens project files, and a project already exists for {board.name}:\n\n"
                f"{target}\n\nOpen it?",
                buttons.Open | buttons.Cancel,
            )
            return answer == buttons.Open and self.open_path(target, confirm_unsaved=False)
        path = self._ask_create_project(board, target)
        return path is not None and self._create_project(path, board)

    def _ask_create_project(self, board: Path, target: Path) -> Path | None:
        """Ask whether to create a project for a KiCad file, and where.

        Args:
            board: The KiCad file the user opened.
            target: The suggested project file location.

        Returns:
            Where to create the project, or None if the user declined.
        """
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Create KITS project")
        box.setText(f"KITS works with project files. Create a project for {board.name}?")
        box.setInformativeText(f"It will be saved as:\n{target}")
        create = box.addButton("Create", QMessageBox.ButtonRole.AcceptRole)
        elsewhere = box.addButton("Save Elsewhere…", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(create)
        box.exec()
        if box.clickedButton() is create:
            return target
        if box.clickedButton() is elsewhere:
            path, _ = QFileDialog.getSaveFileName(self, "New KITS project", str(target), _KITS_FILTER)
            return Path(path).with_suffix(PROJECT_SUFFIX) if path else None
        return None

    def _create_project(self, path: Path, board: Path) -> bool:
        """Create and open a project linking ``board``.

        Args:
            path: The ``.kits`` file to create.
            board: The KiCad board to link.

        Returns:
            True on success.
        """
        if not self._run(
            lambda: self.document.new_project(path, board), f"Creating {path.name}…", "create project"
        ):
            return False
        self._remember(path)
        return True

    def _remember(self, path: Path) -> None:
        """Record a project in the recent list and as the last-used folder.

        Args:
            path: The project file.
        """
        self.settings.add_recent_project(path)
        QSettings().setValue(_LAST_DIR_KEY, str(Path(path).parent))

    @staticmethod
    def _last_dir() -> str:
        """Folder the last project was opened from or saved to.

        Returns:
            A directory path, defaulting to the home folder.
        """
        return str(QSettings().value(_LAST_DIR_KEY, str(Path.home())))

    # -- board -------------------------------------------------------------------------

    def link_board_dialog(self) -> bool:
        """Ask for a KiCad board and link it to the open project.

        Returns:
            True if a board was linked.
        """
        project = self.document.project
        if project is None:
            return False
        start = str(project.board.parent if project.board else project.path.parent)
        board, _ = QFileDialog.getOpenFileName(self, "Link KiCad board", start, _KICAD_FILTER)
        return bool(board) and self.link_board(board)

    def link_board(self, board: str | Path) -> bool:
        """Link a KiCad board to the open project (unsaved until File > Save).

        Args:
            board: The ``.kicad_pro`` or ``.kicad_pcb`` file.

        Returns:
            True if the board loaded and was linked.
        """
        return self._run(lambda: self.document.link_board(board), "Loading board…", "load board")

    def reload_board(self) -> bool:
        """Reload the linked board from disk.

        Returns:
            True if the reload succeeded.
        """
        return self._run(self.document.reload_board, "Reloading board…", "reload board")

    def _run(self, operation: Callable[[], object], message: str, what: str) -> bool:
        """Run a file operation with a busy cursor, reporting any error to the user.

        Args:
            operation: Zero-argument callable to run.
            message: Status-bar text shown while it runs.
            what: Short description for error titles, e.g. ``"open project"``.

        Returns:
            True on success, False if an error was shown.
        """
        self.statusBar().showMessage(message)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            operation()
        except (BoardLoadError, ConfigError) as exc:
            error = str(exc)
        except Exception as exc:  # unexpected bug: report it rather than crash
            log.exception("Unexpected error trying to %s", what)
            error = f"Unexpected error:\n{exc!r}"
        else:
            error = None
        finally:
            QApplication.restoreOverrideCursor()
        # Clear only our own "busy" text, not messages posted while the operation ran.
        if self.statusBar().currentMessage() == message:
            self.statusBar().clearMessage()
        if error is not None:
            QMessageBox.critical(self, f"Could not {what}", error)
            return False
        return True

    # -- drag and drop -------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Accept drags carrying a KITS project or KiCad file.

        Args:
            event: The drag-enter event.
        """
        if self._dropped_path(event) is not None:
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        """Open a dropped KITS project, or offer to create one for a KiCad file.

        Args:
            event: The drop event.
        """
        path = self._dropped_path(event)
        if path is not None:
            event.acceptProposedAction()
            self.open_path(path)

    @staticmethod
    def _dropped_path(event) -> Path | None:
        """Find the first openable file among a drag/drop event's URLs.

        Args:
            event: A drag-enter or drop event.

        Returns:
            Path of the first ``.kits``/``.kicad_pro``/``.kicad_pcb`` file, or ``None``.
        """
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix in (PROJECT_SUFFIX, *_KICAD_SUFFIXES):
                return path
        return None

    # -- meshing ---------------------------------------------------------------

    def build_mesh(self) -> None:
        """Mesh the board in a child process, showing a cancellable progress dialog.

        The project is saved first, since the mesher reads its board link and
        settings from the project file. If the board file changed on disk since
        it was loaded, it is reloaded so the mesh matches what is displayed.
        """
        doc = self.document
        if doc.project is None or doc.board is None or self._mesh_job or self._solve_job:
            return
        if not doc.board.has_outline:
            QMessageBox.warning(self, "Cannot build mesh", "The board has no Edge.Cuts outline.")
            return
        try:
            doc.set_mesh_settings(self.mesh_panel.settings())
        except ConfigError as exc:
            QMessageBox.warning(self, "Cannot build mesh", f"Invalid mesh settings: {exc}")
            return
        if doc.dirty and not self.save():
            return
        if doc.board_changed_on_disk:
            if not self.reload_board():
                return
            self.statusBar().showMessage("The board changed on disk and was reloaded before meshing.", 5000)

        job = MeshJob(doc.project.path, self)
        dialog = QProgressDialog("Starting mesher…", "Cancel", 0, _PROGRESS_STEPS, self)
        dialog.setWindowTitle("Building mesh")
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setValue(0)
        dialog.canceled.connect(job.cancel)
        job.progress.connect(self._on_mesh_progress)
        job.succeeded.connect(self._on_mesh_built)
        job.failed.connect(self._on_mesh_failed)
        job.cancelled.connect(self._on_mesh_cancelled)

        self._mesh_job, self._mesh_progress = job, dialog
        self._update_state()
        job.start()
        dialog.show()

    @property
    def mesh_job_running(self) -> bool:
        """Whether a mesh build is in progress."""
        return self._mesh_job is not None

    def _on_mesh_progress(self, message: str, fraction: float) -> None:
        """Update the progress dialog.

        Args:
            message: Current meshing step.
            fraction: Overall completion, 0 to 1.
        """
        if self._mesh_progress is not None:
            self._mesh_progress.setLabelText(message)
            self._mesh_progress.setValue(round(fraction * _PROGRESS_STEPS))

    def _finish_mesh_job(self) -> None:
        """Close the progress dialog and release the finished job."""
        if self._mesh_progress is not None:
            self._mesh_progress.close()
            self._mesh_progress.deleteLater()
        if self._mesh_job is not None:
            self._mesh_job.deleteLater()
        self._mesh_job = self._mesh_progress = None
        self._update_state()

    # -- solving ------------------------------------------------------------------

    def solve(self) -> None:
        """Solve current flow and temperature in a child process, with a progress dialog.

        The project is saved first (the solver reads it from disk). The solver
        reuses the saved mesh if it is current, otherwise it meshes the board too.
        """
        doc = self.document
        if doc.project is None or doc.board is None or self._mesh_job or self._solve_job:
            return
        sources = doc.current_sources
        if not doc.board.has_outline:
            QMessageBox.warning(self, "Cannot solve", "The board has no Edge.Cuts outline.")
            return
        if not sources:
            QMessageBox.warning(self, "Cannot solve", "Add current sources to pads before solving.")
            return
        if abs(sum(s.current_a for s in sources)) > 1e-9:
            QMessageBox.warning(self, "Cannot solve", "Current sources and sinks must sum to zero.")
            return
        if doc.dirty and not self.save():
            return
        if doc.board_changed_on_disk:
            if not self.reload_board():
                return
            self.statusBar().showMessage("The board changed on disk and was reloaded before solving.", 5000)

        job = SolveJob(doc.project.path, self)
        dialog = QProgressDialog("Starting solver…", "Cancel", 0, _PROGRESS_STEPS, self)
        dialog.setWindowTitle("Solving")
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setValue(0)
        dialog.canceled.connect(job.cancel)
        job.progress.connect(self._on_solve_progress)
        job.succeeded.connect(self._on_solved)
        job.failed.connect(self._on_solve_failed)
        job.cancelled.connect(self._on_solve_cancelled)

        self._solve_job, self._solve_progress = job, dialog
        self._update_state()
        job.start()
        dialog.show()

    @property
    def solve_job_running(self) -> bool:
        """Whether a solve is in progress."""
        return self._solve_job is not None

    def _on_solve_progress(self, message: str, fraction: float) -> None:
        """Update the solve progress dialog.

        Args:
            message: Current step.
            fraction: Overall completion, 0 to 1.
        """
        if self._solve_progress is not None:
            self._solve_progress.setLabelText(message)
            self._solve_progress.setValue(round(fraction * _PROGRESS_STEPS))

    def _finish_solve_job(self) -> None:
        """Close the solve progress dialog and release the finished job."""
        if self._solve_progress is not None:
            self._solve_progress.close()
            self._solve_progress.deleteLater()
        if self._solve_job is not None:
            self._solve_job.deleteLater()
        self._solve_job = self._solve_progress = None
        self._update_state()

    def _on_solved(self, results) -> None:
        """Show new results, and pick up the mesh the solver may have built.

        Args:
            results: The :class:`~kits.solve.results.SolveResults` loaded from the child's output.
        """
        self._finish_solve_job()
        self.document.reload_saved_mesh()
        self.document.set_results(results)
        self._solve_dock.raise_()
        self.statusBar().showMessage(
            f"Solved: {results.total_power_w:.4g} W, max {results.max_temperature_c:.1f} °C", 8000
        )

    def _on_solve_failed(self, message: str) -> None:
        """Report a failed solve.

        Args:
            message: Why solving failed.
        """
        self._finish_solve_job()
        QMessageBox.critical(self, "Solve failed", message)

    def _on_solve_cancelled(self) -> None:
        """Note a cancelled solve; previous results (if any) are kept."""
        self._finish_solve_job()
        self.statusBar().showMessage("Solve cancelled", 5000)

    def _on_results_changed(self, results) -> None:
        """Show (or clear) the results overlay and its legend.

        Args:
            results: The new results, or None.
        """
        if self.scene is not None:
            self.scene.set_results(results, self.solve_panel.display_mode)
            self.scene.set_results_visible(self.show_results_action.isChecked())
            if results is not None:
                self.solve_panel.set_legend(self.solve_panel.display_mode, *self.scene.results_range)

    def _on_display_mode_changed(self, mode) -> None:
        """Recolour the results overlay for a different quantity.

        Args:
            mode: The :class:`~kits.gui.field_display.DisplayMode` to show.
        """
        if self.scene is not None and self.document.results is not None:
            self.scene.set_results(self.document.results, mode)
            self.solve_panel.set_legend(mode, *self.scene.results_range)

    def _on_show_results_toggled(self, checked: bool) -> None:
        """Show or hide the results overlay, keeping the panel checkbox in sync.

        Args:
            checked: New state of the "Show results" action.
        """
        self.solve_panel.show_results.setChecked(checked)
        if self.scene is not None:
            self.scene.set_results_visible(checked)

    def _on_mesh_built(self, mesh: BoardMesh) -> None:
        """Store a freshly built mesh in the document.

        Args:
            mesh: The mesh loaded from the child process's output.
        """
        self._finish_mesh_job()
        self.document.set_mesh(mesh)
        self.statusBar().showMessage(
            f"Mesh built: {mesh.num_triangles:,} triangles on {len(mesh.layers)} layers", 5000
        )

    def _on_mesh_failed(self, message: str) -> None:
        """Report a failed mesh build.

        Args:
            message: Why meshing failed.
        """
        self._finish_mesh_job()
        QMessageBox.critical(self, "Meshing failed", message)

    def _on_mesh_cancelled(self) -> None:
        """Note a cancelled mesh build; the previous mesh (if any) is kept."""
        self._finish_mesh_job()
        self.statusBar().showMessage("Meshing cancelled", 5000)

    def _on_mesh_changed(self, mesh: BoardMesh | None) -> None:
        """Show the document's current mesh as an overlay.

        Args:
            mesh: The new mesh, or None if it was cleared.
        """
        if self.scene is not None:
            self.scene.set_mesh(mesh)
            self.scene.set_mesh_visible(self.show_mesh_action.isChecked())
        if self.document.mesh_notice:
            self.statusBar().showMessage(f"{self.document.mesh_notice}: rebuild the mesh", 10000)
        elif self.document.mesh_from_file:
            self.statusBar().showMessage(f"Loaded the saved mesh ({mesh.num_triangles:,} triangles)", 5000)

    def _on_show_mesh_toggled(self, checked: bool) -> None:
        """Show or hide the mesh overlay, keeping the panel checkbox in sync.

        Args:
            checked: New state of the "Show mesh" action.
        """
        self.mesh_panel.show_mesh.setChecked(checked)
        if self.scene is not None:
            self.scene.set_mesh_visible(checked)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Ask about unsaved changes, and kill any running mesher, before closing.

        Args:
            event: The close event; ignored if the user cancels.
        """
        if not self.maybe_save():
            event.ignore()
            return
        self._cancel_jobs()
        super().closeEvent(event)

    def _cancel_jobs(self) -> None:
        """Kill any running mesher or solver process."""
        for job in (self._mesh_job, self._solve_job):
            if job is not None:
                job.cancel()

    # -- document updates ------------------------------------------------------

    def _on_project_changed(self, project: KitsProject | None) -> None:
        """Update the window for a newly created, opened, renamed or closed project.

        Args:
            project: The project, or ``None`` when it was closed.
        """
        self._update_state()

    def _on_board_changed(self, kicad: Project | None) -> None:
        """Rebuild the scene and panels for a newly loaded (or unloaded) board.

        Args:
            kicad: The loaded KiCad project, or ``None`` if no board is loaded.
        """
        old_scene = self.scene
        self.scene = BoardScene(kicad.board, self) if kicad else None
        self._apply_settings()
        self.view.setScene(self.scene)
        if old_scene is not None:
            old_scene.deleteLater()

        if self.scene is not None:
            self.layer_panel.set_layers(
                [(lay.name, lay.color, lay.is_copper) for lay in self.scene.layers.values()]
            )
            self.view.fit_to_rect(self.scene.board_rect())
            b = kicad.board
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

        self.info_panel.set_project(kicad)
        self._selected_pad = None
        self.sources_panel.set_selected_pad(None)
        self._show_sources()
        self._update_state()

    def _update_state(self, *_args) -> None:
        """Sync the visible page, action states and title with the document.

        Args:
            *_args: Ignored signal arguments.
        """
        doc = self.document
        if doc.project is None:
            self._pages.setCurrentWidget(self.start_page)
        elif doc.board is None:
            if doc.project.board is None:
                message = "This project doesn't link a KiCad board yet."
            else:
                message = (
                    f"The linked KiCad board couldn't be loaded:\n{doc.board_error}\n\n"
                    "If it has moved, link it again."
                )
            self.no_board_page.set_message(message)
            self._pages.setCurrentWidget(self.no_board_page)
        else:
            self._pages.setCurrentWidget(self.view)
        for act in self._project_actions:
            act.setEnabled(doc.project is not None)
        for act in self._board_actions:
            act.setEnabled(doc.board is not None)
        idle = self._mesh_job is None and self._solve_job is None
        self.build_mesh_action.setEnabled(doc.board is not None and idle)
        self.solve_action.setEnabled(doc.board is not None and idle)
        self.mesh_panel.set_busy(not idle)
        self.solve_panel.set_busy(not idle)
        self._update_title()

    def _update_title(self, *_args) -> None:
        """Set the window title from the project name, unsaved state and app version.

        Args:
            *_args: Ignored signal arguments.
        """
        project = self.document.project
        app_title = f"KITS {__version__}"
        if project is None:
            self.setWindowTitle(app_title)
        else:
            marker = "*" if self.document.dirty else ""
            self.setWindowTitle(f"{project.name}{marker} — {app_title}")

    def _update_recent(self) -> None:
        """Refresh the start page's recent-projects list."""
        self.start_page.set_recent(self.settings.recent_projects)

    def _populate_recent_menu(self) -> None:
        """Rebuild the Open Recent menu just before it is shown."""
        recent = self.settings.recent_projects
        self.recent_menu.clear()
        for i, path in enumerate(recent, start=1):
            act = self.recent_menu.addAction(f"&{i % 10} {path.stem}  —  {path}")
            act.setEnabled(path.is_file())
            act.triggered.connect(lambda _checked=False, p=path: self.open_path(p))
        if recent:
            self.recent_menu.addSeparator()
            self.recent_menu.addAction("Clear Recent Projects", self.settings.clear_recent_projects)
        else:
            self.recent_menu.addAction("No recent projects").setEnabled(False)

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

    # -- pad selection and current sources --------------------------------------

    def select_pad(self, key: tuple[str, str] | None) -> None:
        """Select a pad on the board and in the Current Sources panel.

        Args:
            key: ``(ref, pad)``, or None to clear the selection.
        """
        self._selected_pad = tuple(key) if key is not None else None
        if self.scene is not None:
            self.scene.set_selected_pad(self._selected_pad)
        self.sources_panel.set_selected_pad(self._selected_pad)

    @property
    def selected_pad(self) -> tuple[str, str] | None:
        """The pad selected on the board, as ``(ref, pad)``."""
        return self._selected_pad

    def _on_board_clicked(self, pos) -> None:
        """Select the pad under a left click (or clear the selection on empty board).

        Args:
            pos: Scene position of the click, in millimetres.
        """
        if self.scene is None:
            return
        key = self.scene.pad_at(pos)
        self.select_pad(key)
        if key is not None:
            pads = self.document.board.find_pads(*key)
            net = next((p.net for p in pads if p.net), None)
            on_net = f" on net {net}" if net else ""
            self.statusBar().showMessage(
                f"{key[0]} pad {key[1]}{on_net} — double-click or 'Add selected pad' to set a current", 5000
            )

    def _on_board_double_clicked(self, pos) -> None:
        """Add (or go to) a current source on the double-clicked pad.

        Args:
            pos: Scene position of the double-click, in millimetres.
        """
        if self.scene is None:
            return
        key = self.scene.pad_at(pos)
        if key is None:
            return
        self.select_pad(key)
        self._sources_dock.show()
        self._sources_dock.raise_()
        self.sources_panel.add_selected_pad()

    def _on_source_row_selected(self, key: tuple[str, str] | None) -> None:
        """Highlight the pad of the row selected in the Current Sources panel.

        Args:
            key: ``(ref, pad)`` of the selected row, or None.
        """
        if key is None:
            return
        self._selected_pad = tuple(key)
        if self.scene is not None:
            self.scene.set_selected_pad(self._selected_pad)

    def _show_sources(self, *_args) -> None:
        """Redraw the current source markers on the board.

        Args:
            *_args: Ignored signal arguments.
        """
        if self.scene is not None:
            self.scene.set_sources(self.document.current_sources)

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
