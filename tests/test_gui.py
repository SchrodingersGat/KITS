import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QRectF, QSettings, Qt  # noqa: E402
from PySide6.QtGui import QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from kits.gui import colors  # noqa: E402
from kits.gui.board_scene import HOLES, MODEL_ROLE, TH_PADS, VIAS, BoardScene  # noqa: E402
from kits.gui.main_window import MainWindow  # noqa: E402
from kits.gui.settings import DEFAULT_GRID_SPACING_MM, AppSettings  # noqa: E402
from kits.gui.settings_dialog import SettingsDialog  # noqa: E402
from kits.ingest import load_board  # noqa: E402
from kits.ingest.model import Pad  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_open_project_builds_scene(app, board_v9_path):
    window = MainWindow()
    assert window.open_path(board_v9_path.with_suffix(".kicad_pro"))
    scene = window.scene
    assert scene.layer_names() == ["F.Cu", "In1.Cu", "B.Cu", TH_PADS, VIAS, HOLES, "Edge.Cuts"]
    assert window.windowTitle().startswith("demo")

    pads = [i.data(MODEL_ROLE) for i in scene.items() if isinstance(i.data(MODEL_ROLE), Pad)]
    assert sorted(p.number for p in pads) == ["1", "2"]

    scene.set_layer_visible("F.Cu", False)
    assert not scene.layers["F.Cu"].item.isVisible()
    scene.set_active_layer("B.Cu")
    assert scene.layers["B.Cu"].item.zValue() > scene.layers["F.Cu"].item.zValue()


def test_missing_outline_is_reported(app, board_v9_path, board_v10_path):
    window = MainWindow()
    assert window.open_path(board_v10_path)
    assert window.info_panel._labels["Size"].text() == "No board outline"
    assert "no board outline" in window.statusBar().currentMessage()

    assert window.open_path(board_v9_path)
    assert window.info_panel._labels["Size"].text() == "40.00 × 30.00 mm"


def test_failed_open_keeps_previous_project(app, board_v9_path, tmp_path, monkeypatch):
    errors = []
    monkeypatch.setattr("kits.gui.main_window.QMessageBox.critical", lambda *a: errors.append(a))
    window = MainWindow()
    assert window.open_path(board_v9_path)
    assert not window.open_path(tmp_path / "nope.kicad_pro")
    assert errors
    assert window.document.project.name == "demo"


# -- grid -------------------------------------------------------------------


def _render(scene, source: QRectF, size: int = 200) -> QImage:
    image = QImage(size, size, QImage.Format.Format_RGB32)
    painter = QPainter(image)
    scene.render(painter, QRectF(0, 0, size, size), source)
    painter.end()
    return image


def test_settings_defaults_and_persistence(app, tmp_path):
    store = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    settings = AppSettings(store)
    assert settings.grid_spacing_mm == DEFAULT_GRID_SPACING_MM == 10.0
    assert settings.grid_visible

    changes = []
    settings.changed.connect(lambda: changes.append(1))
    settings.grid_spacing_mm = 2.54
    settings.grid_visible = False
    settings.grid_visible = False  # unchanged: no signal
    assert len(changes) == 2

    reloaded = AppSettings(QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat))
    assert reloaded.grid_spacing_mm == pytest.approx(2.54)
    assert not reloaded.grid_visible

    store.setValue("grid/spacing_mm", "garbage")
    assert settings.grid_spacing_mm == DEFAULT_GRID_SPACING_MM


def test_grid_origin_is_bottom_left_of_board(app, board_v9_path):
    scene = BoardScene(load_board(board_v9_path))  # outline is (0, 0)-(40, 30)
    origin = scene.grid_origin()
    assert (origin.x(), origin.y()) == (0, 30)  # Y points down: bottom is max Y


def test_grid_thins_out_when_zoomed_out(app, board_v9_path):
    scene = BoardScene(load_board(board_v9_path))
    scene.set_grid(10, True)
    assert scene.displayed_grid_spacing(pixels_per_mm=5) == 10
    assert scene.displayed_grid_spacing(pixels_per_mm=0.5) == 20
    assert scene.displayed_grid_spacing(pixels_per_mm=0.1) == 100
    with pytest.raises(ValueError):
        scene.set_grid(0, True)


def test_grid_is_drawn_on_origin_aligned_lines(app, board_v9_path):
    scene = BoardScene(load_board(board_v9_path))
    scene.set_grid(10, True)
    # Empty area off the board, at 10 px/mm: origin (0, 30) puts lines at x=50, y=40.
    source = QRectF(42, 32, 20, 20)
    image = _render(scene, source)
    on_line = image.pixelColor(80, 30)  # x = 50 mm
    off_line = image.pixelColor(30, 30)  # x = 45 mm
    assert on_line == colors.GRID
    assert off_line == colors.BACKGROUND
    assert image.pixelColor(30, 80) == colors.GRID  # y = 40 mm

    scene.set_grid(10, False)
    assert _render(scene, source).pixelColor(80, 30) == colors.BACKGROUND


def test_grid_toggle_and_preferences(app, board_v9_path):
    window = MainWindow()
    assert window.open_path(board_v9_path)
    assert window.scene.grid_visible
    assert window.scene.grid_spacing == 10

    window.grid_action.trigger()
    assert not window.scene.grid_visible
    assert not window.settings.grid_visible

    dialog = SettingsDialog(window.settings, window)
    dialog.grid_spacing.setValue(2.5)
    dialog.grid_visible.setChecked(True)
    dialog.accept()
    assert window.scene.grid_spacing == 2.5
    assert window.scene.grid_visible
    assert window.grid_action.isChecked()

    # A newly opened board picks up the stored settings.
    assert window.open_path(board_v9_path)
    assert window.scene.grid_spacing == 2.5


def test_statusbar_coords_are_grid_relative(app, board_v9_path):
    window = MainWindow()
    assert window.open_path(board_v9_path)  # outline (0, 0)-(40, 30): grid origin at KiCad (0, 30)
    assert window.scene.to_grid_coords(0, 30) == (0, 0)
    assert window.scene.to_grid_coords(40, 0) == (40, 30)  # top-right; Y up

    window.view.cursor_moved.emit(10.0, 25.0)
    assert window._cursor_label.text() == f"X {10:9.3f} mm   Y {5:9.3f} mm"


# -- crosshair cursor ----------------------------------------------------------


def _grab_after_move(window, pos):
    from PySide6.QtCore import QPoint, QPointF
    from PySide6.QtGui import QMouseEvent

    vp = window.view.viewport()
    event = QMouseEvent(
        QMouseEvent.Type.MouseMove,
        QPointF(pos),
        QPointF(vp.mapToGlobal(pos)),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    window.view.mouseMoveEvent(event)
    image = vp.grab().toImage()
    return image, QPoint(int(pos.x() * image.devicePixelRatio()), int(pos.y() * image.devicePixelRatio()))


def test_cursor_setting_defaults_on(app, tmp_path):
    settings = AppSettings(QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat))
    assert settings.cursor_visible
    settings.cursor_visible = False
    assert not AppSettings(QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)).cursor_visible


def test_crosshair_follows_mouse_and_toggles(app, board_v9_path):
    from PySide6.QtCore import QPoint

    window = MainWindow()
    window._layer_dock.hide()
    window._info_dock.hide()
    window._mesh_dock.hide()
    window.resize(800, 600)
    window.show()
    assert window.open_path(board_v9_path)
    window.settings.grid_visible = False  # keep grid lines out of the pixel checks
    app.processEvents()

    view = window.view
    # A spot in the empty margin left of the board, so only the crosshair is drawn there.
    board_left = view.mapFromScene(window.scene.board_rect().topLeft()).x()
    assert board_left > 40
    target = QPoint(board_left // 2, view.viewport().height() // 2)
    image, p = _grab_after_move(window, target)
    background = image.pixelColor(p + QPoint(-15, -15))
    assert image.pixelColor(p + QPoint(-10, 0)) != background  # horizontal line
    assert image.pixelColor(p + QPoint(0, -10)) != background  # vertical line

    # Toggling via the View menu hides it and persists the setting.
    window.cursor_action.trigger()
    assert not view.cursor_visible
    assert not window.settings.cursor_visible
    image, p = _grab_after_move(window, target)
    assert image.pixelColor(p + QPoint(-10, 0)) == background

    # The Preferences dialog turns it back on.
    dialog = SettingsDialog(window.settings, window)
    dialog.cursor_visible.setChecked(True)
    dialog.accept()
    assert view.cursor_visible and window.cursor_action.isChecked()
    window.close()


# -- meshing -------------------------------------------------------------------


def _wait_until(app, condition, timeout=60.0):
    import time

    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError("condition not met")
        app.processEvents()
        time.sleep(0.01)


def test_mesh_panel_saves_settings_and_validates(app, board_v9_path):
    from kits.config import config_path_for, load_config

    window = MainWindow()
    assert window.open_path(board_v9_path.with_suffix(".kicad_pro"))
    panel = window.mesh_panel
    assert panel.build_button.isEnabled()

    panel.max_size.setValue(2.5)
    config_file = config_path_for(board_v9_path)
    assert load_config(config_file).mesh.max_size_mm == 2.5

    panel.min_size.setValue(3.0)  # larger than max: rejected, not saved
    assert not panel.settings_error.isHidden()
    assert not panel.build_button.isEnabled()
    assert load_config(config_file).mesh.min_size_mm == 0.1

    panel.min_size.setValue(0.2)
    assert panel.settings_error.isHidden() and panel.build_button.isEnabled()


def test_build_mesh_in_child_process_shows_overlay(app, board_v9_path):
    window = MainWindow()
    window.show()
    assert window.open_path(board_v9_path.with_suffix(".kicad_pro"))
    window.mesh_panel.max_size.setValue(2.0)

    window.build_mesh()
    assert window.mesh_job_running
    assert not window.mesh_panel.build_button.isEnabled()
    _wait_until(app, lambda: not window.mesh_job_running)

    mesh = window.document.mesh
    assert mesh is not None and mesh.settings.max_size_mm == 2.0
    assert mesh.msh_path.is_file()
    assert window.mesh_panel.build_button.text() == "Rebuild mesh"

    scene = window.scene
    assert scene.mesh_layers == ["F.Cu", "In1.Cu", "B.Cu"]
    # Only the active layer's mesh is drawn.
    assert scene.mesh_overlay_visible("F.Cu") and not scene.mesh_overlay_visible("B.Cu")
    scene.set_active_layer("B.Cu")
    assert scene.mesh_overlay_visible("B.Cu") and not scene.mesh_overlay_visible("F.Cu")
    window.show_mesh_action.trigger()
    assert not scene.mesh_overlay_visible("B.Cu")
    assert not window.mesh_panel.show_mesh.isChecked()

    # Changing settings marks the mesh stale; rebuilding replaces it.
    window.mesh_panel.max_size.setValue(3.0)
    assert window.document.mesh_is_stale
    window.build_mesh()
    _wait_until(app, lambda: not window.mesh_job_running)
    assert window.document.mesh.settings.max_size_mm == 3.0
    assert not window.document.mesh_is_stale
    window.close()


def test_cancel_mesh_job_keeps_previous_mesh(app, board_v9_path):
    window = MainWindow()
    assert window.open_path(board_v9_path)
    window.build_mesh()
    window._mesh_job.cancel()
    _wait_until(app, lambda: not window.mesh_job_running)
    assert window.document.mesh is None
    assert "cancelled" in window.statusBar().currentMessage()


def test_build_mesh_without_outline_warns(app, board_v10_path, monkeypatch):
    warnings = []
    monkeypatch.setattr("kits.gui.main_window.QMessageBox.warning", lambda *a: warnings.append(a))
    window = MainWindow()
    assert window.open_path(board_v10_path)
    assert not window.mesh_panel.build_button.isEnabled()
    window.build_mesh()
    assert warnings and not window.mesh_job_running


def test_mesh_failure_is_reported(app, board_v9_path, monkeypatch):
    errors = []
    monkeypatch.setattr("kits.gui.main_window.QMessageBox.critical", lambda *a: errors.append(a))
    window = MainWindow()
    assert window.open_path(board_v9_path)
    # Corrupt the board after loading: the child re-reads it and fails.
    board_v9_path.write_text("(kicad_pcb (version 1)", encoding="utf-8")
    window.document._board_mtime = board_v9_path.stat().st_mtime  # skip the auto-reload
    window.build_mesh()
    _wait_until(app, lambda: not window.mesh_job_running)
    assert errors and "Malformed" in errors[0][2]
