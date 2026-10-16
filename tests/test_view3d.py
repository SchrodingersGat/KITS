"""3D view: extruded geometry (checked by volume) and the 2D/3D toggle."""

import math

import pytest

pytest.importorskip("pyvista")
pytest.importorskip("PySide6")

from shapely.geometry import Point, box  # noqa: E402

from kits.gui.view3d import board_parts, extrude  # noqa: E402
from kits.ingest import load_board  # noqa: E402

from .test_gui_project import _wait, app, settings, strip_kits, window  # noqa: E402, F401
from .test_solve import _HEADER, _pad, _strip  # noqa: E402

_VIA = '  (via (at 25 5) (size 0.8) (drill 0.4) (layers "F.Cu" "B.Cu") (net 1))\n'


def test_extrude_is_a_closed_prism():
    area = box(0, 0, 10, 4).difference(Point(5, 2).buffer(1, quad_segs=16))
    mesh = extrude(area, 1.0, 1.5).clean()
    assert mesh.is_manifold  # watertight, including the hole's wall
    assert mesh.volume == pytest.approx(area.area * 0.5, rel=1e-9)
    # KiCad's y points down; in 3D it points up.
    assert mesh.bounds[2:4] == pytest.approx((-4.0, 0.0))
    assert extrude(Point(0, 0), 0, 1) is None  # nothing to extrude


@pytest.fixture
def strip_with_via(tmp_path):
    path = tmp_path / "via.kicad_pcb"
    path.write_text(_HEADER + _strip("F.Cu") + _pad("P1", 7, "F.Cu") + _VIA + ")\n")
    return load_board(path)


def test_board_parts(strip_with_via):
    parts = {p.key: p for p in board_parts(strip_with_via, z_scale=2.0)}
    assert list(parts) == ["copper:F.Cu", "copper:B.Cu", "dielectric:F.Cu-B.Cu", "barrels"]
    assert parts["copper:F.Cu"].layer == "F.Cu" and parts["dielectric:F.Cu-B.Cu"].layer is None

    # F.Cu: the 40 x 6 strip minus the via's drill, 35 µm thick, scaled x2.
    drill = math.pi * 0.2**2
    copper = parts["copper:F.Cu"].mesh.clean()
    assert copper.volume == pytest.approx((40 * 6 - drill) * 0.035 * 2, rel=2e-3)
    # B.Cu has only the via pad (0.8 mm) with its hole.
    pad = parts["copper:B.Cu"].mesh.clean()
    assert pad.volume == pytest.approx((math.pi * 0.4**2 - drill) * 0.035 * 2, rel=2e-2)

    # The dielectric fills the 50 x 10 board between the copper, minus the drill.
    dielectric = parts["dielectric:F.Cu-B.Cu"].mesh.clean()
    assert dielectric.volume == pytest.approx((50 * 10 - drill) * 1.51 * 2, rel=1e-3)
    z0, z1 = dielectric.bounds[4:6]
    assert z1 - z0 == pytest.approx(1.51 * 2, rel=1e-6)

    # The barrel spans both copper layers, outside the drill.
    barrel = parts["barrels"].mesh
    assert barrel.bounds[4] < z0 and barrel.bounds[5] > z1
    assert math.hypot(barrel.center[0] - 25, barrel.center[1] + 5) < 1e-6


def test_toggle_between_2d_and_3d(app, window, kits_v9_path):  # noqa: F811
    assert not window.view_3d_action.isEnabled()  # no board yet
    assert window.open_path(kits_v9_path)
    assert window.view_3d_action.isEnabled()
    assert window._pages.currentWidget() is window.view

    window.show()  # the 3D scene is built when the view is first shown
    window.view_3d_action.trigger()
    assert window._pages.currentWidget() is window.view3d
    # VTK loads and the solids are built off the GUI thread; the view says so meanwhile.
    assert window.view3d.busy and window.view3d.message.text().startswith("Loading")
    _wait(app, lambda: not window.view3d.busy)
    keys = [p.key for p in window.view3d.parts]
    assert "copper:F.Cu" in keys and any(k.startswith("dielectric:") for k in keys)

    # The Layers panel's toggles apply in 3D too; the dielectric has its own.
    window._on_layer_visibility("F.Cu", False)
    assert "copper:F.Cu" not in window.view3d.visible_part_keys()
    window.view3d.show_dielectric.setChecked(False)
    assert not any(k.startswith("dielectric:") for k in window.view3d.visible_part_keys())

    # Changing the vertical scale rebuilds the solids, keeping the visibility choices.
    window.view3d.z_scale.setValue(10)
    _wait(app, lambda: not window.view3d.busy)
    assert "copper:F.Cu" not in window.view3d.visible_part_keys()
    assert "copper:B.Cu" in window.view3d.visible_part_keys()

    window.view_3d_action.trigger()  # already 3D: stays 3D (2 and 3 select, they don't toggle)
    assert window._pages.currentWidget() is window.view3d
    window.view_2d_action.trigger()
    assert window._pages.currentWidget() is window.view and not window.view_3d_action.isChecked()
    window.view3d.shutdown()


def test_overlay_parts_sit_on_the_copper_faces():
    import numpy as np

    from kits.gui.field_display import DisplayMode, colormap
    from kits.gui.view3d import overlay_parts
    from kits.mesh.model import BoardMesh, LayerMesh

    from .test_results import _results

    results = _results()  # one F.Cu triangle, copper, 3 A/mm²
    nodes, triangles = results.layers[0].nodes, results.layers[0].triangles
    mesh = BoardMesh(
        [LayerMesh("F.Cu", 1.5, nodes, np.vstack([triangles, triangles]), np.array([True, False]))]
    )
    ranges = {"F.Cu": (3.0, 3.1)}
    parts = {p.key: p for p in overlay_parts(ranges, mesh, results, DisplayMode.CURRENT_DENSITY)}
    assert set(parts) == {"results:F.Cu", "mesh:F.Cu", "mesh-bare:F.Cu"}
    assert all(p.layer == "F.Cu" for p in parts.values())

    sheet = parts["results:F.Cu"].mesh
    assert sheet.n_cells == 2  # one copy just below the copper, one just above
    z = sorted(set(sheet.points[:, 2]))
    assert z[0] < 3.0 and z[1] > 3.1 and z[1] - 3.1 < 0.05
    top = colormap(DisplayMode.CURRENT_DENSITY)[-1]  # the only value is the top of the scale
    assert sheet.cell_data["rgb"].tolist() == [[top.red(), top.green(), top.blue()]] * 2
    # The wireframe sits a little further out, so it's drawn over the results.
    assert parts["mesh:F.Cu"].mesh.points[:, 2].max() > z[1]
    assert overlay_parts(ranges) == [] and overlay_parts({}, mesh, results, DisplayMode.TEMPERATURE) == []


def test_mesh_and_results_in_3d(app, window, strip_kits):  # noqa: F811
    assert window.open_path(strip_kits)
    window.show()
    window.view_3d_action.trigger()
    _wait(app, lambda: not window.view3d.busy)
    assert not [k for k in window.view3d.visible_part_keys() if k.startswith(("mesh", "results"))]

    window.solve()  # meshes too
    _wait(app, lambda: not window.solve_job_running and window.document.results is not None)
    _wait(app, lambda: not window.view3d.busy)
    keys = window.view3d.visible_part_keys()
    assert "mesh:F.Cu" in keys and "results:F.Cu" in keys

    # The 2D toggles and display mode apply in 3D.
    window.show_mesh_action.trigger()
    assert "mesh:F.Cu" not in window.view3d.visible_part_keys()
    window.show_results_action.trigger()
    assert "results:F.Cu" not in window.view3d.visible_part_keys()
    window.show_results_action.trigger()
    before = window.view3d.overlays[0].mesh.cell_data["rgb"].copy()
    from kits.gui.field_display import DisplayMode

    window.results_panel.mode.setCurrentIndex(window.results_panel.mode.findData(DisplayMode.TEMPERATURE))
    results = next(p for p in window.view3d.overlays if p.key == "results:F.Cu")
    assert results.mesh.n_cells != len(before) or (results.mesh.cell_data["rgb"] != before).any()
    window._on_layer_visibility("F.Cu", False)
    assert "results:F.Cu" not in window.view3d.visible_part_keys()
    window.view3d.shutdown()


def test_window_is_primed_for_opengl(window):  # noqa: F811
    # A hidden QOpenGLWidget from the start: otherwise Qt re-creates the main window
    # (it vanishes and re-appears) when the 3D view is first shown.
    from PySide6.QtOpenGLWidgets import QOpenGLWidget

    primer = window.findChild(QOpenGLWidget, "OpenGLPrimer")
    assert primer is not None and primer is window._gl_primer
    assert primer.parent() is window and primer.isHidden() and primer.size().isEmpty()


def test_view_switch_pills(app, window, kits_v9_path):  # noqa: F811
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeySequence

    switch = window.view_switch
    assert [b.text() for b in switch.buttons] == ["2D", "3D"]
    assert window.view_2d_action.shortcut() == QKeySequence(Qt.Key.Key_2)
    assert window.view_3d_action.shortcut() == QKeySequence(Qt.Key.Key_3)
    assert switch.isHidden()  # start page: nothing to switch
    assert window.open_path(kits_v9_path)
    window.resize(900, 600)
    window.show()
    app.processEvents()
    assert switch.isVisible() and switch.buttons[0].isChecked() and not switch.buttons[1].isChecked()

    # Bottom-right corner of the viewport, above its pages.
    viewport = window._pages
    assert switch.parent() is viewport
    assert viewport.width() - switch.geometry().right() < 40
    assert viewport.height() - switch.geometry().bottom() < 40

    switch.buttons[1].click()
    assert window._pages.currentWidget() is window.view3d and switch.buttons[1].isChecked()
    assert switch.isVisible()
    switch.buttons[0].click()
    assert window._pages.currentWidget() is window.view and switch.buttons[0].isChecked()

    # Stays in the corner when the window is resized.
    window.resize(1100, 750)
    app.processEvents()
    assert viewport.width() - switch.geometry().right() < 40
    window.view3d.shutdown()


def test_element_bodies_as_blocks(tmp_path):
    from kits.config import PadGroup, ResistiveElement
    from kits.gui.field_display import DisplayMode, colormap
    from kits.gui.view3d import COMPONENT_HEIGHT_MM, ELEMENT_BODY_COLOR, element_parts, layer_z_ranges

    from .test_solve import _FAB_RESISTOR, _footprint

    board, _ = _footprint(tmp_path, _FAB_RESISTOR)  # R1: a 12 x 5 Fab body on top
    element = ResistiveElement("R1", "R1", PadGroup("A", ["1"]), PadGroup("B", ["2"]), 0.01)
    ranges = layer_z_ranges(board, 2.0)
    (part,) = element_parts(board, ranges, [element], 2.0)
    assert part.key == "element:R1" and part.color == ELEMENT_BODY_COLOR
    mesh = part.mesh.clean()
    assert mesh.volume == pytest.approx(60 * COMPONENT_HEIGHT_MM * 2.0, rel=1e-6)
    top = max(z1 for _, z1 in ranges.values())
    assert mesh.bounds[4] == pytest.approx(top)  # sits on the top face

    # Coloured by body temperature on the temperature scale when given.
    (hot,) = element_parts(board, ranges, [element], 2.0, {"R1": 90.0}, (25.0, 80.0))
    assert hot.color == colormap(DisplayMode.TEMPERATURE)[-1]  # above the scale: its top colour
    missing = ResistiveElement("R9", "R9", PadGroup("A", ["1"]), PadGroup("B", ["2"]), 0.01)
    assert element_parts(board, ranges, [missing], 2.0) == []  # footprint not on the board
