import dataclasses

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from kits.config import MeshSettings  # noqa: E402
from kits.gui.main_window import MainWindow  # noqa: E402
from kits.gui.settings import AppSettings  # noqa: E402
from kits.project import KitsProject, load_kits_project  # noqa: E402

Buttons = QMessageBox.StandardButton


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def settings(tmp_path):
    return AppSettings(QSettings(str(tmp_path / "app.ini"), QSettings.Format.IniFormat))


@pytest.fixture
def window(app, settings):
    w = MainWindow(settings=settings)
    yield w
    w.document.close()  # don't prompt about unsaved changes on teardown
    w.close()


@pytest.fixture
def dialogs(monkeypatch):
    """Scripted answers for file dialogs and message boxes; records what was shown."""

    class Dialogs:
        save_paths: list = []
        open_paths: list = []
        shown: list = []

    d = Dialogs()
    monkeypatch.setattr(
        "kits.gui.main_window.QFileDialog.getSaveFileName", lambda *a, **k: (str(d.save_paths.pop(0)), "")
    )
    monkeypatch.setattr(
        "kits.gui.main_window.QFileDialog.getOpenFileName", lambda *a, **k: (str(d.open_paths.pop(0)), "")
    )
    for name in ("critical", "warning"):
        monkeypatch.setattr(
            f"kits.gui.main_window.QMessageBox.{name}", lambda *a, n=name: d.shown.append((n, a))
        )
    return d


def test_start_page_until_a_project_is_opened(window, kits_v9_path):
    assert window._pages.currentWidget() is window.start_page
    assert window.windowTitle() == "KITS 0.1.0"
    assert not window.save_action.isEnabled()
    assert window.open_path(kits_v9_path)
    assert window._pages.currentWidget() is window.view
    assert window.save_action.isEnabled() and window.build_mesh_action.isEnabled()
    window.close_project()
    assert window._pages.currentWidget() is window.start_page
    assert window.document.project is None and window.scene is None


def test_new_project_saves_location_then_links_board(window, dialogs, board_v9_path, tmp_path):
    target = tmp_path / "projects" / "mine"  # suffix is added
    target.parent.mkdir()
    dialogs.save_paths.append(target)
    dialogs.open_paths.append(board_v9_path.with_suffix(".kicad_pro"))
    assert window.new_project()

    created = target.with_suffix(".kits")
    assert created.is_file()
    assert load_kits_project(created).board == board_v9_path.with_suffix(".kicad_pro")
    assert window.document.board is not None and not window.document.dirty
    assert window.settings.recent_projects[0] == created


def test_new_project_with_unreadable_board_creates_nothing(window, dialogs, tmp_path):
    bad = tmp_path / "bad.kicad_pcb"
    bad.write_text("(kicad_pcb", encoding="utf-8")
    dialogs.save_paths.append(tmp_path / "p.kits")
    dialogs.open_paths.append(bad)
    assert not window.new_project()
    assert not (tmp_path / "p.kits").exists()
    assert dialogs.shown[0][0] == "critical"


def test_opening_a_kicad_file_offers_to_create_a_project(window, monkeypatch, board_v9_path):
    offers = []

    def accept(board, target):
        offers.append((board, target))
        return target

    monkeypatch.setattr(window, "_ask_create_project", accept)
    assert window.open_path(board_v9_path)
    ((board, target),) = offers
    assert target == board_v9_path.with_suffix(".kits")
    assert target.is_file() and window.document.project.path == target

    # Next time the existing project is offered instead.
    window.close_project()
    monkeypatch.setattr("kits.gui.main_window.QMessageBox.question", lambda *a, **k: Buttons.Open)
    assert window.open_path(board_v9_path)
    assert len(offers) == 1 and window.document.project.path == target


def test_declining_the_project_offer_opens_nothing(window, monkeypatch, board_v9_path):
    monkeypatch.setattr(window, "_ask_create_project", lambda board, target: None)
    assert not window.open_path(board_v9_path)
    assert window.document.project is None
    assert not board_v9_path.with_suffix(".kits").exists()


@pytest.mark.parametrize("answer", [Buttons.Cancel, Buttons.Discard, Buttons.Save])
def test_unsaved_changes_prompt(window, monkeypatch, kits_v9_path, kits_v10_path, answer):
    assert window.open_path(kits_v9_path)
    window.document.set_mesh_settings(MeshSettings(2.0, 0.1))
    asked = []
    monkeypatch.setattr(window, "_ask_save_changes", lambda: asked.append(1) or answer)

    opened = window.open_path(kits_v10_path)
    assert asked
    assert opened == (answer != Buttons.Cancel)
    saved = load_kits_project(kits_v9_path).mesh.max_size_mm
    assert saved == (2.0 if answer == Buttons.Save else 1.0)


def test_close_event_can_be_cancelled(window, monkeypatch, kits_v9_path):
    assert window.open_path(kits_v9_path)
    window.document.set_mesh_settings(MeshSettings(2.0, 0.1))
    monkeypatch.setattr(window, "_ask_save_changes", lambda: Buttons.Cancel)
    window.show()
    assert not window.close()
    assert window.isVisible()


def test_project_with_missing_board_shows_link_page(window, dialogs, tmp_path, board_v9_path):
    project = KitsProject(tmp_path / "p.kits", board=tmp_path / "gone.kicad_pro")
    project.save()
    assert window.open_path(project.path)
    assert window._pages.currentWidget() is window.no_board_page
    assert "not found" in window.no_board_page.message.text()
    assert not window.build_mesh_action.isEnabled()

    dialogs.open_paths.append(board_v9_path)
    assert window.link_board_dialog()
    assert window._pages.currentWidget() is window.view
    assert window.document.dirty
    assert window.save()
    assert load_kits_project(project.path).board == board_v9_path


def test_save_as_moves_the_project(window, dialogs, kits_v9_path, tmp_path):
    assert window.open_path(kits_v9_path)
    (tmp_path / "copy").mkdir()
    dialogs.save_paths.append(tmp_path / "copy" / "renamed.kits")
    assert window.save_as_dialog()
    assert window.document.project.path == tmp_path / "copy" / "renamed.kits"
    assert window.windowTitle().startswith("renamed —")
    assert load_kits_project(tmp_path / "copy" / "renamed.kits").board == kits_v9_path.with_suffix(
        ".kicad_pro"
    )


def test_recent_projects_menu_and_start_page(app, window, kits_v9_path, kits_v10_path, tmp_path):
    missing = tmp_path / "missing.kits"
    window.settings.add_recent_project(missing)
    assert window.open_path(kits_v10_path)
    assert window.open_path(kits_v9_path)
    assert window.settings.recent_projects == [kits_v9_path, kits_v10_path, missing]
    app.processEvents()  # the start page refresh is deferred

    def menu_entries():
        window._populate_recent_menu()  # normally done when the menu opens
        return [a for a in window.recent_menu.actions() if not a.isSeparator()]

    entries = menu_entries()
    assert "demo" in entries[0].text() and entries[0].isEnabled()
    assert not entries[2].isEnabled()  # missing file greyed out
    items = [window.start_page.recent_list.item(i) for i in range(window.start_page.recent_list.count())]
    assert [i.text().splitlines()[0] for i in items] == ["demo", "v10", "missing"]
    assert not items[2].flags() & Qt.ItemFlag.ItemIsEnabled

    # Choosing a recent entry (menu or start page) opens it.
    entries[1].trigger()
    assert window.document.project.path == kits_v10_path
    app.processEvents()
    window.start_page.recent_list.itemActivated.emit(window.start_page.recent_list.item(1))
    assert window.document.project.path == kits_v9_path
    app.processEvents()

    menu_entries()[-1].trigger()  # "Clear Recent Projects"
    app.processEvents()
    assert window.settings.recent_projects == []
    assert window.start_page.empty_label.isVisibleTo(window.start_page)


def test_recent_list_is_capped_and_deduplicated(settings, tmp_path):
    for i in range(12):
        settings.add_recent_project(tmp_path / f"p{i}.kits")
    settings.add_recent_project(tmp_path / "p5.kits")
    recent = settings.recent_projects
    assert len(recent) == 10
    assert recent[0] == tmp_path / "p5.kits"
    assert recent.count(tmp_path / "p5.kits") == 1


def test_build_mesh_saves_the_project_first(window, kits_v9_path, monkeypatch):
    started = []
    monkeypatch.setattr("kits.gui.main_window.MeshJob.start", lambda self: started.append(self))
    assert window.open_path(kits_v9_path)
    window.mesh_panel.max_size.setValue(2.5)
    assert window.document.dirty
    window.build_mesh()
    assert started and not window.document.dirty
    assert load_kits_project(kits_v9_path).mesh.max_size_mm == 2.5
    window._finish_mesh_job()


def test_saved_mesh_is_reloaded_only_while_current(window, kits_v9_path, board_v9_path):
    from kits.mesh import cli

    assert cli.main([str(kits_v9_path)]) == 0  # build and save a mesh

    # Current: reloaded on open and shown.
    assert window.open_path(kits_v9_path)
    assert window.document.mesh is not None and window.document.mesh_from_file
    assert window.scene.mesh_layers == ["F.Cu", "In1.Cu", "B.Cu"]
    assert "Reloaded" in window.mesh_panel.status.text()
    assert window.mesh_panel.build_button.text() == "Rebuild mesh"
    window.close_project()

    # Settings changed in the project file: not loaded, user told to rebuild.
    project = load_kits_project(kits_v9_path)
    project.mesh = MeshSettings(0.5, 0.1)
    project.save()
    assert window.open_path(kits_v9_path)
    assert window.document.mesh is None
    assert "mesh settings changed" in window.document.mesh_notice
    assert "Rebuild the mesh" in window.mesh_panel.status.text()
    assert "rebuild the mesh" in window.statusBar().currentMessage()
    assert window.scene.mesh_layers == []
    window.close_project()

    # Settings restored but the board edited: still not loaded.
    project.mesh = MeshSettings()
    project.save()
    with open(board_v9_path, "a", encoding="utf-8") as f:
        f.write("\n")
    assert window.open_path(kits_v9_path)
    assert window.document.mesh is None
    assert "board changed" in window.document.mesh_notice

    # Rebuilding clears the notice.
    from kits.mesh.mesher import mesh_board

    window.document.set_mesh(mesh_board(window.document.board, MeshSettings()))
    assert window.document.mesh_notice is None and not window.document.mesh_from_file


def test_mesh_panel_edits_hole_points(window, kits_v9_path):
    assert window.open_path(kits_v9_path)
    panel = window.mesh_panel
    assert panel.circle_points.value() == 36
    panel.circle_points.setValue(48)
    assert window.document.project.mesh.min_circle_points == 48 and window.document.dirty
    assert window.save()
    assert load_kits_project(kits_v9_path).mesh.min_circle_points == 48


def _check_net(panel, net):
    for i in range(panel.nets.count()):
        item = panel.nets.item(i)
        if item.data(Qt.ItemDataRole.UserRole) == net:
            item.setCheckState(Qt.CheckState.Checked)
            return item
    raise AssertionError(f"net {net} not listed")


def test_mesh_panel_track_options(window, kits_v9_path):
    assert window.open_path(kits_v9_path)
    panel = window.mesh_panel
    # The fixture has a +5V track and a GND arc.
    assert sorted(panel.nets.item(i).text() for i in range(panel.nets.count())) == ["+5V (1)", "GND (1)"]
    assert panel.nets.isHidden()
    panel.track_mode.setCurrentIndex(panel.track_mode.findData("nets"))
    assert not panel.nets.isHidden()
    _check_net(panel, "GND")
    mesh = window.document.project.mesh
    assert mesh.tracks == "nets" and mesh.track_nets == ("GND",) and window.document.dirty
    panel.net_filter.setText("5v")
    assert [panel.nets.item(i).isHidden() for i in range(2)] == [False, False]  # GND stays: it is ticked
    panel.net_filter.setText("zzz")
    hidden = {
        panel.nets.item(i).data(Qt.ItemDataRole.UserRole): panel.nets.item(i).isHidden() for i in range(2)
    }
    assert hidden == {"+5V": True, "GND": False}
    # Editing a size keeps the track choice.
    panel.max_size.setValue(1.5)
    assert window.document.project.mesh.track_nets == ("GND",)
    assert window.save()
    reloaded = load_kits_project(kits_v9_path).mesh
    assert reloaded.tracks == "nets" and reloaded.track_nets == ("GND",) and reloaded.max_size_mm == 1.5


def test_mesh_areas_are_drawn_edited_and_saved(window, kits_v9_path, monkeypatch):
    from PySide6.QtCore import QRectF

    assert window.open_path(kits_v9_path)
    areas = window.mesh_panel.areas
    monkeypatch.setattr(areas, "run_dialog", lambda dialog: dialog.area)  # accept the dialog as filled in
    window.draw_mesh_area("refine")
    assert window.view.drawing
    window.view.cancel_rect_draw()
    window._on_rect_drawn(QRectF(20, 10, 8, 6))  # as if the drag finished
    window.draw_mesh_area("exclude")
    window.view.cancel_rect_draw()
    window._on_rect_drawn(QRectF(30, 20, 4, 4))

    project = window.document.project
    assert [(a.name, a.kind) for a in project.mesh.areas] == [("R1", "refine"), ("X1", "exclude")]
    refine = project.mesh.areas[0]
    assert refine.bounds == (20, 10, 28, 16) and refine.size_mm == pytest.approx(0.1)
    assert areas.table.rowCount() == 2 and window.scene.mesh_area_count == 2
    assert window.document.heatsinks == []  # rectangles went to the mesh, not a heatsink

    # Areas are part of the mesh display: hidden with it, whatever their own checkbox says.
    assert window.scene.mesh_areas_visible
    window.show_mesh_action.setChecked(False)
    assert not window.scene.mesh_areas_visible and not areas.show_areas.isEnabled()
    window.document.set_mesh_settings(dataclasses.replace(project.mesh, max_size_mm=1.5))  # redraw
    assert not window.scene.mesh_areas_visible
    window.show_mesh_action.setChecked(True)
    assert window.scene.mesh_areas_visible and areas.show_areas.isEnabled()
    areas.show_areas.setChecked(False)
    assert not window.scene.mesh_areas_visible
    areas.show_areas.setChecked(True)

    # Disabling an area hides it from the board, but keeps it in the project.
    areas.table.item(1, 0).setCheckState(Qt.CheckState.Unchecked)
    assert not project.mesh.areas[1].enabled and window.scene.mesh_area_count == 1
    areas.table.selectRow(0)
    areas.remove_selected_rows()
    assert [a.name for a in project.mesh.areas] == ["X1"] and window.scene.mesh_area_count == 0
    assert window.save()
    assert load_kits_project(kits_v9_path).mesh.areas == project.mesh.areas


def test_mesh_area_dialog_limits_size(app):
    from kits.config import MeshArea
    from kits.gui.heatsinks_panel import GridFrame
    from kits.gui.mesh_areas import MeshAreaDialog

    area = MeshArea("R1", "refine", 0, 0, 5, 5, size_mm=0.1)
    dialog = MeshAreaDialog(area, {"X1"}, GridFrame(), max_size_mm=1.0)
    assert dialog.area == area
    dialog.size.setValue(2.0)
    assert dialog.area is None and "max size" in dialog.error.text()
    dialog.kind.setCurrentIndex(dialog.kind.findData("exclude"))  # exclusions have no size
    assert dialog.area is not None and dialog.area.kind == "exclude"
    dialog.name.setText("X1")
    assert dialog.area is None and "already named" in dialog.error.text()


# -- current sources ---------------------------------------------------------------


def test_document_source_editing_marks_project_dirty(window, kits_v9_path):
    from kits.config import ConfigError, CurrentSource

    doc = window.document
    assert window.open_path(kits_v9_path)
    doc.add_current_source("R1", "1", 2.0)
    assert doc.dirty and doc.current_sources == [CurrentSource("R1", "1", 2.0)]
    with pytest.raises(ConfigError, match="already"):
        doc.add_current_source("R1", "1", 3.0)
    doc.set_source_current("R1", "1", -4.0)
    assert doc.current_source("R1", "1").current_a == -4.0
    doc.remove_current_source("R1", "1")
    assert doc.current_sources == []
    with pytest.raises(ConfigError):
        doc.remove_current_source("R1", "1")


def test_click_double_click_edit_and_remove_sources(app, window, kits_v9_path):
    from PySide6.QtCore import QPointF

    assert window.open_path(kits_v9_path)
    panel, scene = window.sources_panel, window.scene
    smd, th = QPointF(10, 21), QPointF(10.6, 19)  # R1 pad 1 (SMD), R1 pad 2 (TH, off-centre of the hole)

    # Click selects a pad; empty board clears the selection.
    window._on_board_clicked(smd)
    assert window.selected_pad == ("R1", "1")
    assert "R1 pad 1" in panel.selected_label.text() and panel.add_button.isEnabled()
    window._on_board_clicked(QPointF(35, 5))
    assert window.selected_pad is None and not panel.add_button.isEnabled()

    # Double-click adds a source with the default current; again just selects it.
    window._on_board_double_clicked(smd)
    window._on_board_double_clicked(smd)
    assert window.document.current_sources[0].key == ("R1", "1")
    assert len(window.document.current_sources) == 1 and window.document.dirty
    assert scene.source_marker_count == 1
    assert "net: +1 A" in panel.total_label.text() and "sum to zero" in panel.total_label.text()

    # Add a sink on the TH pad via the button, then edit it in the table.
    window._on_board_clicked(th)
    assert window.selected_pad == ("R1", "2")
    panel.add_selected_pad()
    row = panel._row_of(("R1", "2"))
    panel.table.cellWidget(row, 2).setValue(-1.0)
    assert window.document.current_source("R1", "2").current_a == -1.0
    assert "sum to zero" not in panel.total_label.text()
    assert scene.source_marker_count == 2
    assert panel.table.item(row, 1).text() == "+5V"

    # Selecting a row highlights its pad; Remove deletes it.
    panel.table.selectRow(panel._row_of(("R1", "1")))
    assert window.selected_pad == ("R1", "1")
    panel.remove_selected_rows()
    assert [s.key for s in window.document.current_sources] == [("R1", "2")]
    assert scene.source_marker_count == 1

    assert window.save()
    assert [s.to_dict() for s in load_kits_project(kits_v9_path).current_sources] == [
        {"ref": "R1", "pad": "2", "current_a": -1.0}
    ]


def test_hidden_layers_and_missing_pads(window, kits_v9_path):
    from PySide6.QtCore import QPointF

    project = load_kits_project(kits_v9_path)
    from kits.config import CurrentSource

    project.current_sources = [CurrentSource("U99", "3", 1.0)]
    project.save()
    assert window.open_path(kits_v9_path)
    panel = window.sources_panel
    assert panel.table.item(0, 1).text() == "pad not found"
    assert window.scene.source_marker_count == 0  # nothing to draw

    window.scene.set_layer_visible("F.Cu", False)
    assert window.scene.pad_at(QPointF(10, 21)) is None  # SMD pad on a hidden layer can't be picked


# -- solving -------------------------------------------------------------------------


def _wait(app, condition, timeout=120.0):
    import time

    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError("condition not met")
        app.processEvents()
        time.sleep(0.01)


@pytest.fixture
def strip_kits(tmp_path):
    """The solver gate's strip board as a project with +5 A / -5 A sources."""
    from kits.config import CurrentSource

    from .test_solve import _HEADER, _pad, _strip

    board = tmp_path / "strip.kicad_pcb"
    board.write_text(_HEADER + _strip("F.Cu") + _pad("P1", 7, "F.Cu") + _pad("P2", 43, "F.Cu") + ")\n")
    project = KitsProject(
        tmp_path / "strip.kits",
        board=board,
        current_sources=[CurrentSource("P1", "1", 5.0), CurrentSource("P2", "1", -5.0)],
    )
    project.save()
    return project.path


def test_solve_in_child_process_shows_overlay_and_reloads(app, window, strip_kits):
    from kits.gui.field_display import DisplayMode

    assert window.open_path(strip_kits)
    panel, shown = window.solve_panel, window.results_panel
    assert panel.solve_button.isEnabled() and window.solve_action.isEnabled()
    assert window.document.results is None and not shown.legend.isVisibleTo(shown)
    assert "No results yet" in shown.summary.text()
    max_label = window._max_temperature_label
    assert max_label.isHidden()  # nothing to show in the status bar yet

    window.solve()
    assert window.solve_job_running and not panel.solve_button.isEnabled()
    assert not window.build_mesh_action.isEnabled()  # no meshing while solving
    _wait(app, lambda: not window.solve_job_running)

    results = window.document.results
    assert results is not None and results.total_power_w > 0
    assert window.document.mesh is not None  # the mesh the solver built is picked up
    scene = window.scene
    assert scene.results_overlay_visible("F.Cu")
    # The results are summarised in the Results tab (not the Solve tab), which is brought forward.
    assert "Joule power" in shown.summary.text()
    # No resistive elements: all of it is copper loss.
    assert f"Copper losses: {results.total_power_w:.4g} W (100.0%)" in shown.summary.text()
    assert "Resistive element losses: 0 W (0.0%)" in shown.summary.text()
    # Each source in its own table row: pad, current, voltage.
    sources = shown.table_rows("sources")
    assert [row[0] for row in sources] == ["P1.1", "P2.1"]
    assert sources[0][1] == "+5 A" and sources[1][1] == "-5 A" and sources[1][2].endswith("V")
    assert not shown._table_groups["sources"].isHidden() and shown._table_groups["elements"].isHidden()
    # Ctrl+C copies selected cells as tab-separated text.
    from kits.gui.results_panel import copy_cells

    table = shown.tables["sources"]
    table.selectAll()
    assert copy_cells(table).splitlines()[0] == "\t".join(sources[0])
    assert "Joule power" not in panel.status.text()
    # The status bar shows the highest temperature, and where it is.
    assert not max_label.isHidden()
    assert max_label.text() == f"Max {results.max_temperature_c:.3g} °C"
    assert "F.Cu" in max_label.toolTip() or "B.Cu" in max_label.toolTip()
    assert not window._results_dock.isHidden()
    assert shown.legend_high.text().startswith("≥") and "A/mm²" in shown.legend_high.text()

    # Display modes rescale the overlay; the legend follows.
    shown.mode.setCurrentIndex(shown.mode.findData(DisplayMode.TEMPERATURE))
    low, high = scene.results_range
    assert 25 <= low < high and "°C" in shown.legend_high.text()
    shown.mode.setCurrentIndex(shown.mode.findData(DisplayMode.VOLTAGE))
    assert scene.results_range[0] == pytest.approx(0.0, abs=1e-9)

    # Toggle like the mesh; only the active layer is shown.
    window.show_results_action.trigger()
    assert not scene.results_overlay_visible("F.Cu") and not shown.show_results.isChecked()
    window.show_results_action.trigger()
    scene.set_active_layer("B.Cu")
    assert not scene.results_overlay_visible("F.Cu")

    # Reopening reloads the saved results; editing a source makes them stale.
    window.close_project()
    assert window.open_path(strip_kits)
    assert window.document.results_from_file and window.scene.results_overlay_visible("F.Cu")
    window.document.set_source_current("P1", "1", 6.0)
    window.document.set_source_current("P2", "1", -6.0)
    assert "current sources" in window.document.results_staleness
    assert "solve again" in panel.status.text()
    assert window._max_temperature_label.text().endswith("(out of date)")
    assert "solve again" in window._max_temperature_label.toolTip()


def test_solve_disabled_without_balanced_sources(window, strip_kits, monkeypatch):
    warnings = []
    monkeypatch.setattr("kits.gui.main_window.QMessageBox.warning", lambda *a: warnings.append(a))
    assert window.open_path(strip_kits)
    window.document.set_source_current("P2", "1", -4.0)
    assert not window.solve_panel.solve_button.isEnabled()
    assert "sum to zero" in window.solve_panel.status.text()
    window.solve()
    assert warnings and not window.solve_job_running


def test_solve_failure_is_reported(app, window, strip_kits, monkeypatch):
    errors = []
    monkeypatch.setattr("kits.gui.main_window.QMessageBox.critical", lambda *a: errors.append(a))
    assert window.open_path(strip_kits)
    window.document.remove_current_source("P2", "1")
    window.document.add_current_source("Q9", "1", -5.0)  # no such pad
    window.solve()
    _wait(app, lambda: not window.solve_job_running)
    assert errors and "Q9.1" in errors[0][2]


def test_sources_table_leaves_no_stale_editors(window, strip_kits):
    from PySide6.QtWidgets import QDoubleSpinBox

    assert window.open_path(strip_kits)  # refreshes the table more than once
    window.show()
    table = window.sources_panel.table
    visible = [s for s in table.viewport().findChildren(QDoubleSpinBox) if s.isVisible()]
    assert len(visible) == table.rowCount() == 2


def test_solve_panel_edits_materials_and_environment(window, strip_kits):
    from kits.config import SolveSettings

    assert window.open_path(strip_kits)
    panel = window.solve_panel
    boxes = panel.setting_boxes
    assert boxes["copper_conductivity_s_m"].value() == pytest.approx(58.0)  # MS/m
    assert boxes["plating_thickness_mm"].value() == pytest.approx(25.0)  # µm

    boxes["ambient_c"].setValue(40.0)
    assert window.document.solve_settings.ambient_c == 40.0 and window.document.dirty
    boxes["plating_thickness_mm"].setValue(30.0)
    assert window.document.solve_settings.plating_thickness_mm == pytest.approx(0.03)

    # Via fill: the resin conductivity only applies (and is editable) for a resin fill.
    fill, resin = panel.via_fill, boxes["via_fill_thermal_conductivity_w_mk"]
    assert fill.currentData() == "none" and not resin.isEnabled()
    fill.setCurrentIndex(fill.findData("resin"))
    assert window.document.solve_settings.via_fill == "resin" and resin.isEnabled()
    resin.setValue(1.2)
    assert window.document.solve_settings.via_fill_thermal_conductivity_w_mk == pytest.approx(1.2)
    fill.setCurrentIndex(fill.findData("copper"))
    assert window.document.solve_settings.via_fill == "copper" and not resin.isEnabled()

    # Both faces insulated is rejected and blocks solving until fixed.
    boxes["convection_top_w_m2k"].setValue(0.0)
    boxes["convection_bottom_w_m2k"].setValue(0.0)
    assert not panel.settings_error.isHidden() and not panel.solve_button.isEnabled()
    boxes["convection_bottom_w_m2k"].setValue(5.0)
    assert panel.settings_error.isHidden() and panel.solve_button.isEnabled()

    assert window.save()
    saved = load_kits_project(strip_kits).solve
    assert saved == SolveSettings(
        ambient_c=40.0,
        plating_thickness_mm=0.03,
        convection_top_w_m2k=0.0,
        convection_bottom_w_m2k=5.0,
        via_fill="copper",
        via_fill_thermal_conductivity_w_mk=1.2,  # kept for when resin is chosen again
    )
    assert saved.copper_conductivity_s_m == 5.8e7  # untouched fields keep their exact values


def test_show_hide_current_sources(window, strip_kits):
    assert window.open_path(strip_kits)
    scene, panel = window.scene, window.sources_panel
    assert scene.sources_visible and scene.source_marker_count == 2

    window.show_sources_action.trigger()  # View menu / S
    assert not scene.sources_visible and not panel.show_sources.isChecked()

    # Stays hidden when sources change and markers are redrawn.
    window.document.set_source_current("P1", "1", 7.0)
    assert not scene.sources_visible and scene.source_marker_count == 2

    panel.show_sources.setChecked(True)  # panel checkbox
    assert scene.sources_visible and window.show_sources_action.isChecked()

    # Hidden markers don't stop pads being picked, and stay hidden across a board reload.
    from PySide6.QtCore import QPointF

    window.show_sources_action.trigger()
    assert window.scene.pad_at(QPointF(7, 5)) == ("P1", "1")
    window.reload_board()
    assert not window.scene.sources_visible


def test_status_bar_shows_value_under_cursor(app, window, strip_kits):
    from kits.gui.field_display import DisplayMode

    assert window.open_path(strip_kits)
    label = window._probe_label
    window._on_cursor_moved(25.0, 5.0)  # middle of the strip
    assert label.text() == ""  # nothing solved yet

    window.solve()
    _wait(app, lambda: not window.solve_job_running)
    window._on_cursor_moved(25.0, 5.0)
    # Uniform current density in the strip: I / (W t) = 5 A / (6 mm x 0.035 mm).
    text = label.text()
    assert text.startswith("F.Cu · Current density:") and text.endswith("A/mm²")
    value = float(text.split(":")[1].split()[0])
    assert value == pytest.approx(5 / (6 * 0.035), rel=0.01)

    # Changing the display mode updates the readout without moving the mouse.
    window.results_panel.mode.setCurrentIndex(window.results_panel.mode.findData(DisplayMode.TEMPERATURE))
    assert "Temperature:" in label.text() and label.text().endswith("°C")

    window._on_cursor_moved(25.0, 9.5)  # bare board beside the strip: temperature still shown
    assert "Temperature:" in label.text()
    window.results_panel.mode.setCurrentIndex(window.results_panel.mode.findData(DisplayMode.VOLTAGE))
    assert label.text() == ""  # voltage is copper-only
    window._on_cursor_moved(80.0, 80.0)  # off the board
    assert label.text() == ""

    window._on_cursor_moved(25.0, 5.0)
    assert "Voltage:" in label.text()
    window.show_results_action.trigger()  # hiding the overlay clears it
    assert label.text() == ""


def test_layers_panel_overlay_toggles(app, window, strip_kits):
    assert window.open_path(strip_kits)
    window.build_mesh()
    _wait(app, lambda: not window.mesh_job_running)
    window.solve()
    _wait(app, lambda: not window.solve_job_running)
    boxes = window.layer_panel.overlay_boxes
    assert list(boxes) == [
        "Mesh",
        "Results",
        "Vectors",
        "Current sources",
        "Heat loads",
        "Resistive elements",
        "Heatsinks",
        "Markers",
        "Grid",
    ]
    assert all(box.isChecked() for name, box in boxes.items() if name != "Vectors")
    assert not boxes["Vectors"].isChecked()  # flow arrows start off (V)
    scene = window.scene

    # Checkbox -> action and board.
    boxes["Mesh"].setChecked(False)
    assert not window.show_mesh_action.isChecked() and not scene.mesh_overlay_visible("F.Cu")
    boxes["Results"].setChecked(False)
    assert not window.show_results_action.isChecked() and not scene.results_overlay_visible("F.Cu")
    boxes["Current sources"].setChecked(False)
    assert not window.show_sources_action.isChecked() and not scene.sources_visible
    boxes["Grid"].setChecked(False)
    assert not window.settings.grid_visible and not scene.grid_visible

    # Menu actions / other panels -> checkbox.
    window.show_mesh_action.trigger()
    assert boxes["Mesh"].isChecked() and scene.mesh_overlay_visible("F.Cu")
    window.results_panel.show_results.setChecked(True)
    assert boxes["Results"].isChecked() and scene.results_overlay_visible("F.Cu")
    window.sources_panel.show_sources.setChecked(True)
    assert boxes["Current sources"].isChecked()
    window.settings.grid_visible = True
    assert boxes["Grid"].isChecked()


def test_mesh_panel_corner_size(window, kits_v9_path):
    assert window.open_path(kits_v9_path)
    box = window.mesh_panel.corner_size
    assert box.value() == pytest.approx(0.1)
    box.setValue(0.0)
    assert box.text() == "Off"
    assert window.document.project.mesh.corner_size_mm == 0.0 and window.document.dirty
    assert window.save()
    assert load_kits_project(kits_v9_path).mesh.corner_size_mm == 0.0


def test_log_scale_option(app, window, strip_kits):
    from kits.gui.field_display import DisplayMode

    assert window.open_path(strip_kits)
    panel = window.results_panel
    assert not panel.log_scale.isEnabled()  # no results yet
    window.solve()
    _wait(app, lambda: not window.solve_job_running)

    assert panel.log_scale.isChecked() and panel.log_scale.isEnabled()  # on by default for current density
    assert window.scene.results_log
    low, high = window.scene.results_range
    assert 0 < low < high and panel.legend_low.text().startswith("≤")
    assert "Log scale" in panel.legend_note.text()

    panel.log_scale.setChecked(False)
    assert not window.scene.results_log and window.scene.results_range[0] == 0.0
    assert not panel.legend_low.text().startswith("≤")

    panel.log_scale.setChecked(True)
    panel.mode.setCurrentIndex(panel.mode.findData(DisplayMode.TEMPERATURE))
    assert not panel.log_scale.isEnabled() and not window.scene.results_log  # temperature stays linear


def test_markers(app, window, strip_kits):
    from PySide6.QtCore import QPointF

    from kits.gui.field_display import DisplayMode

    assert window.open_path(strip_kits)
    window.build_mesh()
    _wait(app, lambda: not window.mesh_job_running)
    doc, scene, panel = window.document, window.scene, window.results_panel

    window._on_board_clicked(QPointF(25.0, 5.0))  # no results yet: no marker
    assert doc.markers == []

    window.solve()
    _wait(app, lambda: not window.solve_job_running)
    window._on_board_clicked(QPointF(25.0, 5.0))
    window._on_board_clicked(QPointF(80.0, 80.0))  # off the board: no marker
    window._on_board_clicked(QPointF(50.5, 5.0))  # just past the board edge: no marker
    window._on_board_clicked(QPointF(25.0, 9.5))  # on the board but off copper: marker without a value
    assert doc.markers == [(25.0, 5.0), (25.0, 9.5)]
    labels = scene.marker_labels
    assert labels[0].endswith("A/mm²") and labels[1] == "—"
    assert panel.markers_table.rowCount() == 2
    assert panel.markers_table.item(0, 2).text() == labels[0]

    # Labels follow the display mode.
    panel.mode.setCurrentIndex(panel.mode.findData(DisplayMode.TEMPERATURE))
    assert scene.marker_labels[0].endswith("°C") and panel.markers_table.item(0, 2).text().endswith("°C")

    # Toggle: action, Layers checkbox and panel checkbox stay in sync.
    window.show_markers_action.trigger()
    assert not scene.markers_visible and not panel.show_markers.isChecked()
    assert not window.layer_panel.overlay_boxes["Markers"].isChecked()
    panel.show_markers.setChecked(True)
    assert scene.markers_visible and window.show_markers_action.isChecked()

    # Remove the selected marker, then clear.
    panel.markers_table.selectRow(0)
    panel.remove_selected_markers()
    assert doc.markers == [(25.0, 9.5)] and panel.markers_table.rowCount() == 1
    panel.clear_markers_button.click()
    assert doc.markers == [] and scene.marker_labels == []

    # Hiding the results overlay stops clicks adding markers.
    window.show_results_action.trigger()
    window._on_board_clicked(QPointF(25.0, 5.0))
    assert doc.markers == []
    window.show_results_action.trigger()

    # Markers are cleared by new results and by a new mesh.
    window._on_board_clicked(QPointF(25.0, 5.0))
    window.solve()
    _wait(app, lambda: not window.solve_job_running)
    assert doc.markers == []
    window._on_board_clicked(QPointF(25.0, 5.0))
    window.build_mesh()
    _wait(app, lambda: not window.mesh_job_running)
    assert doc.markers == [] and panel.markers_table.rowCount() == 0


def test_double_click_pad_leaves_no_marker(app, window, strip_kits):
    from PySide6.QtCore import QPointF

    assert window.open_path(strip_kits)
    window.build_mesh()
    _wait(app, lambda: not window.mesh_job_running)
    window.solve()
    _wait(app, lambda: not window.solve_job_running)
    pos = QPointF(7, 5)
    assert window.scene.pad_at(pos) == ("P1", "1")
    window._on_board_clicked(pos)  # first click of the double-click
    assert len(window.document.markers) == 1
    window._on_board_double_clicked(pos)
    assert window.document.markers == []

    window._on_board_clicked(QPointF(25.0, 5.0))  # double-click off a pad keeps its one marker
    window._on_board_double_clicked(QPointF(25.0, 5.0))
    assert window.document.markers == [(25.0, 5.0)]


def test_current_multiplier(app, window, strip_kits):
    assert window.open_path(strip_kits)
    doc, panel = window.document, window.sources_panel
    assert panel.multiplier.value() == 1.0 and "Solved with" not in panel.total_label.text()
    window.build_mesh()
    _wait(app, lambda: not window.mesh_job_running)
    window.solve()
    _wait(app, lambda: not window.solve_job_running)
    assert doc.results is not None and doc.results_staleness is None
    base = doc.results.electrode_currents["P1.1"]

    panel.multiplier.setValue(2.0)
    assert doc.current_multiplier == 2.0 and doc.dirty
    assert "Solved with × 2" in panel.total_label.text()
    assert [s.current_a for s in doc.scaled_current_sources] == [2 * s.current_a for s in doc.current_sources]
    assert doc.results_staleness is not None  # results no longer match the inputs

    window.solve()
    _wait(app, lambda: not window.solve_job_running)
    assert doc.results.electrode_currents["P1.1"] == pytest.approx(2 * base, rel=1e-6)

    assert window.save()
    assert load_kits_project(strip_kits).current_multiplier == 2.0


def test_heat_loads_panel(app, window, strip_kits):
    from PySide6.QtCore import QPointF

    from kits.config import HeatLoad, HeatLoadKind

    assert window.open_path(strip_kits)
    doc, panel, scene = window.document, window.heat_panel, window.scene
    sources_before = doc.current_sources

    # Double-click adds to whichever of Current Sources / Heat Loads was shown last.
    window._heat_dock.show()
    window._heat_dock.raise_()
    window._on_load_dock_shown(window._heat_dock, True)  # docks don't report visibility offscreen
    window._on_board_double_clicked(QPointF(7, 5))
    assert doc.heat_loads == [HeatLoad("P1", "1", HeatLoadKind.POWER, 1.0)]
    assert doc.current_sources == sources_before and doc.dirty
    assert panel.table.rowCount() == 1 and scene.heat_load_marker_count == 1

    # Edit the value, then switch kind: the value restarts at that kind's default.
    panel.table.cellWidget(0, 3).setValue(2.5)
    assert doc.heat_load("P1", "1").value == 2.5
    combo = panel.table.cellWidget(0, 2)
    combo.setCurrentIndex(combo.findData(HeatLoadKind.TEMPERATURE))
    assert doc.heat_load("P1", "1") == HeatLoad(
        "P1", "1", HeatLoadKind.TEMPERATURE, doc.solve_settings.ambient_c
    )
    assert panel.table.cellWidget(0, 3).suffix() == " °C"
    combo = panel.table.cellWidget(0, 2)
    combo.setCurrentIndex(combo.findData(HeatLoadKind.RESISTANCE))
    assert doc.heat_load("P1", "1").value == 10.0 and panel.table.cellWidget(0, 3).suffix() == " K/W"

    # Show/hide: action, panel checkbox and Layers overlay box stay in sync.
    window.show_heat_loads_action.trigger()
    assert not scene.heat_loads_visible and not panel.show_heat_loads.isChecked()
    assert not window.layer_panel.overlay_boxes["Heat loads"].isChecked()
    panel.show_heat_loads.setChecked(True)
    assert scene.heat_loads_visible and window.show_heat_loads_action.isChecked()

    # A heat-only solve: no current sources needed.
    for source in doc.current_sources:
        doc.remove_current_source(source.ref, source.pad)
    doc.add_heat_load("P2", "1", HeatLoadKind.POWER, 3.0)
    assert window.solve_panel.solve_button.isEnabled()
    window.solve()
    _wait(app, lambda: not window.solve_job_running)
    results = doc.results
    assert results is not None and results.load_heat["P2.1"] == 3.0
    assert results.load_heat["P1.1"] < 0  # the 10 K/W heatsink takes some of it out
    loads = {row[0]: row for row in window.results_panel.table_rows("loads")}
    assert loads["P2.1"][1] == "+3 W" and loads["P1.1"][1].startswith("-") and loads["P2.1"][2].endswith("°C")
    assert load_kits_project(strip_kits).heat_loads == doc.heat_loads  # saved before solving

    # Remove via the panel; editing a load makes the results stale.
    panel.table.selectRow(0)
    panel.remove_selected_rows()
    assert [h.key for h in doc.heat_loads] == [("P2", "1")]
    assert doc.results_staleness is not None


@pytest.fixture
def bridged_kits(tmp_path):
    """Two strips joined only by resistor R1 (value "10m"), with +5 A / -5 A at the ends."""
    from kits.config import CurrentSource

    from .test_solve import _HEADER, _RESISTOR, _pad, _zone

    board = tmp_path / "bridged.kicad_pcb"
    resistor = _RESISTOR.replace(
        '(property "Reference" "R1")', '(property "Reference" "R1") (property "Value" "10m")'
    )
    pads = _pad("P1", 7, "F.Cu") + _pad("P2", 43, "F.Cu")
    board.write_text(_HEADER + _zone(5, 22) + _zone(28, 45) + pads + resistor + ")\n")
    project = KitsProject(
        tmp_path / "bridged.kits",
        board=board,
        current_sources=[CurrentSource("P1", "1", 5.0), CurrentSource("P2", "1", -5.0)],
    )
    project.save()
    return project.path


def test_resistive_elements_panel(app, window, bridged_kits, monkeypatch):
    from PySide6.QtCore import QPointF

    from kits.config import PadGroup, ResistiveElement

    assert window.open_path(bridged_kits)
    doc, panel, scene = window.document, window.elements_panel, window.scene
    shown = []

    def accept(dialog):
        shown.append(dialog)
        return dialog.element

    monkeypatch.setattr(panel, "run_dialog", accept)

    # Double-click a pad while the Elements tab is the last shown: an element is
    # suggested for its footprint, with the resistance read from the value "10m".
    window._on_load_dock_shown(window._elements_dock, True)
    window._on_board_double_clicked(QPointF(20, 5))
    assert doc.current_sources and not doc.heat_loads
    assert doc.resistive_elements == [
        ResistiveElement("R1", "R1", PadGroup("A", ["1"]), PadGroup("B", ["2"]), resistance_ohm=0.01)
    ]
    assert panel.table.rowCount() == 1 and scene.element_marker_count == 1 and doc.dirty
    dialog = shown[-1]
    assert dialog.ambient_default.isChecked() and "K/W" in dialog.ambient_default.text()

    # A second element on the same footprint gets a fresh name.
    window._on_board_double_clicked(QPointF(20, 5))
    assert [e.name for e in doc.resistive_elements] == ["R1", "R1_2"]
    doc.remove_resistive_element("R1_2")

    # Edit: typed SI values, thermal resistances; invalid entries disable OK.
    def edit(dialog):
        dialog.resistance.setText("4R7")
        dialog.terminal_default[0].setChecked(False)
        dialog.terminal_theta[0].setValue(20.0)
        dialog.group_pads[1].item(0).setCheckState(Qt.CheckState.Checked)  # pad 1 in both groups
        assert dialog.element is None and "both groups" in dialog.error.text()
        dialog.group_pads[1].item(0).setCheckState(Qt.CheckState.Unchecked)
        dialog.resistance.setText("lots")
        assert dialog.element is None
        dialog.resistance.setText("5m")
        dialog.name.setText("Shunt")
        return dialog.element

    monkeypatch.setattr(panel, "run_dialog", edit)
    panel.table.selectRow(0)
    panel.edit_selected()
    shunt = doc.resistive_element("Shunt")
    assert shunt.resistance_ohm == pytest.approx(5e-3) and shunt.thermal_resistance_a_k_w == 20.0
    assert shunt.thermal_resistance_b_k_w is None  # left at the body-type default
    assert shunt.thermal_resistance_ambient_k_w is None

    # Show/hide stays in sync.
    window.show_elements_action.trigger()
    assert not scene.elements_visible and not panel.show_elements.isChecked()
    assert not window.layer_panel.overlay_boxes["Resistive elements"].isChecked()
    panel.show_elements.setChecked(True)
    assert scene.elements_visible

    # Solve: all 5 A goes through the shunt; the panel lists it.
    window.solve()
    _wait(app, lambda: not window.solve_job_running)
    results = doc.results
    assert results.element_currents["Shunt"] == pytest.approx(5.0, rel=1e-6)
    assert results.element_power["Shunt"] == pytest.approx(25 * 5e-3, rel=1e-6)
    assert results.element_temperatures["Shunt"] > 25.0
    if results.max_temperature_c == results.element_temperatures["Shunt"]:  # the body is the hottest
        assert "Shunt body" in window._max_temperature_label.toolTip()
    # The losses split: the shunt's I²R, and the copper's share of the rest.
    assert results.element_losses_w == pytest.approx(25 * 5e-3, rel=1e-6)
    assert results.copper_losses_w + results.element_losses_w == pytest.approx(results.total_power_w)
    summary = window.results_panel.summary.text()
    share = 100 * results.element_losses_w / results.total_power_w
    assert f"Resistive element losses: {results.element_losses_w:.4g} W ({share:.1f}%)" in summary
    assert f"Copper losses: {results.copper_losses_w:.4g} W ({100 - share:.1f}%)" in summary
    (shunt_row,) = window.results_panel.table_rows("elements")
    assert shunt_row[:3] == ["Shunt", "+5 A", "0.125 W"] and shunt_row[3].endswith("°C")
    # Heat out through each terminal and to ambient balances the power.
    to_a, to_b, ambient = (float(cell.split()[0]) for cell in shunt_row[4:])
    assert to_a + to_b + ambient == pytest.approx(0.125, rel=1e-3)
    assert "Results tab" in window.solve_panel.status.text()  # the Solve tab points there
    # The body is drawn on the temperature map of its side (F.Cu), and the scale covers it.
    from kits.gui.field_display import DisplayMode

    scene = window.scene
    mode = window.results_panel.mode
    # The toolbar drop-down switches the display, and follows the panel's choice.
    box = window.display_mode_box
    assert box.isEnabled()
    box.setCurrentIndex(box.findData(DisplayMode.VOLTAGE))
    assert window.results_panel.display_mode is DisplayMode.VOLTAGE
    mode.setCurrentIndex(mode.findData(DisplayMode.TEMPERATURE))
    assert box.currentData() is DisplayMode.TEMPERATURE
    scene.set_active_layer("F.Cu")
    assert scene.body_temperature_count == 1
    assert scene.results_range[1] >= results.element_temperatures["Shunt"]
    assert results.max_temperature_c >= results.element_temperatures["Shunt"]
    scene.set_active_layer("B.Cu")
    assert scene.body_temperature_count == 0  # a top-side part
    mode.setCurrentIndex(mode.findData(DisplayMode.CURRENT_DENSITY))
    scene.set_active_layer("F.Cu")
    assert scene.body_temperature_count == 0  # only on the temperature map
    assert load_kits_project(bridged_kits).resistive_elements == doc.resistive_elements

    # Remove; results go stale.
    panel.table.selectRow(0)
    panel.remove_selected_rows()
    assert doc.resistive_elements == [] and doc.results_staleness is not None


def test_toolbar_quick_actions(window, kits_v9_path):
    from kits.gui import colors

    toolbar = window.toolbar
    buttons = [a for a in toolbar.actions() if not a.isSeparator()][:2]
    assert buttons == [window.build_mesh_action, window.solve_action]
    assert not any(a.isEnabled() for a in buttons)  # no board yet
    # After Solve: the display mode drop-down (disabled until there are results).
    widgets = [toolbar.widgetForAction(a) for a in toolbar.actions()]
    box = window.display_mode_box
    assert box in widgets and not box.isEnabled()
    # Then the layer selector, after it (empty and disabled with no board).
    assert widgets.index(window.layer_box) > widgets.index(box)
    assert window.layer_box.count() == 0 and not window.layer_box.isEnabled()
    assert "F9" in window.solve_action.toolTip()
    # Distinct solid colours so they stand out on the dark theme.
    mesh_style, solve_style = (toolbar.widgetForAction(a).styleSheet() for a in buttons[:2])
    assert colors.MESH_BUTTON.name() in mesh_style and colors.SOLVE_BUTTON.name() in solve_style
    assert colors.MESH_BUTTON != colors.SOLVE_BUTTON
    # The panels' buttons for the same actions match.
    assert colors.MESH_BUTTON.name() in window.mesh_panel.build_button.styleSheet()
    assert colors.SOLVE_BUTTON.name() in window.solve_panel.solve_button.styleSheet()
    assert window.open_path(kits_v9_path)
    assert all(a.isEnabled() for a in buttons)
    view_menu = next(a.menu() for a in window.menuBar().actions() if a.text() == "&View")
    assert window.toolbar.toggleViewAction() in view_menu.actions()  # can be hidden from the View menu


def test_flow_vectors_toggle(app, window, strip_kits):
    assert window.open_path(strip_kits)
    window.solve()
    _wait(app, lambda: not window.solve_job_running)
    scene, action, box = window.scene, window.show_vectors_action, window.results_panel.show_vectors
    assert action.shortcut().toString() == "V" and not action.isChecked() and scene.vector_count == 0
    assert "Vectors" in window.layer_panel.overlay_boxes

    action.trigger()  # V: arrows on, the overlay rebuilt with them
    assert (
        scene.vectors_visible and box.isChecked() and window.layer_panel.overlay_boxes["Vectors"].isChecked()
    )
    assert scene.results_overlay_visible("F.Cu") and scene.vector_count > 50
    # They follow the display mode (heat flow in the temperature view covers the bare board too).
    copper_arrows = scene.vector_count
    mode = window.results_panel.mode
    from kits.gui.field_display import DisplayMode

    mode.setCurrentIndex(mode.findData(DisplayMode.TEMPERATURE))
    assert scene.vector_count > copper_arrows
    box.setChecked(False)  # the Results tab checkbox works too
    assert not action.isChecked() and not scene.vectors_visible and scene.vector_count == 0
    assert scene.results_overlay_visible("F.Cu")  # rebuilt without arrows


def test_flow_vectors_follow_the_zoom(app, window, strip_kits):
    window.resize(1000, 700)
    window.show()
    assert window.open_path(strip_kits)
    window.solve()
    _wait(app, lambda: not window.solve_job_running)
    scene, view = window.scene, window.view
    window.show_vectors_action.trigger()
    fitted = scene.vector_spacing
    # About _VECTOR_SPACING_PX pixels apart on screen, as a power of two in mm.
    assert fitted * view.pixels_per_mm == pytest.approx(30, rel=0.5)
    assert np.log2(fitted) == pytest.approx(round(np.log2(fitted)))

    # Zoom in: once zooming pauses the arrows are redrawn closer together, over the visible area only.
    for _ in range(4):
        view.zoom_in()
    _wait(app, lambda: scene.vector_spacing is not None and scene.vector_spacing < fitted)
    zoomed = scene.vector_spacing
    assert zoomed * view.pixels_per_mm == pytest.approx(30, rel=0.5)
    visible = view.visible_scene_rect()
    assert scene.vector_count * zoomed**2 <= visible.width() * visible.height() * 1.2
    # And back out: further apart again.
    for _ in range(4):
        view.zoom_out()
    _wait(app, lambda: scene.vector_spacing == fitted)


def test_pass_fail_limits(app, window, strip_kits):
    assert window.open_path(strip_kits)
    solve, shown = window.solve_panel, window.results_panel
    window.solve()
    _wait(app, lambda: not window.solve_job_running)
    results = window.document.results
    assert shown.verdict.isHidden() and window._verdict_label.isHidden()  # no limits yet
    assert shown._table_groups["limits"].isHidden()

    # Set limits on the Solve tab: a generous current density, a temperature just under the max.
    checks, values = solve.limit_checks, solve.limit_values
    assert not values["max_temperature_c"].isEnabled()
    checks["max_current_density_a_mm2"].setChecked(True)
    values["max_current_density_a_mm2"].setValue(1000.0)
    checks["max_temperature_c"].setChecked(True)
    values["max_temperature_c"].setValue(results.max_temperature_c - 0.5)
    doc = window.document
    assert doc.limits.max_current_density_a_mm2 == 1000.0 and doc.dirty
    # Judged straight away, without solving again: the results stay current.
    assert doc.results_staleness is None
    rows = shown.table_rows("limits")
    assert [row[1] for row in rows] == ["PASS", "FAIL"]
    assert rows[1][0].startswith("Temperature ≤") and rows[1][3].startswith("-")  # negative margin
    assert shown.verdict.text() == "FAIL" and not shown.verdict.isHidden()
    assert window._verdict_label.text().strip() == "FAIL" and "Temperature" in window._verdict_label.toolTip()

    values["max_temperature_c"].setValue(results.max_temperature_c + 5)
    assert shown.verdict.text() == "PASS" and window._verdict_label.text().strip() == "PASS"
    # Switching a limit off drops it.
    checks["max_current_density_a_mm2"].setChecked(False)
    assert doc.limits.max_current_density_a_mm2 is None and len(shown.table_rows("limits")) == 1

    assert window.save()
    assert load_kits_project(strip_kits).limits == doc.limits
    # Reopened: the limits are shown on the Solve tab again.
    window.close_project()
    assert window.open_path(strip_kits)
    assert checks["max_temperature_c"].isChecked() and not checks["max_current_density_a_mm2"].isChecked()
    assert values["max_temperature_c"].value() == pytest.approx(round(results.max_temperature_c + 5, 1))


def test_generate_report_from_toolbar_and_results_tab(app, window, strip_kits, tmp_path, monkeypatch):
    assert window.open_path(strip_kits)
    action, button = window.report_action, window.results_panel.report_button
    assert action in window.toolbar.actions()
    assert not action.isEnabled() and not button.isEnabled()  # no results yet
    window.solve()
    _wait(app, lambda: not window.solve_job_running)
    assert action.isEnabled() and button.isEnabled()

    # Cancelling the save dialog writes nothing.
    monkeypatch.setattr(window, "ask_report_path", lambda suggestion: None)
    button.click()
    assert not window.report_job_running

    target = tmp_path / "reports" / "strip report.html"
    asked, opened = [], []
    monkeypatch.setattr(window, "ask_report_path", lambda suggestion: asked.append(suggestion) or target)
    monkeypatch.setattr(window, "open_report", lambda path: opened.append(path) or True)
    action.trigger()
    assert asked[0].name == "strip-report.html"  # suggested next to the project
    assert window.report_job_running and not action.isEnabled() and not window.solve_action.isEnabled()
    _wait(app, lambda: not window.report_job_running)
    page = target.read_text(encoding="utf-8")
    assert "Field maps" in page and "Board stackup" in page
    # It opens once written, and the status bar says where it went.
    assert opened == [target]
    message = window.statusBar().currentMessage()
    assert str(target) in message and "opened" in message
    assert action.isEnabled()
    # The next report starts in the same folder; if it can't be opened, the status bar says so.
    monkeypatch.setattr(window, "open_report", lambda path: False)
    action.trigger()
    _wait(app, lambda: not window.report_job_running)
    assert asked[1].parent == target.parent
    assert "couldn't be opened" in window.statusBar().currentMessage()


def test_toolbar_layer_selector_follows_the_active_layer(window, kits_v9_path):
    assert window.open_path(kits_v9_path)
    box, scene = window.layer_box, window.scene
    # The board's copper layers, top to bottom; the active one selected.
    assert [box.itemText(i) for i in range(box.count())] == ["F.Cu", "In1.Cu", "B.Cu"]
    assert box.isEnabled() and box.currentText() == scene.active_layer == "F.Cu"
    # Choosing a layer in the toolbar makes it active (and selects it in the Layers panel)...
    box.setCurrentIndex(box.findText("B.Cu"))
    assert scene.active_layer == "B.Cu"
    assert window.layer_panel._list.currentItem().text() == "B.Cu"
    # ...and selecting one in the Layers panel updates the toolbar.
    window.layer_panel.select_layer("In1.Cu")
    assert scene.active_layer == "In1.Cu" and box.currentText() == "In1.Cu"
    window.layer_panel.select_layer("Edge.Cuts")  # not copper: ignored
    assert scene.active_layer == "In1.Cu"
    # Closing the project empties it.
    window.close_project()
    assert box.count() == 0 and not box.isEnabled()


def test_heatsinks_panel(app, window, strip_kits, monkeypatch):
    from PySide6.QtCore import QPointF, QRectF

    from kits.config import Heatsink

    assert window.open_path(strip_kits)
    doc, panel, scene = window.document, window.heatsinks_panel, window.scene
    shown = []

    def accept(dialog):
        shown.append(dialog)
        return dialog.heatsink

    monkeypatch.setattr(panel, "run_dialog", accept)

    # Draw: the panel's button puts the view in draw mode; a drawn rectangle opens the dialog.
    panel.draw_button.click()
    assert window.view.drawing
    window.view.rect_drawn.emit(QRectF(20.0, 2.0, 10.0, 6.0))
    assert doc.heatsinks == [Heatsink("HS1", "top", 20.0, 2.0, 10.0, 6.0)]
    assert scene.heatsink_count == 1 and doc.dirty
    # The dialog works in grid coordinates: origin at the board's bottom-left (0, 10), Y up.
    dialog = shown[-1]
    assert (dialog.x.value(), dialog.y.value()) == pytest.approx((20.0, 2.0))
    assert panel.table.item(0, 3).text() == "20.00, 2.00"
    window.view.cancel_rect_draw()  # no-op once drawn

    # Double-click the heatsink on the board to edit it: move it and put it underneath.
    def edit(dialog):
        dialog.y.setValue(0.0)
        dialog.side.setCurrentIndex(dialog.side.findData("bottom"))
        dialog.ambient_theta.setValue(2.0)
        return dialog.heatsink

    monkeypatch.setattr(panel, "run_dialog", edit)
    window._on_board_double_clicked(QPointF(25.0, 5.0))
    hs = doc.heatsink("HS1")
    assert (hs.side, hs.y_mm, hs.thermal_resistance_ambient_k_w) == ("bottom", 4.0, 2.0)

    # Show/hide and the enable checkbox.
    window.show_heatsinks_action.trigger()
    assert not scene.heatsinks_visible and not panel.show_heatsinks.isChecked()
    panel.show_heatsinks.setChecked(True)
    assert scene.heatsinks_visible
    panel.table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    assert not doc.heatsink("HS1").enabled and scene.heatsink_count == 0
    panel.table.item(0, 0).setCheckState(Qt.CheckState.Checked)

    # Solve: the heatsink takes heat and is listed; it is saved with the project.
    window.solve()
    _wait(
        app,
        lambda: not window.solve_job_running and doc.results is not None and doc.results_staleness is None,
    )
    assert doc.results.heatsink_heat["HS1"] > 0
    ((name, heat, temperature),) = window.results_panel.table_rows("heatsinks")
    assert name == "HS1" and heat.endswith("W") and temperature.endswith("°C")
    assert load_kits_project(strip_kits).heatsinks == doc.heatsinks

    # 3D: a block under the board.
    window.show()
    window.view_3d_action.trigger()
    _wait(app, lambda: not window.view3d.busy)
    assert "heatsink:HS1" in window.view3d.visible_part_keys()
    window.show_heatsinks_action.trigger()
    assert "heatsink:HS1" not in window.view3d.visible_part_keys()
    window.view3d.shutdown()

    panel.table.selectRow(0)
    panel.remove_selected_rows()
    assert doc.heatsinks == [] and scene.heatsink_count == 0


def test_heatsink_dialog_mount(app):
    from kits.config import Heatsink
    from kits.gui.heatsinks_panel import GridFrame, HeatsinkDialog

    dialog = HeatsinkDialog(Heatsink("HS1", "top", 20.0, 2.0, 10.0, 6.0), set(), GridFrame((0.0, 10.0)))
    assert dialog.heatsink.mount == "board"
    dialog.mount.setCurrentIndex(dialog.mount.findData("components"))
    assert dialog.heatsink == Heatsink("HS1", "top", 20.0, 2.0, 10.0, 6.0, mount="components")


def test_view_menu_has_the_results_and_markers_toggles(window):
    menus = {a.text(): a.menu() for a in window.menuBar().actions()}
    view, solve = menus["&View"].actions(), menus["&Solve"].actions()
    for act in (window.show_results_action, window.show_markers_action):
        assert act in view and act not in solve
    assert window.solve_action in solve and any(a.text() == "Clear markers" for a in solve)
