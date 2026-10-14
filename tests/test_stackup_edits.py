"""Editable layer thicknesses (copper weight, dielectric): model, project file, solver and GUI."""

import pytest
import yaml

pytest.importorskip("skfem")
pytest.importorskip("PySide6")

from kits.config import ConfigError, MeshSettings, StackupEdits  # noqa: E402
from kits.ingest import load_board  # noqa: E402
from kits.mesh.fingerprint import board_hash  # noqa: E402
from kits.mesh.mesher import layer_heights, mesh_board  # noqa: E402
from kits.project import KitsProject, load_kits_project  # noqa: E402
from kits.solve import cli  # noqa: E402
from kits.solve.solver import Electrode, LayeredModel  # noqa: E402

from .test_gui_project import _wait, app, settings, strip_kits, window  # noqa: E402, F401
from .test_solve import _HEADER, R_SQUARE, _pad, _strip  # noqa: E402


@pytest.fixture
def strip_board(tmp_path):
    path = tmp_path / "strip.kicad_pcb"
    path.write_text(_HEADER + _strip("F.Cu") + _pad("P1", 7, "F.Cu") + _pad("P2", 43, "F.Cu") + ")\n")
    return load_board(path)


def test_with_thickness_overrides(strip_board):
    edited = strip_board.with_thickness_overrides({"F.Cu": 0.07, "dielectric 1": 1.0})
    assert edited.copper_thicknesses()["F.Cu"] == (0.07, True)
    assert strip_board.copper_thicknesses()["F.Cu"] == (0.035, True)  # the original is untouched
    heights = layer_heights(edited)
    assert heights["F.Cu"] - heights["B.Cu"] == pytest.approx(0.07 / 2 + 1.0 + 0.035 / 2)
    assert strip_board.with_thickness_overrides({}) is strip_board


def test_overrides_for_copper_missing_from_the_stackup(tmp_path):
    header = _HEADER.replace('    (layer "F.Cu" (type "copper") (thickness 0.035))\n', "")
    path = tmp_path / "no_fcu.kicad_pcb"
    path.write_text(header + _strip("F.Cu") + ")\n")
    board = load_board(path)
    assert board.copper_thicknesses()["F.Cu"][1] is False  # assumed
    edited = board.with_thickness_overrides({"F.Cu": 0.105})
    assert edited.copper_thicknesses()["F.Cu"] == (0.105, True)
    assert edited.copper_thickness_overrides == {"F.Cu": 0.105}


def test_project_round_trip_and_validation(tmp_path):
    project = KitsProject(tmp_path / "p.kits", stackup_edits=StackupEdits("abc", {"F.Cu": 0.07}))
    project.save()
    data = yaml.safe_load(project.path.read_text(encoding="utf-8"))
    assert data["stackup_edits"] == {"board_hash": "abc", "thickness_mm": {"F.Cu": 0.07}}
    loaded = load_kits_project(project.path)
    assert loaded.stackup_edits == project.stackup_edits
    assert loaded.stackup_edits_for("abc") == {"F.Cu": 0.07}
    assert loaded.stackup_edits_for("other") == {}  # made on a different board version

    for bad in ({"thickness_mm": {"F.Cu": 0}}, {"thickness_mm": {"F.Cu": 20}}, {"thickness_mm": [1]}, 3):
        project.path.write_text(yaml.safe_dump({"stackup_edits": bad}), encoding="utf-8")
        with pytest.raises(ConfigError, match="stackup"):
            load_kits_project(project.path)


def test_thicker_copper_needs_a_new_solve_not_a_new_mesh(strip_board):
    mesh = mesh_board(strip_board, MeshSettings(max_size_mm=1.0, min_size_mm=0.1))
    electrodes = [Electrode("P1", "1", voltage=1.0), Electrode("P2", "1", voltage=0.0)]
    base = LayeredModel(strip_board, mesh).solve_dc(electrodes)
    # Same mesh, 2 oz copper: exactly half the resistance.
    thick = LayeredModel(strip_board.with_thickness_overrides({"F.Cu": 0.07}), mesh).solve_dc(electrodes)
    r_base = base.resistance(("P1", "1"), ("P2", "1"))
    assert r_base == pytest.approx(R_SQUARE * 32 / 6, rel=1e-6)
    assert thick.resistance(("P1", "1"), ("P2", "1")) == pytest.approx(r_base / 2, rel=1e-6)
    # Layer heights follow the edited stackup, not the mesh.
    thin = LayeredModel(strip_board.with_thickness_overrides({"dielectric 1": 0.5}), mesh)
    assert thin.by_name["F.Cu"].z - thin.by_name["B.Cu"].z == pytest.approx((0.5 + 0.035) * 1e-3)


def test_command_line_solve_applies_edits_for_the_same_board(strip_kits):  # noqa: F811
    project = load_kits_project(strip_kits)
    plain, _, _ = cli.solve_project(strip_kits)
    project.stackup_edits = StackupEdits(board_hash(project.board), {"F.Cu": 0.07})
    project.save()
    edited, _, _ = cli.solve_project(strip_kits)
    assert edited.total_power_w == pytest.approx(plain.total_power_w / 2, rel=1e-3)  # same current, half R
    assert edited.inputs != plain.inputs
    # Edits made on another version of the board are ignored.
    project.stackup_edits = StackupEdits("someone else's board", {"F.Cu": 0.07})
    project.save()
    stale, _, _ = cli.solve_project(strip_kits)
    assert stale.total_power_w == pytest.approx(plain.total_power_w, rel=1e-9)


def test_editing_in_the_gui(app, window, strip_kits):  # noqa: F811
    from PySide6.QtWidgets import QDoubleSpinBox

    assert window.open_path(strip_kits)
    window.solve()
    _wait(app, lambda: not window.solve_job_running and window.document.results is not None)
    doc, panel = window.document, window.info_panel
    assert doc.results_staleness is None

    # The Weight cell of F.Cu is a spin box in oz.
    table = panel._stackup
    row = next(r for r in range(table.rowCount()) if table.item(r, 0).text() == "F.Cu")
    weight = table.cellWidget(row, 3)
    assert isinstance(weight, QDoubleSpinBox) and weight.value() == pytest.approx(1.0)
    weight.setValue(2.0)
    assert doc.layer_thickness_edits == {"F.Cu": pytest.approx(0.07)}
    assert doc.board.copper_thicknesses()["F.Cu"][0] == pytest.approx(0.07)
    assert doc.file_board.copper_thicknesses()["F.Cu"][0] == pytest.approx(0.035)
    assert doc.dirty and doc.results_staleness is not None  # re-solve needed
    assert doc.mesh is not None and doc.mesh_notice is None  # ...but the mesh is still good
    assert panel.reset_button.isVisibleTo(panel) and "edited" in panel._edits_note.text()
    assert table.cellWidget(row, 3).font().bold()

    # Dielectric thickness too; the 3D view gets the edited board.
    dielectric = next(r for r in range(table.rowCount()) if table.item(r, 0).text() == "dielectric 1")
    table.cellWidget(dielectric, 2).setValue(1.0)
    assert doc.layer_thickness_edits["dielectric 1"] == pytest.approx(1.0)
    assert window.view3d._board is doc.board

    # Saved with the project; setting a value back to the board's drops the edit.
    assert window.save()
    assert load_kits_project(strip_kits).stackup_edits.thickness_mm == doc.layer_thickness_edits
    doc.set_layer_thickness("dielectric 1", 1.51)
    assert set(doc.layer_thickness_edits) == {"F.Cu"}
    with pytest.raises(ConfigError):
        doc.set_layer_thickness("F.Cu", 0.0)

    # Solving again uses the edits: twice the copper, half the power.
    power = doc.results.total_power_w
    window.solve()
    _wait(app, lambda: not window.solve_job_running and doc.results_staleness is None)
    assert doc.results.total_power_w == pytest.approx(power / 2, rel=1e-3)

    # Re-loading the board resets the edits, and says so.
    window.reload_board()
    assert doc.layer_thickness_edits == {} and doc.project.stackup_edits is None
    assert "reset" in doc.stackup_notice and "reset" in panel._edits_note.text()
    assert not panel.reset_button.isVisibleTo(panel)


def test_edits_dropped_when_the_board_file_changes(window, strip_kits):  # noqa: F811
    project = load_kits_project(strip_kits)
    project.stackup_edits = StackupEdits(board_hash(project.board), {"F.Cu": 0.07})
    project.save()
    assert window.open_path(strip_kits)
    assert window.document.layer_thickness_edits == {"F.Cu": 0.07}  # same board: kept
    window.close_project()

    project.board.write_text(project.board.read_text() + "\n")  # the board was edited in KiCad
    assert window.open_path(strip_kits)
    assert window.document.layer_thickness_edits == {} and window.document.stackup_notice
