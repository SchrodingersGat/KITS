import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from kits.gui.board_scene import HOLES, MODEL_ROLE, TH_PADS, VIAS  # noqa: E402
from kits.gui.main_window import MainWindow  # noqa: E402
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
