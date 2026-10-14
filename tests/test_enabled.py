"""Enable/disable flags on current sources, heat loads and resistive elements."""

import dataclasses

import pytest
import yaml

pytest.importorskip("skfem")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402

from kits.config import (  # noqa: E402
    ConfigError,
    CurrentSource,
    HeatLoad,
    HeatLoadKind,
    PadGroup,
    ResistiveElement,
)
from kits.project import KitsProject, load_kits_project  # noqa: E402
from kits.solve import cli  # noqa: E402
from kits.solve.results import inputs_hash  # noqa: E402

from .test_gui_project import _wait, app, bridged_kits, settings, window  # noqa: E402, F401
from .test_solve import _HEADER, _RESISTOR, _pad, _zone  # noqa: E402

POWER = HeatLoadKind.POWER


def _element(name: str = "R1", **kwargs) -> ResistiveElement:
    return ResistiveElement(name, "R1", PadGroup("A", ["1"]), PadGroup("B", ["2"]), 0.1, **kwargs)


def test_enabled_flag_round_trip(tmp_path):
    project = KitsProject(
        tmp_path / "p.kits",
        current_sources=[CurrentSource("J1", "1", 2.0), CurrentSource("J2", "1", -2.0, enabled=False)],
        heat_loads=[HeatLoad("U1", "1", POWER, 1.0, enabled=False)],
        resistive_elements=[_element(enabled=False)],
    )
    project.save()
    data = yaml.safe_load(project.path.read_text(encoding="utf-8"))
    assert "enabled" not in data["current_sources"][0]  # only written when off
    assert data["current_sources"][1]["enabled"] is False
    assert data["heat_loads"][0]["enabled"] is False
    assert data["resistive_elements"][0]["enabled"] is False
    loaded = load_kits_project(project.path)
    assert loaded.current_sources == project.current_sources
    assert loaded.heat_loads == project.heat_loads
    assert loaded.resistive_elements == project.resistive_elements
    # Only enabled ones are solved.
    assert loaded.scaled_current_sources() == [CurrentSource("J1", "1", 2.0)]
    assert loaded.active_heat_loads == [] and loaded.active_resistive_elements == []


@pytest.mark.parametrize("section", ["current_sources", "heat_loads"])
def test_enabled_must_be_a_boolean(tmp_path, section):
    value = "current_a: 1" if section == "current_sources" else "power_w: 1"
    path = tmp_path / "p.kits"
    path.write_text(f"{section}:\n  - {{ref: J1, pad: 1, {value}, enabled: 'no'}}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="true or false"):
        load_kits_project(path)


def test_disabled_items_are_not_solved(tmp_path):
    board = tmp_path / "bridged.kicad_pcb"
    pads = _pad("P1", 7, "F.Cu") + _pad("P2", 43, "F.Cu")
    board.write_text(_HEADER + _zone(5, 22) + _zone(28, 45) + pads + _RESISTOR + ")\n")
    sources = [CurrentSource("P1", "1", 2.0), CurrentSource("P2", "1", -2.0)]
    project = KitsProject(
        tmp_path / "bridged.kits",
        board=board,
        current_sources=sources + [CurrentSource("Q9", "1", 50.0, enabled=False)],  # not even on the board
        heat_loads=[HeatLoad("P1", "1", POWER, 3.0, enabled=False)],
        resistive_elements=[_element(), _element(name="R1 spare", enabled=False)],
    )
    project.save()
    results, _, _ = cli.solve_project(project.path)
    assert results.load_heat == {} and set(results.element_currents) == {"R1"}
    assert results.element_currents["R1"] == pytest.approx(2.0, rel=1e-6)
    # The fingerprint ignores disabled items, so toggling one back changes it.
    assert results.inputs == inputs_hash(sources, project.solve, [], [_element()])

    # With everything disabled there's nothing to solve.
    project.current_sources = [dataclasses.replace(s, enabled=False) for s in project.current_sources]
    project.save()
    with pytest.raises(ConfigError, match="no enabled current sources"):
        cli.solve_project(project.path)


def test_panel_checkboxes(app, window, bridged_kits, monkeypatch):  # noqa: F811
    assert window.open_path(bridged_kits)
    doc, scene = window.document, window.scene
    doc.add_heat_load("P1", "1", POWER, 1.0)
    doc.add_resistive_element(_element())
    assert scene.source_marker_count == 2 and scene.heat_load_marker_count == 1
    assert scene.element_marker_count == 1

    # Untick each kind's checkbox: it is kept but disabled, and leaves the board.
    window.sources_panel.table.item(1, 0).setCheckState(Qt.CheckState.Unchecked)
    assert doc.current_source("P2", "1").enabled is False and scene.source_marker_count == 1
    assert "1 disabled" in window.sources_panel.total_label.text()
    window.heat_panel.table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    assert doc.heat_load("P1", "1").enabled is False and scene.heat_load_marker_count == 0
    window.elements_panel.table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    assert doc.resistive_element("R1").enabled is False and scene.element_marker_count == 0
    assert doc.dirty

    # One source disabled leaves the rest unbalanced: no solve.
    assert not window.solve_panel.solve_button.isEnabled()
    assert window.sources_panel.table.item(1, 0).foreground().color() == Qt.GlobalColor.gray

    # Editing a disabled item keeps it disabled.
    window.sources_panel.table.cellWidget(1, 2).setValue(-3.0)
    assert doc.current_source("P2", "1") == CurrentSource("P2", "1", -3.0, enabled=False)
    doc.set_heat_load("P1", "1", HeatLoadKind.TEMPERATURE, 30.0)
    assert doc.heat_load("P1", "1").enabled is False

    # Tick again: back on the board.
    window.sources_panel.table.item(1, 0).setCheckState(Qt.CheckState.Checked)
    doc.set_source_current("P2", "1", -5.0)
    assert doc.current_source("P2", "1").enabled and scene.source_marker_count == 2
    assert window.solve_panel.solve_button.isEnabled()

    # The flags are saved with the project.
    assert window.save()
    saved = load_kits_project(bridged_kits)
    assert saved.heat_loads[0].enabled is False and saved.resistive_elements[0].enabled is False

    with pytest.raises(ConfigError):
        doc.set_element_enabled("nope", True)
