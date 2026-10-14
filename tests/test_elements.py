"""Resistive elements: project file, value parsing, results fingerprint and a command-line solve."""

import pytest
import yaml

pytest.importorskip("skfem")

from kits.config import ConfigError, CurrentSource, PadGroup, ResistiveElement, SolveSettings  # noqa: E402
from kits.project import KitsProject, load_kits_project  # noqa: E402
from kits.solve import cli  # noqa: E402
from kits.solve.elements import parse_resistance  # noqa: E402
from kits.solve.results import inputs_hash  # noqa: E402

from .test_solve import _HEADER, _RESISTOR, _pad, _zone  # noqa: E402


def _element(name: str = "Q1", **kwargs) -> ResistiveElement:
    values = {"resistance_ohm": 0.01} | kwargs
    return ResistiveElement(name, "Q1", PadGroup("S", ["1", "2", "3"]), PadGroup("D", ["5"]), **values)


def test_round_trip(tmp_path):
    project = KitsProject(
        tmp_path / "p.kits",
        resistive_elements=[_element(), _element(name="Q1 hot", thermal_resistance_board_k_w=2.5)],
    )
    project.save()
    data = yaml.safe_load(project.path.read_text(encoding="utf-8"))
    assert data["resistive_elements"][0] == {
        "name": "Q1",
        "ref": "Q1",
        "resistance_ohm": 0.01,
        "pads_a": {"name": "S", "pads": ["1", "2", "3"]},
        "pads_b": {"name": "D", "pads": ["5"]},
    }
    assert data["resistive_elements"][1]["thermal_resistance_board_k_w"] == 2.5
    assert "thermal_resistance_ambient_k_w" not in data["resistive_elements"][1]  # default: convection
    assert load_kits_project(project.path).resistive_elements == project.resistive_elements

    project.resistive_elements = []
    project.save()
    assert "resistive_elements" not in yaml.safe_load(project.path.read_text(encoding="utf-8"))


_VALID = {
    "name": "Q1",
    "ref": "Q1",
    "resistance_ohm": 0.01,
    "pads_a": {"name": "S", "pads": [1]},
    "pads_b": {"name": "D", "pads": [5]},
}


def _without(key: str) -> dict:
    return {k: v for k, v in _VALID.items() if k != key}


@pytest.mark.parametrize(
    "entries, match",
    [
        (3, "must be a list"),
        ([_without("resistance_ohm")], "missing 'resistance_ohm'"),
        ([_VALID | {"pads_a": {"name": "S"}}], "pads_a"),
        ([_VALID | {"pads_a": {"name": "S", "pads": []}}], "no pads"),
        ([_VALID | {"pads_b": {"name": "D", "pads": [1]}}], "both groups"),
        ([_VALID | {"pads_b": {"name": "", "pads": [5]}}], "needs a name"),
        ([_VALID | {"resistance_ohm": 0}], "resistance must be a positive"),
        ([_VALID | {"thermal_resistance_board_k_w": -1}], "to board must be a positive"),
        ([_VALID | {"thermal_resistance_ambient_k_w": float("nan")}], "to ambient must be"),
        ([_VALID, _VALID], "more than one resistive element"),
    ],
)
def test_invalid_elements_are_rejected(tmp_path, entries, match):
    path = tmp_path / "p.kits"
    path.write_text(yaml.safe_dump({"resistive_elements": entries}), encoding="utf-8")
    with pytest.raises(ConfigError, match=match):
        load_kits_project(path)


@pytest.mark.parametrize(
    "text, ohms",
    [
        ("10k", 10e3),
        ("4k7", 4.7e3),
        ("4R7", 4.7),
        ("R005", 0.005),
        ("0R1", 0.1),
        ("0.01", 0.01),
        ("10m", 0.01),
        ("10mR", 0.01),
        ("100mΩ", 0.1),
        ("10 ohm", 10.0),
        ("2.2M", 2.2e6),
        ("1k 1%", 1e3),
        ("10uF", None),
        ("2N7002", None),
        ("IRF540", None),
        ("0R", None),
        ("", None),
    ],
)
def test_parse_resistance(text, ohms):
    assert parse_resistance(text) == (pytest.approx(ohms) if ohms is not None else None)


def test_inputs_hash_includes_elements_only_when_present():
    sources, settings = [CurrentSource("P1", "1", 5.0)], SolveSettings()
    assert inputs_hash(sources, settings, [], []) == inputs_hash(sources, settings)
    one = inputs_hash(sources, settings, [], [_element()])
    assert one != inputs_hash(sources, settings)
    assert one != inputs_hash(sources, settings, [], [_element(resistance_ohm=0.02)])
    assert one != inputs_hash(sources, settings, [], [_element(thermal_resistance_ambient_k_w=5.0)])


def test_command_line_solve_with_an_element(tmp_path):
    board = tmp_path / "bridged.kicad_pcb"
    pads = _pad("P1", 7, "F.Cu") + _pad("P2", 43, "F.Cu")
    board.write_text(_HEADER + _zone(5, 22) + _zone(28, 45) + pads + _RESISTOR + ")\n")
    element = ResistiveElement("R1", "R1", PadGroup("A", ["1"]), PadGroup("B", ["2"]), resistance_ohm=0.1)
    project = KitsProject(
        tmp_path / "bridged.kits",
        board=board,
        current_sources=[CurrentSource("P1", "1", 2.0), CurrentSource("P2", "1", -2.0)],
        resistive_elements=[element],
    )
    project.save()
    results, _, _ = cli.solve_project(project.path)
    assert results.element_currents["R1"] == pytest.approx(2.0, rel=1e-6)
    assert results.element_power["R1"] == pytest.approx(0.4, rel=1e-6)
    assert results.total_power_w > 0.4  # plus the copper
    assert results.element_temperatures["R1"] > 25.0
    assert results.inputs == inputs_hash(project.scaled_current_sources(), project.solve, [], [element])

    # A missing footprint is reported by name.
    project.resistive_elements = [
        ResistiveElement("X", "R9", PadGroup("A", ["1"]), PadGroup("B", ["2"]), 1.0)
    ]
    project.save()
    with pytest.raises(cli.SolveError, match="R9.1"):
        cli.solve_project(project.path)
