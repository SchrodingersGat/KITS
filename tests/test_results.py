import json

import numpy as np
import pytest

pytest.importorskip("skfem")

from kits.config import ConfigError, CurrentSource  # noqa: E402
from kits.mesh.fingerprint import MeshFingerprint  # noqa: E402
from kits.project import KitsProject, load_kits_project  # noqa: E402
from kits.solve import cli  # noqa: E402
from kits.solve.results import (  # noqa: E402
    LayerResult,
    SolveResults,
    inputs_hash,
    load_saved_results,
)
from kits.solve.solver import SolveError, SolveSettings  # noqa: E402

from .test_solve import _HEADER, _pad, _strip  # noqa: E402


@pytest.fixture
def strip_project(tmp_path):
    """A project for the 40 x 6 mm strip board with +5 A at P1 and -5 A at P2."""
    board = tmp_path / "strip.kicad_pcb"
    board.write_text(_HEADER + _strip("F.Cu") + _pad("P1", 7, "F.Cu") + _pad("P2", 43, "F.Cu") + ")\n")
    project = KitsProject(
        tmp_path / "strip.kits",
        board=board,
        current_sources=[CurrentSource("P1", "1", 5.0), CurrentSource("P2", "1", -5.0)],
    )
    project.save()
    return project.path


def _results() -> SolveResults:
    layer = LayerResult(
        layer="F.Cu",
        nodes=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        triangles=np.array([[0, 1, 2]]),
        copper=np.array([True]),
        voltage=np.array([0.0, 1.0, 2.0]),
        temperature=np.array([25.0, 26.0, 27.0]),
        current_density=np.array([3e6]),
    )
    return SolveResults([layer], 1.5, {"P1.1": 0.01}, {"P1.1": 5.0}, MeshFingerprint("s", "b"), "i")


def test_results_round_trip(tmp_path):
    results = _results()
    results.save_npz(tmp_path / "r.npz")
    loaded = SolveResults.load_npz(tmp_path / "r.npz")
    assert loaded.total_power_w == 1.5 and loaded.max_temperature_c == 27.0
    assert loaded.electrode_voltages == {"P1.1": 0.01}
    assert loaded.mesh_fingerprint == MeshFingerprint("s", "b") and loaded.inputs == "i"
    np.testing.assert_array_equal(loaded.layer("F.Cu").voltage, [0.0, 1.0, 2.0])


def test_inputs_hash_ignores_source_order_but_not_values():
    a, b = CurrentSource("J1", "1", 1.0), CurrentSource("J2", "1", -1.0)
    settings = SolveSettings()
    assert inputs_hash([a, b], settings) == inputs_hash([b, a], settings)
    assert inputs_hash([a, b], settings) != inputs_hash([CurrentSource("J1", "1", 2.0), b], settings)
    assert inputs_hash([a, b], settings) != inputs_hash([a, b], SolveSettings(convection_w_m2k=20))


def test_load_saved_results_checks_fingerprints(tmp_path):
    path = tmp_path / "r.npz"
    assert load_saved_results(path, MeshFingerprint("s", "b"), "i") == (None, None)
    _results().save_npz(path)
    assert load_saved_results(path, MeshFingerprint("s", "b"), "i")[0] is not None
    assert "board changed" in load_saved_results(path, MeshFingerprint("s", "B"), "i")[1]
    assert "mesh settings changed" in load_saved_results(path, MeshFingerprint("S", "b"), "i")[1]
    assert "current sources" in load_saved_results(path, MeshFingerprint("s", "b"), "I")[1]
    path.write_bytes(b"junk")
    assert "couldn't be read" in load_saved_results(path, MeshFingerprint("s", "b"), "i")[1]


def test_solve_project_writes_results_and_reuses_the_mesh(strip_project, monkeypatch):
    results, results_path, mesh_path = cli.solve_project(strip_project)
    assert results_path.is_file() and mesh_path.is_file()
    assert results.electrode_currents["P1.1"] == pytest.approx(5.0, rel=1e-6)
    assert results.total_power_w > 0 and results.max_temperature_c > 25

    # Second run: the saved mesh is current, so no meshing happens.
    monkeypatch.setattr(cli, "mesh_board", lambda *a, **k: pytest.fail("should reuse the saved mesh"))
    again, _, _ = cli.solve_project(strip_project)
    assert again.total_power_w == pytest.approx(results.total_power_w, rel=1e-9)


@pytest.mark.parametrize(
    "sources, error, match",
    [
        ([], ConfigError, "no current sources"),
        ([CurrentSource("P1", "1", 5.0), CurrentSource("P2", "1", -4.0)], ConfigError, "sum to zero"),
        ([CurrentSource("P1", "1", 5.0), CurrentSource("Q9", "1", -5.0)], SolveError, "Q9.1"),
    ],
)
def test_solve_project_rejects_bad_sources(strip_project, sources, error, match):
    project = load_kits_project(strip_project)
    project.current_sources = sources
    project.save()
    with pytest.raises(error, match=match):
        cli.solve_project(strip_project)


def test_unconnected_sources_are_reported(tmp_path):
    # P1 on F.Cu and P2 on a separate B.Cu strip, with no vias: no copper path.
    board = tmp_path / "split.kicad_pcb"
    board.write_text(
        _HEADER + _strip("F.Cu") + _strip("B.Cu") + _pad("P1", 7, "F.Cu") + _pad("P2", 43, "B.Cu") + ")\n"
    )
    project = KitsProject(
        tmp_path / "split.kits",
        board=board,
        current_sources=[CurrentSource("P1", "1", 5.0), CurrentSource("P2", "1", -5.0)],
    )
    project.save()
    with pytest.raises(SolveError, match="no copper path"):
        cli.solve_project(project.path)


def test_cli_progress_protocol(strip_project, capsys):
    assert cli.main([str(strip_project), "--progress"]) == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert all("progress" in r for r in records[:-1])
    done = records[-1]
    assert done["done"] and done["power_w"] > 0 and done["results"].endswith(".results.npz")


# -- display ------------------------------------------------------------------------


def test_field_display_values_and_ranges(app_or_skip):
    from kits.gui.field_display import DisplayMode, color_indices, format_value, triangle_values, value_range

    results = _results()
    layer = results.layer("F.Cu")
    values, shown = triangle_values(layer, DisplayMode.CURRENT_DENSITY)
    assert values[0] == pytest.approx(3.0) and shown.all()  # A/m² -> A/mm²
    values, _ = triangle_values(layer, DisplayMode.VOLTAGE)
    assert values[0] == pytest.approx(1.0)  # mean of the corners
    assert value_range(results, DisplayMode.TEMPERATURE) == pytest.approx((26.0, 26.0 + 26e-6))  # one value
    assert list(color_indices(np.array([-1.0, 0.0, 0.5, 1.0, 2.0]), 0.0, 1.0, steps=4)) == [0, 0, 2, 3, 3]
    assert format_value(0.0123, DisplayMode.VOLTAGE) == "12.3 mV"
    assert format_value(151.2, DisplayMode.CURRENT_DENSITY) == "151 A/mm²"


@pytest.fixture
def app_or_skip():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])
