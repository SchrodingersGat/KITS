"""Pass/fail limits: project file, the checks, and the command-line verdict."""

import json

import numpy as np
import pytest
import yaml

pytest.importorskip("skfem")

from kits.config import ConfigError, CurrentSource, Limits, PadGroup, ResistiveElement  # noqa: E402
from kits.postprocess.limits import (  # noqa: E402
    _weighted_percentile,
    check_limits,
    hottest,
    layer_peak_current_density,
    peak_current_density,
    verdict,
)
from kits.project import KitsProject, load_kits_project  # noqa: E402
from kits.solve import cli  # noqa: E402

from .test_solve import _HEADER, _RESISTOR, _pad, _zone  # noqa: E402


def test_limits_in_the_project_file(tmp_path):
    project = KitsProject(tmp_path / "p.kits")
    project.save()
    assert "limits" not in yaml.safe_load(project.path.read_text(encoding="utf-8"))  # none set: not written
    project.limits = Limits(max_current_density_a_mm2=35.0, max_temperature_c=105.0)
    project.save()
    data = yaml.safe_load(project.path.read_text(encoding="utf-8"))
    assert data["limits"] == {"max_current_density_a_mm2": 35.0, "max_temperature_c": 105.0}
    assert load_kits_project(project.path).limits == project.limits
    project.limits = Limits(max_temperature_c=85.0)  # just one
    project.save()
    assert load_kits_project(project.path).limits == Limits(None, 85.0)
    # The corner spike setting is written only when it isn't the default, even with no limits.
    project.limits = Limits(current_density_spike_area_pct=0.0)
    project.save()
    data = yaml.safe_load(project.path.read_text(encoding="utf-8"))
    assert data["limits"] == {"current_density_spike_area_pct": 0.0}
    assert load_kits_project(project.path).limits == project.limits


@pytest.mark.parametrize(
    "limits, message",
    [
        (Limits(max_current_density_a_mm2=0), "current density limit must be a positive"),
        (Limits(max_current_density_a_mm2=float("nan")), "current density limit must be a positive"),
        (Limits(max_temperature_c=-300), "above absolute zero"),
        (Limits(current_density_spike_area_pct=-1), "spike area must be between"),
        (Limits(current_density_spike_area_pct=50), "spike area must be between"),
    ],
)
def test_invalid_limits(limits, message):
    with pytest.raises(ConfigError, match=message):
        limits.validate()
    with pytest.raises(ConfigError, match="invalid limit"):
        Limits.from_dict({"max_temperature_c": "hot"})


def test_weighted_percentile():
    values = np.array([1.0, 2.0, 100.0])
    # The 100 covers 0.1% of the area, so it's above the 99.5th area percentile.
    assert _weighted_percentile(values, np.array([500.0, 499.0, 1.0]), 99.5) == 2.0
    assert _weighted_percentile(values, np.array([1.0, 1.0, 98.0]), 99.5) == 100.0
    assert _weighted_percentile(np.array([]), np.array([]), 99.5) == 0.0


@pytest.fixture
def solved(tmp_path):
    """The bridged strips solved with 10 A through R1, as a saved project."""
    board = tmp_path / "bridged.kicad_pcb"
    pads = _pad("P1", 7, "F.Cu") + _pad("P2", 43, "F.Cu")
    board.write_text(_HEADER + _zone(5, 22) + _zone(28, 45) + pads + _RESISTOR + ")\n")
    project = KitsProject(
        tmp_path / "bridged.kits",
        board=board,
        current_sources=[CurrentSource("P1", "1", 10.0), CurrentSource("P2", "1", -10.0)],
        resistive_elements=[ResistiveElement("R1", "R1", PadGroup("A", ["1"]), PadGroup("B", ["2"]), 0.01)],
    )
    project.save()
    results, _, _ = cli.solve_project(project.path)
    return project, results


def test_checks_against_the_results(solved):
    _project, results = solved
    density, layer = peak_current_density(results)
    # 10 A along 6 mm × 35 µm strips ≈ 47.6 A/mm²; the peak excludes corner spikes.
    assert layer == "F.Cu" and 40 < density < 70
    absolute = max(r.current_density[r.copper].max() / 1e6 for r in results.layers if r.copper.any())
    assert density <= absolute
    temperature, where = hottest(results)
    assert temperature == pytest.approx(results.max_temperature_c) and where == "R1 body"

    assert check_limits(results, Limits()) == [] and verdict([]) is None
    passing = check_limits(results, Limits(density * 1.01, temperature + 1))
    assert [c.passed for c in passing] == [True, True] and verdict(passing) is True
    assert passing[0].margin == pytest.approx(density * 0.01)
    failing = check_limits(results, Limits(density * 0.99, temperature + 1))
    assert [c.passed for c in failing] == [False, True] and verdict(failing) is False
    assert failing[0].margin < 0 and failing[0].quantity == "Current density"
    assert check_limits(results, Limits(max_temperature_c=temperature))[0].passed  # equal passes


def test_corner_spike_exclusion(solved):
    _project, results = solved
    absolute = max(r.current_density[r.copper].max() / 1e6 for r in results.layers if r.copper.any())
    # Off: the true maximum, spikes and all. More area ignored: a lower peak.
    assert peak_current_density(results, 0.0) == (pytest.approx(absolute), "F.Cu")
    peaks = [peak_current_density(results, pct)[0] for pct in (0.0, 0.1, 0.5, 2.0)]
    assert peaks == sorted(peaks, reverse=True) and peaks[-1] < absolute
    check = check_limits(results, Limits(max_current_density_a_mm2=1.0, current_density_spike_area_pct=0))[0]
    assert check.value == pytest.approx(absolute) and "spikes included" in check.basis
    check = check_limits(results, Limits(max_current_density_a_mm2=1.0))[0]
    assert check.value == pytest.approx(peaks[2]) and "99.5%" in check.basis

    # A layer's peak sits on a triangle at or below the cut-off, not on the spike.
    layer = results.layer("F.Cu")
    value, triangle = layer_peak_current_density(layer, 0.5)
    assert layer.current_density[triangle] / 1e6 == pytest.approx(value)
    assert value <= peak_current_density(results, 0.5)[0] + 1e-9 and value < absolute
    assert layer_peak_current_density(layer, 0.0)[0] == pytest.approx(absolute)


def test_command_line_reports_pass_or_fail(solved, capsys):
    project, results = solved
    assert cli.main([str(project.path)]) == 0
    assert "Result:" not in capsys.readouterr().out  # no limits, no verdict

    project.limits = Limits(max_temperature_c=results.max_temperature_c - 1)
    project.save()
    assert cli.main([str(project.path)]) == 0  # a failed check isn't a failed solve
    out = capsys.readouterr().out
    assert "limit: Temperature" in out and "R1 body" in out and "Result: FAIL" in out
    assert cli.main([str(project.path), "--progress"]) == 0
    done = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert done["passed"] is False

    # Limits don't affect the results: changing one doesn't make them stale.
    from kits.solve.results import inputs_hash

    assert results.inputs == inputs_hash(
        project.scaled_current_sources(), project.solve, [], project.active_resistive_elements, {}, []
    )
