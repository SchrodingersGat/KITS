"""HTML results report: data encoding, contents, and the kits-report command line."""

import base64
import json
import re
import zlib

import numpy as np
import pytest

pytest.importorskip("skfem")

from kits.config import CurrentSource, HeatLoad, HeatLoadKind, PadGroup, ResistiveElement  # noqa: E402
from kits.project import KitsProject, load_kits_project  # noqa: E402
from kits.report import cli  # noqa: E402
from kits.report.builder import encode_raster  # noqa: E402
from kits.solve.cli import solve_project  # noqa: E402

from .test_solve import _HEADER, _RESISTOR, _pad, _zone  # noqa: E402


def _decode(raster: dict) -> np.ndarray:
    codes = np.frombuffer(zlib.decompress(base64.b64decode(raster["data"])), "<u2").astype(float)
    values = raster["lo"] + (codes - 1) / 65533 * (raster["hi"] - raster["lo"])
    if raster["q"] == "log":
        values = 10**values
    values[codes == 0] = np.nan
    return values


def _data(page: str) -> dict:
    found = re.search(r'<script id="report-data" type="application/json">(.*?)</script>', page, re.S)
    return json.loads(found.group(1))


@pytest.mark.parametrize("log", [False, True])
def test_raster_round_trip(log):
    values = np.array([np.nan, 0.5, 1.0, 10.0, 100.0, np.nan])
    decoded = _decode(encode_raster(values, log))
    assert np.isnan(decoded[[0, 5]]).all()
    if log:  # 16 bits over the decades: a tiny relative error everywhere
        np.testing.assert_allclose(decoded[1:5], values[1:5], rtol=1e-4)
    else:  # 16 bits over the range: within half a step
        np.testing.assert_allclose(decoded[1:5], values[1:5], atol=(100 - 0.5) / 65533)
    assert np.isnan(_decode(encode_raster(np.full(3, np.nan), log))).all()  # nothing to show


@pytest.fixture
def bridged_project(tmp_path):
    """Two strips joined by resistor R1, 10 A through it, and a heat load on P1."""
    board = tmp_path / "bridged.kicad_pcb"
    pads = _pad("P1", 7, "F.Cu") + _pad("P2", 43, "F.Cu")
    board.write_text(_HEADER + _zone(5, 22) + _zone(28, 45) + pads + _RESISTOR + ")\n")
    project = KitsProject(
        tmp_path / "bridged.kits",
        board=board,
        current_sources=[CurrentSource("P1", "1", 10.0), CurrentSource("P2", "1", -10.0)],
        heat_loads=[HeatLoad("P1", "1", HeatLoadKind.POWER, 0.5)],
        resistive_elements=[
            ResistiveElement("R1 <b>", "R1", PadGroup("A", ["1"]), PadGroup("B", ["2"]), 0.01)
        ],
    )
    project.save()
    return project.path


def test_report_contents(bridged_project, tmp_path):
    from kits.config import Limits

    results, _, _ = solve_project(bridged_project)
    project = load_kits_project(bridged_project)
    project.limits = Limits(max_temperature_c=results.max_temperature_c - 1)  # a failing check
    project.save()
    out = tmp_path / "out" / "report.html"
    path, staleness = cli.generate_report(bridged_project, out, max_pixels=200)
    assert path == out and staleness is None
    page = out.read_text(encoding="utf-8")
    data = _data(page)

    # Summary and every simulation element, with their figures.
    assert data["summary"]["power"] == pytest.approx(results.total_power_w)
    assert data["summary"]["max_temperature"] == pytest.approx(results.max_temperature_c)
    for heading in ("Current sources", "Heat loads", "Resistive elements", "Settings", "Board stackup"):
        assert heading in page
    # Total losses and where they go: the copper and each element, with their shares.
    summary = data["summary"]
    assert summary["element_losses"] == pytest.approx(results.element_losses_w)
    assert summary["copper_losses"] + summary["element_losses"] == pytest.approx(summary["power"])
    losses = page[page.index("<h3>Losses</h3>") : page.index("<h3>Current sources</h3>")]
    assert (
        "Copper (tracks, pours, vias, solder joints)" in losses and "Resistive element R1 &lt;b&gt;" in losses
    )
    share = 100 * results.element_losses_w / results.total_power_w
    assert f"{share:.1f}%" in losses and "100%" in losses
    # The verdict, and the limits table.
    assert data["verdict"] is False
    assert "Pass/fail limits" in page and '<b class="fail">FAIL</b>' in page
    # The stackup, top to bottom, with each copper layer's weight and height, and the total.
    stackup = page[page.index("Board stackup") :]
    assert stackup.index("F.Cu") < stackup.index("dielectric 1") < stackup.index("B.Cu")
    assert "1 oz" in stackup and "35 µm" in stackup and "1510 µm" in stackup and "FR4" in stackup
    assert "1.58 mm" in stackup  # the total
    assert "P1.1" in page and "default, unclassified body" in page
    # Element names are escaped in the HTML (they come from the project file). The JSON data
    # keeps them raw, inside its script element, and the page's script escapes them.
    html_part = re.sub(r'<script id="report-data".*?</script>', "", page, flags=re.S)
    assert "R1 &lt;b&gt;" in html_part and "R1 <b>" not in html_part

    # One map per layer and quantity, sampled like the GUI shows them.
    grid = data["grid"]
    assert max(grid["width"], grid["height"]) == 200
    assert [layer["name"] for layer in data["layers"]] == ["F.Cu", "B.Cu"]
    f_cu = data["layers"][0]["rasters"]
    temperature = _decode(f_cu["TEMPERATURE"])
    assert temperature.size == grid["width"] * grid["height"]
    layer = results.layer("F.Cu")
    assert np.nanmin(temperature) >= layer.temperature.min() - 1e-3
    assert np.nanmax(temperature) <= layer.temperature.max() + 1e-3
    density = _decode(f_cu["CURRENT_DENSITY"])
    assert f_cu["CURRENT_DENSITY"]["q"] == "log" and np.nanmax(density) > 0
    # Off copper (the gap between the strips) there's temperature but no current density.
    assert np.isnan(density).sum() > np.isnan(temperature).sum()
    assert [m["key"] for m in data["modes"]] == ["CURRENT_DENSITY", "VOLTAGE", "TEMPERATURE"]

    # The element body is drawn on its side's temperature map.
    (body,) = data["bodies"]
    assert body["layer"] == "F.Cu" and body["temperature"] == pytest.approx(
        results.element_temperatures["R1 <b>"]
    )
    # Fresh results carry each element's heat through its terminals.
    assert "Heat to A, B" in page and results.element_terminal_heat["R1 <b>"]


def test_report_command_line(bridged_project, capsys):
    # No results yet.
    assert cli.main([str(bridged_project)]) == 1
    assert "no saved results" in capsys.readouterr().err

    # --solve solves first; the default path is in kits_output.
    assert cli.main([str(bridged_project), "--solve", "--max-pixels", "100"]) == 0
    project = load_kits_project(bridged_project)
    report = cli.default_report_path(project)
    assert report.is_file() and f"Wrote {report}" in capsys.readouterr().out

    # Edited inputs: the old results are still reported, with a warning in the page.
    project.current_sources[0].current_a = 12.0
    project.current_sources[1].current_a = -12.0
    project.save()
    assert cli.main([str(bridged_project), "--max-pixels", "100", "--progress"]) == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    done = records[-1]
    assert done["done"] and "changed since the results were computed" in done["stale"]
    assert any("Rendering F.Cu" in r.get("message", "") for r in records)
    assert "changed since the results were computed" in _data(report.read_text(encoding="utf-8"))["staleness"]

    assert cli.main([str(bridged_project.with_suffix(".txt"))]) == 1
