"""Manual heatsinks: project file, thermal model, and contact only through solder mask openings."""

import numpy as np
import pytest
import yaml

pytest.importorskip("skfem")

from kits.config import ConfigError, CurrentSource, HeatLoad, HeatLoadKind, Heatsink  # noqa: E402
from kits.ingest import load_board  # noqa: E402
from kits.project import KitsProject, load_kits_project  # noqa: E402
from kits.solve import cli  # noqa: E402
from kits.solve.results import inputs_hash  # noqa: E402
from kits.solve.solver import Electrode, SolveError, SolveSettings, _triangle_areas  # noqa: E402

from .test_mask import _MASKED_HEADER  # noqa: E402
from .test_mask import _pad as _masked_pad  # noqa: E402
from .test_solve import _HEADER, _model, _pad, _strip  # noqa: E402

POWER = HeatLoadKind.POWER
SETTINGS = SolveSettings(convection_top_w_m2k=10.0, convection_bottom_w_m2k=10.0, ambient_c=25.0)


def _heatsink(**kwargs) -> Heatsink:
    values = {"name": "HS1", "side": "top", "x_mm": 20.0, "y_mm": 2.0, "width_mm": 10.0, "height_mm": 6.0}
    return Heatsink(**(values | kwargs))


def test_project_round_trip_and_validation(tmp_path):
    project = KitsProject(
        tmp_path / "p.kits", heatsinks=[_heatsink(), _heatsink(name="HS2", side="bottom", enabled=False)]
    )
    project.save()
    data = yaml.safe_load(project.path.read_text(encoding="utf-8"))
    assert data["heatsinks"][0] == {
        "name": "HS1",
        "side": "top",
        "x_mm": 20.0,
        "y_mm": 2.0,
        "width_mm": 10.0,
        "height_mm": 6.0,
        "thermal_resistance_board_k_w": 1.0,
        "thermal_resistance_ambient_k_w": 10.0,
    }
    loaded = load_kits_project(project.path)
    assert loaded.heatsinks == project.heatsinks
    assert [h.name for h in loaded.active_heatsinks] == ["HS1"]

    for bad, match in (
        ({"side": "left"}, "side"),
        ({"width_mm": 0}, "width"),
        ({"thermal_resistance_ambient_k_w": -1}, "ambient"),
        ({"x_mm": float("inf")}, "position"),
    ):
        project.path.write_text(yaml.safe_dump({"heatsinks": [_heatsink(**bad).to_dict()]}), encoding="utf-8")
        with pytest.raises(ConfigError, match=match):
            load_kits_project(project.path)


@pytest.fixture
def strip_board(tmp_path):
    path = tmp_path / "strip.kicad_pcb"
    path.write_text(_HEADER + _strip("F.Cu") + _pad("P1", 7, "F.Cu") + _pad("P2", 43, "F.Cu") + ")\n")
    return load_board(path)


def test_heatsink_cools_the_board(strip_board):
    model = _model(strip_board, SETTINGS)
    source = [HeatLoad("P1", "1", POWER, 2.0)]
    plain = model.solve_thermal(None, source)
    sink = _heatsink(thermal_resistance_board_k_w=1.0, thermal_resistance_ambient_k_w=2.0)
    cooled = model.solve_thermal(None, source, heatsinks=[sink])

    flow = cooled.heatsink_heat["HS1"]
    assert 0 < flow < 2.0
    # Heat balance: board convection plus the heatsink take out all 2 W.
    assert cooled.convected_power + flow == pytest.approx(2.0, rel=1e-9)
    # The heatsink obeys both resistances: to ambient, and from the board's contact area.
    t_hs = cooled.heatsink_temperatures["HS1"]
    assert flow == pytest.approx((t_hs - 25.0) / 2.0, rel=1e-9)
    nodes, g = model.heatsink_conductances(sink)
    temperature = np.concatenate(list(cooled.temperature.values()))
    assert float(g @ (temperature[nodes] - t_hs)) == pytest.approx(flow, rel=1e-6)
    # No mask on this board: the whole contact conducts at 1 / R.
    assert g.sum() == pytest.approx(1.0 / 1.0, rel=1e-9)
    assert cooled.temperature["F.Cu"].max() < plain.temperature["F.Cu"].max()

    # Contact is the rectangle on the top face (one element of slack at the edges).
    xy = model.layers[0].nodes_mm[nodes - model.layers[0].offset]
    assert xy[:, 0].min() >= 19 and xy[:, 0].max() <= 31
    bottom = model.heatsink_conductances(_heatsink(side="bottom"))[0]
    assert bottom.min() >= model.layers[-1].offset
    # The board doesn't convect under the heatsink.
    h = model._surface_coefficient(model.layers[0], True, False, [sink])
    covered = model.heatsink_triangles(sink)
    assert (h[covered] == 0).all() and (h[~covered] == 10.0).all()

    # A better heatsink takes more; one off the board is an error.
    better = model.solve_thermal(None, source, heatsinks=[_heatsink(thermal_resistance_ambient_k_w=0.5)])
    assert better.heatsink_heat["HS1"] > flow
    with pytest.raises(SolveError, match="HS1 is off the board"):
        model.solve_thermal(None, source, heatsinks=[_heatsink(x_mm=100.0)])


def test_heatsink_contact_through_copper_and_mask(tmp_path):
    """Full exposure conducts 1/R; all mask 1/(R + t/kA); a partial opening, the mix."""
    opening = '  (gr_rect (start 20 2) (end 25 8) (stroke (width 0)) (fill solid) (layer "F.Mask"))\n'

    def model(extra, header=_MASKED_HEADER):
        path = tmp_path / "masked.kicad_pcb"
        pads = _masked_pad("P1", 7) + _masked_pad("P2", 43)
        path.write_text(header + _strip("F.Cu") + pads + extra + ")\n")
        return _model(load_board(path), SETTINGS)

    sink = _heatsink(thermal_resistance_board_k_w=1.0)  # 10 x 6 mm over the strip, x 20..30
    area = 60e-6  # m²
    mask_r = 0.02e-3 / 0.25  # 20 µm of mask (from the stackup) at 0.25 W/mK, K·m²/W

    exposed = model("", header=_HEADER)  # no solder mask at all
    assert exposed.heatsink_conductances(sink)[1].sum() == pytest.approx(1.0, rel=1e-9)

    covered = model("")  # all under mask: still conducts, much less
    g_mask = covered.heatsink_conductances(sink)[1].sum()
    # The covered area is that of the mesh triangles under the heatsink, ~60 mm².
    lf = covered.layers[0]
    mesh_area = _triangle_areas(lf.nodes_mm, lf.mesh.t[:, covered.heatsink_triangles(sink)]).sum() * 1e-6
    assert mesh_area == pytest.approx(area, rel=0.02)
    assert g_mask == pytest.approx(mesh_area / (1.0 * mesh_area + mask_r), rel=1e-9)
    assert g_mask < 0.5

    partial = model(opening)  # the left half (x 20..25) is open
    nodes, g = partial.heatsink_conductances(sink)
    lf = partial.layers[0]
    tri = lf.mesh.t[:, partial.heatsink_triangles(sink)]
    open_area = (partial.masked_triangles(lf)[partial.heatsink_triangles(sink)] == 0).mean()
    assert 0.3 < open_area < 0.7  # about half (by mesh triangles)
    assert g_mask < g.sum() < 1.0
    # Exactly the mix: open area at 1/(R·A), masked area at 1/(R·A + t/k), A the covered area.
    tri_area = _triangle_areas(lf.nodes_mm, tri) * 1e-6
    is_masked = partial.masked_triangles(lf)[partial.heatsink_triangles(sink)]
    covered_area = tri_area.sum()
    expected = tri_area[~is_masked].sum() / (1.0 * covered_area) + tri_area[is_masked].sum() / (
        1.0 * covered_area + mask_r
    )
    assert g.sum() == pytest.approx(expected, rel=1e-9)
    # Open nodes (left) conduct more than masked ones (right) for the same area.
    x = lf.nodes_mm[nodes - lf.offset][:, 0]
    assert g[x < 24].sum() > g[x > 26].sum()

    # The heat balance closes in every case.
    for m in (exposed, covered, partial):
        thermal = m.solve_thermal(None, [HeatLoad("P1", "1", POWER, 1.0)], heatsinks=[sink])
        assert thermal.convected_power + thermal.heatsink_heat["HS1"] == pytest.approx(1.0, rel=1e-9)
    assert (
        exposed.solve_thermal(None, [HeatLoad("P1", "1", POWER, 1.0)], heatsinks=[sink]).heatsink_heat["HS1"]
        > partial.solve_thermal(None, [HeatLoad("P1", "1", POWER, 1.0)], heatsinks=[sink]).heatsink_heat[
            "HS1"
        ]
        > covered.solve_thermal(None, [HeatLoad("P1", "1", POWER, 1.0)], heatsinks=[sink]).heatsink_heat[
            "HS1"
        ]
    )


def test_element_body_heat_goes_through_mask_openings(tmp_path):
    from kits.config import PadGroup, ResistiveElement

    resistor = """  (footprint "T:R" (layer "F.Cu") (at 25 5)
    (property "Reference" "R1")
    (fp_rect (start -6 -3) (end 6 3) (stroke (width 0.1) (type solid)) (fill none) (layer "F.Fab"))
    (pad "1" smd rect (at -5 0) (size 2 6) (layers "F.Cu" "F.Mask") (net 1 "N"))
    (pad "2" smd rect (at 5 0) (size 2 6) (layers "F.Cu" "F.Mask") (net 1 "N")))
"""
    path = tmp_path / "masked_r.kicad_pcb"
    path.write_text(_MASKED_HEADER + _strip("F.Cu") + resistor + ")\n")
    model = _model(load_board(path), SETTINGS)
    element = ResistiveElement("R1", "R1", PadGroup("A", ["1"]), PadGroup("B", ["2"]), 0.01)
    nodes, _ = model.body_nodes(element)
    x = model.layers[0].nodes_mm[nodes - model.layers[0].offset][:, 0]
    # The 12 mm body is all under mask except its two exposed pads (x 19..21 and 29..31).
    assert ((x < 22) | (x > 28)).all()


def test_command_line_solve_with_a_heatsink(tmp_path):
    board = tmp_path / "strip.kicad_pcb"
    board.write_text(_HEADER + _strip("F.Cu") + _pad("P1", 7, "F.Cu") + _pad("P2", 43, "F.Cu") + ")\n")
    project = KitsProject(
        tmp_path / "strip.kits",
        board=board,
        current_sources=[CurrentSource("P1", "1", 10.0), CurrentSource("P2", "1", -10.0)],
        heatsinks=[_heatsink(), _heatsink(name="spare", enabled=False)],
    )
    project.save()
    results, results_path, _ = cli.solve_project(project.path)
    assert set(results.heatsink_temperatures) == {"HS1"}  # the disabled one isn't solved
    assert results.heatsink_heat["HS1"] > 0
    assert results.inputs == inputs_hash(
        project.scaled_current_sources(), project.solve, [], [], {}, [project.heatsinks[0]]
    )
    from kits.solve.results import SolveResults

    assert SolveResults.load_npz(results_path).heatsink_heat == results.heatsink_heat


def test_mount_round_trip_and_validation(tmp_path):
    project = KitsProject(
        tmp_path / "p.kits", heatsinks=[_heatsink(mount="components"), _heatsink(name="HS2")]
    )
    project.save()
    data = yaml.safe_load(project.path.read_text(encoding="utf-8"))
    assert data["heatsinks"][0]["mount"] == "components"
    assert "mount" not in data["heatsinks"][1]  # board is the default
    assert load_kits_project(project.path).heatsinks == project.heatsinks
    project.path.write_text(
        yaml.safe_dump({"heatsinks": [_heatsink(mount="lid").to_dict()]}), encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="mount"):
        load_kits_project(project.path)


@pytest.fixture
def bridged(tmp_path):
    """Two strips joined only by resistor R1 (body = its pads' extent, x 18..32, y 2..8)."""
    from kits.config import PadGroup, ResistiveElement

    from .test_solve import _RESISTOR, _zone

    path = tmp_path / "bridged.kicad_pcb"
    pads = _pad("P1", 7, "F.Cu") + _pad("P2", 43, "F.Cu")
    path.write_text(_HEADER + _zone(5, 22) + _zone(28, 45) + pads + _RESISTOR + ")\n")
    model = _model(load_board(path), SETTINGS)
    element = ResistiveElement("R1", "R1", PadGroup("A", ["1"]), PadGroup("B", ["2"]), resistance_ohm=0.01)
    dc = model.solve_dc([Electrode("P1", "1", current=10.0), Electrode("P2", "1", current=-10.0)], [element])
    return model, element, dc


def test_heatsink_on_a_component_cools_it_through_its_body(bridged):
    model, element, dc = bridged
    plain = model.solve_thermal(dc, elements=[element])
    on_top = _heatsink(
        x_mm=18.0,
        y_mm=2.0,
        width_mm=14.0,
        height_mm=6.0,
        mount="components",
        thermal_resistance_board_k_w=2.0,
        thermal_resistance_ambient_k_w=5.0,
    )
    cooled = model.solve_thermal(dc, elements=[element], heatsinks=[on_top])

    flow = cooled.heatsink_heat["HS1"]
    t_hs, t_body = cooled.heatsink_temperatures["HS1"], cooled.element_temperatures["R1"]
    # The path is body -> (2 K/W) -> heatsink -> (5 K/W) -> ambient.
    assert flow == pytest.approx((t_hs - 25.0) / 5.0, rel=1e-9)
    assert t_body - t_hs == pytest.approx(flow * 2.0, rel=1e-6)
    # The body is fully covered, so it no longer convects from its top.
    assert cooled.element_ambient_heat["R1"] == 0.0
    total_out = cooled.convected_power + flow
    assert total_out == pytest.approx(dc.total_power, rel=1e-8)
    assert t_body < plain.element_temperatures["R1"]


def test_partly_covered_component_and_nothing_underneath(bridged):
    from kits.solve.elements import default_ambient_conductance

    model, element, dc = bridged
    half = _heatsink(x_mm=18.0, y_mm=2.0, width_mm=7.0, height_mm=6.0, mount="components")  # half of 14 x 6
    thermal = model.solve_thermal(dc, elements=[element], heatsinks=[half])
    footprint = next(fp for fp in model.board.footprints if fp.reference == "R1")
    g_half = default_ambient_conductance(footprint, SETTINGS) * 0.5
    expected = g_half * (thermal.element_temperatures["R1"] - 25.0)
    assert thermal.element_ambient_heat["R1"] == pytest.approx(expected, rel=1e-9)
    assert model.heatsink_contacts(half, [element]) == [("R1", pytest.approx(42.0), pytest.approx(0.5))]

    with pytest.raises(SolveError, match="covers no resistive element on the top side"):
        model.solve_thermal(
            dc, elements=[element], heatsinks=[_heatsink(x_mm=5.0, width_mm=4.0, mount="components")]
        )
    with pytest.raises(SolveError, match="covers no resistive element on the bottom side"):  # R1 is on top
        model.solve_thermal(dc, elements=[element], heatsinks=[_heatsink(side="bottom", mount="components")])
