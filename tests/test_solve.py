"""Gate tests for the scikit-fem 2.5D solver against hand calculations.

Copper: σ = 5.8e7 S/m, 35 µm thick, so one square of sheet is
1 / (5.8e7 * 35e-6) = 0.4926 mΩ.
"""

import math

import numpy as np
import pytest

pytest.importorskip("skfem")

from kits.config import HeatLoad, HeatLoadKind, MeshSettings  # noqa: E402
from kits.ingest import load_board  # noqa: E402
from kits.mesh.mesher import layer_heights, mesh_board  # noqa: E402
from kits.solve.fem import mass_matrix  # noqa: E402
from kits.solve.solver import Electrode, LayeredModel, SolveError, SolveSettings  # noqa: E402

SIGMA = 5.8e7
T_CU = 35e-6
R_SQUARE = 1 / (SIGMA * T_CU)

_HEADER = """(kicad_pcb (version 20241229) (generator "pcbnew")
  (general (thickness 1.58))
  (layers (0 "F.Cu" signal) (2 "B.Cu" signal) (25 "Edge.Cuts" user))
  (setup (stackup
    (layer "F.Cu" (type "copper") (thickness 0.035))
    (layer "dielectric 1" (type "core") (thickness 1.51) (material "FR4"))
    (layer "B.Cu" (type "copper") (thickness 0.035))))
  (net 0 "")
  (net 1 "N")
  (gr_rect (start 0 0) (end 50 10) (stroke (width 0.05)) (fill none) (layer "Edge.Cuts"))
"""


def _strip(layer: str) -> str:
    """A 40 x 6 mm filled zone from x=5..45, y=2..8."""
    pts = "(xy 5 2) (xy 45 2) (xy 45 8) (xy 5 8)"
    return f"""  (zone (net 1) (net_name "N") (layer "{layer}")
    (polygon (pts {pts})) (filled_polygon (layer "{layer}") (pts {pts})))
"""


def _pad(ref: str, x: float, layer: str) -> str:
    """A 4 x 6 mm SMD pad covering the strip's full width."""
    return f"""  (footprint "T:P" (layer "{layer}") (at {x} 5)
    (property "Reference" "{ref}")
    (pad "1" smd rect (at 0 0) (size 4 6) (layers "{layer}") (net 1 "N")))
"""


@pytest.fixture
def strip_board(tmp_path):
    """One F.Cu strip with full-width pads P1 (x=5..9) and P2 (x=41..45)."""
    path = tmp_path / "strip.kicad_pcb"
    path.write_text(_HEADER + _strip("F.Cu") + _pad("P1", 7, "F.Cu") + _pad("P2", 43, "F.Cu") + ")\n")
    return load_board(path)


@pytest.fixture
def two_layer_board(tmp_path):
    """F.Cu and B.Cu strips joined by a column of 5 vias at x=25; P1 on F.Cu, P2 on B.Cu."""
    vias = "".join(
        f'  (via (at 25 {y}) (size 0.6) (drill 0.3) (layers "F.Cu" "B.Cu") (net 1))\n'
        for y in (3, 4, 5, 6, 7)
    )
    text = _HEADER + _strip("F.Cu") + _strip("B.Cu") + _pad("P1", 7, "F.Cu") + _pad("P2", 43, "B.Cu")
    path = tmp_path / "two.kicad_pcb"
    path.write_text(text + vias + ")\n")
    return load_board(path)


def _model(board, settings=None):
    mesh = mesh_board(board, MeshSettings(max_size_mm=1.0, min_size_mm=0.1))
    return LayeredModel(board, mesh, settings)


def test_strip_resistance_matches_sheet_resistance(strip_board):
    model = _model(strip_board)
    dc = model.solve_dc([Electrode("P1", "1", voltage=1.0), Electrode("P2", "1", voltage=0.0)])
    expected = R_SQUARE * (41 - 9) / 6  # 32 mm long, 6 mm wide between pad edges
    assert dc.resistance(("P1", "1"), ("P2", "1")) == pytest.approx(expected, rel=1e-6)
    # Current conservation and Joule power.
    i1, i2 = dc.electrode_currents[("P1", "1")], dc.electrode_currents[("P2", "1")]
    assert i1 == pytest.approx(-i2, rel=1e-7)
    assert dc.total_power == pytest.approx(1.0**2 / expected, rel=1e-6)
    # Uniform current density I / (W t) in the strip between the pads.
    j = dc.current_density["F.Cu"]
    assert np.median(j[j > 0]) == pytest.approx(i1 / (6e-3 * T_CU), rel=1e-3)


def test_current_sources_give_the_same_resistance(strip_board):
    model = _model(strip_board)
    dc = model.solve_dc([Electrode("P1", "1", current=10.0), Electrode("P2", "1", current=-10.0)])
    # P2 becomes the 0 V reference; current is injected over P1's area, so P1 isn't
    # quite equipotential: the mean-voltage resistance is close to, not exactly, the strip's.
    r = dc.electrode_voltages[("P1", "1")] / 10.0
    assert r == pytest.approx(R_SQUARE * 32 / 6, rel=0.05)
    assert dc.electrode_currents[("P2", "1")] == pytest.approx(-10.0, rel=1e-7)


def test_unbalanced_currents_are_rejected(strip_board):
    model = _model(strip_board)
    with pytest.raises(SolveError):
        model.solve_dc([Electrode("P1", "1", current=10.0), Electrode("P2", "1", current=-5.0)])
    with pytest.raises(SolveError, match="no pad"):
        model.solve_dc([Electrode("P9", "1", voltage=1.0), Electrode("P2", "1", voltage=0.0)])


def test_two_layers_through_vias_match_network_estimate(two_layer_board):
    model = _model(two_layer_board)
    dc = model.solve_dc([Electrode("P1", "1", voltage=1.0), Electrode("P2", "1", voltage=0.0)])
    r = dc.resistance(("P1", "1"), ("P2", "1"))

    heights = layer_heights(two_layer_board)
    barrel_length = (heights["F.Cu"] - heights["B.Cu"]) * 1e-3
    r_via = barrel_length / (SIGMA * math.pi * 0.3e-3 * 25e-6)
    r_strips = 2 * R_SQUARE * (25 - 9) / 6  # F.Cu from P1 to the vias, B.Cu from the vias to P2
    estimate = r_strips + r_via / 5
    # Current crowding into the via column (and the drilled-out holes) adds a little.
    assert estimate < r < 1.15 * estimate
    # All current crosses the barrels: the B.Cu side carries it to P2.
    assert dc.electrode_currents[("P2", "1")] == pytest.approx(-dc.electrode_currents[("P1", "1")], rel=1e-6)


def test_heat_balance_and_mean_temperature_rise(strip_board):
    settings = SolveSettings(convection_top_w_m2k=10.0, convection_bottom_w_m2k=10.0, ambient_c=25.0)
    model = _model(strip_board, settings)
    dc = model.solve_dc([Electrode("P1", "1", current=10.0), Electrode("P2", "1", current=-10.0)])
    thermal = model.solve_thermal(dc)

    # Steady state: everything generated leaves by convection.
    assert thermal.convected_power == pytest.approx(dc.total_power, rel=1e-9)
    # So the area-weighted mean surface rise is exactly P / (h A) over both faces.
    area = 2 * 50e-3 * 10e-3
    mean_rise = 0.0
    for lf in (model.by_name["F.Cu"], model.by_name["B.Cu"]):
        rise = thermal.temperature[lf.layer] - 25.0
        weights = np.asarray(mass_matrix(lf, np.ones(lf.mesh.nelements)).sum(axis=0)).ravel()
        mean_rise += float(weights @ rise)  # ∫ ΔT dA
    mean_rise /= area
    assert mean_rise == pytest.approx(dc.total_power / (10.0 * area), rel=1e-9)
    # The strip runs hotter than the bare FR4 around it, and the top (copper) side hotter than the bottom.
    assert thermal.max_temperature > 25.0 + mean_rise
    assert thermal.temperature["F.Cu"].max() > thermal.temperature["B.Cu"].max()


def test_two_layer_thermal_conserves_energy(two_layer_board):
    model = _model(two_layer_board)
    dc = model.solve_dc([Electrode("P1", "1", current=5.0), Electrode("P2", "1", current=-5.0)])
    thermal = model.solve_thermal(dc)
    assert dc.barrel_heat.sum() > 0
    assert thermal.convected_power == pytest.approx(dc.total_power, rel=1e-9)


def test_electrode_covers_every_pad_with_the_number(tmp_path):
    # A footprint with two separate pads both numbered "1" (like split thermal pads):
    # current entering "P1.1" must use both.
    split = """  (footprint "T:P" (layer "F.Cu") (at 7 5)
    (property "Reference" "P1")
    (pad "1" smd rect (at 0 -1.5) (size 4 3) (layers "F.Cu") (net 1 "N"))
    (pad "1" smd rect (at 0 1.5) (size 4 3) (layers "F.Cu") (net 1 "N")))
"""
    path = tmp_path / "split.kicad_pcb"
    path.write_text(_HEADER + _strip("F.Cu") + split + _pad("P2", 43, "F.Cu") + ")\n")
    board = load_board(path)
    assert len(board.find_pads("P1", "1")) == 2
    dc = _model(board).solve_dc([Electrode("P1", "1", voltage=1.0), Electrode("P2", "1", voltage=0.0)])
    # Together the two halves span the full strip width, so the strip result is unchanged.
    assert dc.resistance(("P1", "1"), ("P2", "1")) == pytest.approx(R_SQUARE * 32 / 6, rel=1e-6)


def test_one_sided_convection_still_balances(strip_board):
    # Bottom face insulated: all heat must leave from the top, which then runs hotter.
    both = _model(strip_board, SolveSettings(convection_top_w_m2k=10, convection_bottom_w_m2k=10))
    top_only = _model(strip_board, SolveSettings(convection_top_w_m2k=10, convection_bottom_w_m2k=0))
    electrodes = [Electrode("P1", "1", current=10.0), Electrode("P2", "1", current=-10.0)]
    t_both = both.solve_thermal(both.solve_dc(electrodes))
    dc = top_only.solve_dc(electrodes)
    t_top = top_only.solve_thermal(dc)
    assert t_top.convected_power == pytest.approx(dc.total_power, rel=1e-9)
    assert t_top.max_temperature > t_both.max_temperature


def test_ambient_shifts_temperature_uniformly(strip_board):
    # The model is linear in temperature, so ambient is a pure offset. (One model,
    # so both solves share a mesh: threaded meshing isn't bit-for-bit repeatable.)
    model = _model(strip_board, SolveSettings(ambient_c=25.0))
    dc = model.solve_dc([Electrode("P1", "1", current=10.0), Electrode("P2", "1", current=-10.0)])
    t_cool = model.solve_thermal(dc).temperature["F.Cu"]
    model.settings = SolveSettings(ambient_c=60.0)
    t_hot = model.solve_thermal(dc).temperature["F.Cu"]
    np.testing.assert_allclose(t_hot - t_cool, 35.0, atol=1e-9)


def test_copper_weight_from_stackup_sets_sheet_resistance(tmp_path):
    # Same strip, 2 oz copper in the stackup: exactly half the resistance.
    header = _HEADER.replace(
        '(layer "F.Cu" (type "copper") (thickness 0.035))', '(layer "F.Cu" (type "copper") (thickness 0.07))'
    )
    path = tmp_path / "heavy.kicad_pcb"
    path.write_text(header + _strip("F.Cu") + _pad("P1", 7, "F.Cu") + _pad("P2", 43, "F.Cu") + ")\n")
    dc = _model(load_board(path)).solve_dc(
        [Electrode("P1", "1", voltage=1.0), Electrode("P2", "1", voltage=0.0)]
    )
    assert dc.resistance(("P1", "1"), ("P2", "1")) == pytest.approx(R_SQUARE * 32 / 6 / 2, rel=1e-6)


# -- heat loads ---------------------------------------------------------------------------

POWER, FIXED, RESISTANCE = HeatLoadKind.POWER, HeatLoadKind.TEMPERATURE, HeatLoadKind.RESISTANCE


def _surface_mean_rise(model, thermal, ambient):
    """Area-weighted mean rise over the top and bottom faces, and that total area (m²)."""
    total, area = 0.0, 0.0
    for lf in (model.layers[0], model.layers[-1]):
        weights = np.asarray(mass_matrix(lf, np.ones(lf.mesh.nelements)).sum(axis=0)).ravel()
        total += float(weights @ (thermal.temperature[lf.layer] - ambient))
        area += float(weights.sum())
    return total / area, area


def test_heat_only_power_source_matches_convection(strip_board):
    # No current: a 2 W pad source must all leave by convection, so the mean
    # surface rise is exactly P / (h A), as for Joule heat.
    settings = SolveSettings(convection_top_w_m2k=10.0, convection_bottom_w_m2k=10.0, ambient_c=25.0)
    model = _model(strip_board, settings)
    thermal = model.solve_thermal(None, [HeatLoad("P1", "1", POWER, 2.0)])
    assert thermal.convected_power == pytest.approx(2.0, rel=1e-9)
    mean_rise, area = _surface_mean_rise(model, thermal, 25.0)
    assert mean_rise == pytest.approx(2.0 / (10.0 * area), rel=1e-9)
    assert thermal.load_heat[("P1", "1")] == 2.0
    # The source pad is the hottest place on the board.
    assert thermal.load_temperatures[("P1", "1")] > thermal.temperature["F.Cu"].mean()
    # Linear: twice the power, twice the rise.
    double = model.solve_thermal(None, [HeatLoad("P1", "1", POWER, 4.0)])
    np.testing.assert_allclose(
        double.temperature["F.Cu"] - 25, 2 * (thermal.temperature["F.Cu"] - 25), rtol=1e-9
    )


def test_fixed_temperature_sink_balances_heat(strip_board):
    model = _model(strip_board, SolveSettings(ambient_c=25.0))
    dc = model.solve_dc([Electrode("P1", "1", current=10.0), Electrode("P2", "1", current=-10.0)])
    loads = [HeatLoad("P1", "1", POWER, 1.5), HeatLoad("P2", "1", FIXED, 20.0)]
    thermal = model.solve_thermal(dc, loads)

    # Every node of the sink pad is held at 20 °C.
    nodes, _ = model.pad_nodes("P2", "1")
    f_cu = model.by_name["F.Cu"]
    assert np.allclose(np.concatenate(list(thermal.temperature.values()))[nodes], 20.0)
    assert thermal.load_temperatures[("P2", "1")] == pytest.approx(20.0)
    # Energy: Joule + source + sink (negative here: it removes heat) = convected.
    sink = thermal.load_heat[("P2", "1")]
    assert sink < 0
    assert dc.total_power + 1.5 + sink == pytest.approx(thermal.convected_power, rel=1e-8)
    # A sink cooler than ambient also pulls heat in from the air: the board runs cooler overall.
    plain = model.solve_thermal(dc, [HeatLoad("P1", "1", POWER, 1.5)])
    assert thermal.temperature[f_cu.layer].mean() < plain.temperature[f_cu.layer].mean()


def test_thermal_resistance_sink(strip_board):
    model = _model(strip_board, SolveSettings(ambient_c=25.0))
    source = HeatLoad("P1", "1", POWER, 2.0)
    thermal = model.solve_thermal(None, [source, HeatLoad("P2", "1", RESISTANCE, 5.0)])
    flow = -thermal.load_heat[("P2", "1")]
    # The heatsink carries (T_pad - T_ambient) / R, and the books balance.
    assert flow == pytest.approx((thermal.load_temperatures[("P2", "1")] - 25.0) / 5.0, rel=1e-9)
    assert 0 < flow < 2.0
    assert 2.0 - flow == pytest.approx(thermal.convected_power, rel=1e-8)
    # A better heatsink takes more of the heat and cools the source.
    better = model.solve_thermal(None, [source, HeatLoad("P2", "1", RESISTANCE, 0.5)])
    assert -better.load_heat[("P2", "1")] > flow
    assert better.load_temperatures[("P1", "1")] < thermal.load_temperatures[("P1", "1")]
    # A near-perfect heatsink approaches a pad fixed at ambient.
    ideal = model.solve_thermal(None, [source, HeatLoad("P2", "1", RESISTANCE, 1e-9)])
    fixed = model.solve_thermal(None, [source, HeatLoad("P2", "1", FIXED, 25.0)])
    assert ideal.load_heat[("P2", "1")] == pytest.approx(fixed.load_heat[("P2", "1")], rel=1e-5)


def test_heat_load_on_missing_pad_is_an_error(strip_board):
    model = _model(strip_board)
    with pytest.raises(SolveError, match="Q9.1"):
        model.solve_thermal(None, [HeatLoad("Q9", "1", POWER, 1.0)])


# -- resistive elements -------------------------------------------------------------------


def _zone(x0: float, x1: float, layer: str = "F.Cu") -> str:
    pts = f"(xy {x0} 2) (xy {x1} 2) (xy {x1} 8) (xy {x0} 8)"
    return f"""  (zone (net 1) (net_name "N") (layer "{layer}")
    (polygon (pts {pts})) (filled_polygon (layer "{layer}") (pts {pts})))
"""


_RESISTOR = """  (footprint "T:R" (layer "F.Cu") (at 25 5)
    (property "Reference" "R1")
    (pad "1" smd rect (at -5 0) (size 4 6) (layers "F.Cu") (net 1 "N"))
    (pad "2" smd rect (at 5 0) (size 4 6) (layers "F.Cu") (net 1 "N")))
"""


@pytest.fixture
def bridged_board(tmp_path):
    """Two F.Cu strips (x=5..22 and 28..45) joined only by resistor R1 (pads at x=18..22 and 28..32)."""
    path = tmp_path / "bridged.kicad_pcb"
    text = _HEADER + _zone(5, 22) + _zone(28, 45) + _pad("P1", 7, "F.Cu") + _pad("P2", 43, "F.Cu") + _RESISTOR
    path.write_text(text + ")\n")
    return load_board(path)


def _element(**kwargs):
    from kits.config import PadGroup, ResistiveElement

    values = {"resistance_ohm": 1e-3} | kwargs
    return ResistiveElement("R1", "R1", PadGroup("A", ["1"]), PadGroup("B", ["2"]), **values)


def test_resistive_element_in_series(bridged_board):
    model = _model(bridged_board)
    element = _element()
    electrodes = [Electrode("P1", "1", voltage=1e-3), Electrode("P2", "1", voltage=0.0)]
    with pytest.raises(SolveError):  # without the element the strips aren't connected
        model.solve_thermal(
            model.solve_dc([Electrode("P1", "1", current=1), Electrode("P2", "1", current=-1)])
        )
    dc = model.solve_dc(electrodes, [element])
    # Series: left strip (pad edge to pad edge) + element + right strip; each
    # terminal is equipotential, like a soldered end cap.
    strips = R_SQUARE * ((18 - 9) + (41 - 32)) / 6
    assert dc.resistance(("P1", "1"), ("P2", "1")) == pytest.approx(strips + 1e-3, rel=0.01)
    # All the current goes through the element, and its power is I²R.
    current = dc.electrode_currents[("P1", "1")]
    assert dc.element_currents["R1"] == pytest.approx(current, rel=1e-9)
    assert dc.element_power["R1"] == pytest.approx(current**2 * 1e-3, rel=1e-9)
    assert dc.total_power == pytest.approx(current * 1e-3, rel=1e-9)  # P = V I, element included


def test_resistive_element_heat(bridged_board):
    from kits.solve.elements import body_area_mm2

    settings = SolveSettings(convection_top_w_m2k=10.0, convection_bottom_w_m2k=10.0, ambient_c=25.0)
    model = _model(bridged_board, settings)
    electrodes = [Electrode("P1", "1", current=10.0), Electrode("P2", "1", current=-10.0)]
    ideal = _element(resistance_ohm=0.01)
    dc = model.solve_dc(electrodes, [ideal])
    # 10 A through 10 mΩ (to the 1e-12 leakage that keeps floating copper solvable).
    assert dc.element_power["R1"] == pytest.approx(1.0, rel=1e-6)

    # Ideal contact: the body is its pads; by default it also convects from its body area.
    thermal = model.solve_thermal(dc, elements=[ideal])
    footprint = next(fp for fp in bridged_board.footprints if fp.reference == "R1")
    assert body_area_mm2(footprint) == pytest.approx(14 * 6)
    nodes, weights = model.pad_nodes("R1", ["1", "2"])
    temperature = np.concatenate(list(thermal.temperature.values()))
    body = thermal.element_temperatures["R1"]
    assert body == pytest.approx(float(weights @ temperature[nodes]))
    assert thermal.element_ambient_heat["R1"] == pytest.approx(10.0 * 84e-6 * (body - 25.0), rel=1e-9)
    # Energy: board convection + the body's own loss to ambient = all Joule heat.
    total_out = thermal.convected_power + thermal.element_ambient_heat["R1"]
    assert total_out == pytest.approx(dc.total_power, rel=1e-8)

    # Through a 20 K/W junction-to-board resistance the body runs hotter, by exactly
    # (heat into the board) x R above its pads.
    jb = _element(resistance_ohm=0.01, thermal_resistance_board_k_w=20.0, thermal_resistance_ambient_k_w=50.0)
    hot = model.solve_thermal(dc, elements=[jb])
    pads = float(weights @ np.concatenate(list(hot.temperature.values()))[nodes])
    to_ambient = hot.element_ambient_heat["R1"]
    assert to_ambient == pytest.approx((hot.element_temperatures["R1"] - 25.0) / 50.0, rel=1e-9)
    assert hot.element_temperatures["R1"] - pads == pytest.approx((1.0 - to_ambient) * 20.0, rel=1e-6)
    assert hot.convected_power + to_ambient == pytest.approx(dc.total_power, rel=1e-8)
    # Same default cooling as the ideal case: the junction runs hotter than the pads did.
    junction = model.solve_thermal(
        dc, elements=[_element(resistance_ohm=0.01, thermal_resistance_board_k_w=20.0)]
    )
    assert junction.element_temperatures["R1"] > body


def test_auto_pad_groups(tmp_path):
    from kits.solve.elements import auto_pad_groups

    mosfet = """  (footprint "T:Q" (layer "F.Cu") (at 25 5)
    (property "Reference" "Q1")
    (pad "1" smd rect (at -2 -2) (size 1 1.5) (layers "F.Cu") (net 2 "S"))
    (pad "2" smd rect (at -1 -2) (size 1 1.5) (layers "F.Cu") (net 2 "S"))
    (pad "3" smd rect (at 0 -2) (size 1 1.5) (layers "F.Cu") (net 2 "S"))
    (pad "4" smd rect (at 1 -2) (size 1 1.5) (layers "F.Cu") (net 3 "G"))
    (pad "5" smd rect (at 0 1) (size 5 4) (layers "F.Cu") (net 4 "D"))
    (pad "5" smd rect (at 3 1) (size 1 1) (layers "F.Cu") (net 4 "D"))
    (pad "" np_thru_hole circle (at 4 4) (size 1 1) (drill 1) (layers "*.Cu")))
"""
    header = _HEADER.replace('(net 1 "N")', '(net 1 "N") (net 2 "S") (net 3 "G") (net 4 "D")')
    path = tmp_path / "q.kicad_pcb"
    path.write_text(header + mosfet + _RESISTOR + ")\n")
    board = load_board(path)
    q1 = next(fp for fp in board.footprints if fp.reference == "Q1")
    a, b = auto_pad_groups(q1)
    assert (a.name, a.pads) == ("S", ["1", "2", "3"])  # the gate is left out
    assert (b.name, b.pads) == ("D", ["5"])
    # Both pads on one net: split by number.
    r1 = next(fp for fp in board.footprints if fp.reference == "R1")
    a, b = auto_pad_groups(r1)
    assert (a.name, a.pads, b.name, b.pads) == ("A", ["1"], "B", ["2"])
