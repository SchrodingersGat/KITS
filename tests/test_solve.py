"""Gate tests for the scikit-fem 2.5D solver against hand calculations.

Copper: σ = 5.8e7 S/m, 35 µm thick, so one square of sheet is
1 / (5.8e7 * 35e-6) = 0.4926 mΩ.
"""

import math

import numpy as np
import pytest

pytest.importorskip("skfem")

from kits.config import MeshSettings  # noqa: E402
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
    settings = SolveSettings(convection_w_m2k=10.0, ambient_c=25.0)
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
