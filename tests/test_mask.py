"""Solder mask: parsing openings, convection through the mask, and pad contacts blocked by it."""

import numpy as np
import pytest

pytest.importorskip("skfem")

from kits.config import HeatLoad, HeatLoadKind  # noqa: E402
from kits.ingest import load_board  # noqa: E402
from kits.mesh.geometry import mask_openings  # noqa: E402
from kits.solve.solver import Electrode, SolveError, SolveSettings  # noqa: E402

from .test_solve import _HEADER, _model, _strip  # noqa: E402

_MASKED_HEADER = _HEADER.replace(
    '(25 "Edge.Cuts" user))', '(25 "Edge.Cuts" user) (39 "F.Mask" user) (40 "B.Mask" user))'
).replace(
    '    (layer "F.Cu" (type "copper") (thickness 0.035))\n',
    '    (layer "F.Mask" (type "Top Solder Mask") (thickness 0.02))\n'
    '    (layer "F.Cu" (type "copper") (thickness 0.035))\n',
)


def _pad(ref: str, x: float, mask: str = ' "F.Mask"', extra: str = "") -> str:
    """A 4 x 6 mm SMD pad across the strip; ``mask`` adds its mask opening layer."""
    return f"""  (footprint "T:P" (layer "F.Cu") (at {x} 5)
    (property "Reference" "{ref}")
    (pad "1" smd rect (at 0 0) (size 4 6) (layers "F.Cu"{mask}) (net 1 "N"){extra}))
"""


def _board(tmp_path, *pads, header=_MASKED_HEADER, extra=""):
    path = tmp_path / "masked.kicad_pcb"
    path.write_text(header + _strip("F.Cu") + "".join(pads) + extra + ")\n")
    return load_board(path)


def test_openings_from_pads_graphics_and_clearance(tmp_path):
    graphic = '  (gr_rect (start 20 2) (end 24 4) (stroke (width 0)) (fill solid) (layer "F.Mask"))\n'
    setup = _MASKED_HEADER.replace("  (setup (stackup", "  (setup (pad_to_mask_clearance 0.1) (stackup")
    board = _board(
        tmp_path,
        _pad("P1", 7),  # board clearance 0.1: 4.2 x 6.2 opening (rounded corners)
        _pad("P2", 43, extra=" (solder_mask_margin -0.5)"),  # own margin: 3 x 5
        _pad("P3", 25, mask=""),  # no opening: covered
        header=setup,
        extra=graphic,
    )
    assert board.pad_to_mask_clearance_mm == 0.1 and len(board.mask_graphics) == 1
    pads = {p.footprint_ref: p for p in board.pads()}
    assert pads["P1"].mask_layers == ["F.Mask"] and pads["P1"].mask_margin is None
    assert pads["P2"].mask_margin == -0.5 and pads["P3"].mask_layers == []

    openings = mask_openings(board, "F")
    rounded = 4.2 * 6.2 - (4 - np.pi) * 0.1**2  # 0.1 mm outward offset rounds the corners
    assert openings.area == pytest.approx(rounded + 3 * 5 + 4 * 2, rel=1e-3)
    assert not openings.contains(_point(25, 5))  # P3 stays under the mask
    assert mask_openings(board, "B").is_empty  # B.Mask exists but nothing is open
    no_mask = _board(tmp_path, _pad("P1", 7), header=_HEADER)
    assert mask_openings(no_mask, "F") is None


def _point(x, y):
    from shapely.geometry import Point

    return Point(x, y)


def test_mask_adds_resistance_to_convection(tmp_path):
    settings = SolveSettings(convection_top_w_m2k=10.0, convection_bottom_w_m2k=10.0, ambient_c=25.0)
    masked = _model(_board(tmp_path, _pad("P1", 7), _pad("P2", 43)), settings)
    top = masked.layers[0]
    covered = masked.masked_triangles(top)
    assert covered.any() and not covered.all()  # the pads are open
    h = masked._surface_coefficient(top, 10.0, True, False)
    expected = 1 / (1 / 10.0 + 0.02e-3 / 0.25)  # 20 µm of mask at 0.25 W/mK
    np.testing.assert_allclose(h[covered], expected)
    np.testing.assert_allclose(h[~covered], 10.0)
    # The bottom has no B.Mask thickness in the stackup: the default 10 µm, nothing open.
    bottom = masked._surface_coefficient(masked.layers[-1], 10.0, False, True)
    np.testing.assert_allclose(bottom, 1 / (1 / 10.0 + 0.01e-3 / 0.25))

    # Heat still balances, and a more insulating mask runs the board hotter.
    load = [HeatLoad("P1", "1", HeatLoadKind.POWER, 2.0)]
    thermal = masked.solve_thermal(None, load)
    assert thermal.convected_power == pytest.approx(2.0, rel=1e-9)
    masked.settings = SolveSettings(
        convection_top_w_m2k=10.0, convection_bottom_w_m2k=10.0, solder_mask_thermal_conductivity_w_mk=0.001
    )
    insulated = masked.solve_thermal(None, load)
    assert insulated.temperature["F.Cu"].mean() > thermal.temperature["F.Cu"].mean()


def test_pads_only_connect_through_mask_openings(tmp_path):
    # P2's opening is 1 mm smaller all round (solder-mask-defined): current enters only there.
    board = _board(tmp_path, _pad("P1", 7), _pad("P2", 43, extra=" (solder_mask_margin -1)"))
    model = _model(board)
    nodes, weights = model.pad_nodes("P2", "1")
    xy = model.layers[0].nodes_mm[nodes - model.layers[0].offset]
    assert xy[:, 0].min() > 41.0 and xy[:, 0].max() < 45.0  # opening x 42..44 (+ one element)
    assert weights.sum() == pytest.approx(1.0)
    dc = model.solve_dc([Electrode("P1", "1", current=1.0), Electrode("P2", "1", current=-1.0)])
    assert dc.electrode_currents[("P2", "1")] == pytest.approx(-1.0, rel=1e-6)

    # A pad with no opening at all can't take a current source.
    covered = _model(_board(tmp_path, _pad("P1", 7), _pad("P2", 43, mask="")))
    with pytest.raises(SolveError, match="P2.1 is covered by solder mask"):
        covered.solve_dc([Electrode("P1", "1", current=1.0), Electrode("P2", "1", current=-1.0)])
    with pytest.raises(SolveError, match="covered by solder mask"):
        covered.solve_thermal(None, [HeatLoad("P2", "1", HeatLoadKind.POWER, 1.0)])


def test_mask_layers_in_the_board_view(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from kits.gui.board_scene import BoardScene

    _app = QApplication.instance() or QApplication([])  # scenes need one
    board = _board(tmp_path, _pad("P1", 7), _pad("P2", 43))
    scene = BoardScene(board)
    assert "F.Mask" in scene.layer_names() and "B.Mask" in scene.layer_names()
    assert not scene.layers["F.Mask"].is_copper
    scene.set_layer_visible("F.Mask", False)
    assert not scene.layers["F.Mask"].item.isVisible()
    assert "F.Mask" not in BoardScene(_board(tmp_path, _pad("P1", 7), header=_HEADER)).layer_names()
