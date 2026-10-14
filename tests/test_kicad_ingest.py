import math
from pathlib import Path

import pytest

from kits.ingest import BoardLoadError, load_board, load_project
from kits.ingest.geometry import arc_extent, arc_points, rotate
from kits.ingest.model import Graphic

from .conftest import kicad_demo_projects


def test_project_loads_board(board_v9_path):
    project = load_project(board_v9_path.with_suffix(".kicad_pro"))
    assert project.name == "demo"
    assert project.settings == {"meta": {"version": 3}}
    assert project.board.path == board_v9_path


def test_layers_and_stackup(board_v9_path):
    b = load_board(board_v9_path)
    assert b.version == 20241229
    assert b.thickness_mm == pytest.approx(1.6)
    assert b.copper_layers == ["F.Cu", "In1.Cu", "B.Cu"]
    assert [s.name for s in b.stackup][:2] == ["F.Cu", "dielectric 1"]
    assert b.stackup[1].material == "FR4"
    assert b.stackup[1].epsilon_r == pytest.approx(4.5)
    assert b.stackup[3].thickness_mm == pytest.approx(1.2)


def test_nets_v9(board_v9_path):
    b = load_board(board_v9_path)
    assert b.tracks[0].net == "+5V"  # resolved via the net table
    assert b.arcs[0].net == "GND"
    pads = {p.number: p for p in b.pads()}
    assert pads["1"].net == "GND"
    assert pads["2"].net == "+5V"
    assert set(b.nets) == {"GND", "+5V"}


def test_nets_v10(board_v10_path):
    b = load_board(board_v10_path)
    assert b.tracks[0].net == "/VPP{slash}MCLR"
    assert next(b.pads()).net == "VCC"
    assert b.zones[0].net == "GND"
    assert b.zones[0].layers == ["B.Cu"]


def test_footprint_and_pad_transform(board_v9_path):
    b = load_board(board_v9_path)
    fp = b.footprints[0]
    assert (fp.reference, fp.value, fp.rotation) == ("R1", "10k", 90)
    pads = {p.number: p for p in fp.pads}
    # Local (-1, 0) rotated 90 deg CCW (as displayed, Y down) -> (0, +1).
    assert pads["1"].position == pytest.approx((10, 21))
    assert pads["2"].position == pytest.approx((10, 19))
    assert pads["1"].layers == ["F.Cu"]
    assert pads["1"].shape == "roundrect"
    assert pads["2"].layers == ["F.Cu", "In1.Cu", "B.Cu"]  # *.Cu expanded
    assert pads["2"].drill.width == pytest.approx(0.8)
    assert pads["2"].footprint_ref == "R1"


def test_vias(board_v9_path):
    through, blind = load_board(board_v9_path).vias
    assert through.via_type == "through" and through.layers == ("F.Cu", "B.Cu")
    assert blind.via_type == "blind" and blind.layers == ("F.Cu", "In1.Cu")


def test_zones(board_v9_path):
    pour, keepout = load_board(board_v9_path).zones
    assert pour.name == "gnd pour"
    assert pour.layers == ["F.Cu", "B.Cu"]
    assert set(pour.fills) == {"F.Cu", "B.Cu"}
    assert len(pour.outline) == 4
    assert keepout.is_rule_area


def test_graphics_and_outline(board_v9_path):
    b = load_board(board_v9_path)
    # Silkscreen and text are dropped; the outline rect becomes a polygon.
    assert {g.layer for g in b.graphics} == {"Edge.Cuts", "B.Cu"}
    (outline,) = b.outline_graphics()
    assert outline.kind == "poly" and len(outline.points) == 4
    copper = next(g for g in b.graphics if g.layer == "B.Cu")
    assert copper.filled and copper.net == "GND"
    assert b.bounding_box() == (0, 0, 40, 30)


def test_errors(tmp_path):
    with pytest.raises(BoardLoadError):
        load_project(tmp_path / "missing.kicad_pro")
    (tmp_path / "x.kicad_pro").write_text("{}")
    with pytest.raises(BoardLoadError, match="No board file"):
        load_project(tmp_path / "x.kicad_pro")
    bad = tmp_path / "bad.kicad_pcb"
    bad.write_text("(kicad_pcb (version 1)")
    with pytest.raises(BoardLoadError, match="Malformed"):
        load_board(bad)
    sch = tmp_path / "s.kicad_pcb"
    sch.write_text("(kicad_sch (version 1))")
    with pytest.raises(BoardLoadError, match="Not a KiCad board"):
        load_board(sch)


def test_rotate_matches_kicad_convention():
    assert rotate((1, 0), 90) == pytest.approx((0, -1))  # CCW on a Y-down screen


def test_arc_points_pass_through_mid():
    pts = arc_points((0, 0), (1, 1), (2, 0))
    assert pts[0] == (0, 0) and pts[-1] == (2, 0)
    assert all(math.hypot(x - 1, y) == pytest.approx(1) for x, y in pts)
    assert max(y for _, y in pts) == pytest.approx(1, abs=1e-3)


def test_arc_extent_includes_bulge():
    # Arc over the top of the unit circle with an off-centre mid, so none of
    # its three defining points is the extreme (0, -1).
    s = 2**-0.5
    xs, ys = zip(*arc_extent((s, -s), (0.2, -((1 - 0.04) ** 0.5)), (-s, -s)), strict=True)
    assert min(ys) == pytest.approx(-1)
    # Same arc going the long way round (through the bottom).
    xs, ys = zip(*arc_extent((s, -s), (0, 1), (-s, -s)), strict=True)
    assert (min(xs), max(xs), max(ys)) == pytest.approx((-1, 1, 1))


def test_bounding_box_includes_arc_bulge(board_v9_path):
    b = load_board(board_v9_path)
    # A rounded corner bulging past the 40 x 30 outline.
    b.graphics.append(Graphic("arc", "Edge.Cuts", [(40, 0), (45, 5), (40, 10)], 0.1))
    assert b.bounding_box() == pytest.approx((0, 0, 45, 30))


def test_board_without_outline(board_v10_path):
    b = load_board(board_v10_path)
    assert not b.has_outline
    assert b.outline_bounding_box() is None
    # Falls back to the copper extent, including pad size and zones, rather
    # than collapsing to pad centres.
    assert b.bounding_box() == b.copper_bounding_box()
    x0, y0, x1, y1 = b.bounding_box()
    assert x0 == pytest.approx(-0.1) and y0 == pytest.approx(-0.1)  # 0.2 mm track at (0, 0)
    assert x1 == pytest.approx(5 + 0.5**0.5) and y1 == pytest.approx(5 + 0.5**0.5)


def test_example_simple_in_out():
    # The example's rectangle is on Eco2.User, not Edge.Cuts, so KiCad itself
    # reports no board outline either.
    path = Path(__file__).parent.parent / "examples" / "Simple-In-Out" / "Simple-In-Out.kicad_pro"
    if not path.is_file():
        pytest.skip("example project not present")
    b = load_project(path).board
    assert b.version == 20260206
    assert not b.has_outline
    assert {p.net for p in b.pads()} == {"/NET"}
    x0, y0, x1, y1 = b.bounding_box()
    assert (x1 - x0, y1 - y0) > (20, 5)  # covers the zones, not just pad centres


@pytest.mark.parametrize("project", kicad_demo_projects(), ids=lambda p: p.stem)
def test_kicad_demo_projects_load(project):
    b = load_project(project).board
    assert b.copper_layers
    assert b.footprints
    assert b.bounding_box() is not None
