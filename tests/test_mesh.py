import json

import numpy as np
import pytest
import shapely
from shapely.geometry import Point

from kits.config import ConfigError, MeshSettings, ProjectConfig, config_path_for, load_config
from kits.ingest import load_board
from kits.ingest.geometry import rotate
from kits.ingest.model import Graphic, Pad
from kits.mesh import cli
from kits.mesh.geometry import (
    GeometryError,
    board_outline,
    extract_layers,
    layer_anchors,
    pad_area,
)
from kits.mesh.mesher import layer_heights, mesh_board
from kits.mesh.model import BoardMesh

# -- config ----------------------------------------------------------------------


def test_config_defaults_when_missing(tmp_path):
    config = load_config(tmp_path / "x.kits.yaml")
    assert not config.exists
    assert config.mesh == MeshSettings(1.0, 0.1)


def test_config_round_trip_preserves_unknown_keys(tmp_path):
    path = tmp_path / "x.kits.yaml"
    path.write_text("ambient_temperature_c: 25\nmesh:\n  max_size_mm: 2\n", encoding="utf-8")
    config = load_config(path)
    assert config.mesh == MeshSettings(2.0, 0.1)
    config.mesh = MeshSettings(0.5, 0.05)
    config.save()
    reloaded = load_config(path)
    assert reloaded.mesh == MeshSettings(0.5, 0.05)
    assert reloaded.extra == {"ambient_temperature_c": 25}


@pytest.mark.parametrize(
    "text",
    [
        "mesh:\n  max_size_mm: 0.1\n  min_size_mm: 0.5\n",  # min > max
        "mesh:\n  max_size_mm: -1\n",
        "mesh:\n  max_size_mm: big\n",
        "mesh: 3\n",
        "- a list\n",
        "mesh: [unclosed\n",
    ],
)
def test_config_rejects_invalid(tmp_path, text):
    path = tmp_path / "x.kits.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_config_path_for(tmp_path):
    assert config_path_for(tmp_path / "board.kicad_pro") == tmp_path / "board.kits.yaml"
    assert config_path_for(tmp_path / "board.kicad_pcb") == tmp_path / "board.kits.yaml"


def test_config_save_rejects_invalid(tmp_path):
    config = ProjectConfig(tmp_path / "x.kits.yaml", MeshSettings(0.1, 1.0))
    with pytest.raises(ConfigError):
        config.save()
    assert not config.exists


# -- geometry --------------------------------------------------------------------


def test_pad_rotation_matches_kicad_convention():
    # A custom pad whose only primitive sits at local +X; rotating by 90 deg
    # must move it where ingest's KiCad rotation sends (1, 0).
    prim = Graphic("poly", "", [(0.9, -0.1), (1.1, -0.1), (1.1, 0.1), (0.9, 0.1)], 0, True)
    pad = Pad("1", "smd", "custom", (10, 20), 90, (0.2, 0.2), ["F.Cu"], primitives=[prim])
    centroid = pad_area(pad).difference(Point(10, 20).buffer(0.15)).centroid
    expected = rotate((1, 0), 90)
    assert (centroid.x, centroid.y) == pytest.approx((10 + expected[0], 20 + expected[1]), abs=1e-6)


def test_rect_pad_area_and_rotation():
    pad = Pad("1", "smd", "rect", (0, 0), 90, (2, 1), ["F.Cu"])
    shape = pad_area(pad)
    assert shape.area == pytest.approx(2)
    x0, y0, x1, y1 = shape.bounds
    assert (x1 - x0, y1 - y0) == pytest.approx((1, 2))


def test_board_outline(board_v9_path):
    board = load_board(board_v9_path)
    assert board_outline(board).area == pytest.approx(40 * 30)
    # A closed circle inside the outline is a cutout.
    board.graphics.append(Graphic("circle", "Edge.Cuts", [(20, 15), (22, 15)]))
    assert board_outline(board).area == pytest.approx(1200 - np.pi * 4, rel=1e-3)


def test_board_outline_errors(board_v9_path, board_v10_path):
    with pytest.raises(GeometryError, match="no outline"):
        board_outline(load_board(board_v10_path))
    board = load_board(board_v9_path)
    board.graphics = [
        Graphic("line", "Edge.Cuts", [(0, 0), (10, 0)]),
        Graphic("line", "Edge.Cuts", [(10, 0), (10, 5)]),
    ]
    with pytest.raises(GeometryError, match="not closed"):
        board_outline(board)


def test_extract_layers_partitions_board(board_v9_path):
    board = load_board(board_v9_path)
    layers = {g.layer: g for g in extract_layers(board, MeshSettings())}
    assert list(layers) == ["F.Cu", "In1.Cu", "B.Cu"]
    for geom in layers.values():
        assert geom.total_area == pytest.approx(1200, rel=1e-6)
        union = shapely.union_all([f.polygon for f in geom.faces])
        assert union.area == pytest.approx(geom.total_area, rel=1e-9)  # faces don't overlap
    assert layers["F.Cu"].copper_area > layers["In1.Cu"].copper_area > 0


def test_via_anchors_follow_via_span(board_v9_path):
    board = load_board(board_v9_path)
    # Through via at (5, 5) spans everything; blind via at (6, 6) is F.Cu-In1.Cu only.
    # The through-hole pad of R1 sits at (10, 19).
    anchors = sorted(layer_anchors(board, "F.Cu"))
    assert anchors == pytest.approx([(5, 5), (6, 6), (10, 19)])
    assert (6, 6) in layer_anchors(board, "In1.Cu")
    assert (6, 6) not in layer_anchors(board, "B.Cu")
    assert (5, 5) in layer_anchors(board, "B.Cu")


def test_layer_heights_from_stackup(board_v9_path):
    heights = layer_heights(load_board(board_v9_path))
    assert heights["B.Cu"] == pytest.approx(0.0175)
    assert heights["In1.Cu"] == pytest.approx(0.035 + 1.2 + 0.0175)
    assert heights["F.Cu"] == pytest.approx(0.035 + 1.2 + 0.035 + 0.2 + 0.0175)


def test_layer_heights_without_stackup(board_v9_path):
    board = load_board(board_v9_path)
    board.stackup = []
    assert layer_heights(board) == pytest.approx({"F.Cu": 1.6, "In1.Cu": 0.8, "B.Cu": 0.0})


# -- mesher ----------------------------------------------------------------------


def _has_node(lm, xy, tol=1e-6):
    return bool((np.linalg.norm(lm.nodes - np.array(xy), axis=1) < tol).any())


def test_mesh_board_covers_board_and_embeds_anchors(board_v9_path, tmp_path):
    board = load_board(board_v9_path)
    settings = MeshSettings(max_size_mm=2.0, min_size_mm=0.1)
    progress = []
    mesh = mesh_board(board, settings, tmp_path / "out" / "demo.msh", lambda m, f: progress.append(f))

    assert [lm.layer for lm in mesh.layers] == ["F.Cu", "In1.Cu", "B.Cu"]
    geometry = {g.layer: g for g in extract_layers(board, settings)}
    for lm in mesh.layers:
        areas = lm.triangle_areas()
        assert areas.min() > 0
        assert areas.sum() == pytest.approx(1200, rel=1e-6)
        assert areas[lm.copper].sum() == pytest.approx(geometry[lm.layer].copper_area, rel=1e-6)
        edges, _ = lm.edges()
        lengths = np.linalg.norm(lm.nodes[edges[:, 0]] - lm.nodes[edges[:, 1]], axis=1)
        assert lengths.max() < 1.6 * settings.max_size_mm  # gmsh treats max size as a target

    assert _has_node(mesh.layer("B.Cu"), (5, 5))  # through via
    assert _has_node(mesh.layer("In1.Cu"), (6, 6))  # blind via
    assert progress == sorted(progress) and progress[-1] == 1.0
    assert mesh.msh_path.is_file()


def test_msh_file_has_named_groups_and_heights(board_v9_path, tmp_path):
    import gmsh

    board = load_board(board_v9_path)
    mesh = mesh_board(board, MeshSettings(2.0, 0.1), tmp_path / "demo.msh")
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(tmp_path / "demo.msh"))
        names = {gmsh.model.getPhysicalName(d, t) for d, t in gmsh.model.getPhysicalGroups()}
        _, coords, _ = gmsh.model.mesh.getNodes()
    finally:
        gmsh.finalize()
    assert names == {
        f"{layer}/{region}" for layer in board.copper_layers for region in ("copper", "dielectric")
    }
    assert len(coords) // 3 == mesh.num_nodes
    assert sorted(set(np.round(coords[2::3], 6))) == pytest.approx(sorted(layer_heights(board).values()))


def test_npz_round_trip(board_v9_path, tmp_path):
    mesh = mesh_board(load_board(board_v9_path), MeshSettings(2.0, 0.1), tmp_path / "a.msh")
    mesh.save_npz(tmp_path / "m.npz")
    loaded = BoardMesh.load_npz(tmp_path / "m.npz")
    assert loaded.settings == mesh.settings
    assert loaded.msh_path == mesh.msh_path
    for a, b in zip(mesh.layers, loaded.layers, strict=True):
        assert a.layer == b.layer and a.z_mm == b.z_mm
        np.testing.assert_array_equal(a.triangles, b.triangles)
        np.testing.assert_array_equal(a.copper, b.copper)


# -- command line ----------------------------------------------------------------


def test_cli_uses_project_config_and_default_output(board_v9_path, capsys):
    project = board_v9_path.with_suffix(".kicad_pro")
    config_path_for(project).write_text("mesh:\n  max_size_mm: 3\n  min_size_mm: 0.2\n", encoding="utf-8")
    npz = board_v9_path.parent / "m.npz"
    assert cli.main([str(project), "--progress", "--npz", str(npz)]) == 0

    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert all("progress" in r for r in records[:-1])
    assert records[-1]["done"] and records[-1]["triangles"] > 0
    assert (board_v9_path.parent / "kits_output" / "demo.msh").is_file()
    assert BoardMesh.load_npz(npz).settings == MeshSettings(3.0, 0.2)


def test_cli_reports_errors_as_json(board_v10_path, capsys):
    assert cli.main([str(board_v10_path), "--progress"]) == 1
    (record,) = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert "outline" in record["error"]


def test_cli_reports_unwritable_output_before_meshing(board_v9_path, tmp_path, capsys, monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "mesh_board", lambda *a, **k: calls.append(a))
    blocker = tmp_path / "file"
    blocker.write_text("not a folder")
    assert cli.main([str(board_v9_path), "--progress", "-o", str(blocker / "sub" / "x.msh")]) == 1
    (record,) = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert "cannot create output folder" in record["error"]
    assert not calls  # failed before any meshing
