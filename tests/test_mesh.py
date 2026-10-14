import json

import numpy as np
import pytest
import shapely
from shapely.geometry import Point

from kits.config import MeshSettings
from kits.ingest import load_board
from kits.ingest.geometry import rotate
from kits.ingest.model import Drill, Graphic, Pad
from kits.mesh import cli
from kits.mesh.fingerprint import MeshFingerprint
from kits.mesh.geometry import (
    GeometryError,
    board_holes,
    board_outline,
    circle_points,
    extract_layers,
    hole_polygon,
    pad_area,
)
from kits.mesh.mesher import layer_heights, mesh_board
from kits.mesh.model import BoardMesh, load_saved_mesh

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
        holes = shapely.union_all(geom.holes).area
        assert geom.total_area == pytest.approx(1200 - holes, rel=1e-6)
        union = shapely.union_all([f.polygon for f in geom.faces])
        assert union.area == pytest.approx(geom.total_area, rel=1e-9)  # faces don't overlap
    assert layers["F.Cu"].copper_area > layers["In1.Cu"].copper_area > 0


def test_holes_follow_via_span(board_v9_path):
    board = load_board(board_v9_path)
    holes = {
        (round(h.polygon.centroid.x, 3), round(h.polygon.centroid.y, 3)): h
        for h in board_holes(board, MeshSettings())
    }
    # Through via at (5, 5) spans everything; blind via at (6, 6) is F.Cu-In1.Cu only;
    # R1's through-hole pad at (10, 19) is drilled through the whole board.
    assert set(holes) == {(5, 5), (6, 6), (10, 19)}
    assert holes[(5, 5)].layers == ["F.Cu", "In1.Cu", "B.Cu"]
    assert holes[(6, 6)].layers == ["F.Cu", "In1.Cu"]
    assert holes[(10, 19)].layers == ["F.Cu", "In1.Cu", "B.Cu"] and holes[(10, 19)].plated


def test_circle_points():
    settings = MeshSettings(max_size_mm=1.0, min_size_mm=0.1, min_circle_points=36)
    assert circle_points(0.15, settings) == 36  # small hole: the minimum applies
    assert circle_points(10.0, settings) == 63  # large hole: segments capped at max size
    assert circle_points(0.15, MeshSettings(1.0, 0.1, 48)) == 48


def test_round_hole_polygon_vertices_lie_on_the_drill_edge():
    settings = MeshSettings()
    hole = hole_polygon(Drill(0.3, 0.3), (5, 5), 0, settings)
    pts = np.array(hole.exterior.coords)[:-1]
    assert len(pts) == 36
    assert np.linalg.norm(pts - [5, 5], axis=1) == pytest.approx(np.full(36, 0.15), abs=1e-4)


def test_slot_hole_polygon_follows_pad_rotation():
    settings = MeshSettings()
    slot = hole_polygon(Drill(2.0, 1.0), (0, 0), 0, settings)
    x0, y0, x1, y1 = slot.bounds
    assert (x1 - x0, y1 - y0) == pytest.approx((2.0, 1.0), abs=1e-3)
    assert slot.area == pytest.approx(1.0 * 1.0 + np.pi * 0.25, rel=1e-2)  # stadium
    x0, y0, x1, y1 = hole_polygon(Drill(2.0, 1.0), (0, 0), 90, settings).bounds
    assert (x1 - x0, y1 - y0) == pytest.approx((1.0, 2.0), abs=1e-3)
    x0, y0, x1, y1 = hole_polygon(Drill(1.0, 2.0), (0, 0), 0, settings).bounds
    assert (x1 - x0, y1 - y0) == pytest.approx((1.0, 2.0), abs=1e-3)


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


def test_mesh_board_covers_board_minus_holes(board_v9_path, tmp_path):
    board = load_board(board_v9_path)
    settings = MeshSettings(max_size_mm=2.0, min_size_mm=0.1)
    progress = []
    mesh = mesh_board(board, settings, tmp_path / "out" / "demo.msh", lambda m, f: progress.append(f))

    assert [lm.layer for lm in mesh.layers] == ["F.Cu", "In1.Cu", "B.Cu"]
    geometry = {g.layer: g for g in extract_layers(board, settings)}
    for lm in mesh.layers:
        areas = lm.triangle_areas()
        assert areas.min() > 0
        assert areas.sum() == pytest.approx(geometry[lm.layer].total_area, rel=1e-6)
        assert areas.sum() < 1200  # holes are cut out
        assert areas[lm.copper].sum() == pytest.approx(geometry[lm.layer].copper_area, rel=1e-6)
        edges, _ = lm.edges()
        lengths = np.linalg.norm(lm.nodes[edges[:, 0]] - lm.nodes[edges[:, 1]], axis=1)
        assert lengths.max() < 1.6 * settings.max_size_mm  # gmsh treats max size as a target

    # Every hole-edge vertex is a node on every layer the hole cuts (so plated
    # barrels can link layers there), and no node lies inside a hole.
    for lm in mesh.layers:
        for hole in geometry[lm.layer].holes:
            for xy in list(hole.exterior.coords)[:-1]:
                assert _has_node(lm, xy, tol=1e-6)
            centroid = np.array(hole.centroid.coords[0])
            radius = hole.exterior.distance(hole.centroid)
            assert np.linalg.norm(lm.nodes - centroid, axis=1).min() >= radius - 1e-6
    via_ring = list(hole_polygon(Drill(0.3, 0.3), (5, 5), 0, settings).exterior.coords)[:-1]
    assert all(_has_node(mesh.layer(layer), xy) for layer in ("F.Cu", "In1.Cu", "B.Cu") for xy in via_ring)
    blind_ring = list(hole_polygon(Drill(0.2, 0.2), (6, 6), 0, settings).exterior.coords)[:-1]
    assert not any(_has_node(mesh.layer("B.Cu"), xy) for xy in blind_ring)
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


def test_cli_meshes_project_with_its_settings(kits_v9_path, capsys):
    from kits.project import load_kits_project

    project = load_kits_project(kits_v9_path)
    project.mesh = MeshSettings(3.0, 0.2)
    project.save()
    assert cli.main([str(kits_v9_path), "--progress"]) == 0

    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert all("progress" in r for r in records[:-1])
    done = records[-1]
    assert done["done"] and done["triangles"] > 0
    assert (kits_v9_path.parent / "kits_output" / "demo.msh").is_file()
    assert done["mesh"] == str(project.saved_mesh_path)
    saved = BoardMesh.load_npz(project.saved_mesh_path)
    assert saved.settings == MeshSettings(3.0, 0.2)
    assert saved.fingerprint == MeshFingerprint.of(project.board.with_suffix(".kicad_pcb"), project.mesh)


# -- saved meshes ------------------------------------------------------------------


def test_settings_hash_tracks_settings_and_mesher_version(monkeypatch):
    from kits.mesh import fingerprint

    a = fingerprint.settings_hash(MeshSettings(1.0, 0.1))
    assert a == fingerprint.settings_hash(MeshSettings(1.0, 0.1))
    assert a != fingerprint.settings_hash(MeshSettings(1.0, 0.2))
    monkeypatch.setattr(fingerprint, "MESHER_VERSION", fingerprint.MESHER_VERSION + 1)
    assert a != fingerprint.settings_hash(MeshSettings(1.0, 0.1))


def test_staleness_reasons():
    built = MeshFingerprint(settings="s1", board="b1")
    assert built.staleness(MeshFingerprint("s1", "b1")) is None
    assert (
        built.staleness(MeshFingerprint("s2", "b1"))
        == "The mesh settings changed since the saved mesh was built"
    )
    assert built.staleness(MeshFingerprint("s1", "b2")) == "The board changed since the saved mesh was built"
    assert "board and the mesh settings" in built.staleness(MeshFingerprint("s2", "b2"))


@pytest.fixture
def saved_mesh(board_v9_path, tmp_path):
    settings = MeshSettings(2.0, 0.1)
    mesh = mesh_board(load_board(board_v9_path), settings, tmp_path / "out" / "demo.msh")
    path = tmp_path / "out" / "demo.mesh.npz"
    mesh.save_npz(path)
    return path, settings


def test_load_saved_mesh_when_current(saved_mesh, board_v9_path):
    path, settings = saved_mesh
    mesh, notice = load_saved_mesh(path, board_v9_path, settings)
    assert notice is None and mesh.num_triangles > 0


def test_load_saved_mesh_rejects_changed_settings(saved_mesh, board_v9_path):
    path, _ = saved_mesh
    mesh, notice = load_saved_mesh(path, board_v9_path, MeshSettings(1.0, 0.1))
    assert mesh is None and "mesh settings changed" in notice


def test_load_saved_mesh_rejects_changed_board(saved_mesh, board_v9_path):
    path, settings = saved_mesh
    with open(board_v9_path, "a", encoding="utf-8") as f:
        f.write("\n")  # any byte change counts
    mesh, notice = load_saved_mesh(path, board_v9_path, settings)
    assert mesh is None and "board changed" in notice


def test_load_saved_mesh_missing_or_corrupt(saved_mesh, board_v9_path, tmp_path):
    path, settings = saved_mesh
    assert load_saved_mesh(tmp_path / "none.mesh.npz", board_v9_path, settings) == (None, None)
    path.write_bytes(b"not a zip")
    mesh, notice = load_saved_mesh(path, board_v9_path, settings)
    assert mesh is None and "couldn't be read" in notice


def test_mesh_saved_without_fingerprint_is_stale(saved_mesh, board_v9_path):
    path, settings = saved_mesh
    mesh = BoardMesh.load_npz(path)
    mesh.fingerprint = MeshFingerprint()  # as written before fingerprints existed
    mesh.save_npz(path)
    assert load_saved_mesh(path, board_v9_path, settings)[0] is None


def test_cli_reports_errors_as_json(kits_v10_path, capsys):
    assert cli.main([str(kits_v10_path), "--progress"]) == 1
    (record,) = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert "outline" in record["error"]


def test_cli_requires_a_kits_project(board_v9_path, capsys):
    assert cli.main([str(board_v9_path)]) == 1
    assert "expected a KITS project file" in capsys.readouterr().err


def test_cli_reports_unlinked_board(tmp_path, capsys):
    from kits.project import KitsProject

    KitsProject(tmp_path / "empty.kits").save()
    assert cli.main([str(tmp_path / "empty.kits")]) == 1
    assert "doesn't link a KiCad board" in capsys.readouterr().err


def test_cli_reports_unwritable_output_before_meshing(kits_v9_path, tmp_path, capsys, monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "mesh_board", lambda *a, **k: calls.append(a))
    blocker = tmp_path / "file"
    blocker.write_text("not a folder")
    assert cli.main([str(kits_v9_path), "--progress", "-o", str(blocker / "sub" / "x.msh")]) == 1
    (record,) = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert "cannot create output folder" in record["error"]
    assert not calls  # failed before any meshing
