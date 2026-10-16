"""Mesh simplification options: which tracks are meshed, refine areas and exclusion areas."""

import dataclasses

import numpy as np
import pytest
import shapely

from kits.config import ConfigError, MeshArea, MeshSettings
from kits.ingest import load_board
from kits.mesh import fingerprint
from kits.mesh.geometry import GeometryError, extract_layers, layer_copper, meshed_holes
from kits.mesh.mesher import mesh_board
from kits.mesh.model import BoardMesh


def _area(name="A1", kind="refine", x=4.0, y=4.0, w=3.0, h=3.0, **kw):
    return MeshArea(name, kind, x, y, w, h, **kw)


# -- settings ---------------------------------------------------------------------


def test_settings_round_trip_with_tracks_and_areas():
    settings = MeshSettings(
        tracks="nets",
        track_nets=("GND", "+5V"),
        areas=(_area(size_mm=0.05), _area("X1", "exclude", enabled=False)),
    )
    data = settings.to_dict()
    assert data["tracks"] == "nets" and data["track_nets"] == ["GND", "+5V"]
    assert "size_mm" not in data["areas"][1] and data["areas"][1]["enabled"] is False
    assert MeshSettings.from_dict(data) == settings


def test_default_settings_keep_their_old_form_and_fingerprint():
    # Older projects (and their saved meshes) mustn't change.
    assert set(MeshSettings().to_dict()) == {
        "max_size_mm",
        "min_size_mm",
        "min_circle_points",
        "corner_size_mm",
    }


@pytest.mark.parametrize(
    "settings, message",
    [
        (MeshSettings(tracks="some"), "tracks must be one of"),
        (MeshSettings(areas=(_area(), _area())), "more than one mesh area"),
        (MeshSettings(max_size_mm=1.0, areas=(_area(size_mm=2.0),)), "must not exceed the max size"),
        (MeshSettings(areas=(_area(kind="blur"),)), "kind must be"),
        (MeshSettings(areas=(_area(w=0),)), "width must be a positive number"),
        (MeshSettings(areas=(_area(name=""),)), "needs a name"),
    ],
)
def test_invalid_settings(settings, message):
    with pytest.raises(ConfigError, match=message):
        settings.validate()


def test_bad_area_entry_is_reported():
    with pytest.raises(ConfigError, match="mesh area is missing 'kind'"):
        MeshSettings.from_dict(
            {"areas": [{"name": "A", "x_mm": 0, "y_mm": 0, "width_mm": 1, "height_mm": 1}]}
        )


def test_fingerprint_ignores_what_doesnt_shape_the_mesh():
    base = MeshSettings()
    h = fingerprint.settings_hash(base)
    # Nets chosen while all tracks are meshed, and disabled areas, change nothing.
    assert fingerprint.settings_hash(dataclasses.replace(base, track_nets=("GND",))) == h
    assert fingerprint.settings_hash(dataclasses.replace(base, areas=(_area(enabled=False),))) == h
    assert fingerprint.settings_hash(dataclasses.replace(base, areas=(_area(),))) != h
    assert fingerprint.settings_hash(dataclasses.replace(base, tracks="none")) != h


def test_keeps_track():
    assert MeshSettings().keeps_track("GND") and MeshSettings().keeps_track(None)
    assert not MeshSettings(tracks="none").keeps_track("GND")
    by_net = MeshSettings(tracks="nets", track_nets=("GND",))
    assert by_net.keeps_track("GND") and not by_net.keeps_track("+5V") and not by_net.keeps_track(None)


# -- geometry ---------------------------------------------------------------------


def test_track_modes_choose_which_tracks_are_copper(board_v9_path):
    board = load_board(board_v9_path)
    # The fixture's only F.Cu track is 1..3, 2..4 on +5V; its B.Cu arc is on GND.
    track = shapely.LineString([(1, 2), (3, 4)]).buffer(0.125)
    probe = track.difference(layer_copper(board, "F.Cu", settings=MeshSettings(tracks="none")))

    def has_track(settings, layer="F.Cu"):
        return layer_copper(board, layer, settings=settings).covers(probe.representative_point())

    assert has_track(None) and has_track(MeshSettings())
    assert not has_track(MeshSettings(tracks="none"))
    assert has_track(MeshSettings(tracks="nets", track_nets=("+5V",)))
    assert not has_track(MeshSettings(tracks="nets", track_nets=("GND",)))
    arc_area = layer_copper(board, "B.Cu").area
    assert layer_copper(board, "B.Cu", settings=MeshSettings(tracks="none")).area < arc_area
    assert (
        layer_copper(board, "B.Cu", settings=MeshSettings(tracks="nets", track_nets=("GND",))).area
        == arc_area
    )


def test_exclusion_cuts_the_board_and_drops_holes(board_v9_path):
    board = load_board(board_v9_path)
    plain = MeshSettings()
    # Covers both vias (5,5) and (6,6), but not R1's through-hole pad (at (10,19): R1 is rotated 90°).
    excluded = MeshSettings(areas=(_area("X1", "exclude", x=4, y=4, w=3, h=3),))
    assert len(meshed_holes(board, plain)) == 3
    holes = meshed_holes(board, excluded)
    assert len(holes) == 1 and holes[0].polygon.centroid.distance(shapely.Point(10, 19)) < 1e-6
    before = {g.layer: g.total_area for g in extract_layers(board, plain)}
    after = {g.layer: g for g in extract_layers(board, excluded)}
    for layer, geom in after.items():
        # 9 mm² removed (the via holes inside it no longer need cutting).
        assert geom.total_area == pytest.approx(before[layer] - 9.0, abs=0.5)
        assert not any(
            f.polygon.intersection(shapely.box(4.1, 4.1, 6.9, 6.9)).area > 1e-9 for f in geom.faces
        )
    disabled = MeshSettings(areas=(dataclasses.replace(excluded.areas[0], enabled=False),))
    assert len(meshed_holes(board, disabled)) == 3


def test_excluding_everything_is_an_error(board_v9_path):
    settings = MeshSettings(areas=(_area("X1", "exclude", x=-1, y=-1, w=50, h=50),))
    with pytest.raises(GeometryError, match="cover the whole board"):
        extract_layers(load_board(board_v9_path), settings)


def _edge_lengths(layer, inside):
    p, t = layer.nodes, layer.triangles
    centroids = p[t].mean(axis=1)
    keep = inside(centroids)
    tri = t[keep]
    edges = np.concatenate(
        [p[tri[:, 0]] - p[tri[:, 1]], p[tri[:, 1]] - p[tri[:, 2]], p[tri[:, 2]] - p[tri[:, 0]]]
    )
    return np.hypot(edges[:, 0], edges[:, 1])


def test_refine_area_meshes_finer_inside(board_v9_path):
    board = load_board(board_v9_path)
    coarse = MeshSettings(max_size_mm=2.0, min_size_mm=0.2, corner_size_mm=0)
    area = _area("R1", x=20, y=10, w=8, h=8, size_mm=0.25)  # bare board, away from features
    fine = dataclasses.replace(coarse, areas=(area,))

    def inside(c):
        return (c[:, 0] > 21) & (c[:, 0] < 27) & (c[:, 1] > 11) & (c[:, 1] < 17)

    plain = mesh_board(board, coarse).layers[0]
    refined = mesh_board(board, fine).layers[0]
    assert np.median(_edge_lengths(plain, inside)) > 1.0
    assert np.median(_edge_lengths(refined, inside)) < 0.4
    assert refined.num_triangles > plain.num_triangles


def test_via_pad_edge_is_a_ring_of_nodes(tmp_path):
    from .test_solve import _HEADER, _strip

    # A 0.8 mm via (0.4 mm drill) in the middle of a pour on both layers: its pad edge
    # isn't part of any copper outline, so without a ring the mesh ignores it.
    via = '  (via (at 25 5) (size 0.8) (drill 0.4) (layers "F.Cu" "B.Cu") (net 1))\n'
    path = tmp_path / "via.kicad_pcb"
    path.write_text(_HEADER + _strip("F.Cu") + _strip("B.Cu") + via + ")\n")
    board = load_board(path)
    settings = MeshSettings(max_size_mm=1.0, min_size_mm=0.1, min_circle_points=36, corner_size_mm=0)

    def on_circle(mesh, radius):
        layer = mesh.layers[0]
        distance = np.hypot(*(layer.nodes - (25.0, 5.0)).T)
        return layer.nodes[np.abs(distance - radius) < 1e-3]

    with_rings = mesh_board(board, settings)
    for layer in with_rings.layers:
        assert layer.layer in ("F.Cu", "B.Cu")
    ring = on_circle(with_rings, 0.4)
    assert len(ring) == 36  # a regular 36-gon...
    angles = np.sort(np.degrees(np.arctan2(ring[:, 1] - 5, ring[:, 0] - 25)) % 360)
    np.testing.assert_allclose(angles, np.arange(36) * 10.0, atol=0.05)  # ...from angle 0, like the hole
    assert len(on_circle(with_rings, 0.2)) == 36  # the drill hole edge, concentric

    without = mesh_board(board, dataclasses.replace(settings, via_rings=False))
    assert len(on_circle(without, 0.4)) < 6
    assert with_rings.num_triangles > without.num_triangles

    # The option is saved only when off, and changes the mesh fingerprint.
    assert "via_rings" not in settings.to_dict()
    off = dataclasses.replace(settings, via_rings=False)
    assert MeshSettings.from_dict(off.to_dict()) == off
    assert fingerprint.settings_hash(off) != fingerprint.settings_hash(settings)


def test_saved_mesh_keeps_all_settings(board_v9_path, tmp_path):
    settings = MeshSettings(
        max_size_mm=2.0, corner_size_mm=0.2, tracks="none", areas=(_area("X1", "exclude", x=30, y=20),)
    )
    mesh = mesh_board(load_board(board_v9_path), settings)
    mesh.save_npz(tmp_path / "m.npz")
    assert BoardMesh.load_npz(tmp_path / "m.npz").settings == settings
