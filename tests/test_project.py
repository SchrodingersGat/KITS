import pytest
import yaml

from kits.config import ConfigError, MeshSettings
from kits.project import KitsProject, default_project_path, load_kits_project


def test_round_trip_with_relative_board_path(tmp_path):
    (tmp_path / "boards").mkdir()
    board = tmp_path / "boards" / "b.kicad_pro"
    project = KitsProject(tmp_path / "proj" / "p.kits", board=board, mesh=MeshSettings(0.5, 0.05))
    (tmp_path / "proj").mkdir()
    project.save()

    data = yaml.safe_load(project.path.read_text(encoding="utf-8"))
    assert data["board"] == "../boards/b.kicad_pro"  # relative, POSIX separators
    assert data["mesh"] == {
        "max_size_mm": 0.5,
        "min_size_mm": 0.05,
        "min_circle_points": 36,
        "corner_size_mm": 0.1,
    }

    loaded = load_kits_project(project.path)
    assert loaded.board == board.absolute()
    assert loaded.mesh == MeshSettings(0.5, 0.05)
    assert loaded.name == "p"
    assert loaded.msh_path == tmp_path / "proj" / "kits_output" / "p.msh"


def test_save_as_rewrites_relative_board_path(tmp_path):
    board = tmp_path / "b.kicad_pcb"
    project = KitsProject(tmp_path / "p.kits", board=board)
    project.save()
    (tmp_path / "sub").mkdir()
    project.save(tmp_path / "sub" / "q.kits")
    assert project.path == tmp_path / "sub" / "q.kits"
    assert yaml.safe_load(project.path.read_text(encoding="utf-8"))["board"] == "../b.kicad_pcb"
    assert load_kits_project(project.path).board == board


def test_unknown_keys_are_preserved(tmp_path):
    path = tmp_path / "p.kits"
    path.write_text(
        "board: b.kicad_pro\nheat_sinks: [{name: tab}]\nmesh:\n  max_size_mm: 2\n", encoding="utf-8"
    )
    project = load_kits_project(path)
    assert project.mesh == MeshSettings(2.0, 0.1)
    project.mesh = MeshSettings(0.5, 0.05)
    project.save()
    reloaded = load_kits_project(path)
    assert reloaded.mesh == MeshSettings(0.5, 0.05)
    assert reloaded.extra == {"heat_sinks": [{"name": "tab"}]}


def test_empty_file_gives_defaults(tmp_path):
    path = tmp_path / "p.kits"
    path.write_text("", encoding="utf-8")
    project = load_kits_project(path)
    assert project.board is None
    assert project.mesh == MeshSettings()


@pytest.mark.parametrize(
    "text",
    [
        "mesh:\n  max_size_mm: 0.1\n  min_size_mm: 0.5\n",  # min > max
        "mesh:\n  max_size_mm: -1\n",
        "mesh:\n  max_size_mm: big\n",
        "mesh: 3\n",
        "board: [1, 2]\n",
        "mesh:\n  min_circle_points: 4\n",  # below the allowed range
        "mesh:\n  min_circle_points: 36.5\n",
        "mesh:\n  min_circle_points: true\n",
        "- a list\n",
        "mesh: [unclosed\n",
    ],
)
def test_invalid_files_are_rejected(tmp_path, text):
    path = tmp_path / "p.kits"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_kits_project(path)


def test_missing_file_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="could not read"):
        load_kits_project(tmp_path / "nope.kits")


def test_save_rejects_invalid_settings(tmp_path):
    project = KitsProject(tmp_path / "p.kits", mesh=MeshSettings(0.1, 1.0))
    with pytest.raises(ConfigError):
        project.save()
    assert not project.path.exists()


def test_default_project_path(tmp_path):
    assert default_project_path(tmp_path / "board.kicad_pro") == tmp_path / "board.kits"
    assert default_project_path(tmp_path / "board.kicad_pcb") == tmp_path / "board.kits"


def test_current_sources_round_trip(tmp_path):
    from kits.config import CurrentSource

    path = tmp_path / "p.kits"
    path.write_text(
        "current_sources:\n  - {ref: J1, pad: 1, current_a: 10}\n  - {ref: J2, pad: '1', current_a: -10.5}\n",
        encoding="utf-8",
    )
    project = load_kits_project(path)
    assert project.current_sources == [CurrentSource("J1", "1", 10.0), CurrentSource("J2", "1", -10.5)]
    project.save()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["current_sources"] == [
        {"ref": "J1", "pad": "1", "current_a": 10.0},
        {"ref": "J2", "pad": "1", "current_a": -10.5},
    ]
    assert load_kits_project(path).current_sources == project.current_sources


@pytest.mark.parametrize(
    "text, match",
    [
        ("current_sources: 3\n", "must be a list"),
        ("current_sources:\n  - {ref: J1, pad: 1}\n", "missing 'current_a'"),
        ("current_sources:\n  - {ref: J1, pad: 1, current_a: lots}\n", "invalid current source"),
        ("current_sources:\n  - {ref: J1, pad: 1, current_a: .nan}\n", "finite"),
        ("current_sources:\n  - {ref: '', pad: 1, current_a: 1}\n", "reference"),
        (
            "current_sources:\n  - {ref: J1, pad: 1, current_a: 1}\n  - {ref: J1, pad: '1', current_a: 2}\n",
            "more than one",
        ),
    ],
)
def test_invalid_current_sources_are_rejected(tmp_path, text, match):
    path = tmp_path / "p.kits"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match=match):
        load_kits_project(path)


def test_no_current_sources_key_when_empty(tmp_path):
    project = KitsProject(tmp_path / "p.kits")
    project.save()
    assert "current_sources" not in yaml.safe_load(project.path.read_text(encoding="utf-8"))


# -- materials, convection and ambient ---------------------------------------------------


def test_solve_settings_round_trip(tmp_path):
    from kits.config import SolveSettings

    path = tmp_path / "p.kits"
    path.write_text(
        "ambient_temperature_c: 40\n"
        "materials:\n  copper: {conductivity_s_m: 5.0e7, plating_thickness_mm: 0.02}\n"
        "  dielectric: {thermal_conductivity_w_mk: 0.5}\n"
        "convection: {top_w_m2k: 25, bottom_w_m2k: 0}\n",
        encoding="utf-8",
    )
    project = load_kits_project(path)
    assert project.solve == SolveSettings(
        copper_conductivity_s_m=5.0e7,
        copper_thermal_conductivity_w_mk=390.0,  # default kept
        dielectric_thermal_conductivity_w_mk=0.5,
        plating_thickness_mm=0.02,
        convection_top_w_m2k=25.0,
        convection_bottom_w_m2k=0.0,
        ambient_c=40.0,
    )
    project.save()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["ambient_temperature_c"] == 40.0
    assert data["materials"]["copper"]["thermal_conductivity_w_mk"] == 390.0  # always written
    assert data["convection"] == {"top_w_m2k": 25.0, "bottom_w_m2k": 0.0}
    assert load_kits_project(path).solve == project.solve


def test_solve_settings_default_when_absent(tmp_path):
    from kits.config import SolveSettings

    path = tmp_path / "p.kits"
    path.write_text("board: b.kicad_pro\n", encoding="utf-8")
    assert load_kits_project(path).solve == SolveSettings()


@pytest.mark.parametrize(
    "text, match",
    [
        ("convection: {top_w_m2k: 0, bottom_w_m2k: 0}\n", "nowhere to go"),
        ("convection: {top_w_m2k: -1}\n", "negative"),
        ("materials: {copper: {conductivity_s_m: 0}}\n", "copper conductivity must be positive"),
        ("materials: {dielectric: {thermal_conductivity_w_mk: -0.3}}\n", "dielectric"),
        ("materials: {copper: {plating_thickness_mm: lots}}\n", "invalid material"),
        ("materials: [1, 2]\n", "'materials' must be a mapping"),
        ("materials: {copper: 5}\n", "'materials.copper' must be a mapping"),
        ("ambient_temperature_c: -300\n", "absolute zero"),
        ("ambient_temperature_c: .inf\n", "finite"),
    ],
)
def test_invalid_solve_settings_are_rejected(tmp_path, text, match):
    path = tmp_path / "p.kits"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match=match):
        load_kits_project(path)


@pytest.mark.parametrize("value, ok", [(0, True), (0.05, True), (1.0, True), (-0.1, False), (2.0, False)])
def test_corner_size_validation(tmp_path, value, ok):
    path = tmp_path / "p.kits"
    path.write_text(f"mesh:\n  max_size_mm: 1.0\n  corner_size_mm: {value}\n", encoding="utf-8")
    if ok:
        assert load_kits_project(path).mesh.corner_size_mm == value
    else:
        with pytest.raises(ConfigError, match="corner size"):
            load_kits_project(path)
