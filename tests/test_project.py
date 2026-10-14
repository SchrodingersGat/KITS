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
    assert data["mesh"] == {"max_size_mm": 0.5, "min_size_mm": 0.05}

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
        "board: b.kicad_pro\nambient_temperature_c: 25\nmesh:\n  max_size_mm: 2\n", encoding="utf-8"
    )
    project = load_kits_project(path)
    assert project.mesh == MeshSettings(2.0, 0.1)
    project.mesh = MeshSettings(0.5, 0.05)
    project.save()
    reloaded = load_kits_project(path)
    assert reloaded.mesh == MeshSettings(0.5, 0.05)
    assert reloaded.extra == {"ambient_temperature_c": 25}


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
