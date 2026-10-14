import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from kits.config import MeshSettings  # noqa: E402
from kits.gui.main_window import MainWindow  # noqa: E402
from kits.gui.settings import AppSettings  # noqa: E402
from kits.project import KitsProject, load_kits_project  # noqa: E402

Buttons = QMessageBox.StandardButton


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def settings(tmp_path):
    return AppSettings(QSettings(str(tmp_path / "app.ini"), QSettings.Format.IniFormat))


@pytest.fixture
def window(app, settings):
    w = MainWindow(settings=settings)
    yield w
    w.document.close()  # don't prompt about unsaved changes on teardown
    w.close()


@pytest.fixture
def dialogs(monkeypatch):
    """Scripted answers for file dialogs and message boxes; records what was shown."""

    class Dialogs:
        save_paths: list = []
        open_paths: list = []
        shown: list = []

    d = Dialogs()
    monkeypatch.setattr(
        "kits.gui.main_window.QFileDialog.getSaveFileName", lambda *a, **k: (str(d.save_paths.pop(0)), "")
    )
    monkeypatch.setattr(
        "kits.gui.main_window.QFileDialog.getOpenFileName", lambda *a, **k: (str(d.open_paths.pop(0)), "")
    )
    for name in ("critical", "warning"):
        monkeypatch.setattr(
            f"kits.gui.main_window.QMessageBox.{name}", lambda *a, n=name: d.shown.append((n, a))
        )
    return d


def test_start_page_until_a_project_is_opened(window, kits_v9_path):
    assert window._pages.currentWidget() is window.start_page
    assert window.windowTitle() == "KITS 0.1.0"
    assert not window.save_action.isEnabled()
    assert window.open_path(kits_v9_path)
    assert window._pages.currentWidget() is window.view
    assert window.save_action.isEnabled() and window.build_mesh_action.isEnabled()
    window.close_project()
    assert window._pages.currentWidget() is window.start_page
    assert window.document.project is None and window.scene is None


def test_new_project_saves_location_then_links_board(window, dialogs, board_v9_path, tmp_path):
    target = tmp_path / "projects" / "mine"  # suffix is added
    target.parent.mkdir()
    dialogs.save_paths.append(target)
    dialogs.open_paths.append(board_v9_path.with_suffix(".kicad_pro"))
    assert window.new_project()

    created = target.with_suffix(".kits")
    assert created.is_file()
    assert load_kits_project(created).board == board_v9_path.with_suffix(".kicad_pro")
    assert window.document.board is not None and not window.document.dirty
    assert window.settings.recent_projects[0] == created


def test_new_project_with_unreadable_board_creates_nothing(window, dialogs, tmp_path):
    bad = tmp_path / "bad.kicad_pcb"
    bad.write_text("(kicad_pcb", encoding="utf-8")
    dialogs.save_paths.append(tmp_path / "p.kits")
    dialogs.open_paths.append(bad)
    assert not window.new_project()
    assert not (tmp_path / "p.kits").exists()
    assert dialogs.shown[0][0] == "critical"


def test_opening_a_kicad_file_offers_to_create_a_project(window, monkeypatch, board_v9_path):
    offers = []

    def accept(board, target):
        offers.append((board, target))
        return target

    monkeypatch.setattr(window, "_ask_create_project", accept)
    assert window.open_path(board_v9_path)
    ((board, target),) = offers
    assert target == board_v9_path.with_suffix(".kits")
    assert target.is_file() and window.document.project.path == target

    # Next time the existing project is offered instead.
    window.close_project()
    monkeypatch.setattr("kits.gui.main_window.QMessageBox.question", lambda *a, **k: Buttons.Open)
    assert window.open_path(board_v9_path)
    assert len(offers) == 1 and window.document.project.path == target


def test_declining_the_project_offer_opens_nothing(window, monkeypatch, board_v9_path):
    monkeypatch.setattr(window, "_ask_create_project", lambda board, target: None)
    assert not window.open_path(board_v9_path)
    assert window.document.project is None
    assert not board_v9_path.with_suffix(".kits").exists()


@pytest.mark.parametrize("answer", [Buttons.Cancel, Buttons.Discard, Buttons.Save])
def test_unsaved_changes_prompt(window, monkeypatch, kits_v9_path, kits_v10_path, answer):
    assert window.open_path(kits_v9_path)
    window.document.set_mesh_settings(MeshSettings(2.0, 0.1))
    asked = []
    monkeypatch.setattr(window, "_ask_save_changes", lambda: asked.append(1) or answer)

    opened = window.open_path(kits_v10_path)
    assert asked
    assert opened == (answer != Buttons.Cancel)
    saved = load_kits_project(kits_v9_path).mesh.max_size_mm
    assert saved == (2.0 if answer == Buttons.Save else 1.0)


def test_close_event_can_be_cancelled(window, monkeypatch, kits_v9_path):
    assert window.open_path(kits_v9_path)
    window.document.set_mesh_settings(MeshSettings(2.0, 0.1))
    monkeypatch.setattr(window, "_ask_save_changes", lambda: Buttons.Cancel)
    window.show()
    assert not window.close()
    assert window.isVisible()


def test_project_with_missing_board_shows_link_page(window, dialogs, tmp_path, board_v9_path):
    project = KitsProject(tmp_path / "p.kits", board=tmp_path / "gone.kicad_pro")
    project.save()
    assert window.open_path(project.path)
    assert window._pages.currentWidget() is window.no_board_page
    assert "not found" in window.no_board_page.message.text()
    assert not window.build_mesh_action.isEnabled()

    dialogs.open_paths.append(board_v9_path)
    assert window.link_board_dialog()
    assert window._pages.currentWidget() is window.view
    assert window.document.dirty
    assert window.save()
    assert load_kits_project(project.path).board == board_v9_path


def test_save_as_moves_the_project(window, dialogs, kits_v9_path, tmp_path):
    assert window.open_path(kits_v9_path)
    (tmp_path / "copy").mkdir()
    dialogs.save_paths.append(tmp_path / "copy" / "renamed.kits")
    assert window.save_as_dialog()
    assert window.document.project.path == tmp_path / "copy" / "renamed.kits"
    assert window.windowTitle().startswith("renamed —")
    assert load_kits_project(tmp_path / "copy" / "renamed.kits").board == kits_v9_path.with_suffix(
        ".kicad_pro"
    )


def test_recent_projects_menu_and_start_page(app, window, kits_v9_path, kits_v10_path, tmp_path):
    missing = tmp_path / "missing.kits"
    window.settings.add_recent_project(missing)
    assert window.open_path(kits_v10_path)
    assert window.open_path(kits_v9_path)
    assert window.settings.recent_projects == [kits_v9_path, kits_v10_path, missing]
    app.processEvents()  # the start page refresh is deferred

    def menu_entries():
        window._populate_recent_menu()  # normally done when the menu opens
        return [a for a in window.recent_menu.actions() if not a.isSeparator()]

    entries = menu_entries()
    assert "demo" in entries[0].text() and entries[0].isEnabled()
    assert not entries[2].isEnabled()  # missing file greyed out
    items = [window.start_page.recent_list.item(i) for i in range(window.start_page.recent_list.count())]
    assert [i.text().splitlines()[0] for i in items] == ["demo", "v10", "missing"]
    assert not items[2].flags() & Qt.ItemFlag.ItemIsEnabled

    # Choosing a recent entry (menu or start page) opens it.
    entries[1].trigger()
    assert window.document.project.path == kits_v10_path
    app.processEvents()
    window.start_page.recent_list.itemActivated.emit(window.start_page.recent_list.item(1))
    assert window.document.project.path == kits_v9_path
    app.processEvents()

    menu_entries()[-1].trigger()  # "Clear Recent Projects"
    app.processEvents()
    assert window.settings.recent_projects == []
    assert window.start_page.empty_label.isVisibleTo(window.start_page)


def test_recent_list_is_capped_and_deduplicated(settings, tmp_path):
    for i in range(12):
        settings.add_recent_project(tmp_path / f"p{i}.kits")
    settings.add_recent_project(tmp_path / "p5.kits")
    recent = settings.recent_projects
    assert len(recent) == 10
    assert recent[0] == tmp_path / "p5.kits"
    assert recent.count(tmp_path / "p5.kits") == 1


def test_build_mesh_saves_the_project_first(window, kits_v9_path, monkeypatch):
    started = []
    monkeypatch.setattr("kits.gui.main_window.MeshJob.start", lambda self: started.append(self))
    assert window.open_path(kits_v9_path)
    window.mesh_panel.max_size.setValue(2.5)
    assert window.document.dirty
    window.build_mesh()
    assert started and not window.document.dirty
    assert load_kits_project(kits_v9_path).mesh.max_size_mm == 2.5
    window._finish_mesh_job()
