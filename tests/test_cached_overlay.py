"""CachedOverlayItem: dense overlays drawn from a base image, a mipmap and settled detail images."""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QRectF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QTransform  # noqa: E402
from PySide6.QtWidgets import QApplication, QGraphicsScene, QGraphicsView  # noqa: E402

from kits.gui import cached_overlay  # noqa: E402
from kits.gui.cached_overlay import BASE_MAX_PX, CachedOverlayItem  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _triangles(n: int = 20, size: float = 10.0) -> list:
    """An n x n grid of filled triangles over a size x size square, in two colours."""
    step = size / n
    red, blue = QPainterPath(), QPainterPath()
    for i in range(n):
        for j in range(n):
            x, y = i * step, j * step
            path = red if (i + j) % 2 else blue
            path.moveTo(x, y)
            path.lineTo(x + step, y)
            path.lineTo(x, y + step)
            path.closeSubpath()
    return [(red, QColor(255, 0, 0)), (blue, QColor(0, 0, 255))]


def _render(view: QGraphicsView) -> QImage:
    """What the view shows, painted the way a frame is."""
    image = QImage(view.viewport().size(), QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.black)
    painter = QPainter(image)
    view.render(painter)
    painter.end()
    return image


def _settle(app, item, timeout: float = 5.0) -> None:
    """Fire the settle timer now and wait for the render thread to deliver the detail image."""

    item._timer.stop()
    item._request_detail()
    _wait_idle(app, item, timeout)


def _wait_idle(app, item, timeout: float = 5.0) -> None:
    """Wait for the render thread to finish and deliver."""
    import time

    deadline = time.monotonic() + timeout
    while (item.pending or not cached_overlay.WORKER.idle) and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()


def _view(scene: QGraphicsScene, scale: float, center=(5.0, 5.0)) -> QGraphicsView:
    view = QGraphicsView(scene)
    view.resize(400, 400)
    view.setTransform(QTransform.fromScale(scale, scale))
    view.centerOn(*center)
    return view


def test_overlay_is_indexed_and_painted(app):
    # Regression: the item must be in the scene's index with its real bounds, or it is never drawn.
    scene = QGraphicsScene()
    container = scene.addRect(QRectF())  # any parent item
    item = CachedOverlayItem(_triangles(), filled=True, parent=container)
    assert item.boundingRect() == QRectF(0, 0, 10, 10)
    assert scene.items(QRectF(4, 4, 2, 2)), "not found in the scene index"
    image = _render(_view(scene, 30.0))
    colors = {QColor(image.pixel(x, y)).name() for x in range(0, 400, 7) for y in range(0, 400, 7)}
    assert "#ff0000" in colors and "#0000ff" in colors


def test_mipmap_and_detail_image(app, monkeypatch):
    scene = QGraphicsScene()
    item = CachedOverlayItem(_triangles(), filled=True)
    scene.addItem(item)
    base_resolution = BASE_MAX_PX / 10.0  # px per unit for a 10-unit-wide overlay
    assert item.base_resolution == pytest.approx(base_resolution)

    # Zoomed out: drawn from a halved copy of the base image, no detail image.
    _render(_view(scene, 20.0))
    assert item._base is not None and item._levels and item.detail is None

    # Zoomed in past the base image: the base is shown, and the visible area is
    # repainted sharply once the view settles (the timer, fired here directly).
    zoom = base_resolution * 4
    view = _view(scene, zoom)
    _render(view)
    assert item._timer.isActive() and item.detail is None
    _settle(app, item)
    detail = item.detail
    assert detail is not None and detail.requested == pytest.approx(zoom)
    visible = view.mapToScene(view.viewport().rect()).boundingRect()
    assert detail.rect.contains(visible.intersected(item.boundingRect()))
    # Painting again at the same zoom uses the detail image and asks for nothing new.
    _render(view)
    assert not item._timer.isActive()


def test_detail_resolution_is_capped(app, monkeypatch):
    monkeypatch.setattr(cached_overlay, "DETAIL_MAX_PX", 256)
    scene = QGraphicsScene()
    item = CachedOverlayItem(_triangles(), filled=True)
    scene.addItem(item)
    view = _view(scene, BASE_MAX_PX)  # absurdly far in
    _render(view)
    _settle(app, item)
    assert max(item.detail.pixmap.width(), item.detail.pixmap.height()) <= 257
    # Requested resolution is remembered, so it isn't re-rendered on every paint.
    _render(view)
    assert not item._timer.isActive()


def test_lines_are_stroked(app):
    path = QPainterPath()
    path.moveTo(0, 0)
    path.lineTo(10, 10)
    scene = QGraphicsScene()
    scene.addItem(CachedOverlayItem([(path, QColor(0, 255, 0))], filled=False))

    def brightest(scale):
        image = _render(_view(scene, scale))
        return max(QColor(image.pixel(x, y)).green() for x in range(0, 400, 2) for y in range(400))

    # At the base image's resolution the one-pixel (antialiased) line is drawn at full strength...
    crisp = brightest(BASE_MAX_PX / 10.0)
    assert crisp >= 100
    # ...and zoomed far out it is averaged into a faint tint (dense meshes thin out).
    assert 0 < brightest(10.0) < crisp / 4


def test_detail_images_are_painted_on_one_background_thread(app, monkeypatch):
    import threading

    painted_on = []
    original = cached_overlay._paint_image

    def spy(*args, **kwargs):
        painted_on.append(threading.current_thread())
        return original(*args, **kwargs)

    monkeypatch.setattr(cached_overlay, "_paint_image", spy)
    scene = QGraphicsScene()
    items = [CachedOverlayItem(_triangles(), filled=True), CachedOverlayItem(_triangles(), filled=True)]
    for item in items:
        scene.addItem(item)
    view = _view(scene, BASE_MAX_PX / 10.0 * 4)
    _render(view)  # base images: painted here, on the GUI thread
    assert all(t is threading.main_thread() for t in painted_on)
    painted_on.clear()
    for item in items:
        _settle(app, item)
        assert item.detail is not None
    worker = cached_overlay.WORKER.thread
    assert painted_on and all(t is worker for t in painted_on) and worker is not threading.main_thread()
    assert sum(t.name == "kits-overlay-render" for t in threading.enumerate()) == 1


def test_superseded_renders_are_cancelled(app, monkeypatch):
    import threading

    gate = threading.Event()
    calls = []
    original = cached_overlay._paint_image

    def held(groups, filled, rect, resolution, cancelled=None):
        calls.append(resolution)
        if len(calls) == 1:
            gate.wait(5)  # hold the first detail render while the user "zooms on"
        return original(groups, filled, rect, resolution, cancelled)

    scene = QGraphicsScene()
    item = CachedOverlayItem(_triangles(), filled=True)
    scene.addItem(item)
    zoom = BASE_MAX_PX / 10.0 * 3
    _render(_view(scene, zoom))  # base image, before the hold is installed
    monkeypatch.setattr(cached_overlay, "_paint_image", held)
    worker = cached_overlay.WORKER
    before = worker.cancelled

    item._timer.stop()
    item._request_detail()  # zoom A: starts painting, held
    first = item._pending
    _render(_view(scene, zoom * 1.5))  # the view moves on: zoom A's render is now stale
    assert first.cancelled and not item.pending
    item._timer.stop()
    item._request_detail()  # zoom B: queued behind the held render
    gate.set()
    _wait_idle(app, item)
    assert item.detail is not None and item.detail.requested == pytest.approx(zoom * 1.5)
    assert worker.cancelled > before  # zoom A was abandoned, not delivered


def test_moving_while_a_render_is_on_its_way_keeps_it_if_it_still_fits(app):
    scene = QGraphicsScene()
    item = CachedOverlayItem(_triangles(), filled=True)
    scene.addItem(item)
    view = _view(scene, BASE_MAX_PX / 10.0 * 3)
    _render(view)
    item._timer.stop()
    item._request_detail()
    pending = item._pending
    _render(view)  # same view: the job on its way is still the right one
    assert item._pending is pending and not pending.cancelled
    _settle(app, item)
