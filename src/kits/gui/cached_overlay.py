"""A dense vector overlay (mesh edges, coloured result triangles) drawn from cached images.

Drawing tens or hundreds of thousands of small shapes every frame makes panning and
zooming slow. :class:`CachedOverlayItem` instead paints its shapes into a *base image*
once (at most :data:`BASE_MAX_PX` on its long side) and blits that image every frame,
so a frame costs the same however many shapes there are. Zoomed out, the base image
also thins dense mesh edges into a tint that shows the mesh density.

Zoomed in beyond the base image's resolution, the scaled-up (slightly blurred) base
image is shown while the view moves; once it has been still for :data:`SETTLE_MS`,
the visible area is repainted at screen resolution into a *detail image*. Shapes are
grouped in tiles, so that repaint only draws the tiles on screen.

Detail images are painted on one background thread shared by all overlays (painting
into a ``QImage`` is safe off the GUI thread), so the window never waits for them.
Each overlay numbers its requests; a newer request, or the view moving again, makes
the running render stop at its next tile and queued stale ones get skipped, so at
most one render runs at a time and none is wasted on an intermediate zoom.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import shiboken6
from PySide6.QtCore import QObject, QRectF, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsObject

BASE_MAX_PX = 2048  # long side of the base image
DETAIL_MAX_PX = 4096  # long side of a detail image
DETAIL_MARGIN = 0.25  # detail images cover this much extra on each side, for small pans
SETTLE_MS = 60  # quiet time before the visible area is repainted sharply
_BASE_SLACK = 1.2  # use the base image until the view is this much finer than it
_SAME_ZOOM = 0.05  # a detail image (or a render on its way) is reused within this zoom change


@dataclass
class _Group:
    """Shapes of one colour in one tile.

    Attributes:
        bounds: Bounding rectangle of the path, in item coordinates.
        path: The shapes (closed triangles to fill, or line segments to stroke).
        color: Fill or line colour.
    """

    bounds: QRectF
    path: QPainterPath
    color: QColor


@dataclass
class _Image:
    """A rendered image of part of the overlay.

    Attributes:
        rect: The area it covers, in item coordinates.
        pixmap: The image.
        resolution: Pixels per item unit it was rendered at.
        requested: The view resolution it was rendered for (may exceed ``resolution`` if capped).
    """

    rect: QRectF
    pixmap: QPixmap
    resolution: float
    requested: float


def _paint_image(
    groups: list[_Group],
    filled: bool,
    rect: QRectF,
    resolution: float,
    cancelled: Callable[[], bool] | None = None,
) -> QImage | None:
    """Paint the shapes in an area into an image (safe on any thread).

    Args:
        groups: The shape groups.
        filled: Fill the paths (True) or stroke them as one-pixel lines.
        rect: Area to cover, item coordinates.
        resolution: Pixels per item unit.
        cancelled: Checked before each group; painting stops if it returns True.

    Returns:
        The image, or None if cancelled.
    """
    width = max(1, math.ceil(rect.width() * resolution))
    height = max(1, math.ceil(rect.height() * resolution))
    image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    # Filled triangles share edges exactly: without antialiasing there are no seams.
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, not filled)
    painter.scale(width / rect.width(), height / rect.height())
    painter.translate(-rect.left(), -rect.top())
    try:
        for group in groups:
            if cancelled is not None and cancelled():
                return None
            if not group.bounds.intersects(rect):
                continue
            if filled:
                painter.fillPath(group.path, group.color)
            else:
                painter.strokePath(group.path, QPen(group.color, 0))  # cosmetic: one pixel wide
    finally:
        painter.end()
    return image


@dataclass
class _RenderState:
    """Shared between an overlay and its render jobs: the latest request number.

    Attributes:
        item: The overlay the jobs are for.
        generation: Bumped for every new request; jobs with an older number are stale.
    """

    item: CachedOverlayItem
    generation: int = 0


@dataclass
class _Job:
    """A detail image to paint on the render thread.

    Attributes:
        state: The overlay's shared state.
        generation: The request number when the job was made.
        groups: The overlay's shapes (never modified once made, so safe to read).
        filled: Fill or stroke.
        rect: Area to cover, item coordinates.
        resolution: Pixels per item unit to paint at.
        requested: The view resolution it is for.
    """

    state: _RenderState
    generation: int
    groups: list[_Group] = field(repr=False)
    filled: bool
    rect: QRectF
    resolution: float
    requested: float

    @property
    def cancelled(self) -> bool:
        """Whether a newer request (or the view moving again) has made this job stale."""
        return self.generation != self.state.generation


class _Bridge(QObject):
    """Lives on the GUI thread and hands finished images back to their overlays.

    Signals:
        finished(object, object): ``(job, image)`` from the render thread.
    """

    finished = Signal(object, object)

    def __init__(self):
        """Create the bridge; connect it to itself with a queued connection."""
        super().__init__()
        self.finished.connect(self._deliver, Qt.ConnectionType.QueuedConnection)

    @Slot(object, object)
    def _deliver(self, job: _Job, image: QImage) -> None:
        """On the GUI thread: give the overlay its image, unless it is stale or gone.

        Args:
            job: The finished job.
            image: The painted image.
        """
        item = job.state.item
        if job.cancelled or not shiboken6.isValid(item):
            return
        item._accept_detail(job, image)


class _RenderWorker:
    """The one background thread that paints detail images, a job at a time.

    Attributes:
        completed: Jobs painted to the end.
        cancelled: Jobs stopped part-way or skipped because they were stale.
    """

    def __init__(self):
        """Create the (not yet started) worker."""
        self._jobs: queue.SimpleQueue[_Job] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._bridge: _Bridge | None = None
        self._lock = threading.Lock()
        self._outstanding = 0  # jobs queued or being painted
        self.completed = 0
        self.cancelled = 0

    def submit(self, job: _Job) -> None:
        """Queue a job (call on the GUI thread); starts the thread on first use.

        Args:
            job: The job.
        """
        if self._bridge is None:
            self._bridge = _Bridge()  # created here, so it belongs to the GUI thread
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="kits-overlay-render", daemon=True)
            self._thread.start()
        with self._lock:
            self._outstanding += 1
        self._jobs.put(job)

    @property
    def thread(self) -> threading.Thread | None:
        """The render thread, once started."""
        return self._thread

    @property
    def idle(self) -> bool:
        """Whether no job is queued or being painted."""
        with self._lock:
            return self._outstanding == 0

    def _run(self) -> None:
        """Paint jobs as they come, skipping or stopping stale ones."""
        while True:
            job = self._jobs.get()
            image = None
            if not job.cancelled:
                image = _paint_image(
                    job.groups, job.filled, job.rect, job.resolution, lambda job=job: _yield_then(job)
                )
            if image is None:
                self.cancelled += 1
            else:
                self.completed += 1
                self._bridge.finished.emit(job, image)
            with self._lock:
                self._outstanding -= 1


def _yield_then(job: _Job) -> bool:
    """Between tiles on the render thread: let the GUI thread run, then check for cancellation.

    Qt's painting calls keep Python's interpreter lock, and the GUI thread needs
    that lock for every Python callback (item painting, timers). Without an explicit
    hand-over it would wait up to the interpreter's switch interval each time, which
    adds up to the whole render; ``sleep(0)`` hands the lock over at once.

    Args:
        job: The job being painted.

    Returns:
        Whether the job has been cancelled.
    """
    time.sleep(0)
    return job.cancelled


WORKER = _RenderWorker()


class CachedOverlayItem(QGraphicsObject):
    """Filled or stroked shapes drawn from cached images rather than vector by vector."""

    def __init__(self, groups: list[tuple[QPainterPath, QColor]], filled: bool, parent=None):
        """Create the item.

        Args:
            groups: ``(path, colour)`` per tile and colour, in item (scene) coordinates.
            filled: True to fill the paths (result triangles), False to stroke them as
                one-pixel lines (mesh edges).
            parent: Optional parent item.
        """
        super().__init__()  # parented below, once the bounds are known (the scene indexes them)
        self._groups = [
            _Group(path.boundingRect(), path, QColor(color)) for path, color in groups if not path.isEmpty()
        ]
        self._filled = filled
        self._bounds = QRectF()
        for group in self._groups:
            self._bounds = self._bounds.united(group.bounds)
        self._base: _Image | None = None
        self._levels: dict[int, _Image] = {}  # base image halved k times (a mipmap), built on demand
        self._detail: _Image | None = None
        self._pending: _Job | None = None  # the detail render on its way, if any
        self._state = _RenderState(self)
        self._wanted: tuple[QRectF, float] | None = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(SETTLE_MS)
        self._timer.timeout.connect(self._request_detail)
        if parent is not None:
            self.setParentItem(parent)

    # -- QGraphicsItem -----------------------------------------------------------------

    def boundingRect(self) -> QRectF:  # noqa: N802 - Qt override
        """The area covered by the shapes.

        Returns:
            The bounding rectangle in item coordinates.
        """
        return self._bounds

    def paint(self, painter: QPainter, option, widget=None) -> None:
        """Blit the cached image that best matches the current zoom.

        Args:
            painter: Painter in item coordinates.
            option: Style options (gives the level of detail).
            widget: Widget being painted on, if any.
        """
        if self._bounds.isEmpty():
            return
        resolution = option.levelOfDetailFromTransform(painter.worldTransform())
        if self._base is None:
            self._base = self._render(self._bounds, self.base_resolution)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        inverse, ok = painter.worldTransform().inverted()
        device = painter.device()
        visible = inverse.mapRect(QRectF(0, 0, device.width(), device.height())) if ok else self._bounds
        visible = visible.intersected(self._bounds)
        if resolution <= self._base.resolution * _BASE_SLACK or visible.isEmpty():
            self._blit(painter, self._level_for(resolution))
            return
        if self._serves(self._detail, visible, resolution):
            self._blit(painter, self._detail)
            return
        # Zoomed in past the base image: show it (blurred) now, sharpen once the view is still.
        self._blit(painter, self._base)
        if self._serves(self._pending, visible, resolution):
            return  # the right detail image is already being painted
        self._state.generation += 1  # the view moved on: stop any render for the old view
        self._pending = None
        self._wanted = (visible, resolution)
        self._timer.start()

    # -- images ------------------------------------------------------------------------

    @property
    def base_resolution(self) -> float:
        """Pixels per item unit of the base image."""
        return BASE_MAX_PX / max(self._bounds.width(), self._bounds.height(), 1e-9)

    @property
    def detail(self) -> _Image | None:
        """The current detail image, if one has been rendered."""
        return self._detail

    @property
    def pending(self) -> bool:
        """Whether a detail image is being painted for this overlay."""
        return self._pending is not None

    @staticmethod
    def _serves(image, visible: QRectF, resolution: float) -> bool:
        """Whether an image (or a job on its way) covers the view at its zoom.

        Args:
            image: An :class:`_Image` or :class:`_Job`, or None.
            visible: The visible area, item coordinates.
            resolution: The view's pixels per item unit.

        Returns:
            True if it can be shown for this view.
        """
        return (
            image is not None
            and abs(image.requested / resolution - 1) < _SAME_ZOOM
            and image.rect.contains(visible)
        )

    def _level_for(self, resolution: float) -> _Image:
        """The base image, or a halved copy of it, for drawing at a lower resolution.

        Shrinking an image of one-pixel lines a lot in one step aliases into moiré
        stripes; copies averaged down by halves (a mipmap) shrink cleanly.

        Args:
            resolution: The view's pixels per item unit.

        Returns:
            The smallest copy that is still at least as fine as the view.
        """
        k = 0
        while (
            self._base.resolution / 2 ** (k + 1) >= resolution and self._base.pixmap.width() >> (k + 1) >= 2
        ):
            k += 1
        if k == 0:
            return self._base
        if k not in self._levels:
            image = self._base.pixmap.toImage()
            size = image.size() / 2**k
            smaller = image.scaled(
                size, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            self._levels[k] = _Image(
                self._base.rect, QPixmap.fromImage(smaller), self._base.resolution / 2**k, 0.0
            )
        return self._levels[k]

    def _blit(self, painter: QPainter, image: _Image) -> None:
        """Draw a cached image over the area it covers.

        Args:
            painter: Painter in item coordinates.
            image: The image.
        """
        painter.drawPixmap(image.rect, image.pixmap, QRectF(image.pixmap.rect()))

    def _render(self, rect: QRectF, resolution: float) -> _Image:
        """Paint the shapes in an area into an image, here and now (for the base image).

        Args:
            rect: Area to cover, item coordinates.
            resolution: Pixels per item unit.

        Returns:
            The rendered image.
        """
        image = _paint_image(self._groups, self._filled, rect, resolution)
        return _Image(QRectF(rect), QPixmap.fromImage(image), resolution, resolution)

    def _request_detail(self) -> None:
        """Ask the render thread to paint the visible area (plus a margin) at the view's resolution."""
        if self._wanted is None:
            return
        visible, resolution = self._wanted
        dx, dy = visible.width() * DETAIL_MARGIN, visible.height() * DETAIL_MARGIN
        rect = visible.adjusted(-dx, -dy, dx, dy).intersected(self._bounds)
        if rect.isEmpty():
            return
        capped = min(resolution, DETAIL_MAX_PX / max(rect.width(), rect.height()))
        self._state.generation += 1  # supersedes anything still queued or running
        self._pending = _Job(
            self._state, self._state.generation, self._groups, self._filled, QRectF(rect), capped, resolution
        )
        WORKER.submit(self._pending)

    def _accept_detail(self, job: _Job, image: QImage) -> None:
        """On the GUI thread: show a finished detail image.

        Args:
            job: The job it was painted for.
            image: The image.
        """
        self._detail = _Image(job.rect, QPixmap.fromImage(image), job.resolution, job.requested)
        self._pending = None
        self.update()
