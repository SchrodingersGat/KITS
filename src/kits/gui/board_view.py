"""Zoomable, pannable view onto a :class:`BoardScene`.

Mouse: wheel zooms about the cursor; middle- or right-drag pans. Left click is
left free for selection / click-to-edit tools.
"""

from __future__ import annotations

from PySide6.QtCore import QLineF, QPoint, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QMouseEvent, QPainter, QPalette, QPen, QWheelEvent
from PySide6.QtWidgets import QGraphicsView

from . import colors

_ZOOM_STEP = 1.25
_MIN_SCALE = 0.5  # pixels per mm
_MAX_SCALE = 5000.0
_PAN_BUTTONS = (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton)
_CURSOR_STRIP_PX = 2  # half-height of the strip repainted around each crosshair line


class BoardView(QGraphicsView):
    cursor_moved = Signal(float, float)  # board coordinates, mm

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Colour for when no board is open. Deliberately not setBackgroundBrush():
        # a view brush stops QGraphicsView from calling the scene's
        # drawBackground(), which draws the grid.
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Base, colors.BACKGROUND)
        self.setPalette(palette)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.setMouseTracking(True)
        self._pan_origin: QPoint | None = None
        self._fit_rect: QRectF | None = None
        self._cursor_visible = True
        self._cursor_pos: QPoint | None = None  # viewport pixels; None when mouse is outside

    # -- crosshair cursor ----------------------------------------------------

    def set_cursor_visible(self, visible: bool) -> None:
        self._cursor_visible = visible
        self._update_cursor_lines(self._cursor_pos)

    @property
    def cursor_visible(self) -> bool:
        return self._cursor_visible

    def _move_cursor(self, pos: QPoint | None) -> None:
        old, self._cursor_pos = self._cursor_pos, pos
        if old != pos:
            self._update_cursor_lines(old)
            self._update_cursor_lines(pos)

    def _update_cursor_lines(self, pos: QPoint | None) -> None:
        """Repaint only the thin strips under the crosshair lines at ``pos``."""
        if pos is None:
            return
        vp = self.viewport()
        m = _CURSOR_STRIP_PX
        vp.update(QRect(0, pos.y() - m, vp.width(), 2 * m + 1))
        vp.update(QRect(pos.x() - m, 0, 2 * m + 1, vp.height()))

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        super().drawForeground(painter, rect)
        if not self._cursor_visible or self._cursor_pos is None or self.scene() is None:
            return
        center = self.mapToScene(self._cursor_pos)
        painter.save()
        pen = QPen(colors.CURSOR, 1)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.drawLine(QLineF(rect.left(), center.y(), rect.right(), center.y()))
        painter.drawLine(QLineF(center.x(), rect.top(), center.x(), rect.bottom()))
        painter.restore()

    def scrollContentsBy(self, dx: int, dy: int) -> None:
        # Scrolling blits the viewport, carrying the old lines along with the
        # board; clear them where they landed and redraw at the mouse.
        super().scrollContentsBy(dx, dy)
        if self._cursor_pos is not None:
            self._update_cursor_lines(self._cursor_pos + QPoint(dx, dy))
            self._update_cursor_lines(self._cursor_pos)

    def leaveEvent(self, event) -> None:
        self._move_cursor(None)
        super().leaveEvent(event)

    # -- zoom ----------------------------------------------------------------

    def fit_to_rect(self, rect: QRectF) -> None:
        self._fit_rect = rect
        if not rect.isEmpty():
            margin = max(rect.width(), rect.height()) * 0.03
            self.fitInView(
                rect.adjusted(-margin, -margin, margin, margin), Qt.AspectRatioMode.KeepAspectRatio
            )

    def zoom_to_fit(self) -> None:
        if self._fit_rect is not None:
            self.fit_to_rect(self._fit_rect)

    def zoom(self, factor: float) -> None:
        current = self.transform().m11()
        factor = max(_MIN_SCALE / current, min(_MAX_SCALE / current, factor))
        self.scale(factor, factor)

    def zoom_in(self) -> None:
        self._zoom_about_center(_ZOOM_STEP)

    def zoom_out(self) -> None:
        self._zoom_about_center(1 / _ZOOM_STEP)

    def _zoom_about_center(self, factor: float) -> None:
        anchor = self.transformationAnchor()
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.zoom(factor)
        self.setTransformationAnchor(anchor)

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta:
            self.zoom(_ZOOM_STEP ** (delta / 120))
        event.accept()

    # -- pan / cursor tracking ---------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() in _PAN_BUTTONS:
            self._pan_origin = event.position().toPoint()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position().toPoint()
        if self._pan_origin is not None:
            delta = pos - self._pan_origin
            self._pan_origin = pos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
        else:
            super().mouseMoveEvent(event)
        self._move_cursor(pos)
        scene_pos = self.mapToScene(pos)
        self.cursor_moved.emit(scene_pos.x(), scene_pos.y())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() in _PAN_BUTTONS and self._pan_origin is not None:
            self._pan_origin = None
            self.viewport().unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event) -> None:
        event.accept()  # right button is used for panning
