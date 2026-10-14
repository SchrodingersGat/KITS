"""Zoomable, pannable view onto a :class:`BoardScene`.

Mouse: wheel zooms about the cursor; middle- or right-drag pans. Left click is
left free for selection / click-to-edit tools.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRectF, Qt, Signal
from PySide6.QtGui import QMouseEvent, QPainter, QWheelEvent
from PySide6.QtWidgets import QGraphicsView

from . import colors

_ZOOM_STEP = 1.25
_MIN_SCALE = 0.5  # pixels per mm
_MAX_SCALE = 5000.0
_PAN_BUTTONS = (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton)


class BoardView(QGraphicsView):
    cursor_moved = Signal(float, float)  # board coordinates, mm

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setBackgroundBrush(colors.BACKGROUND)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.setMouseTracking(True)
        self._pan_origin: QPoint | None = None
        self._fit_rect: QRectF | None = None

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
