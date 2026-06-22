"""Anotaciones de depuracion sobre frames BGR."""
from __future__ import annotations

import cv2
import numpy as np

from inference.types import FaceDetections
from ui.types import FrameView


class DebugOverlay:
    """Dibuja estado FSM, MOG2 y bbox de caras sobre una copia del frame."""

    def __init__(self) -> None:
        self._keep_alive_phase = 0

    def render(self, frame_bgr: np.ndarray, view: FrameView) -> np.ndarray:
        vis = frame_bgr.copy()
        self._draw_keep_alive(vis)
        self._draw_faces(vis, view.dets)
        self._draw_status(vis, view)
        return vis

    def _next_keep_alive(self) -> str:
        dots = "." * (self._keep_alive_phase + 1)
        self._keep_alive_phase = (self._keep_alive_phase + 1) % 3
        return dots

    def _draw_keep_alive(self, vis: np.ndarray) -> None:
        _, w = vis.shape[:2]
        alive = self._next_keep_alive()
        (tw, th), _ = cv2.getTextSize(alive, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(
            vis,
            alive,
            (w - tw - 8, th + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
        )

    def _draw_faces(self, vis: np.ndarray, dets: FaceDetections | None) -> None:
        if dets is None or not dets.has_faces:
            return
        for idx, row in enumerate(dets.dets):
            x1, y1, x2, y2 = map(int, row[:4])
            score = float(row[4])
            color = (0, 255, 0)
            label = f"#{idx + 1} {score:.2f}"
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                vis,
                label,
                (x1, max(y1 - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )

    def _draw_status(self, vis: np.ndarray, view: FrameView) -> None:
        cv2.putText(
            vis,
            f"estado: {view.fsm.state.value}",
            (8, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            vis,
            f"MOG2 px={view.mov.pixel_count} mov={view.mov.hay_mov}",
            (8, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
        )
