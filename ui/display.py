"""Ventana OpenCV para depuracion del pipeline (opcional)."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ui.types import FrameView

if TYPE_CHECKING:
    import numpy as np

    from ui.overlay import DebugOverlay


class PipelineDisplay:
    """
    UI de depuracion: overlay + ventana OpenCV.

    Si ``enabled=False``, no carga cv2 ni overlay (headless / RK3568).
    """

    def __init__(
        self,
        *,
        enabled: bool,
        window_name: str = "pipeline_mov",
    ) -> None:
        self._enabled = enabled
        self._window_name = window_name
        self._overlay: DebugOverlay | None = None
        self._opened = False
        if enabled:
            from ui.overlay import DebugOverlay

            self._overlay = DebugOverlay()

    @classmethod
    def from_settings(cls) -> PipelineDisplay:
        from configs import settings as s

        return cls(enabled=s.DISPLAY_IS_ENABLE)

    def setup(self) -> None:
        if not self._enabled:
            return
        import cv2

        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        self._opened = True
        logging.info("Display activo (q en ventana para salir).")

    def show(self, frame_bgr: np.ndarray, view: FrameView) -> None:
        if not self._enabled or self._overlay is None:
            return
        import cv2

        vis = self._overlay.render(frame_bgr, view)
        cv2.imshow(self._window_name, vis)

    def poll_quit(self) -> bool:
        if not self._enabled:
            return False
        import cv2

        return cv2.waitKey(1) & 0xFF == ord("q")

    def teardown(self) -> None:
        if not self._enabled or not self._opened:
            return
        import cv2

        cv2.destroyAllWindows()
        self._opened = False
