"""Ventana OpenCV para depuracion / demo del pipeline (opcional)."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ui.types import FrameView
from utils.image_utils import letterbox_bgr

if TYPE_CHECKING:
    import numpy as np

    from ui.banner import DisplayBanner
    from ui.overlay import DebugOverlay
    from ui.overlay_theme import OverlayTheme


class PipelineDisplay:
    """
    Ventana OpenCV: overlay de pipeline + letterbox fullscreen + banner.

    Si ``enabled=False``, no carga cv2 ni overlay (headless / RK3568).
    ``banner`` se inyecta desde main (``DisplayBanner.try_resolve_from_path`` o None).
    ``overlay_theme`` define color MATCH y tipografia (legacy vs CTK).
    ``show_identity_bar`` controla la franja inferior de nombre/ID.
    ``warning_object_bar`` avisa si hay Bidon visible (objeto no autorizado).
    """

    def __init__(
        self,
        *,
        enabled: bool,
        window_name: str = "pipeline_mov",
        forceFullScreen: bool = False,
        window_width: int = 0,
        window_height: int = 0,
        banner: DisplayBanner | None = None,
        overlay_theme: OverlayTheme | None = None,
        show_identity_bar: bool = True,
        warning_object_bar: bool = False,
    ) -> None:
        self._enabled = enabled
        self._window_name = window_name
        self._forceFullScreen = forceFullScreen
        self._banner = banner
        self._overlay: DebugOverlay | None = None
        self._opened = False
        self._window_size: tuple[int, int] | None = None
        if enabled:
            if window_width > 0 and window_height > 0:
                self._window_size = (window_width, window_height)
            elif window_width != 0 or window_height != 0:
                logging.warning(
                    "Display: DISPLAY_WIDTH/DISPLAY_HEIGHT invalidos (%d x %d); "
                    "sin resize de ventana ni letterbox.",
                    window_width,
                    window_height,
                )
        if enabled:
            from ui.overlay import DebugOverlay

            self._overlay = DebugOverlay(
                theme=overlay_theme,
                show_identity_bar=show_identity_bar,
                warning_object_bar=warning_object_bar,
            )

    @classmethod
    def from_settings(
        cls,
        *,
        enabled: bool,
        force_full_screen: bool,
        display_width: int,
        display_height: int,
        window_name: str = "pipeline_mov",
        banner: DisplayBanner | None = None,
        overlay_theme: OverlayTheme | None = None,
        show_identity_bar: bool = True,
        warning_object_bar: bool = False,
    ) -> PipelineDisplay:
        return cls(
            enabled=enabled,
            window_name=window_name,
            forceFullScreen=force_full_screen,
            window_width=display_width,
            window_height=display_height,
            banner=banner,
            overlay_theme=overlay_theme,
            show_identity_bar=show_identity_bar,
            warning_object_bar=warning_object_bar,
        )

    def setup(self) -> None:
        if not self._enabled:
            return
        import cv2

        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)

        if self._window_size is not None:
            width, height = self._window_size
            cv2.resizeWindow(self._window_name, width, height)
            cv2.moveWindow(self._window_name, 0, 0)

        if self._forceFullScreen:
            cv2.setWindowProperty(
                self._window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )

        cv2.waitKey(1)
        self._opened = True
        logging.debug("Display activo (q en ventana para salir).")

    def show(self, frame_bgr: np.ndarray, view: FrameView) -> None:
        if not self._enabled or self._overlay is None:
            return
        import cv2

        vis = self._overlay.render(frame_bgr, view)
        if self._banner is not None:
            self._banner.paste_top(vis)
        if self._window_size is not None:
            vis, _ = letterbox_bgr(vis, self._window_size, fill_value=0)
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
