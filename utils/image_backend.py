"""
Backend de imagen: OpenCV (PC / fallback) y RGA (RK3568 + USE_RGA).

Main arranca la instancia activa con ``ImageBackend.start(use_rga=...)`` tras
``validar_todo()``. El resto del pipeline usa ``image_utils`` sin pasar flags.
"""
from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

_my_rga_module: Any | None = None
_my_rga_import_failed = False
_rga_fallback_logged = False
_active: ImageBackend | None = None


def _log_rga_fallback_once(reason: str) -> None:
    global _rga_fallback_logged
    if _rga_fallback_logged:
        return
    _rga_fallback_logged = True
    logging.debug("RGA no disponible (%s); usando OpenCV.", reason)


def _try_import_my_rga() -> Any | None:
    global _my_rga_module, _my_rga_import_failed
    if _my_rga_import_failed:
        return None
    if _my_rga_module is not None:
        return _my_rga_module
    try:
        import my_rga as mod

        _my_rga_module = mod
        return mod
    except ImportError as exc:
        _my_rga_import_failed = True
        _log_rga_fallback_once(str(exc))
        return None


def opencv_resize(
    frame: np.ndarray,
    out_wh: tuple[int, int],
    interpolation: int,
) -> np.ndarray:
    return cv2.resize(frame, out_wh, interpolation=interpolation)


def opencv_bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def opencv_letterbox_bgr(
    image_bgr: np.ndarray,
    out_wh: tuple[int, int],
    fill_value: int,
) -> tuple[np.ndarray, float, int, int]:
    target_width, target_height = out_wh[0], out_wh[1]
    image_height, image_width = image_bgr.shape[:2]

    aspect_ratio = min(target_width / image_width, target_height / image_height)
    new_width = int(image_width * aspect_ratio)
    new_height = int(image_height * aspect_ratio)

    resized = cv2.resize(
        image_bgr,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    canvas = (np.ones((target_height, target_width, 3), dtype=np.uint8) * fill_value).astype(
        np.uint8
    )
    offset_x = (target_width - new_width) // 2
    offset_y = (target_height - new_height) // 2
    canvas[offset_y : offset_y + new_height, offset_x : offset_x + new_width] = resized

    return canvas, aspect_ratio, offset_x, offset_y


def rga_resize(
    frame: np.ndarray,
    out_wh: tuple[int, int],
    interpolation: int,
) -> np.ndarray | None:
    mod = _try_import_my_rga()
    if mod is None:
        return None
    try:
        src = np.ascontiguousarray(frame, dtype=np.uint8)
        out, _used = mod.resize_bgr(src, out_wh[0], out_wh[1])
        return np.asarray(out, dtype=np.uint8)
    except Exception as exc:
        _log_rga_fallback_once(str(exc))
        return None


def rga_letterbox_bgr(
    image_bgr: np.ndarray,
    out_wh: tuple[int, int],
    fill_value: int,
) -> tuple[np.ndarray, float, int, int] | None:
    mod = _try_import_my_rga()
    if mod is None:
        return None
    try:
        src = np.ascontiguousarray(image_bgr, dtype=np.uint8)
        canvas, scale, pad_x, pad_y, _used = mod.letterbox_bgr(
            src, out_wh[0], out_wh[1], int(fill_value) & 0xFF
        )
        return np.asarray(canvas, dtype=np.uint8), float(scale), int(pad_x), int(pad_y)
    except Exception as exc:
        _log_rga_fallback_once(str(exc))
        return None


def rga_bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray | None:
    mod = _try_import_my_rga()
    if mod is None:
        return None
    try:
        src = np.ascontiguousarray(frame_bgr, dtype=np.uint8)
        rgb, _used = mod.bgr_to_rgb(src)
        return np.asarray(rgb, dtype=np.uint8)
    except Exception as exc:
        _log_rga_fallback_once(str(exc))
        return None


class ImageBackend:
    """Resize, letterbox y BGR->RGB; RGA opcional segun flag de instancia."""

    __slots__ = ("_use_rga",)

    def __init__(self, *, use_rga: bool = False) -> None:
        self._use_rga = bool(use_rga)

    @property
    def use_rga(self) -> bool:
        return self._use_rga

    @classmethod
    def start(cls, *, use_rga: bool = False) -> ImageBackend:
        """Registra la instancia activa del backend (llamar una vez desde main)."""
        global _active
        backend = cls(use_rga=use_rga)
        _active = backend
        if use_rga:
            logging.debug("ImageBackend: RGA habilitado")
        return backend

    def resize_bgr(
        self,
        frame: np.ndarray,
        out_wh: tuple[int, int],
        interpolation: int = cv2.INTER_AREA,
    ) -> np.ndarray:
        if self._use_rga:
            out = rga_resize(frame, out_wh, interpolation)
            if out is not None:
                return out
        return opencv_resize(frame, out_wh, interpolation)

    def letterbox_bgr(
        self,
        image_bgr: np.ndarray,
        out_wh: tuple[int, int],
        fill_value: int,
    ) -> tuple[np.ndarray, float, int, int]:
        if self._use_rga:
            out = rga_letterbox_bgr(image_bgr, out_wh, fill_value)
            if out is not None:
                return out
        return opencv_letterbox_bgr(image_bgr, out_wh, fill_value)

    def bgr_to_rgb(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self._use_rga:
            out = rga_bgr_to_rgb(frame_bgr)
            if out is not None:
                return out
        return opencv_bgr_to_rgb(frame_bgr)


def active_backend() -> ImageBackend:
    """Instancia activa; OpenCV-only si main no llamo ``start`` (scripts offline)."""
    global _active
    if _active is None:
        _active = ImageBackend(use_rga=False)
    return _active


def resize_bgr(
    frame: np.ndarray,
    out_wh: tuple[int, int],
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    return active_backend().resize_bgr(frame, out_wh, interpolation)


def letterbox_bgr_backend(
    image_bgr: np.ndarray,
    out_wh: tuple[int, int],
    fill_value: int,
) -> tuple[np.ndarray, float, int, int]:
    return active_backend().letterbox_bgr(image_bgr, out_wh, fill_value)


def bgr_to_rgb_backend(frame_bgr: np.ndarray) -> np.ndarray:
    return active_backend().bgr_to_rgb(frame_bgr)
