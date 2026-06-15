"""Sensor de movimiento MOG2 sobre frame reducido."""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from mov_detect.types import Mog2Config, MotionResult
from utils.image_utils import resize_frame

GetFrameFn = Callable[[], tuple[bool, Any]]


class Mog2MotionSensor:
    """
    OpenCV MOG2 sobre un frame BGR reducido (independiente de la resolucion de captura).

    Calibracion al arranque: ``warmup_from_first_frame`` — un frame de captura
    aplicado N veces al buffer MOG2. Luego el bucle principal usa ``evaluate``
    con frames nuevos del stream con normalidad.
    """

    def __init__(self, config: Mog2Config | None = None) -> None:
        self._cfg = config or Mog2Config()
        self._umbral_base = self._cfg.movimiento_pixeles
        self._umbral_pixeles = self._umbral_base
        self._process_wh = (self._cfg.process_width, self._cfg.process_height)
        self._fgbg = cv2.createBackgroundSubtractorMOG2(
            history=self._cfg.history,
            varThreshold=self._cfg.var_threshold,
            detectShadows=False,
        )

    @property
    def umbral_pixeles(self) -> int:
        return self._umbral_pixeles

    def set_umbral_idle(self) -> None:
        """Umbral base (IDLE): mas estricto para entrar a fase activa."""
        self._umbral_pixeles = self._umbral_base

    def set_umbral_activo(self) -> None:
        """Umbral reducido (fuera de IDLE): histeresis para mantener fase activa."""
        self._umbral_pixeles = self._umbral_base // 3

    def _wait_first_frame(
        self,
        get_frame: GetFrameFn,
        timeout_s: float,
        poll_s: float,
    ) -> np.ndarray | None:
        t_ini = time.monotonic()
        while time.monotonic() - t_ini <= timeout_s:
            ok, frame = get_frame()
            if ok and frame is not None and getattr(frame, "size", 0) > 0:
                return np.asarray(frame).copy()
            time.sleep(poll_s)
        return None

    def warmup_frame(self, frame_bgr: np.ndarray) -> None:
        """Un frame de calibracion con learning rate alto (fondo)."""
        small = resize_frame(frame_bgr, self._process_wh, interpolation=cv2.INTER_AREA)
        self._fgbg.apply(small, learningRate=self._cfg.warmup_learning_rate)

    def warmup_from_first_frame(
        self,
        get_frame: GetFrameFn,
        n_frames: int | None = None,
        timeout_s: float = 120.0,
        poll_s: float = 0.05,
    ) -> int:
        """
        Calibra MOG2 con el primer frame valido de captura, aplicado N veces.

        No requiere N frames nuevos del hilo (~2 FPS). Util al arranque con escena estatica.

        Returns:
            Cantidad de applies al modelo (``n_frames`` si hubo exito).

        Raises:
            RuntimeError: Si no llego ningun frame en ``timeout_s``.
        """
        target = n_frames if n_frames is not None else self._cfg.warmup_frames
        if target < 1:
            return 0

        logging.info(
            "MOG2 warmup: esperando primer frame (timeout %.0f s)...",
            timeout_s,
        )
        first = self._wait_first_frame(get_frame, timeout_s, poll_s)
        if first is None:
            raise RuntimeError(
                "MOG2 warmup: no se recibio ningun frame en {:.0f} s".format(timeout_s)
            )

        h, w = first.shape[:2]
        logging.info(
            "MOG2 warmup: frame %dx%d, aplicando %d veces (learningRate=%.2f)...",
            w,
            h,
            target,
            self._cfg.warmup_learning_rate,
        )
        for i in range(target):
            self.warmup_frame(first)
            if (i + 1) % max(1, target // 5) == 0 or (i + 1) == target:
                logging.info("MOG2 warmup: %d / %d", i + 1, target)

        logging.info("MOG2 warmup listo (%d applies, mismo frame).", target)
        return target

    def evaluate(self, frame_bgr: np.ndarray) -> MotionResult:
        """Evalua movimiento en un frame BGR (resolucion de captura)."""
        small = resize_frame(frame_bgr, self._process_wh, interpolation=cv2.INTER_AREA)
        mask = self._fgbg.apply(small)
        pixel_count = int(cv2.countNonZero(mask))
        hay_mov = pixel_count > self._umbral_pixeles
        return MotionResult(hay_mov=hay_mov, pixel_count=pixel_count)
