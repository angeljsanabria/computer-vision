"""Snapshot periodico de calidad de imagen (headless; pisa img_snap.jpg)."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from configs import settings as s
from configs.paths import resolve_repo_path

_SNAP_NAME = "img_snap.jpg"


class ImgQualitySnapSaver:
    """Guarda frame BGR cada N s en un unico archivo (sobrescribe)."""

    def __init__(self) -> None:
        self._interval_s = s.IMG_QUALITY_CHECK_INTERVAL_S
        self._out_path: Path = resolve_repo_path(s.IMG_QUALITY_CHECK_DIR) / _SNAP_NAME
        self._last_save_mono: float | None = None
        self._out_path.parent.mkdir(parents=True, exist_ok=True)
        logging.debug(
            "img_quality_check: activo cada %.1f s -> %s",
            self._interval_s,
            self._out_path,
        )

    def maybe_save(
        self, frame_bgr: np.ndarray, now_mono: float | None = None
    ) -> None:
        now = now_mono if now_mono is not None else time.monotonic()
        if (
            self._last_save_mono is not None
            and (now - self._last_save_mono) < self._interval_s
        ):
            return
        try:
            # Sufijo .jpg para que cv2.imwrite elija codec JPEG (no usar .tmp al final).
            tmp = self._out_path.with_name(
                f"{self._out_path.stem}.tmp{self._out_path.suffix}"
            )
            if not cv2.imwrite(str(tmp), frame_bgr):
                raise RuntimeError("cv2.imwrite devolvio False")
            tmp.replace(self._out_path)
            self._last_save_mono = now
            logging.debug("img_quality_check: guardado %s", self._out_path)
        except Exception as exc:
            logging.warning("img_quality_check: fallo al guardar: %s", exc)
