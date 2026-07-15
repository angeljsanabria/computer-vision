"""Snapshot periodico de calidad de imagen (headless; pisa img_snap.jpg)."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

_NS_PER_S = 1_000_000_000
_SNAP_NAME = "img_snap.jpg"


class ImgQualitySnapSaver:
    """Guarda frame BGR cada N s en un unico archivo (sobrescribe)."""

    def __init__(self, interval_s: float, out_dir: str) -> None:
        self._interval_s = float(interval_s)
        self._interval_ns = int(self._interval_s * _NS_PER_S)
        self._out_path: Path = Path(out_dir) / _SNAP_NAME
        self._last_save_ns: int | None = None
        self._out_path.parent.mkdir(parents=True, exist_ok=True)
        logging.debug(
            "img_quality_check: activo cada %.1f s -> %s",
            self._interval_s,
            self._out_path,
        )

    def maybe_save(
        self, frame_bgr: np.ndarray, now_ns: int | None = None
    ) -> None:
        now = now_ns if now_ns is not None else time.monotonic_ns()
        if (
            self._last_save_ns is not None
            and (now - self._last_save_ns) < self._interval_ns
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
            self._last_save_ns = now
            logging.debug("img_quality_check: guardado %s", self._out_path)
        except Exception as exc:
            logging.warning("img_quality_check: fallo al guardar: %s", exc)
