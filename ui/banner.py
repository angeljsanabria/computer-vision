"""Banner estatico de marca para la ventana de demo."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np


class DisplayBanner:
    """
    Franja superior opaca (JPG/PNG BGR).

    Se construye una sola vez al arranque. Por frame solo copia un ROI
    (escala cacheada por ancho de canvas; sin deformar aspect ratio).
    """

    def __init__(self, image_bgr: np.ndarray) -> None:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("banner debe ser imagen BGR HxWx3")
        self._source = image_bgr
        self._scaled: np.ndarray | None = None
        self._scaled_width = 0

    @classmethod
    def try_from_path(cls, path: str | None) -> DisplayBanner | None:
        """
        Resuelve el asset una sola vez.

        Returns:
            Instancia si el archivo existe y se lee; ``None`` si path vacio,
            no existe o no se puede decodificar (sin reintentos posteriores).
        """
        if not path:
            return None

        file_path = Path(path)
        if not file_path.is_file():
            logging.info("Display banner desactivado: no existe %s", file_path)
            return None

        image = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            logging.warning(
                "Display banner desactivado: no se pudo leer %s", file_path
            )
            return None

        logging.info(
            "Display banner activo: %s (%dx%d)",
            file_path,
            image.shape[1],
            image.shape[0],
        )
        return cls(image)

    def paste_top(self, frame_bgr: np.ndarray) -> int:
        """
        Escribe el banner en la franja superior del frame (in-place).

        Returns:
            Alto en pixeles ocupado por el banner (0 si no se dibujo).
        """
        height, width = frame_bgr.shape[:2]
        if width <= 0 or height <= 0:
            return 0

        strip = self._scaled_to_width(width)
        strip_h = min(strip.shape[0], height)
        frame_bgr[0:strip_h, 0:width] = strip[0:strip_h]
        return strip_h

    def _scaled_to_width(self, width: int) -> np.ndarray:
        if self._scaled is not None and self._scaled_width == width:
            return self._scaled

        src_h, src_w = self._source.shape[:2]
        if src_w == width:
            scaled = self._source
        else:
            new_h = max(1, int(round(src_h * (width / float(src_w)))))
            scaled = cv2.resize(
                self._source,
                (width, new_h),
                interpolation=cv2.INTER_AREA,
            )

        self._scaled = scaled
        self._scaled_width = width
        return scaled
