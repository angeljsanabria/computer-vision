"""Banner estatico de marca para la ventana de demo."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

_WIDTH_VARIANT_SUFFIXES = (".png", ".jpg", ".jpeg")


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

    @staticmethod
    def _read_bgr(file_path: Path) -> np.ndarray | None:
        image = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            return None
        return image

    @classmethod
    def try_from_path(cls, path: str | None) -> DisplayBanner | None:
        """
        Resuelve el asset una sola vez (ruta fija).

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

        image = cls._read_bgr(file_path)
        if image is None:
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

    @classmethod
    def try_resolve_from_path(
        cls,
        path: str | None,
        display_width: int = 0,
    ) -> DisplayBanner | None:
        """
        Resuelve banner por ancho de display o fallback al path base.

        ``path`` apunta al asset por defecto (ej. ``../data/baner_test.jpg``).
        Si ``display_width > 0``, busca antes ``{stem}_{width}.png|.jpg|.jpeg``
        en la misma carpeta; si no hay variante usable, usa el archivo de
        ``path``. Mismo contrato que ``try_from_path`` si nada sirve: ``None``
        y pipeline sin banner.
        """
        if not path:
            return None

        base_path = Path(path)
        parent = base_path.parent
        stem = base_path.stem

        candidates: list[Path] = []
        if display_width > 0:
            for ext in _WIDTH_VARIANT_SUFFIXES:
                candidates.append(parent / f"{stem}_{display_width}{ext}")
        if base_path not in candidates:
            candidates.append(base_path)

        for candidate in candidates:
            if not candidate.is_file():
                continue
            image = cls._read_bgr(candidate)
            if image is None:
                logging.warning(
                    "Display banner desactivado: no se pudo leer %s", candidate
                )
                continue
            logging.info(
                "Display banner activo: %s (%dx%d)",
                candidate,
                image.shape[1],
                image.shape[0],
            )
            if display_width > 0 and candidate == base_path:
                logging.info(
                    "Display banner: sin variante %s_%d.*; usando %s",
                    stem,
                    display_width,
                    base_path.name,
                )
            return cls(image)

        if not base_path.is_file():
            logging.info("Display banner desactivado: no existe %s", base_path)
        return None

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
