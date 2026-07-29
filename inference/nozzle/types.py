"""Tipos de salida del detector nozzle YOLOv8."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from inference.nozzle.constants import CLASS_NAME


@dataclass(frozen=True)
class NozzleDetections:
    """Detecciones nozzle en pixeles del frame original (filas N x 5: xyxy + score)."""

    dets: np.ndarray

    @property
    def has_detections(self) -> bool:
        return self.dets.shape[0] > 0

    @classmethod
    def empty(cls) -> NozzleDetections:
        return cls(dets=np.zeros((0, 5), dtype=np.float32))

    def class_name(self, row_idx: int) -> str:
        """Nombre de clase de la fila (1 clase fija)."""
        if row_idx < 0 or row_idx >= self.dets.shape[0]:
            return "?"
        return CLASS_NAME
