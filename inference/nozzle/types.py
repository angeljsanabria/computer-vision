"""Tipos de salida del detector nozzle YOLOv8."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
