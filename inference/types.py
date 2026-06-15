"""Tipos compartidos de salida de modelos de inferencia."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FaceDetections:
    """Detecciones RetinaFace en pixeles del frame original (filas N x 15)."""

    dets: np.ndarray

    @property
    def has_faces(self) -> bool:
        return self.dets.shape[0] > 0

    @classmethod
    def empty(cls) -> FaceDetections:
        return cls(dets=np.zeros((0, 15), dtype=np.float32))
