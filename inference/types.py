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


@dataclass(frozen=True)
class FaceSelection:
    """Indices en ``FaceDetections.dets`` a procesar (mejor cara primero)."""

    indices: tuple[int, ...]

    @property
    def count(self) -> int:
        return len(self.indices)

    @classmethod
    def empty(cls) -> FaceSelection:
        return cls(indices=())

    def rank_of(self, det_index: int) -> int | None:
        """1-based rank si esta seleccionada; None si no."""
        try:
            return self.indices.index(det_index) + 1
        except ValueError:
            return None

    def rows(self, dets: FaceDetections) -> np.ndarray:
        """Filas (K, 15) en orden de ranking."""
        if not self.indices:
            return np.zeros((0, 15), dtype=np.float32)
        return dets.dets[list(self.indices)]


@dataclass(frozen=True)
class FaceEmbedding:
    """Vector de embedding facial L2-normalizado (MobileFaceNet, tipicamente 128-D)."""

    vector: np.ndarray

    @property
    def dim(self) -> int:
        return int(self.vector.reshape(-1).size)
