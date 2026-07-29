"""Tipos de salida del detector Bidon/Pico (YOLOv8)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from inference.nozzle_bidon.constants import CLASS_NAMES


@dataclass(frozen=True)
class NozzleBidonDetections:
    """
    Detecciones en pixeles del frame original.

    Filas N x 6: xyxy + score + class_id (0=Bidon, 1=Pico).
    """

    dets: np.ndarray

    @property
    def has_detections(self) -> bool:
        return self.dets.shape[0] > 0

    @classmethod
    def empty(cls) -> NozzleBidonDetections:
        return cls(dets=np.zeros((0, 6), dtype=np.float32))

    def class_name(self, row_idx: int) -> str:
        if row_idx < 0 or row_idx >= self.dets.shape[0]:
            return "?"
        cls_id = int(self.dets[row_idx, 5])
        if 0 <= cls_id < len(CLASS_NAMES):
            return CLASS_NAMES[cls_id]
        return str(cls_id)
