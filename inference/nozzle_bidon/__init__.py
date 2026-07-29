"""Inferencia detector Bidon/Pico YOLOv8 (nozzle_bidones_v4)."""
from __future__ import annotations

from inference.nozzle_bidon.select_best import mejores_bidones
from inference.nozzle_bidon.types import NozzleBidonDetections

__all__ = [
    "NozzleBidonDetections",
    "mejores_bidones",
]
