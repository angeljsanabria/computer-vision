"""Inferencia detector Bidon/Pico YOLOv8 (nozzle_bidones)."""
from __future__ import annotations

from inference.nozzle_bidon.color_verify import HsvRange, verificar_color_bidones
from inference.nozzle_bidon.select_best import mejores_bidones
from inference.nozzle_bidon.types import NozzleBidonDetections

__all__ = [
    "HsvRange",
    "NozzleBidonDetections",
    "mejores_bidones",
    "verificar_color_bidones",
]
