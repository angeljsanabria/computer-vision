"""Inferencia detector fuel nozzle YOLOv8."""
from __future__ import annotations

from inference.nozzle.select_best import mejores_nozzles
from inference.nozzle.types import NozzleDetections

__all__ = [
    "NozzleDetections",
    "mejores_nozzles",
]
