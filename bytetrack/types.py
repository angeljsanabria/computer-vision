"""Contratos publicos del paquete bytetrack (Fase 0: sin logica de tracking)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from inference.types import FaceDetections


@dataclass(frozen=True)
class ByteTrackConfig:
    """Parametros del tracker (equivalente limpio a args YOLOX)."""

    track_thresh: float = 0.8
    match_thresh: float = 0.8
    track_buffer: int = 30
    frame_rate: float = 3.0


@dataclass(frozen=True)
class FaceTrack:
    """Un track activo listo para overlay (coords en pixeles del frame)."""

    track_id: int
    tlbr: np.ndarray  # shape (4,) float: x1, y1, x2, y2
    score: float
    # UI only: nozzle usa sticky show_bbox; caras dejan True (siempre dibujar).
    show_bbox: bool = True
    # UI only: nozzle cachea class_id (0=Bidon, 1=Pico); caras = None.
    class_id: int | None = None


@dataclass(frozen=True)
class TrackResult:
    """Salida de un update; inmutable para FrameView."""

    tracks: tuple[FaceTrack, ...]

    @classmethod
    def empty(cls) -> TrackResult:
        return cls(tracks=())


class FaceTracker(Protocol):
    """
    Frontera usada por main.py.

    Lee FaceDetections sin mutarla. Devuelve tracks solo para UI.
    """

    def update(self, dets: FaceDetections | None) -> TrackResult: ...
