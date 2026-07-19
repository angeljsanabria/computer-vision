"""Datos de un frame listos para mostrar (sin logica de pipeline)."""
from __future__ import annotations

from dataclasses import dataclass

from bytetrack.types import TrackResult
from inference.identity.types import IdentityMatch
from inference.types import FaceDetections, FaceMeshLandmarks
from mov_detect.types import FsmTickResult, MotionResult


@dataclass(frozen=True)
class FrameView:
    """Snapshot de un frame procesado para overlay / debug."""

    mov: MotionResult
    fsm: FsmTickResult
    dets: FaceDetections | None = None
    identity: IdentityMatch | None = None
    identity_is_stale: bool = False
    tracks: TrackResult | None = None
    identity_track_id: int | None = None
    identity_by_track: dict[int, IdentityMatch] | None = None
    # Experimental FaceMesh 468 (ver ui/overlay._draw_facemesh). Eliminar si no se aprueba el modelo.
    facemesh: FaceMeshLandmarks | None = None
