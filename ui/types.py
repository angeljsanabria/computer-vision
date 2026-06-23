"""Datos de un frame listos para mostrar (sin logica de pipeline)."""
from __future__ import annotations

from dataclasses import dataclass

from inference.identity.types import IdentityMatch
from inference.types import FaceDetections
from mov_detect.types import FsmTickResult, MotionResult


@dataclass(frozen=True)
class FrameView:
    """Snapshot de un frame procesado para overlay / debug."""

    mov: MotionResult
    fsm: FsmTickResult
    dets: FaceDetections | None = None
    identity: IdentityMatch | None = None
