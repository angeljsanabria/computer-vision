"""Tipos compartidos del paquete mov_detect."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FlowState(str, Enum):
    IDLE = "IDLE"
    MOV_DETECTED = "MOV_DETECTED"
    MOV_OUT = "MOV_OUT"
    FACE_PROCESSED = "FACE_PROCESSED"
    FACE_OUT = "FACE_OUT"
    FACE_PROCESSED_TIMEOUT = "FACE_PROCESSED_TIMEOUT"


@dataclass(frozen=True)
class Mog2Config:
    process_width: int = 320
    process_height: int = 240
    history: int = 20
    var_threshold: int = 40
    movimiento_pixeles: int = 1000
    warmup_frames: int = 20
    warmup_learning_rate: float = 0.5


@dataclass(frozen=True)
class FsmConfig:
    timeout_mov_s: float = 10.0
    timeout_face_s: float = 10.0


@dataclass(frozen=True)
class MotionResult:
    hay_mov: bool
    pixel_count: int


@dataclass(frozen=True)
class FsmTickResult:
    state: FlowState
    run_face_detector: bool
    run_embedding: bool
    transitions: tuple[str, ...]
