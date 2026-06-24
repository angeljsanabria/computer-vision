"""Deteccion de movimiento MOG2 y FSM para pipeline facial edge."""
from mov_detect.fsm import MotionFaceFsm
from mov_detect.sensor_mog2 import Mog2MotionSensor
from mov_detect.types import (
    FlowState,
    FsmConfig,
    FsmTickResult,
    Mog2Config,
    MotionResult,
)

__all__ = [
    "FlowState",
    "FsmConfig",
    "FsmTickResult",
    "Mog2Config",
    "Mog2MotionSensor",
    "MotionFaceFsm",
    "MotionResult",
    "config_from_settings",
]


def config_from_settings() -> tuple[Mog2Config, FsmConfig]:
    """Construye configuracion MOG2/FSM desde ``configs.settings``."""
    from configs import settings as s

    mog2 = Mog2Config(
        process_width=s.MOG2_PROCESS_WIDTH,
        process_height=s.MOG2_PROCESS_HEIGHT,
        history=s.MOG2_HISTORY,
        var_threshold=s.MOG2_VAR_THRESHOLD,
        movimiento_pixeles=s.MOG2_MOVIMIENTO_PIXELES,
        warmup_frames=s.MOG2_WARMUP_FRAMES,
        warmup_learning_rate=s.MOG2_WARMUP_LEARNING_RATE,
    )
    fsm = FsmConfig(
        timeout_mov_s=s.FSM_TIMEOUT_MOV_S,
        timeout_face_s=s.FSM_TIMEOUT_FACE_S,
        recognized_refresh_s=s.FSM_RECOGNIZED_REFRESH_S,
    )
    return mog2, fsm
