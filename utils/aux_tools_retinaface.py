"""
DEPRECADO: usar ``inference.retinaface`` (postprocess, constants, detector_pc / detector_rk3568).

Este modulo re-exporta simbolos por compatibilidad con scripts en ``export_models/``.
No agregar codigo nuevo aqui.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from inference.retinaface.constants import (
    BOX_SCALE as RETINAFACE_BOX_SCALE,
    INPUT_HEIGHT as RETINAFACE_INPUT_HEIGHT,
    INPUT_HW as RETINAFACE_INPUT_HW,
    INPUT_WIDTH as RETINAFACE_INPUT_WIDTH,
    LANDMARK_SCALE as RETINAFACE_LANDMARK_SCALE,
    LETTERBOX_FILL as RETINAFACE_LETTERBOX_FILL,
)
from inference.retinaface.postprocess import (
    box_decode,
    decode_landm,
    dets_desde_salidas_modelo,
    nms,
    prior_box,
    split_outputs,
)


def retinaface_dets_desde_rknn_outputs(
    outputs: list[Any],
    *,
    img_width: int,
    img_height: int,
    aspect_ratio: float,
    offset_x: int,
    offset_y: int,
    score_deteccion: float,
    score_pre_nms: float = 0.02,
    nms_iou: float = 0.5,
    log_priors: bool = False,
) -> np.ndarray:
    """Alias legacy. Preferir ``inference.retinaface.postprocess.dets_desde_salidas_modelo``."""
    return dets_desde_salidas_modelo(
        outputs,
        img_width=img_width,
        img_height=img_height,
        aspect_ratio=aspect_ratio,
        offset_x=offset_x,
        offset_y=offset_y,
        score_deteccion=score_deteccion,
        score_pre_nms=score_pre_nms,
        nms_iou=nms_iou,
        log_priors=log_priors,
    )


__all__ = [
    "RETINAFACE_BOX_SCALE",
    "RETINAFACE_INPUT_HEIGHT",
    "RETINAFACE_INPUT_HW",
    "RETINAFACE_INPUT_WIDTH",
    "RETINAFACE_LANDMARK_SCALE",
    "RETINAFACE_LETTERBOX_FILL",
    "box_decode",
    "decode_landm",
    "nms",
    "prior_box",
    "retinaface_dets_desde_rknn_outputs",
    "split_outputs",
]
