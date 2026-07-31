"""Preproceso de frame BGR para inferencia nozzle_bidon."""
from __future__ import annotations

import cv2
import numpy as np

from utils.image_utils import LetterboxMeta, bgr_to_rgb, letterbox_bgr, resize_frame


def stretch_bgr_to_rknn_input(frame_bgr: np.ndarray, input_size: int) -> np.ndarray:
    """Resize cuadrado + RGB uint8 NHWC (mean 0 / std 255 en export RKNN)."""
    img = resize_frame(
        frame_bgr,
        (input_size, input_size),
        interpolation=cv2.INTER_LINEAR,
    )
    rgb = bgr_to_rgb(img)
    return np.expand_dims(rgb, axis=0)


def letterbox_bgr_for_onnx(
    frame_bgr: np.ndarray,
    fill_value: int,
    input_size: int,
) -> tuple[np.ndarray, LetterboxMeta]:
    """Letterbox BGR para ONNX Ultralytics (input_size x input_size)."""
    return letterbox_bgr(frame_bgr, (input_size, input_size), fill_value)


def letterbox_to_nchw_float01(canvas_bgr: np.ndarray) -> np.ndarray:
    """Canvas BGR letterbox -> tensor NCHW float32 [0, 1] RGB."""
    rgb = bgr_to_rgb(canvas_bgr)
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return np.expand_dims(chw, axis=0)


def scale_boxes_letterbox(
    xyxy: np.ndarray,
    meta: LetterboxMeta,
    orig_w: int,
    orig_h: int,
) -> np.ndarray:
    """Mapea cajas del lienzo letterbox al frame original."""
    if xyxy.size == 0:
        return xyxy
    ar = meta.aspect_ratio
    ox, oy = meta.offset_x, meta.offset_y
    out = xyxy.astype(np.float32, copy=True)
    out[:, [0, 2]] = (out[:, [0, 2]] - ox) / ar
    out[:, [1, 3]] = (out[:, [1, 3]] - oy) / ar
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, orig_w)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, orig_h)
    return out
