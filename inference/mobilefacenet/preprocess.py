"""Tensor de entrada MobileFaceNet desde parche BGR 112x112."""
from __future__ import annotations

import cv2
import numpy as np

from inference.mobilefacenet.constants import IMAGENET_MEAN, IMAGENET_STD, INPUT_HW


def _assert_bgr112(face_bgr: np.ndarray) -> None:
    if face_bgr.ndim != 3 or face_bgr.shape[2] != 3:
        raise ValueError(f"face_bgr debe ser (H, W, 3), got {face_bgr.shape}")
    h, w = face_bgr.shape[:2]
    if (h, w) != INPUT_HW:
        raise ValueError(f"face_bgr debe ser {INPUT_HW}, got ({h}, {w})")


def bgr112_to_onnx_nchw(face_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 112x112 -> float32 NCHW (1, 3, 112, 112). Mismo pipeline que export_models."""
    _assert_bgr112(face_bgr)
    if face_bgr.size == 0:
        raise ValueError("recorte vacio")
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    norm = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.transpose(norm, (2, 0, 1))
    return np.expand_dims(chw.astype(np.float32), axis=0)


def bgr112_to_rknn_nhwc(face_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 112x112 -> RGB uint8 NHWC (1, 112, 112, 3); mean/std en el .rknn."""
    _assert_bgr112(face_bgr)
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    return np.expand_dims(rgb, axis=0)
