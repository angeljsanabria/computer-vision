"""Tensor de entrada FaceMesh desde parche BGR recortado."""
from __future__ import annotations

import cv2
import numpy as np

from inference.facemesh.constants import INPUT_HW, INPUT_SIZE


def _assert_bgr192(face_bgr: np.ndarray) -> None:
    if face_bgr.ndim != 3 or face_bgr.shape[2] != 3:
        raise ValueError(f"face_bgr debe ser (H, W, 3), got {face_bgr.shape}")
    h, w = face_bgr.shape[:2]
    if (h, w) != INPUT_HW:
        raise ValueError(f"face_bgr debe ser {INPUT_HW}, got ({h}, {w})")


def crop_to_bgr192(face_crop_bgr: np.ndarray) -> np.ndarray:
    """Recorte arbitrario -> BGR uint8 192x192 (resize lineal)."""
    if face_crop_bgr.size == 0:
        raise ValueError("recorte vacio")
    return cv2.resize(
        face_crop_bgr,
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )


def bgr192_to_onnx_nchw(face_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 192x192 -> RGB float32 NCHW (1, 3, 192, 192) en [0, 1]."""
    _assert_bgr192(face_bgr)
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))
    return np.expand_dims(chw.astype(np.float32), axis=0)


def bgr192_to_rknn_nhwc(face_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 192x192 -> RGB uint8 NHWC (1, 192, 192, 3); pendiente mean/std en .rknn."""
    _assert_bgr192(face_bgr)
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    return np.expand_dims(rgb, axis=0)
