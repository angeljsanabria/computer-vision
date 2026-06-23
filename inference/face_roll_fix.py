"""
Correccion de roll sobre recorte bbox (mas barato que align ArcFace 5 puntos).

Estilo imutils FaceAligner simplificado: solo endereza la linea entre ojos
en el crop, sin mapear a plantilla ArcFace.
"""
from __future__ import annotations

import cv2
import numpy as np

from inference.face_align import MOBILEFACENET_ALIGN_SIZE, landmarks_from_det_row
from inference.face_pose import eye_roll_deg_from_landmarks
from inference.face_crop import bbox_crop_with_margin


def roll_fix_bbox_crop(
    crop_bgr: np.ndarray,
    lmk_in_crop: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Rota el recorte para horizontalizar la linea entre ojos.

    Returns:
        parche rotado (mismo canvas que crop), roll en grados antes de corregir.
    """
    if lmk_in_crop.shape != (5, 2):
        raise ValueError(f"lmk_in_crop debe ser (5, 2), got {lmk_in_crop.shape}")

    roll_deg = eye_roll_deg_from_landmarks(lmk_in_crop.astype(np.float32))

    if abs(roll_deg) < 0.05:
        return crop_bgr, roll_deg

    left = lmk_in_crop[0].astype(np.float32)
    right = lmk_in_crop[1].astype(np.float32)
    eyes_center = ((left + right) * 0.5).astype(np.float32)
    h, w = crop_bgr.shape[:2]
    center = (float(eyes_center[0]), float(eyes_center[1]))
    M = cv2.getRotationMatrix2D(center, roll_deg, 1.0)
    rotated = cv2.warpAffine(
        crop_bgr,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, roll_deg


def crop_roll_fix_to_size(
    frame_bgr: np.ndarray,
    det_row: np.ndarray,
    *,
    margin_frac: float,
    out_size: int = MOBILEFACENET_ALIGN_SIZE,
) -> tuple[np.ndarray, float]:
    """Recorte bbox + roll-fix + resize a out_size x out_size."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox_crop_with_margin(det_row, w, h, margin_frac)
    crop = frame_bgr[y1 : y2 + 1, x1 : x2 + 1]
    if crop.size == 0:
        raise ValueError("recorte bbox vacio")

    lmk = landmarks_from_det_row(det_row)
    lmk_crop = lmk - np.array([x1, y1], dtype=np.float32)
    fixed, roll_deg = roll_fix_bbox_crop(crop, lmk_crop)
    out = cv2.resize(fixed, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return out, roll_deg
