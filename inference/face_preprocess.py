"""
Preproceso cara 112x112 para MobileFaceNet.

Modos (``FACE_ALIGNMENT_ENABLE`` en settings):
  - false (defecto): siempre crop bbox + resize (export_models/RetinaFace_from_cam_with_id.py).
  - true: hibrido; align si |roll| > umbral, si no crop.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from inference.face_align import MOBILEFACENET_ALIGN_SIZE, align_face_from_det_row
from inference.face_pose import eye_roll_deg_from_det_row, roll_within_frontal_range


@dataclass(frozen=True)
class FacePatch:
    """Parche BGR 112x112 listo para el embedder."""

    bgr: np.ndarray
    used_align: bool
    roll_deg: float


def bbox_crop_with_margin(
    det_row: np.ndarray,
    img_w: int,
    img_h: int,
    margin_frac: float,
) -> tuple[int, int, int, int]:
    """Misma politica que export_models/RetinaFace_from_cam_with_id.py."""
    x1, y1, x2, y2 = det_row[0], det_row[1], det_row[2], det_row[3]
    bw = max(float(x2 - x1), 1.0)
    bh = max(float(y2 - y1), 1.0)
    mx = bw * margin_frac
    my = bh * margin_frac
    nx1 = max(0, int(np.floor(x1 - mx)))
    ny1 = max(0, int(np.floor(y1 - my)))
    nx2 = min(img_w - 1, int(np.ceil(x2 + mx)))
    ny2 = min(img_h - 1, int(np.ceil(y2 + my)))
    if nx2 <= nx1 or ny2 <= ny1:
        return int(x1), int(y1), int(x2), int(y2)
    return nx1, ny1, nx2, ny2


def crop_bbox_to_size(
    frame_bgr: np.ndarray,
    det_row: np.ndarray,
    *,
    margin_frac: float,
    out_size: int = MOBILEFACENET_ALIGN_SIZE,
) -> np.ndarray:
    """Recorte bbox con margen y resize a out_size x out_size."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox_crop_with_margin(det_row, w, h, margin_frac)
    crop = frame_bgr[y1 : y2 + 1, x1 : x2 + 1]
    if crop.size == 0:
        raise ValueError("recorte bbox vacio")
    return cv2.resize(
        crop,
        (out_size, out_size),
        interpolation=cv2.INTER_LINEAR,
    )


def prepare_face_patch(
    frame_bgr: np.ndarray,
    det_row: np.ndarray,
    *,
    alignment_enable: bool,
    max_abs_roll_deg: float,
    crop_margin_frac: float,
    out_size: int = MOBILEFACENET_ALIGN_SIZE,
) -> FacePatch:
    """
    Genera parche BGR para embedding.

    alignment_enable=False: siempre crop (sin warpAffine), modo export.
    alignment_enable=True: crop si frontal (|roll| <= umbral), si no align.
    """
    if not alignment_enable:
        bgr = crop_bbox_to_size(
            frame_bgr,
            det_row,
            margin_frac=crop_margin_frac,
            out_size=out_size,
        )
        return FacePatch(bgr=bgr, used_align=False, roll_deg=0.0)

    roll_deg = eye_roll_deg_from_det_row(det_row)

    if roll_within_frontal_range(roll_deg, max_abs_roll_deg):
        bgr = crop_bbox_to_size(
            frame_bgr,
            det_row,
            margin_frac=crop_margin_frac,
            out_size=out_size,
        )
        return FacePatch(bgr=bgr, used_align=False, roll_deg=roll_deg)

    bgr = align_face_from_det_row(frame_bgr, det_row, image_size=out_size)
    return FacePatch(bgr=bgr, used_align=True, roll_deg=roll_deg)


def prepare_face_patch_from_settings(
    frame_bgr: np.ndarray,
    det_row: np.ndarray,
) -> FacePatch:
    """Atajo: lee FACE_ALIGNMENT_ENABLE, FACE_ROLL_MAX_DEG y FACE_CROP_MARGIN_FRAC."""
    from configs import settings as s

    return prepare_face_patch(
        frame_bgr,
        det_row,
        alignment_enable=s.FACE_ALIGNMENT_ENABLE,
        max_abs_roll_deg=s.FACE_ROLL_MAX_DEG,
        crop_margin_frac=s.FACE_CROP_MARGIN_FRAC,
    )
