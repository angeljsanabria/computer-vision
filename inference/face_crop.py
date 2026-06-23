"""Recorte bbox con margen (compartido por crop y roll-fix)."""
from __future__ import annotations

import cv2
import numpy as np

from inference.face_align import MOBILEFACENET_ALIGN_SIZE


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
