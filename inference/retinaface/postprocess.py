"""
Postproceso RetinaFace mobile 320 (priors, decode, NMS).

Valido solo para Pytorch_Retinaface / rknn_model_zoo. No usar con YOLO u otros detectores.
"""
from __future__ import annotations

from itertools import product
from math import ceil
from typing import Any

import numpy as np

from inference.retinaface.constants import (
    BOX_SCALE,
    INPUT_HW,
    LANDMARK_SCALE,
    NMS_IOU,
)


def prior_box(image_size: tuple[int, int], *, log_priors: bool = False) -> np.ndarray:
    """Genera priors (anclas) en coordenadas normalizadas. Forma (N, 4) [cx, cy, w, h]."""
    anchors: list[float] = []
    min_sizes = [[16, 32], [64, 128], [256, 512]]
    steps = [8, 16, 32]
    feature_maps = [
        [ceil(image_size[0] / step), ceil(image_size[1] / step)] for step in steps
    ]
    for k, f in enumerate(feature_maps):
        min_sizes_ = min_sizes[k]
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in min_sizes_:
                s_kx = min_size / image_size[1]
                s_ky = min_size / image_size[0]
                dense_cx = [x * steps[k] / image_size[1] for x in [j + 0.5]]
                dense_cy = [y * steps[k] / image_size[0] for y in [i + 0.5]]
                for cy, cx in product(dense_cy, dense_cx):
                    anchors += [cx, cy, s_kx, s_ky]
    output = np.array(anchors, dtype=np.float64).reshape(-1, 4)
    if log_priors:
        print("image_size:", image_size, " num_priors=", output.shape[0])
    return output


def box_decode(loc: np.ndarray, priors: np.ndarray) -> np.ndarray:
    variances = [0.1, 0.2]
    boxes = np.concatenate(
        (
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(loc[:, 2:] * variances[1]),
        ),
        axis=1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landm(pre: np.ndarray, priors: np.ndarray) -> np.ndarray:
    variances = [0.1, 0.2]
    return np.concatenate(
        (
            priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
        ),
        axis=1,
    )


def nms(dets: np.ndarray, thresh: float) -> list[int]:
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def split_outputs(outputs: list[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrs = [np.array(o) for o in outputs]
    if len(arrs) == 3:
        return arrs[0], arrs[1], arrs[2]
    loc = next(t for t in arrs if t.shape[-1] == 4)
    conf = next(t for t in arrs if t.shape[-1] == 2)
    landm = next(t for t in arrs if t.shape[-1] == 10)
    return loc, conf, landm


def dets_desde_salidas_modelo(
    outputs: list[Any],
    *,
    img_width: int,
    img_height: int,
    aspect_ratio: float,
    offset_x: int,
    offset_y: int,
    score_deteccion: float,
    score_pre_nms: float,
    nms_iou: float = NMS_IOU,
    log_priors: bool = False,
) -> np.ndarray:
    """
    Postproceso completo de salidas ONNX/RKNN -> filas (N, 15) en pixeles del frame original.

    Cada fila: [x1, y1, x2, y2, score, 10 coords landmarks].
    """
    loc, conf, landmarks = split_outputs(outputs)
    priors = prior_box(INPUT_HW, log_priors=log_priors)
    boxes = box_decode(loc.squeeze(0), priors)
    boxes = boxes * BOX_SCALE // 1
    boxes[..., 0::2] = np.clip(
        (boxes[..., 0::2] - offset_x) / aspect_ratio,
        0,
        img_width,
    )
    boxes[..., 1::2] = np.clip(
        (boxes[..., 1::2] - offset_y) / aspect_ratio,
        0,
        img_height,
    )
    scores = conf.squeeze(0)[:, 1]
    landmarks = decode_landm(landmarks.squeeze(0), priors)
    landmarks = landmarks * LANDMARK_SCALE // 1
    landmarks[..., 0::2] = np.clip(
        (landmarks[..., 0::2] - offset_x) / aspect_ratio,
        0,
        img_width,
    )
    landmarks[..., 1::2] = np.clip(
        (landmarks[..., 1::2] - offset_y) / aspect_ratio,
        0,
        img_height,
    )

    inds = np.where(scores > score_pre_nms)[0]
    boxes = boxes[inds]
    landmarks = landmarks[inds]
    scores = scores[inds]

    order = scores.argsort()[::-1]
    boxes = boxes[order]
    landmarks = landmarks[order]
    scores = scores[order]

    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = nms(dets, nms_iou)
    dets = dets[keep, :]
    landmarks = landmarks[keep]
    dets = np.concatenate((dets, landmarks), axis=1)

    mask = dets[:, 4] >= score_deteccion
    return dets[mask]
