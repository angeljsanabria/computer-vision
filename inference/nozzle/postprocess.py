"""Postproceso YOLOv8 Ultralytics (salida tipica 1 x (4+nc) x 8400)."""
from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88.0, 88.0)))


def _class_scores_to_prob(cls_values: np.ndarray) -> np.ndarray:
    """
    Ultralytics ONNX suele exportar probabilidades ya calibradas en [0, 1].
    Aplicar sigmoid otra vez infla scores (~0.5 minimo) y rompe el umbral.
    """
    if cls_values.size == 0:
        return cls_values
    vmin = float(np.min(cls_values))
    vmax = float(np.max(cls_values))
    if vmin >= 0.0 and vmax <= 1.0:
        return cls_values.astype(np.float32)
    return sigmoid(cls_values).astype(np.float32)


def postprocess_yolov8_ultralytics(
    pred: np.ndarray,
    conf_thres: float,
    iou_thres: float,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Devuelve (xyxy, scores) en espacio del tensor de entrada (640x640).

    ``pred``: (1, 4+nc, 8400) o (4+nc, 8400); nc=1 para nozzle fine-tune.

    Export ONNX Ultralytics: la columna de clase ya viene en [0, 1] (probabilidad).
    Export RKNN / logits crudos: aplicar sigmoid.
    """
    if pred.ndim == 3:
        pred = pred[0]
    pred = pred.T
    boxes_xywh = pred[:, :4]
    cls_values = pred[:, 4:]
    cls_prob = _class_scores_to_prob(cls_values)
    scores = np.max(cls_prob, axis=1)

    mask = scores >= conf_thres
    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    if len(scores) == 0:
        return None, None

    cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    xyxy = np.stack([x1, y1, x2, y2], axis=1)

    keep = _nms_xyxy(xyxy, scores, iou_thres)
    return xyxy[keep], scores[keep]


def _nms_xyxy(xyxy: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    keep: list[int] = []
    order = scores.argsort()[::-1]
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(xyxy[i, 0], xyxy[rest, 0])
        yy1 = np.maximum(xyxy[i, 1], xyxy[rest, 1])
        xx2 = np.minimum(xyxy[i, 2], xyxy[rest, 2])
        yy2 = np.minimum(xyxy[i, 3], xyxy[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = (xyxy[i, 2] - xyxy[i, 0]) * (xyxy[i, 3] - xyxy[i, 1])
        area_r = (xyxy[rest, 2] - xyxy[rest, 0]) * (xyxy[rest, 3] - xyxy[rest, 1])
        union = area_i + area_r - inter
        iou = inter / np.maximum(union, 1e-6)
        inds = np.where(iou <= iou_thres)[0]
        order = rest[inds]
    return np.array(keep, dtype=np.int64)


def scale_boxes_stretch(xyxy: np.ndarray, orig_w: int, orig_h: int, input_size: int) -> np.ndarray:
    """Mapea cajas desde tensor cuadrado stretch (RKNN) al frame original."""
    if xyxy.size == 0:
        return xyxy
    out = xyxy.astype(np.float32, copy=True)
    sx = orig_w / float(input_size)
    sy = orig_h / float(input_size)
    out[:, [0, 2]] *= sx
    out[:, [1, 3]] *= sy
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, orig_w)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, orig_h)
    return out
