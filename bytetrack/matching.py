"""Costo IoU y asignacion de detecciones a tracks (solo NumPy)."""
from __future__ import annotations

import numpy as np


def ious(atlbrs: np.ndarray, btlbrs: np.ndarray) -> np.ndarray:
    """IoU pareado (len(a), len(b)) entre cajas tlbr (x1, y1, x2, y2)."""
    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)

    a = np.asarray(atlbrs, dtype=np.float32)
    b = np.asarray(btlbrs, dtype=np.float32)

    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def _is_array_like(tracks) -> bool:
    return len(tracks) > 0 and isinstance(tracks[0], np.ndarray)


def iou_distance(atracks, btracks) -> np.ndarray:
    """Costo = 1 - IoU. Acepta arrays tlbr directos o objetos con atributo .tlbr."""
    atlbrs = atracks if _is_array_like(atracks) else [t.tlbr for t in atracks]
    btlbrs = btracks if _is_array_like(btracks) else [t.tlbr for t in btracks]
    return 1.0 - ious(atlbrs, btlbrs)


def fuse_score(cost_matrix: np.ndarray, detections) -> np.ndarray:
    """Funde IoU con score de deteccion (favorece cajas de alta confianza)."""
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1.0 - cost_matrix
    det_scores = np.array([d.score for d in detections], dtype=np.float32)
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    return 1.0 - fuse_sim


def linear_assignment(
    cost_matrix: np.ndarray, thresh: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Asignacion voraz por costo minimo global (sin lap/scipy).

    Exacta para N<=2 (caso tipico con FACE_PROCESS_TOP_N); aproximada con mas
    filas/columnas. Alcanza para la escala de caras de este pipeline.
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(cost_matrix.shape[0]),
            np.arange(cost_matrix.shape[1]),
        )

    cost = cost_matrix.copy()
    row_mask = np.ones(cost.shape[0], dtype=bool)
    col_mask = np.ones(cost.shape[1], dtype=bool)
    matches: list[tuple[int, int]] = []

    while True:
        masked = np.where(row_mask[:, None] & col_mask[None, :], cost, np.inf)
        if not np.isfinite(masked).any():
            break
        i, j = np.unravel_index(np.argmin(masked), masked.shape)
        if masked[i, j] > thresh:
            break
        matches.append((int(i), int(j)))
        row_mask[i] = False
        col_mask[j] = False

    matches_arr = np.asarray(matches, dtype=int) if matches else np.empty((0, 2), dtype=int)
    return matches_arr, np.where(row_mask)[0], np.where(col_mask)[0]
