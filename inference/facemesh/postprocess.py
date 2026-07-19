"""Remap de landmarks FaceMesh al sistema de coords del frame original."""
from __future__ import annotations

import numpy as np

from inference.facemesh.constants import INPUT_SIZE, LANDMARK_DIM, NUM_LANDMARKS


def landmarks_mesh_to_frame(
    mesh_points: np.ndarray,
    crop_xyxy: tuple[int, int, int, int],
    *,
    mesh_size: int = INPUT_SIZE,
) -> np.ndarray:
    """
    Convierte salida del modelo (coords en espacio 192x192) a pixeles del frame.

    ``mesh_points``: (468, 3) o flat; x/y en [0, mesh_size], z profundidad relativa.
    """
    pts = np.asarray(mesh_points, dtype=np.float32).reshape(-1, LANDMARK_DIM)
    if pts.shape[0] != NUM_LANDMARKS:
        raise ValueError(
            f"FaceMesh devolvio {pts.shape[0]} puntos, esperado {NUM_LANDMARKS}"
        )

    x1, y1, x2, y2 = crop_xyxy
    crop_w = max(x2 - x1 + 1, 1)
    crop_h = max(y2 - y1 + 1, 1)
    scale = float(mesh_size)

    out = pts.copy()
    out[:, 0] = x1 + (pts[:, 0] / scale) * crop_w
    out[:, 1] = y1 + (pts[:, 1] / scale) * crop_h
    return out
