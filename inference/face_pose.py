"""Heuristicas de pose desde landmarks RetinaFace (5 puntos)."""
from __future__ import annotations

import numpy as np

from inference.face_align import landmarks_from_det_row


def eye_roll_deg_from_landmarks(lmk: np.ndarray) -> float:
    """
    Roll en grados: angulo de la linea ojo_izq -> ojo_der vs horizontal.

    lmk: (5, 2) en orden RetinaFace (ojo_izq, ojo_der, nariz, boca_izq, boca_der).
    Frontal ~ 0°; cabeza inclinada aumenta |roll|.
    """
    if lmk.shape != (5, 2):
        raise ValueError(f"lmk debe ser (5, 2), got {lmk.shape}")
    dx = float(lmk[1, 0] - lmk[0, 0])
    dy = float(lmk[1, 1] - lmk[0, 1])
    return float(np.degrees(np.arctan2(dy, dx)))


def eye_roll_deg_from_det_row(det_row: np.ndarray) -> float:
    """Atajo: fila RetinaFace (15,) -> roll en grados."""
    return eye_roll_deg_from_landmarks(landmarks_from_det_row(det_row))


def roll_within_frontal_range(roll_deg: float, max_abs_roll_deg: float) -> bool:
    return abs(roll_deg) <= max_abs_roll_deg
