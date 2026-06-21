"""Ranking y seleccion de las mejores caras RetinaFace para procesar (embed, etc.)."""
from __future__ import annotations

import math

import numpy as np

from inference.types import FaceDetections, FaceSelection


def bbox_area(det: np.ndarray) -> float:
    """Area del bbox en pixeles cuadrados."""
    x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
    return max(float(x2 - x1), 0.0) * max(float(y2 - y1), 0.0)


def distancia_interocular(det: np.ndarray) -> float:
    """Distancia entre landmarks de ojos (indices 5-8 en fila RetinaFace)."""
    return math.hypot(float(det[7] - det[5]), float(det[8] - det[6]))


def rank_cara(det: np.ndarray) -> float:
    """
    Metrica compuesta barata: area * score * distancia_ojos.

    Favorece caras grandes, confiables y en primer plano (ojos separados).
    """
    score = float(det[4])
    area = bbox_area(det)
    ojos = distancia_interocular(det)
    return area * score * max(ojos, 1.0)


def seleccionar_caras(
    dets: FaceDetections,
    *,
    top_n: int,
) -> FaceSelection:
    """
    Devuelve hasta ``top_n`` indices (mejor primero).

    Asume que ``dets`` ya viene filtrado por score en el postproceso del detector.

    ``top_n=1`` -> solo la mejor; ``top_n=2`` -> mejor y segunda, etc.

    Si hay <= ``top_n`` caras, devuelve todas sin calcular rank. Con mas, rankea y corta.
    """
    if top_n < 1 or not dets.has_faces:
        return FaceSelection.empty()

    rows = dets.dets
    n = rows.shape[0]

    if n <= top_n:
        return FaceSelection(tuple(range(n)))

    if top_n == 1:
        mejor = max(range(n), key=lambda i: rank_cara(rows[i]))
        return FaceSelection((mejor,))

    candidatas = [(idx, rank_cara(rows[idx])) for idx in range(n)]
    candidatas.sort(key=lambda t: t[1], reverse=True)
    return FaceSelection(tuple(idx for idx, _ in candidatas[:top_n]))


def mejores_caras(dets: FaceDetections, *, top_n: int) -> FaceDetections:
    """
    Devuelve hasta ``top_n`` caras utiles (mejor primero).

    Si ya hay <= ``top_n`` caras, reutiliza ``dets`` sin copiar. Solo copia al
    rankear y recortar cuando hay mas candidatas que ``top_n``.
    """
    if top_n < 1 or not dets.has_faces:
        return FaceDetections.empty()

    rows = dets.dets
    n = rows.shape[0]
    if n <= top_n:
        return dets

    selection = seleccionar_caras(dets, top_n=top_n)
    return FaceDetections(dets=selection.rows(dets))
