"""Dibujo de landmarks FaceMesh sobre frames BGR."""
from __future__ import annotations

import cv2
import numpy as np

from inference.facemesh.landmark_indices import SMALL_FACE_DISPLAY_INDICES
from inference.types import FaceMeshLandmarks

_SMALL_FACE_INDEX_ARRAY = np.asarray(SMALL_FACE_DISPLAY_INDICES, dtype=np.int32)


def filter_landmarks_for_bbox(
    landmarks: FaceMeshLandmarks,
    *,
    bbox_width: int,
    frame_width: int,
    full_points_bbox_frac: float,
) -> FaceMeshLandmarks:
    """
    Landmarks listos para overlay segun tamano del bbox en pantalla.

    Si ``bbox_width`` supera ``frame_width * full_points_bbox_frac``, devuelve
    los 468 puntos. Si no, omite ojos y region boca/nariz (nucleo + anillo perioral);
    devuelve el resto del mesh (ver ``inference/facemesh/landmark_indices.py``).
    """
    if frame_width <= 0 or bbox_width <= 0:
        return landmarks
    threshold = frame_width * full_points_bbox_frac
    if bbox_width > threshold:
        return landmarks
    return FaceMeshLandmarks(
        points=landmarks.points[_SMALL_FACE_INDEX_ARRAY],
        crop_xyxy=landmarks.crop_xyxy,
    )


def draw_facemesh_points(
    vis: np.ndarray,
    landmarks: FaceMeshLandmarks | None,
    *,
    point_color: tuple[int, int, int] = (0, 255, 0),
    point_radius: int = 1,
) -> None:
    """Dibuja los landmarks recibidos sobre ``vis`` (in-place, sin bbox)."""
    if landmarks is None:
        return

    frame_h, frame_w = vis.shape[:2]
    for x_f, y_f, _ in landmarks.points:
        x, y = int(x_f), int(y_f)
        if 0 <= x < frame_w and 0 <= y < frame_h:
            cv2.circle(vis, (x, y), point_radius, point_color, -1)


def draw_facemesh_landmarks(
    frame_bgr: np.ndarray,
    landmarks: FaceMeshLandmarks | None,
    *,
    point_color: tuple[int, int, int] = (0, 255, 0),
    point_radius: int = 1,
    crop_color: tuple[int, int, int] = (255, 0, 0),
    draw_crop_rect: bool = False,
) -> np.ndarray:
    """
    Copia el frame y dibuja landmarks.

    Por defecto solo puntos (``draw_crop_rect=False``) para no duplicar bbox
    cuando el recuadro ya lo dibuja ``ui/overlay`` (track o deteccion).
    """
    vis = frame_bgr.copy()
    draw_facemesh_points(
        vis, landmarks, point_color=point_color, point_radius=point_radius
    )
    if draw_crop_rect and landmarks is not None:
        x1, y1, x2, y2 = landmarks.crop_xyxy
        cv2.rectangle(vis, (x1, y1), (x2, y2), crop_color, 1)
    return vis
