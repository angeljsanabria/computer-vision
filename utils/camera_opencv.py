"""
Apertura de camara alineada con 2_ocv_cam.py: V4L2 en Linux, fallback, calentamiento.

Uso tipico en RK3568: CAMERA_INDEX 10 u 11 para webcam USB UVC.
"""
from __future__ import annotations

import sys

import cv2

CALENTAMIENTO_LECTURAS = 25


def abrir_camara(indice: int) -> cv2.VideoCapture | None:
    if sys.platform.startswith("linux") and hasattr(cv2, "CAP_V4L2"):
        cap = cv2.VideoCapture(indice, cv2.CAP_V4L2)
        if cap.isOpened():
            return cap
        cap.release()
    cap = cv2.VideoCapture(indice)
    return cap if cap.isOpened() else None


def preparar_camara(cap: cv2.VideoCapture) -> bool:
    for _ in range(CALENTAMIENTO_LECTURAS):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            return True
    return False
