"""Inferencia de modelos (RetinaFace, MobileFaceNet pendiente)."""
from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from inference.types import FaceDetections, FaceSelection


class FaceDetector(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> FaceDetections: ...


def build_face_detector() -> FaceDetector | None:
    """
    Factory segun ``settings.INFERENCE_BACKEND``.

    Valores: ``none`` (sin detector), ``pc`` (ONNX), ``rk3568`` (RKNN).
    """
    from configs import settings as s

    backend = s.INFERENCE_BACKEND
    if backend == "none":
        return None
    if backend == "pc":
        from inference.retinaface.detector_pc import RetinaFaceDetectorPc

        return RetinaFaceDetectorPc.from_settings()
    if backend == "rk3568":
        from inference.retinaface.detector_rk3568 import RetinaFaceDetectorRk3568

        return RetinaFaceDetectorRk3568.from_settings()

    logging.critical(
        "INFERENCE_BACKEND invalido: '%s'. Usar none, pc o rk3568.",
        backend,
    )
    return None


__all__ = [
    "FaceDetections",
    "FaceDetector",
    "FaceSelection",
    "build_face_detector",
]
