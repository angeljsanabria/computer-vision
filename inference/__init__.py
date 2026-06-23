"""Inferencia de modelos (RetinaFace, MobileFaceNet)."""
from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from inference.types import FaceDetections, FaceEmbedding, FaceSelection


class FaceDetector(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> FaceDetections: ...


class FaceEmbedder(Protocol):
    def embed(self, face_bgr: np.ndarray) -> np.ndarray: ...

    def release(self) -> None: ...


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


def build_embedder() -> FaceEmbedder | None:
    """
    Factory segun ``settings.INFERENCE_BACKEND``.

    Mismo backend que RetinaFace: ``none`` sin embedder, ``pc`` ONNX, ``rk3568`` RKNN.
    """
    from configs import settings as s

    backend = s.INFERENCE_BACKEND
    if backend == "none":
        return None
    if backend == "pc":
        from inference.mobilefacenet.embedder_pc import MobileFaceNetEmbedderPc

        return MobileFaceNetEmbedderPc.from_settings()
    if backend == "rk3568":
        from inference.mobilefacenet.embedder_rk3568 import (
            MobileFaceNetEmbedderRk3568,
        )

        return MobileFaceNetEmbedderRk3568.from_settings()

    logging.critical(
        "INFERENCE_BACKEND invalido: '%s'. Usar none, pc o rk3568.",
        backend,
    )
    return None


__all__ = [
    "FaceDetections",
    "FaceDetector",
    "FaceEmbedder",
    "FaceEmbedding",
    "FaceSelection",
    "build_embedder",
    "build_face_detector",
]
