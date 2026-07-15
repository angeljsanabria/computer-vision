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


def build_face_detector(
    backend: str,
    model_path: str,
    score_deteccion: float,
    score_pre_nms: float,
    *,
    use_rga: bool = False,
) -> FaceDetector | None:
    """
    Factory segun backend.

    Valores: ``none`` (sin detector), ``pc`` (ONNX), ``rk3568`` (RKNN).
    """
    if backend == "none":
        return None
    if backend == "pc":
        from inference.retinaface.detector_pc import RetinaFaceDetectorPc

        return RetinaFaceDetectorPc(
            model_path=model_path,
            score_deteccion=score_deteccion,
            score_pre_nms=score_pre_nms,
            use_rga=use_rga,
        )
    if backend == "rk3568":
        from inference.retinaface.detector_rk3568 import RetinaFaceDetectorRk3568

        return RetinaFaceDetectorRk3568(
            model_path=model_path,
            score_deteccion=score_deteccion,
            score_pre_nms=score_pre_nms,
            use_rga=use_rga,
        )

    logging.critical(
        "INFERENCE_BACKEND invalido: '%s'. Usar none, pc o rk3568.",
        backend,
    )
    return None


def build_embedder(backend: str, model_path: str) -> FaceEmbedder | None:
    """
    Factory segun backend.

    Mismo backend que RetinaFace: ``none`` sin embedder, ``pc`` ONNX, ``rk3568`` RKNN.
    """
    if backend == "none":
        return None
    if backend == "pc":
        from inference.mobilefacenet.embedder_pc import MobileFaceNetEmbedderPc

        return MobileFaceNetEmbedderPc(model_path=model_path)
    if backend == "rk3568":
        from inference.mobilefacenet.embedder_rk3568 import (
            MobileFaceNetEmbedderRk3568,
        )

        return MobileFaceNetEmbedderRk3568(model_path=model_path)

    logging.critical(
        "INFERENCE_BACKEND invalido: '%s'. Usar none, pc o rk3568.",
        backend,
    )
    return None


def build_identity_matcher(
    backend: str,
    gallery_dir: str,
    min_similarity: float,
    npy_name: str,
    meta_name: str,
) -> "FaceGalleryMatcher | None":
    """Matcher 1:N vs galeria .npy. ``None`` si ``backend=none``."""
    if backend == "none":
        return None

    from inference.identity.matcher import FaceGalleryMatcher

    return FaceGalleryMatcher(
        gallery_dir=gallery_dir,
        min_similarity=min_similarity,
        npy_name=npy_name,
        meta_name=meta_name,
    )


__all__ = [
    "FaceDetections",
    "FaceDetector",
    "FaceEmbedder",
    "FaceEmbedding",
    "FaceSelection",
    "build_embedder",
    "build_face_detector",
    "build_identity_matcher",
]
