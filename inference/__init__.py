"""Inferencia de modelos (RetinaFace, MobileFaceNet, FaceMesh, Nozzle YOLO)."""
from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from inference.nozzle.types import NozzleDetections
from inference.types import (
    FaceDetections,
    FaceEmbedding,
    FaceMeshLandmarks,
    FaceSelection,
)


class FaceDetector(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> FaceDetections: ...


class FaceEmbedder(Protocol):
    def embed(self, face_bgr: np.ndarray) -> np.ndarray: ...

    def release(self) -> None: ...


class FaceMeshEstimator(Protocol):
    def estimate(self, face_bgr: np.ndarray) -> np.ndarray: ...

    def release(self) -> None: ...


class NozzleDetector(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> NozzleDetections: ...

    def release(self) -> None: ...


def build_face_detector(
    backend: str,
    model_path: str,
    score_deteccion: float,
    score_pre_nms: float,
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
        )
    if backend == "rk3568":
        from inference.retinaface.detector_rk3568 import RetinaFaceDetectorRk3568

        return RetinaFaceDetectorRk3568(
            model_path=model_path,
            score_deteccion=score_deteccion,
            score_pre_nms=score_pre_nms,
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


def build_face_mesh(backend: str, model_path: str) -> FaceMeshEstimator | None:
    """
    Factory segun backend.

    Mismo backend que RetinaFace: ``none`` sin FaceMesh, ``pc`` ONNX, ``rk3568`` RKNN.
    """
    if backend == "none":
        return None
    if backend == "pc":
        from inference.facemesh.estimator_pc import FaceMeshEstimatorPc

        return FaceMeshEstimatorPc(model_path=model_path)
    if backend == "rk3568":
        from inference.facemesh.estimator_rk3568 import FaceMeshEstimatorRk3568

        return FaceMeshEstimatorRk3568(model_path=model_path)

    logging.critical(
        "INFERENCE_BACKEND invalido: '%s'. Usar none, pc o rk3568.",
        backend,
    )
    return None


def build_nozzle_detector(
    backend: str,
    model_path: str,
    score_deteccion: float,
    nms_iou: float,
) -> NozzleDetector | None:
    """
    Factory segun backend.

    Mismo backend que RetinaFace: ``none`` sin detector, ``pc`` ONNX, ``rk3568`` RKNN.
    """
    if backend == "none":
        return None
    if backend == "pc":
        from inference.nozzle.detector_pc import NozzleDetectorPc

        return NozzleDetectorPc(
            model_path=model_path,
            score_deteccion=score_deteccion,
            nms_iou=nms_iou,
        )
    if backend == "rk3568":
        from inference.nozzle.detector_rk3568 import NozzleDetectorRk3568

        return NozzleDetectorRk3568(
            model_path=model_path,
            score_deteccion=score_deteccion,
            nms_iou=nms_iou,
        )

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
    "FaceMeshEstimator",
    "FaceMeshLandmarks",
    "FaceSelection",
    "NozzleDetections",
    "NozzleDetector",
    "build_embedder",
    "build_face_detector",
    "build_face_mesh",
    "build_identity_matcher",
    "build_nozzle_detector",
]
