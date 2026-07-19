"""FaceMesh 468 landmarks en RK3568 via RKNN Lite (pendiente export .rknn)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from inference.facemesh.constants import LANDMARK_DIM, NUM_LANDMARKS
from inference.facemesh.preprocess import bgr192_to_rknn_nhwc

try:
    from rknnlite.api import RKNNLite
except ImportError:
    RKNNLite = None  # type: ignore[assignment,misc]


class FaceMeshEstimatorRk3568:
    """
    Stub para paridad con ``retinaface/detector_rk3568`` y ``mobilefacenet/embedder_rk3568``.

    Completar tras exportar ``face_mesh_192x192.rknn`` (entrada RGB uint8 192x192).
    """

    def __init__(self, model_path: str | Path) -> None:
        self._model_path = Path(model_path)
        self._rknn: RKNNLite | None = None
        # TODO: load_rknn + init_runtime (copiar patron de MobileFaceNetEmbedderRk3568).

    def release(self) -> None:
        if self._rknn is not None:
            self._rknn.release()
            self._rknn = None

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass

    def estimate(self, face_bgr: np.ndarray) -> np.ndarray:
        """Parche BGR 192x192 -> (468, 3) float32 en espacio mesh."""
        del face_bgr, LANDMARK_DIM, NUM_LANDMARKS, bgr192_to_rknn_nhwc
        raise NotImplementedError(
            "FaceMesh RK3568 pendiente: exportar .rknn e implementar inferencia "
            f"(modelo esperado: {self._model_path})."
        )
