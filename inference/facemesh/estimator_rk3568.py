"""FaceMesh 468 landmarks en RK3568 via RKNN Lite."""
from __future__ import annotations

import logging
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
    FaceMesh 468 landmarks (RKNN) para RK3568.

    Entrada RGB uint8 192x192; mean/std (/255) en el .rknn al export.
    """

    def __init__(self, model_path: str | Path) -> None:
        if RKNNLite is None:
            raise RuntimeError(
                "rknnlite no instalado. Instala RKNN-Toolkit-Lite2 en la placa "
                "(aarch64), p. ej. rknn_toolkit_lite2-2.3.2-...whl"
            )
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"No existe modelo FaceMesh RKNN: {path}")

        self._rknn: RKNNLite | None = RKNNLite()
        if self._rknn.load_rknn(str(path)) != 0:
            raise RuntimeError(f"load_rknn failed: {path}")
        if self._rknn.init_runtime() != 0:
            self._rknn.release()
            raise RuntimeError(f"init_runtime failed: {path}")

        logging.debug("FaceMesh RK3568 (RKNN) cargado: %s", path)

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
        if self._rknn is None:
            raise RuntimeError("FaceMesh RK3568: runtime ya liberado")

        feed = bgr192_to_rknn_nhwc(face_bgr)
        outputs = self._rknn.inference(inputs=[feed])
        if not outputs:
            raise RuntimeError("FaceMesh RK3568: inference sin salida")

        pts = np.asarray(outputs[0], dtype=np.float32).reshape(-1, LANDMARK_DIM)
        if pts.shape[0] != NUM_LANDMARKS:
            raise RuntimeError(
                f"FaceMesh RKNN devolvio {pts.shape[0]} puntos, esperado {NUM_LANDMARKS}"
            )
        return pts
